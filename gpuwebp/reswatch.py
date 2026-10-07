"""Real-time resource governor: pure-ctypes sampling (no subprocess spawn),
throttles the pipeline when free RAM runs low and aborts before the machine
freezes. PowerShell/nvidia-smi polling itself once contributed to a freeze,
so everything here is in-process Win32 calls (~microseconds)."""
import ctypes
import threading
import time


class _MEMSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_uint64),
        ("ullAvailPhys", ctypes.c_uint64),
        ("ullTotalPageFile", ctypes.c_uint64),
        ("ullAvailPageFile", ctypes.c_uint64),
        ("ullTotalVirtual", ctypes.c_uint64),
        ("ullAvailVirtual", ctypes.c_uint64),
        ("ullAvailExtendedVirtual", ctypes.c_uint64),
    ]


_k32 = ctypes.windll.kernel32


def mem_status():
    """Returns (avail_gb, load_pct). Never raises."""
    try:
        st = _MEMSTATUSEX()
        st.dwLength = ctypes.sizeof(_MEMSTATUSEX)
        if not _k32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return 999.0, 0
        return st.ullAvailPhys / 1e9, st.dwMemoryLoad
    except Exception:
        return 999.0, 0


class Governor:
    """Background sampler. .throttle is set while avail RAM < low_gb;
    .abort is set if it ever drops below crit_gb (machine-protection)."""

    def __init__(self, log=None, low_gb=8.0, crit_gb=3.0, interval=0.5):
        self.low_gb = low_gb
        self.crit_gb = crit_gb
        self.interval = interval
        self.log = log or (lambda msg: None)
        self.throttle = threading.Event()
        self.abort = threading.Event()
        self.avail_gb = 999.0
        self.load_pct = 0
        self.min_avail = 999.0
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def _run(self):
        warned_low = False
        while not self._stop.is_set():
            avail, load = mem_status()
            self.avail_gb = avail
            self.load_pct = load
            self.min_avail = min(self.min_avail, avail)
            if avail < self.crit_gb:
                self.abort.set()
                self.throttle.set()
                self.log(f"[资源保护] 可用内存{avail:.1f}GB<{self.crit_gb}GB, "
                         f"中止新任务防止死机")
                break
            if avail < self.low_gb:
                self.throttle.set()
                if not warned_low:
                    self.log(f"[资源保护] 可用内存{avail:.1f}GB<{self.low_gb}GB, "
                             f"暂停供给")
                    warned_low = True
            else:
                if self.throttle.is_set() and warned_low:
                    self.log(f"[资源保护] 内存恢复至{avail:.1f}GB, 继续供给")
                self.throttle.clear()
                warned_low = False
            self._stop.wait(self.interval)

    def checkpoint(self, where=""):
        """Cooperative gate: block while throttled; raise if aborting."""
        while self.throttle.is_set() and not self.abort.is_set():
            time.sleep(0.05)
        if self.abort.is_set():
            raise RuntimeError(f"资源保护中止 {where}")

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
