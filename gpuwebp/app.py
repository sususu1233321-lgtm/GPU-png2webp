"""GPU压图 — 中文 tkinter 界面 + 命令行批量模式。

无参数启动图形界面；带参数运行命令行批处理：
  GPU压图.exe --src <文件夹> [--dst <文件夹>] [--quality 90] [--cpu]
              [--recursive] [--no-verify] [--workers 3] [--device 0]
"""
import argparse
import os
import subprocess
import sys
import threading
import time
import traceback

DEFAULT_SRC = r"L:\图片备份8\nai3_240531"


def _fill_devices(box, want_default_label=True):
    """Populate GPU combo box; returns list of (label, device_id)."""
    items = [("自动", 0)]
    try:
        import cupy as cp
        for i in range(cp.cuda.runtime.getDeviceCount()):
            name = cp.cuda.runtime.getDeviceProperties(i)["name"].decode(
                "latin1", "replace")
            items.append((f"{name.strip()} (#{i})", i))
    except Exception:
        pass
    box["values"] = [lbl for lbl, _ in items]
    box.current(0)
    return items


def run_gui():
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext, ttk

    from gpuwebp.pipeline import run_batch

    root = tk.Tk()
    root.title("GPU压图 — PNG → WebP(自研GPU编码器)")
    root.geometry("860x640")

    state = {"stop": threading.Event(), "running": False}
    stat = {"labels": {}}

    frm = ttk.Frame(root, padding=8)
    frm.pack(fill="both", expand=True)

    # ---- row: source / dest
    ttk.Label(frm, text="源文件夹:").grid(row=0, column=0, sticky="w")
    src_var = tk.StringVar(value=DEFAULT_SRC if os.path.isdir(DEFAULT_SRC) else "")
    ttk.Entry(frm, textvariable=src_var, width=64).grid(row=0, column=1, sticky="we")
    ttk.Button(frm, text="浏览…", command=lambda: src_var.set(
        filedialog.askdirectory() or src_var.get())).grid(row=0, column=2)

    ttk.Label(frm, text="输出文件夹:").grid(row=1, column=0, sticky="w")
    dst_var = tk.StringVar(value="")
    ttk.Entry(frm, textvariable=dst_var, width=64).grid(row=1, column=1, sticky="we")
    ttk.Button(frm, text="浏览…", command=lambda: dst_var.set(
        filedialog.askdirectory() or dst_var.get())).grid(row=1, column=2)

    # ---- row: options
    opt = ttk.LabelFrame(frm, text="选项", padding=6)
    opt.grid(row=2, column=0, columnspan=3, sticky="we", pady=6)

    ttk.Label(opt, text="质量:").grid(row=0, column=0, sticky="w")
    q_var = tk.IntVar(value=90)
    q_scale = ttk.Scale(opt, from_=50, to=100, variable=q_var, length=140)
    q_scale.grid(row=0, column=1)
    q_lbl = ttk.Label(opt, text="90")
    q_lbl.grid(row=0, column=2)
    q_scale.config(command=lambda v: (q_var.set(int(float(v))),
                                      q_lbl.config(text=str(int(float(v))))))

    ttk.Label(opt, text="引擎:").grid(row=0, column=3, sticky="w")
    eng_var = tk.StringVar(value="GPU(批量高速)")
    ttk.Combobox(opt, textvariable=eng_var, state="readonly", width=14,
                 values=["GPU(批量高速)", "GPU(单张)", "CPU(兼容)"]).grid(row=0, column=4)

    ttk.Label(opt, text="GPU:").grid(row=0, column=5, sticky="w")
    dev_box = ttk.Combobox(opt, state="readonly", width=22)
    dev_box.grid(row=0, column=6)
    dev_map = _fill_devices(dev_box)

    ttk.Label(opt, text="线程:").grid(row=1, column=0, sticky="w")
    workers_var = tk.IntVar(value=3)
    ttk.Spinbox(opt, from_=1, to=8, textvariable=workers_var, width=4).grid(row=1, column=1)

    rec_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(opt, text="包含子目录", variable=rec_var).grid(row=1, column=3)
    skip_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(opt, text="跳过已存在", variable=skip_var).grid(row=1, column=4)
    verify_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(opt, text="逐张校验(PSNR≥34+元数据字节比对,失败自动CPU兜底)",
                    variable=verify_var).grid(row=1, column=5, columnspan=2, sticky="w")
    del_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(opt, text="完成后删除原PNG(二次确认)", variable=del_var).grid(
        row=2, column=0, columnspan=3, sticky="w")

    # ---- progress
    prog = ttk.Progressbar(frm, maximum=100)
    prog.grid(row=3, column=0, columnspan=3, sticky="we", pady=(6, 0))
    info = ttk.Label(frm, text="就绪")
    info.grid(row=4, column=0, columnspan=3, sticky="w")

    # ---- log
    log = scrolledtext.ScrolledText(frm, height=14, state="disabled", font=("Consolas", 9))
    log.grid(row=5, column=0, columnspan=3, sticky="nsew", pady=4)
    frm.rowconfigure(5, weight=1)
    frm.columnconfigure(1, weight=1)

    btns = ttk.Frame(frm)
    btns.grid(row=6, column=0, columnspan=3, sticky="we")
    start_btn = ttk.Button(btns, text="开始压缩")
    start_btn.pack(side="left")
    stop_btn = ttk.Button(btns, text="停止", state="disabled")
    stop_btn.pack(side="left", padx=6)

    def open_out():
        out = dst_var.get() or default_dst()
        if os.path.isdir(out):
            os.startfile(out)                            # noqa: S606

    def default_dst():
        s = src_var.get()
        return os.path.join(s, "webp") if s else ""

    def logline(s):
        log.configure(state="normal")
        log.insert("end", s + "\n")
        log.see("end")
        log.configure(state="disabled")

    def set_running(r):
        state["running"] = r
        start_btn.config(state="disabled" if r else "normal")
        stop_btn.config(state="normal" if r else "disabled")

    def on_done(stats, t0):
        set_running(False)
        pct = (100 * stats.dst_bytes / stats.src_bytes) if stats.src_bytes else 0
        logline("=" * 60)
        logline(f"完成: 成功 {stats.done} / 失败 {stats.failed} / 跳过 {stats.skipped}"
                f" / CPU兜底 {stats.fallbacks}")
        logline(f"体积: {stats.src_bytes/1e6:.1f} MB → {stats.dst_bytes/1e6:.1f} MB "
                f"({pct:.1f}%, 节省 {100-pct:.1f}%)")
        logline(f"耗时 {time.time()-t0:.1f}s, 平均 {stats.total/max(time.time()-t0,1e-9):.1f} 张/秒")
        info.config(text=f"完成 — 节省 {100-pct:.1f}%")

    def worker():
        t0 = time.time()
        try:
            stats = run_batch(
                src_var.get(), dst_var.get() or default_dst(),
                quality=int(q_var.get()),
                engine=("cpu" if "CPU" in eng_var.get()
                        else "gpu" if "单张" in eng_var.get() else "gpu-batch"),
                device=dev_map[dev_box.current()][1],
                recursive=rec_var.get(), skip_existing=skip_var.get(),
                verify_meta=verify_var.get(), workers=int(workers_var.get()),
                progress_cb=on_progress, log_cb=logline,
                stop_event=state["stop"])
            root.after(0, lambda: on_done(stats, t0))
            if del_var.get():
                root.after(0, ask_delete)
        except Exception:
            logline(traceback.format_exc())
            root.after(0, lambda: set_running(False))

    def ask_delete():
        if messagebox.askyesno("删除原PNG",
                               "压缩已完成并通过校验。\n确定要删除原始PNG文件吗?\n"
                               "(此操作不可恢复,建议先抽查输出结果)"):
            n = 0
            for fn in os.listdir(src_var.get()):
                if fn.lower().endswith(".png"):
                    webp = os.path.join(dst_var.get() or default_dst(),
                                        os.path.splitext(fn)[0] + ".webp")
                    if os.path.exists(webp):
                        try:
                            os.remove(os.path.join(src_var.get(), fn))
                            n += 1
                        except OSError:
                            pass
            logline(f"已删除 {n} 个原PNG")

    def on_progress(stats):
        done_all = stats.done + stats.failed + stats.skipped
        pct = 100 * done_all / max(stats.total, 1)
        rate = stats.speed
        eta = stats.eta
        saved = (100 - 100 * stats.dst_bytes / stats.src_bytes) if stats.src_bytes else 0
        text = (f"进度 {done_all}/{stats.total} ({pct:.0f}%)  "
                f"速度 {rate:.1f} 张/秒  剩余 ~{eta/60:.1f} 分钟  "
                f"已省 {saved:.1f}%  当前: {stats.cur_name[:40]}")
        root.after(0, lambda: (prog.config(value=pct), info.config(text=text)))

    def start():
        s = src_var.get()
        if not os.path.isdir(s):
            messagebox.showerror("错误", "源文件夹不存在")
            return
        d = dst_var.get() or default_dst()
        os.makedirs(d, exist_ok=True)
        state["stop"].clear()
        set_running(True)
        logline(f"开始: {s} → {d} (质量{q_var.get()}, {eng_var.get()}引擎)")
        threading.Thread(target=worker, daemon=True).start()

    start_btn.config(command=start)
    stop_btn.config(command=lambda: (state["stop"].set(), logline("停止中…")))
    ttk.Button(btns, text="打开输出目录", command=open_out).pack(side="left", padx=6)
    ttk.Button(btns, text="清空日志", command=lambda: (log.delete("1.0", "end"))).pack(side="right")

    root.mainloop()


def run_cli(argv):
    from gpuwebp.pipeline import run_batch

    ap = argparse.ArgumentParser(prog="GPU压图", description="PNG→WebP 批量压缩")
    ap.add_argument("--src", help="源文件夹")
    ap.add_argument("--dst", default=None, help="输出文件夹(默认 源/webp)")
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--cpu", action="store_true", help="使用CPU(Pillow)引擎")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--no-verify", action="store_true", help="关闭逐张校验")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--diag", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.diag or not args.src:
        print("== 诊断模式 ==")
        print("python:", sys.version)
        print("frozen:", getattr(sys, "frozen", False))
        import numpy
        print("numpy", numpy.__version__)
        import PIL
        print("pillow", PIL.__version__)
        try:
            import numba
            print("numba", numba.__version__)
        except Exception:
            traceback.print_exc()
        try:
            import cupy
            print("cupy", cupy.__version__)
            with cupy.cuda.Device(0):
                print("gpu0:", cupy.cuda.runtime.getDeviceProperties(0)["name"])
                print("compute test:", int(cupy.arange(100).sum()))
        except Exception:
            traceback.print_exc()
        try:
            from gpuwebp import gpu_engine
            print("gpu_engine import OK")
        except Exception:
            traceback.print_exc()
        return

    dst = args.dst or os.path.join(args.src, "webp")
    print(f"源: {args.src}\n出: {dst}  质量: {args.quality}  "
          f"引擎: {'CPU' if args.cpu else 'GPU'}")
    t0 = time.time()
    stats = run_batch(args.src, dst, quality=args.quality,
                      engine="cpu" if args.cpu else "gpu-batch", device=args.device,
                      recursive=args.recursive, skip_existing=True,
                      verify_meta=not args.no_verify, workers=args.workers,
                      log_cb=print)
    pct = 100 * stats.dst_bytes / max(stats.src_bytes, 1)
    print("=" * 50)
    print(f"成功 {stats.done} 失败 {stats.failed} 跳过 {stats.skipped} CPU兜底 {stats.fallbacks}")
    print(f"{stats.src_bytes/1e6:.1f}MB → {stats.dst_bytes/1e6:.1f}MB ({pct:.1f}%) "
          f"耗时 {time.time()-t0:.1f}s")


def main():
    argv = sys.argv[1:]
    if argv:
        run_cli(argv)
    else:
        run_gui()


if __name__ == "__main__":
    main()
