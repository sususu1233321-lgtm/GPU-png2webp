# -*- coding: utf-8 -*-
"""Build the source-protected gpuwebp package:

  纯 Python 模块           -> Nuitka 编译为机器码 (.pyd)
  含 numba @njit 的模块    -> 字节码加密(zlib+XOR+b85)嵌进编译后的加载器,
                              运行时解密到内存执行(numba 需要真实字节码)
  CUDA 内核源码字符串       -> 压缩加密为数据块,同样嵌进 .pyd

产出: build_pkg/gpuwebp.cp312-win_amd64.pyd(单文件、无 .py 源码)
"""
import base64
import compileall
import importlib.util
import marshal
import os
import shutil
import sys
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "gpuwebp")
OUT = os.path.join(ROOT, "build_pkg")

# modules whose functions use numba @njit (need real bytecode at runtime)
NUMBA_MODULES = [
    "alpha_enc", "closed_loop_jit", "closed_loop", "bool_coder",
    "metafast", "vp8_encode", "pngdec", "encoder", "pipeline",
]
# pure-python modules compiled to machine code (pipeline/encoder appear in
# both lists: they are shipped as encrypted bytecode because of their njit
# parts — see above)
PURE_MODULES = [m for m in os.listdir(PKG)
                if m.endswith(".py")
                and m[:-3] not in NUMBA_MODULES
                and m[:-3] not in ("gen_tables", "__init__")]
# dev-only, never shipped
SKIP = {"gen_tables.py"}

KEY = b"GPUYaTu-2024-" + bytes.fromhex("9d012abde0") + b"#webp"


def encrypt_bytes(data: bytes) -> str:
    x = bytes(b ^ KEY[i % len(KEY)] for i, b in enumerate(zlib.compress(data, 9)))
    return base64.b85encode(x).decode()


def decrypt_bytes(blob: str) -> bytes:
    raw = base64.b85decode(blob)
    return zlib.decompress(
        bytes(b ^ KEY[i % len(KEY)] for i, b in enumerate(raw)))


def build():
    shutil.rmtree(OUT, ignore_errors=True)
    dst = os.path.join(OUT, "gpuwebp")
    os.makedirs(dst)

    # ---------- 1) encrypted bytecode blobs for numba modules ----------
    blobs = {}
    pyc_dir = os.path.join(OUT, "_pyc")
    for mod in NUMBA_MODULES:
        src = os.path.join(PKG, mod + ".py")
        spec = importlib.util.spec_from_file_location("gpuwebp." + mod, src)
        text = open(src, encoding="utf-8").read()
        # encrypted modules have no real file for numba's cache locator;
        # disable on-disk caching for them (JIT happens in memory per run)
        text = text.replace("cache=True", "cache=False")
        code = compile(text, f"gpuwebp/{mod}.py", "exec", dont_inherit=True)
        blobs[mod] = encrypt_bytes(marshal.dumps(code))

    nbdata = ("# generated: encrypted bytecode payloads\n"
              "BLOBS = {\n")
    for mod, blob in blobs.items():
        nbdata += f'    "{mod}":\n        "{blob}",\n'
    nbdata += "}\n"
    open(os.path.join(dst, "_nbdata.py"), "w", encoding="utf-8").write(nbdata)

    # ---------- 2) loader (compiled to machine code by Nuitka) ----------
    loader = '''
import importlib.abc, importlib.util, marshal, sys, types
from . import _nbdata

_KEY = b"GPUYaTu-2024-" + bytes.fromhex("9d012abde0") + b"#webp"


def _decrypt(blob):
    import base64, zlib
    raw = base64.b85decode(blob)
    return zlib.decompress(
        bytes(b ^ _KEY[i % len(_KEY)] for i, b in enumerate(raw)))


class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Serves the numba-dependent submodules from encrypted bytecode."""
    _names = set(_nbdata.BLOBS)

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith("gpuwebp."):
            return None
        short = fullname[len("gpuwebp."):]
        if "." in short or short not in self._names:
            return None
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec):
        return types.ModuleType(spec.name)

    def exec_module(self, module):
        code = marshal.loads(_decrypt(_nbdata.BLOBS[module.__name__.split(".")[1]]))
        module.__file__ = f"<gpuwebp/{module.__name__.split('.')[1]}.py>"
        exec(code, module.__dict__)


_installed = False


def install():
    global _installed
    if _installed:
        return
    sys.meta_path.insert(0, _Finder())
    _installed = True
'''
    open(os.path.join(dst, "_nbloader.py"), "w", encoding="utf-8").write(loader)

    # ---------- 3) __init__ installs the loader ----------
    open(os.path.join(dst, "__init__.py"), "w", encoding="utf-8").write(
        '"""GPU压图 编码器(源码保护版)。"""\n'
        "from . import _nbloader\n"
        "_nbloader.install()\n")

    # ---------- 4) pure modules; obfuscate the CUDA kernel source ----------
    import re
    for name in PURE_MODULES:
        if name in SKIP:
            continue
        src = open(os.path.join(PKG, name), encoding="utf-8").read()
        if name == "closed_loop_gpu.py":
            m = re.search(r'_CUDA_SRC = r"""(.*?)"""(\n\n_CUDA_SRC)', src, re.S)
            # find the actual triple-quoted block
            m = re.search(r'_CUDA_SRC = r"""', src)
            if m:
                start = m.end()
                end = src.index('"""', start)
                kernel_src = src[m.start():end + 3]
                enc = "_CUDA_ENC = (\n    \"" + encrypt_bytes(
                    src[start:end].encode()).replace(
                    "", "") + "\")" if False else None
                # build replacement
                blob = encrypt_bytes(src[start:end].encode())
                wrapped = "\n".join(
                    '    "%s"' % (blob[i:i + 96])
                    for i in range(0, len(blob), 96))
                repl = ("import base64, zlib as _zlib\n"
                        "from . import _prot\n"
                        "_CUDA_SRC = _prot.dec(\n" + wrapped + "\n)\n")
                src = src[:m.start()] + repl + src[end + 3:]
        shutil.copyfile(os.path.join(PKG, name),
                        os.path.join(dst, name))
        if name == "closed_loop_gpu.py":
            open(os.path.join(dst, name), "w", encoding="utf-8").write(src)

    # shared protection helpers module
    open(os.path.join(dst, "_prot.py"), "w", encoding="utf-8").write(
        'import base64, zlib\n'
        '_KEY = b"GPUYaTu-2024-" + bytes.fromhex("9d012abde0") + b"#webp"\n'
        "\n"
        "def dec(blob):\n"
        "    raw = base64.b85decode(blob)\n"
        "    return zlib.decompress(bytes(\n"
        "        b ^ _KEY[i % len(_KEY)] for i, b in enumerate(raw))\n"
        "    ).decode()\n")

    print("build copy ready:", dst)
    print("  pure modules:", sorted(m for m in PURE_MODULES if m not in SKIP))
    print("  encrypted bytecode modules:", NUMBA_MODULES)


if __name__ == "__main__":
    build()
