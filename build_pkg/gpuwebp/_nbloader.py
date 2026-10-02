
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
