# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for GPU压图 (onedir, noconsole)
# 构建: python -m PyInstaller --noconfirm GPU压图.spec
from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = [
    'graphlib', 'imagecodecs',
    'cuda', 'cuda.pathfinder',
    'cuda.pathfinder._dynamic_libs.dynamic_lib_subprocess',
]
for pkg in ('cupy', 'cupy_backends', 'cupyx', 'numba', 'llvmlite',
             'imagecodecs'):
    tmp = collect_all(pkg)
    datas += tmp[0]
    binaries += tmp[1]
    hiddenimports += tmp[2]

EXCLUDES = [
    'yt_dlp', 'altair', 'accelerate', 'aiohttp', 'av', 'bcrypt', 'bs4',
    'bitsandbytes', 'brotli', 'clr_loader', 'contourpy', 'cryptography',
    'cv2', 'curl_cffi', 'datasets', 'dill', 'emoji', 'fastapi', 'filelock',
    'frozenlist', 'fsspec', 'ftfy', 'h2', 'hf_xet', 'httpcore', 'httpx',
    'huggingface_hub', 'invoke', 'jsonschema', 'mutagen', 'pyarrow',
    'Crypto', 'pycryptodome', 'pydantic', 'pytest', 'requests', 'urllib3',
    'secretstorage', 'soundfile', 'sqlalchemy', 'torch', 'torchvision',
    'transformers', 'websockets', 'websocket', 'yaml', 'google',
    'selenium', 'setuptools', 'pip', 'docutils', 'jinja2', 'dateutil',
    'pandas', 'matplotlib', 'scipy', 'sklearn', 'IPython', 'jupyter',
    'tornado', 'pytz', 'six', 'numpy.f2py',
]

# plain-source build: the Sep-25 encrypted pyd is intentionally NOT shipped —
# it would shadow the current sources and run a month-old pipeline
# C++ fast path DLLs -> _internal/cpp/, matching pipeline.py's
# <root>/cpp/ lookup when gpuwebp ships as plain sources (_internal/gpuwebp)
binaries += [
    (r'cpp/gpu_pipeline_v2.dll', 'cpp'),
    (r'cpp/entropy.dll', 'cpp'),
    (r'cpp/pngdec.dll', 'cpp'),
]

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='GPU压图',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
_drop_prefixes = ('cublas', 'cusparse', 'cufft', 'cusolver', 'curand')
_pruned = [b for b in a.binaries
           if not any(b[0].lower().startswith(p) for p in _drop_prefixes)]

coll = COLLECT(
    exe,
    _pruned,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='GPU压图',
)
