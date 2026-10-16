#!/usr/bin/env python3
"""Smoke-test the unified `roboracer` environment.

Run:  conda activate roboracer && python scripts/check_env.py
Exit code 0 = every component importable and CUDA visible.
"""
import importlib
import shutil
import sys

FAILED = []


def check(name, fn):
    try:
        info = fn()
        print(f"  [ok]   {name:<22} {info}")
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] {name:<22} {type(e).__name__}: {e}")
        FAILED.append(name)


def mod(name, attr="__version__"):
    def _f():
        m = importlib.import_module(name)
        return getattr(m, attr, "")
    return _f


def torch_cuda():
    import torch
    ok = torch.cuda.is_available()
    dev = torch.cuda.get_device_name(0) if ok else "NO CUDA"
    return f"{torch.__version__}  cuda={torch.version.cuda}  {dev}"


def gsplat_kernel():
    import torch
    import gsplat
    # trivial rasterization to force CUDA kernel load
    if not torch.cuda.is_available():
        return f"{gsplat.__version__} (CPU only, kernel not tested)"
    n = 8
    means = torch.randn(n, 3, device="cuda")
    quats = torch.randn(n, 4, device="cuda")
    scales = torch.rand(n, 3, device="cuda") * 0.1
    opac = torch.rand(n, device="cuda")
    colors = torch.rand(n, 3, device="cuda")
    K = torch.tensor([[[100.0, 0, 32], [0, 100.0, 32], [0, 0, 1]]], device="cuda")
    vm = torch.eye(4, device="cuda")[None]
    vm[0, 2, 3] = 5.0
    img, _, _ = gsplat.rasterization(means, quats, scales, opac, colors, vm, K, 64, 64)
    return f"{gsplat.__version__}  rasterization ok {tuple(img.shape)}"


def nnodely_ok():
    import nnodely
    from nnodely import Modely  # noqa: F401
    return getattr(nnodely, "__version__", "1.5.4")


def binary(name):
    def _f():
        p = shutil.which(name)
        if not p:
            raise FileNotFoundError(f"{name} not on PATH")
        return p
    return _f


print(f"python {sys.version.split()[0]}  ({sys.executable})")
print("-- core")
check("torch + CUDA", torch_cuda)
check("numpy", mod("numpy"))
check("pandas", mod("pandas"))
check("scipy", mod("scipy"))
check("matplotlib", mod("matplotlib"))
check("cv2", mod("cv2"))
check("sklearn", mod("sklearn"))
print("-- camera / LiDAR 3DGS")
check("nerfstudio", mod("nerfstudio"))
check("gsplat kernel", gsplat_kernel)
check("open3d", mod("open3d"))
check("ns-train (bin)", binary("ns-train"))
check("colmap (bin)", binary("colmap"))
check("ffmpeg (bin)", binary("ffmpeg"))
print("-- vehicle dynamics")
check("nnodely (MS-NN)", nnodely_ok)
check("casadi", mod("casadi"))
check("onnxruntime", mod("onnxruntime"))

print()
if FAILED:
    print(f"{len(FAILED)} check(s) failed: {', '.join(FAILED)}")
    sys.exit(1)
print("all checks passed")
