"""Adapter registry: model YAML `adapter:` key -> class (imported lazily so CPU-only runs need no GPU stack)."""

import importlib

_REGISTRY = {
    "constant_velocity": ("bench.adapters.constant_velocity", "ConstantVelocityAdapter"),
    "alpamayo": ("bench.adapters.alpamayo", "AlpamayoAdapter"),
    "flashdrive": ("bench.adapters.flashdrive", "FlashDriveAdapter"),
    "qwen_drive": ("bench.adapters.qwen_drive", "QwenDriveAdapter"),
}


def create_adapter(cfg: dict, task_prompt: str | None):
    name = cfg["adapter"]
    if name not in _REGISTRY:
        raise KeyError(f"Unknown adapter {name!r}; registered: {sorted(_REGISTRY)}")
    mod, cls = _REGISTRY[name]
    return getattr(importlib.import_module(mod), cls)(cfg, task_prompt)
