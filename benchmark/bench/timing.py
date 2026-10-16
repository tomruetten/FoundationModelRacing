"""Per-stage GPU timing via synchronised wall-clock hooks.

Stages are accumulated per prediction in a `StageTimer`. Hooks synchronise CUDA
before and after the timed region, so stage sums are exact but the hooks add a few
tens of microseconds of sync overhead; total latency is measured separately by the
adapter without hooks being the only source of truth.
"""

import functools
import time
from collections import defaultdict
from contextlib import contextmanager

STAGES = ("vision", "prefill", "decode", "diffusion")


def _sync() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


class StageTimer:
    def __init__(self) -> None:
        self.ms: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)

    def reset(self) -> None:
        self.ms.clear()
        self.calls.clear()

    @contextmanager
    def stage(self, name: str):
        _sync()
        t = time.perf_counter()
        try:
            yield
        finally:
            _sync()
            self.ms[name] += (time.perf_counter() - t) * 1000.0
            self.calls[name] += 1

    def wrap_method(self, obj, method: str, stage: str) -> None:
        """Replace `obj.method` by a version that times each call into `stage`."""
        orig = getattr(obj, method)

        @functools.wraps(orig)
        def timed(*args, **kwargs):
            with self.stage(stage):
                return orig(*args, **kwargs)

        setattr(obj, method, timed)

    def hook_module(self, module, stage: str) -> None:
        """Time every forward of a torch module into `stage`."""
        starts = []

        def pre(_m, _args, _kwargs=None):
            _sync()
            starts.append(time.perf_counter())

        def post(_m, _args, _kwargs=None, _out=None):
            _sync()
            self.ms[stage] += (time.perf_counter() - starts.pop()) * 1000.0
            self.calls[stage] += 1

        module.register_forward_pre_hook(lambda m, a: pre(m, a))
        module.register_forward_hook(lambda m, a, o: post(m, a, None, o))

    def hook_prefill_decode(self, module) -> None:
        """Time forwards of an autoregressive LLM: first call per prediction = prefill, rest = decode."""
        starts = []

        def pre(_m, _a):
            _sync()
            starts.append(time.perf_counter())

        def post(_m, _a, _o):
            _sync()
            dt = (time.perf_counter() - starts.pop()) * 1000.0
            stage = "prefill" if self.calls["prefill"] == 0 else "decode"
            self.ms[stage] += dt
            self.calls[stage] += 1

        module.register_forward_pre_hook(pre)
        module.register_forward_hook(post)

    def snapshot(self, total_ms: float) -> dict[str, float]:
        """Stage times in ms; missing stages are NaN, `other` = total - sum(stages)."""
        out = {s: (self.ms[s] if self.calls[s] else float("nan")) for s in STAGES}
        out["other"] = total_ms - sum(v for v in out.values() if v == v)
        out["total"] = total_ms
        out["decode_tokens"] = max(self.calls["decode"], 0)
        return out
