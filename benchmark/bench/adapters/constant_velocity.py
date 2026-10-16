"""GPU-free sanity baseline: extrapolate the last history velocity (and zero yaw rate)."""

import time

import numpy as np

from .base import ModelAdapter, Prediction


class ConstantVelocityAdapter(ModelAdapter):
    def load(self) -> None:
        pass

    def prepare(self, sample, num_samples):
        return sample

    def predict(self, inputs, num_samples) -> Prediction:
        t = time.perf_counter()
        h = inputs.hist_xyz[:, :2]
        v = (h[-1] - h[-2]) / 0.1
        steps = np.arange(1, inputs.fut_xyz.shape[0] + 1)[:, None] * 0.1
        xy = (h[-1][None] + v[None] * steps)[None].repeat(num_samples, axis=0)
        total = (time.perf_counter() - t) * 1000.0
        nan = float("nan")
        return Prediction(xy, {"vision": nan, "prefill": nan, "decode": nan, "diffusion": nan,
                               "other": total, "total": total, "decode_tokens": 0})
