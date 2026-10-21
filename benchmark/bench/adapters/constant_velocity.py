"""GPU-free baselines that extrapolate the ego history. `direction` selects the variant:

    velocity  (default) last velocity vector, kept constant. Follows the direction of travel.
    heading   last speed (norm of the last step), driven straight along the vehicle heading (ego x axis).
    ctrv      last speed and the yaw rate of the last `yaw_rate_window_s` seconds of the heading history
              (constant turn rate and velocity), driven along the heading.

The recorded sim car travels up to ~17 deg beside its heading in a turn (side slip of the GRU dynamics), so `velocity`
and `heading` differ a lot there; the models see the heading in the pose history, so `heading` is the fair trivial baseline.
"""

import time

import numpy as np

from .base import ModelAdapter, Prediction

DT = 0.1


class ConstantVelocityAdapter(ModelAdapter):
    def load(self) -> None:
        self.direction = self.cfg.get("direction", "velocity")
        if self.direction not in ("velocity", "heading", "ctrv"):
            raise ValueError(f"direction must be velocity, heading or ctrv, got {self.direction!r}")
        self.window = max(int(round(float(self.cfg.get("yaw_rate_window_s", 0.5)) / DT)), 1)

    def prepare(self, sample, num_samples):
        return sample

    def _plan(self, hist_xyz, hist_rot, steps: int) -> np.ndarray:
        h = hist_xyz[:, :2].astype(np.float64)
        t = np.arange(1, steps + 1)[:, None] * DT
        if self.direction == "velocity":
            return h[-1][None] + (h[-1] - h[-2])[None] / DT * t
        speed = np.linalg.norm(h[-1] - h[-2]) / DT
        if self.direction == "heading":
            return h[-1][None] + np.concatenate([speed * t, np.zeros_like(t)], axis=1)
        yaw = np.unwrap(np.arctan2(hist_rot[:, 1, 0], hist_rot[:, 0, 0]))
        rate = (yaw[-1] - yaw[-1 - self.window]) / (DT * self.window)
        theta = yaw[-1] + rate * t[:, 0]
        return h[-1][None] + np.cumsum(np.stack([np.cos(theta), np.sin(theta)], axis=1) * speed * DT, axis=0)

    def predict(self, inputs, num_samples) -> Prediction:
        t = time.perf_counter()
        xy = self._plan(inputs.hist_xyz, inputs.hist_rot, inputs.fut_xyz.shape[0])[None].repeat(num_samples, axis=0)
        total = (time.perf_counter() - t) * 1000.0
        nan = float("nan")
        return Prediction(xy, {"vision": nan, "prefill": nan, "decode": nan, "diffusion": nan,
                               "other": total, "total": total, "decode_tokens": 0})
