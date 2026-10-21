"""Sim recording -> benchmark samples.

A recording is the output directory of `racing --save-frames`:
    trajectory.jsonl   one row per dynamics tick (episode_id, step, time_s, vehicles.ego.{x,y,yaw})
    frames/ep<E>_[step]<step>.png   ego camera frames (20 Hz, quantised to the 50 Hz tick)

Everything is resampled on the *time axis*, so the sim rate does not matter: for a sample at
camera time t0 the model gets frames at t0-0.3..t0 (nearest recorded frame), 16 history poses
at t0-1.5..t0 and 64 future poses t0+0.1..t0+6.4 (all @10 Hz, linearly interpolated), expressed
in the ego frame at t0 (x forward, y left; sim vehicle frame is FLU).
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HZ = 10.0
DT = 1.0 / HZ
FRAME_RE = re.compile(r"ep(\d+)_(?:step)?(\d+)\.png$")


@dataclass
class Sample:
    episode: int
    step: int
    t0: float
    frame_paths: list[Path]  # oldest first, last = t0
    hist_xyz: np.ndarray  # (H, 3) ego frame at t0, last row = origin
    hist_rot: np.ndarray  # (H, 3, 3)
    fut_xyz: np.ndarray  # (F, 3) ego frame at t0
    speed: float  # m/s at t0 (from history)

    def load_frames(self) -> np.ndarray:
        """(n_frames, H, W, 3) uint8."""
        from PIL import Image

        return np.stack([np.asarray(Image.open(p).convert("RGB")) for p in self.frame_paths])


def _yaw_rot(yaw: np.ndarray) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    r = np.zeros(yaw.shape + (3, 3))
    r[..., 0, 0], r[..., 0, 1], r[..., 1, 0], r[..., 1, 1], r[..., 2, 2] = c, -s, s, c, 1.0
    return r


def to_ego_frame(x, y, yaw, x0, y0, yaw0):
    """World poses -> ego frame at (x0, y0, yaw0). Returns xyz (N,3) and rot (N,3,3)."""
    dx, dy = np.asarray(x) - x0, np.asarray(y) - y0
    c, s = np.cos(yaw0), np.sin(yaw0)
    xyz = np.stack([c * dx + s * dy, -s * dx + c * dy, np.zeros_like(dx)], axis=-1)
    rot = _yaw_rot(np.asarray(yaw) - yaw0)
    return xyz, rot


class Episode:
    def __init__(self, ep: int, t, x, y, yaw, step_to_time):
        self.ep, self.t, self.x, self.y, self.yaw = ep, t, x, y, np.unwrap(yaw)
        self.step_to_time = step_to_time

    def pose_at(self, times: np.ndarray):
        return (np.interp(times, self.t, self.x), np.interp(times, self.t, self.y),
                np.interp(times, self.t, self.yaw))


def load_recording(rec_dir: str | Path) -> dict[int, Episode]:
    rec_dir = Path(rec_dir)
    rows: dict[int, list] = {}
    with open(rec_dir / "trajectory.jsonl") as f:
        for line in f:
            r = json.loads(line)
            ego = r["vehicles"]["ego"]
            rows.setdefault(r["episode_id"], []).append((r["step"], r["time_s"], ego["x"], ego["y"], ego["yaw"]))
    eps = {}
    for ep, rs in rows.items():
        a = np.array(sorted(rs), dtype=np.float64)
        _, idx = np.unique(a[:, 1], return_index=True)  # drop repeated timestamps (paused ticks)
        a = a[idx]
        eps[ep] = Episode(ep, a[:, 1], a[:, 2], a[:, 3], a[:, 4], {int(s): float(t) for s, t in zip(a[:, 0], a[:, 1])})
    return eps


def build_samples(rec_dir: str | Path, num_frames: int = 4, num_history: int = 16, num_future: int = 64,
                  period_s: float = 0.5, max_samples: int | None = None, min_speed: float = 0.0,
                  frame_tolerance_s: float = 0.04, frame_dt_s: float = DT) -> list[Sample]:
    """Cut samples from a recording, one every `period_s` seconds of sim time (0.1 = consecutive
    10 Hz windows, required by streaming models). The `num_frames` input frames end at t0 and are
    `frame_dt_s` apart (0.1 s for the Alpamayo models, 0.5 s for Qwen-Drive); poses stay at 10 Hz."""
    rec_dir = Path(rec_dir)
    eps = load_recording(rec_dir)
    frames: dict[int, list[tuple[float, Path]]] = {}
    for p in sorted((rec_dir / "frames").glob("ep*_*.png")):
        m = FRAME_RE.search(p.name)
        ep, step = int(m.group(1)), int(m.group(2))
        if ep in eps and step in eps[ep].step_to_time:
            frames.setdefault(ep, []).append((eps[ep].step_to_time[step], p))

    samples: list[Sample] = []
    for ep, fl in sorted(frames.items()):
        e = eps[ep]
        fl.sort()
        ft = np.array([t for t, _ in fl])
        next_t = -np.inf
        for i in range(len(fl)):
            t0 = ft[i]
            if t0 < next_t - frame_tolerance_s:
                continue
            frame_times = t0 - frame_dt_s * np.arange(num_frames - 1, -1, -1)
            hist_times = t0 - DT * np.arange(num_history - 1, -1, -1)
            fut_times = t0 + DT * np.arange(1, num_future + 1)
            if hist_times[0] < e.t[0] or fut_times[-1] > e.t[-1] or frame_times[0] < ft[0]:
                continue
            idx = np.abs(ft[None] - frame_times[:, None]).argmin(axis=1)
            if np.abs(ft[idx] - frame_times).max() > frame_tolerance_s:
                continue
            x, y, yaw = e.pose_at(np.concatenate([hist_times, fut_times]))
            x0, y0, yaw0 = x[num_history - 1], y[num_history - 1], yaw[num_history - 1]
            xyz, rot = to_ego_frame(x, y, yaw, x0, y0, yaw0)
            speed = float(np.linalg.norm(xyz[num_history - 1] - xyz[num_history - 2]) / DT)
            if speed < min_speed:
                continue
            next_t = t0 + period_s
            samples.append(Sample(ep, int(round(t0 / DT)), float(t0), [fl[j][1] for j in idx],
                                  xyz[:num_history].astype(np.float32), rot[:num_history].astype(np.float32),
                                  xyz[num_history:].astype(np.float32), speed))
            if max_samples and len(samples) >= max_samples:
                return samples
    return samples
