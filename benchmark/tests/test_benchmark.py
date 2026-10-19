"""CPU-only tests: ego-frame transform, metrics, dataset slicing, end-to-end run with the constant-velocity baseline."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.dataset import build_samples, to_ego_frame  # noqa: E402
from bench.metrics import trajectory_metrics  # noqa: E402


def make_recording(root: Path, seconds=14.0, radius=5.0, speed=2.0):
    """Circle at constant speed, 50 Hz ticks, camera frames at 20 Hz (40/60 ms alternating)."""
    from PIL import Image

    (root / "frames").mkdir(parents=True)
    w = speed / radius
    cam_steps = set()
    t_cam = 0.0
    k = 0
    while t_cam < seconds:
        cam_steps.add(int(round(t_cam * 50)))
        t_cam += 0.04 if k % 2 == 0 else 0.06
        k += 1
    with open(root / "trajectory.jsonl", "w") as f:
        for step in range(int(seconds * 50)):
            t = step / 50
            ego = {"x": radius * np.sin(w * t), "y": radius * (1 - np.cos(w * t)), "yaw": w * t}
            f.write(json.dumps({"episode_id": 0, "step": step, "time_s": t, "vehicles": {"ego": ego}}) + "\n")
            if step in cam_steps:
                Image.fromarray(np.full((8, 8, 3), step % 255, np.uint8)).save(root / "frames" / f"ep0_{step:06d}.png")
    return w


def test_ego_frame_identity_and_forward():
    xyz, rot = to_ego_frame(np.array([1.0, 2.0]), np.array([1.0, 1.0]), np.array([0.5, 0.5]), 2.0, 1.0, 0.5)
    assert np.allclose(xyz[-1], 0) and np.allclose(rot[-1], np.eye(3))
    # a point 1 m ahead along the heading is +x in the ego frame
    xyz, _ = to_ego_frame(np.array([2.0 + np.cos(0.5)]), np.array([1.0 + np.sin(0.5)]), np.array([0.5]), 2.0, 1.0, 0.5)
    assert np.allclose(xyz[0], [1, 0, 0], atol=1e-9)


def test_metrics():
    gt = np.stack([np.arange(64) * 0.1, np.zeros(64)], -1)
    pred = np.stack([gt, gt + np.array([0, 1.0]), gt + np.array([0, 2.0])])
    m = trajectory_metrics(pred, gt, [2.0, 6.4])
    assert m["ade"] == pytest.approx(0) and m["min_ade"] == pytest.approx(0)
    pred2 = pred[1:]  # sample 0 now off by 1 m, sample 1 by 2 m
    m2 = trajectory_metrics(pred2, gt, [2.0])
    assert m2["ade"] == pytest.approx(1.0) and m2["min_ade"] == pytest.approx(1.0)
    assert m2["fde"] == pytest.approx(1.0) and m2["ade@2s"] == pytest.approx(1.0)


def test_dataset_circle(tmp_path):
    w = make_recording(tmp_path)
    samples = build_samples(tmp_path, period_s=0.5)
    assert len(samples) > 5
    s = samples[0]
    assert len(s.frame_paths) == 4 and s.hist_xyz.shape == (16, 3) and s.fut_xyz.shape == (64, 3)
    assert np.allclose(s.hist_xyz[-1], 0, atol=1e-6) and np.allclose(s.hist_rot[-1], np.eye(3), atol=1e-6)
    assert s.speed == pytest.approx(2.0, abs=0.05)
    # on a circle with yaw rate w the point 1 s ahead has y > 0 (turning left) and x > 0
    p = s.fut_xyz[9]
    assert p[0] > 1.5 and p[1] > 0.1
    # distance travelled along the arc after 6.4 s equals v * t (chord < arc, but close for r=5 m)
    assert np.hypot(*s.fut_xyz[-1][:2]) == pytest.approx(2 * 5 * np.sin(w * 6.4 / 2), abs=0.02)
    # consecutive samples are >= period apart
    assert all(b.t0 - a.t0 >= 0.45 for a, b in zip(samples, samples[1:]))


def test_end_to_end_constant_velocity(tmp_path):
    make_recording(tmp_path / "rec")
    import run_benchmark as rb
    import yaml

    cfg_path = tmp_path / "config.yaml"
    (tmp_path / "m.yaml").write_text(yaml.safe_dump({"name": "cv", "adapter": "constant_velocity"}))
    cfg_path.write_text(yaml.safe_dump({"model": "m.yaml", "data": {"recording": "rec", "period_s": 0.5,
                                         "max_samples": 10}, "task_prompt": "x", "num_samples": 3,
                                        "warmup": 1, "output_dir": "out"}))
    summary = rb.run(rb.load_config(cfg_path, None, ["num_samples=2"]), tmp_path)
    assert summary["n_scored"] == 9
    # chord vs arc: constant velocity leaves the circle, so ADE is positive but small at short horizons
    assert 0 < summary["accuracy_m"]["ade@2s"]["mean"] < summary["accuracy_m"]["ade"]["mean"]
    out = next((tmp_path / "out").iterdir())
    for name in ("per_sample.csv", "summary.json", "summary.md", "config.snapshot.yaml", "predictions.npz"):
        assert (out / name).exists()
    d = np.load(out / "predictions.npz")
    assert d["pred_xy"].shape[1:] == (2, 64, 2) and d["gt_xy"].shape[1:] == (64, 2) and d["hist_xy"].shape[1:] == (16, 2)
    assert len(d["idx"]) == summary["n_scored"] + 1  # warmup sample is saved too
