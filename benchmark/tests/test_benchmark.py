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


def test_resolve_repo_relative_vlm(tmp_path):
    from bench.hf_utils import resolve_repo_relative_vlm as resolve

    # Hub id of the VLM or absolute path: unchanged, nothing downloaded
    cfg = {"vlm_name_or_path": "Qwen/Qwen3-VL-8B-Instruct"}
    boom = lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not be called"))
    assert resolve(cfg, "nvidia/Alpamayo-1.5-10B", file_exists=lambda r, f: False, snapshot_download=boom) == \
        "nvidia/Alpamayo-1.5-10B" and cfg["vlm_name_or_path"] == "Qwen/Qwen3-VL-8B-Instruct"
    cfg = {"vlm_name_or_path": str(tmp_path / "abs")}
    assert resolve(cfg, "x/y", file_exists=boom, snapshot_download=boom) == "x/y"
    assert resolve({}, "x/y", file_exists=boom, snapshot_download=boom) == "x/y"

    # relative "vlm" with a Hub id: snapshot is downloaded, path points into it
    snap = tmp_path / "snap"
    (snap / "vlm").mkdir(parents=True)
    cfg = {"vlm_name_or_path": "vlm"}
    seen = {}
    out = resolve(cfg, "SIMFORGE-AI/SIMFORGE-D1.5",
                  file_exists=lambda repo, f: seen.setdefault("f", (repo, f)) and True,
                  snapshot_download=lambda repo, **kw: str(snap))
    assert out == str(snap) and cfg["vlm_name_or_path"] == str(snap / "vlm")
    assert seen["f"] == ("SIMFORGE-AI/SIMFORGE-D1.5", "vlm/config.json")

    # relative "vlm" with a local checkpoint directory (e.g. hf download --local-dir)
    cfg = {"vlm_name_or_path": "vlm"}
    assert resolve(cfg, str(snap), file_exists=boom, snapshot_download=boom) == str(snap)
    assert cfg["vlm_name_or_path"] == str(snap / "vlm")


def test_sample_seed_stable_and_distinct():
    from bench.seeding import sample_seed, set_seed

    assert sample_seed(42, 5) == sample_seed(42, 5)
    assert len({sample_seed(42, i) for i in range(200)}) == 200
    assert sample_seed(1, 5) != sample_seed(2, 5)
    set_seed(sample_seed(42, 5))
    a = np.random.rand(3)
    set_seed(sample_seed(42, 5))
    assert np.array_equal(a, np.random.rand(3))


def test_multi_seed_run_and_aggregate(tmp_path):
    make_recording(tmp_path / "rec")
    import run_benchmark as rb
    import yaml

    (tmp_path / "m.yaml").write_text(yaml.safe_dump({"name": "cv", "adapter": "constant_velocity"}))
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({"model": "m.yaml", "data": {"recording": "rec", "period_s": 0.5,
                                         "max_samples": 8}, "task_prompt": None, "num_samples": 2,
                                        "warmup": 1, "output_dir": "out", "seed": 5}))
    cfg = rb.load_config(cfg_path, None, [])
    # single run: seed from config ends up in the report
    single = rb.run(cfg, tmp_path)
    assert single["meta"]["seed"] == 5
    # several seeds: one folder per seed plus aggregate files
    cfg["seeds"] = [1, 2, 3]
    cfg["run_tag"] = "multi"
    agg = rb.run(cfg, tmp_path)
    root = next(p for p in (tmp_path / "out").iterdir() if "3seeds" in p.name)
    for s in (1, 2, 3):
        assert (root / f"seed_{s}" / "summary.md").exists()
        assert yaml.safe_load((root / f"seed_{s}" / "config.snapshot.yaml").read_text())["seed"] == s
    assert (root / "aggregate.md").exists() and (root / "aggregate.json").exists()
    ade = agg["metrics"]["ade (m)"]
    assert set(ade["per_seed"]) == {"1", "2", "3"}
    # the constant-velocity baseline is deterministic: identical across seeds, std 0
    assert ade["std_over_seeds"] == 0.0


def test_seeds_in_separate_processes(tmp_path):
    import subprocess

    import yaml

    make_recording(tmp_path / "rec")
    (tmp_path / "m.yaml").write_text(yaml.safe_dump({"name": "cv", "adapter": "constant_velocity",
                                                      "separate_processes": True}))
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({"model": "m.yaml", "data": {"recording": "rec", "period_s": 0.5,
                                         "max_samples": 8}, "task_prompt": None, "num_samples": 2,
                                        "warmup": 1, "output_dir": "out", "run_tag": "proc"}))
    script = Path(__file__).resolve().parents[1] / "run_benchmark.py"
    subprocess.run([sys.executable, str(script), "--config", str(cfg_path), "--seeds", "4", "5"],
                   check=True, capture_output=True)
    root = next(p for p in (tmp_path / "out").iterdir() if "2seeds" in p.name)
    for s in (4, 5):
        assert (root / f"seed_{s}" / "summary.md").exists()
        assert yaml.safe_load((root / f"seed_{s}" / "config.snapshot.yaml").read_text())["seed"] == s
    agg = json.loads((root / "aggregate.json").read_text())
    assert set(agg["metrics"]["ade (m)"]["per_seed"]) == {"4", "5"}


def test_resume_completes_missing_seed(tmp_path):
    import shutil
    import subprocess

    import yaml

    make_recording(tmp_path / "rec")
    (tmp_path / "m.yaml").write_text(yaml.safe_dump({"name": "cv", "adapter": "constant_velocity",
                                                      "separate_processes": True}))
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({"model": "m.yaml", "data": {"recording": "rec", "period_s": 0.5,
                                         "max_samples": 8}, "task_prompt": None, "num_samples": 2,
                                        "warmup": 1, "output_dir": "out", "run_tag": "res"}))
    script = Path(__file__).resolve().parents[1] / "run_benchmark.py"
    run = lambda *extra: subprocess.run([sys.executable, str(script), "--config", str(cfg_path), "--seeds", "1", "2",
                                         "3", *extra], check=True, capture_output=True, text=True)
    run()
    root = next(p for p in (tmp_path / "out").iterdir() if "3seeds" in p.name)
    # simulate a failed seed 3 (no result, no aggregate)
    shutil.rmtree(root / "seed_3")
    (root / "aggregate.json").unlink()
    (root / "aggregate.md").unlink()
    before = (root / "seed_1" / "summary.json").stat().st_mtime_ns
    out = run("--resume", str(root))
    assert (root / "seed_3" / "summary.json").exists() and (root / "aggregate.md").exists()
    assert (root / "seed_1" / "summary.json").stat().st_mtime_ns == before   # finished seeds are not rerun
    assert set(json.loads((root / "aggregate.json").read_text())["metrics"]["ade (m)"]["per_seed"]) == {"1", "2", "3"}
    assert "skipping" in out.stderr


def test_metrics_horizon_beyond_plan_is_nan():
    gt = np.stack([np.arange(64) * 0.1, np.zeros(64)], -1)
    pred = np.stack([gt[:50] + np.array([0, 1.0])])  # a 5 s plan (Qwen-Drive)
    m = trajectory_metrics(pred, gt, [2.0, 5.0, 6.4])
    assert m["pred_horizon_s"] == pytest.approx(5.0) and m["ade"] == pytest.approx(1.0)
    assert m["ade@5s"] == pytest.approx(1.0)
    assert np.isnan(m["ade@6.4s"]) and np.isnan(m["min_ade@6.4s"])


def test_frame_spacing_half_second(tmp_path):
    make_recording(tmp_path)
    s = build_samples(tmp_path, period_s=0.5, frame_dt_s=0.5)
    assert s, "no samples"
    import re

    steps = [int(re.search(r"ep0_(\d+)", p.name).group(1)) for p in s[3].frame_paths]
    assert np.allclose(np.diff(steps) / 50, 0.5, atol=0.04)
    assert abs(steps[-1] / 50 - s[3].t0) < 0.04
    assert len(s[3].hist_xyz) == 16  # poses stay at 10 Hz


def test_qwen_ego_inputs_circle(tmp_path):
    from bench.adapters.qwen_drive import ego_inputs

    w = make_recording(tmp_path, radius=5.0, speed=2.0)
    s = build_samples(tmp_path, period_s=0.5)[5]
    pose, vel, acc = ego_inputs(s.hist_xyz, s.hist_rot)
    assert pose.shape == (16, 3) and vel.shape == (16, 2) and acc.shape == (16, 2)
    assert np.allclose(pose[-1], 0)
    assert np.allclose(pose[:, 2], np.unwrap(pose[:, 2]))
    assert pose[0, 2] == pytest.approx(-w * 1.5, abs=1e-3)   # left turn: older headings are negative
    assert np.allclose(vel[:, 0], 2.0, atol=0.02) and np.allclose(vel[:, 1], 0, atol=0.02)  # body frame
    # centripetal acceleration points left (+y in the body frame): v^2 / r = 0.8
    assert np.allclose(acc[2:-1, 1], 0.8, atol=0.05) and np.allclose(acc[2:-1, 0], 0, atol=0.05)


def test_qwen_registered_and_prompt_dropped():
    import run_benchmark as rb
    from bench.adapters import create_adapter

    a = create_adapter({"adapter": "qwen_drive", "mode": "direct_planning"}, "x")  # lazy: no torch/qwen_drive needed
    assert a.frame_spacing_s == 0.5 and not a.uses_task_prompt
    cfg = {"model_cfg": {"name": "q", "adapter": "qwen_drive"}, "task_prompt": "drive", "run_tag": "track_prompt"}
    rb.apply_adapter_defaults(cfg)
    assert cfg["task_prompt"] is None and cfg["run_tag"] == "no_prompt"
    cfg = {"model_cfg": {"name": "cv", "adapter": "constant_velocity"}, "task_prompt": "drive", "run_tag": "t"}
    rb.apply_adapter_defaults(cfg)
    assert cfg["task_prompt"] == "drive" and cfg["run_tag"] == "t"


def test_qwen_model_yamls_load():
    import yaml

    for name, mode, planner in (("qwen_drive_direct", "direct_planning", "planner-sft"),
                                ("qwen_drive_reasoning", "reasoning_planning", "planner-sft"),
                                ("qwen_drive_reasoning_rl", "reasoning_planning", "planner-rl")):
        c = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/models" / f"{name}.yaml").read_text())
        assert c["name"] == name and c["adapter"] == "qwen_drive" and c["mode"] == mode and c["planner"] == planner


def _cv_plan(direction, tmp_path, **cfg):
    from bench.adapters import create_adapter

    w = make_recording(tmp_path, radius=5.0, speed=2.0)
    s = build_samples(tmp_path, period_s=0.5)[5]
    a = create_adapter({"adapter": "constant_velocity", "direction": direction, **cfg}, None)
    a.load()
    return a.predict(a.prepare(s, 1), 1).xy[0], s, w


def test_constant_velocity_variants_on_a_circle(tmp_path):
    # circle, v = 2 m/s, r = 5 m: velocity and heading coincide, so all three start the same way
    for d in ("velocity", "heading", "ctrv"):
        xy, s, w = _cv_plan(d, tmp_path / d)
        assert xy.shape == (64, 2) and np.allclose(xy[0], [0.2, 0], atol=0.02)
    xy_h, s, w = _cv_plan("heading", tmp_path / "h2")
    xy_c, _, _ = _cv_plan("ctrv", tmp_path / "c2")
    gt = s.fut_xyz[:, :2]
    err = lambda p: np.linalg.norm(p - gt, axis=1)
    assert err(xy_c)[-1] < 0.3 < 1.0 < err(xy_h)[-1]  # constant yaw rate follows the circle, a straight line leaves it
    assert np.isclose(xy_h[-1, 1], 0)  # heading baseline never moves sideways


def test_heading_baseline_ignores_side_slip(tmp_path):
    from bench.adapters import create_adapter
    from bench.dataset import Sample

    n = 16  # moving along y (90 deg beside the heading, x axis)
    hist = np.zeros((n, 3)); hist[:, 1] = np.arange(n) * 0.1 * 1.5
    hist -= hist[-1]
    rot = np.tile(np.eye(3), (n, 1, 1))
    s = Sample(0, 0, 0.0, [], hist.astype(np.float32), rot.astype(np.float32), np.zeros((64, 3), np.float32), 1.5)
    out = {}
    for d in ("velocity", "heading"):
        a = create_adapter({"adapter": "constant_velocity", "direction": d}, None)
        a.load()
        out[d] = a.predict(a.prepare(s, 1), 1).xy[0]
    assert out["velocity"][-1, 0] == pytest.approx(0, abs=1e-6) and out["velocity"][-1, 1] == pytest.approx(9.6, abs=1e-3)
    assert out["heading"][-1, 0] == pytest.approx(9.6, abs=1e-3) and out["heading"][-1, 1] == pytest.approx(0, abs=1e-6)
