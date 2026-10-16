#!/usr/bin/env python
# 16_raceline_retest_correct_checkpoint.py -- 2026-08-28
# Task 5: the original B/C/D raceline speed-envelope comparison (2026-08-26)
# turned out to have used the wrong GRU checkpoint (onboard-only 6in/4out,
# not the real 7in/5out Run30 -- see new_task_list_Lab.txt 2026-08-28 entry
# (not included in this repo) and project_vehicle_dynamics_v2 memory).
# Re-runs the same three configs with
# the now-fixed checkpoint. The original driver script that built the
# 8-lap-concatenated raceline path was never persisted (same situation as
# the task-2 comparison-video script) -- rebuilt here, reusing PurePursuit/
# GRUInference/StateIntegrator/run_v2 directly via importlib rather than
# reimplementing the closed-loop logic.
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import splprep, splev

_SCRIPT_DIR = Path(__file__).parent
_REPO_ROOT = _SCRIPT_DIR.parent

V2_DEMO_PATH = str(_SCRIPT_DIR / "10_v2_main_demo.py")
_spec = importlib.util.spec_from_file_location("v2_main_demo", V2_DEMO_PATH)
v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v2)

GRU_CHECKPOINT = str(_REPO_ROOT / "dynamics/deployed" / "trained_models" / "gru_run30_best.pth")
RACELINE_JSON = str(_REPO_ROOT / "reference" / "raceline_FTM_Halle" / "reftraj_FTM_Halle_min_time.json")
N_LAPS = 8
POINTS_PER_LAP = 2000
DURATION_S = 60.0


def build_raceline_path():
    d = json.loads(Path(RACELINE_JSON).read_text())
    px, py = np.array(d["pos_x"]), np.array(d["pos_y"])
    # closed loop (start == end); drop the duplicate last point before
    # spline fitting with per=1 (periodic), matching the closed shape.
    px, py = px[:-1], py[:-1]
    tck, _ = splprep([px, py], per=1, s=0)
    u = np.linspace(0, 1, POINTS_PER_LAP, endpoint=False)
    lap_x, lap_y = splev(u, tck)
    lap = np.stack([lap_x, lap_y], axis=1)
    return np.concatenate([lap] * N_LAPS, axis=0)


def run_config(name, ref_path, **pp_kwargs):
    gru = v2.GRUInference(GRU_CHECKPOINT, device="cpu")
    controller = v2.PurePursuit(wheelbase=0.3225, max_steering=0.29,
                                 train_speed_min=0.3, train_speed_max=2.5,
                                 **pp_kwargs)
    integrator = v2.StateIntegrator(dt=0.02)
    n_steps = int(DURATION_S / integrator.dt)
    df = v2.run_v2(gru, ref_path, controller, integrator, n_steps)
    skip = int(2.0 / integrator.dt)  # matches --skip-startup-s default
    te = df["tracking_error"].iloc[skip:]
    print(f"\n=== Config {name} ({pp_kwargs}) ===")
    print(f"  tracking_error: mean={te.mean():.3f}m  p95={te.quantile(0.95):.3f}m  max={te.max():.3f}m")
    print(f"  GRU vx: mean={df['vx_pred'].mean():.2f}  max={df['vx_pred'].max():.2f} m/s")
    return {"config": name, "mean_te": float(te.mean()), "p95_te": float(te.quantile(0.95)),
            "max_te": float(te.max()), "mean_vx": float(df["vx_pred"].mean())}


if __name__ == "__main__":
    ref_path = build_raceline_path()
    print(f"raceline path: {len(ref_path)} points ({N_LAPS} laps x {POINTS_PER_LAP}), "
          f"closed-loop, {DURATION_S:.0f}s per config")

    results = []
    results.append(run_config("B (speed=0.6, no adaptive-lookahead)",
                               ref_path, target_speed=0.6, adaptive_lookahead=False))
    results.append(run_config("C (speed=1.0, adaptive-lookahead)",
                               ref_path, target_speed=1.0, adaptive_lookahead=True))
    results.append(run_config("D (speed=0.8, adaptive-lookahead)",
                               ref_path, target_speed=0.8, adaptive_lookahead=True))

    print("\n" + "=" * 60)
    print("SUMMARY (correct Run30 checkpoint) vs 2026-08-26 baseline (wrong checkpoint)")
    print("=" * 60)
    baseline = {"B (speed=0.6, no adaptive-lookahead)": 0.078,
                "C (speed=1.0, adaptive-lookahead)": 0.061,
                "D (speed=0.8, adaptive-lookahead)": 0.033}
    for r in results:
        old = baseline[r["config"]]
        print(f"  {r['config']}: mean_te={r['mean_te']:.3f}m  (old checkpoint: {old:.3f}m)")

    out = _SCRIPT_DIR / "output" / "v2_raceline_retest_correct_checkpoint_20260828.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nsaved -> {out}")
