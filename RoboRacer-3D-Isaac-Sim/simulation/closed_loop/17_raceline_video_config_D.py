#!/usr/bin/env python
# 17_raceline_video_config_D.py -- 2026-09-02
"""
Task 5 raceline verification video. Re-runs run_v2 config D (target_speed=0.8,
adaptive_lookahead=True -- the best config from 16_raceline_retest_correct_
checkpoint.py, mean tracking_error=0.032m) with the CORRECT Run30 checkpoint
(gru_run30_best.pth, 7in/5out MoCap spec) and renders an animated top-down
replay (raceline vs actual closed-loop path, car marker + trail) plus a
tracking-error/speed telemetry panel, as an mp4.

Why this exists: 16_ only prints numbers to stdout, no video. The 3 existing
mp4s in raceline_FTM_Halle/videos/ (v2_raceline_test1/test2_D_tuned/
D_multilap_12x, all dated 2026-08-26) were recorded BEFORE the 2026-08-28
checkpoint-path fix, with the wrong onboard-only (6in/4out) checkpoint --
they do not verify the actual config-D result and must not be reused.

Usage:
  conda run -n roboracer python3 17_raceline_video_config_D.py
"""
import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
OUT_DIR = _SCRIPT_DIR / "output" / "videos"


def build_raceline_path():
    d = json.loads(Path(RACELINE_JSON).read_text())
    px, py = np.array(d["pos_x"]), np.array(d["pos_y"])
    # closed loop (start == end); drop the duplicate last point before
    # spline fitting with per=1 (periodic), matching 16_'s own construction.
    px, py = px[:-1], py[:-1]
    tck, _ = splprep([px, py], per=1, s=0)
    u = np.linspace(0, 1, POINTS_PER_LAP, endpoint=False)
    lap_x, lap_y = splev(u, tck)
    lap = np.stack([lap_x, lap_y], axis=1)
    return np.concatenate([lap] * N_LAPS, axis=0)


def main():
    ref_path = build_raceline_path()
    print(f"raceline path: {len(ref_path)} points ({N_LAPS} laps x {POINTS_PER_LAP}), closed-loop")

    gru = v2.GRUInference(GRU_CHECKPOINT, device="cpu")
    controller = v2.PurePursuit(wheelbase=0.3225, max_steering=0.29,
                                 train_speed_min=0.3, train_speed_max=2.5,
                                 target_speed=0.8, adaptive_lookahead=True)
    integrator = v2.StateIntegrator(dt=0.02)
    n_steps = int(DURATION_S / integrator.dt)
    df = v2.run_v2(gru, ref_path, controller, integrator, n_steps)

    skip = int(2.0 / integrator.dt)
    te = df["tracking_error"].iloc[skip:]
    print(f"\n=== Config D (target_speed=0.8, adaptive_lookahead=True) ===")
    print(f"  tracking_error: mean={te.mean():.3f}m  p95={te.quantile(0.95):.3f}m  max={te.max():.3f}m")
    print(f"  GRU vx: mean={df['vx_pred'].mean():.2f}  max={df['vx_pred'].max():.2f} m/s")

    # -------- render frames --------
    frames_dir = OUT_DIR / "_frames_tmp_configD"
    frames_dir.mkdir(parents=True, exist_ok=True)

    STRIDE = 2  # 50Hz sim (dt=0.02) -> 25fps video, real-time-paced (3000 steps -> 1500 frames -> 60s)
    FPS = int(round(1.0 / integrator.dt / STRIDE))
    idxs = list(range(0, len(df), STRIDE))
    print(f"[video] {len(df)} sim steps -> {len(idxs)} frames @ {FPS}fps "
          f"({len(idxs) / FPS:.1f}s output, real-time-paced)")

    margin = 1.0
    xlim = (float(ref_path[:, 0].min() - margin), float(ref_path[:, 0].max() + margin))
    ylim = (float(ref_path[:, 1].min() - margin), float(ref_path[:, 1].max() + margin))

    fig, (ax_map, ax_tel) = plt.subplots(1, 2, figsize=(14, 6), dpi=110)

    ax_map.plot(ref_path[:POINTS_PER_LAP, 0], ref_path[:POINTS_PER_LAP, 1],
                "k--", lw=1, alpha=0.5, label="Raceline (optimal, Ziad Elnaggar)")
    ax_map.set_xlim(*xlim)
    ax_map.set_ylim(*ylim)
    ax_map.set_aspect("equal")
    ax_map.set_xlabel("x [m]")
    ax_map.set_ylabel("y [m]")
    ax_map.set_title("Task 5: PurePursuit + GRU(Run30) Raceline Tracking")
    trail_line, = ax_map.plot([], [], "b-", lw=2, label="V2 closed-loop path")
    car_pt = ax_map.scatter([], [], c="red", s=60, zorder=5, label="Car")
    txt = ax_map.text(0.02, 0.02, "", transform=ax_map.transAxes, fontsize=9,
                       va="bottom", bbox=dict(fc="white", alpha=0.7))
    ax_map.legend(fontsize=8, loc="upper right")

    ax_tel.set_xlim(0, DURATION_S)
    te_max = float(df["tracking_error"].max()) * 1.2
    ax_tel.set_ylim(0, max(te_max, 0.1))
    ax_tel.set_xlabel("t [s]")
    ax_tel.set_ylabel("tracking error [m]", color="r")
    err_line, = ax_tel.plot([], [], "r-", lw=1.2, label="tracking error")
    ax_tel.axhline(float(te.mean()), color="r", ls="--", alpha=0.4)
    ax_spd = ax_tel.twinx()
    ax_spd.set_ylim(0, float(df["vx_pred"].max()) * 1.2)
    ax_spd.set_ylabel("vx [m/s]", color="b")
    spd_line, = ax_spd.plot([], [], "b-", lw=1.0, alpha=0.6, label="vx (GRU)")
    cursor = ax_tel.axvline(0, color="gray", lw=1)
    ax_tel.set_title(f"Tracking Error / Speed vs Time  (mean err={te.mean():.3f}m)")

    plt.tight_layout()

    t_arr = df["t"].to_numpy()
    x_arr = df["x"].to_numpy()
    y_arr = df["y"].to_numpy()
    te_arr = df["tracking_error"].to_numpy()
    vx_arr = df["vx_pred"].to_numpy()

    for fi, i in enumerate(idxs):
        trail_line.set_data(x_arr[:i + 1], y_arr[:i + 1])
        car_pt.set_offsets([[x_arr[i], y_arr[i]]])
        txt.set_text(f"t={t_arr[i]:5.1f}s   tracking_err={te_arr[i]:.3f}m   vx={vx_arr[i]:.2f}m/s")
        err_line.set_data(t_arr[:i + 1], te_arr[:i + 1])
        spd_line.set_data(t_arr[:i + 1], vx_arr[:i + 1])
        cursor.set_xdata([t_arr[i], t_arr[i]])
        fig.savefig(frames_dir / f"frame_{fi:05d}.png")
        if fi % 200 == 0:
            print(f"  [{fi:5d}/{len(idxs)}]")

    plt.close(fig)

    out_video = OUT_DIR / "v2_raceline_configD_correct_checkpoint_20260902.mp4"
    print("[video] encoding mp4...")
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(FPS),
        "-i", str(frames_dir / "frame_%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(out_video),
    ], check=True)
    print(f"[video] saved -> {out_video}")


if __name__ == "__main__":
    main()
