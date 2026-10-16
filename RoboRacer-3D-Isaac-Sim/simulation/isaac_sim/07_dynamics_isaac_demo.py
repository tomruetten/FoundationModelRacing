#!/usr/bin/env python3
# 07_dynamics_isaac_demo.py -- 2026-08-13
"""
Isaac Sim client ("Prozess A") for the vehicle-dynamics V1 demo. Replays a
real recorded Car02 trajectory (Day1 test CSV, never used in training) and
at every 20ms step sends control inputs to the dynamics IPC servers,
collecting predictions alongside the real ground truth.

MS-NN and GRU-OL run open-loop/teacher-forced (always fed the REAL previous
state, never their own prior prediction). GRU-CL (2026-08-13, per user
decision) runs a genuine closed-loop/autoregressive rollout in parallel,
in the SAME pass: a second, independently-stateful GRU server instance
(own --gru-cl-queue-dir, own GRUInference buffer) is fed real data only
during its own warmup (rows [0:GRU_WARMUP), matching GRU-OL's warmup
exactly so both are compared over the identical row range); from
GRU_WARMUP onward it receives its OWN previous prediction for ALL FOUR
state channels (speed, yr, ax, ay) instead of ground truth. Feeding back
only yr/ax while leaving speed/ay pinned to ground truth would not be a
real closed loop -- errors couldn't accumulate through the other channels
that also feed the GRU's own history window, and the resulting CL-RMSE
would look artificially good. steering_cmd/current_motor are actuator
commands, not model outputs -- always real, in both OL and CL.

GRU-MoCap (2026-08-13, added after GRU-CL): a THIRD GRU variant, open-loop
only, using a genuine 7in/5out MoCap-state checkpoint (GRUMoCapInference in
gru_inference.py -- structurally the "Run30 family", see that class's
docstring for exact provenance/caveats) instead of the onboard-only 6in/4out
checkpoint the other two GRU paths use. Column requirements (v_x_local,
v_y_local, mocap_angular_velocity, a_y_corr, speed_cmd) were verified
present in the Day1 CSV before wiring this in, per explicit go/no-go
criteria from the user.

CAVEAT on yr_gru_mocap vs yr_true: GRU-MoCap's yaw-rate output predicts
`mocap_angular_velocity` (MoCap-derived), but yr_true in this CSV/plot is
`IMU_angular_velocity_z` (IMU-derived) -- the same physical quantity from
TWO DIFFERENT sensing pipelines, not guaranteed to agree exactly (this
project has a separately-flagged, unresolved ~mocap/IMU frame-offset
finding from earlier work). A RMSE(gru_mocap, yr_true) number therefore
conflates model error with sensor-frame disagreement -- do not present it
as directly comparable to MS-NN/GRU-onboard's IMU-vs-IMU RMSE without this
caveat.

Modeled directly on 1707-Test/scripts/isaac_sim_demo_okession.py's
structure (SimulationApp init incl. OMNI_KIT_ACCEPT_EULA ordering,
request/poll/response helper, DISPLAY=:0 requirement for --gui).

BEFORE RUNNING THIS:
  Terminal A (nnodely-env): conda run -n nnodely-env python3 05_msnn_dynamics_server.py \\
                  --model ../../dynamics/msnn/trained_models/msnn_exp9_phase2_backup \\
                  --queue-dir isaac_sim_output/msnn_queue
  Terminal B (roboracer env): conda run -n roboracer python3 06_gru_dynamics_server.py \\
                  --checkpoint ../../dynamics/msnn/trained_models/gru_run30_best.pth \\
                  --queue-dir isaac_sim_output/gru_queue
  Terminal C (roboracer env): conda run -n roboracer python3 06_gru_dynamics_server.py \\
                  --checkpoint ../../dynamics/msnn/trained_models/gru_run30_best.pth \\
                  --queue-dir isaac_sim_output/gru_cl_queue
  Terminal D (roboracer env): conda run -n roboracer python3 08_gru_mocap_dynamics_server.py \\
                  --checkpoint ../../dynamics/msnn/local_run___20260731204259241445.pth \\
                  --queue-dir isaac_sim_output/gru_mocap_queue
  Terminal E (Isaac Sim Python): python3 07_dynamics_isaac_demo.py

Output: isaac_sim_output/dynamics_comparison.csv + dynamics_comparison.png

KNOWN LIMITATIONS (see plan doc, deliberate V1 scope):
  - Not real-time: MS-NN's nnodely forward pass isn't batched, so 6304 rows
    takes several minutes wall-clock. This is an offline batch comparison,
    not a live demo.
  - Isaac Sim scene (Probe/experiments/isaac_sim_export/scene.usda) is not
    confirmed to be the same physical space as the Car02 Dynamik recordings
    (different dates, no doc links them) -- the car may visually float off
    or clip into the floor mesh. Cosmetic only, does not affect the
    yaw_rate/ax comparison numbers, which are V1's actual point.
  - RESOLVED 2026-08-13: MS-NN yr RMSE was ~0.408 vs the documented 0.1158
    due to an inference_msnn.py buffer-sizing bug (see
    05_msnn_dynamics_server.py's docstring for the root cause). Fixed --
    Day3 Val now reproduces 0.1158 exactly, Day1 cross-day is 0.1731.
  - RESOLVED 2026-08-13: GRU closed-loop rollout added (see module
    docstring above) -- V1 originally only had open-loop for both models.
    MS-NN closed-loop was deliberately NOT added: yaw_rate is structurally
    never an MS-NN input (confirmed via source), so yr closed-loop is a
    non-starter for it; only acceleration (ax) is self-referential, and
    doing that alone has no established baseline to validate against --
    left for a future pass if it turns out to matter.
  - Native Isaac Sim physics as a third comparison source, and real
    control-driven (not CSV-replay) closed-loop simulation, remain out of
    scope for V1 -- both are substantial separate follow-ups, not bugs.
  - ADDED 2026-08-13: GRU-MoCap (open-loop) as a third GRU variant -- see
    module docstring above for the checkpoint provenance caveat (likely a
    sibling of "Run30", not bit-identical) and the yr_gru_mocap-vs-yr_true
    sensor-frame caveat (MoCap-derived prediction compared against an
    IMU-derived ground truth column).
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ["OMNI_KIT_ACCEPT_EULA"] = "yes"

_SCRIPT_DIR = Path(__file__).parent
DYNAMIK_DIR = _SCRIPT_DIR.parent / "dynamics/msnn" / "Code von Felix" / "ma_jonathanmohr-main"
DEFAULT_CSV = DYNAMIK_DIR / "processed_data" / "session_20260721_20ms.csv"

# MS-NN warmup (NUM_SAMPLES_PAST) / GRU warmup (HISTORY) -- comparison rows
# are only recorded once BOTH buffers have filled, see module docstring.
MSNN_WARMUP = 20
GRU_WARMUP = 10
WARMUP = max(MSNN_WARMUP, GRU_WARMUP)


def request_dynamics(queue_dir: Path, idx: int, payload: dict, timeout_s: float = 30.0,
                      retry_delay_s: float = 0.01, max_retries: int = 20) -> dict:
    """Write req_NNNNNN.json, poll for res_NNNNNN.json, return its parsed
    contents. Retries on JSONDecodeError (defense in depth against reading
    a response file mid-write, even though the servers write atomically via
    tmp-then-rename -- cheap insurance)."""
    req_path = queue_dir / f"req_{idx:06d}.json"
    res_path = queue_dir / f"res_{idx:06d}.json"
    json.dump(payload, open(req_path, "w"))

    t0 = time.time()
    while not res_path.exists():
        if time.time() - t0 > timeout_s:
            raise TimeoutError(
                f"dynamics server did not respond within {timeout_s}s -- "
                f"is it running and watching {queue_dir}?")
        time.sleep(0.02)

    for attempt in range(max_retries):
        try:
            res = json.loads(res_path.read_text())
            break
        except json.JSONDecodeError:
            if attempt == max_retries - 1:
                raise
            time.sleep(retry_delay_s)
    res_path.unlink(missing_ok=True)
    return res


def load_trajectory(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col=0)
    df = df.dropna(subset=['IMU_linear_acceleration_x', 'IMU_linear_acceleration_y'])
    return df.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(DEFAULT_CSV),
                         help="real trajectory to replay (default: Day1 test set, never trained on)")
    parser.add_argument("--scene", required=True,
                         help="path to the exported .usda track scene (not included in this repo)")
    parser.add_argument("--msnn-queue-dir", required=True,
                         help="MUST match 05_msnn_dynamics_server.py's --queue-dir")
    parser.add_argument("--gru-queue-dir", required=True,
                         help="Open-loop GRU. MUST match a running 06_gru_dynamics_server.py's --queue-dir")
    parser.add_argument("--gru-cl-queue-dir", required=True,
                         help="Closed-loop GRU. MUST match a SECOND, separately-running "
                              "06_gru_dynamics_server.py instance's --queue-dir (own buffer state, "
                              "must not share a queue-dir with --gru-queue-dir)")
    parser.add_argument("--gru-mocap-queue-dir", required=True,
                         help="Open-loop MoCap-state GRU. MUST match 08_gru_mocap_dynamics_server.py's --queue-dir")
    parser.add_argument("--output-dir", default=str(_SCRIPT_DIR.parent / "isaac_sim_output"))
    parser.add_argument("--stride", type=int, default=1, help="use every Nth row of the real trajectory")
    parser.add_argument("--gui", action="store_true", default=False,
                         help="show a real window (default: headless)")
    args = parser.parse_args()
    args.headless = not args.gui

    import isaacsim
    from isaacsim import SimulationApp
    _FULL_EXPERIENCE = str(Path(isaacsim.__file__).parent / "apps" / "isaacsim.exp.full.kit")
    app = SimulationApp({
        "headless": args.headless,
        "width": 1280, "height": 800,
        **({"experience": _FULL_EXPERIENCE} if not args.headless else {}),
    })

    from omni.isaac.core import World
    from omni.isaac.core.prims import XFormPrim
    import omni.isaac.core.utils.stage as stage_utils

    world = World()
    stage_utils.add_reference_to_stage(os.path.abspath(args.scene), "/World/Track")
    car = world.scene.add(XFormPrim(
        prim_path="/World/Car02", name="car02", position=np.array([0.0, 0.0, 0.0]),
    ))
    world.reset()
    print(f"[isaac] scene ({args.scene}) + car loaded")
    if not args.headless:
        print("[isaac] GUI window should be open now (if you don't see it, you are not "
              "looking at the machine's :0 display -- SSH alone never shows it).")

    df = load_trajectory(Path(args.csv))
    df = df.iloc[::args.stride].reset_index(drop=True)
    print(f"[isaac] replaying {len(df)} rows from {args.csv} (stride={args.stride})")

    msnn_queue = Path(args.msnn_queue_dir)
    gru_queue = Path(args.gru_queue_dir)
    gru_cl_queue = Path(args.gru_cl_queue_dir)
    gru_mocap_queue = Path(args.gru_mocap_queue_dir)
    output_dir = Path(args.output_dir)
    for d in (msnn_queue, gru_queue, gru_cl_queue, gru_mocap_queue, output_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Episode reset on all four servers before replay starts.
    request_dynamics(msnn_queue, -1, {"step": -1, "reset": True})
    request_dynamics(gru_queue, -1, {"step": -1, "reset": True})
    request_dynamics(gru_cl_queue, -1, {"step": -1, "reset": True})
    request_dynamics(gru_mocap_queue, -1, {"step": -1, "reset": True})

    rows = []
    t_start = time.time()
    # Holds the CL server's own previous prediction, used as the NEXT
    # step's speed/yr/ax/ay once warmup is over (see module docstring).
    prev_cl_pred = None
    for i in range(len(df)):
        row = df.iloc[i]

        # Drive the Isaac Sim car proxy from the row's real MoCap world pose.
        pos = np.array([row['p_x'], row['p_y'], row['p_z']], dtype=np.float64)
        quat_wxyz = np.array([row['q_w'], row['q_x'], row['q_y'], row['q_z']], dtype=np.float64)
        car.set_world_pose(position=pos, orientation=quat_wxyz)
        for _ in range(2):
            world.step()

        msnn_res = request_dynamics(msnn_queue, i, {
            "step": i,
            "delta": float(row['steering_cmd']),
            "vel": float(row['odom_speed']),
            "acc": float(row['IMU_linear_acceleration_x']),
            "accy": float(row['IMU_linear_acceleration_y']),
            "crt": float(row['current_motor']),
        })
        gru_res = request_dynamics(gru_queue, i, {
            "step": i,
            "speed": float(row['odom_speed']),
            "yr": float(row['IMU_angular_velocity_z']),
            "ax": float(row['IMU_linear_acceleration_x']),
            "ay": float(row['IMU_linear_acceleration_y']),
            "steering_cmd": float(row['steering_cmd']),
            "current_motor": float(row['current_motor']),
        })

        # GRU-CL: during warmup (i < GRU_WARMUP), feed real data -- SAME
        # threshold as the OL server's own natural warmup, so both are
        # comparable from the same starting row. From GRU_WARMUP onward,
        # feed back ALL FOUR of the CL server's own previous predictions
        # (not just yr/ax) -- steering_cmd/current_motor stay real always.
        if i < GRU_WARMUP or prev_cl_pred is None:
            cl_state = {
                "speed": float(row['odom_speed']),
                "yr": float(row['IMU_angular_velocity_z']),
                "ax": float(row['IMU_linear_acceleration_x']),
                "ay": float(row['IMU_linear_acceleration_y']),
            }
        else:
            cl_state = {
                "speed": prev_cl_pred["speed_pred"],
                "yr": prev_cl_pred["yr_pred"],
                "ax": prev_cl_pred["ax_pred"],
                "ay": prev_cl_pred["ay_pred"],
            }
        gru_cl_res = request_dynamics(gru_cl_queue, i, {
            "step": i,
            **cl_state,
            "steering_cmd": float(row['steering_cmd']),
            "current_motor": float(row['current_motor']),
        })
        prev_cl_pred = gru_cl_res

        # GRU-MoCap: open-loop only, real MoCap-state columns from the row.
        gru_mocap_res = request_dynamics(gru_mocap_queue, i, {
            "step": i,
            "vx": float(row['v_x_local']),
            "vy": float(row['v_y_local']),
            "yr": float(row['mocap_angular_velocity']),
            "ax": float(row['IMU_linear_acceleration_x']),
            "ay": float(row['a_y_corr']),
            "speed_cmd": float(row['speed_cmd']),
            "steering_cmd": float(row['steering_cmd']),
        })

        if i >= WARMUP:
            rows.append({
                "t": i * 0.02 * args.stride,
                "yr_true": float(row['IMU_angular_velocity_z']),
                # BUGFIX 2026-08-13: GRU-MoCap predicts mocap_angular_velocity,
                # NOT IMU_angular_velocity_z -- two different sensors for the
                # same physical quantity. Comparing gru_mocap's prediction
                # against yr_true (IMU) conflates real model error with
                # sensor-frame disagreement (measured: the two ground-truth
                # sensors themselves disagree by RMS 0.780 rad/s, LARGER than
                # gru_mocap's RMSE against the wrong column ever was). Record
                # gru_mocap's own correct target separately so RMSE against it
                # is computed against the right column, not retrofitted later.
                "yr_true_mocap": float(row['mocap_angular_velocity']),
                "yr_msnn": msnn_res["yr_pred"],
                "yr_gru": gru_res["yr_pred"],
                "yr_gru_cl": gru_cl_res["yr_pred"],
                "yr_gru_mocap": gru_mocap_res["yr_pred"],
                "ax_true": float(row['IMU_linear_acceleration_x']),
                "ax_msnn": msnn_res["ax_pred"],
                "ax_gru": gru_res["ax_pred"],
                "ax_gru_cl": gru_cl_res["ax_pred"],
                "ax_gru_mocap": gru_mocap_res["ax_pred"],
            })

        if i % 100 == 0:
            elapsed = time.time() - t_start
            print(f"  [{i:5d}/{len(df)}]  ({elapsed:.1f}s elapsed)")

    out_csv = output_dir / "dynamics_comparison.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nDone: {len(rows)} comparison rows (first {WARMUP} rows skipped as model warmup) -> {out_csv}")

    # Plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    dfc = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle(f'Vehicle Dynamics V1 -- Isaac Sim replay ({Path(args.csv).name})',
                 fontsize=13, fontweight='bold')
    for ax_idx, (key, label, unit) in enumerate([('yr', 'psi_dot (yaw rate)', 'rad/s'),
                                                   ('ax', 'a_x (longitudinal accel)', 'm/s^2')]):
        ax = axes[ax_idx]
        ax.plot(dfc['t'], dfc[f'{key}_true'], color='#333333', lw=1.0, alpha=0.7, label='Ground Truth (IMU)')
        ax.plot(dfc['t'], dfc[f'{key}_msnn'], color='#1B4FBA', lw=1.0, alpha=0.85, label='MS-NN')
        ax.plot(dfc['t'], dfc[f'{key}_gru'], color='#15803D', lw=1.0, alpha=0.85, label='GRU onboard (open-loop)')
        ax.plot(dfc['t'], dfc[f'{key}_gru_cl'], color='#B45309', lw=1.0, alpha=0.85, ls='--', label='GRU onboard (closed-loop)')
        ax.plot(dfc['t'], dfc[f'{key}_gru_mocap'], color='#7C3AED', lw=1.0, alpha=0.85, label='GRU MoCap (open-loop)')
        if key == 'yr':
            # GRU-MoCap's real target is mocap_angular_velocity, not the IMU
            # column used for the other three models -- plot it too so the
            # purple GRU-MoCap line is compared against the right reference,
            # not silently against a different sensor's signal.
            ax.plot(dfc['t'], dfc['yr_true_mocap'], color='#999999', lw=1.0,
                    alpha=0.6, ls=':', label='Ground Truth (MoCap, for GRU-MoCap only)')
        ax.set_ylabel(f'{label} [{unit}]')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('t [s]')
    plt.tight_layout()
    out_png = output_dir / "dynamics_comparison.png"
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"Plot saved: {out_png}")

    if not args.headless:
        print("[isaac] replay finished -- window stays open, car parked at last "
              "row's pose. Press Ctrl+C in this terminal to close it.")
        try:
            while app.is_running():
                world.step(render=True)
        except KeyboardInterrupt:
            print("\n[isaac] closing")

    app.close()


if __name__ == "__main__":
    main()
