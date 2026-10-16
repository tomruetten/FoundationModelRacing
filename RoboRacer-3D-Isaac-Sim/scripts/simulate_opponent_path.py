#!/usr/bin/env python3
"""Drive an opponent around a recorded lap with the learned GRU dynamics instead of replaying
the ego path with a fixed gap.

Reference path = the recorded camera trajectory projected onto the scene's floor plane.
Controller = Pure Pursuit (simulation/closed_loop/10_v2_main_demo.py) with the recorded ego
speed as per-point target speed and a gap controller that keeps the opponent `--gap` metres
ahead of where the ego was at that moment (race-like, collision free). Plant = the deployed
GRU (state: v_x, v_y, yaw rate, a_x, a_y; inputs: speed and steering commands), integrated
at 50 Hz. The integrated motion is then made vehicle-plausible for rendering: body heading =
travel direction minus a small, low-passed sideslip (|beta| <= 4 deg), speed and yaw rate
low-passed at 5 Hz, accelerations from the smoothed kinematics (not the raw GRU outputs).
Output: a JSON trajectory in the scene's world frame that the render demo and the Isaac
clients play back (--opponent-trajectory), with everything the object animation needs.

    conda activate roboracer && source scripts/env.sh
    python scripts/simulate_opponent_path.py --transforms $NEW1_ROOT/dataset/transforms.json \\
        --floor-plane $NEW1_ROOT/dataset/floor_plane.json --gru dynamics/deployed/trained_models/gru_run30_best.pth \\
        --target-speed 1.5 --duration 40 --start-frame 560 --lead 2.5 --out $NEW1_ROOT/dataset/opponent_gru.json
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simulation" / "isaac_sim"))
import object_splats as osp  # noqa: E402


def load_v2():
    spec = importlib.util.spec_from_file_location("v2demo", ROOT / "simulation" / "closed_loop" / "10_v2_main_demo.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transforms", required=True)
    ap.add_argument("--floor-plane", required=True)
    ap.add_argument("--plant", choices=["kinematic", "gru"], default="kinematic",
                    help="kinematic: bicycle model with vehicle limits (default, stable at race pace); "
                         "gru: deployed GRU dynamics as plant (unstable above ~1 m/s on the tight New1 track)")
    ap.add_argument("--gru", default=str(ROOT / "dynamics/deployed/trained_models/gru_run30_best.pth"))
    ap.add_argument("--a-max", type=float, default=3.0, help="kinematic: max acceleration [m/s2]")
    ap.add_argument("--brake-max", type=float, default=4.0, help="kinematic: max deceleration [m/s2]")
    ap.add_argument("--a-lat-max", type=float, default=4.0, help="kinematic: lateral acceleration limit -> corner speed")
    ap.add_argument("--steer-rate", type=float, default=3.0, help="kinematic: steering rate limit [rad/s]")
    ap.add_argument("--max-steer", type=float, default=0.30)
    ap.add_argument("--target-speed", type=float, default=None,
                    help="constant target speed [m/s]; default: the recorded ego speed along the path")
    ap.add_argument("--speed-factor", type=float, default=1.0, help="scale on the recorded ego speed profile")
    ap.add_argument("--gap", type=float, default=2.5, help="distance to keep ahead of the ego's recorded progress [m]")
    ap.add_argument("--gap-gain", type=float, default=0.8, help="speed correction per metre of gap error [1/s]")
    ap.add_argument("--max-sideslip-deg", type=float, default=6.0)
    ap.add_argument("--smooth-hz", type=float, default=5.0, help="low-pass cutoff for speed / yaw rate / sideslip")
    ap.add_argument("--lookahead", type=float, default=0.6)
    ap.add_argument("--wheelbase", type=float, default=0.33)
    ap.add_argument("--duration", type=float, default=40.0, help="simulated seconds")
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--start-frame", type=int, default=0, help="ego trajectory frame at which the opponent's t=0 begins")
    ap.add_argument("--frame-rate", type=float, default=15.0, help="frame rate of the ego trajectory (New1: 15 Hz)")
    ap.add_argument("--lead", type=float, default=2.5, help="start this many metres ahead along the path")
    ap.add_argument("--lateral", type=float, default=0.0, help="reference path offset to the left [m]")
    ap.add_argument("--laps", type=int, default=3, help="repeat the reference lap this often")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    v2 = load_v2() if args.plant == "gru" else None
    meta = json.load(open(args.transforms))
    frames = meta["frames"]
    plane = osp.FloorPlane.load(args.floor_plane)
    # floor-aligned 2-D frame: origin = plane point, x along the first camera heading projected, z = normal
    C = np.array([np.array(f["transform_matrix"])[:3, 3] for f in frames], dtype=np.float64)
    P0 = plane.project(C)
    fwd0 = -np.array(frames[0]["transform_matrix"])[:3, 2]
    ex = fwd0 - np.dot(fwd0, plane.normal) * plane.normal; ex /= np.linalg.norm(ex)
    ey = np.cross(plane.normal, ex)
    R_lw = np.stack([ex, ey, plane.normal])                     # world -> levelled
    o = plane.point
    path2d = (P0 - o) @ R_lw.T
    from scipy.ndimage import uniform_filter1d
    t_ego = np.arange(len(frames)) / args.frame_rate
    # lateral offset and resampling at 5 cm; recorded ego speed and time along the path
    d = np.diff(path2d[:, :2], axis=0); heading = np.arctan2(d[:, 1], d[:, 0]); heading = np.r_[heading, heading[-1]]
    left = np.c_[-np.sin(heading), np.cos(heading)]
    xy = path2d[:, :2] + args.lateral * left
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    ego_speed = uniform_filter1d(np.gradient(seg) / np.gradient(t_ego), 15)     # 1 s window
    s_new = np.arange(0, seg[-1], 0.05)
    ref = np.c_[np.interp(s_new, seg, xy[:, 0]), np.interp(s_new, seg, xy[:, 1])]
    ref_v = np.interp(s_new, seg, ego_speed) * args.speed_factor
    if args.target_speed:
        ref_v = np.full_like(ref_v, args.target_speed)
    ref_v = np.clip(ref_v, 0.4, 2.5)
    closed = np.linalg.norm(xy[0] - xy[-1]) < 1.0
    print(f"reference path: {len(ref)} points, {seg[-1]:.1f} m, {'closed lap' if closed else 'open'}; ego speed along it "
          f"mean {ego_speed.mean():.2f} max {ego_speed.max():.2f} m/s")

    # start pose: `lead` metres ahead of the ego's position at the start frame
    s_ego0 = seg[min(args.start_frame, len(seg) - 1)]
    i0 = int(np.clip(np.searchsorted(s_new, s_ego0 + args.lead), 0, len(ref) - 3))
    x0, y0 = ref[i0]
    th0 = float(np.arctan2(*(ref[i0 + 2] - ref[i0])[::-1]))
    v0 = float(ref_v[i0])

    # corner speed limit from the path curvature (smoothed over 0.5 m)
    tang = np.gradient(ref, axis=0); head_ref = np.unwrap(np.arctan2(tang[:, 1], tang[:, 0]))
    kappa = uniform_filter1d(np.gradient(head_ref) / 0.05, 10)
    v_corner = np.sqrt(args.a_lat_max / np.maximum(np.abs(kappa), 1e-3))
    ref_v = uniform_filter1d(np.minimum(ref_v, v_corner), 20)

    from scipy.spatial import cKDTree
    n_steps = int(args.duration / args.dt)
    rows, gaps = [], []
    i_prog = i0                       # progress index; the recorded path loops the track, so search forward only

    def progress(x, y):
        nonlocal i_prog
        lo, hi = max(i_prog - 4, 0), min(i_prog + 40, len(ref) - 1)      # +2 m window at 5 cm spacing
        i_prog = lo + int(np.argmin(np.linalg.norm(ref[lo:hi + 1] - [x, y], axis=1)))
        return i_prog

    def gap_at(t, i_near):
        s_ego = np.interp(t_ego[min(args.start_frame, len(t_ego) - 1)] + t, t_ego, seg)
        return s_new[i_near] - s_ego

    if args.plant == "gru":
        gru = v2.GRUInference(args.gru)
        ctrl = v2.PurePursuit(wheelbase=args.wheelbase, lookahead=args.lookahead, target_speed=v0,
                              adaptive_lookahead=True, ld_min=0.45, ld_max=1.2, path_v=ref_v)
        integ = v2.StateIntegrator(dt=args.dt)
        integ.reset(x=x0, y=y0, theta=th0)
        state = {"vx": v0, "vy": 0.0, "yr": 0.0, "ax": 0.0, "ay": 0.0}
        for _ in range(gru.HISTORY):
            gru.step(vx=v0, vy=0.0, yr=0.0, ax=0.0, ay=0.0, speed_cmd=v0, steering_cmd=0.0)
        for k in range(n_steps):
            t = k * args.dt
            steer, speed = ctrl.compute(integ.x, integ.y, integ.theta, ref)
            gap = gap_at(t, progress(integ.x, integ.y)); gaps.append(gap)
            speed = float(np.clip(speed + args.gap_gain * (args.gap - gap), 0.3, 2.5))
            state = gru.step(vx=state["vx"], vy=state["vy"], yr=state["yr"], ax=state["ax"], ay=state["ay"],
                             speed_cmd=speed, steering_cmd=steer)
            integ.update(state["vx"], state["vy"], state["yr"])
            rows.append({"t": round(t, 4), "xl": integ.x, "yl": integ.y, "theta_raw": integ.theta,
                         "v_raw": state["vx"], "v_y_raw": state["vy"], "yaw_rate_raw": state["yr"], "steer_cmd": steer, "speed_cmd": speed})
    else:
        # kinematic bicycle at the rear axle with vehicle limits; pure pursuit with speed-dependent lookahead
        L = args.wheelbase
        x, y, th, v, delta = float(x0), float(y0), th0, v0, 0.0
        for k in range(n_steps):
            t = k * args.dt
            i_near = progress(x, y)
            if i_near >= len(ref) - 30:              # end of the recorded path: stop here
                print(f"reached the end of the recorded path at t = {t:.1f} s")
                break
            gap = gap_at(t, i_near); gaps.append(gap)
            v_cmd = float(np.clip(ref_v[min(i_near + 10, len(ref) - 1)] + args.gap_gain * (args.gap - gap), 0.3, 2.5))
            # lookahead point
            ld = float(np.clip(0.35 + 0.45 * v, 0.5, 1.4))
            j = i_near
            while j < len(ref) - 1 and np.hypot(*(ref[j] - [x, y])) < ld:
                j += 1
            dx, dy = ref[j] - [x, y]
            alpha = np.arctan2(dy, dx) - th
            alpha = np.arctan2(np.sin(alpha), np.cos(alpha))
            delta_cmd = float(np.clip(np.arctan(2 * L * np.sin(alpha) / max(ld, 1e-3)), -args.max_steer, args.max_steer))
            delta += float(np.clip(delta_cmd - delta, -args.steer_rate * args.dt, args.steer_rate * args.dt))
            acc = float(np.clip((v_cmd - v) / 0.4, -args.brake_max, args.a_max))
            v = max(v + acc * args.dt, 0.0)
            yaw_rate = v * np.tan(delta) / L
            beta = np.arctan(0.5 * np.tan(delta))                  # kinematic sideslip at the CoG (l_r = L/2)
            x += v * np.cos(th + beta) * args.dt
            y += v * np.sin(th + beta) * args.dt
            th += yaw_rate * args.dt
            rows.append({"t": round(t, 4), "xl": x, "yl": y, "theta_raw": th, "v_raw": v, "v_y_raw": v * np.sin(beta),
                         "yaw_rate_raw": yaw_rate, "steer_cmd": delta, "speed_cmd": v_cmd})
    tree_ref = cKDTree(ref)

    # ---- vehicle-plausible post-processing of the integrated motion
    from scipy.signal import butter, filtfilt
    b, a = butter(2, args.smooth_hz / (0.5 / args.dt))
    xl = np.array([r["xl"] for r in rows]); yl = np.array([r["yl"] for r in rows]); tt = np.array([r["t"] for r in rows])
    xs, ys = filtfilt(b, a, xl), filtfilt(b, a, yl)
    vx_, vy_ = np.gradient(xs, args.dt), np.gradient(ys, args.dt)
    v = np.hypot(vx_, vy_)
    travel = np.unwrap(np.arctan2(vy_, vx_))
    beta = np.arctan2(np.array([r["v_y_raw"] for r in rows]), np.maximum(np.array([r["v_raw"] for r in rows]), 0.2))
    beta = np.clip(filtfilt(b, a, beta), -np.radians(args.max_sideslip_deg), np.radians(args.max_sideslip_deg))
    theta = travel - beta                                     # body points slightly inside the travel direction
    yaw_rate = np.gradient(np.unwrap(theta), args.dt)
    a_x = filtfilt(b, a, np.gradient(v, args.dt))
    a_y = v * yaw_rate
    steer = filtfilt(b, a, np.clip(np.array([r["steer_cmd"] for r in rows]), -0.3, 0.3))
    for i, r in enumerate(rows):
        pw = o + R_lw.T @ np.array([xs[i], ys[i], 0.0])
        r.update({"x": pw[0], "y": pw[1], "z": pw[2], "theta_levelled": float(theta[i]), "v": float(v[i]),
                  "v_y": float(-v[i] * np.sin(beta[i])), "yaw_rate": float(yaw_rate[i]), "a_x": float(a_x[i]), "a_y": float(a_y[i]),
                  "steer": float(steer[i]), "sideslip": float(beta[i])})
        for kk in ("xl", "yl", "theta_raw", "v_raw", "v_y_raw", "yaw_rate_raw"):
            r.pop(kk)
    xy_sim = np.array([[r["x"], r["y"], r["z"]] for r in rows])
    sim2d = (xy_sim - o) @ R_lw.T
    err, _ = tree_ref.query(sim2d[:, :2])
    gaps = np.array(gaps)
    out = {"frame": "world frame of transforms.json; heading = theta_levelled about the floor normal, x axis = first camera heading",
           "floor_plane": str(Path(args.floor_plane).resolve()), "levelled_R_lw": R_lw.tolist(), "levelled_origin": o.tolist(),
           "plant": args.plant, "gru": str(Path(args.gru).resolve()) if args.plant == "gru" else None,
           "target_speed": args.target_speed, "dt": args.dt,
           "start_frame": args.start_frame, "frame_rate": args.frame_rate,
           "time_note": "t = 0 when the ego is at trajectory frame start_frame; ego frame i -> t = (i - start_frame) / frame_rate",
           "stats": {"tracking_error_median_m": float(np.median(err)), "tracking_error_p95_m": float(np.percentile(err, 95)),
                     "speed_mean": float(v.mean()), "speed_max": float(v.max()), "distance_m": float(np.sum(np.linalg.norm(np.diff(xy_sim, axis=0), axis=1))),
                     "gap_to_ego_min_m": float(gaps.min()), "gap_to_ego_mean_m": float(gaps.mean()),
                     "sideslip_max_deg": float(np.degrees(np.abs(beta).max())), "yaw_rate_std": float(yaw_rate.std())},
           "rows": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"))
    print(f"simulated {args.duration:.0f} s: {out['stats']['distance_m']:.1f} m driven, speed mean {v.mean():.2f} max {v.max():.2f} m/s, "
          f"tracking error median {np.median(err)*100:.0f} cm p95 {np.percentile(err,95)*100:.0f} cm, gap to ego min {gaps.min():.2f} mean {gaps.mean():.2f} m, "
          f"sideslip max {np.degrees(np.abs(beta).max()):.1f} deg -> {args.out}")


if __name__ == "__main__":
    main()
