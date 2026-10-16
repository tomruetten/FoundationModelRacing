"""LiDAR 3DGS evaluation against real Livox Mid-360 scans.

Session models (2026-08-27, two tracks; the 2026-07-14 model has no raw scans available
locally and is only inventoried): for every 10th scan in the bag
     (the split the training script held out) the model is queried along the
     REAL scan's ray directions from the FAST-LIO odometry pose at that time:
        - hit rate (fraction of real rays the model also returns)
        - per-ray range error (median, RMSE, share within 10 cm)
        - Chamfer / F-score between the real and the synthetic scan
        - error vs range, error along the trajectory, a top-down overlay
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .common import DATA_LOCAL, REPO, Report, chamfer_fscore, load_module, log, raw_session_dir

MAX_RANGE = 6.0        # training range of the session models
VAL_EVERY_N = 10       # held-out scans in the session training script
ALPHA_THRESH = 0.99    # session models: raydrop head off, alpha filter on
MAX_SCANS = 60         # per model (runtime)


# ----------------------------------------------------------------------------- bag reading
def read_odometry(bag: Path, topic: str = "/Odometry"):
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore
    ts = get_typestore(Stores.ROS2_HUMBLE)
    t, p, q = [], [], []
    with Reader(bag) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        for conn, _, raw in reader.messages(connections=conns):
            m = ts.deserialize_cdr(raw, conn.msgtype)
            t.append(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)
            pp = m.pose.pose.position; qq = m.pose.pose.orientation
            p.append([pp.x, pp.y, pp.z]); q.append([qq.x, qq.y, qq.z, qq.w])
    return np.array(t), np.array(p), np.array(q)


def iter_scans(bag: Path, topic: str = "/livox/lidar"):
    """Yield (stamp_s, xyz (N,3) float32, intensity (N,)) for every PointCloud2 on the topic."""
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore
    ts = get_typestore(Stores.ROS2_HUMBLE)
    with Reader(bag) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        for conn, _, raw in reader.messages(connections=conns):
            m = ts.deserialize_cdr(raw, conn.msgtype)
            fields = {f.name: (f.offset, f.datatype) for f in m.fields}
            dt = np.dtype({"names": ["x", "y", "z"], "formats": ["<f4"] * 3,
                           "offsets": [fields["x"][0], fields["y"][0], fields["z"][0]], "itemsize": m.point_step})
            n = m.width * m.height
            arr = np.frombuffer(bytes(m.data), dtype=dt, count=n)
            xyz = np.stack([arr["x"], arr["y"], arr["z"]], 1).astype(np.float32)
            inten = None
            if "intensity" in fields:
                dti = np.dtype({"names": ["i"], "formats": ["<f4"], "offsets": [fields["intensity"][0]], "itemsize": m.point_step})
                inten = np.frombuffer(bytes(m.data), dtype=dti, count=n)["i"].astype(np.float32)
            yield m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, xyz, inten


# ----------------------------------------------------------------------------- ray-cast with per-ray bookkeeping
def cast_rays(renderer_mod, renderer, R_wl, t_wl, dirs, alpha_thresh=ALPHA_THRESH):
    """Same maths as LidarRenderer.render_along_rays but returns per-ray depth and validity."""
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    N = len(dirs)
    depth = np.zeros(N, np.float32); valid = np.zeros(N, bool)
    W, H = renderer_mod.VCAM_W, renderer_mod.VCAM_H
    for _, R_face in renderer_mod.CUBEMAP_FACES.items():
        pr = (R_face.T @ dirs.T).T
        x_cam, y_cam, z_cam = -pr[:, 1], -pr[:, 2], pr[:, 0]
        front = z_cam > 0.15
        if not front.any():
            continue
        u = renderer.K[0, 0] * x_cam / np.where(front, z_cam, 1) + renderer.K[0, 2]
        v = renderer.K[1, 1] * y_cam / np.where(front, z_cam, 1) + renderer.K[1, 2]
        inb = front & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
        idx = np.where(inb)[0]
        if len(idx) == 0:
            continue
        viewmat = renderer_mod._make_viewmat(R_wl, t_wl, R_face)
        d_img, _, a_img = renderer_mod._render_face(renderer.splats, viewmat, renderer.K, renderer.device, return_alpha=True)
        ui = np.clip(np.round(u[idx]).astype(int), 0, W - 1); vi = np.clip(np.round(v[idx]).astype(int), 0, H - 1)
        d = d_img[vi, ui]
        ok = (d > 0.1) & (a_img[vi, ui] >= alpha_thresh)
        # the face image stores z-depth along the face axis; range along the ray is z / cos(angle)
        depth[idx[ok]] = d[ok] / z_cam[idx[ok]]; valid[idx[ok]] = True
    return depth, valid


# ----------------------------------------------------------------------------- cases
def eval_prerendered(report: Report):
    """The 2026-07-14 model ships synthetic.ply / real_holdout.ply, but the two clouds cover
    different parts of the scene (checked: centroids 4 m apart, different extents), so a set
    distance between them is not a model error. Reported here only as an inventory note."""
    d = DATA_LOCAL / "New1/checkpoints/lidar_3dgs/synthetic_lidar"
    if (d / "synthetic.ply").exists():
        report.h(3, "2026-07-14 model")
        report.note("No raw LiDAR recording of the 2026-07-14 session is available locally, so this model cannot "
                    "be scored against real scans here (the 13.44 cm Chamfer in the original notes was computed "
                    "on the training machine). The shipped synthetic/real hold-out clouds cover different regions "
                    "and are not a valid pair for a set distance; skipped.")


def session_cases():
    cases = []
    for token, label in (("20260827_142014", "2026-08-27 track A (142014)"), ("20260827_143212", "2026-08-27 track B (143212)")):
        raw = raw_session_dir(token)
        if raw is None:
            continue
        odom = raw / "odometry_bag/recording/recording_0.db3"
        scans = raw / "all_data/all_data_0.db3"
        for exp in sorted((raw / "lidar_3dgs_experiments").glob("*/best.pt")) if (raw / "lidar_3dgs_experiments").exists() else []:
            if odom.exists() and scans.exists():
                cases.append(dict(label=f"{label} / {exp.parent.name}", ckpt=exp, odom=odom, scans=scans))
    return cases


def eval_session(report: Report, case: dict, lr_mod, quick: bool, fig_tag: str):
    from scipy.spatial.transform import Rotation, Slerp
    import matplotlib.pyplot as plt
    log(f"lidar: {case['label']}")
    renderer = lr_mod.LidarRenderer(checkpoint_path=case["ckpt"])
    t_od, p_od, q_od = read_odometry(case["odom"])
    slerp = Slerp(t_od, Rotation.from_quat(q_od))
    rows, overlay = [], None
    n_used = 0
    for i, (ts, xyz, inten) in enumerate(iter_scans(case["scans"])):
        if i % VAL_EVERY_N != 0 or ts < t_od[0] or ts > t_od[-1]:
            continue
        if quick and n_used >= 15 or n_used >= MAX_SCANS:
            break
        rng = np.linalg.norm(xyz, axis=1)
        keep = (rng > 0.3) & (rng < MAX_RANGE) & np.isfinite(rng)
        xyz, rng = xyz[keep], rng[keep]
        if len(xyz) < 500:
            continue
        R_wl = slerp(ts).as_matrix()
        t_wl = np.array([np.interp(ts, t_od, p_od[:, k]) for k in range(3)])
        depth, valid = cast_rays(lr_mod, renderer, R_wl, t_wl, xyz)
        err = depth[valid] - rng[valid]
        syn = (xyz[valid] / rng[valid, None]) * depth[valid, None]
        cf = chamfer_fscore(syn, xyz)
        rows.append(dict(scan=i, t=ts - t_od[0], n_real=len(xyz), hit_rate=float(valid.mean()),
                         range_err_median=float(np.median(np.abs(err))) if len(err) else np.nan,
                         range_err_rmse=float(np.sqrt(np.mean(err ** 2))) if len(err) else np.nan,
                         within10cm=float((np.abs(err) < 0.10).mean()) if len(err) else np.nan,
                         chamfer=cf["chamfer_m"], f5=cf["fscore@5cm"], f10=cf["fscore@10cm"], f20=cf["fscore@20cm"]))
        if overlay is None or (rows[-1]["chamfer"] < overlay[0] and n_used > 3):
            overlay = (rows[-1]["chamfer"], xyz, syn, rng[valid], np.abs(err))
        n_used += 1
    if not rows:
        report.note(f"{case['label']}: no usable scans"); return None
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(report.out_dir / f"lidar_{fig_tag}_per_scan.csv", index=False)
    summ = dict(n_scans=int(len(df)), hit_rate=float(df.hit_rate.mean()), range_err_median_cm=float(df.range_err_median.median() * 100),
                range_err_rmse_cm=float(np.sqrt(np.mean(df.range_err_rmse ** 2)) * 100), within10cm=float(df.within10cm.mean()),
                chamfer_cm=float(df.chamfer.mean() * 100), f10=float(df.f10.mean()), f20=float(df.f20.mean()))
    report.put("lidar", case["label"], value=summ)
    # figures
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    ax[0].plot(df.t, df.chamfer * 100, "o-", ms=3); ax[0].set_xlabel("t [s]"); ax[0].set_ylabel("Chamfer [cm]"); ax[0].grid(alpha=.3); ax[0].set_title("per held-out scan", fontsize=9)
    _, xyz_o, syn_o, rng_o, aerr_o = overlay
    bins = np.arange(0, MAX_RANGE + 0.01, 0.5)
    med = [np.median(aerr_o[(rng_o >= a) & (rng_o < b)]) * 100 if ((rng_o >= a) & (rng_o < b)).sum() > 20 else np.nan for a, b in zip(bins[:-1], bins[1:])]
    ax[1].bar(bins[:-1] + 0.25, med, 0.45); ax[1].set_xlabel("real range [m]"); ax[1].set_ylabel("median |range error| [cm]"); ax[1].grid(alpha=.3); ax[1].set_title("error vs range (one scan)", fontsize=9)
    ax[2].scatter(xyz_o[:, 0], xyz_o[:, 1], s=1, c="k", label="real"); ax[2].scatter(syn_o[:, 0], syn_o[:, 1], s=1, c="r", alpha=.6, label="model"); ax[2].set_aspect("equal"); ax[2].legend(fontsize=7); ax[2].set_title("top-down overlay (sensor frame)", fontsize=9)
    fig.suptitle(case["label"], fontsize=10)
    report.savefig(fig, f"lidar_{fig_tag}.png", f"{case['label']}: Chamfer over the lap, range error vs distance, and a real vs rendered scan.")
    return summ


def run(report: Report, quick: bool = False):
    import matplotlib
    matplotlib.use("Agg")
    report.h(2, "LiDAR model (LiDAR 3D Gaussian Splatting)")
    report.p("The LiDAR model is a Gaussian-splat scene rendered through a 6-face cubemap and read back "
             "along the Livox scan pattern. Chamfer distance is the mean nearest-neighbour distance between "
             "real and rendered scan (both directions); F-score is the harmonic mean of precision and recall "
             "at a distance threshold; hit rate is the share of real rays for which the model returns a surface.")
    try:
        eval_prerendered(report)
    except Exception as e:  # noqa: BLE001
        report.note(f"pre-rendered evaluation failed: {e}")
    cases = session_cases()
    if not cases:
        report.note("no session LiDAR models/bags found on the data drive (RAW_SESSION_* in scripts/env.sh)")
        return True
    lr_mod = load_module(REPO / "simulation/isaac_sim/lidar_renderer.py", "lidar_renderer_eval")
    rows = []
    for k, case in enumerate(cases):
        try:
            s = eval_session(report, case, lr_mod, quick, f"session_{k}")
        except Exception as e:  # noqa: BLE001
            report.note(f"{case['label']}: failed ({e})"); s = None
        if s:
            rows.append([case["label"], s["n_scans"], s["hit_rate"], s["range_err_median_cm"], s["range_err_rmse_cm"], s["within10cm"], s["chamfer_cm"], s["f10"], s["f20"]])
    report.h(3, "Session models vs real held-out scans")
    report.table(["model", "scans", "hit rate", "median |Δrange| [cm]", "RMSE range [cm]", "within 10 cm", "Chamfer [cm]", "F@10cm", "F@20cm"], rows)
    report.p("Poses come from FAST-LIO odometry of the same recording, so pose error is part of the number. "
             "Hit rate below 1 means the model has holes where the real sensor saw a surface (or the alpha "
             "filter removed thin geometry); range errors are measured only on rays both agree on.")
    return True
