#!/usr/bin/env python3
"""
calibrate_tcl_v2.py — T_cl calibration from UNIFIED bags (camera + LiDAR same bag).

New data format: /oak/rgb/image_raw + /livox/lidar in the same db3.
No separate video file, no timing offset needed.

Usage:
    conda run -n nerfstudio-py310 python3 scripts/calibrate_tcl_v2.py
    conda run -n nerfstudio-py310 python3 scripts/calibrate_tcl_v2.py --vis
"""

import argparse
import json
import os
import sqlite3
import struct
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# New unified bag paths
# Raw data isn't included in this repo (see root README) -- point these at
# your own copies, e.g. export NEW_DATA_ROOT=/path/to/New_probe
# ---------------------------------------------------------------------------
NEW_DATA_ROOT = Path(os.environ.get('NEW_DATA_ROOT', './data/New_probe'))
PROBE_ROOT    = Path(os.environ.get('PROBE_ROOT', './data/Probe'))
OUT_DIR       = PROBE_ROOT / 'experiments/calibration_tcl_2026-07-14'
OUT_DIR.mkdir(parents=True, exist_ok=True)

UNIFIED_BAGS = [
    NEW_DATA_ROOT / 'test_run_20260714_141624',  # main: 7 positions 1m/1.5m/2m
    NEW_DATA_ROOT / 'test_run_20260714_141809',  # supplement: tilted board
    NEW_DATA_ROOT / 'test_run_20260714_141025',  # extra
]

# Camera intrinsics — OAK-D Pro W 1280x800, rational_polynomial
K = np.array([[566.2562, 0.0, 641.9587],
              [0.0, 565.9493, 388.6107],
              [0.0, 0.0, 1.0]], dtype=np.float64)
D = np.array([3.6221985816955566, 0.9726417660713196,
              9.896762639982626e-05, -4.5452539779944345e-05,
              0.009347673505544662, 3.98378849029541,
              2.0535287857055664, 0.11342744529247284], dtype=np.float64)

TAG_SIZE    = 0.088
TAG_SPACING = 0.3 * TAG_SIZE
PITCH       = TAG_SIZE + TAG_SPACING
TAG_COLS    = 6
TAG_ROWS    = 6
MAX_TAG_ID  = TAG_COLS * TAG_ROWS - 1

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
_PARAMS = cv2.aruco.DetectorParameters()
_PARAMS.adaptiveThreshWinSizeMin = 3
_PARAMS.adaptiveThreshWinSizeMax = 53
_PARAMS.adaptiveThreshWinSizeStep = 4
_PARAMS.minMarkerPerimeterRate = 0.01
_PARAMS.maxMarkerPerimeterRate = 4.0
_DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, _PARAMS)


# ---------------------------------------------------------------------------
# CDR helpers (copied from original)
# ---------------------------------------------------------------------------

def _align4(off):
    return off + (4 - (off % 4)) % 4

def _read_u8(buf, off):
    return struct.unpack_from('B', buf, off)[0], off + 1

def _read_u32(buf, off):
    off = _align4(off)
    return struct.unpack_from('<I', buf, off)[0], off + 4

def _read_str(buf, off):
    length, off = _read_u32(buf, off)
    if length == 0:
        return '', off
    s = buf[off:off + length - 1].decode('utf-8', errors='replace')
    off += length
    return s, off


# ---------------------------------------------------------------------------
# ROS2 sensor_msgs/Image CDR decoder
# ---------------------------------------------------------------------------

def parse_ros2_image(data: bytes):
    """
    Decode a serialized ROS2 sensor_msgs/Image CDR message.
    Returns (timestamp_ns, bgr_frame) or None on failure.
    """
    try:
        off = 4  # skip 4-byte CDR encapsulation header
        # Header: stamp (sec int32, nanosec uint32) + frame_id string
        off = _align4(off)
        sec  = struct.unpack_from('<i', data, off)[0]; off += 4
        off = _align4(off)
        nsec = struct.unpack_from('<I', data, off)[0]; off += 4
        timestamp_ns = sec * 10**9 + nsec
        _fid, off = _read_str(data, off)

        # Image fields
        height, off = _read_u32(data, off)
        width,  off = _read_u32(data, off)
        encoding, off = _read_str(data, off)
        is_bigendian, off = _read_u8(data, off)
        step, off = _read_u32(data, off)
        n_bytes, off = _read_u32(data, off)

        pixel_data = bytes(data[off:off + n_bytes])

        enc = encoding.lower()
        if enc in ('bgr8', 'bgr'):
            frame = np.frombuffer(pixel_data, dtype=np.uint8).reshape(height, width, 3).copy()
        elif enc in ('rgb8', 'rgb'):
            frame = np.frombuffer(pixel_data, dtype=np.uint8).reshape(height, width, 3).copy()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif enc in ('mono8', 'mono'):
            gray = np.frombuffer(pixel_data, dtype=np.uint8).reshape(height, width).copy()
            frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            return None

        return timestamp_ns, frame
    except Exception:
        return None


# ---------------------------------------------------------------------------
# PointCloud2 CDR decoder (same as original)
# ---------------------------------------------------------------------------

def parse_pointcloud2(data: bytes) -> np.ndarray:
    off = 4
    _sec,  off = _read_u32(data, off)
    _nsec, off = _read_u32(data, off)
    _fid,  off = _read_str(data, off)
    _height, off = _read_u32(data, off)
    width,   off = _read_u32(data, off)
    n_fields, off = _read_u32(data, off)
    field_offsets = {}
    for _ in range(n_fields):
        fname, off = _read_str(data, off)
        foff,  off = _read_u32(data, off)
        dtype, off = _read_u8(data, off)
        count, off = _read_u32(data, off)
        field_offsets[fname] = foff
    _big, off = _read_u8(data, off)
    point_step, off = _read_u32(data, off)
    _row_step,  off = _read_u32(data, off)
    n_data,     off = _read_u32(data, off)
    raw = np.frombuffer(data[off:off + n_data], dtype=np.uint8).reshape(-1, point_step)
    xo = field_offsets.get('x', 0)
    yo = field_offsets.get('y', 4)
    zo = field_offsets.get('z', 8)
    x = np.frombuffer(raw[:, xo:xo+4].tobytes(), dtype=np.float32)
    y = np.frombuffer(raw[:, yo:yo+4].tobytes(), dtype=np.float32)
    z = np.frombuffer(raw[:, zo:zo+4].tobytes(), dtype=np.float32)
    xyz = np.stack([x, y, z], axis=1)
    valid = np.isfinite(xyz).all(axis=1) & (np.linalg.norm(xyz, axis=1) > 0.01)
    return xyz[valid].astype(np.float32)


# ---------------------------------------------------------------------------
# Load unified bag (camera + LiDAR from same db3)
# ---------------------------------------------------------------------------

def load_unified_bag(bag_dir: Path):
    """
    Load camera frames and LiDAR scans from a unified bag.
    Returns:
        cam_frames: list of (timestamp_ns, bgr_frame)
        lidar_scans: list of (timestamp_ns, xyz_array)
    """
    db_files = sorted(bag_dir.glob('*.db3'))
    if not db_files:
        raise FileNotFoundError(f"No .db3 in {bag_dir}")

    conn = sqlite3.connect(str(db_files[0]))
    topics = {r[1]: r[0] for r in conn.execute('SELECT id, name FROM topics')}

    cam_topic_id   = topics.get('/oak/rgb/image_raw')
    lidar_topic_id = topics.get('/livox/lidar')

    if cam_topic_id is None:
        print(f"  [WARN] No /oak/rgb/image_raw in {bag_dir.name}")
    if lidar_topic_id is None:
        print(f"  [WARN] No /livox/lidar in {bag_dir.name}")

    # Load camera frames
    cam_frames = []
    if cam_topic_id is not None:
        rows = conn.execute(
            'SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp',
            (cam_topic_id,)
        ).fetchall()
        for ts, data in rows:
            result = parse_ros2_image(bytes(data))
            if result is not None:
                cam_frames.append((int(ts), result[1]))
        print(f"  Camera frames decoded: {len(cam_frames)}/{len(rows)}")

    # Load LiDAR scans
    lidar_scans = []
    if lidar_topic_id is not None:
        rows = conn.execute(
            'SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp',
            (lidar_topic_id,)
        ).fetchall()
        for ts, data in rows:
            try:
                xyz = parse_pointcloud2(bytes(data))
                lidar_scans.append((int(ts), xyz))
            except Exception:
                pass
        print(f"  LiDAR scans loaded:   {len(lidar_scans)}/{len(rows)}")

    conn.close()
    return cam_frames, lidar_scans


# ---------------------------------------------------------------------------
# Board detection (same logic as original)
# ---------------------------------------------------------------------------

LOCAL_3D = np.array([[0, 0, 0], [TAG_SIZE, 0, 0],
                     [TAG_SIZE, TAG_SIZE, 0], [0, TAG_SIZE, 0]], dtype=np.float64)


def detect_board_plane_camera(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = _DETECTOR.detectMarkers(gray)
    if ids is None or len(ids) < 3:
        return None
    normals, distances = [], []
    for i, tag_id in enumerate(ids.flatten()):
        if tag_id > MAX_TAG_ID:
            continue
        pts2d = corners[i][0].astype(np.float64)
        ret, rvec, tvec = cv2.solvePnP(LOCAL_3D, pts2d, K, D,
                                        flags=cv2.SOLVEPNP_ITERATIVE)
        if not ret:
            continue
        refine = cv2.solvePnPRefineLM(LOCAL_3D, pts2d.reshape(-1, 1, 2),
                                       K, D, rvec, tvec)
        rvec, tvec = refine[-2], refine[-1]
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.flatten()
        n = R @ np.array([0.0, 0.0, 1.0])
        d = float(n @ t)
        normals.append(n)
        distances.append(d)
    if len(normals) < 3:
        return None
    normals   = np.array(normals)
    distances = np.array(distances)
    n_med = np.median(normals, axis=0)
    n_med /= np.linalg.norm(n_med)
    angles = np.arccos(np.clip(normals @ n_med, -1, 1))
    keep = angles < np.radians(15)
    if keep.sum() < 3:
        return None
    n_avg = normals[keep].mean(axis=0)
    n_avg /= np.linalg.norm(n_avg)
    d_avg = float(distances[keep].mean())
    if n_avg[2] < 0:
        n_avg, d_avg = -n_avg, -d_avg
    return n_avg, d_avg, int(keep.sum())


def ransac_plane(xyz, n_iter=300, dist_thresh=0.012, min_inliers=50):
    n = len(xyz)
    if n < min_inliers:
        return None
    best_n, best_d, best_count = None, 0.0, 0
    rng = np.random.default_rng(0)
    for _ in range(n_iter):
        idx = rng.choice(n, 3, replace=False)
        p0, p1, p2 = xyz[idx]
        v1, v2 = p1 - p0, p2 - p0
        normal = np.cross(v1, v2)
        nm = np.linalg.norm(normal)
        if nm < 1e-6:
            continue
        normal /= nm
        d = float(normal @ p0)
        count = int((np.abs(xyz @ normal - d) < dist_thresh).sum())
        if count > best_count:
            best_count, best_n, best_d = count, normal, d
    if best_n is None or best_count < min_inliers:
        return None
    inliers = np.abs(xyz @ best_n - best_d) < dist_thresh
    pts = xyz[inliers]
    _, _, Vt = np.linalg.svd(pts - pts.mean(0))
    best_n = Vt[-1]
    best_d = float(best_n @ pts.mean(0))
    inliers = np.abs(xyz @ best_n - best_d) < dist_thresh
    return best_n, best_d, inliers


def detect_board_plane_lidar(scans_xyz):
    all_pts = np.vstack(scans_xyz).astype(np.float32)
    ranges  = np.linalg.norm(all_pts, axis=1)
    mask = ((ranges > 0.3) & (ranges < 2.5) &
            (all_pts[:, 2] > -0.4) & (all_pts[:, 2] < 0.8) &
            (all_pts[:, 0] > 0.1))
    pts = all_pts[mask]
    if len(pts) < 60:
        return None
    result = ransac_plane(pts)
    if result is None:
        return None
    normal, dist, inliers = result
    board_center = pts[inliers].mean(0)
    if float(normal @ board_center) < 0:
        normal, dist = -normal, -dist
    patch = pts[inliers]
    if len(patch) < 40 or (patch.max(0) - patch.min(0)).max() < 0.2:
        return None
    if abs(normal[2]) > 0.7:
        return None
    return normal, dist, int(inliers.sum())


# ---------------------------------------------------------------------------
# Process unified bag → plane pairs
# ---------------------------------------------------------------------------

def process_unified_bag(bag_dir: Path, args):
    """Process a unified bag, return list of (n_c, d_c, n_l, d_l) pairs."""
    print(f"\n=== {bag_dir.name} ===")
    cam_frames, lidar_scans = load_unified_bag(bag_dir)

    if not cam_frames or not lidar_scans:
        print("  [SKIP] Missing camera or LiDAR data")
        return []

    lidar_ts = np.array([ts for ts, _ in lidar_scans], dtype=np.int64)
    pairs = []
    # Time-based diversity: accept at most 1 pair per 2 seconds
    last_accepted_ts = None
    MIN_INTERVAL_NS = int(2.0 * 1e9)

    for cam_ts, frame in cam_frames:
        # Detect AprilGrid
        cam_result = detect_board_plane_camera(frame)
        if cam_result is None:
            continue
        n_c, d_c, n_tags = cam_result
        board_dist = d_c

        # Time-based diversity: skip if too close in time to last accepted frame
        if last_accepted_ts is not None and (cam_ts - last_accepted_ts) < MIN_INTERVAL_NS:
            continue

        # Find nearest LiDAR scans (same bag = same timestamp space, no offset)
        nearest_idx = int(np.argmin(np.abs(lidar_ts - cam_ts)))
        time_diff_ms = abs(lidar_ts[nearest_idx] - cam_ts) / 1e6
        if time_diff_ms > 300:
            continue

        # Accumulate 3 scans around nearest
        lidar_xyz_list = []
        for di in range(-1, 2):
            idx = nearest_idx + di
            if 0 <= idx < len(lidar_scans):
                lidar_xyz_list.append(lidar_scans[idx][1])
        if not lidar_xyz_list:
            continue

        # Detect board plane in LiDAR
        lidar_result = detect_board_plane_lidar(lidar_xyz_list)
        if lidar_result is None:
            continue
        n_l, d_l, n_inliers = lidar_result

        if abs(board_dist - abs(d_l)) > 1.5:
            continue

        pairs.append((n_c, d_c, n_l, d_l))
        last_accepted_ts = cam_ts

        if args.verbose:
            print(f"  cam_ts={cam_ts} cam_d={d_c:.3f}m n_tags={n_tags} "
                  f"lidar_d={d_l:.3f}m n_inliers={n_inliers} dt={time_diff_ms:.0f}ms")

        if args.vis:
            vis = frame.copy()
            cv2.putText(vis, f"d={d_c:.2f}m tags={n_tags}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("AprilGrid detection", vis)
            cv2.waitKey(300)

    print(f"  Collected {len(pairs)} plane pairs")
    return pairs


# ---------------------------------------------------------------------------
# Optimization (copied from original)
# ---------------------------------------------------------------------------

def tcl_cost_rv(params, pairs):
    R = Rotation.from_rotvec(params[:3]).as_matrix()
    t = params[3:]
    rot_err, trans_err = 0.0, 0.0
    for n_c, d_c, n_l, d_l in pairs:
        n_pred = R @ n_l
        rot_err   += float(np.sum((n_pred - n_c) ** 2))
        trans_err += float((float(n_c @ t) + d_l - d_c) ** 2)
    return rot_err + trans_err


def lstsq_translation(pairs, R):
    n_c_mat = np.array([n_c for n_c, _, _, _ in pairs])
    b       = np.array([d_c - d_l for n_c, d_c, n_l, d_l in pairs])
    t, _, _, _ = np.linalg.lstsq(n_c_mat, b, rcond=None)
    return t


def pair_residual_angle(n_c, n_l, R):
    n_pred = R @ n_l
    return float(np.degrees(np.arccos(np.clip(n_pred @ n_c, -1.0, 1.0))))


def robust_rotation_from_reliable_pairs(pairs):
    R_ideal = Rotation.from_quat([0.5, -0.5, 0.5, 0.5]).as_matrix()
    residuals = np.array([pair_residual_angle(n_c, n_l, R_ideal)
                          for n_c, d_c, n_l, d_l in pairs])
    inliers = residuals < 20.0
    n_in = inliers.sum()
    print(f"\n  Outlier rejection vs R_ideal:")
    for i, (r, ok) in enumerate(zip(residuals, inliers)):
        print(f"    pair {i:2d}: {r:5.1f}°{'  ← OUTLIER' if not ok else ''}")
    print(f"  Inliers: {n_in}/{len(pairs)}")
    if n_in < 3:
        print("  Too few inliers — using R_ideal")
        return R_ideal, inliers, residuals
    inlier_pairs = [p for p, ok in zip(pairs, inliers) if ok]
    n_c_inlier = np.array([n_c for n_c, _, _, _ in inlier_pairs])
    n_l_inlier = np.array([n_l for _, _, n_l, _ in inlier_pairs])
    weights = np.maximum(n_l_inlier[:, 0], 0.5)
    r1 = np.average(n_c_inlier, axis=0, weights=weights)
    r1 /= np.linalg.norm(r1)
    r2_ideal = R_ideal[:, 1]
    r2 = r2_ideal - float(r2_ideal @ r1) * r1
    r2 /= np.linalg.norm(r2)
    r3 = np.cross(r1, r2)
    r3 /= np.linalg.norm(r3)
    R_robust = np.column_stack([r1, r2, r3])
    if np.linalg.det(R_robust) < 0:
        r3 = -r3
        R_robust = np.column_stack([r1, r2, r3])
    return R_robust, inliers, residuals


def optimize_tcl(pairs, n_restarts=6):
    T_BOUNDS = [(-0.10, 0.15), (-0.05, 0.15), (-0.10, 0.02)]
    R_ideal  = Rotation.from_quat([0.5, -0.5, 0.5, 0.5]).as_matrix()
    t_init   = np.array([0.023, 0.070, -0.044])

    R_robust, inliers, residuals = robust_rotation_from_reliable_pairs(pairs)
    t_robust = lstsq_translation(pairs, R_robust)
    t_robust_clipped = np.clip(t_robust,
                               [b[0] for b in T_BOUNDS],
                               [b[1] for b in T_BOUNDS])
    print(f"\n  Robust R: q={Rotation.from_matrix(R_robust).as_quat().round(4)}")
    print(f"  Robust t: {t_robust.round(4)} → clipped: {t_robust_clipped.round(4)}")

    def bounded_cost(params, pairs, t_bounds):
        base = tcl_cost_rv(params, pairs)
        t = params[3:]
        penalty = sum(1000*(lo-t[i])**2 if t[i]<lo else 1000*(t[i]-hi)**2 if t[i]>hi else 0.0
                      for i, (lo, hi) in enumerate(t_bounds))
        return base + penalty

    seeds = [
        (Rotation.from_matrix(R_robust), t_robust_clipped),
        (Rotation.from_quat([0.5, -0.5, 0.5, 0.5]), t_init),
        (Rotation.from_quat([-0.46552296, 0.48718884, -0.52270603, 0.52220090]), t_init),
    ]
    best_result = None
    np.random.seed(42)
    for restart in range(n_restarts):
        if restart < len(seeds):
            R0, t0 = seeds[restart]
        else:
            perturb = Rotation.from_rotvec(np.random.randn(3) * 0.3)
            R0 = Rotation.from_matrix(R_robust) * perturb
            t0 = t_robust_clipped + np.random.randn(3) * 0.02
        x0 = np.concatenate([R0.as_rotvec(), t0])
        res = minimize(bounded_cost, x0, args=(pairs, T_BOUNDS),
                       method='Nelder-Mead',
                       options={'maxiter': 200000, 'xatol': 1e-10, 'fatol': 1e-10})
        if best_result is None or res.fun < best_result.fun:
            best_result = res
        print(f"  restart {restart}: cost={res.fun:.6f}  t={res.x[3:].round(4)}")

    params = best_result.x
    R = Rotation.from_rotvec(params[:3]).as_matrix()
    t = params[3:]
    return R, t, float(best_result.fun)


def evaluate_pairs(pairs, R, t, label=""):
    rot_errs = []
    for n_c, d_c, n_l, d_l in pairs:
        angle = pair_residual_angle(n_c, n_l, R)
        rot_errs.append(angle)
    rot_errs = np.array(rot_errs)
    inlier_mask = rot_errs < 10.0
    print(f"\n  {label} — {len(pairs)} pairs")
    print(f"  rotation residual (all):    mean={rot_errs.mean():.2f}° std={rot_errs.std():.2f}°")
    if inlier_mask.sum() > 0:
        print(f"  rotation residual (inliers <10°): mean={rot_errs[inlier_mask].mean():.2f}° "
              f"n={inlier_mask.sum()}/{len(pairs)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vis',     action='store_true', help='show detection windows')
    parser.add_argument('--verbose', action='store_true', help='print per-frame details')
    args = parser.parse_args()

    print("=" * 60)
    print("T_cl Calibration v2 — Unified Bag Format")
    print("=" * 60)

    all_pairs = []
    for bag_dir in UNIFIED_BAGS:
        if not bag_dir.exists():
            print(f"\n[SKIP] {bag_dir} not found")
            continue
        pairs = process_unified_bag(bag_dir, args)
        all_pairs.extend(pairs)

    print(f"\nTotal plane pairs collected: {len(all_pairs)}")

    if len(all_pairs) < 5:
        print("ERROR: Too few pairs for calibration (need >= 5)")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Optimizing T_cl ...")
    print("=" * 60)
    R, t, cost = optimize_tcl(all_pairs)

    q = Rotation.from_matrix(R).as_quat()
    euler = Rotation.from_matrix(R).as_euler('xyz', degrees=True)

    print(f"\n{'='*60}")
    print("RESULT")
    print(f"{'='*60}")
    print(f"  translation [x,y,z] m:  {t.round(5).tolist()}")
    print(f"  quaternion [x,y,z,w]:   {q.round(6).tolist()}")
    print(f"  euler RPY deg (xyz):     {euler.round(2).tolist()}")
    print(f"  definition: X_camera = R_cl @ X_lidar + t_cl")

    evaluate_pairs(all_pairs, R, t, "Final")

    # Compare with previous result
    prev_q = np.array([0.48149282347616235, -0.4701392520594065,
                       0.5167592769821906, 0.5292386930930304])
    prev_t = np.array([0.06931279334640274, -0.04650674937368852, -0.02003236137774059])
    prev_R = Rotation.from_quat(prev_q).as_matrix()
    angle_diff = np.degrees(np.arccos(np.clip(
        (np.trace(prev_R.T @ R) - 1) / 2, -1, 1)))
    t_diff = np.linalg.norm(t - prev_t) * 100
    print(f"\n  vs previous (2026-07-03):")
    print(f"    rotation diff: {angle_diff:.2f}°")
    print(f"    translation diff: {t_diff:.1f} cm")

    # Save result
    result = {
        "date": "2026-07-14",
        "method": "unified_bag_plane_pair_optimization",
        "n_pairs_total": len(all_pairs),
        "T_cl_result": {
            "translation_xyz_m": t.tolist(),
            "quaternion_xyzw": q.tolist(),
            "euler_rpy_deg_xyz": euler.tolist(),
            "rotation_matrix": R.tolist(),
            "definition": "X_camera = R_cl @ X_lidar + t_cl"
        },
        "optimization_cost": cost,
        "bags_used": [str(b) for b in UNIFIED_BAGS if b.exists()]
    }
    out_path = OUT_DIR / 'tcl_result.json'
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved to: {out_path}")

    if args.vis:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
