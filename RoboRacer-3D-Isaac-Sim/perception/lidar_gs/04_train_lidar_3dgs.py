"""
train_lidar_3dgs.py — Pure LiDAR 3DGS: train Gaussians using only LiDAR depth supervision.

LiDAR in → 3DGS → LiDAR out. No camera images needed.

Pipeline:
  1. Extract per-frame LiDAR scans from ROS2 bag
  2. Transform each scan to world frame via MoCap poses
  3. Initialize Gaussians from accumulated LiDAR point cloud
  4. For each frame: render depth from LiDAR sensor viewpoint → L1 loss vs real depth
  5. Render synthetic LiDAR from novel viewpoints for evaluation

The Livox Mid-360 has 360° horizontal FOV. To render with gsplat (pinhole model),
we use a forward-facing virtual camera at the LiDAR sensor position. This covers
the main driving direction. Future: cubemap for full 360° coverage.

Usage:
  conda run -n nerfstudio-py310 python3 scripts/train_lidar_3dgs.py [--steps 10000]
"""
import argparse
import csv
import json
import struct
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp

# ── Config ────────────────────────────────────────────────────────────────────
import os
# Raw data isn't included in this repo (see root README) -- point these at
# your own copies, e.g. export PROBE_ROOT=/path/to/Probe
PROBE_ROOT = Path(os.environ.get('PROBE_ROOT', './data/Probe'))
ROOT = Path(os.environ.get('LIDAR3DGS_ROOT', './data/Newmocap/layout_02_slow_run_02'))
BAG  = ROOT / os.environ.get('LIDAR3DGS_BAG', 'run_20260327_105224_0.db3')
CSYNC = ROOT / 'camera_sync.csv'

# COLMAP pose source (used when --use-colmap is set)
COLMAP_TRANSFORMS = Path(os.environ.get(
    'LIDAR3DGS_COLMAP_TRANSFORMS',
    str(PROBE_ROOT / 'experiments/layout02_colmap_metric/dataset/transforms.json')
))

# Camera-LiDAR extrinsic raw values from extrinsics_candidate_checked_v2.yaml.
# WARNING: The YAML uses ROS TF child-frame convention:
#   T_CL_YAML_Q is the camera frame's orientation *expressed in* LiDAR frame (NOT R_cl directly).
#   T_CL_YAML_T is the camera ORIGIN in LiDAR frame (NOT the t_cl translation vector).
# Correct T_cl (LiDAR→camera optical):  R_cl = R_yaml.T,  t_cl = -R_cl @ T_CL_YAML_T
T_CL_YAML_T = np.array([0.02267000, 0.06987000, -0.04441000])  # camera origin in LiDAR frame (physical, unchanged)
T_CL_YAML_Q = np.array([-0.47431095, 0.49897995, -0.50794995, 0.51772095])  # xyzw, child-in-parent (2026-07-14, residual 2.54°)

T_BASE_MOCAP_t = np.array([0.10, 0.00, 0.20])
T_BASE_LIDAR_t = np.array([0.27, 0.00, 0.11])
POSE_OFFSET    = 28

SG_WIN, SG_POLY = 21, 3
OUTLIER_POS_JUMP_M = 0.50
MAX_RANGE_M = 8.0
INIT_VOXEL_SIZE = 0.10

# Virtual depth camera params for rendering
VCAM_W, VCAM_H = 256, 256
VCAM_FOV_DEG = 90.0

# Cubemap face rotations: sensor-frame rotation to look in each direction
# Sensor frame: X-forward, Y-left, Z-up
# Each R_face rotates sensor frame so that the face's direction becomes +X (forward)
CUBEMAP_FACES = {
    'front': np.eye(3),                                           # +X
    'back':  Rotation.from_euler('z', 180, degrees=True).as_matrix(),  # -X
    'left':  Rotation.from_euler('z',  90, degrees=True).as_matrix(),  # +Y
    'right': Rotation.from_euler('z', -90, degrees=True).as_matrix(),  # -Y
    'up':    Rotation.from_euler('y', -90, degrees=True).as_matrix(),  # +Z
    'down':  Rotation.from_euler('y',  90, degrees=True).as_matrix(),  # -Z
}

# ── Data loading (reuse from build_lidar_pointcloud.py) ──────────────────────

def load_colmap_poses():
    """Load COLMAP camera poses and camera_sync mapping for COLMAP-based LiDAR training.

    Returns:
        colmap_by_idx: dict {camera_frame_idx(int) -> c2w_4x4 (nerfstudio/OpenGL convention)}
        cam_sync:      dict {lidar_ts_ns(int) -> (camera_frame_idx, matched)}
        T_cl:          4x4 float64, LiDAR-to-nerfstudio-camera transform
    """
    # Load COLMAP poses (nerfstudio convention: X-right, Y-up, Z-back)
    tf = json.load(open(str(COLMAP_TRANSFORMS)))
    colmap_by_idx = {}
    for frame in tf['frames']:
        stem = Path(frame['file_path']).stem  # 'frame_000042' → 42
        idx = int(stem.split('_')[1])
        colmap_by_idx[idx] = np.array(frame['transform_matrix'], dtype=np.float64)
    print(f"  COLMAP: {len(colmap_by_idx)} poses loaded from {COLMAP_TRANSFORMS.name}")

    # Load camera_sync: LiDAR timestamp → camera frame index
    cam_sync = {}
    with open(str(CSYNC)) as f:
        for row in csv.DictReader(f):
            lt = int(row['lidar_timestamp_ns'])
            cam_idx = int(row['camera_frame_idx'])
            matched = row['matched'].strip().lower() == 'true'
            cam_sync[lt] = (cam_idx, matched)
    n_matched = sum(1 for _, (_, m) in cam_sync.items() if m)
    print(f"  camera_sync: {len(cam_sync)} entries, {n_matched} matched")

    # Build T_cl: LiDAR sensor frame → nerfstudio camera frame
    # Step 1: T_cl_opencv = LiDAR → camera optical (OpenCV: X-right, Y-down, Z-fwd)
    # YAML quaternion is child-frame-in-parent (ROS TF), so R_cl = R_yaml.T
    # YAML translation is camera origin in LiDAR frame, so t_cl = -R_cl @ cam_origin
    R_cl_opencv = Rotation.from_quat(T_CL_YAML_Q).as_matrix().T  # transpose = invert convention
    T_cl_opencv = np.eye(4, dtype=np.float64)
    T_cl_opencv[:3, :3] = R_cl_opencv
    T_cl_opencv[:3, 3]  = -R_cl_opencv @ T_CL_YAML_T  # = [0.069, -0.047, -0.017]
    # Step 2: convert camera convention: OpenCV (Y-down, Z-fwd) → nerfstudio OpenGL (Y-up, Z-back)
    OPENCV2OPENGL = np.diag([1.0, -1.0, -1.0, 1.0])
    T_cl = OPENCV2OPENGL @ T_cl_opencv  # LiDAR → nerfstudio camera

    return colmap_by_idx, cam_sync, T_cl

def load_mocap():
    conn = sqlite3.connect(str(BAG))
    cur  = conn.cursor()
    cur.execute("""
        SELECT m.timestamp, m.data FROM messages m
        JOIN topics t ON m.topic_id=t.id
        WHERE t.name='/vrpn_mocap/Car2/pose' ORDER BY m.timestamp
    """)
    rows = cur.fetchall()
    conn.close()

    n = len(rows)
    ts_ns = np.empty(n, dtype=np.float64)
    pos   = np.empty((n, 3))
    quat  = np.empty((n, 4))
    for i, (ts, data) in enumerate(rows):
        b = bytes(data)
        px, py, pz, qx, qy, qz, qw = struct.unpack_from('<7d', b, POSE_OFFSET)
        ts_ns[i] = float(ts)
        pos[i]   = [px, py, pz]
        quat[i]  = [qx, qy, qz, qw]

    d = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    ok = np.ones(n, dtype=bool)
    ok[1:] &= (d < OUTLIER_POS_JUMP_M)
    ok[:-1] &= (d < OUTLIER_POS_JUMP_M)
    ts_ns, pos, quat = ts_ns[ok], pos[ok], quat[ok]

    _, uniq = np.unique(ts_ns, return_index=True)
    ts_ns, pos, quat = ts_ns[uniq], pos[uniq], quat[uniq]

    for ax in range(3): pos[:, ax]  = savgol_filter(pos[:, ax],  SG_WIN, SG_POLY)
    for ax in range(4): quat[:, ax] = savgol_filter(quat[:, ax], SG_WIN, SG_POLY)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)

    slerp = Slerp(ts_ns, Rotation.from_quat(quat))
    print(f"  Mocap: {len(ts_ns)} messages")
    return ts_ns, pos, slerp


def _align(idx, n):
    if idx % n != 0: idx += n - (idx % n)
    return idx


def parse_pointcloud2(data, return_intensity=False):
    b = bytes(data)
    idx = 4
    idx += 8
    frame_id_len = struct.unpack_from('<I', b, idx)[0]; idx += 4
    idx += frame_id_len
    idx = _align(idx, 4)
    height = struct.unpack_from('<I', b, idx)[0]; idx += 4
    width  = struct.unpack_from('<I', b, idx)[0]; idx += 4
    n_fields = struct.unpack_from('<I', b, idx)[0]; idx += 4
    fields = {}
    for _ in range(n_fields):
        idx = _align(idx, 4)
        name_len = struct.unpack_from('<I', b, idx)[0]; idx += 4
        name = b[idx:idx+name_len].rstrip(b'\x00').decode(); idx += name_len
        idx = _align(idx, 4)
        f_offset = struct.unpack_from('<I', b, idx)[0]; idx += 4
        f_dtype  = b[idx]; idx += 1
        idx = _align(idx, 4)
        f_count  = struct.unpack_from('<I', b, idx)[0]; idx += 4
        fields[name] = (f_offset, f_dtype)
    idx += 1
    idx = _align(idx, 4)
    point_step = struct.unpack_from('<I', b, idx)[0]; idx += 4
    idx += 4
    data_len = struct.unpack_from('<I', b, idx)[0]; idx += 4
    data_start = idx

    if 'x' not in fields:
        if return_intensity:
            return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=np.float32)
        return np.empty((0, 3), dtype=np.float32)
    x_off, _ = fields['x']
    y_off, _ = fields['y']
    z_off, _ = fields['z']
    pts = np.frombuffer(b[data_start:data_start+data_len], dtype=np.uint8).reshape(-1, point_step)
    xs = np.frombuffer(pts[:, x_off:x_off+4].tobytes(), dtype=np.float32)
    ys = np.frombuffer(pts[:, y_off:y_off+4].tobytes(), dtype=np.float32)
    zs = np.frombuffer(pts[:, z_off:z_off+4].tobytes(), dtype=np.float32)
    xyz = np.stack([xs, ys, zs], axis=1)
    valid = np.isfinite(xyz).all(axis=1) & (np.linalg.norm(xyz, axis=1) > 0.1)

    if return_intensity and 'intensity' in fields:
        i_off, _ = fields['intensity']
        intensity = np.frombuffer(pts[:, i_off:i_off+4].tobytes(), dtype=np.float32)
        return xyz[valid], intensity[valid]
    if return_intensity:
        return xyz[valid], np.zeros(valid.sum(), dtype=np.float32)
    return xyz[valid]


def voxel_downsample(pts, voxel):
    if len(pts) == 0: return pts
    vox_idx = np.floor(pts / voxel).astype(np.int32)
    _, uniq = np.unique(vox_idx, axis=0, return_index=True)
    return pts[uniq]


# ── Virtual camera for depth rendering ────────────────────────────────────────

def make_virtual_K(W, H, fov_deg):
    f = W / (2.0 * np.tan(np.radians(fov_deg / 2.0)))
    return np.array([[f, 0, W/2], [0, f, H/2], [0, 0, 1]], dtype=np.float32)


def project_lidar_to_depth(xyz_sensor, K, W, H, R_face=None, intensity=None):
    """Project LiDAR points (in sensor frame) into a virtual pinhole depth image.
    Also projects intensity if provided.
    """
    if len(xyz_sensor) == 0:
        result = np.zeros((H, W), dtype=np.float32)
        return (result, np.zeros((H, W), dtype=np.float32)) if intensity is not None else result

    pts = xyz_sensor
    if R_face is not None:
        pts = (R_face.T @ pts.T).T

    x_cam = -pts[:, 1]
    y_cam = -pts[:, 2]
    z_cam =  pts[:, 0]

    valid = z_cam > 0.1
    x_cam, y_cam, z_cam = x_cam[valid], y_cam[valid], z_cam[valid]
    if intensity is not None:
        inten_valid = intensity[valid]

    u = (K[0, 0] * x_cam / z_cam + K[0, 2]).astype(np.int32)
    v = (K[1, 1] * y_cam / z_cam + K[1, 2]).astype(np.int32)

    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z_cam = u[in_bounds], v[in_bounds], z_cam[in_bounds]
    if intensity is not None:
        inten_valid = inten_valid[in_bounds]

    depth = np.zeros((H, W), dtype=np.float32)
    inten_img = np.zeros((H, W), dtype=np.float32) if intensity is not None else None
    for i in range(len(u)):
        if depth[v[i], u[i]] == 0 or z_cam[i] < depth[v[i], u[i]]:
            depth[v[i], u[i]] = z_cam[i]
            if inten_img is not None:
                inten_img[v[i], u[i]] = inten_valid[i]

    return (depth, inten_img) if intensity is not None else depth


# Lidar-to-camera base rotation: sensor (X-fwd,Y-left,Z-up) → cam (X-right,Y-down,Z-fwd)
R_CL_BASE = np.array([[0, -1, 0],
                       [0,  0, -1],
                       [1,  0,  0]], dtype=np.float64)


def make_viewmat(R_wl, t_wl, R_face=None):
    """Build 4x4 world-to-camera viewmat for gsplat.
    R_face: cubemap face rotation in sensor frame (optional).
    """
    R_lw = R_wl.T
    t_lw = -R_lw @ t_wl

    if R_face is not None:
        # Apply face rotation: rotated_sensor = R_face^T @ sensor
        R_cl = R_CL_BASE @ R_face.T
    else:
        R_cl = R_CL_BASE

    R_cw = R_cl @ R_lw
    t_cw = R_cl @ t_lw

    viewmat = np.eye(4, dtype=np.float32)
    viewmat[:3, :3] = R_cw
    viewmat[:3, 3]  = t_cw
    return viewmat


# ── Data preparation ──────────────────────────────────────────────────────────

def prepare_data(max_frames=None, train_ratio=0.9, use_cubemap=True, use_intensity=False,
                 use_colmap=False):
    """Extract per-frame LiDAR data. Returns train/val splits.

    use_colmap=True: derive LiDAR poses from COLMAP camera poses via camera_sync.csv
                     and the camera-LiDAR extrinsic calibration. Eliminates MoCap errors.
    use_colmap=False: classic MoCap interpolation (original behaviour).
    """
    if use_colmap:
        print("Loading COLMAP poses (use_colmap mode)...")
        colmap_by_idx, colmap_cam_sync, T_cl = load_colmap_poses()
    else:
        print("Loading mocap...")
        ts_f, pos_f, slerp = load_mocap()

    print("Loading LiDAR scans...")
    conn = sqlite3.connect(str(BAG))
    cur  = conn.cursor()
    cur.execute("""
        SELECT m.timestamp, m.data FROM messages m
        JOIN topics t ON m.topic_id=t.id
        WHERE t.name='/livox/lidar' ORDER BY m.timestamp
    """)
    lidar_rows = cur.fetchall()
    conn.close()
    print(f"  LiDAR: {len(lidar_rows)} scans")

    if not use_colmap:
        cam_sync = {}
        with open(str(CSYNC)) as f:
            for row in csv.DictReader(f):
                lt = int(row['lidar_timestamp_ns'])
                cam_sync[lt] = row['matched'].strip().lower() == 'true'

    K = make_virtual_K(VCAM_W, VCAM_H, VCAM_FOV_DEG)

    faces = list(CUBEMAP_FACES.items()) if use_cubemap else [('front', np.eye(3))]
    print(f"  Using {'cubemap (' + str(len(faces)) + ' faces)' if use_cubemap else 'single front camera'}")
    if use_intensity:
        print(f"  Intensity supervision enabled")

    frames = []
    all_world_pts = []
    intensity_stats = []
    n_scans = 0

    for ts, data in lidar_rows:
        if use_colmap:
            lidar_ts_int = int(ts)
            if lidar_ts_int not in colmap_cam_sync:
                continue
            cam_idx, matched = colmap_cam_sync[lidar_ts_int]
            if not matched or cam_idx < 0 or cam_idx not in colmap_by_idx:
                continue
            # T_wl = T_wc (COLMAP c2w) @ T_cl (LiDAR → nerfstudio camera)
            T_wc = colmap_by_idx[cam_idx]
            T_wl_4x4 = T_wc @ T_cl
            R_wl = T_wl_4x4[:3, :3]
            t_wl = T_wl_4x4[:3, 3]
        else:
            lidar_ts = float(ts)
            if not cam_sync.get(ts, False):
                continue
            if lidar_ts < ts_f[0] or lidar_ts > ts_f[-1]:
                continue

            i = max(0, min(int(np.searchsorted(ts_f, lidar_ts)) - 1, len(ts_f) - 2))
            alpha = (lidar_ts - ts_f[i]) / max(float(ts_f[i+1] - ts_f[i]), 1.0)
            pos_m = (1.0 - alpha) * pos_f[i] + alpha * pos_f[i+1]
            R_wm  = slerp(lidar_ts).as_matrix()

            t_wb = R_wm @ (-T_BASE_MOCAP_t) + pos_m
            R_wb = R_wm
            t_wl = R_wb @ T_BASE_LIDAR_t + t_wb
            R_wl = R_wb

        if use_intensity:
            xyz_lidar, intensity = parse_pointcloud2(bytes(data), return_intensity=True)
        else:
            xyz_lidar = parse_pointcloud2(bytes(data))
            intensity = None

        if len(xyz_lidar) == 0:
            continue

        ranges = np.linalg.norm(xyz_lidar, axis=1)
        mask = ranges < MAX_RANGE_M
        xyz_lidar = xyz_lidar[mask]
        if intensity is not None:
            intensity = intensity[mask]
            intensity_stats.append(intensity)

        xyz_world = (R_wl @ xyz_lidar.T).T + t_wl
        all_world_pts.append(xyz_world.astype(np.float32))

        for face_name, R_face in faces:
            if use_intensity:
                depth_gt, inten_gt = project_lidar_to_depth(
                    xyz_lidar, K, VCAM_W, VCAM_H, R_face, intensity)
            else:
                depth_gt = project_lidar_to_depth(xyz_lidar, K, VCAM_W, VCAM_H, R_face)
                inten_gt = None

            n_valid = (depth_gt > 0).sum()
            if n_valid < 50:
                continue

            viewmat = make_viewmat(R_wl, t_wl, R_face)
            frame = {
                'viewmat': viewmat,
                'depth_gt': depth_gt,
                'n_valid': int(n_valid),
            }
            if inten_gt is not None:
                frame['intensity_gt'] = inten_gt
            frames.append(frame)

        n_scans += 1
        if max_frames and n_scans >= max_frames:
            break

    avg_valid = np.mean([f['n_valid'] for f in frames]) if frames else 0
    print(f"  {n_scans} scans → {len(frames)} frames (avg {avg_valid:.0f} depth pixels/frame)")

    if use_intensity and intensity_stats:
        all_inten = np.concatenate(intensity_stats)
        inten_scale = float(np.percentile(all_inten[all_inten > 0], 99))
        print(f"  Intensity range: [{all_inten.min():.1f}, {all_inten.max():.1f}], "
              f"99th pct={inten_scale:.1f} (used for normalization)")
        # Normalize intensity images to [0, 1]
        for f in frames:
            if 'intensity_gt' in f:
                f['intensity_gt'] = np.clip(f['intensity_gt'] / max(inten_scale, 1.0), 0, 1).astype(np.float32)
    else:
        inten_scale = None

    pts_world = np.concatenate(all_world_pts, axis=0)
    pts_world = voxel_downsample(pts_world, INIT_VOXEL_SIZE)
    print(f"  Init point cloud: {len(pts_world)} points (voxel={INIT_VOXEL_SIZE}m)")

    n_train = int(len(frames) * train_ratio)
    train_frames = frames[:n_train]
    val_frames = frames[n_train:]
    print(f"  Train: {len(train_frames)}, Val: {len(val_frames)}")

    return train_frames, val_frames, pts_world, K


# ── 3DGS Model ───────────────────────────────────────────────────────────────

def init_gaussians(pts_world, device='cuda'):
    """Initialize Gaussians from LiDAR point cloud."""
    points = torch.from_numpy(pts_world).float().to(device)
    N = len(points)

    # Use voxel size as initial scale (simpler than KNN for large point clouds)
    init_scale = np.log(INIT_VOXEL_SIZE)
    scales = torch.full((N, 3), init_scale, device=device)

    quats = torch.zeros((N, 4), device=device)
    quats[:, 0] = 1.0  # identity rotation
    opacities = torch.logit(torch.full((N,), 0.5, device=device))

    # Dummy colors (gray) - we don't need them but gsplat requires colors arg
    colors = torch.full((N, 3), 0.5, device=device)

    splats = {
        'means': torch.nn.Parameter(points),
        'scales': torch.nn.Parameter(scales),
        'quats': torch.nn.Parameter(quats),
        'opacities': torch.nn.Parameter(opacities),
        'colors': torch.nn.Parameter(colors),
    }
    return splats


def render_depth(splats, viewmat, K, W, H, device='cuda'):
    """Render expected depth (and optionally intensity/raydrop) using gsplat rasterization.
    Returns (depth, colors_img, alpha, info).
    colors_img: [H,W,3] — ch0=intensity, ch1=raydrop logit, ch2=unused.
    alpha: [H,W] — accumulated opacity along ray.
    """
    from gsplat import rasterization

    viewmat_t = torch.from_numpy(viewmat).float().unsqueeze(0).to(device)
    K_t = torch.from_numpy(K).float().unsqueeze(0).to(device)

    renders, alphas, info = rasterization(
        means=splats['means'],
        quats=splats['quats'],
        scales=torch.exp(splats['scales']),
        opacities=torch.sigmoid(splats['opacities']),
        colors=splats['colors'],
        viewmats=viewmat_t,
        Ks=K_t,
        width=W,
        height=H,
        render_mode="RGB+ED",
        near_plane=0.1,
        far_plane=MAX_RANGE_M + 1.0,
        sh_degree=None,
    )
    depth = renders[0, :, :, 3]  # [H, W]
    colors_img = renders[0, :, :, :3]  # [H, W, 3]
    alpha = alphas[0, :, :, 0]  # [H, W]
    return depth, colors_img, alpha, info


# ── Densification ────────────────────────────────────────────────────────────

def densify_and_prune(splats, optimizer, grad_thresh=0.0002, min_opacity=0.05,
                      max_gaussians=500000):
    """Simple densification: clone high-gradient Gaussians, prune low-opacity ones."""
    means = splats['means']
    if means.grad is None:
        return

    grad_norms = means.grad.norm(dim=1)
    N = len(means)

    # Clone Gaussians with high position gradient
    clone_mask = grad_norms > grad_thresh
    n_clone = clone_mask.sum().item()

    # Prune low-opacity Gaussians
    opac = torch.sigmoid(splats['opacities'].data)
    prune_mask = opac < min_opacity
    keep_mask = ~prune_mask
    n_prune = prune_mask.sum().item()

    if n_clone == 0 and n_prune == 0:
        return

    # Cap cloning to avoid memory explosion
    if N + n_clone - n_prune > max_gaussians:
        n_clone = max(0, max_gaussians - N + n_prune)
        if n_clone > 0:
            _, topk_idx = grad_norms.topk(n_clone)
            clone_mask = torch.zeros(N, dtype=torch.bool, device=means.device)
            clone_mask[topk_idx] = True
        else:
            clone_mask = torch.zeros(N, dtype=torch.bool, device=means.device)

    with torch.no_grad():
        for key in splats:
            old = splats[key].data
            if clone_mask.any():
                cloned = old[clone_mask]
                if key == 'means':
                    cloned = cloned + torch.randn_like(cloned) * 0.01
                new_data = torch.cat([old, cloned], dim=0)
            else:
                new_data = old

            # Now prune (extend prune mask to include cloned — they're kept)
            full_keep = torch.ones(len(new_data), dtype=torch.bool, device=means.device)
            full_keep[:N] = keep_mask
            new_data = new_data[full_keep]

            splats[key] = torch.nn.Parameter(new_data)

        # Rebuild optimizer
        for pg in optimizer.param_groups:
            for p_key in ['means', 'scales', 'quats', 'opacities', 'colors']:
                if pg['params'][0].data_ptr() == id(splats[p_key]):
                    pg['params'] = [splats[p_key]]

    return n_clone, n_prune


def rebuild_optimizer(splats, args):
    """Create fresh optimizer after densification changes parameter sizes."""
    return torch.optim.Adam([
        {'params': [splats['means']], 'lr': args.lr_means},
        {'params': [splats['scales']], 'lr': args.lr_scales},
        {'params': [splats['quats']], 'lr': args.lr_quats},
        {'params': [splats['opacities']], 'lr': args.lr_opacities},
        {'params': [splats['colors']], 'lr': args.lr_means},
    ])


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    out_dir = PROBE_ROOT / 'experiments' / args.exp
    out_dir.mkdir(parents=True, exist_ok=True)

    use_intensity = args.intensity
    use_densify = args.densify
    use_raydrop = args.raydrop

    # Prepare data
    train_frames, val_frames, pts_world, K = prepare_data(
        max_frames=args.max_frames,
        train_ratio=0.9,
        use_cubemap=args.cubemap,
        use_intensity=use_intensity,
        use_colmap=args.use_colmap,
    )

    # Initialize Gaussians
    print(f"Initializing {len(pts_world)} Gaussians...")
    splats = init_gaussians(pts_world, device)

    if use_intensity:
        # Initialize colors to 0.5 (will be trained as intensity)
        splats['colors'].data.fill_(0.0)

    optimizer = rebuild_optimizer(splats, args)

    n_train = len(train_frames)
    features = []
    if use_intensity: features.append("intensity")
    if use_raydrop: features.append("raydrop")
    if use_densify: features.append("densification")
    print(f"\nStarting training: {args.steps} steps, {n_train} train frames"
          + (f" [{', '.join(features)}]" if features else ""))

    best_val_loss = float('inf')
    t0 = time.time()

    for step in range(args.steps):
        optimizer.zero_grad()

        idx = np.random.randint(n_train)
        frame = train_frames[idx]

        depth_gt = torch.from_numpy(frame['depth_gt']).float().to(device)
        depth_pred, colors_pred, alpha_pred, info = render_depth(splats, frame['viewmat'], K, VCAM_W, VCAM_H, device)

        valid = depth_gt > 0
        if valid.sum() == 0:
            continue

        loss = F.l1_loss(depth_pred[valid], depth_gt[valid])

        # Intensity loss (channel 0)
        if use_intensity and 'intensity_gt' in frame:
            inten_pred = colors_pred[:, :, 0]
            inten_gt = torch.from_numpy(frame['intensity_gt']).float().to(device)
            inten_valid = inten_gt > 0
            if inten_valid.sum() > 0:
                inten_loss = F.l1_loss(inten_pred[inten_valid], inten_gt[inten_valid])
                loss = loss + 0.1 * inten_loss

        # Ray-drop loss (channel 1): predict which pixels have real LiDAR returns
        if use_raydrop:
            raydrop_logit = colors_pred[:, :, 1]
            raydrop_gt = (depth_gt > 0).float()
            n_pos = raydrop_gt.sum()
            n_neg = (1 - raydrop_gt).sum()
            pos_weight = torch.clamp(n_neg / (n_pos + 1), 1.0, 50.0)
            raydrop_loss = F.binary_cross_entropy_with_logits(
                raydrop_logit, raydrop_gt,
                pos_weight=pos_weight)
            loss = loss + 0.05 * raydrop_loss

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            splats['opacities'].data.clamp_(-10, 10)

        # Densification every 1000 steps during first half of training
        if use_densify and step % 1000 == 0 and step > 0 and step < args.steps * 0.6:
            result = densify_and_prune(splats, optimizer)
            if result:
                n_clone, n_prune = result
                optimizer = rebuild_optimizer(splats, args)
                if n_clone > 0 or n_prune > 0:
                    print(f"  [DENSIFY] step={step} +{n_clone}/-{n_prune} → {len(splats['means'])} gaussians")

        if step % 500 == 0:
            elapsed = time.time() - t0
            n_gauss = len(splats['means'])
            print(f"  Step {step:6d}/{args.steps} | loss={loss.item():.4f} | "
                  f"gaussians={n_gauss} | {elapsed:.0f}s")

        if step % 2000 == 0 and step > 0 and val_frames:
            val_loss = evaluate(splats, val_frames, K, device)
            print(f"  [VAL] step={step} val_depth_L1={val_loss:.4f}m")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(splats, out_dir / 'best.pt')

    if val_frames:
        val_loss = evaluate(splats, val_frames, K, device)
        print(f"\n  [FINAL VAL] depth_L1={val_loss:.4f}m (best={best_val_loss:.4f}m)")

    save_checkpoint(splats, out_dir / 'final.pt')
    print(f"Saved to {out_dir}")
    export_gaussians_ply(splats, out_dir / 'gaussians.ply')

    return val_loss if val_frames else loss.item()


def evaluate(splats, val_frames, K, device):
    """Evaluate depth L1 on validation frames."""
    losses = []
    with torch.no_grad():
        for frame in val_frames:
            depth_gt = torch.from_numpy(frame['depth_gt']).float().to(device)
            depth_pred, _, _, _ = render_depth(splats, frame['viewmat'], K, VCAM_W, VCAM_H, device)
            valid = depth_gt > 0
            if valid.sum() > 0:
                l1 = F.l1_loss(depth_pred[valid], depth_gt[valid]).item()
                losses.append(l1)
    return np.mean(losses) if losses else float('inf')


def save_checkpoint(splats, path):
    torch.save({k: v.data.cpu() for k, v in splats.items()}, str(path))


def export_gaussians_ply(splats, path):
    """Export Gaussian means as PLY for geometric evaluation."""
    import open3d as o3d
    pts = splats['means'].detach().cpu().numpy()
    opac = torch.sigmoid(splats['opacities']).detach().cpu().numpy()
    # Filter low-opacity Gaussians
    mask = opac > 0.1
    pts = pts[mask]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.full((len(pts), 3), 0.5))
    o3d.io.write_point_cloud(str(path), pcd)
    print(f"  Exported {len(pts)} Gaussians to {path}")


# ── Synthetic LiDAR rendering ────────────────────────────────────────────────

def render_synthetic_lidar(args):
    """Load trained model and render synthetic LiDAR from held-out poses."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = PROBE_ROOT / 'experiments' / args.exp

    ckpt_path = out_dir / 'best.pt'
    if not ckpt_path.exists():
        ckpt_path = out_dir / 'final.pt'
    print(f"Loading checkpoint: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    splats = {k: torch.nn.Parameter(v.to(device)) for k, v in ckpt.items()}

    _, val_frames, _, K = prepare_data(train_ratio=0.9, use_cubemap=args.cubemap,
                                       use_intensity=getattr(args, 'intensity', False),
                                       use_colmap=getattr(args, 'use_colmap', False))

    synth_dir = out_dir / 'synthetic_lidar'
    synth_dir.mkdir(exist_ok=True)

    all_synth_pts = []
    all_real_pts = []

    with torch.no_grad():
        for i, frame in enumerate(val_frames):
            depth_pred, _, _, _ = render_depth(splats, frame['viewmat'], K, VCAM_W, VCAM_H, device)
            depth_pred_np = depth_pred.cpu().numpy()

            # Convert depth image to 3D points (in camera frame)
            v_coords, u_coords = np.where(depth_pred_np > 0.1)
            d = depth_pred_np[v_coords, u_coords]
            x = (u_coords - K[0, 2]) * d / K[0, 0]
            y = (v_coords - K[1, 2]) * d / K[1, 1]
            z = d
            synth_cam = np.stack([x, y, z], axis=1)

            # Camera → world
            viewmat = frame['viewmat']
            R_cw = viewmat[:3, :3]
            t_cw = viewmat[:3, 3]
            R_wc = R_cw.T
            t_wc = -R_wc @ t_cw
            synth_world = (R_wc @ synth_cam.T).T + t_wc
            all_synth_pts.append(synth_world)

            # Same for ground truth
            depth_gt = frame['depth_gt']
            v_gt, u_gt = np.where(depth_gt > 0)
            d_gt = depth_gt[v_gt, u_gt]
            x_gt = (u_gt - K[0, 2]) * d_gt / K[0, 0]
            y_gt = (v_gt - K[1, 2]) * d_gt / K[1, 1]
            z_gt = d_gt
            real_cam = np.stack([x_gt, y_gt, z_gt], axis=1)
            real_world = (R_wc @ real_cam.T).T + t_wc
            all_real_pts.append(real_world)

    synth_all = np.concatenate(all_synth_pts, axis=0)
    real_all = np.concatenate(all_real_pts, axis=0)
    print(f"Synthetic: {len(synth_all)} points, Real: {len(real_all)} points")

    # Geometric evaluation
    from scipy.spatial import cKDTree

    tree_real = cKDTree(real_all)
    tree_synth = cKDTree(synth_all)

    d_synth2real, _ = tree_real.query(synth_all, k=1)
    d_real2synth, _ = tree_synth.query(real_all, k=1)

    chamfer = (d_synth2real.mean() + d_real2synth.mean()) / 2
    for thresh in [0.05, 0.10, 0.20]:
        prec = (d_synth2real < thresh).mean()
        rec  = (d_real2synth < thresh).mean()
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        print(f"  F-score@{thresh:.2f}m: {f1:.3f} (P={prec:.3f}, R={rec:.3f})")
    print(f"  Chamfer Distance: {chamfer:.4f}m")

    # Save
    import open3d as o3d
    for name, pts in [('synthetic', synth_all), ('real_holdout', real_all)]:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts[:500000].astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(
            np.full((min(len(pts), 500000), 3), 0.3 if name == 'synthetic' else 0.7)
        )
        o3d.io.write_point_cloud(str(synth_dir / f'{name}.ply'), pcd)

    return chamfer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default='lidar_3dgs_v1')
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--max-frames', type=int, default=None)
    parser.add_argument('--lr-means', type=float, default=1e-4)
    parser.add_argument('--lr-scales', type=float, default=5e-3)
    parser.add_argument('--lr-quats', type=float, default=1e-3)
    parser.add_argument('--lr-opacities', type=float, default=5e-2)
    parser.add_argument('--cubemap', action='store_true', help='Use 6-face cubemap instead of single front camera')
    parser.add_argument('--intensity', action='store_true', help='Enable intensity supervision')
    parser.add_argument('--densify', action='store_true', help='Enable densification (clone/prune)')
    parser.add_argument('--raydrop', action='store_true', help='Enable ray-drop prediction (colors ch1)')
    parser.add_argument('--render', action='store_true', help='Render synthetic LiDAR from trained model')
    parser.add_argument('--use-colmap', action='store_true',
                        help='Use COLMAP camera poses (via camera_sync) instead of MoCap. '
                             'Eliminates MoCap pitch/Z errors for better 3D geometry.')
    args = parser.parse_args()

    if args.render:
        render_synthetic_lidar(args)
    else:
        train(args)
