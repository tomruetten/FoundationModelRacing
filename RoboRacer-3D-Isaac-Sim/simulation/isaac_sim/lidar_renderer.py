"""
Pose -> LiDAR point cloud rendering bridge for the trained LiDAR-3DGS model.

Analogous to splat_camera_renderer.py, but for LiDAR rather than RGB camera.
The LiDAR-3DGS (experiments/lidar_3dgs_v7_raydrop/best.pt, MoCap-posed
training -- see scripts/train_lidar_3dgs.py) is a raw gsplat
checkpoint trained to reproduce Livox Mid-360 scans.  It uses a 6-face
cubemap renderer (each face: 256x256 pinhole, 90° FOV) to cover the full
360° horizontal + ~180° vertical FOV of the Livox sensor.

Key differences from the camera renderer:
  - Checkpoint format: raw dict {means, scales, quats, opacities, colors}
    NOT a nerfstudio pipeline -- load directly with torch.load()
  - Output: 3-D point cloud (Nx3, sensor frame) + intensity (N,)
    NOT an RGB image
  - Coordinate convention: LiDAR sensor frame  X-forward, Y-left, Z-up
    (same as Livox Mid-360's native frame)
  - Isaac Sim receives the point cloud as an NPZ file:
    {"xyz": Nx3 float32, "intensity": N float32}  -- values in sensor frame

Usage (standalone selftest):
    conda run -n nerfstudio-py310 python3 scripts/lidar_renderer.py --selftest

Usage (as module, from lidar_render_server.py):
    from lidar_renderer import LidarRenderer
    r = LidarRenderer()
    xyz, intensity = r.render(R_wl, t_wl)   # -> Nx3, N  (sensor frame, metres)
"""
from pathlib import Path

import numpy as np
import torch

from object_splats import LIDAR_OBJECT_MIN_OPACITY

OBJECT_ALPHA_SOLID = 0.9   # object layer counts as a surface where its accumulated alpha reaches this
OBJECT_LIDAR_STRIDE = 4    # LiDAR layer uses every 4th object gaussian with a 2x footprint (see ObjectSplat.proxy)

# Paths relative to this script so they work on any machine after extraction.
_SCRIPTS_DIR = Path(__file__).parent
_EXPERIMENTS_DIR = _SCRIPTS_DIR.parent / "experiments"

CHECKPOINT_PATH = _EXPERIMENTS_DIR / "lidar_3dgs_v7_raydrop" / "best.pt"
# REVERTED to the original MoCap-posed model on 2026-07-03, after switching to
# lidar_3dgs_colmap_v2_50k on 2026-07-02 turned out to be a mistake: the home
# machine ran a controlled, T_cl-bug-fixed comparison the same day and found
# the COLMAP-pose path fundamentally worse (Chamfer 21.5cm, F@10cm=69%, 65k
# gaussians) than the MoCap path (Chamfer 4.3cm, F@10cm=93%, 230k gaussians).
# The MoCap path doesn't route through T_cl at all, so it was never affected
# by that bug and remains the validated best. See the session notes for the
# full writeup (2026-07-02 COLMAP experiment + 2026-07-03 correction).

# Virtual pinhole camera for each cubemap face
VCAM_W, VCAM_H = 256, 256
VCAM_FOV_DEG = 90.0
MAX_RANGE_M = 8.0

# Cubemap face rotations in sensor frame (X-fwd, Y-left, Z-up).
# Each R_face rotates the sensor frame so the face's direction becomes +X.
CUBEMAP_FACES = {
    "front": np.eye(3),
    "back":  np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float64),
    "left":  np.array([[0, -1, 0], [1,  0, 0], [0, 0, 1]], dtype=np.float64),
    "right": np.array([[0,  1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64),
    "up":    np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float64),
    "down":  np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.float64),
}

# LiDAR sensor frame (X-fwd, Y-left, Z-up) → pinhole camera frame (X-right, Y-down, Z-fwd)
R_CL_BASE = np.array([[0, -1, 0],
                       [0,  0, -1],
                       [1,  0,  0]], dtype=np.float64)

RAYDROP_THRESHOLD = 0.5   # sigmoid(logit) > threshold → point dropped (not a real return)
# REVERTED to the original default on 2026-07-03. Was lowered to 0.3 on 2026-07-02
# while lidar_3dgs_colmap_v2_50k was the active checkpoint (see git history / prior
# comment), but that checkpoint itself was reverted the same day (see CHECKPOINT_PATH
# above). Re-tested 0.3 against the reinstated lidar_3dgs_v7_raydrop/best.pt: it wipes
# out nearly all of the central/near structure (13k pts, mostly scattered peripheral
# fragments) rather than cleaning noise -- the threshold's effect is not portable
# across differently-trained checkpoints, each has its own raydrop logit calibration.
# 0.5 is the value the validated Chamfer=4.3cm/F@10cm=93% result was measured with;
# don't change it without a fresh, model-specific real-vs-synthetic comparison.


def _make_K(W: int, H: int, fov_deg: float) -> np.ndarray:
    f = W / (2.0 * np.tan(np.radians(fov_deg / 2.0)))
    return np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], dtype=np.float32)


def _make_viewmat(R_wl: np.ndarray, t_wl: np.ndarray, R_face: np.ndarray = None) -> np.ndarray:
    """Build 4x4 world-to-camera viewmat for gsplat.
    R_wl: 3x3 rotation  world→LiDAR sensor
    t_wl: (3,) translation world→LiDAR (i.e. sensor origin in world frame)
    R_face: cubemap face rotation in sensor frame (None = front face only)
    """
    R_lw = R_wl.T
    t_lw = -R_lw @ t_wl

    R_cl = R_CL_BASE @ R_face.T if R_face is not None else R_CL_BASE
    R_cw = R_cl @ R_lw
    t_cw = R_cl @ t_lw

    viewmat = np.eye(4, dtype=np.float32)
    viewmat[:3, :3] = R_cw
    viewmat[:3, 3] = t_cw
    return viewmat


def _depth_to_points_sensor(depth: np.ndarray, K: np.ndarray, R_face: np.ndarray) -> np.ndarray:
    """Back-project a depth image to 3-D points in LiDAR sensor frame.
    depth: [H,W] float32, metres (0 = invalid)
    R_face: cubemap face rotation (3x3)
    Returns: [N,3] in sensor frame (X-fwd, Y-left, Z-up)
    """
    H, W = depth.shape
    v_idx, u_idx = np.where(depth > 0)
    if len(v_idx) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    # unproject to pinhole camera frame
    x_cam = (u_idx - K[0, 2]) / K[0, 0]
    y_cam = (v_idx - K[1, 2]) / K[1, 1]
    z_cam = depth[v_idx, u_idx]

    pts_cam = np.stack([x_cam * z_cam, y_cam * z_cam, z_cam], axis=1).astype(np.float32)

    # camera frame → sensor frame: invert R_CL_BASE then invert R_face
    # R_cl = R_CL_BASE @ R_face.T  →  R_lc = R_face @ R_CL_BASE.T
    R_lc = R_face @ R_CL_BASE.T
    pts_sensor = (R_lc @ pts_cam.T).T
    return pts_sensor


def _render_face(splats: dict, viewmat: np.ndarray, K: np.ndarray,
                 device: str, return_alpha: bool = False) -> tuple:
    """Render one cubemap face. Returns (depth [H,W], colors [H,W,3]), plus
    alpha [H,W] (gsplat accumulated opacity) if return_alpha=True."""
    from gsplat import rasterization

    viewmat_t = torch.from_numpy(viewmat).float().unsqueeze(0).to(device)
    K_t = torch.from_numpy(K).float().unsqueeze(0).to(device)

    renders, alphas, _ = rasterization(
        means=splats["means"],
        quats=splats["quats"],
        scales=torch.exp(splats["scales"]),
        opacities=torch.sigmoid(splats["opacities"]),
        colors=splats["colors"],
        viewmats=viewmat_t,
        Ks=K_t,
        width=VCAM_W,
        height=VCAM_H,
        render_mode="RGB+ED",
        near_plane=0.1,
        far_plane=MAX_RANGE_M + 1.0,
        sh_degree=None,
    )
    depth = renders[0, :, :, 3].cpu().numpy()
    colors_img = renders[0, :, :, :3].cpu().numpy()
    if return_alpha:
        alpha = alphas[0, :, :, 0].cpu().numpy()
        return depth, colors_img, alpha
    return depth, colors_img


class LidarRenderer:
    def __init__(self, checkpoint_path: Path = CHECKPOINT_PATH, device: str = "cuda"):
        self.device = device
        self.K = _make_K(VCAM_W, VCAM_H, VCAM_FOV_DEG)

        raw = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.splats = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in raw.items()
        }
        # ensure requires_grad=False for inference
        for p in self.splats.values():
            if isinstance(p, torch.Tensor):
                p.requires_grad_(False)
        n = self.splats["means"].shape[0]
        print(f"[LidarRenderer] loaded {n:,} Gaussians from {checkpoint_path.name}")

    def object_splats(self, objects) -> dict | None:
        """Posed object gaussians (object_splats.ObjectSplat, T_wo in the LiDAR world frame) as
        one gsplat dict, or None. Objects are made opaque (solid surfaces) and get a
        never-dropped raydrop channel."""
        if not objects:
            return None
        parts = {k: [] for k in ("means", "scales", "quats", "opacities", "colors")}
        n_ch = self.splats["colors"].shape[1]
        stride = getattr(self, "object_lidar_stride", OBJECT_LIDAR_STRIDE)
        for entry in objects:
            obj, T_wo, part_tf = (entry + (None,))[:3] if len(entry) == 2 else entry
            obj = obj.proxy(stride)
            p = obj.posed(np.asarray(T_wo, dtype=np.float64), device=self.device,
                          opaque_min=LIDAR_OBJECT_MIN_OPACITY, part_transforms=part_tf, for_lidar=True)
            parts["means"].append(p["means"])
            parts["scales"].append(p["scales_log"])
            parts["quats"].append(p["quats"])
            parts["opacities"].append(p["opacities"])
            parts["colors"].append(obj.lidar_colors(n_ch, device=self.device))
        return {k: torch.cat(v) for k, v in parts.items()}

    def _render_face_composited(self, obj_splats, viewmat: np.ndarray) -> tuple:
        """Scene face render with objects composited by FIRST SURFACE: a pixel takes the object
        depth where the object layer is solid (alpha >= OBJECT_ALPHA_SOLID) and nearer than the
        scene surface. Rendering objects inside the map instead would mix their depth with the
        map's low-opacity fog through gsplat's expected depth (13 cm bias measured on the
        F1TENTH splat 2 m ahead in the New1 map). Returns (depth, colors, alpha)."""
        depth, colors_img, alpha = _render_face(self.splats, viewmat, self.K, self.device, return_alpha=True)
        if obj_splats is None:
            return depth, colors_img, alpha
        d_o, c_o, a_o = _render_face(obj_splats, viewmat, self.K, self.device, return_alpha=True)
        take = (a_o >= OBJECT_ALPHA_SOLID) & (d_o > 0.05) & ((d_o < depth) | (depth <= 0) | (alpha < OBJECT_ALPHA_SOLID))
        depth = np.where(take, d_o, depth)
        colors_img = np.where(take[..., None], c_o, colors_img)
        alpha = np.where(take, 1.0, alpha)
        return depth, colors_img, alpha

    def render(self, R_wl: np.ndarray, t_wl: np.ndarray,
               use_raydrop: bool = True, alpha_thresh: float = None, objects=None) -> tuple:
        """
        Render a full 360° LiDAR scan by rendering all 6 cubemap faces.

        Args:
            R_wl: (3,3) LIDAR-TO-WORLD rotation matrix (same convention as
                  train_lidar_3dgs.py -- confusingly named, but matches the
                  training code's make_viewmat() which does R_lw = R_wl.T)
            t_wl: (3,) LiDAR sensor origin in world frame (metres)
            use_raydrop: apply raydrop mask (ch1 of colors is raydrop logit)
            alpha_thresh: if set, drop pixels whose accumulated opacity is
                  below this value (use ~0.99 for checkpoints without raydrop)
            objects: optional list of (ObjectSplat, T_wo) rendered as a separate
                  layer and composited by first surface (opponents, cones);
                  T_wo object -> world frame

        Returns:
            xyz:       (N,3) float32 point cloud in LiDAR sensor frame (metres)
            intensity: (N,)  float32 intensity (0-1), or zeros if not available
        """
        all_xyz = []
        all_intensity = []
        obj_splats = self.object_splats(objects)

        for face_name, R_face in CUBEMAP_FACES.items():
            viewmat = _make_viewmat(R_wl, t_wl, R_face)
            depth, colors_img, alpha = self._render_face_composited(obj_splats, viewmat)

            if use_raydrop:
                raydrop_prob = 1.0 / (1.0 + np.exp(-colors_img[:, :, 1]))
                depth[raydrop_prob > RAYDROP_THRESHOLD] = 0.0
            if alpha_thresh is not None:
                # Checkpoints trained without a raydrop head (e.g. New1
                # checkpoints/lidar_3dgs/best.pt) use gsplat's accumulated
                # opacity instead: drop rays that did not hit solid geometry.
                depth[alpha < alpha_thresh] = 0.0

            pts = _depth_to_points_sensor(depth, self.K, R_face)
            if len(pts) == 0:
                continue

            v_idx, u_idx = np.where(depth > 0)
            inten = colors_img[v_idx, u_idx, 0].astype(np.float32)

            all_xyz.append(pts)
            all_intensity.append(inten)

        if not all_xyz:
            return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.float32)

        xyz = np.concatenate(all_xyz, axis=0)
        intensity = np.concatenate(all_intensity, axis=0)
        return xyz, intensity

    def render_along_rays(self, R_wl: np.ndarray, t_wl: np.ndarray,
                          ray_dirs_sensor: np.ndarray,
                          use_raydrop: bool = True,
                          alpha_thresh: float = None, objects=None) -> tuple:
        """Render depth/intensity along a FIXED set of ray directions (sensor
        frame, unit vectors) instead of every pixel of the dense cubemap grid.

        Why: the dense per-pixel cubemap render (`render()` above) queries a
        virtual pinhole camera at every pixel -- a uniform sampling of the
        scene surface. Real Livox Mid-360 scans are a non-repetitive rotating
        scan: a sparse set of rays along curved scan-line trajectories, NOT a
        uniform grid. Random-subsampling the dense output to match real point
        counts still looks like a diffuse "blob" (verified 2026-07-03 -- see
        chat/session log), because the points being kept are still a uniform
        random sample, not the actual scan-line structure.

        This method instead queries the SAME trained model, but only at pixel
        locations corresponding to a real scan's actual ray directions
        (`ray_dirs_sensor`, e.g. from `ScanLookup.scan(idx)` normalized to unit
        vectors). Livox's scan pattern is non-repetitive but its per-~100ms-
        window angular coverage is characteristic of the sensor, not of any
        one specific pose -- so a template borrowed from ANY real scan can be
        reused for ANY query pose (novel or replay). The output keeps the
        network's learned, pose-conditioned geometry (works for genuinely
        novel poses, unlike the MoCap+LiDAR hybrid's real-scan lookup) while
        matching the real sensor's sparse, structured scan-line appearance.

        Args:
            ray_dirs_sensor: (N,3) ray directions in LiDAR sensor frame
                (X-fwd, Y-left, Z-up). Need not be unit vectors -- normalized
                internally.
        Returns:
            xyz:       (M,3) float32, M <= N (rays with no valid return dropped)
            intensity: (M,) float32
        """
        dirs = ray_dirs_sensor / np.linalg.norm(ray_dirs_sensor, axis=1, keepdims=True)
        N = len(dirs)
        out_depth = np.zeros(N, dtype=np.float32)
        out_intensity = np.zeros(N, dtype=np.float32)
        out_valid = np.zeros(N, dtype=bool)
        obj_splats = self.object_splats(objects)

        for face_name, R_face in CUBEMAP_FACES.items():
            pts_rot = (R_face.T @ dirs.T).T
            x_cam = -pts_rot[:, 1]
            y_cam = -pts_rot[:, 2]
            z_cam = pts_rot[:, 0]
            in_front = z_cam > 0.15
            if not in_front.any():
                continue

            u = self.K[0, 0] * x_cam / z_cam + self.K[0, 2]
            v = self.K[1, 1] * y_cam / z_cam + self.K[1, 2]
            in_bounds = in_front & (u >= 0) & (u < VCAM_W - 1) & (v >= 0) & (v < VCAM_H - 1)
            idxs = np.where(in_bounds)[0]
            if len(idxs) == 0:
                continue

            viewmat = _make_viewmat(R_wl, t_wl, R_face)
            depth_img, colors_img, alpha_img = self._render_face_composited(obj_splats, viewmat)

            ui = np.clip(np.round(u[idxs]).astype(np.int32), 0, VCAM_W - 1)
            vi = np.clip(np.round(v[idxs]).astype(np.int32), 0, VCAM_H - 1)
            d = depth_img[vi, ui]
            inten = colors_img[vi, ui, 0]
            ok = d > 0.1
            if use_raydrop:
                raydrop_prob = 1.0 / (1.0 + np.exp(-colors_img[:, :, 1]))
                ok &= raydrop_prob[vi, ui] <= RAYDROP_THRESHOLD
            if alpha_thresh is not None:
                ok &= alpha_img[vi, ui] >= alpha_thresh

            # depth image = z along the face axis; convert to range along the ray
            # (up to 42 % short at cubemap face corners otherwise; fixed 2026-09-10)
            out_depth[idxs[ok]] = d[ok] / z_cam[idxs[ok]]
            out_intensity[idxs[ok]] = inten[ok]
            out_valid[idxs[ok]] = True

        xyz_sensor = dirs[out_valid] * out_depth[out_valid, None]
        intensity = out_intensity[out_valid]
        return xyz_sensor.astype(np.float32), intensity.astype(np.float32)


def _selftest():
    """Render from a known training pose, compare point count to expectations."""
    import json

    renderer = LidarRenderer()

    # Load a training pose from the LiDAR 3DGS training data directory
    # The training script uses MoCap poses; use the LiDAR pointcloud PLY origin
    # as a sanity-check pose (identity: sensor at world origin, looking +X)
    R_wl = np.eye(3)
    t_wl = np.zeros(3)
    print("[selftest] rendering from world-origin pose (all 6 cubemap faces)...")

    xyz, intensity = renderer.render(R_wl, t_wl, use_raydrop=True)
    print(f"[selftest] got {len(xyz):,} points, intensity range [{intensity.min():.3f}, {intensity.max():.3f}]")

    out_dir = _SCRIPTS_DIR.parent / "isaac_sim_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "lidar_renderer_selftest.npz", xyz=xyz, intensity=intensity)
    print(f"[selftest] saved to {out_dir}/lidar_renderer_selftest.npz")
    print("[selftest] DONE -- open the NPZ in Open3D or numpy to inspect the point cloud")

    # Also save as ASCII PLY for quick visual check
    ply_path = out_dir / "lidar_renderer_selftest.ply"
    with open(ply_path, "w") as f:
        f.write(f"ply\nformat ascii 1.0\n"
                f"element vertex {len(xyz)}\n"
                f"property float x\nproperty float y\nproperty float z\n"
                f"property float intensity\nend_header\n")
        for pt, iv in zip(xyz, intensity):
            f.write(f"{pt[0]:.4f} {pt[1]:.4f} {pt[2]:.4f} {iv:.4f}\n")
    print(f"[selftest] also saved PLY to {ply_path} for open3d/meshlab inspection")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        _selftest()
