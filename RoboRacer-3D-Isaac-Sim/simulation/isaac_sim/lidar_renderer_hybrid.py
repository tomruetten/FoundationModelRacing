"""
lidar_renderer_hybrid.py — MoCap+LiDAR hybrid renderer (2026-07-03).

For a query pose that corresponds to a literal replay frame (we know its
nerfstudio camera_frame_idx), returns the REAL captured Livox scan from that
exact instant -- zero pose error, guaranteed-crisp real points, no synthetic
model involved.

For a genuinely novel query pose (no known camera_frame_idx, e.g. a future
MLP-extrapolated pose), falls back to nearest-real-scan lookup gated by a
distance/angle threshold, and only invokes the trained LiDAR-3DGS synthetic
model when no real scan is close enough.

2026-07-03 (later same day): the synthetic-model fallback used to call
LidarRenderer.render() (dense per-pixel cubemap -> uniform "blob", the
original streaking problem). Replaced with render_along_rays(), which queries
the SAME trained model but only at pixel locations matching a real scan's
actual ray-angle pattern (borrowed from the nearest real scan, any distance)
-- this reproduces genuine Livox scan-line structure while still being the
network's own pose-conditioned inference, so it generalizes to truly novel
poses (unlike branches 1/2, which just replay captured data). Verified
cross-pose (template from one scan, query pose from a different one):
Chamfer/F-score both improve over the old dense method, and the point cloud
visually shows scan lines instead of a diffuse blob. See
diagnostics/eval_raycast.py and chat history for the comparison.

Same .render(R_wl, t_wl, use_raydrop, camera_frame_idx) interface as
LidarRenderer, so this is a drop-in replacement in lidar_render_server.py.
"""
import numpy as np

from lidar_scan_lookup import ScanLookup

NEAREST_POS_THRESH_M = 0.05
# 2026-07-04: after the COLMAP/MoCap position correction in
# isaac_sim_lidar_splat_demo.py (c2w_to_lidar_pose -> _apply_colmap_mocap_correction),
# position error dropped from median 14.7cm to 1.6cm, but ORIENTATION error stayed
# ~10 deg -- found to be per-frame COLMAP rotation noise, not a fixable systematic
# bias (a rotation-averaging fit across all 1066 matched frames didn't reduce it).
# 3.0 deg was unrealistically tight given this noise floor and made this branch
# almost never fire; 15.0 deg matches the actual noise level and lifts coverage
# from ~0% to ~85% of frames (measured on the replay trajectory).
NEAREST_ANGLE_THRESH_DEG = 15.0


class HybridLidarRenderer:
    def __init__(self, checkpoint_path=None, device: str = "cuda"):
        self.lut = ScanLookup()
        self._synth = None  # lazy-loaded LidarRenderer, only if a fallback is needed
        self._checkpoint_path = checkpoint_path
        self._device = device
        self.n_real = 0
        self.n_synth = 0

    def _synth_renderer(self):
        if self._synth is None:
            from lidar_renderer import LidarRenderer, CHECKPOINT_PATH
            ckpt = self._checkpoint_path or CHECKPOINT_PATH
            self._synth = LidarRenderer(checkpoint_path=ckpt, device=self._device)
        return self._synth

    def render(self, R_wl: np.ndarray, t_wl: np.ndarray,
               use_raydrop: bool = True, camera_frame_idx: int = None) -> tuple:
        # 1) exact replay match
        if camera_frame_idx is not None:
            idx = self.lut.exact(camera_frame_idx)
            if idx is not None:
                self.n_real += 1
                xyz, intensity = self.lut.scan(idx)
                return xyz.astype(np.float32), intensity.astype(np.float32)

        # 2) nearest-real-scan, gated by distance/angle
        idx, dist = self.lut.nearest(t_wl)
        if dist < NEAREST_POS_THRESH_M:
            ang = self.lut.angle_deg(idx, R_wl)
            if ang < NEAREST_ANGLE_THRESH_DEG:
                self.n_real += 1
                xyz, intensity = self.lut.scan(idx)
                return xyz.astype(np.float32), intensity.astype(np.float32)

        # 3) genuinely novel pose: ray-cast the trained model along the nearest
        # real scan's angular pattern (used only as an angle TEMPLATE -- the
        # depth/intensity values are genuine model inference at this pose, not
        # copied from that scan). raydrop is intentionally off here: its
        # per-pixel threshold was tuned for the dense grid and drops ~90% of
        # legitimate points when queried at these sparse ray-template pixel
        # locations (see chat history 2026-07-03); use gsplat's own accumulated
        # alpha (opacity) as the validity filter instead -- alpha_thresh=0.99
        # measured (2026-07-04, full 107-scan held-out set, 3 templates/frame
        # averaged) to give Chamfer=26.3cm/F@10cm=46.1%, vs 35.1cm/40.9% with
        # depth>0.1 alone. See lidar_evaluation_comparison_2026-07-04.txt.
        self.n_synth += 1
        xyz_template, _ = self.lut.scan(idx)
        return self._synth_renderer().render_along_rays(
            R_wl, t_wl, xyz_template, use_raydrop=False, alpha_thresh=0.99)
