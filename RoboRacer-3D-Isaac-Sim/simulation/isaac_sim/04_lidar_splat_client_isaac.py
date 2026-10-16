#!/usr/bin/env python3
# isaac_sim_lidar_splat_demo.py
"""
"LiDAR Splat in Isaac Sim" -- Phase 2 (Felix, 2026-06-30: "Sobald es
funktioniert ergänzen wir dann LiDAR")

DEPENDS ON Phase 1 (isaac_sim_camera_splat_demo.py) having confirmed that
Camera-3DGS renders correctly inside Isaac Sim.  Do NOT run this before
Phase 1 is validated.

What this adds:
  - The Livox Mid-360 LiDAR sensor is now replaced by the trained LiDAR-3DGS
    (experiments/lidar_3dgs_v7_raydrop/best.pt) -- a separate network that
    learned REAL Livox scan patterns (raydrop, intensity, geometry), per
    Felix's "eigenes Netzwerk für LiDAR" directive.
  - Each sim step: the LiDAR's 6-face cubemap is rendered by
    lidar_render_server.py (nerfstudio env, separate process) and returned as
    a point cloud NPZ file.  Isaac Sim reads the NPZ and can use it exactly
    as it would use the native Livox scan output.
  - Camera continues to use splat_render_server.py (same as Phase 1).

BEFORE RUNNING THIS:
  Terminal A: conda run -n nerfstudio-py310 python3 scripts/splat_render_server.py \\
                  --queue-dir isaac_sim_output/splat_queue
  Terminal B: conda run -n nerfstudio-py310 python3 scripts/lidar_render_server.py \\
                  --queue-dir isaac_sim_output/lidar_queue
  Terminal C (Isaac Sim Python): python3 scripts/isaac_sim_lidar_splat_demo.py --stride 2

Architecture:
  Isaac Sim (this script)
      │  camera pose c2w ──────────────────► splat_render_server.py → PNG
      │  lidar pose (R_wl, t_wl) ──────────► lidar_render_server.py → NPZ (Nx3 xyz)
      │
      ▼ saves combined output:
        isaac_sim_output/lidar_splat_demo/frame_XXXX_camera.png
        isaac_sim_output/lidar_splat_demo/frame_XXXX_lidar.npz

Coordinate conventions:
  Camera:  c2w is raw nerfstudio/OpenGL 4x4 (splat_render_server handles transform)
  LiDAR:   R_wl (3x3) + t_wl (3,) in world frame, sensor frame = X-fwd Y-left Z-up
           derived from c2w using the known T_camera_lidar extrinsic offset.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

os.environ["OMNI_KIT_ACCEPT_EULA"] = "yes"
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from object_splats import pose_from_camera_frame, trajectory_up, path_arclength, frame_ahead, FloorPlane, ObjectSplat, driving_state_from_path, attitude_from_accel  # noqa: E402

# ── T_camera_lidar: LiDAR origin relative to the camera, in camera frame ─────
# From lab_preparation_guide.md / TECHNICAL_NOTES.md:
#   T_BASE_LIDAR_t = [0.27, 0.00, 0.11]  (forward, left, up) in BASE frame
#   T_BASE_CAMERA approx [0, 0, ~0.20] in BASE frame (OAK-D on top of car)
# Approximate: LiDAR is ~27cm forward, ~9cm below the camera.
# NOTE: this is intentionally coarse; for production use basalt to calibrate
# the exact T_camera_lidar and replace these values.
T_BASE_LIDAR = np.array([0.27, 0.00, 0.11])   # LiDAR in vehicle base frame
T_BASE_CAMERA = np.array([0.00, 0.00, 0.20])  # Camera in vehicle base frame
T_CAMERA_LIDAR = T_BASE_LIDAR - T_BASE_CAMERA  # approx, metres


def load_real_trajectory(transforms_path: Path):
    data = json.load(open(transforms_path))
    frames = data["frames"]
    print(f"[trajectory] loaded {len(frames)} real frames from {transforms_path}")
    return frames


def nerfstudio_c2w_to_isaac_pose(c2w: np.ndarray):
    """Convert nerfstudio camera-to-world → Isaac Sim (Z-up) vehicle pose.
    See isaac_sim_camera_splat_demo.py for the full derivation.
    COLMAP world is Z-up (after Umeyama to MoCap) = Isaac Sim world → identity
    position mapping; only the camera→vehicle axis convention needs conversion.
    """
    from scipy.spatial.transform import Rotation as R

    pos_isaac = c2w[:3, 3].copy()   # Z-up world: direct mapping, no axis swap
    rot = c2w[:3, :3]
    R_vehicle_to_world = np.column_stack([-rot[:, 2], -rot[:, 0], rot[:, 1]])
    quat_xyzw = R.from_matrix(R_vehicle_to_world).as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    return pos_isaac, quat_wxyz


def vehicle_to_lidar_pose(pos_vehicle: np.ndarray, quat_wxyz: np.ndarray):
    """Compute LiDAR-3DGS pose from Isaac Sim vehicle (base_link) pose.

    lidar_renderer.py's make_viewmat() applies R_CL_BASE internally, so it
    expects R_wl = pure LiDAR sensor orientation in world (columns = LiDAR
    X-fwd/Y-left/Z-up axes expressed in world).  Composing c2w @ T_cl first
    and then passing that as R_wl causes make_viewmat() to double-apply the
    lidar->camera rotation, scrambling the output point cloud.

    Instead: derive R_wl directly from the vehicle pose.  Vehicle (base_link)
    and LiDAR share the same axis convention (X-fwd, Y-left, Z-up), so
    R_wl = R_vehicle_to_world.  Position adds the known T_base_lidar offset.

    T_BASE_LIDAR = [0.27, 0.00, 0.11] m (from ROS bag TF / extrinsics YAML).
    pos_vehicle approximates the camera position (c2w[:3,3]); the ~29 cm
    camera-to-base offset is intentionally ignored here for simplicity.
    """
    from scipy.spatial.transform import Rotation as R

    T_BASE_LIDAR_t = np.array([0.27, 0.0, 0.11])
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    R_wl = R.from_quat(quat_xyzw).as_matrix()   # pure LiDAR orientation in world
    t_wl = pos_vehicle + R_wl @ T_BASE_LIDAR_t   # LiDAR position in world
    return R_wl, t_wl


def request_render(queue_dir: Path, idx: int, payload: dict,
                   response_suffix: str, timeout_s: float = 30.0):
    """Write a request JSON, block until the response file appears (any extension for
    images: the camera server picks bmp/png/jpg)."""
    req_path = queue_dir / f"request_{idx:06d}.json"
    json.dump(payload, open(req_path, "w"))
    pattern = f"response_{idx:06d}.npz" if response_suffix == "npz" else f"response_{idx:06d}.*"
    t0 = time.time()
    while True:
        hits = [p for p in queue_dir.glob(pattern) if ".tmp" not in p.name]
        if hits:
            return hits[0]
        if time.time() - t0 > timeout_s:
            raise TimeoutError(
                f"render server did not respond within {timeout_s}s -- "
                f"is the server running and watching {queue_dir}?"
            )
        time.sleep(0.005)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transforms",
        default="experiments/layout02_colmap_metric/dataset/transforms.json",
    )
    parser.add_argument("--scene", default="experiments/isaac_sim_export/scene.usda")
    parser.add_argument("--camera-queue-dir", default="isaac_sim_output/splat_queue")
    parser.add_argument("--lidar-queue-dir", default="isaac_sim_output/lidar_queue")
    parser.add_argument("--output-dir", default="isaac_sim_output/lidar_splat_demo")
    parser.add_argument("--gui", action="store_true",
                        help="open the Isaac Sim window (default: headless, frames only written to --output-dir)")
    parser.add_argument("--stride", type=int, default=2,
                        help="use every Nth frame of the real trajectory")
    parser.add_argument("--camera-only", action="store_true",
                        help="skip LiDAR rendering (Phase 1 mode, for fallback)")
    parser.add_argument("--opponent-splat", default=None,
                        help="object splat npz (scripts/extract_object_splat.py) inserted into BOTH the camera "
                             "and the LiDAR map, driving --opponent-lead frames ahead on the same trajectory")
    parser.add_argument("--opponent-usd", default=None, help="the same object as USD, shown in the Isaac viewport")
    parser.add_argument("--opponent-gap", type=float, default=2.5, help="metres ahead along the path")
    parser.add_argument("--opponent-lead", type=int, default=None, help="frames ahead instead of --opponent-gap")
    parser.add_argument("--opponent-lateral", type=float, default=0.0)
    parser.add_argument("--ground-offset", type=float, default=0.15, help="camera height above the floor [m] (without a floor plane)")
    parser.add_argument("--floor-plane", default=None, help="floor_plane.json (default: next to --transforms)")
    parser.add_argument("--no-articulation", action="store_true", help="rigid opponent (no wheel spin, steering, pitch/roll)")
    args = parser.parse_args()

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": not args.gui, "width": 1280, "height": 800})

    try:  # Isaac Sim >= 4.5 API (only option in 6.x)
        from isaacsim.core.api import World
        from isaacsim.core.prims import XFormPrim
        import isaacsim.core.utils.stage as stage_utils
    except ImportError:  # Isaac Sim <= 4.2
        from omni.isaac.core import World
        from omni.isaac.core.prims import XFormPrim
        import omni.isaac.core.utils.stage as stage_utils
    from PIL import Image

    world = World()
    # typeless prim so a native 3DGS scene root (ParticleField3DGaussianSplat) composes
    stage_utils.get_current_stage().DefinePrim("/World/Track").GetReferences().AddReference(os.path.abspath(args.scene))
    car = world.scene.add(XFormPrim(
        prim_path="/World/F1Tenth",
        name="f1tenth_car",
        position=np.array([0.0, 0.0, 0.0]),
    ))
    opponent_prim = None
    if args.opponent_usd:
        from pxr import UsdGeom
        _st = stage_utils.get_current_stage()
        UsdGeom.Xform.Define(_st, "/World/Opponent")
        _st.DefinePrim("/World/Opponent/Splat").GetReferences().AddReference(os.path.abspath(args.opponent_usd))
        opponent_prim = XFormPrim("/World/Opponent")
    world.reset()
    print("[isaac] scene + car loaded")

    frames_all = load_real_trajectory(Path(args.transforms))
    frames = frames_all[:: args.stride]
    print(f"[isaac] using {len(frames)} frames (stride={args.stride})")

    floor_up = trajectory_up(frames_all)    # COLMAP frames are not gravity aligned
    fp = Path(args.floor_plane) if args.floor_plane else Path(args.transforms).parent / "floor_plane.json"
    floor = FloorPlane.load(fp) if fp.exists() else None
    print(f"[isaac] floor plane: {floor if floor else 'none, using camera path minus ground offset'}")
    cum = path_arclength(frames_all)
    if args.opponent_lead is not None:
        args.opponent_gap = None
    P_all = np.array([np.array(f["transform_matrix"])[:3, 3] for f in frames_all])
    path_state = driving_state_from_path(floor.project(P_all) if floor else P_all, np.arange(len(frames_all)) / 15.0)
    articulate = (not args.no_articulation) and args.opponent_splat and ObjectSplat.load(args.opponent_splat).parts is not None

    def opponent_objects(i):
        """Pose of the opponent for ego frame i (same frame as c2w); moves the Isaac prim."""
        if not (args.opponent_splat or args.opponent_usd):
            return None
        j = frame_ahead(i * args.stride, cum, args.opponent_gap, args.opponent_lead)
        T = pose_from_camera_frame(np.array(frames_all[j]["transform_matrix"]),
                                   ground_offset=args.ground_offset, lateral=args.opponent_lateral, up=floor_up, floor=floor)
        if opponent_prim is not None:
            from scipy.spatial.transform import Rotation as _R
            q = _R.from_matrix(T[:3, :3]).as_quat()
            opponent_prim.set_world_poses(positions=np.array([T[:3, 3]]), orientations=np.array([[q[3], q[0], q[1], q[2]]]))
        if not args.opponent_splat:
            return None
        entry = {"path": os.path.abspath(args.opponent_splat), "T_wo": T.tolist()}
        if articulate:
            pitch, roll = attitude_from_accel(float(path_state["a_x"][j]), float(path_state["a_y"][j]))
            entry["articulation"] = {"wheel_angle": float(cum[j]) / 0.057, "steer_angle": float(path_state["steer"][j]), "pitch": pitch, "roll": roll}
        return [entry]

    cam_queue = Path(args.camera_queue_dir)
    lidar_queue = Path(args.lidar_queue_dir)
    output_dir = Path(args.output_dir)
    for d in (cam_queue, lidar_queue, output_dir):
        d.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    for i, frame in enumerate(frames):
        c2w = np.array(frame["transform_matrix"])

        # nerfstudio dataset frame index -- lets lidar_render_server.py --hybrid
        # return the exact real scan captured at this instant (via lidar_scan_lookup.py)
        camera_frame_idx = int(Path(frame["file_path"]).stem.split("_")[-1])

        # Move car in Isaac Sim viewport (visualization only)
        pos_isaac, quat_wxyz = nerfstudio_c2w_to_isaac_pose(c2w)
        car.set_world_pose(position=pos_isaac, orientation=quat_wxyz)
        objects = opponent_objects(i)
        for _ in range(2):
            world.step()

        # ── Camera: gsplat render ─────────────────────────────────────────────
        cam_payload = {"c2w": c2w.tolist()}
        if objects:
            cam_payload["objects"] = objects
        cam_resp = request_render(cam_queue, i, payload=cam_payload, response_suffix="png")
        img = Image.open(cam_resp).convert("RGB")
        img.save(output_dir / f"frame_{i:04d}_camera.jpg", quality=95)
        cam_resp.unlink(missing_ok=True)

        # ── LiDAR: 3DGS cubemap render ────────────────────────────────────────
        if not args.camera_only:
            R_wl, t_wl = vehicle_to_lidar_pose(pos_isaac, quat_wxyz)
            lidar_resp = request_render(
                lidar_queue, i,
                payload={
                    "R_wl": R_wl.tolist(),
                    "t_wl": t_wl.tolist(),
                    "use_raydrop": True,
                    "camera_frame_idx": camera_frame_idx,
                    **({"objects": objects} if objects else {}),
                },
                response_suffix="npz",
            )
            # Copy the NPZ alongside the camera frame
            import shutil
            shutil.copy(lidar_resp, output_dir / f"frame_{i:04d}_lidar.npz")
            lidar_resp.unlink(missing_ok=True)

        if i % 20 == 0:
            elapsed = time.time() - t_start
            print(f"  [{i:4d}/{len(frames)}]  cam+lidar  ({elapsed:.1f}s elapsed)")

    print(f"\n✅ LiDAR-Splat-in-Isaac-Sim demo complete: {len(frames)} frames")
    print(f"   Output dir: {output_dir}")
    print(f"   Camera frames: frame_XXXX_camera.png  (compare with real photos)")
    print(f"   LiDAR scans:   frame_XXXX_lidar.npz   (Nx3 xyz + N intensity)")
    print(f"   To inspect a scan:")
    print(f"     import numpy as np, open3d as o3d")
    print(f"     d = np.load('frame_0000_lidar.npz'); xyz = d['xyz']")
    print(f"     pcd = o3d.geometry.PointCloud()")
    print(f"     pcd.points = o3d.utility.Vector3dVector(xyz)")
    print(f"     o3d.visualization.draw_geometries([pcd])")

    app.close()


if __name__ == "__main__":
    main()
