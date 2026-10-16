#!/usr/bin/env python3
# isaac_sim_camera_splat_demo.py
"""
"Kamera Splat in Isaac Sim" -- Phase 1 (Felix, 2026-06-30: "Implementiere
bitte schon einmal das Kamera Splat in Issac Sim damit wir langsam mit
unserer Pipeline anfangen.")

What this does, and why it's different from the old task4/task5 scripts in
lab_preparation_guide.md:
  - OLD (task4/task5): Isaac Sim's own native `Camera` sensor renders Isaac's
    PBR shading of the imported mesh (experiments/isaac_sim_export/scene.usda)
    -- NOT the trained Camera-3DGS model. Also drove a fake 50-point circular
    demo trajectory instead of the real recorded track.
  - NEW (this script): drives the car through the REAL recorded COLMAP
    trajectory (1066 frames, experiments/layout02_colmap_metric/dataset/
    transforms.json -- the pure-COLMAP, 22.29 dB PSNR checkpoint, per Felix's
    "use COLMAP not MoCap" directive) and, at each frame, the "camera sensor
    output" is the actual Gaussian-Splatting render from that pose -- produced
    by a separate process (splat_render_server.py) to avoid mixing Isaac
    Sim's bundled torch/CUDA with nerfstudio/gsplat's in one process.

LiDAR is intentionally left on Isaac's native sensor for this phase --
Felix: "Sobald es funktioniert ergänzen wir dann LiDAR" (add LiDAR once
camera works). Do not try to add the real LiDAR-3DGS network here yet.

BEFORE RUNNING THIS: start splat_render_server.py in a separate terminal,
in the nerfstudio/gsplat-capable conda env:
    conda run -n nerfstudio-py310 python3 splat_render_server.py \
        --queue-dir isaac_sim_output/splat_queue
(on the lab machine, substitute whatever conda env has nerfstudio+gsplat --
check TECHNICAL_NOTES.md, likely named "probe")

Then run THIS script with Isaac Sim's own Python:
    python3 isaac_sim_camera_splat_demo.py --stride 2
(--stride 2 renders every 2nd frame of the 1066-frame trajectory, ~533
frames, to keep runtime reasonable for a first test; use --stride 1 for the
full trajectory once the pipeline is confirmed working)
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
from object_splats import pose_from_camera_frame, similarity_from_frames, trajectory_up, path_arclength, frame_ahead, FloorPlane, ObjectSplat, driving_state_from_path, attitude_from_accel, rot_z  # noqa: E402


def load_real_trajectory(transforms_path: Path):
    """Load the real, recorded COLMAP camera trajectory -- replaces the fake
    50-point circular demo trajectory in the old task5 script."""
    data = json.load(open(transforms_path))
    frames = data["frames"]
    print(f"[trajectory] loaded {len(frames)} real frames from {transforms_path}")
    return frames


def nerfstudio_c2w_to_isaac_pose(c2w: np.ndarray):
    """
    Convert a nerfstudio camera-to-world matrix to Isaac Sim (Z-up) position
    + quaternion for the vehicle body.

    Key insight confirmed by data analysis (2026-07-01):
      The COLMAP-metric world frame is Z-up (after Umeyama alignment to MoCap).
      Camera positions: X in [-3.2, 3.5]m, Y in [-0.8, 4.3]m, Z in [0.15, 0.21]m.
      Z ~= 0.18m = camera mount height above ground -- Z IS the height axis.
      Isaac Sim (robotics mode) is also Z-up → position mapping is identity.

    Previous code swapped Y↔Z (wrong: it was designed for a synthetic Y-up
    nerfstudio scene, but our COLMAP frame is Z-up). That placed the car 5m
    above the ground inside Isaac Sim.

    Camera convention (nerfstudio/OpenGL): X-right, Y-up, Z-back.
    Vehicle convention (Isaac/ROS, Z-up): X-forward, Y-left, Z-up.
    Conversion: vehicle_X = -cam_Z, vehicle_Y = -cam_X, vehicle_Z = cam_Y.
    """
    from scipy.spatial.transform import Rotation as R

    # Position: COLMAP world == Isaac world (both Z-up after Umeyama to MoCap)
    pos_isaac = c2w[:3, 3].copy()

    # Rotation: convert camera axes to vehicle body axes
    # camera cols are [cam_X_in_world, cam_Y_in_world, cam_Z_in_world]
    # vehicle forward = -cam_Z, vehicle left = -cam_X, vehicle up = cam_Y
    rot = c2w[:3, :3]
    R_vehicle_to_world = np.column_stack([-rot[:, 2], -rot[:, 0], rot[:, 1]])
    quat_xyzw = R.from_matrix(R_vehicle_to_world).as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    return pos_isaac, quat_wxyz


def request_splat_render(queue_dir: Path, idx: int, c2w: np.ndarray, timeout_s: float = 30.0, objects=None):
    """Write a pose request, block until splat_render_server.py produces the
    corresponding PNG, return its path. objects: list of {"path"|"name", "T_wo"} inserted
    into the scene for this frame (object splats, see object_splats.py)."""
    req_path = queue_dir / f"request_{idx:06d}.json"
    payload = {"c2w": c2w.tolist()}
    if objects:
        payload["objects"] = objects
    json.dump(payload, open(req_path, "w"))

    t0 = time.time()
    while True:   # the server picks the image format (bmp default, png, jpg)
        hits = [p for p in queue_dir.glob(f"response_{idx:06d}.*") if ".tmp" not in p.name]
        if hits:
            return hits[0]
        if time.time() - t0 > timeout_s:
            raise TimeoutError(
                f"splat_render_server did not respond to {req_path.name} within "
                f"{timeout_s}s -- is splat_render_server.py running and watching "
                f"{queue_dir}?"
            )
        time.sleep(0.005)


def quat_wxyz_from_matrix(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as _R
    q = _R.from_matrix(R).as_quat()   # x,y,z,w
    return np.array([q[3], q[0], q[1], q[2]])


class Opponent:
    """A second vehicle (or any object splat) that drives `lead` frames ahead of the ego
    car on the recorded trajectory. Visible twice: in Isaac Sim as its own gaussian-splat
    prim under /World/Opponent (--opponent-usd) and in the pseudo camera images from the
    render server, which inserts the same gaussians into the camera map (--opponent-splat).
    """

    def __init__(self, args, frames, stage, XFormPrim):
        self.lead, self.gap, self.lateral, self.ground = args.opponent_lead, args.opponent_gap, args.opponent_lateral, args.ground_offset
        self.frames = frames
        self.cum = path_arclength(frames)
        self.times = np.arange(len(frames)) / 15.0        # recorded at 15 Hz (only ratios matter here)
        self.articulate = not args.no_articulation
        self.obj = ObjectSplat.load(args.opponent_splat) if args.opponent_splat else None
        if self.obj is not None and self.obj.parts is None:
            self.articulate = False
        self.traj = None
        if args.opponent_trajectory:
            d = json.load(open(args.opponent_trajectory))
            self.traj = {"T": np.array([r["t"] for r in d["rows"]]), "R_lw": np.array(d["levelled_R_lw"]),
                         "start_frame": int(d.get("start_frame", 0)), "fps": float(d.get("frame_rate", 15.0)),
                         **{k: np.array([r[k] for r in d["rows"]]) for k in ("x", "y", "z", "theta_levelled", "v", "a_x", "a_y", "steer")}}
            self.traj["odo"] = np.r_[0, np.cumsum(np.abs(self.traj["v"][:-1]) * np.diff(self.traj["T"]))]
            print(f"[isaac] opponent follows the GRU trajectory {args.opponent_trajectory} ({self.traj['T'][-1]:.0f} s)")
        self.up = trajectory_up(frames)      # COLMAP frames are not gravity aligned (New1: 9.5 deg)
        self.splat = os.path.abspath(args.opponent_splat) if args.opponent_splat else None
        # Isaac frame -> render-server frame (3DGRUT trajectories carry both poses)
        S = similarity_from_frames(frames)
        self.isaac_to_server = np.linalg.inv(S) if S is not None else np.eye(4)
        # floor plane of the scene (scripts/build_floor_plane.py), stored in the server frame
        self.floor = None
        fp = Path(args.floor_plane) if args.floor_plane else Path(args.transforms).parent / "floor_plane.json"
        if fp.exists():
            self.floor = FloorPlane.load(fp)
            if S is not None:
                self.floor = self.floor.transformed(S)
            print(f"[isaac] opponent stands on the reconstructed floor: {self.floor} from {fp}")
        else:
            print(f"[isaac] no floor plane ({fp}); opponent hangs {self.ground:.2f} m below the camera path")
        self.prim = None
        if args.opponent_usd:
            from pxr import UsdGeom
            UsdGeom.Xform.Define(stage, "/World/Opponent")
            stage.DefinePrim("/World/Opponent/Splat").GetReferences().AddReference(os.path.abspath(args.opponent_usd))
            self.prim = XFormPrim("/World/Opponent")
            print(f"[isaac] opponent splat {args.opponent_usd} under /World/Opponent")
        where = f"{self.gap:.1f} m ahead along the path" if self.gap is not None else f"{self.lead} frames ahead"
        print(f"[isaac] opponent drives {where}, lateral {self.lateral:+.2f} m, "
              f"ground {self.ground:.2f} m below the camera, floor normal {np.round(self.up, 3).tolist()}")

    def pose_isaac(self, i: int):
        """Pose (4x4) plus driving state and odometer for ego frame i."""
        if self.traj is not None:
            t = np.clip((i - self.traj["start_frame"]) / self.traj["fps"], self.traj["T"][0], self.traj["T"][-1])
            p = np.array([np.interp(t, self.traj["T"], self.traj[k]) for k in ("x", "y", "z")])
            th = float(np.interp(t, self.traj["T"], self.traj["theta_levelled"]))
            T = np.eye(4); T[:3, :3] = self.traj["R_lw"].T @ rot_z(th); T[:3, 3] = p
            if self.floor is not None:
                T[:3, 3] = self.floor.project(p) + self.floor.clearance * self.floor.normal
            st = {k: float(np.interp(t, self.traj["T"], self.traj[k])) for k in ("v", "a_x", "a_y", "steer")}
            return T, st, float(np.interp(t, self.traj["T"], self.traj["odo"]))
        j = frame_ahead(i, self.cum, self.gap, self.lead)
        c2w = np.array(self.frames[j]["transform_matrix"])
        T = pose_from_camera_frame(c2w, ground_offset=self.ground, lateral=self.lateral, up=self.up, floor=self.floor)
        if not hasattr(self, "_path_state"):
            P = np.array([np.array(f["transform_matrix"])[:3, 3] for f in self.frames])
            self._path_state = driving_state_from_path(self.floor.project(P) if self.floor else P, self.times)
        st = {k: float(v[j]) for k, v in self._path_state.items() if k != "heading"}
        return T, st, float(self.cum[j])

    def update(self, i: int):
        """Move the Isaac prim; return the request "objects" list for the render server."""
        T, st, odo = self.pose_isaac(i)
        if self.prim is not None:
            self.prim.set_world_poses(positions=np.array([T[:3, 3]]),
                                      orientations=np.array([quat_wxyz_from_matrix(T[:3, :3])]))
        if self.splat is None:
            return None
        entry = {"path": self.splat, "T_wo": (self.isaac_to_server @ T).tolist()}
        if self.articulate:
            pitch, roll = attitude_from_accel(st["a_x"], st["a_y"])
            entry["articulation"] = {"wheel_angle": odo / 0.057, "steer_angle": st["steer"], "pitch": pitch, "roll": roll}
        return [entry]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transforms",
        default="experiments/layout02_near_triage_pure_colmap_cropped/dataset/transforms.json",
    )
    parser.add_argument("--opponent-usd", default=None,
                        help="object splat USD (scripts/splat_to_usd_gsplat.py --object-npz) shown in Isaac Sim "
                             "as a second vehicle driving ahead on the same trajectory")
    parser.add_argument("--opponent-splat", default=None,
                        help="object splat npz (scripts/extract_object_splat.py) inserted into the camera map "
                             "by the render server at the same pose")
    parser.add_argument("--opponent-gap", type=float, default=2.5,
                        help="opponent drives this many metres ahead along the recorded path (speed independent)")
    parser.add_argument("--opponent-lead", type=int, default=None,
                        help="alternative to --opponent-gap: fixed number of trajectory frames ahead (before --stride)")
    parser.add_argument("--opponent-lateral", type=float, default=0.0, help="offset to the left of the path [m]")
    parser.add_argument("--ground-offset", type=float, default=0.15,
                        help="camera height above the floor [m] (New1 lap: 0.15 +- 0.07 measured); "
                             "the opponent's origin sits on the floor")
    parser.add_argument("--floor-plane", default=None,
                        help="floor_plane.json of the scene (scripts/build_floor_plane.py); default: next to --transforms")
    parser.add_argument("--opponent-trajectory", default=None,
                        help="GRU-driven opponent trajectory json (scripts/simulate_opponent_path.py) instead of the recorded path")
    parser.add_argument("--no-articulation", action="store_true", help="rigid opponent (no wheel spin, steering, pitch/roll)")
    parser.add_argument("--start-frame", type=int, default=0, help="first trajectory frame (before --stride)")
    parser.add_argument("--max-frames", type=int, default=None, help="stop after this many frames (checks)")
    parser.add_argument("--frame-format", choices=["jpg", "png", "bmp"], default="jpg",
                        help="format of the saved pseudo-camera frames (png costs ~150 ms per frame)")
    parser.add_argument("--scene", default="experiments/isaac_sim_export/scene.usda")
    parser.add_argument("--queue-dir", default="isaac_sim_output/splat_queue")
    parser.add_argument("--output-dir", default="isaac_sim_output/camera_splat_demo")
    parser.add_argument("--screenshot", default=None,
                        help="save a viewport capture (PNG) after the first frame, also works headless")
    parser.add_argument("--nurec", action="store_true",
                        help="scene is a 3DGRUT/NuRec export (PPISP post-processing): enable omni.rtx.spg and "
                             "apply Isaac's NuRec render overrides before the first frame")
    parser.add_argument("--view", choices=["onboard", "chase", "overview"], default="onboard",
                        help="viewport camera: onboard = the F1TENTH camera pose itself (default), "
                             "chase = low third-person behind the car, overview = static top view")
    parser.add_argument("--chase", action="store_true", help="alias for --view chase")
    parser.add_argument("--gui", action="store_true",
                        help="open the Isaac Sim window (default: headless, frames only written to --output-dir)")
    parser.add_argument("--stride", type=int, default=2,
                         help="use every Nth frame of the real trajectory")
    args = parser.parse_args()
    if args.chase:
        args.view = "chase"
    if args.opponent_lead is not None:
        args.opponent_gap = None

    from isaacsim import SimulationApp
    # Gaussian-splat render caps are read once at the first Hydra sync, so they must
    # be launch args (runtime carb.settings changes are ignored). Defaults 48/500 are
    # tuned for speed; higher values converge a bit cleaner on the native splat scene.
    extra_args = ["--/rtx/rtpt/gaussian/maxGaussiansToAccumulate=512",
                  "--/rtx/raytracing/gaussian/maxIntersections=4000",
                  "--/renderer/multiGpu/enabled=false"]
    if args.nurec:
        # SPG/PPISP render graph for 3DGRUT exports + Isaac's NuRec helper extension
        extra_args += ["--enable", "omni.rtx.spg", "--enable", "isaacsim.replicator.nurec_utils"]
    app = SimulationApp({"headless": not args.gui, "width": 1280, "height": 800,
                         "extra_args": extra_args})

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

    scene_path = os.path.abspath(args.scene)
    # Reference onto a TYPELESS prim so the scene file's own prim type wins.
    # add_reference_to_stage() authors an Xform, which silently overrides a
    # ParticleField3DGaussianSplat root (native 3DGS scene) -> nothing renders.
    _stage = stage_utils.get_current_stage()
    _stage.DefinePrim("/World/Track").GetReferences().AddReference(scene_path)
    if args.nurec:
        # NuRec engine reads these carb settings once at the first Hydra sync.
        from isaacsim.replicator.nurec_utils.rendering_setup import setup_for_rendering
        ok, is_nurec, spg, problems = setup_for_rendering(_stage)
        print(f"[isaac] nurec setup: ok={ok} nurec={is_nurec} spg={spg} problems={problems}")
        if not ok:
            raise SystemExit("NuRec prerequisites unmet: " + "; ".join(problems))

    # Visible car marker (Car02 footprint 0.42 x 0.26 x 0.10 m, blue) instead of
    # an empty XForm, plus a light: scene.usda is a bare mesh, so without these
    # the RTX viewport stays black.
    try:
        from isaacsim.core.api.objects import VisualCuboid
        from isaacsim.core.utils.prims import create_prim
        from isaacsim.core.utils.viewports import set_camera_view
    except ImportError:  # Isaac Sim < 4.5
        from omni.isaac.core.objects import VisualCuboid
        from omni.isaac.core.utils.prims import create_prim
        from omni.isaac.core.utils.viewports import set_camera_view
    create_prim("/World/DomeLight", "DomeLight", attributes={"inputs:intensity": 400.0})
    car = world.scene.add(VisualCuboid(
        prim_path="/World/F1Tenth",
        name="f1tenth_car",
        position=np.array([0.0, 0.0, 0.0]),
        scale=np.array([0.42, 0.26, 0.10]),
        color=np.array([0.1, 0.4, 1.0]),
    ))
    world.reset()
    print("[isaac] scene + car loaded")

    frames_all = load_real_trajectory(Path(args.transforms))
    frames = frames_all[args.start_frame :: args.stride]
    if args.max_frames:
        frames = frames[: args.max_frames]
    print(f"[isaac] using {len(frames)} frames (stride={args.stride})")
    opponent = None
    if args.opponent_usd or args.opponent_splat:
        opponent = Opponent(args, frames_all, _stage, XFormPrim)

    # Onboard view: a dedicated USD camera with the real OAK-D intrinsics (from
    # transforms.json), posed every frame from c2w and made the active viewport
    # camera. Kit's built-in /OmniverseKit_Persp keeps its own manipulator state
    # and cannot be posed reliably from a script.
    meta = json.load(open(args.transforms))
    cam_xf = None
    if args.view == "onboard":
        from pxr import UsdGeom
        from omni.kit.viewport.utility import get_active_viewport
        cam_path = "/World/OnboardCam"
        cam = UsdGeom.Camera.Define(stage_utils.get_current_stage(), cam_path)
        w_px, h_px = float(meta["w"]), float(meta["h"])
        h_ap = 20.955  # mm, Kit default horizontal aperture
        cam.GetFocalLengthAttr().Set(float(meta["fl_x"]) * h_ap / w_px)
        cam.GetHorizontalApertureAttr().Set(h_ap)
        v_ap = h_ap * h_px / w_px * float(meta["fl_x"]) / float(meta["fl_y"])
        cam.GetVerticalApertureAttr().Set(v_ap)
        # principal point: film origin must land on pixel (cx, cy); film +Y is up, image +y is down
        cx, cy = float(meta.get("cx", w_px / 2)), float(meta.get("cy", h_px / 2))
        cam.GetHorizontalApertureOffsetAttr().Set((cx - w_px / 2) * h_ap / w_px)
        cam.GetVerticalApertureOffsetAttr().Set((cy - h_px / 2) * v_ap / h_px)
        cam.GetClippingRangeAttr().Set((0.05, 1000.0))
        cam_xf = XFormPrim(cam_path)
        get_active_viewport().camera_path = cam_path
        car.set_visibility(False)   # eye sits exactly at the camera pose -> keep the marker out of the lens
        print(f"[isaac] onboard camera {cam_path}: fl_x={meta['fl_x']:.1f}px {int(w_px)}x{int(h_px)} cx={cx:.1f} cy={cy:.1f}")

    # Overview camera: look at the trajectory centre from above/behind.
    traj_xyz = np.array([np.array(f["transform_matrix"])[:3, 3] for f in frames])
    centre = traj_xyz.mean(axis=0)
    span = float(np.max(traj_xyz.max(axis=0) - traj_xyz.min(axis=0)))
    if (args.gui or args.screenshot) and args.view == "overview":
        set_camera_view(eye=centre + np.array([-0.8 * span, -0.8 * span, 0.9 * span]),
                        target=centre)

    queue_dir = Path(args.queue_dir)
    queue_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    for i, frame in enumerate(frames):
        c2w = np.array(frame["transform_matrix"])

        pos_isaac, quat_wxyz = nerfstudio_c2w_to_isaac_pose(c2w)
        car.set_world_pose(position=pos_isaac, orientation=quat_wxyz)
        req_objects = opponent.update(args.start_frame + i * args.stride) if opponent else None
        if (args.gui or args.screenshot) and args.view != "overview":
            fwd = -c2w[:3, 2]                      # camera looks along -Z (OpenGL)
            fwd_xy = fwd.copy(); fwd_xy[2] = 0.0
            fwd_xy /= (np.linalg.norm(fwd_xy) + 1e-9)
            if args.view == "onboard":
                # exact OAK-D pose incl. roll/pitch: USD cameras look down -Z with +Y up,
                # the same convention as the nerfstudio c2w -> rotation carries over 1:1.
                from scipy.spatial.transform import Rotation as _R
                q = _R.from_matrix(c2w[:3, :3]).as_quat()            # x,y,z,w
                cam_xf.set_world_poses(positions=np.array([c2w[:3, 3]]),
                                       orientations=np.array([[q[3], q[0], q[1], q[2]]]))
            else:                                  # low chase cam just behind/above the car
                eye = pos_isaac - 1.0 * fwd_xy + np.array([0, 0, 0.35])
                tgt = pos_isaac + 2.0 * fwd_xy
                set_camera_view(eye=eye, target=tgt)
        for _ in range(2):
            world.step()

        # The render server wants the pose in the checkpoint's own dataset frame. A
        # trajectory converted by scripts/wrap_3dgrut_usd.py keeps that pose in
        # transform_matrix_source; plain transforms.json files are already in it.
        c2w_server = np.array(frame.get("transform_matrix_source", frame["transform_matrix"]))
        resp_path = request_splat_render(queue_dir, i, c2w_server, objects=req_objects)
        img = Image.open(resp_path).convert("RGB")
        img.save(output_dir / f"camera_splat_{i:04d}.{args.frame_format}", **({"quality": 95} if args.frame_format == "jpg" else {}))
        resp_path.unlink(missing_ok=True)

        if i == 0 and args.screenshot:
            from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
            for _ in range(60):   # let RTX converge / camera move settle
                app.update()
            capture_viewport_to_file(get_active_viewport(), args.screenshot)
            for _ in range(60):   # capture is asynchronous
                app.update()
            print(f"[isaac] viewport screenshot -> {args.screenshot}")

        if i % 20 == 0:
            elapsed = time.time() - t_start
            print(f"  [{i:4d}/{len(frames)}] {frame['file_path']} "
                  f"({elapsed:.1f}s elapsed)")

    print(f"\n✅ Camera-Splat-in-Isaac-Sim demo complete: {len(frames)} frames")
    print(f"   Output: {output_dir}")
    print(f"   Next: visually compare against the original real photos "
          f"(experiments/layout02_colmap_metric/dataset/images/) at the same "
          f"frame indices, and against the offline nerfstudio render (task 7) "
          f"to confirm the bridge reproduces identical output inside Isaac Sim.")

    # signal splat_render_server.py to stop, if desired
    # (queue_dir / "STOP").touch()

    app.close()


if __name__ == "__main__":
    main()
