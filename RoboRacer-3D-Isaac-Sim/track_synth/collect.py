"""Collect ego camera frames + exact poses on a synthetic track scene.

    # Isaac Sim 6 python (env_isaac6); output is a benchmark-compatible recording
    PYTHONPATH=. python -m track_synth.collect --scene outputs/iros2026 \
        --dynamics-checkpoint dynamics/deployed/trained_models/gru_run30_best.pth \
        --episodes 20 --seconds 25 --output outputs/iros2026_data

Output (same layout as `racing --save-frames`, consumed by Code/benchmark):
    frames/ep<E>_<step>.png     640x400 RGB, 20 Hz
    trajectory.jsonl            one row per 50 Hz tick; rows with a camera frame also hold
                                `camera_to_world` (4x4, optical RDF sensor frame in the world)
    dataset.json                scene, arguments, per-episode spawn / speed / light settings
"""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import numpy as np
from racing.config import Schedule, sensor_pose
from racing.core import Command, RaceEngine, VehicleState
from . import vehicle
from .raceline import curvature


def load_scene(scene_dir):
    scene_dir = Path(scene_dir)
    scenario = json.loads((scene_dir / 'scenario.json').read_text())
    scenario['usd_path'] = str((scene_dir / scenario['usd']).resolve())
    for key in ('reference_path', 'half_width', 'raceline', 'raceline_speed', 'raceline_clearance'):
        if key in scenario:
            scenario[key] = np.asarray(scenario[key], dtype=float)
    return scenario


def driving_line(scenario, driver):
    """(path, clearance to the free edge, reference speed or None) of the line the autopilot follows."""
    if driver == 'raceline':
        if 'raceline' not in scenario:
            raise SystemExit('scene has no raceline: rebuild it with track_synth.build (without --no-raceline)')
        return scenario['raceline'], scenario['raceline_clearance'], scenario['raceline_speed']
    return scenario['reference_path'], scenario['half_width'], None


def camera_pose(scenario, state, camera_height=None):
    """World pose of the optical (RDF) camera frame. base_link sits `base_z` above the floor;
    `camera_height` (optical centre above floor) overrides it."""
    mount = scenario['sensors']['camera']
    base_z = scenario.get('base_z', 0.) if camera_height is None else camera_height - mount['translation'][2]
    values = asdict(state) if not isinstance(state, dict) else dict(state)
    return sensor_pose({**values, 'z': base_z}, mount)


def scaled_intrinsics(camera, scale):
    return {k: (round(v * scale) if k in ('width', 'height') else v * scale) for k, v in camera.items()}


class ProfileEngine(RaceEngine):
    """Autopilot = the repo's validated Pure Pursuit (simulation/closed_loop/10_v2_main_demo.py, config D):
    curvature-adaptive lookahead ld = clip(0.5 / (1 + 3 |kappa|), 0.3, 2.0), goal = first path point at
    distance >= ld, delta = atan2(2 L sin(alpha), ld). Speed = per-point reference profile (racing line)
    or a constant, clamped to the GRU training range 0.3..2.5 m/s."""
    def __init__(self, *args, speed=None, lookahead_time=0., yaw_damping=0., steer_filter=0., **kwargs):
        self.profile = None if speed is None else np.asarray(speed, dtype=float)
        self.speed_scale = 1.; self.lookahead_time = lookahead_time
        self.yaw_damping = yaw_damping; self.steer_filter = steer_filter; self._steer_prev = 0.
        super().__init__(*args, **kwargs)
        if self.profile is not None and len(self.profile) != len(self.path):
            raise ValueError('speed profile and path must have the same number of points')
        self._kappa_signed = curvature(self.path, smooth_m=.3)
        self._kappa = np.abs(self._kappa_signed)

    def reset(self, seed=0):
        self._steer_prev = 0.
        return super().reset(seed)

    def reference_speed(self, progress):
        if self.profile is None:
            v = self.opponent_speed
        else:
            v = self.speed_scale * float(np.interp(progress, self._cumulative[:-1], self.profile, period=self.path_length))
        return float(np.clip(v, vehicle.SPEED_MIN, vehicle.SPEED_MAX))

    def _autopilot(self, state):
        progress = self._project(state)
        n = len(self.path)
        closest = min(int(np.searchsorted(self._cumulative, progress, side='right') - 1), n - 1)
        base = vehicle.LOOKAHEAD + self.lookahead_time * max(state.vx, 0.)
        ld = float(np.clip(base / (1. + vehicle.LD_CURVE_K * self._kappa[closest]), vehicle.LD_MIN, vehicle.LD_MAX))
        ahead = self.path[(closest + np.arange(min(n, int(4. / max(self._lengths.mean(), 1e-3))))) % n]
        distance = np.hypot(ahead[:, 0] - state.x, ahead[:, 1] - state.y)
        far = np.nonzero(distance >= ld)[0]
        goal = ahead[far[0] if len(far) else -1]
        alpha = math.atan2(goal[1] - state.y, goal[0] - state.x) - state.yaw
        alpha = (alpha + math.pi) % (2 * math.pi) - math.pi
        steering = math.atan2(2. * vehicle.WHEELBASE * math.sin(alpha), ld)
        if self.yaw_damping:          # damp the yaw rate towards the one the path needs (v * kappa)
            steering -= self.yaw_damping * (state.yr - state.vx * self._kappa_signed[closest])
        if self.steer_filter:         # first-order lag on the steering command
            gain = self.dt / (self.steer_filter + self.dt)
            steering = self._steer_prev + gain * (steering - self._steer_prev)
        steering = float(np.clip(steering, -vehicle.MAX_STEERING, vehicle.MAX_STEERING))
        self._steer_prev = steering
        return self._clamp(Command(self.reference_speed(progress), steering))


def make_engine(scenario, model, args):
    path, _, speed = driving_line(scenario, args.driver)
    return ProfileEngine({'ego': model}, path, {'ego': VehicleState(x=float(path[0, 0]), y=float(path[0, 1]))},
                         outer_boundary=scenario['outer_boundary'], inner_boundaries=scenario['inner_boundaries'],
                         dt=1 / scenario['rates']['dynamics'], opponent_speed=args.speed_max,
                         lookahead=args.lookahead, max_speed=vehicle.SPEED_MAX, max_steering=vehicle.MAX_STEERING,
                         speed=speed, lookahead_time=args.lookahead_time, yaw_damping=args.yaw_damping,
                         steer_filter=args.steer_filter)


def sample_episode(scenario, rng, args):
    """Random spawn on the driving line (small lateral / heading offset), rolling at the line's speed.

    Returns speed = constant target speed (centerline driver) or the speed scale (raceline driver)."""
    path, clearance, profile = driving_line(scenario, args.driver)
    i = int(rng.integers(len(path))); j = (i + 3) % len(path)
    tangent = path[j] - path[i]; yaw = math.atan2(tangent[1], tangent[0])
    lateral_limit = max(0., min(args.max_lateral, clearance[i] - args.hose_clearance))
    lateral = rng.uniform(-lateral_limit, lateral_limit)
    left = np.array([-math.sin(yaw), math.cos(yaw)])
    xy = path[i] + lateral * left
    if profile is None:
        speed = float(rng.uniform(args.speed_min, args.speed_max)); v0 = 0.
    else:
        speed = float(rng.uniform(args.speed_scale_min, args.speed_scale_max)); v0 = speed * float(profile[i])
    return dict(path_index=i, lateral_m=float(lateral), driver=args.driver,
                state=VehicleState(x=float(xy[0]), y=float(xy[1]), yaw=float(yaw + rng.normal(0, args.yaw_noise)), vx=v0),
                speed=speed, light_scale=float(rng.uniform(.6, 1.3)), light_tint=[float(v) for v in rng.uniform(.92, 1.0, 3)])


def episodes(scenario, model, args):
    """Pure simulation loop. Yields (episode, setup, snapshot, camera_due) for every dynamics tick.

    The first `--lead-in` seconds of each episode (controller and dynamics settling after the rolling
    start) are in the trajectory but get no camera frames."""
    engine = make_engine(scenario, model, args)
    schedule = Schedule(scenario['rates']['dynamics'], {'camera': scenario['rates']['camera']})
    for episode in range(args.episodes):
        rng = np.random.default_rng(args.seed * 100003 + episode)
        setup = sample_episode(scenario, rng, args)
        engine.spawns['ego'] = setup['state']
        engine.opponent_speed = setup['speed']; engine.speed_scale = setup['speed'] if engine.profile is not None else 1.
        engine.set_control_mode('autopilot'); engine.reset(args.seed); engine.resume(); schedule.reset()
        engine.episode_id = episode      # recording-wide unique ids
        snapshot = engine.snapshot()
        yield episode, setup, snapshot, ({'camera'} if args.lead_in <= 0 else set())
        for _ in range(int((args.lead_in + args.seconds) * scenario['rates']['dynamics'])):
            snapshot = engine.step()
            if snapshot.status != 'running' and snapshot.step_index == 0:
                raise RuntimeError(f'episode {episode} could not start: {snapshot.reason}')
            due = schedule.due(snapshot.step_index)
            if snapshot.time_s < args.lead_in:
                due = set()
            yield episode, setup, snapshot, due
            if snapshot.status != 'running':
                break


def hose_clearance(scenario, state, tree=None):
    """Distance from the car rectangle (nearest corner) to the hose surface, m; negative = inside."""
    from racing.geometry import rectangle
    if tree is None:
        from scipy.spatial import cKDTree
        tree = cKDTree(np.asarray(scenario['hose_points']))
    return float(tree.query(rectangle(state))[0].min() - scenario['hose']['radius_m'])


def row(episode, snapshot, frame_pose=None, clearance=None):
    out = {'episode_id': episode, 'step': snapshot.step_index, 'time_s': snapshot.time_s,
           'status': snapshot.status, 'reason': snapshot.reason, 'laps': dict(snapshot.laps),
           'vehicles': {k: asdict(v) for k, v in snapshot.vehicles.items()}}
    if frame_pose is not None:
        out['camera_to_world'] = np.asarray(frame_pose).tolist()
    if clearance is not None:
        out['hose_clearance_m'] = round(clearance, 3)
    return out


def select_model(args):
    from racing.dynamics import GRUDynamics, KinematicDynamics
    return KinematicDynamics() if args.kinematic else GRUDynamics(args.dynamics_checkpoint)


def run(args):
    scenario = load_scene(args.scene)
    model = select_model(args)
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'width': 1280, 'height': 800, 'multi_gpu': False,
                         'renderer': args.renderer, 'anti_aliasing': 3,
                         'extra_args': ['--/renderer/multiGpu/enabled=false']})
    output = Path(args.output); (output / 'frames').mkdir(parents=True, exist_ok=True)
    exit_code = 0
    try:
        import carb
        import omni.usd
        import omni.replicator.core as rep
        from pxr import UsdGeom, Gf
        from PIL import Image
        from racing.isaac_app import _camera
        context = omni.usd.get_context(); context.new_stage(); stage = context.get_stage()
        stage.DefinePrim('/World', 'Xform').GetReferences().AddReference(scenario['usd_path'])
        UsdGeom.SetStageUpAxis(stage, 'Z'); UsdGeom.SetStageMetersPerUnit(stage, 1.)
        settings = carb.settings.get_settings()
        # Fixed exposure by default. Auto exposure adapts over time, so identical poses get different
        # brightness depending on how many frames came before (first frame != later frames) and it
        # cancels out --light-scale. Opt in with --auto-exposure.
        settings.set('/rtx/post/histogram/enabled', bool(args.auto_exposure))
        intrinsics = scaled_intrinsics(scenario['camera'], args.image_scale)
        camera, camera_op = _camera(stage, '/World/EgoCamera', intrinsics)
        product = rep.create.render_product(str(camera.GetPath()), (intrinsics['width'], intrinsics['height']))
        rgb = rep.AnnotatorRegistry.get_annotator('rgb'); rgb.attach([product])
        lights = {}
        for prim in stage.Traverse():
            if str(prim.GetPath()).startswith('/World/Lighting/') and prim.HasAttribute('inputs:intensity'):
                lights[prim.GetPath()] = (prim.GetAttribute('inputs:intensity').Get(),
                                          prim.GetAttribute('inputs:color').Get())
        def set_lighting(scale, tint):
            for path, (intensity, color) in lights.items():
                prim = stage.GetPrimAtPath(path)
                factor = args.dome_scale if path.name == 'HallPanorama' else args.fill_scale if path.name.startswith('FillSun') else 1.
                prim.GetAttribute('inputs:intensity').Set(float(intensity * scale * factor))
                prim.GetAttribute('inputs:color').Set(Gf.Vec3f(*(np.array(color) * tint)))
        def place_camera(state):
            pose = camera_pose(scenario, state, args.camera_height)   # optical RDF in world
            usd_pose = pose.copy(); usd_pose[:3, :3] = pose[:3, :3] @ np.diag([1., -1., -1.])  # -> USD camera axes
            camera_op.Set(Gf.Matrix4d(usd_pose.T.tolist()))
            return pose
        def render_frame():
            rep.orchestrator.step(rt_subframes=args.subframes, delta_time=0., pause_timeline=False)
            data = np.asarray(rgb.get_data())
            if data.ndim != 3 or data.shape[:2] != (intrinsics['height'], intrinsics['width']) or data.dtype != np.uint8:
                raise RuntimeError('camera did not produce a valid RGB frame')
            return data[:, :, :3]
        from scipy.spatial import cKDTree
        hose_tree = cKDTree(np.asarray(scenario['hose_points']))
        trace = open(output / 'trajectory.jsonl', 'w'); meta = {}; frames = 0
        current = None
        for episode, setup, snapshot, due in episodes(scenario, model, args):
            if episode != current:
                current = episode; meta[episode] = {k: v for k, v in setup.items() if k != 'state'}
                meta[episode]['spawn'] = asdict(setup['state'])
                set_lighting(setup['light_scale'] * args.light_scale, np.array(setup['light_tint']))
                place_camera(snapshot.vehicles['ego'])
                # let shaders, lights and the camera pose settle BEFORE the first saved frame
                for _ in range(args.warmup if episode == 0 else args.episode_settle):
                    app.update()
                if episode == 0 and args.probe_frames:
                    means = [render_frame().reshape(-1, 3).mean(0) for _ in range(args.probe_frames)]
                    lum = [float(m @ [.2126, .7152, .0722]) for m in means]
                    spread = (max(lum) - min(lum)) / max(np.mean(lum), 1e-6)
                    print('probe: mean luminance of %d renders of the same pose: %s  (spread %.1f %%)%s' % (
                        args.probe_frames, [round(v, 1) for v in lum], 100 * spread,
                        '' if spread < .03 else '  -> brightness drifts between frames (auto exposure / temporal effect)'), flush=True)
            pose = place_camera(snapshot.vehicles['ego'])
            frame_pose = None
            if 'camera' in due:
                Image.fromarray(render_frame()).save(output / 'frames' / f'ep{episode}_{snapshot.step_index:06d}.png')
                frame_pose = pose; frames += 1
            clearance = hose_clearance(scenario, snapshot.vehicles['ego'], hose_tree)
            trace.write(json.dumps(row(episode, snapshot, frame_pose, clearance)) + '\n')
            meta[episode]['min_hose_clearance_m'] = min(meta[episode].get('min_hose_clearance_m', 9.), round(clearance, 3))
            meta[episode]['end'] = dict(step=snapshot.step_index, time_s=snapshot.time_s, status=snapshot.status,
                                        reason=snapshot.reason, laps=dict(snapshot.laps))
        trace.close()
        (output / 'dataset.json').write_text(json.dumps(dict(
            scene=args.scene, frames=frames, args={k: v for k, v in vars(args).items()},
            camera=intrinsics, episodes=meta), indent=2, default=str))
        print(f'saved {frames} frames in {args.episodes} episodes to {output}', flush=True)
    except BaseException:
        import traceback
        exit_code = 1; traceback.print_exc()
    finally:
        app.close(exit_code=exit_code)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scene', required=True, help='directory written by track_synth.build')
    parser.add_argument('--output', default='outputs/synthetic_recording')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--seconds', type=float, default=25., help='recorded duration per episode, after --lead-in')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--driver', choices=('raceline', 'centerline'), default='raceline',
                        help='raceline: lap-time-optimised line + speed profile; centerline: constant speed on the centre line')
    parser.add_argument('--speed-scale-min', type=float, default=.45,
                        help='raceline driver: per-episode speed factor range. With the Run30 GRU in the loop the tracking is '
                             'stable up to ~0.6 (about 1.0-1.5 m/s); from ~0.75 the car snakes and touches the hose')
    parser.add_argument('--speed-scale-max', type=float, default=.6)
    parser.add_argument('--speed-min', type=float, default=1.0, help='centerline driver: constant speed range (m/s)')
    parser.add_argument('--speed-max', type=float, default=2.0, help='(GRU training range: 0.3 .. 2.5 m/s)')
    parser.add_argument('--lead-in', type=float, default=2.0, help='seconds of driving before the first camera frame')
    parser.add_argument('--lookahead', type=float, default=vehicle.LOOKAHEAD, help=argparse.SUPPRESS)
    parser.add_argument('--yaw-damping', type=float, default=0., help='steering damping on the yaw-rate error (rad per rad/s)')
    parser.add_argument('--steer-filter', type=float, default=0., help='steering command low-pass time constant (s)')
    parser.add_argument('--lookahead-time', type=float, default=0.,
                        help='seconds added to the 0.5 m base lookahead per m/s of speed (0 = the repo controller as validated at <= 1 m/s)')
    parser.add_argument('--max-lateral', type=float, default=.15, help='max spawn offset from the driving line (m)')
    parser.add_argument('--hose-clearance', type=float, default=.5, help='keep spawns this far from the free edge (m)')
    parser.add_argument('--yaw-noise', type=float, default=.05, help='spawn heading noise (rad, 1 sigma)')
    parser.add_argument('--dynamics-checkpoint', help='Run30 GRU checkpoint (learned dynamics)')
    parser.add_argument('--kinematic', action='store_true', help='explicit diagnostic bicycle model instead of the GRU')
    parser.add_argument('--light-scale', type=float, default=1., help='global brightness factor applied at run time (no rebuild needed)')
    parser.add_argument('--dome-scale', type=float, default=1., help='brightness of the visible hall panorama only')
    parser.add_argument('--fill-scale', type=float, default=1., help='overhead fill light on hoses and floor (distant lights)')
    parser.add_argument('--auto-exposure', action='store_true',
                        help='temporal auto exposure (brightness then depends on the frames before)')
    parser.add_argument('--probe-frames', type=int, default=4,
                        help='render the first pose this many times and print the brightness spread (0 = off)')
    parser.add_argument('--episode-settle', type=int, default=5, help='render updates after the lighting change of a new episode')
    parser.add_argument('--camera-height', type=float, help='optical centre above the floor (m); default from scene (~0.25)')
    parser.add_argument('--image-scale', type=float, default=1., help='2 renders 1280x800 (same intrinsics, scaled)')
    parser.add_argument('--renderer', default='RealTimePathTracing')
    parser.add_argument('--subframes', type=int, default=3, help='RTX sub-frames per saved image (noise vs. speed)')
    parser.add_argument('--warmup', type=int, default=60)
    return parser


def main():
    args = build_parser().parse_args()
    if bool(args.dynamics_checkpoint) == bool(args.kinematic):
        raise SystemExit('Pass exactly one of --dynamics-checkpoint or --kinematic')
    run(args)


if __name__ == '__main__':
    main()
