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
from racing.core import RaceEngine, VehicleState


def load_scene(scene_dir):
    scene_dir = Path(scene_dir)
    scenario = json.loads((scene_dir / 'scenario.json').read_text())
    scenario['usd_path'] = str((scene_dir / scenario['usd']).resolve())
    scenario['reference_path'] = np.asarray(scenario['reference_path'], dtype=float)
    scenario['half_width'] = np.asarray(scenario['half_width'], dtype=float)
    return scenario


def camera_pose(scenario, state, camera_height=None):
    """World pose of the optical (RDF) camera frame. base_link sits `base_z` above the floor;
    `camera_height` (optical centre above floor) overrides it."""
    mount = scenario['sensors']['camera']
    base_z = scenario.get('base_z', 0.) if camera_height is None else camera_height - mount['translation'][2]
    values = asdict(state) if not isinstance(state, dict) else dict(state)
    return sensor_pose({**values, 'z': base_z}, mount)


def scaled_intrinsics(camera, scale):
    return {k: (round(v * scale) if k in ('width', 'height') else v * scale) for k, v in camera.items()}


def make_engine(scenario, model, args):
    return RaceEngine({'ego': model}, scenario['reference_path'], {'ego': VehicleState(x=float(scenario['reference_path'][0, 0]),
                      y=float(scenario['reference_path'][0, 1]))},
                      outer_boundary=scenario['outer_boundary'], inner_boundaries=scenario['inner_boundaries'],
                      dt=1 / scenario['rates']['dynamics'], opponent_speed=args.speed_max,
                      lookahead=args.lookahead, max_speed=max(2.5, args.speed_max))


def sample_episode(scenario, rng, args):
    """Random spawn on the centre line with lateral / heading offset and a target speed."""
    path, half_width = scenario['reference_path'], scenario['half_width']
    i = int(rng.integers(len(path))); j = (i + 3) % len(path)
    tangent = path[j] - path[i]; yaw = math.atan2(tangent[1], tangent[0])
    lateral_limit = max(0., min(args.max_lateral, half_width[i] - args.hose_clearance))
    lateral = rng.uniform(-lateral_limit, lateral_limit)
    left = np.array([-math.sin(yaw), math.cos(yaw)])
    xy = path[i] + lateral * left
    return dict(path_index=i, lateral_m=float(lateral),
                state=VehicleState(x=float(xy[0]), y=float(xy[1]), yaw=float(yaw + rng.normal(0, args.yaw_noise))),
                speed=float(rng.uniform(args.speed_min, args.speed_max)),
                light_scale=float(rng.uniform(.6, 1.3)), light_tint=[float(v) for v in rng.uniform(.92, 1.0, 3)])


def episodes(scenario, model, args):
    """Pure simulation loop. Yields (episode, setup, snapshot, camera_due) for every dynamics tick."""
    engine = make_engine(scenario, model, args)
    schedule = Schedule(scenario['rates']['dynamics'], {'camera': scenario['rates']['camera']})
    for episode in range(args.episodes):
        rng = np.random.default_rng(args.seed * 100003 + episode)
        setup = sample_episode(scenario, rng, args)
        engine.spawns['ego'] = setup['state']; engine.opponent_speed = setup['speed']
        engine.set_control_mode('autopilot'); engine.reset(args.seed); engine.resume(); schedule.reset()
        engine.episode_id = episode      # recording-wide unique ids
        snapshot = engine.snapshot()
        yield episode, setup, snapshot, {'camera'}   # frame at t=0
        for _ in range(int(args.seconds * scenario['rates']['dynamics'])):
            snapshot = engine.step()
            if snapshot.status != 'running' and snapshot.step_index == 0:
                raise RuntimeError(f'episode {episode} could not start: {snapshot.reason}')
            due = schedule.due(snapshot.step_index)
            yield episode, setup, snapshot, due
            if snapshot.status != 'running':
                break


def row(episode, snapshot, frame_pose=None):
    out = {'episode_id': episode, 'step': snapshot.step_index, 'time_s': snapshot.time_s,
           'status': snapshot.status, 'reason': snapshot.reason, 'laps': dict(snapshot.laps),
           'vehicles': {k: asdict(v) for k, v in snapshot.vehicles.items()}}
    if frame_pose is not None:
        out['camera_to_world'] = np.asarray(frame_pose).tolist()
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
            trace.write(json.dumps(row(episode, snapshot, frame_pose)) + '\n')
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
    parser.add_argument('--seconds', type=float, default=25., help='max duration per episode')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--speed-min', type=float, default=1.0); parser.add_argument('--speed-max', type=float, default=2.0)
    parser.add_argument('--lookahead', type=float, default=.9)
    parser.add_argument('--max-lateral', type=float, default=.5, help='max spawn offset from the centre line (m)')
    parser.add_argument('--hose-clearance', type=float, default=.5, help='keep spawns this far from the free edge (m)')
    parser.add_argument('--yaw-noise', type=float, default=.15, help='spawn heading noise (rad, 1 sigma)')
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
