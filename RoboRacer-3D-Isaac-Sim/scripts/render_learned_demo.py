#!/usr/bin/env python3
"""Render a five-second learned-camera/LiDAR replay with shared Gaussian objects.

Run in the roboracer environment. Uses New1's 30 Hz source frame numbers to
interpolate its 15 Hz COLMAP poses. No recorded images or scans enter the video.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'simulation/isaac_sim'))
from splat_camera_renderer import SplatCameraRenderer
from lidar_renderer import LidarRenderer
sys.path.insert(0, str(ROOT))
from object_splats import ObjectSplat, trajectory_up, pose_from_camera_frame, FloorPlane, driving_state_from_path, attitude_from_accel, rot_z


def font(size, bold=False):
    name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/' + name, size)


def load_livox_patterns(bag, seconds=6.0, max_range=8.0):
    """Recorded Livox return directions (unit vectors, sensor frame) and their relative
    timestamps from a ROS2 bag: the Mid-360's non-repetitive scan pattern at ~10 Hz. The
    directions come from successful returns of another session, so they carry that session's
    sampling, not New1 geometry."""
    from evaluation.eval_lidar import iter_scans
    scans = []
    for stamp, xyz, _ in iter_scans(Path(bag)):
        ok = np.isfinite(xyz).all(axis=1) & (np.linalg.norm(xyz, axis=1) > 0.1)
        xyz = xyz[ok]
        if not scans and not len(xyz):
            continue
        scans.append((stamp, xyz / np.linalg.norm(xyz, axis=1, keepdims=True)))
        if stamp - scans[0][0] >= seconds:
            break
    t0 = scans[0][0]
    return [(st - t0, d) for st, d in scans]


def ortho_panel(points, labels=None, bev=False, size=(768, 430), title=None):
    """Orthographic LiDAR view in the sensor frame: 3D isometric or bird's eye, 1 m grid,
    8 m range, height colouring (cyan -1 m to gold +3 m); returns on inserted objects are
    drawn in red-orange."""
    w, h = size
    im = Image.new('RGB', size, (12, 22, 34))
    d = ImageDraw.Draw(im)
    scale = min((w - 70) / 16, (h - 85) / 16)
    basis = np.array([[0, -1, 0], [-1, 0, 0.]]) if bev else np.array([[.7071, -.7071, 0], [-.35355, -.35355, -.8660254]])

    def project(p):
        return np.asarray(p, dtype=float).reshape(-1, 3) @ basis.T * scale + [w / 2, (h + 35) / 2]
    for k in range(-8, 9):
        for line in ([[k, -8, 0], [k, 8, 0]], [[-8, k, 0], [8, k, 0]]):
            d.line([tuple(x) for x in project(line)], fill=(30, 44, 57))
    if len(points):
        xy = project(points).astype(int)
        good = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 52) & (xy[:, 1] < h - 24)
        z = np.clip((points[:, 2] + 1) / 4, 0, 1)
        colors = np.column_stack((60 + 195 * z, 210 - 100 * z, 250 - 195 * z)).astype('uint8')
        if labels is not None:
            colors[labels > 0] = (255, 120, 60)
        a = np.array(im)
        a[xy[good, 1], xy[good, 0]] = colors[good]
        im = Image.fromarray(a); d = ImageDraw.Draw(im)
    d.text((20, 12), title or ('BEV / sensor frame' if bev else '3D / sensor frame'), font=font(22, True), fill='white')
    for p, label, color in [([2, 0, 0], '+X', (255, 120, 100)), ([0, 2, 0], '+Y', (90, 230, 130))]:
        origin, end = project([[0, 0, 0], p])
        d.line([tuple(origin), tuple(end)], fill=color, width=2)
        d.text(tuple(end), label, font=font(13), fill=color)
    d.text((20, h - 23), 'Grid 1 m | range <= 8 m | height: cyan -1 m to gold +3 m | orange: inserted objects', font=font(14), fill=(165, 185, 205))
    return im


def cloud_panel(points, labels):
    """Perspective view of actual rendered returns, with inserted returns colored."""
    w, h = 768, 720
    im = Image.new('RGB', (w, h), (12, 22, 34))
    d = ImageDraw.Draw(im)
    # An elevated view following the sensor, looking forward along +X.
    eye = np.array([-3.0, -4.0, 5.0])
    forward = np.array([3.0, 0.0, 0.0]) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.]); right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    def project(p):
        q = p - eye
        z = q @ forward
        return np.column_stack([w / 2 + 760 * (q @ right) / z,
                                h / 2 - 760 * (q @ up) / z]), z

    for i in range(-6, 10):
        for p in (np.array([[i, -6, -.15], [i, 6, -.15]]),
                  np.array([[-2, i, -.15], [9, i, -.15]])):
            uv, z = project(p)
            if (z > .1).all():
                d.line([tuple(v) for v in uv], fill=(29, 43, 58), width=1)
    uv, z = project(points)
    valid = (z > .2) & (points[:, 0] > -2) & (points[:, 0] < 8)
    valid &= (points[:, 2] > -.8) & (points[:, 2] < 2.5)
    valid &= (uv[:, 0] >= 1) & (uv[:, 0] < w-1) & (uv[:, 1] >= 1) & (uv[:, 1] < h-1)
    ids = np.where(valid)[0]
    ids = ids[np.argsort(z[ids])[::-1]]
    # Rasterize nearest depth last; foreground labels come from geometric membership.
    a = np.array(im)
    colors = np.zeros((len(points), 3), np.uint8)
    height = np.clip((points[:, 2] + .2) / 1.8, 0, 1)
    colors[:] = np.column_stack([65 + height*100, 125 + height*65, 180 + height*60]).astype(np.uint8)
    colors[labels == 1] = [80, 230, 190]
    colors[labels == 2] = [255, 159, 70]
    xy = uv[ids].astype(int)
    for dx, dy in ((0, 0), (1, 0), (0, 1)):
        a[xy[:, 1]+dy, xy[:, 0]+dx] = colors[ids]
    im = Image.fromarray(a)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, w, 92), fill=(12, 22, 34))
    d.rectangle((0, 623, w, h), fill=(12, 22, 34))
    d.text((24, 22), 'LEARNED LiDAR', font=font(25, True), fill='white')
    d.text((24, 58), '360-degree model / forward detail', font=font(17), fill=(158, 178, 198))
    d.text((24, 638), 'Car returns', font=font(18), fill=(80, 230, 190))
    d.text((210, 638), 'Cone returns', font=font(18), fill=(255, 159, 70))
    d.text((24, 675), 'Grid: 1 m  |  Colors identify inserted objects', font=font(16), fill=(158,178,198))
    return im


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=ROOT/'outputs/learned_demo_5s_detailed')
    ap.add_argument('--start', type=int, default=560, help='Starting COLMAP pose index')
    ap.add_argument('--preview', action='store_true')
    ap.add_argument('--car', type=Path, default=ROOT/'data/local/objects/f1tenth.npz',
                    help='Gaussian opponent asset; default is the installed F1TENTH')
    ap.add_argument('--opponent-trajectory', type=Path, default=None,
                    help='GRU-driven opponent trajectory (scripts/simulate_opponent_path.py) for the first car; '
                         'default: data/local/New1/dataset/opponent_gru.json if present, else the recorded path 1.7 m ahead')
    ap.add_argument('--no-articulation', action='store_true', help='keep wheels and body rigid')
    ap.add_argument('--blur-samples', type=int, default=3,
                    help='sub-frame samples averaged per camera frame (camera and object motion, wheel spin) over the exposure; 1 = off')
    ap.add_argument('--exposure-ms', type=float, default=8.0, help='exposure time simulated by the sub-frame average (16 ms over-blurs fast turns)')
    ap.add_argument('--object-soften', type=float, default=1.3, help='footprint factor for inserted objects in the camera render')
    ap.add_argument('--livox-bag', type=Path, default=None,
                    help='ROS2 bag with /livox/lidar scans whose return directions and 10 Hz timing sample the LiDAR '
                         '(default: $RAW_SESSION_142014/all_data/all_data_0.db3; missing bag -> dense cubemap sampling)')
    ap.add_argument('--lidar-sampling', choices=['livox', 'cubemap'], default='livox')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    frames_dir = args.out/'frames'; frames_dir.mkdir(exist_ok=True)
    scan_dir = args.out/'scans'; scan_dir.mkdir(exist_ok=True)
    data = ROOT/'data/local/New1'
    meta = json.loads((data/'dataset/transforms.json').read_text())
    frames = meta['frames']
    poses = np.array([f['transform_matrix'] for f in frames])
    times = np.array([int(Path(f['file_path']).stem.split('_')[-1])/30 for f in frames])
    assert np.all(np.diff(times) > 0)
    slerp = Slerp(times, Rotation.from_matrix(poses[:, :3, :3]))
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1))]
    up = trajectory_up(frames)

    def at(t):
        t = np.clip(t, times[0], times[-1])
        p = np.eye(4); p[:3, :3] = slerp(t).as_matrix()
        p[:3, 3] = [np.interp(t, times, poses[:, k, 3]) for k in range(3)]
        return p

    # Objects stand on the reconstructed floor plane (scripts/build_floor_plane.py) when it
    # exists; hanging them 0.15 m below the camera path lets the path's height jitter
    # (3.5 cm frame to frame in New1) push the wheels through the floor.
    floor_file = data / 'dataset/floor_plane.json'
    floor = FloorPlane.load(floor_file) if floor_file.exists() else None
    print('floor plane:', floor if floor else 'none (camera path minus 0.15 m)')

    def ahead(t, gap, lateral=0):
        distance = np.interp(t, times, arc) + gap
        return pose_from_camera_frame(at(np.interp(distance, arc, times)),
                                      ground_offset=.15, lateral=lateral, up=up, floor=floor)

    # Driving state along the recorded path (speed, accelerations, Ackermann steering) for
    # wheel spin, steering and body pitch/roll of path-following opponents.
    P_floor = floor.project(poses[:, :3, 3]) if floor else poses[:, :3, 3]
    path_state = driving_state_from_path(P_floor, times)

    def state_ahead(t, gap):
        distance = np.interp(t, times, arc) + gap
        tt = np.interp(distance, arc, times)
        return {k: float(np.interp(tt, times, v)) for k, v in path_state.items() if k != 'heading'}

    def articulation(state, odometer):
        """Request-style articulation dict from a driving state and distance travelled [m]."""
        pitch, roll = attitude_from_accel(state['a_x'], state['a_y'])
        return {'wheel_angle': odometer / 0.057, 'steer_angle': state['steer'], 'pitch': pitch, 'roll': roll}

    class GRUOpponent:
        """Plays back a trajectory from scripts/simulate_opponent_path.py (world frame)."""
        def __init__(self, path):
            d = json.loads(Path(path).read_text())
            self.rows = d['rows']; self.T = np.array([r['t'] for r in self.rows])
            self.t0 = times[min(int(d.get('start_frame', args.start)), len(times) - 1)]   # ego time of the opponent's t = 0
            self.R_lw = np.array(d['levelled_R_lw']); self.up = self.R_lw[2]
            self.cols = {k: np.array([r[k] for r in self.rows]) for k in ('x', 'y', 'z', 'theta_levelled', 'v', 'a_x', 'a_y', 'steer')}
            self.odo = np.r_[0, np.cumsum(np.abs(self.cols['v'][:-1]) * np.diff(self.T))]
            print(f'GRU opponent trajectory: {len(self.rows)} samples, {self.T[-1]:.1f} s, speed max {self.cols["v"].max():.2f} m/s')

        def pose(self, t):
            t = np.clip(t, self.T[0], self.T[-1])
            p = np.array([np.interp(t, self.T, self.cols[k]) for k in ('x', 'y', 'z')])
            th = np.interp(t, self.T, self.cols['theta_levelled'])
            R = self.R_lw.T @ rot_z(th)              # heading about the floor normal, in world axes
            T = np.eye(4); T[:3, :3] = R; T[:3, 3] = p
            if floor:
                T[:3, 3] = floor.project(p) + floor.clearance * floor.normal
            return T

        def state(self, t):
            t = np.clip(t, self.T[0], self.T[-1])
            return {k: float(np.interp(t, self.T, self.cols[k])) for k in ('v', 'a_x', 'a_y', 'steer')}, float(np.interp(t, self.T, self.odo))

    # maps: the env-selected ones (scripts/env.sh: CAMERA_3DGS_CONFIG / LIDAR_3DGS_CKPT, e.g. the
    # floor-flattened variants from scripts/flatten_floor.py), else the original New1 checkpoints
    camera_ckpt = Path(os.environ.get('CAMERA_3DGS_CONFIG', data/'checkpoints/camera_3dgs/config.yml'))
    lidar_ckpt = Path(os.environ.get('LIDAR_3DGS_CKPT', data/'checkpoints/lidar_3dgs/best.pt'))
    print('camera map:', camera_ckpt, '| lidar map:', lidar_ckpt)
    cam = SplatCameraRenderer(camera_ckpt)
    cam.object_soften = args.object_soften
    lidar = LidarRenderer(lidar_ckpt)
    patterns = None
    if args.lidar_sampling == 'livox':
        bag = args.livox_bag or (Path(os.environ['RAW_SESSION_142014']) / 'all_data/all_data_0.db3' if os.environ.get('RAW_SESSION_142014') else None)
        if bag and Path(bag).exists():
            patterns = load_livox_patterns(bag)
            print(f'Livox sampling: {len(patterns)} recorded scan patterns over {patterns[-1][0]:.1f} s from {bag}')
        else:
            print('no Livox bag found, falling back to dense cubemap sampling')
    objects_dir = ROOT/'data/local/objects'
    car = ObjectSplat.load(args.car)
    # The CAD conversion contains constant material colors and no illumination.
    # Bake a soft overhead key into its surface Gaussians to reveal its geometry.
    if not car.meta.get('appearance_baked', False):
        normals = Rotation.from_quat(car.quats[:, [1, 2, 3, 0]]).as_matrix()[:, :, 2]
        light = np.array([-.35, -.45, 1.]); light /= np.linalg.norm(light)
        diffuse = .32 + .95 * np.maximum(normals @ light, 0)
        c0 = 0.28209479177387814
        rgb_car = np.clip((car.sh[:, 0]*c0+.5) * diffuse[:, None], 0, 1)
        car.sh[:, 0] = (rgb_car-.5)/c0
        car.meta['demo_lighting'] = 'Soft key baked from mesh surface normals in object frame'
    cone = ObjectSplat.load(objects_dir/'cone_photoscan.npz')
    # Exact calibration convention used by this LiDAR checkpoint's training script.
    rcl = Rotation.from_quat([-.47431095, .49897995, -.50794995, .51772095]).as_matrix().T
    tcl = np.eye(4); tcl[:3, :3] = rcl
    tcl[:3, 3] = -rcl @ np.array([.02267, .06987, -.04441])
    tcl = np.diag([1., -1., -1., 1.]) @ tcl
    start = times[args.start]
    assert start + 5 < times[-1]
    traj_file = args.opponent_trajectory or (data / 'dataset/opponent_gru.json')
    gru_opp = GRUOpponent(traj_file) if (traj_file.exists() and not args.no_articulation or args.opponent_trajectory) else None
    if gru_opp is None:
        print('first car follows the recorded path (no GRU trajectory)')
    articulate = not args.no_articulation and car.parts is not None
    print('articulation:', 'wheels/steering/pitch/roll' if articulate else 'off (rigid asset)')
    cones = [(cone, ahead(start, gap, lateral)) for gap, lateral in
             [(1.4, -.5), (2.5, .55), (3.8, -.5), (5.0, .55)]]
    indices = [0, 75, 149] if args.preview else range(150)
    records = []
    scan_cache = {}
    for i in indices:
        t = start + i/30
        pose = at(t)
        def scene_at(tq):
            """Camera pose and object list at time tq (opponent 1: GRU trajectory when available,
            opponent 2: recorded path 3.3 m ahead)."""
            if gru_opp is not None:
                T1 = gru_opp.pose(tq - gru_opp.t0); st1, odo1 = gru_opp.state(tq - gru_opp.t0)
            else:
                T1 = ahead(tq, 1.7); st1 = state_ahead(tq, 1.7); odo1 = np.interp(tq, times, arc) + 1.7
            T2 = ahead(tq, 3.3, -.15); st2 = state_ahead(tq, 3.3); odo2 = np.interp(tq, times, arc) + 3.3
            if articulate:
                cars = [(car, T1, car.articulated(**articulation(st1, odo1))), (car, T2, car.articulated(**articulation(st2, odo2)))]
            else:
                cars = [(car, T1), (car, T2)]
            return at(tq), cars + cones
        pose, objects = scene_at(t)
        if args.blur_samples > 1:
            # motion blur: average sub-frame renders over the exposure (camera, opponents, wheels move)
            acc = np.zeros((cam.height, cam.width, 3), np.float64)
            for dtq in np.linspace(-args.exposure_ms / 2000, args.exposure_ms / 2000, args.blur_samples):
                p_q, o_q = scene_at(t + dtq)
                acc += cam.render(p_q, objects=o_q).astype(np.float64)
            rgb = np.clip(acc / args.blur_samples + 0.5, 0, 255).astype(np.uint8)
        else:
            rgb = cam.render(pose, objects=objects)
        # LiDAR: a new scan only at the recorded Livox scan times (~10 Hz), rendered along the
        # recorded return directions from the pose and object state of that instant
        if patterns is not None:
            j = max(0, int(np.searchsorted([p[0] for p in patterns], i / 30, side='right') - 1))
            t_scan = start + patterns[j][0]
        else:
            j, t_scan = i, t
        if j not in scan_cache:
            pose_s, objects_s = scene_at(t_scan)
            wl = pose_s @ tcl
            if patterns is not None:
                points, intensity = lidar.render_along_rays(wl[:3, :3], wl[:3, 3], patterns[j][1],
                                                            use_raydrop=False, alpha_thresh=.99, objects=objects_s)
            else:
                points, intensity = lidar.render(wl[:3, :3], wl[:3, 3], use_raydrop=False, alpha_thresh=.99, objects=objects_s)
            keep = np.isfinite(points).all(axis=1) & (np.linalg.norm(points, axis=1) <= 8.0)
            points, intensity = points[keep], intensity[keep]
            assert len(points) > 0
            world = points @ wl[:3, :3].T + wl[:3, 3]
            labels = np.zeros(len(points), dtype=np.uint8)
            for k_o, entry in enumerate(objects_s):
                obj, xf = entry[0], entry[1]
                local = (world - xf[:3, 3]) @ xf[:3, :3]
                lo, hi = obj.bbox()
                mask = ((local > lo - .025) & (local < hi + .025)).all(axis=1)
                labels[mask] = 1 if k_o < 2 else 2
            np.savez_compressed(scan_dir / f'{j:04d}.npz', xyz=points, intensity=intensity, labels=labels,
                                camera_to_world=pose_s, lidar_to_world=wl, scan_time_s=t_scan,
                                ray_dirs=patterns[j][1] if patterns is not None else np.zeros((0, 3)),
                                object_to_world=np.array([e[1] for e in objects_s]))
            scan_cache[j] = (points, labels, ortho_panel(points, labels), ortho_panel(points, labels, bev=True))
        points, labels, p3d, pbev = scan_cache[j]
        canvas = Image.new('RGB', (1920, 1080), (9, 16, 26))
        canvas.paste(Image.fromarray(rgb).resize((1152, 720), Image.Resampling.LANCZOS), (0, 150))
        canvas.paste(p3d, (1152, 110)); canvas.paste(pbev, (1152, 560))
        d = ImageDraw.Draw(canvas)
        d.text((28, 20), 'RoboRacer  /  Learned sensor world', font=font(32, True), fill='white')
        d.text((30, 67), 'Camera + LiDAR Gaussian maps  |  Livox-pattern LiDAR (recorded return directions, 10 Hz)  |  shared object placement and occlusion',
               font=font(20), fill=(175, 195, 215))
        d.rounded_rectangle((22, 170, 292, 210), radius=8, fill=(9, 16, 26))
        d.text((36, 175), 'LEARNED CAMERA', font=font(22, True), fill='white')
        d.text((1640, 33), f'{i/30:04.2f} / 5.00 s', font=font(24), fill=(80, 230, 190))
        d.text((28, 905), f'{len(points):,} LiDAR returns  |  scan {j:02d}  |  on objects: {int((labels == 1).sum())} car, {int((labels == 2).sum())} cone',
               font=font(23), fill='white')
        d.text((28, 954), '2 Gaussian cars  +  4 photo-scanned cones', font=font(23, True), fill='white')
        sub = ('Lead car: kinematic driver on the recorded line, gap-keeping' if gru_opp is not None else 'Sensor replay') + \
              ('  |  wheels, steering, pitch/roll animated' if articulate else '') + (f'  |  {args.exposure_ms:.0f} ms global-shutter blur' if args.blur_samples > 1 else '') + '  |  Cones: CC0 scan by Nik Kottmann'
        d.text((28, 995), sub, font=font(17), fill=(153, 174, 194))
        d.text((28, 1030), 'LiDAR directions from session 142014 returns (sampling only, not a full firing pattern); geometry and objects from the New1 maps.',
               font=font(15), fill=(165, 185, 205))
        d.rectangle((0, 1076, round(1920 * (i + 1) / 150), 1079), fill=(80, 230, 190))
        canvas.save(frames_dir / f'{i:04d}.png')
        records.append({'frame': i, 'time_s': i / 30, 'source_time_s': t, 'scan': int(j), 'returns': len(points),
                        'car_returns': int((labels == 1).sum()), 'cone_returns': int((labels == 2).sum())})
        if i % 15 == 0 or args.preview:
            print(records[-1], flush=True)
    (args.out/('preview_metrics.json' if args.preview else 'render_metrics.json')).write_text(json.dumps(records,indent=2))
    if not args.preview:
        subprocess.run(['ffmpeg','-y','-v','error','-framerate','30','-i',str(frames_dir/'%04d.png'),
                        '-frames:v','150','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',
                        '-movflags','+faststart',str(args.out/'roboracer_camera_lidar_5s.mp4')],check=True)
        print('Saved', args.out/'roboracer_camera_lidar_5s.mp4', flush=True)


if __name__ == '__main__':
    main()
