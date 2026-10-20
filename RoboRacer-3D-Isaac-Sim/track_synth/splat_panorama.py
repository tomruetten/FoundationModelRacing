"""360° equirectangular panorama of the real hall, rendered from the camera 3DGS.

Used as the dome light texture of the synthetic track: real hall in the background and
image-based lighting from the real hall on hoses and floor. Run in the `roboracer` env (GPU):

    python -m track_synth.splat_panorama \
        --config data/local/New1/checkpoints/camera_3dgs/config.yml --out reference/iros2026/hall_pano.png

The panorama is taken at the centre of the recorded camera path, `--height` above the floor.
The real track's hoses/cones near the centre stay below the horizon, where the synthetic
floor covers the dome.
"""
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def level_frame(up, heading):
    """World rotation whose columns are (forward, left, up) of a gravity-levelled frame."""
    up = up / np.linalg.norm(up)
    fwd = heading - np.dot(heading, up) * up
    fwd /= np.linalg.norm(fwd)
    return np.column_stack([fwd, np.cross(up, fwd), up])


FACES = {  # face name -> viewing direction in the levelled (forward, left, up) frame
    'front': (1, 0, 0), 'left': (0, 1, 0), 'back': (-1, 0, 0), 'right': (0, -1, 0), 'up': (0, 0, 1), 'down': (0, 0, -1)}


def face_rotation(direction):
    """Levelled-frame rotation of an OpenGL camera (x right, y up, z back) looking along `direction`."""
    d = np.asarray(direction, float)
    up_hint = np.array([1., 0, 0]) if abs(d[2]) > .9 else np.array([0, 0, 1.])
    right = np.cross(d, up_hint); right /= np.linalg.norm(right)
    cam_up = np.cross(right, d)
    return np.column_stack([right, cam_up, -d])


def equirect_from_faces(faces, width):
    """Resample six 90° cube faces (levelled frame) into a latlong image (top = up, centre = forward)."""
    height = width // 2
    lon = (0.5 - (np.arange(width) + .5) / width) * 2 * np.pi        # +lon = to the left
    lat = (0.5 - (np.arange(height) + .5) / height) * np.pi
    lon, lat = np.meshgrid(lon, lat)
    d = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1)
    out = np.zeros((height, width, 3), np.uint8); best = np.full((height, width), -np.inf)
    for name, img in faces.items():
        R = face_rotation(FACES[name])
        c = d @ R                                                    # camera frame (x right, y up, z back)
        depth = -c[..., 2]
        size = img.shape[0]
        with np.errstate(divide='ignore', invalid='ignore'):
            u = (c[..., 0] / depth + 1) * size / 2
            v = (1 - c[..., 1] / depth) * size / 2
        ok = (depth > 0) & (u >= 0) & (u < size) & (v >= 0) & (v < size) & (depth > best)
        out[ok] = img[v[ok].astype(int), u[ok].astype(int)]
        best[ok] = depth[ok]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=ROOT / 'data/local/New1/checkpoints/camera_3dgs/config.yml')
    parser.add_argument('--data-dir', type=Path, help='dataset dir with the transforms.json of this checkpoint')
    parser.add_argument('--out', type=Path, default=ROOT / 'reference/iros2026/hall_pano.png')
    parser.add_argument('--height', type=float, default=1.0, help='panorama height above the floor (m)')
    parser.add_argument('--path-height', type=float, default=.25,
                        help='camera path height above the floor, used when no floor_plane.json exists')
    parser.add_argument('--face', type=int, default=1536, help='cube face size (px)')
    parser.add_argument('--width', type=int, default=4096, help='panorama width (px)')
    args = parser.parse_args()

    os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
    sys.path.insert(0, str(ROOT / 'simulation/isaac_sim'))
    from splat_camera_renderer import SplatCameraRenderer
    from object_splats import trajectory_up, FloorPlane
    from PIL import Image

    renderer = SplatCameraRenderer(args.config, data_dir=args.data_dir)
    frames = json.loads((renderer.data_dir / 'transforms.json').read_text())['frames']
    poses = np.array([f['transform_matrix'] for f in frames], float)
    centre = poses[:, :3, 3].mean(0)
    plane_path = renderer.data_dir / 'floor_plane.json'
    if plane_path.exists():
        floor = FloorPlane.load(plane_path); up = floor.normal
        origin = floor.project(centre) + args.height * up
    else:
        up = trajectory_up(frames)
        origin = centre + (args.height - args.path_height) * up
    heading = -poses[0, :3, 2]
    level = level_frame(up, heading)

    renderer.width = renderer.height = args.face
    renderer.fx = renderer.fy = renderer.cx = renderer.cy = args.face / 2
    faces = {}
    for name, direction in FACES.items():
        c2w = np.eye(4); c2w[:3, :3] = level @ face_rotation(direction); c2w[:3, 3] = origin
        faces[name] = renderer.render(c2w)
        print('rendered', name, flush=True)
    pano = equirect_from_faces(faces, args.width)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pano).save(args.out)
    (args.out.with_suffix('.json')).write_text(json.dumps(dict(
        config=str(args.config), origin=origin.tolist(), up=up.tolist(), height_m=args.height,
        floor_plane=plane_path.exists(), convention='latlong, top=up, image centre=forward of first frame, +lon=left'), indent=2))
    print('wrote', args.out)


if __name__ == '__main__':
    main()
