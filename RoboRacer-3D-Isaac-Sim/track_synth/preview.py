"""Dependency-light software preview: rasterise the USD hoses/floor from the ego camera.

Only a geometry sanity check (proportions, camera axes, clearances); not a photoreal render.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def load_triangles(usd_path, prims=('/World/Track',)):
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(str(usd_path)); tris = []
    for prim in stage.Traverse():
        if prim.GetTypeName() != 'Mesh' or not any(str(prim.GetPath()).startswith(p) for p in prims):
            continue
        mesh = UsdGeom.Mesh(prim); pts = np.array(mesh.GetPointsAttr().Get())
        counts = np.array(mesh.GetFaceVertexCountsAttr().Get()); idx = np.array(mesh.GetFaceVertexIndicesAttr().Get())
        start = np.r_[0, np.cumsum(counts)[:-1]]
        for c in (3, 4):
            sel = start[counts == c]
            if not len(sel):
                continue
            faces = idx[sel[:, None] + np.arange(c)]
            tris.append(pts[faces[:, [0, 1, 2]]])
            if c == 4:
                tris.append(pts[faces[:, [0, 2, 3]]])
    return np.concatenate(tris)


def render(tris, scenario, pose_xy_yaw, size=None):
    cam = scenario['camera']; w, h = size or (cam['width'], cam['height'])
    from .collect import camera_pose
    x, y, yaw = pose_xy_yaw
    c2w = camera_pose(scenario, dict(x=x, y=y, yaw=yaw))
    # 1 m checker floor tiles around the car (the scene floor is one huge quad, unusable here)
    gx, gy = np.meshgrid(np.arange(-30, 30), np.arange(-30, 30))
    corners = np.stack([np.c_[gx.ravel() + a, gy.ravel() + b] for a, b in ((0, 0), (1, 0), (1, 1))], 1)
    corners2 = np.stack([np.c_[gx.ravel() + a, gy.ravel() + b] for a, b in ((0, 0), (1, 1), (0, 1))], 1)
    floor = np.concatenate([corners, corners2]) + np.round([x, y])
    floor = np.concatenate([floor, np.zeros(floor.shape[:2] + (1,))], -1)
    checker = np.tile(((gx + gy).ravel() % 2).astype(bool), 2)
    tris = np.concatenate([floor, tris]); is_floor = np.r_[np.ones(len(floor), bool), np.zeros(len(tris) - len(floor), bool)]
    checker = np.r_[checker, np.zeros(len(tris) - len(floor), bool)]
    w2c = np.linalg.inv(c2w)
    p = tris @ w2c[:3, :3].T + w2c[:3, 3]                 # optical frame: x right, y down, z forward
    depth = p[..., 2]; keep = (depth > .05).all(1) & (depth < 40).all(1)
    tris, p, is_floor, checker = tris[keep], p[keep], is_floor[keep], checker[keep]
    uv = np.stack([cam['fx'] * p[..., 0] / p[..., 2] + cam['cx'], cam['fy'] * p[..., 1] / p[..., 2] + cam['cy']], -1)
    normal = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-12
    shade = .25 + .75 * np.abs(normal @ np.array([.3, .2, .93]))
    order = np.argsort(-p[..., 2].mean(1))
    img = Image.new('RGB', (w, h), (90, 100, 115)); draw = ImageDraw.Draw(img)
    for i in order:
        col = (np.array((170, 168, 160) if checker[i] else (150, 148, 140)) if is_floor[i]
               else np.array((225, 225, 220)) * shade[i])
        draw.polygon([tuple(q) for q in uv[i]], fill=tuple(col.astype(int)))
    return img


def plot_raceline(scenario, out):
    """Top view: map boundaries, centre line, racing line coloured by reference speed."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14, 8.5))
    for key in ('outer_boundary',):
        poly = np.array(scenario[key]); ax.plot(*np.vstack([poly, poly[:1]]).T, color='gray', lw=1)
    for poly in scenario['inner_boundaries']:
        poly = np.array(poly); ax.plot(*np.vstack([poly, poly[:1]]).T, color='gray', lw=1)
    center = np.array(scenario['reference_path']); ax.plot(*np.vstack([center, center[:1]]).T, 'k--', lw=.5)
    line, speed = np.array(scenario['raceline']), np.array(scenario['raceline_speed'])
    sc = ax.scatter(line[:, 0], line[:, 1], c=speed, s=6, cmap='viridis'); fig.colorbar(sc, label='reference speed (m/s)')
    info = scenario['raceline_info']
    ax.set_title(f"racing line: lap {info['lap_time_s']:.1f} s (centre line {info['centerline_lap_time_s']:.1f} s), "
                 f"v_max {info['limits']['v_max']} m/s, a_lat {info['limits']['a_lat']} m/s^2")
    ax.set_aspect('equal'); fig.savefig(out, dpi=70)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scene_dir'); parser.add_argument('--fractions', type=float, nargs='*', default=[0, .25, .5, .75])
    args = parser.parse_args(); d = Path(args.scene_dir)
    scenario = json.loads((d / 'scenario.json').read_text())
    path = np.array(scenario['reference_path']); tris = load_triangles(d / scenario['usd'])
    for frac in args.fractions:
        i = int(frac * len(path)) % len(path); j = (i + 3) % len(path)
        yaw = np.arctan2(*(path[j] - path[i])[::-1])
        render(tris, scenario, (*path[i], yaw)).save(d / f'preview_{int(frac * 100):02d}.png')
    if 'raceline' in scenario:
        plot_raceline(scenario, d / 'raceline.png')
    print('wrote previews to', d)


if __name__ == '__main__':
    main()
