"""Dependency-light software preview: rasterise the USD hoses/floor from the ego camera.

Only a geometry sanity check (proportions, camera axes, clearances); not a photoreal render.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from racing.config import sensor_pose


def load_triangles(usd_path, prims=('/World/Track', '/World/Hall/Floor')):
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
    x, y, yaw = pose_xy_yaw
    c2w = sensor_pose(dict(x=x, y=y, yaw=yaw), scenario['sensors']['camera'])
    w2c = np.linalg.inv(c2w)
    p = tris @ w2c[:3, :3].T + w2c[:3, 3]                 # optical frame: x right, y down, z forward
    depth = p[..., 2]; keep = (depth > .05).all(1) & (depth < 40).all(1)
    tris, p = tris[keep], p[keep]
    uv = np.stack([cam['fx'] * p[..., 0] / p[..., 2] + cam['cx'], cam['fy'] * p[..., 1] / p[..., 2] + cam['cy']], -1)
    normal = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-12
    shade = .25 + .75 * np.abs(normal @ np.array([.3, .2, .93]))
    order = np.argsort(-p[..., 2].mean(1))
    img = Image.new('RGB', (w, h), (150, 160, 170)); draw = ImageDraw.Draw(img)
    for i in order:
        flat = tris[i, :, 2].max() < .001
        col = np.array((125, 125, 118) if flat else (225, 225, 220)) * shade[i]
        draw.polygon([tuple(q) for q in uv[i]], fill=tuple(col.astype(int)))
    return img


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
    print('wrote previews to', d)


if __name__ == '__main__':
    main()
