#!/usr/bin/env python3
"""Cut an object (opponent car, cone, ...) out of a trained camera-3DGS scene and save it as
a standalone object splat (npz) in its own metric frame, ready to be inserted into the
camera and LiDAR maps (render servers, --object / request "objects") and exported for
Isaac Sim (scripts/splat_to_usd_gsplat.py --object-npz).

    conda activate roboracer && source scripts/env.sh
    python scripts/extract_object_splat.py \
        --ckpt  $NEW1_ROOT/checkpoints/camera_3dgs/nerfstudio_models/step-000049999.ckpt \
        --dataparser-json $NEW1_ROOT/checkpoints/camera_3dgs/dataparser_transforms.json \
        --transforms $NEW1_ROOT/dataset/transforms.json \
        --center -2.82 -3.91 -0.40 --size 0.40 0.40 0.50 --auto-floor \
        --name cone --out $NEW1_ROOT/objects/cone.npz --preview

Box centre/size are in the metric world frame of transforms.json (the frame the Isaac
scene and the render servers use). --yaw-deg is the heading of the object's forward axis in
that frame so the saved object has +x forward. Use --list-clusters first to find candidate
objects (opaque, compact gaussian clusters standing on the ground).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simulation" / "isaac_sim"))
import object_splats as osp  # noqa: E402


def list_clusters(gauss, world_from_train, floor_band=(0.05, 0.6), min_pts=150, eps=0.08):
    """Print compact clusters of opaque gaussians standing on the ground (candidate objects)."""
    from sklearn.cluster import DBSCAN
    R, s, t = osp.decompose_similarity(world_from_train)
    W = gauss["means"] @ (s * R).T + t
    opac = 1 / (1 + np.exp(-gauss["opacities"]))
    # ground = most populated 2 cm z-bin of opaque gaussians
    hist, edges = np.histogram(W[opac > 0.5, 2], bins=np.arange(W[:, 2].min(), W[:, 2].max(), 0.02))
    floor = edges[np.argmax(hist)] + 0.01
    sel = (opac > 0.5) & (W[:, 2] > floor + floor_band[0]) & (W[:, 2] < floor + floor_band[1])
    P = W[sel]
    print(f"ground z ~ {floor:.2f} m; {len(P)} opaque gaussians in the band {floor_band} m above it")
    lab = DBSCAN(eps=eps, min_samples=20).fit_predict(P)
    rows = []
    for l in sorted(set(lab) - {-1}):
        Q = P[lab == l]
        if len(Q) < min_pts:
            continue
        ext = Q.max(0) - Q.min(0)
        if ext[0] > 1.5 or ext[1] > 1.5:
            continue
        rgb = np.clip(0.5 + osp.SH_C0 * gauss["sh"][sel][lab == l][:, 0], 0, 1).mean(0)
        rows.append((len(Q), Q.mean(0), ext, rgb))
    rows.sort(key=lambda r: -r[0])
    print(f"{'n':>6}  {'centre x y z [m]':>24}  {'extent x y z [m]':>20}  mean rgb")
    for n, c, e, rgb in rows[:40]:
        print(f"{n:6d}  {c[0]:8.2f} {c[1]:7.2f} {c[2]:7.2f}  {e[0]:6.2f} {e[1]:6.2f} {e[2]:6.2f}  {rgb.round(2).tolist()}")
    print("suggested --center: cluster centre with z = ground + half extent; --size: extent + 0.1 m margin")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="nerfstudio splatfacto checkpoint (.ckpt)")
    ap.add_argument("--dataparser-json", default=None, help="dataparser_transforms.json next to config.yml")
    ap.add_argument("--transforms", default=None, help="dataset transforms.json (for applied_transform)")
    ap.add_argument("--list-clusters", action="store_true", help="print candidate objects and exit")
    ap.add_argument("--center", type=float, nargs=3, metavar=("X", "Y", "Z"))
    ap.add_argument("--size", type=float, nargs=3, metavar=("SX", "SY", "SZ"))
    ap.add_argument("--yaw-deg", type=float, default=0.0, help="heading of the object's forward axis in the world frame")
    ap.add_argument("--floor-z", type=float, default=None, help="ground height (default: box bottom)")
    ap.add_argument("--auto-floor", action="store_true", help="estimate the ground height from the gaussians around the box")
    ap.add_argument("--above-floor", type=float, default=0.015, help="drop gaussians closer than this to the ground [m]")
    ap.add_argument("--opacity-thresh", type=float, default=0.3)
    ap.add_argument("--max-scale", type=float, default=0.3, help="drop gaussians larger than this [m] (floaters)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default=None, help="output .npz (default: <ckpt dir>/objects/<name>.npz)")
    ap.add_argument("--preview", action="store_true", help="also write <out>_preview.png (turntable, needs a GPU)")
    args = ap.parse_args()

    gauss = osp.load_nerfstudio_gaussians(args.ckpt)
    if args.dataparser_json:
        W = osp.nerfstudio_world_from_train(args.dataparser_json, args.transforms)
    else:
        print("warning: no --dataparser-json, treating the training frame as the metric world frame")
        W = np.eye(4)
    print(f"{len(gauss['means'])} gaussians, world scale {osp.decompose_similarity(W)[1]:.4f} m/unit")

    if args.list_clusters:
        list_clusters(gauss, W)
        return
    if args.center is None or args.size is None:
        sys.exit("need --center and --size (or --list-clusters)")

    floor_z = args.floor_z
    if args.auto_floor:
        R, s, t = osp.decompose_similarity(W)
        means_w = gauss["means"] @ (s * R).T + t
        floor_z = osp.estimate_floor_z(means_w, 1 / (1 + np.exp(-gauss["opacities"])), args.center,
                                       radius=max(args.size[:2]) / 2 + 0.6)
        print(f"estimated ground z = {floor_z:.3f} m")
    name = args.name or "object"
    obj = osp.extract_object(gauss, W, args.center, args.size, yaw_deg=args.yaw_deg, floor_z=floor_z,
                             above_floor=args.above_floor, opacity_thresh=args.opacity_thresh,
                             max_scale=args.max_scale,
                             meta={"name": name, "source_ckpt": str(Path(args.ckpt).resolve()),
                                   "frame": "object: metres, z up, +x forward, origin on the ground"})
    out = Path(args.out) if args.out else Path(args.ckpt).resolve().parents[1] / "objects" / f"{name}.npz"
    obj.save(out)
    lo, hi = obj.bbox()
    print(f"wrote {out}: {obj.n} gaussians, extent {np.round(hi - lo, 3).tolist()} m, "
          f"height above ground {hi[2]:.3f} m")
    json.dump(obj.meta, open(out.with_suffix(".json"), "w"), indent=1)
    if args.preview:
        png = out.with_name(out.stem + "_preview.png")
        osp.render_preview(obj, png)
        print(f"preview: {png}")


if __name__ == "__main__":
    main()
