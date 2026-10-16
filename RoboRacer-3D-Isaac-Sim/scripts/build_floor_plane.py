#!/usr/bin/env python3
"""Fit the reconstructed floor of a trained camera-3DGS scene as a plane, so opponents and
obstacles stand on the floor the cameras see (object_splats.FloorPlane).

    conda activate roboracer && source scripts/env.sh
    python scripts/build_floor_plane.py \
        --ckpt $NEW1_ROOT/checkpoints/camera_3dgs/nerfstudio_models/step-000049999.ckpt \
        --dataparser-json $NEW1_ROOT/checkpoints/camera_3dgs/dataparser_transforms.json \
        --transforms $NEW1_ROOT/dataset/transforms.json          # -> $NEW1_ROOT/dataset/floor_plane.json

The plane is written next to transforms.json (the frame the Isaac clients and render servers
use); the clients pick it up automatically. Prints the fit quality and the camera height above
the floor along the lap (should be nearly constant; jitter there is exactly what the plane
removes from the opponent's motion).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simulation" / "isaac_sim"))
import object_splats as osp  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataparser-json", required=True)
    ap.add_argument("--transforms", required=True, help="trajectory transforms.json (frame of the plane)")
    ap.add_argument("--out", default=None, help="output json (default: floor_plane.json next to --transforms)")
    ap.add_argument("--corridor", type=float, default=1.2, help="use floor within this distance of the path [m]")
    ap.add_argument("--trim", type=float, default=0.03, help="outlier cut for the robust fit [m]")
    ap.add_argument("--clearance", type=float, default=0.02,
                    help="visual clearance objects get above the plane [m]; the splat floor is a band of opaque "
                         "gaussians (New1: -6..+7.5 cm around the plane) that swallows the lowest centimetres of an "
                         "object placed exactly on it; 2 cm keeps wheels visible without a visible gap (sweep 0-4 cm)")
    args = ap.parse_args()

    g = osp.load_nerfstudio_gaussians(args.ckpt)
    W = osp.nerfstudio_world_from_train(args.dataparser_json, args.transforms)
    R, s, t = osp.decompose_similarity(W)
    means_w = g["means"] @ (s * R).T + t
    frames = json.load(open(args.transforms))["frames"]
    C = np.array([np.array(f["transform_matrix"])[:3, 3] for f in frames])
    up = osp.trajectory_up(frames)
    samples = osp.floor_samples_from_scene(means_w, g["opacities"], g["scales_log"] + np.log(s), C, up, corridor=args.corridor)
    plane = osp.FloorPlane.fit(samples, up_hint=up, trim=args.trim,
                               meta={"source_ckpt": str(Path(args.ckpt).resolve()), "transforms": str(Path(args.transforms).resolve()),
                                     "frame": "world frame of transforms.json"})
    plane.clearance = args.clearance
    band = plane.height(means_w)
    near = (np.abs(band) < 0.10) & (1 / (1 + np.exp(-g["opacities"])) > 0.3)
    plane.meta["floor_band_p5_p95_m"] = [float(np.percentile(band[near], 5)), float(np.percentile(band[near], 95))]
    h = plane.height(C)
    plane.meta.update({"camera_height_mean_m": float(h.mean()), "camera_height_std_m": float(h.std()),
                       "camera_height_min_m": float(h.min()), "camera_height_max_m": float(h.max()),
                       "tilt_vs_z_deg": float(np.degrees(np.arccos(abs(plane.normal[2]))))})
    out = Path(args.out) if args.out else Path(args.transforms).parent / "floor_plane.json"
    plane.save(out)
    print(f"{plane}: {plane.meta['n_inliers']}/{plane.meta['n_samples']} floor cells within {args.trim} m, "
          f"tilt vs world z {plane.meta['tilt_vs_z_deg']:.2f} deg")
    print(f"camera height above the floor along the lap: mean {h.mean():.3f} m, std {h.std():.3f}, "
          f"min {h.min():.3f}, max {h.max():.3f}  (this jitter no longer reaches the opponent)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
