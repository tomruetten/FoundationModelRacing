#!/usr/bin/env python3
"""Bake a nerfstudio splatfacto checkpoint into a plain USD Points prim.

Gives Isaac Sim (any version) a visually faithful, colored version of the
trained 3DGS scene that coexists with ordinary USD geometry (car, lights).
The native gaussian-splat prim of Isaac Sim 6 renders nicer but hides all
other geometry in the same stage (observed 2026-08-29), hence this fallback.

Positions are the checkpoint's raw `means`: that is the nerfstudio training
frame, i.e. exactly the frame `SplatCameraRenderer.to_model_frame()` maps
poses into. Pass --to-world with the checkpoint's dataparser_transforms.json
to instead express the points in the dataset/COLMAP world frame that
transforms.json poses (and the Isaac Sim replay client) use.

    conda activate env_isaaclab      # any env with torch + pxr (usd-core)
    python scripts/splat_to_usd_points.py \\
        --ckpt data/local/New1/checkpoints/camera_3dgs/nerfstudio_models/step-000049999.ckpt \\
        --dataparser-json data/local/New1/checkpoints/camera_3dgs/dataparser_transforms.json \\
        --to-world --out-usd data/local/New1/scene_splat_points.usda
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

SH_C0 = 0.28209479177387814


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="nerfstudio_models/step-XXXX.ckpt")
    ap.add_argument("--out-usd", required=True)
    ap.add_argument("--dataparser-json", default=None,
                    help="dataparser_transforms.json next to config.yml (needed for --to-world)")
    ap.add_argument("--to-world", action="store_true",
                    help="undo the dataparser transform+scale so points are in the "
                         "transforms.json world frame (metres)")
    ap.add_argument("--transforms", default=None,
                    help="dataset transforms.json; its applied_transform is divided out of the "
                         "dataparser transform (same fix as splat_camera_renderer.py). Required "
                         "for --to-world when the dataset has an applied_transform.")
    ap.add_argument("--prim-path", default="/World/Splat")
    ap.add_argument("--opacity-thresh", type=float, default=0.3,
                    help="drop gaussians with sigmoid(opacity) below this (floaters)")
    ap.add_argument("--max-scale", type=float, default=None,
                    help="drop gaussians whose largest axis exceeds this (metres, world frame)")
    ap.add_argument("--point-width", type=float, default=0.02, help="rendered point size [m]")
    ap.add_argument("--z-range", type=float, nargs=2, default=None,
                    help="keep only points with world z in [lo, hi] (crop ceiling/floor noise)")
    ap.add_argument("--bbox", type=float, nargs=6, default=None, metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
                    help="keep only points inside this world-frame box (drops far floaters)")
    ap.add_argument("--around-trajectory", default=None, metavar="TRANSFORMS_JSON",
                    help="keep points within --margin metres of the camera trajectory bbox")
    ap.add_argument("--margin", type=float, default=8.0)
    ap.add_argument("--max-points", type=int, default=0, help="random subsample (0 = all)")
    args = ap.parse_args()

    from pxr import Usd, UsdGeom, Vt

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    gp = ckpt["pipeline"]
    means = gp["_model.gauss_params.means"].numpy().astype(np.float64)
    opac = 1.0 / (1.0 + np.exp(-gp["_model.gauss_params.opacities"].numpy().astype(np.float64).squeeze(-1)))
    f_dc = gp["_model.gauss_params.features_dc"].numpy().astype(np.float32)
    scales = np.exp(gp["_model.gauss_params.scales"].numpy().astype(np.float64))
    n0 = len(means)
    print(f"{n0} gaussians in checkpoint")

    scale_factor = 1.0
    if args.to_world:
        if not args.dataparser_json:
            raise SystemExit("--to-world needs --dataparser-json")
        dp = json.load(open(args.dataparser_json))
        T = np.eye(4)
        T[:3, :4] = np.array(dp["transform"])
        s = float(dp["scale"])
        # dataparser_transforms.json stores transform_matrix @ applied_transform;
        # poses in transforms.json are already past applied_transform.
        if args.transforms:
            meta = json.load(open(args.transforms))
            if "applied_transform" in meta:
                A = np.eye(4)
                A[:3, :4] = np.array(meta["applied_transform"])
                T = T @ np.linalg.inv(A)
                print("applied_transform divided out")
        # training frame = (T @ world) * s   ->   world = T^-1 @ (training / s)
        Tinv = np.linalg.inv(T)
        pts = means / s
        means = pts @ Tinv[:3, :3].T + Tinv[:3, 3]
        scale_factor = 1.0 / s
        print(f"mapped to world frame (scale {s:.4f}, transform from {Path(args.dataparser_json).name})")
    scales = scales * scale_factor

    keep = opac >= args.opacity_thresh
    if args.max_scale:
        keep &= scales.max(axis=1) <= args.max_scale
    if args.z_range:
        keep &= (means[:, 2] >= args.z_range[0]) & (means[:, 2] <= args.z_range[1])
    bbox = args.bbox
    if args.around_trajectory:
        fr = json.load(open(args.around_trajectory))["frames"]
        traj = np.array([np.array(f["transform_matrix"])[:3, 3] for f in fr])
        lo, hi = traj.min(axis=0) - args.margin, traj.max(axis=0) + args.margin
        bbox = [lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]]
        print(f"crop box from trajectory +- {args.margin} m: {np.round(bbox, 1)}")
    if bbox:
        keep &= ((means[:, 0] >= bbox[0]) & (means[:, 0] <= bbox[1]) &
                 (means[:, 1] >= bbox[2]) & (means[:, 1] <= bbox[3]) &
                 (means[:, 2] >= bbox[4]) & (means[:, 2] <= bbox[5]))
    means, opac, f_dc = means[keep], opac[keep], f_dc[keep]
    if args.max_points and len(means) > args.max_points:
        sel = np.random.default_rng(0).choice(len(means), args.max_points, replace=False)
        means, opac, f_dc = means[sel], opac[sel], f_dc[sel]
    print(f"keeping {len(means)}/{n0}")
    colors = np.clip(0.5 + SH_C0 * f_dc, 0.0, 1.0).astype(np.float32)

    out = Path(args.out_usd)
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)
    pts = UsdGeom.Points.Define(stage, args.prim_path)
    pts.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(means.astype(np.float32)))
    pts.CreateWidthsAttr(Vt.FloatArray([float(args.point_width)] * len(means)))
    pts.CreateDisplayColorAttr(Vt.Vec3fArray.FromNumpy(colors))
    pts.CreateDisplayOpacityAttr(Vt.FloatArray.FromNumpy(opac.astype(np.float32)))
    stage.GetRootLayer().Save()
    lo, hi = means.min(axis=0), means.max(axis=0)
    print(f"saved {out}  extent x[{lo[0]:.1f},{hi[0]:.1f}] y[{lo[1]:.1f},{hi[1]:.1f}] z[{lo[2]:.1f},{hi[2]:.1f}]")


if __name__ == "__main__":
    main()
