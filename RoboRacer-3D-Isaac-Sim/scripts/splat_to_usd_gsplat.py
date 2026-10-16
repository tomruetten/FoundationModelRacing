#!/usr/bin/env python3
"""Convert a nerfstudio splatfacto checkpoint into Isaac Sim's native
Gaussian-splat USD (UsdVol.ParticleField3DGaussianSplat) so the trained 3DGS
scene renders photoreal inside the Isaac Sim 6.x viewport.

Needs Isaac Sim >= 6.0 installed via pip (extension omni.kit.converter.gsplat
ships the `usd_convert_gsplat` package). Run inside that env:

    conda activate env_isaac6
    python scripts/splat_to_usd_gsplat.py \\
        --ckpt data/local/New1/checkpoints/camera_3dgs/nerfstudio_models/step-000049999.ckpt \\
        --dataparser-json data/local/New1/checkpoints/camera_3dgs/dataparser_transforms.json \\
        --transforms data/local/New1/dataset/transforms.json --to-world \\
        --out-usd data/local/New1/scene_splat_gsplat.usd

--object-npz converts an object splat written by scripts/extract_object_splat.py
(already metric, z up, +x forward, origin on the ground) into a standalone USD; the
Isaac client poses it under an Xform (--opponent-usd).

--to-world writes TWO files: <out>_raw.usd with the gaussians untouched in
nerfstudio's training frame, and <out> (the file to load) that references it
under an Xform carrying the training->world transform (rotation + uniform
scale). Isaac Sim applies the xform, so view-dependent colour (spherical
harmonics) stays correct. Rotating the gaussians themselves without rotating
the SH coefficients gives colour casts (verified 2026-09-10).
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


def _add_converter_to_path():
    try:
        import isaacsim  # noqa: F401
    except ImportError:
        sys.exit("run inside the Isaac Sim 6 env (conda activate env_isaac6)")
    root = os.path.dirname(isaacsim.__file__)
    hits = glob.glob(os.path.join(root, "extscache", "omni.kit.converter.gsplat-*", "pip_prebundle"))
    if not hits:
        sys.exit("omni.kit.converter.gsplat not found in this Isaac Sim install (needs >= 6.0)")
    sys.path.insert(0, hits[0])
    return hits[0]


def quat_mul(q1, q2):
    """Hamilton product, quaternions as (w, x, y, z)."""
    w1, x1, y1, z1 = q1.T
    w2, x2, y2, z2 = q2.T
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="nerfstudio splatfacto checkpoint")
    ap.add_argument("--object-npz", default=None, help="object splat (extract_object_splat.py) instead of --ckpt")
    ap.add_argument("--out-usd", required=True)
    ap.add_argument("--dataparser-json", default=None)
    ap.add_argument("--transforms", default=None)
    ap.add_argument("--to-world", action="store_true")
    ap.add_argument("--opacity-thresh", type=float, default=0.05)
    ap.add_argument("--around-trajectory", default=None, metavar="TRANSFORMS_JSON")
    ap.add_argument("--margin", type=float, default=10.0)
    ap.add_argument("--max-scale", type=float, default=None,
                    help="drop gaussians whose largest axis exceeds this [m, world frame]; "
                         "kills the big blurry floaters that dominate novel views")
    ap.add_argument("--z-range", type=float, nargs=2, default=None,
                    help="keep gaussians with world z in [lo, hi] (crop ceiling / floor noise)")
    ap.add_argument("--prim-name", default="Splat3DGS")
    ap.add_argument("--no-sh", action="store_true", help="drop higher-order SH (smaller file)")
    args = ap.parse_args()

    pre = _add_converter_to_path()
    from usd_convert_gsplat import GaussianSplatData, write_gaussian_splat_usd
    print(f"converter: {pre}")

    if args.object_npz:
        if args.to_world:
            sys.exit("--object-npz is already metric; drop --to-world")
        z = np.load(args.object_npz, allow_pickle=False)
        means = z["means"].astype(np.float64)
        scales_log = z["scales_log"].astype(np.float64)
        quats = z["quats"].astype(np.float64)
        opac_logit = z["opacities"].astype(np.float32).reshape(-1)
        f_dc = z["sh"][:, 0].astype(np.float32)
        f_rest_raw = z["sh"][:, 1:].astype(np.float32)
        source = args.object_npz
    elif args.ckpt:
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        gp = ckpt["pipeline"]
        means = gp["_model.gauss_params.means"].numpy().astype(np.float64)
        scales_log = gp["_model.gauss_params.scales"].numpy().astype(np.float64)
        quats = gp["_model.gauss_params.quats"].numpy().astype(np.float64)  # nerfstudio: (w,x,y,z)
        opac_logit = gp["_model.gauss_params.opacities"].numpy().astype(np.float32).squeeze(-1)
        f_dc = gp["_model.gauss_params.features_dc"].numpy().astype(np.float32)
        f_rest_raw = gp["_model.gauss_params.features_rest"].numpy().astype(np.float32)  # (N,15,3)
        source = args.ckpt
    else:
        sys.exit("need --ckpt or --object-npz")
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    n0 = len(means)
    print(f"{n0} gaussians")

    M_world_from_train = None
    means_world, scales_log_world = means, scales_log
    if args.to_world:
        if not args.dataparser_json:
            sys.exit("--to-world needs --dataparser-json")
        dp = json.load(open(args.dataparser_json))
        T = np.eye(4)
        T[:3, :4] = np.array(dp["transform"])
        s = float(dp["scale"])
        if args.transforms:
            meta = json.load(open(args.transforms))
            if "applied_transform" in meta:
                A = np.eye(4)
                A[:3, :4] = np.array(meta["applied_transform"])
                T = T @ np.linalg.inv(A)
        # training = (T @ world) * s   ->   world = T^-1 @ (training / s)
        M_world_from_train = np.linalg.inv(T) @ np.diag([1 / s, 1 / s, 1 / s, 1.0])
        means_world = means @ M_world_from_train[:3, :3].T + M_world_from_train[:3, 3]
        scales_log_world = scales_log - np.log(s)
        print(f"world transform: scale {1/s:.4f}, filters applied in world metres")

    keep = 1.0 / (1.0 + np.exp(-opac_logit)) >= args.opacity_thresh
    if args.max_scale:
        keep &= np.exp(scales_log_world).max(axis=1) <= args.max_scale
    if args.z_range:
        keep &= (means_world[:, 2] >= args.z_range[0]) & (means_world[:, 2] <= args.z_range[1])
    if args.around_trajectory:
        fr = json.load(open(args.around_trajectory))["frames"]
        traj = np.array([np.array(f["transform_matrix"])[:3, 3] for f in fr])
        lo, hi = traj.min(axis=0) - args.margin, traj.max(axis=0) + args.margin
        keep &= np.all((means_world >= lo) & (means_world <= hi), axis=1)
    idx = np.where(keep)[0]
    print(f"keeping {len(idx)}/{n0}")

    # PLY/converter convention: f_rest channel-major (N, 3, 15) flattened -> (N, 45)
    f_rest = None if args.no_sh else np.ascontiguousarray(f_rest_raw[idx].transpose(0, 2, 1)).reshape(len(idx), -1)
    data = GaussianSplatData(
        positions=means[idx].astype(np.float32),
        scales=scales_log[idx].astype(np.float32),
        rotations=quats[idx].astype(np.float32),
        opacities=opac_logit[idx],
        f_dc=f_dc[idx],
        f_rest=f_rest,
        count=len(idx),
    )
    out = Path(args.out_usd)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw_out = out.with_name(out.stem + "_raw" + out.suffix) if M_world_from_train is not None else out
    res = write_gaussian_splat_usd(data, str(raw_out), source_file=source,
                                   prim_name=args.prim_name, up_axis="Z")
    print(f"wrote {res}  ({raw_out.stat().st_size / 1e6:.1f} MB)")
    if M_world_from_train is not None:
        from pxr import Usd, UsdGeom, Gf
        stage = Usd.Stage.CreateNew(str(out))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        root = UsdGeom.Xform.Define(stage, "/" + args.prim_name)
        stage.SetDefaultPrim(root.GetPrim())
        root.MakeMatrixXform().Set(Gf.Matrix4d(M_world_from_train.T.tolist()))  # USD: row vectors
        child = stage.DefinePrim(f"/{args.prim_name}/Gaussians")           # typeless -> takes referenced type
        child.GetReferences().AddReference("./" + raw_out.name)
        stage.GetRootLayer().Save()
        print(f"wrote {out}  (wrapper: world xform + reference to {raw_out.name})")


if __name__ == "__main__":
    main()
