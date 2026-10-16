#!/usr/bin/env python3
"""Place a 3DGRUT export (ParticleField USDZ, COLMAP frame) into the metric,
Z-up Isaac Sim frame and convert the recorded trajectory into the same frame.

The 3DGRUT asset is exported with apply_normalizing_transform=false, i.e. in
the COLMAP world of the session (arbitrary orientation and scale). The earlier
Isaac scenes use the frame nerfstudio's dataparser produces (auto-oriented,
Z-up, centred) scaled to metres with an odometry-fitted factor
(METERS_PER_NERFUNIT, 3.9868 for session 142014_noflare, see
COLMAP位姿_3DGS点云_小车轨迹同步完整流程_20260902.md). So:

    X_isaac = m * s * (R_dp @ X_colmap + t_dp)

with (R_dp, t_dp, s) from dataparser_transforms.json and m = metres per unit.
Writes <out>.usda (Xform with that matrix referencing the USDZ) and, if
--transforms is given, <out>_trajectory.json with every camera pose mapped
into the same frame (for 02_camera_splat_client_isaac.py --transforms).

    conda activate roboracer   # any env with pxr (usd-core) + numpy
    python scripts/wrap_3dgrut_usd.py \\
        --usdz  /media/.../runs/<exp>/<run>/export_last_lightfield.usdz \\
        --dataparser-json data/local/sessions/session_20260827_142014_noflare/outputs/colmap_dataset_full/splatfacto/2026-08-28_234736/dataparser_transforms.json \\
        --meters-per-unit 3.9868 \\
        --transforms data/local/sessions/session_20260827_142014_noflare/dataset/transforms_colmap_full_ALL1521_ALIGNED.json \\
        --out data/local/sessions/session_20260827_142014_noflare/scene_3dgrut.usda
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--usdz", required=True)
    ap.add_argument("--dataparser-json", required=True)
    ap.add_argument("--meters-per-unit", type=float, required=True,
                    help="metres per nerfstudio unit (odometry-fitted, session specific)")
    ap.add_argument("--transforms", default=None, help="trajectory transforms.json in the COLMAP frame")
    ap.add_argument("--out", required=True, help="output .usda wrapper")
    ap.add_argument("--prim-name", default="Splat3DGRUT")
    args = ap.parse_args()

    from pxr import Usd, UsdGeom, Gf

    dp = json.load(open(args.dataparser_json))
    T = np.eye(4)
    T[:3, :4] = np.array(dp["transform"])
    s = float(dp["scale"]) * args.meters_per_unit
    M = np.diag([s, s, s, 1.0]) @ T          # X_isaac = s * (R X + t)
    print(f"dataparser scale {dp['scale']:.5f} x {args.meters_per_unit} m/unit -> total scale {s:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    usdz = Path(args.usdz).resolve()
    rel = os.path.relpath(usdz, out.parent.resolve())
    stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/" + args.prim_name)
    stage.SetDefaultPrim(root.GetPrim())
    root.MakeMatrixXform().Set(Gf.Matrix4d(M.T.tolist()))   # USD row-vector convention
    child = stage.DefinePrim(f"/{args.prim_name}/Asset")     # typeless -> takes the referenced type
    child.GetReferences().AddReference(rel)
    stage.GetRootLayer().Save()
    print(f"wrote {out} -> {rel}")

    if args.transforms:
        meta = json.load(open(args.transforms))
        R = M[:3, :3] / s
        for f in meta["frames"]:
            c2w = np.array(f["transform_matrix"])
            f["transform_matrix_source"] = c2w.tolist()   # original frame, for the gsplat render server
            new = np.eye(4)
            new[:3, :3] = R @ c2w[:3, :3]                 # camera axes rotated, no scale
            new[:3, 3] = M[:3, :3] @ c2w[:3, 3] + M[:3, 3]  # position scaled + shifted
            f["transform_matrix"] = new.tolist()
        meta.pop("applied_transform", None)
        tj = out.with_name(out.stem + "_trajectory.json")
        json.dump(meta, open(tj, "w"))
        P = np.array([np.array(f["transform_matrix"])[:3, 3] for f in meta["frames"]])
        print(f"wrote {tj}: {len(meta['frames'])} poses, extent x{P[:,0].min():.1f}..{P[:,0].max():.1f} "
              f"y{P[:,1].min():.1f}..{P[:,1].max():.1f} z{P[:,2].min():.2f}..{P[:,2].max():.2f} m")


if __name__ == "__main__":
    main()
