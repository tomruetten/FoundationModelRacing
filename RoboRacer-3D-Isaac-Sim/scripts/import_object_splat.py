#!/usr/bin/env python3
"""Turn an existing asset into an object splat (npz) for the opponent pipeline:

  * a mesh (USD from Isaac's asset converter, GLB/GLTF/OBJ via trimesh) -> surface-sampled
    gaussians (mesh2splat recipe, flat colours from materials / textures / vertex colours)
  * a ready-made Gaussian splat PLY (Scaniverse, Polycam, Postshot, SuperSplat, ...) -> same
    container, re-oriented and scaled to metres

Output frame: metres, +x forward, +y left, +z up, origin on the ground under the footprint
centre (what simulation/isaac_sim/object_splats.py expects).

    conda activate roboracer && source scripts/env.sh
    python scripts/import_object_splat.py --usd .../F1TENTH.usd --up Y --forward X \\
        --spacing-mm 4 --name f1tenth --out data/local/objects/f1tenth.npz --preview
    python scripts/import_object_splat.py --mesh cone.glb --up Y --forward Z --height 0.33 \\
        --name cone_mesh --out data/local/objects/cone_mesh.npz --preview
    python scripts/import_object_splat.py --ply capture.ply --up Y --forward -Z --height 0.5 \\
        --floor-percentile 1 --name car_capture --out data/local/objects/car_capture.npz

Units: USD uses its metersPerUnit; meshes and PLYs are assumed metres unless --scale or
--height (rescale so the object is this tall) is given.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simulation" / "isaac_sim"))
import object_splats as osp  # noqa: E402


def _triangulate(counts, indices):
    """USD polygon faces -> triangle index array (fan triangulation)."""
    tris, k = [], 0
    for c in counts:
        for i in range(1, c - 1):
            tris.append((indices[k], indices[k + i], indices[k + i + 1]))
        k += c
    return np.array(tris, dtype=np.int64).reshape(-1, 3)


def load_usd_mesh(path, linear_colors=False, part_rule=None):
    """All UsdGeom.Mesh prims -> (vertices [stage units], faces, face colours 0..1, metersPerUnit).
    Colour per mesh: bound material's diffuse constant (OmniPBR / UsdPreviewSurface, linear ->
    sRGB), else the displayColor primvar, else grey."""
    from pxr import Usd, UsdGeom, UsdShade
    st = Usd.Stage.Open(str(path))
    mpu = float(UsdGeom.GetStageMetersPerUnit(st)) or 1.0
    V, Fc, C, FP, names = [], [], [], [], []
    n_off = 0
    for prim in st.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)
        if len(pts) == 0:
            continue
        xf = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        pw = (np.c_[pts, np.ones(len(pts))] @ xf)[:, :3]
        tris = _triangulate(mesh.GetFaceVertexCountsAttr().Get(), mesh.GetFaceVertexIndicesAttr().Get())
        col = None
        mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if mat:
            for sp in Usd.PrimRange(mat.GetPrim()):
                if sp.IsA(UsdShade.Shader):
                    sh = UsdShade.Shader(sp)
                    for key in ("diffuse_color_constant", "diffuseColor", "base_color", "baseColor"):
                        inp = sh.GetInput(key)
                        if inp and inp.Get() is not None:
                            col = np.array(inp.Get(), dtype=np.float64)[:3]
                            if linear_colors:
                                col = osp.linear_to_srgb(col)
                            break
                if col is not None:
                    break
        if col is None:
            dc = mesh.GetDisplayColorAttr().Get()
            col = np.array(dc[0], dtype=np.float64) if dc else np.array([0.5, 0.5, 0.5])
        V.append(pw); Fc.append(tris + n_off); C.append(np.tile(col, (len(tris), 1)))
        pname = part_rule(prim.GetParent().GetName()) if part_rule else "body"
        if pname not in names:
            names.append(pname)
        FP.append(np.full(len(tris), names.index(pname), np.int16))
        n_off += len(pw)
    if not V:
        raise ValueError("no meshes in USD")
    return np.vstack(V), np.vstack(Fc), np.vstack(C), mpu, (np.concatenate(FP), names)


def load_trimesh(path, linear_colors=True):
    """GLB/GLTF/OBJ -> (vertices, faces, face colours 0..1). Textures sampled at face centroids,
    PBR baseColorFactor and vertex colours handled; node transforms applied."""
    import trimesh
    scene = trimesh.load(str(path), force="scene")
    V, Fc, C = [], [], []
    n_off = 0
    for node in scene.graph.nodes_geometry:
        T, gname = scene.graph[node]
        g = scene.geometry[gname]
        if not isinstance(g, trimesh.Trimesh) or len(g.faces) == 0:
            continue
        pw = trimesh.transform_points(g.vertices, T)
        vis = g.visual
        col = None
        if hasattr(vis, "material"):
            mat = vis.material
            tex = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
            uv = getattr(vis, "uv", None)
            if tex is not None and uv is not None and len(uv) == len(g.vertices):
                img = np.asarray(tex.convert("RGB"), dtype=np.float64) / 255.0
                H, W = img.shape[:2]
                uvc = uv[g.faces].mean(1)                              # face centroid uv
                iu = np.clip((uvc[:, 0] % 1.0 * (W - 1)).astype(int), 0, W - 1)
                iv = np.clip(((1 - uvc[:, 1] % 1.0) * (H - 1)).astype(int), 0, H - 1)
                col = img[iv, iu]
            else:
                fac = getattr(mat, "baseColorFactor", None)
                if fac is None:
                    fac = getattr(mat, "diffuse", None)
                if fac is not None:
                    fac = np.asarray(fac, np.float64)[:3] / (255.0 if np.max(fac) > 1.0 else 1.0)
                    col = np.tile(osp.linear_to_srgb(fac) if linear_colors else fac, (len(g.faces), 1))
        if col is None and hasattr(vis, "face_colors") and vis.face_colors is not None and len(vis.face_colors) == len(g.faces):
            col = np.asarray(vis.face_colors, np.float64)[:, :3] / 255.0
        if col is None and hasattr(vis, "vertex_colors") and vis.vertex_colors is not None and len(vis.vertex_colors) == len(g.vertices):
            col = np.asarray(vis.vertex_colors, np.float64)[g.faces, :3].mean(1) / 255.0
        if col is None:
            col = np.tile([0.5, 0.5, 0.5], (len(g.faces), 1))
        V.append(pw); Fc.append(np.asarray(g.faces) + n_off); C.append(col)
        n_off += len(pw)
    if not V:
        raise ValueError("no triangle geometry in file")
    return np.vstack(V), np.vstack(Fc), np.vstack(C)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--usd", help="mesh asset as USD (e.g. Isaac asset-converter output)")
    src.add_argument("--mesh", help="mesh asset as GLB/GLTF/OBJ")
    src.add_argument("--ply", help="ready-made Gaussian splat PLY")
    ap.add_argument("--up", default="Z", help="source axis pointing up (X, -X, Y, -Y, Z, -Z)")
    ap.add_argument("--forward", default="X", help="source axis pointing forward")
    ap.add_argument("--scale", type=float, default=None, help="source units -> metres (default: USD metersPerUnit, else 1)")
    ap.add_argument("--height", type=float, default=None, help="rescale so the object is this tall [m]")
    ap.add_argument("--spacing-mm", type=float, default=4.0, help="mesh sampling: gaussian spacing on the surface")
    ap.add_argument("--overlap", type=float, default=2.0, help="mesh sampling: gaussians per spacing^2")
    ap.add_argument("--footprint", type=float, default=1.0,
                    help="mesh sampling: scale factor on the in-plane gaussian size (>1 closes gaps, blurs edges)")
    ap.add_argument("--linear-colors", choices=["auto", "yes", "no"], default="auto",
                    help="material constants are linear RGB and need the sRGB transfer (auto: yes for glTF, "
                         "no for USD converted from FBX/Unity whose constants are already sRGB values)")
    ap.add_argument("--floor-percentile", type=float, default=0.0,
                    help="PLY: ground = this percentile of the gaussians' height (floaters below are cut)")
    ap.add_argument("--opacity-thresh", type=float, default=0.0, help="PLY: drop gaussians below this opacity")
    ap.add_argument("--parts", choices=["none", "f1tenth"], default="none",
                    help="tag gaussians by part for articulation (f1tenth: wheels by mesh name, front wheels steerable)")
    ap.add_argument("--shadow", action="store_true", help="add a soft contact shadow part under the footprint (camera only)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--preview", action="store_true", help="also write <out>_preview.png (needs a GPU)")
    args = ap.parse_args()

    R = osp.frame_from_axes(args.forward, args.up)
    if args.ply:
        gauss = osp.load_splat_ply(args.ply)
        keep = 1 / (1 + np.exp(-gauss["opacities"])) >= args.opacity_thresh
        gauss = {k: v[keep] for k, v in gauss.items()}
        source, scale = args.ply, args.scale or 1.0
        print(f"{len(gauss['means'])} gaussians from {args.ply} (SH coefficients: {gauss['sh'].shape[1]})")
    else:
        face_part = part_names = None
        if args.usd:
            lin = args.linear_colors == "yes"
            V, F, C, mpu, (face_part, part_names) = load_usd_mesh(args.usd, linear_colors=lin,
                                                                 part_rule=osp.classify_part if args.parts == "f1tenth" else None)
            source, scale = args.usd, args.scale or mpu
        else:
            lin = args.linear_colors != "no"
            V, F, C = load_trimesh(args.mesh, linear_colors=lin)
            source, scale = args.mesh, args.scale or 1.0
        V_m = V * scale
        if args.height:                                              # rescale the mesh first, then sample at the wanted spacing
            h = float(np.ptp((V_m @ R.T)[:, 2]))
            V_m *= args.height / h
            print(f"rescaled mesh by {args.height / h:.4f} to {args.height} m height")
        print(f"mesh: {len(V)} vertices, {len(F)} triangles, extent {np.round((V_m.max(0) - V_m.min(0)), 3).tolist()} m before re-orientation")
        gauss = osp.gaussians_from_mesh(V_m, F, C, spacing=args.spacing_mm / 1000.0, overlap=args.overlap,
                                        footprint=args.footprint, face_part=face_part)
        scale = 1.0                                                  # already metres
        args.height = None
        print(f"{len(gauss['means'])} gaussians sampled ({args.spacing_mm} mm spacing, overlap {args.overlap})")

    if args.height:
        z = (gauss["means"] @ R.T)[:, 2] * scale
        h = float(np.percentile(z, 99.9) - np.percentile(z, args.floor_percentile))
        scale *= args.height / h
        print(f"rescaled by {args.height / h:.4f} to {args.height} m height")
    z = (gauss["means"] @ R.T)[:, 2] * scale
    floor = float(np.percentile(z, args.floor_percentile)) if args.floor_percentile > 0 else "min"
    obj = osp.gaussians_to_object(gauss, R, scale=scale, floor=floor,
                                  meta={"name": args.name, "source": str(Path(source).resolve()),
                                        "frame": "object: metres, z up, +x forward, origin on the ground",
                                        "src_up": args.up, "src_forward": args.forward},
                                  part_names=part_names if not args.ply else None)
    if args.shadow:
        lo, hi = obj.bbox()
        obj = obj.with_part(osp.contact_shadow(hi[0] - lo[0], hi[1] - lo[1]), {"name": "shadow", "kind": "shadow", "lidar": False})
        print(f"contact shadow added: {obj.n - sum(1 for _ in [])} gaussians total")
    if args.ply and args.floor_percentile > 0:
        keep = obj.means[:, 2] >= -0.02
        obj = osp.ObjectSplat(obj.means[keep], obj.scales_log[keep], obj.quats[keep], obj.opacities[keep], obj.sh[keep], obj.meta)
    out = Path(args.out)
    obj.save(out)
    lo, hi = obj.bbox()
    print(f"wrote {out}: {obj.n} gaussians, size x {hi[0]-lo[0]:.3f} y {hi[1]-lo[1]:.3f} z {hi[2]-lo[2]:.3f} m")
    json.dump(obj.meta, open(out.with_suffix(".json"), "w"), indent=1)
    if args.preview:
        png = out.with_name(out.stem + "_preview.png")
        osp.render_preview(obj, png)
        print(f"preview: {png}")


if __name__ == "__main__":
    main()
