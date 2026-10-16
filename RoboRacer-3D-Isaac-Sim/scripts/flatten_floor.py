#!/usr/bin/env python3
"""Make the floor of a trained scene flat.

The floor of a splat is a band of opaque gaussians (New1: -6 to +7.5 cm around the fitted
plane) although the hall floor is planar to a centimetre. LiDAR clouds are no help here
(the accumulated New1 cloud has the same +-7 cm floor noise and disagrees with the camera map
by 21 cm median), but the plane itself is an exact geometric prior. This script

  1. projects the floor gaussians of the CAMERA map (within the camera corridor and +-band of
     the plane) onto the plane,
  2. fine-tunes the checkpoint for a few thousand steps with the photometric loss of
     splatfacto (L1 + 0.2 D-SSIM) plus an L1 depth loss that keeps floor pixels on the plane,
     so the texture re-fits the moved geometry and the band does not re-open,
  3. optionally flattens the LiDAR map's floor the same way (hard projection, no appearance).

    conda activate roboracer && source scripts/env.sh
    python scripts/flatten_floor.py --config $CAMERA_3DGS_CONFIG --transforms $NEW1_ROOT/dataset/transforms.json \\
        --floor-plane $NEW1_ROOT/dataset/floor_plane.json --images $DATA_DRIVE_ROOT/New1/dataset/images_real \\
        --camera-params $NEW1_ROOT/camera_params.json --out $NEW1_ROOT/checkpoints/camera_3dgs_flat \\
        --lidar-ckpt $LIDAR_3DGS_CKPT --steps 3000

Writes <out>/config.yml, dataparser_transforms.json and nerfstudio_models/<same step>.ckpt (a
drop-in for CAMERA_3DGS_CONFIG), <lidar ckpt dir>/best_flat.pt, and a report.json with the
floor band and held-out PSNR before and after.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simulation" / "isaac_sim"))
import object_splats as osp  # noqa: E402


def load_images(frames, image_dir, cam_params, cache_dir=None):
    """Undistorted frames (uint8 HxWx3) keyed by frame index; images_real are raw frames and the
    model was trained on cv2.undistort(K, D) of them (see evaluation/eval_camera.py)."""
    import cv2
    K = D = None
    if cam_params:
        cp = json.load(open(cam_params))
        i = cp["intrinsics"]
        K = np.array([[i["fx"], 0, i["cx"]], [0, i["fy"], i["cy"]], [0, 0, 1]], dtype=np.float64)
        D = np.array(cp["distortion_coefficients"]["D"], dtype=np.float64)
    out = {}
    for i, f in enumerate(frames):
        name = Path(f["file_path"]).name
        for cand in (Path(image_dir) / name, Path(image_dir) / f["file_path"]):
            if cand.exists():
                img = cv2.imread(str(cand), cv2.IMREAD_COLOR)
                if K is not None:
                    img = cv2.undistort(img, K, D)
                out[i] = np.ascontiguousarray(img[:, :, ::-1])
                break
    return out


def viewmat_from_c2w(c2w):
    """nerfstudio/OpenGL c2w -> gsplat world-to-camera (OpenCV)."""
    c = np.array(c2w, dtype=np.float64)
    if c.shape[0] == 3:
        c = np.vstack([c, [0, 0, 0, 1]])
    c = c.copy()
    c[:3, 1:3] *= -1
    return np.linalg.inv(c)


def plane_depth(viewmat, K, W, H, plane_model):
    """z-depth (camera space) of the ray-plane intersection for every pixel; nan where the ray
    does not hit the plane in front of the camera."""
    n, p0 = plane_model
    c2w = np.linalg.inv(viewmat)
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    d_cam = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
    d_w = d_cam @ c2w[:3, :3].T
    o = c2w[:3, 3]
    denom = d_w @ n
    t = ((p0 - o) @ n) / np.where(np.abs(denom) < 1e-9, np.nan, denom)
    z = t * d_cam[..., 2]          # depth along the camera axis = t * (d_cam z = 1)
    z[(t <= 0.05)] = np.nan
    return z


def ssim_loss(a, b):
    from pytorch_msssim import ssim
    return 1.0 - ssim(a.permute(2, 0, 1)[None], b.permute(2, 0, 1)[None], data_range=1.0, size_average=True)


def flatten_lidar(args, plane, C_w, report, out):
    """Hard-flatten the LiDAR map's floor onto its own plane (appearance does not matter there).
    The LiDAR map is sparse (105k gaussians for the hall), so floor cells are 20 cm with >= 3
    gaussians."""
    L = torch.load(args.lidar_ckpt, map_location="cpu", weights_only=False)
    means = L["means"].numpy().astype(np.float64)
    op = torch.sigmoid(L["opacities"]).numpy().reshape(-1)
    sc = torch.exp(L["scales"]).numpy().max(1)
    samples = osp.floor_samples_from_scene(means, L["opacities"].numpy().reshape(-1), L["scales"].numpy(), C_w, plane.normal,
                                           corridor=args.corridor, cell=0.20, min_per_cell=3)
    if len(samples) < 20:
        raise RuntimeError(f"only {len(samples)} LiDAR floor cells found")
    lplane = osp.FloorPlane.fit(samples, up_hint=plane.normal)
    from scipy.spatial import cKDTree
    d, _ = cKDTree(lplane.project(C_w)[:, :2]).query(means[:, :2])
    h = lplane.height(means)
    sel = (d < args.corridor) & (np.abs(h) < args.band) & (sc < 0.15)
    band = h[sel & (op > 0.3)]
    stats_before = {"p5_cm": float(np.percentile(band, 5) * 100), "p95_cm": float(np.percentile(band, 95) * 100)}
    means[sel] -= np.outer(h[sel], lplane.normal)
    L["means"] = torch.from_numpy(means.astype(np.float32))
    lout = Path(args.lidar_ckpt).with_name("best_flat.pt")
    torch.save(L, lout)
    report["lidar"] = {"plane": {"normal": lplane.normal.tolist(), "point": lplane.point.tolist(), "rms_m": lplane.meta["inlier_rms_m"]},
                       "offset_to_camera_plane_cm": float(plane.height(lplane.point) * 100), "n_projected": int(sel.sum()),
                       "band_before": stats_before, "out": str(lout)}
    print(f"LiDAR map: plane {lplane}, {plane.height(lplane.point)*100:+.1f} cm relative to the camera plane, "
          f"floor band before p5/p95 {stats_before['p5_cm']:+.1f}/{stats_before['p95_cm']:+.1f} cm, projected {sel.sum()} gaussians -> {lout}")
    json.dump(report, open(out / "report.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="splatfacto config.yml")
    ap.add_argument("--transforms", required=True)
    ap.add_argument("--floor-plane", required=True)
    ap.add_argument("--images", required=True, help="directory with the (raw) training frames")
    ap.add_argument("--camera-params", default=None, help="camera_params.json with K/D if the frames are raw")
    ap.add_argument("--out", required=True, help="output checkpoint directory")
    ap.add_argument("--lidar-ckpt", default=None, help="LiDAR gsplat best.pt to flatten as well")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--band", type=float, default=0.10, help="project gaussians within +-band of the plane [m]")
    ap.add_argument("--corridor", type=float, default=1.5, help="only within this distance of the camera path [m]")
    ap.add_argument("--depth-weight", type=float, default=2.0)
    ap.add_argument("--depth-tol", type=float, default=0.15, help="pixel counts as floor if rendered depth is within this of the plane [m]")
    ap.add_argument("--max-depth", type=float, default=6.0)
    ap.add_argument("--no-project", action="store_true", help="skip the hard projection, fine-tune only")
    ap.add_argument("--top", type=float, default=0.01,
                    help="only gaussians higher than this above the plane are projected (the part below carries the "
                         "floor's reflections and does not occlude objects); they are moved to --top-target and held there")
    ap.add_argument("--top-target", type=float, default=0.005)
    ap.add_argument("--lr-scale", type=float, default=0.25, help="learning rates relative to splatfacto's initial ones")
    ap.add_argument("--lidar-only", action="store_true", help="only flatten the LiDAR map")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = "cuda"

    from gsplat import rasterization
    from nerfstudio.data.utils.dataparsers_utils import get_train_eval_split_fraction

    cfg = Path(args.config)
    ckpt_dir = cfg.parent / "nerfstudio_models"
    ckpt_path = sorted(ckpt_dir.glob("step-*.ckpt"))[-1]
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    gp = ck["pipeline"]
    P = {k: gp[f"_model.gauss_params.{k}"].clone().float().to(dev).requires_grad_(True)
         for k in ("means", "scales", "quats", "opacities", "features_dc", "features_rest")}
    n = len(P["means"])
    meta = json.load(open(args.transforms))
    frames = meta["frames"]
    W_from_train = osp.nerfstudio_world_from_train(cfg.parent / "dataparser_transforms.json", args.transforms)
    M = np.linalg.inv(W_from_train)                      # world -> model
    Rm, sm, tm = osp.decompose_similarity(M)
    plane = osp.FloorPlane.load(args.floor_plane)
    plane_m = plane.transformed(M)                        # plane in model units
    C_w = np.array([np.array(f["transform_matrix"])[:3, 3] for f in frames])
    print(f"{n} gaussians, model scale {sm:.4f} units/m, plane {plane}")

    def band_stats(means_t):
        means_w = means_t.detach().cpu().numpy().astype(np.float64) @ W_from_train[:3, :3].T + W_from_train[:3, 3]
        from scipy.spatial import cKDTree
        d, _ = cKDTree(plane.project(C_w)[:, :2]).query(means_w[:, :2])
        h = plane.height(means_w)
        op = torch.sigmoid(P["opacities"]).detach().cpu().numpy().reshape(-1)
        sel = (d < 1.0) & (np.abs(h) < 0.10) & (op > 0.3)
        return {"n": int(sel.sum()), "p5_cm": float(np.percentile(h[sel], 5) * 100), "p50_cm": float(np.percentile(h[sel], 50) * 100),
                "p95_cm": float(np.percentile(h[sel], 95) * 100), "std_cm": float(h[sel].std() * 100)}, (d, h)

    report = {"checkpoint": str(ckpt_path), "band_before": band_stats(P["means"])[0]}
    print("floor band before:", report["band_before"])
    if args.lidar_only:
        args.steps = 0

    # ---- 1. hard projection of the floor band onto the plane
    hold_idx = None
    n_m = torch.tensor(plane_m.normal, dtype=torch.float32, device=dev)
    p0_m = torch.tensor(plane_m.point, dtype=torch.float32, device=dev)
    top_m = args.top_target * sm
    if not args.no_project:
        _, (d, h) = band_stats(P["means"])
        scales_w = np.exp(P["scales"].detach().cpu().numpy() - np.log(sm)).max(1)
        sel = (d < args.corridor) & (h > args.top) & (h < args.band) & (scales_w < 0.15)
        hold_idx = torch.from_numpy(np.where(sel)[0]).to(dev)
        with torch.no_grad():
            shift = torch.tensor((args.top_target - h[sel]) * sm, dtype=torch.float32, device=dev)[:, None] * n_m[None]
            P["means"].data[hold_idx] += shift
        report["n_projected"] = int(sel.sum())
        print(f"projected {sel.sum()} above-plane floor gaussians (h > {args.top*100:.0f} cm) down to {args.top_target*100:.1f} cm")

    def clamp_held():
        """Projected gradient: the moved gaussians may not rise above top_target again."""
        if hold_idx is None:
            return
        with torch.no_grad():
            hh = (P["means"].data[hold_idx] - p0_m) @ n_m
            over = hh > top_m
            if bool(over.any()):
                P["means"].data[hold_idx[over]] -= (hh[over] - top_m)[:, None] * n_m[None]

    # ---- data
    if args.lidar_only:
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        flatten_lidar(args, plane, C_w, report, out)
        return
    names = [Path(f["file_path"]).name for f in frames]
    import re
    frac = float(re.search(r"train_split_fraction:\s*([0-9.]+)", cfg.read_text()).group(1)) if re.search(r"train_split_fraction", cfg.read_text()) else 0.9
    i_train, i_eval = get_train_eval_split_fraction(sorted(names), frac)
    order = np.argsort(names)                         # split is defined on sorted names
    i_train, i_eval = order[np.array(i_train)], order[np.array(i_eval)]
    t0 = time.time()
    images = load_images(frames, args.images, args.camera_params)
    print(f"{len(images)} images loaded in {time.time()-t0:.0f} s ({len(i_train)} train / {len(i_eval)} eval frames in the split)")
    i_train = np.array([i for i in i_train if i in images]); i_eval = np.array([i for i in i_eval if i in images])
    fx, fy, cx, cy = float(meta["fl_x"]), float(meta["fl_y"]), float(meta.get("cx", meta["w"] / 2)), float(meta.get("cy", meta["h"] / 2))
    Wpx, Hpx = int(meta["w"]), int(meta["h"])
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    K_t = torch.tensor(K, dtype=torch.float32, device=dev)[None]

    def cam(i):
        c2w_w = np.array(frames[i]["transform_matrix"])
        c2w_m = M @ (np.vstack([c2w_w[:3, :4], [0, 0, 0, 1]]) if c2w_w.shape[0] == 3 else c2w_w)
        c2w_m[:3, :3] /= sm                                 # rotation stays orthonormal
        vm = viewmat_from_c2w(c2w_m)
        return vm, torch.tensor(vm, dtype=torch.float32, device=dev)[None]

    def render(vm_t, mode="RGB"):
        img, alpha, _ = rasterization(means=P["means"], quats=P["quats"], scales=torch.exp(P["scales"]),
                                      opacities=torch.sigmoid(P["opacities"]).squeeze(-1),
                                      colors=torch.cat([P["features_dc"][:, None], P["features_rest"]], 1),
                                      viewmats=vm_t, Ks=K_t, width=Wpx, height=Hpx, sh_degree=3, render_mode=mode)
        return img[0], alpha[0, ..., 0]

    def psnr_eval():
        vals = []
        with torch.no_grad():
            for i in i_eval[::max(1, len(i_eval) // 30)]:
                _, vm_t = cam(i)
                img, _ = render(vm_t)
                gt = torch.from_numpy(images[i]).to(dev).float() / 255
                mse = ((img.clamp(0, 1) - gt) ** 2).mean().item()
                vals.append(10 * np.log10(1 / max(mse, 1e-10)))
        return float(np.mean(vals))

    if args.no_project:
        report["psnr_eval_original"] = psnr_eval()
        print(f"held-out PSNR of the original checkpoint: {report['psnr_eval_original']:.2f} dB")
    report["psnr_eval_before_finetune"] = psnr_eval()
    print(f"held-out PSNR after projection, before fine-tune: {report['psnr_eval_before_finetune']:.2f} dB")

    # ---- 2. fine-tune
    lrs = {k: v * args.lr_scale for k, v in {"means": 1.6e-4 * sm, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2,
                                             "features_dc": 2.5e-3, "features_rest": 1.25e-4}.items()}
    print("learning rates:", {k: f"{v:.2e}" for k, v in lrs.items()})
    opt = torch.optim.Adam([{"params": [P[k]], "lr": lr} for k, lr in lrs.items()], eps=1e-15)
    plane_model = (plane_m.normal, plane_m.point)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        i = int(rng.choice(i_train))
        vm, vm_t = cam(i)
        out, alpha = render(vm_t, "RGB+ED")
        rgb, depth = out[..., :3], out[..., 3]
        gt = torch.from_numpy(images[i]).to(dev).float() / 255
        l_photo = 0.8 * (rgb - gt).abs().mean() + 0.2 * ssim_loss(rgb.clamp(0, 1), gt)
        zp = plane_depth(vm, K, Wpx, Hpx, plane_model)          # model units
        zp_t = torch.from_numpy(np.nan_to_num(zp, nan=-1.0)).to(dev).float()
        with torch.no_grad():
            floor = (zp_t > 0) & (zp_t < args.max_depth * sm) & ((depth - zp_t).abs() < args.depth_tol * sm) & (alpha > 0.5)
        l_depth = ((depth - zp_t).abs()[floor].mean() / sm) if floor.any() else torch.zeros((), device=dev)
        loss = l_photo + args.depth_weight * l_depth
        if not torch.isfinite(loss):
            print(f"step {step}: non-finite loss, skipped")
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        for k in P:
            g = P[k].grad
            if g is not None:
                torch.nan_to_num_(g, nan=0.0, posinf=0.0, neginf=0.0)
        opt.step()
        with torch.no_grad():
            P["quats"].data /= P["quats"].data.norm(dim=1, keepdim=True).clamp(min=1e-8)
            P["scales"].data.clamp_(min=np.log(1e-5), max=np.log(5.0))
        clamp_held()
        if step % 100 == 0:
            print(f"step {step:5d}  photo {l_photo.item():.4f}  floor depth L1 {l_depth.item()*100:.2f} cm  ({floor.float().mean().item()*100:.0f} % floor px)  {(time.time()-t0)/step*1000:.0f} ms/step")
        if step % args.eval_every == 0 or step == args.steps:
            print(f"   held-out PSNR {psnr_eval():.2f} dB")
    with torch.no_grad():
        finite = torch.ones(n, dtype=torch.bool, device=dev)
        for k in P:
            finite &= torch.isfinite(P[k].detach()).flatten(1).all(1) if P[k].dim() > 1 else torch.isfinite(P[k].detach())
        if not bool(finite.all()):
            print(f"dropping {int((~finite).sum())} gaussians with non-finite parameters")
            for k in P:
                P[k] = torch.nn.Parameter(P[k].detach()[finite])
            report["n_dropped_nonfinite"] = int((~finite).sum())

    # ---- write the checkpoint first (stats below must not lose the result)
    out = Path(args.out)
    (out / "nerfstudio_models").mkdir(parents=True, exist_ok=True)
    for k in P:
        gp[f"_model.gauss_params.{k}"] = P[k].detach().cpu().to(gp[f"_model.gauss_params.{k}"].dtype)
    torch.save(ck, out / "nerfstudio_models" / ckpt_path.name)
    shutil.copy(cfg, out / "config.yml")
    shutil.copy(cfg.parent / "dataparser_transforms.json", out / "dataparser_transforms.json")
    report["out"] = str(out)
    print(f"wrote {out}/nerfstudio_models/{ckpt_path.name}")
    report["psnr_eval_after"] = psnr_eval()
    report["band_after"] = band_stats(P["means"])[0]
    print("floor band after:", report["band_after"], f"PSNR {report['psnr_eval_after']:.2f} dB")
    json.dump(report, open(out / "report.json", "w"), indent=1)

    # ---- 3. LiDAR map
    if args.lidar_ckpt:
        flatten_lidar(args, plane, C_w, report, out)
    json.dump(report, open(out / "report.json", "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
