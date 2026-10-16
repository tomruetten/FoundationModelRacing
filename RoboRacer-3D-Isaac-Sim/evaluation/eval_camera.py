"""Camera 3DGS (splatfacto) evaluation against the recorded frames.

For every available checkpoint (2026-07-14 recording + newer sessions):
  * reproduce nerfstudio's train/eval split (frames sorted by file name, 90/10 fraction)
  * render every eval frame and every k-th train frame with the same renderer the
    Isaac Sim render server uses (SplatCameraRenderer)
  * PSNR / SSIM / LPIPS per frame, summary statistics, distribution along the lap
  * qualitative panels (truth | render | abs. error) for best / median / worst eval frame
  * novel-view probe: renders at poses the car never visited (lateral offset, higher
    camera) with a sharpness proxy, to show how far off-trajectory the model stays usable
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import torch

from .common import DATA_LOCAL, REPO, Report, env, load_module, log, raw_session_dir

TRAIN_STRIDE = 10
NOVEL_OFFSETS = [("on trajectory", np.zeros(3)), ("0.3 m left", np.array([-0.3, 0, 0])),
                 ("0.3 m right", np.array([0.3, 0, 0])), ("0.5 m up", np.array([0, 0.5, 0])),
                 ("1.0 m back", np.array([0, 0, 1.0]))]  # camera-frame offsets (x right, y up, z back)


def discover_models():
    models = []
    cfg = Path(env("CAMERA_3DGS_CONFIG", DATA_LOCAL / "New1/checkpoints/camera_3dgs/config.yml"))
    if cfg.exists():
        # images_real are the RAW camera frames; training used cv2.undistort with the factory
        # calibration (camera_params.json), so the truth has to be undistorted the same way.
        und = None
        cp = DATA_LOCAL / "New1/camera_params.json"
        if cp.exists():
            c = json.load(open(cp)); i = c["intrinsics"]
            und = dict(K=np.array([[i["fx"], 0, i["cx"]], [0, i["fy"], i["cy"]], [0, 0, 1]]), D=np.array(c["distortion_coefficients"]["D"]))
        models.append(dict(name="2026-07-14 recording (splatfacto 50k)", config=cfg,
                           transforms=DATA_LOCAL / "New1/dataset/transforms.json",
                           image_dirs=[DATA_LOCAL / "New1/dataset/images", DATA_LOCAL / "New1/dataset/images_real"],
                           undistort=und, reference_psnr=27.61))
    for sd in sorted((DATA_LOCAL / "sessions").glob("session_*")):
        cfgs = sorted(sd.glob("outputs/*/splatfacto/*/config.yml"))
        tj = sd / "dataset/colmap_dataset_full/transforms.json"
        if not cfgs or not tj.exists():
            continue
        token = re.search(r"(\d{8}_\d{6})", sd.name).group(1)
        raw = raw_session_dir(token)
        dirs = [sd / "dataset"]
        if raw:
            dirs.append(raw / "dataset")
        ref = None
        ev = sd / "eval_colmap_full_ptcloud.json"
        if ev.exists():
            try:
                ref = json.load(open(ev)).get("results", {}).get("psnr")
            except Exception:  # noqa: BLE001
                ref = None
        models.append(dict(name=f"session {token} (splatfacto)", config=cfgs[-1], transforms=tj, image_dirs=dirs, reference_psnr=ref))
    return models


def resolve_image(file_path: str, image_dirs):
    fp = Path(file_path)
    for d in image_dirs:
        for cand in (d / fp, d / fp.name, d / "images_full" / fp.name, d / "images_real" / fp.name):
            if cand.exists():
                return cand
    return None


def split_fraction_from_config(cfg_path: Path) -> float:
    m = re.search(r"train_split_fraction:\s*([0-9.]+)", cfg_path.read_text())
    return float(m.group(1)) if m else 0.9


def sharpness(img: np.ndarray) -> float:
    import cv2
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def run(report: Report, quick: bool = False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from nerfstudio.data.utils.dataparsers_utils import get_train_eval_split_fraction

    device = "cuda" if torch.cuda.is_available() else "cpu"
    psnr_m = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_m = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_m = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)
    scr = load_module(REPO / "simulation/isaac_sim/splat_camera_renderer.py", "splat_camera_renderer_eval")

    report.h(2, "Camera model (3D Gaussian Splatting)")
    report.p("Renders are produced by the same renderer the Isaac Sim camera server uses and compared "
             "with the recorded, undistorted frames. Eval frames are the ones nerfstudio held out during "
             "training (10 %, deterministic split); train frames are shown for reference. LPIPS uses AlexNet "
             "(lower is better).")

    summary_rows = []
    for mi, m in enumerate(discover_models()):
        log(f"camera model: {m['name']}")
        try:
            r = scr.SplatCameraRenderer(config_path=m["config"])
        except Exception as e:  # noqa: BLE001
            report.note(f"{m['name']}: could not load ({e})"); continue
        meta = json.load(open(m["transforms"]))
        frames = sorted(meta["frames"], key=lambda f: f["file_path"])
        names = [f["file_path"] for f in frames]
        i_train, i_eval = get_train_eval_split_fraction(names, split_fraction_from_config(m["config"]))
        todo = [(int(i), "eval") for i in i_eval] + [(int(i), "train") for i in i_train[::TRAIN_STRIDE]]
        if quick:
            todo = todo[::6]
        rows, missing, t_render = [], 0, []
        best = {}
        for idx, split in todo:
            f = frames[idx]
            img_path = resolve_image(f["file_path"], m["image_dirs"])
            if img_path is None:
                missing += 1; continue
            gt = np.asarray(Image.open(img_path).convert("RGB"))
            if m.get("undistort") and "images_real" in str(img_path):
                import cv2
                gt = cv2.undistort(gt, m["undistort"]["K"], m["undistort"]["D"])
            c2w = np.array(f["transform_matrix"])
            t0 = time.time(); rgb = r.render(c2w, already_in_model_frame=False); t_render.append(time.time() - t0)
            if rgb.shape != gt.shape:
                gt = np.asarray(Image.fromarray(gt).resize((rgb.shape[1], rgb.shape[0])))
            a = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().div(255).to(device)
            b = torch.from_numpy(gt).permute(2, 0, 1)[None].float().div(255).to(device)
            ps = float(psnr_m(a, b)); ss = float(ssim_m(a, b)); lp = float(lpips_m(a, b))
            rows.append(dict(idx=idx, split=split, file=f["file_path"], psnr=ps, ssim=ss, lpips=lp, path=str(img_path)))
            if split == "eval":
                best[idx] = (ps, rgb, gt)
        if not rows:
            report.note(f"{m['name']}: no images found for {m['transforms']} (looked in {[str(d) for d in m['image_dirs']]})")
            continue
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_csv(report.out_dir / f"camera_{mi}_per_frame.csv", index=False)
        ev = df[df.split == "eval"]; tr = df[df.split == "train"]
        stats = {k: {"psnr_mean": float(d.psnr.mean()), "psnr_median": float(d.psnr.median()), "psnr_p10": float(d.psnr.quantile(0.1)),
                     "ssim_mean": float(d.ssim.mean()), "lpips_mean": float(d.lpips.mean()), "n": int(len(d))}
                 for k, d in (("eval", ev), ("train", tr)) if len(d)}
        report.put("camera", m["name"], value=dict(stats=stats, reference_psnr=m["reference_psnr"],
                                                   render_ms=float(np.mean(t_render) * 1000), missing_images=missing, config=str(m["config"])))
        summary_rows.append([m["name"], stats["eval"]["n"], stats["eval"]["psnr_mean"], stats["eval"]["psnr_p10"], stats["eval"]["ssim_mean"],
                             stats["eval"]["lpips_mean"], stats.get("train", {}).get("psnr_mean", float("nan")),
                             m["reference_psnr"] if m["reference_psnr"] else float("nan"), float(np.mean(t_render) * 1000)])

        # per-frame curve along the lap
        fig, ax = plt.subplots(figsize=(11, 3.2))
        ax.plot(tr.idx, tr.psnr, ".", ms=3, alpha=.6, label="train frames")
        ax.plot(ev.idx, ev.psnr, "o", ms=4, label="held-out frames")
        ax.set_xlabel("frame index (sorted by file name = time)"); ax.set_ylabel("PSNR [dB]"); ax.grid(alpha=.3); ax.legend(fontsize=8)
        ax.set_title(m["name"], fontsize=9)
        report.savefig(fig, f"camera_{mi}_psnr_along_lap.png", f"{m['name']}: PSNR along the recording. Dips mark parts of the track the model reconstructs worse.")

        # qualitative panels
        if best:
            order = sorted(best, key=lambda k: best[k][0])
            picks = [("worst", order[0]), ("median", order[len(order) // 2]), ("best", order[-1])]
            fig, axes = plt.subplots(3, 3, figsize=(13, 7.2))
            for row, (lab, k) in enumerate(picks):
                ps, rgb, gt = best[k]
                err = np.abs(rgb.astype(np.int16) - gt.astype(np.int16)).mean(axis=2)
                axes[row, 0].imshow(gt); axes[row, 0].set_title(f"{lab} held-out frame: truth", fontsize=8)
                axes[row, 1].imshow(rgb); axes[row, 1].set_title(f"render  PSNR {ps:.1f} dB", fontsize=8)
                im = axes[row, 2].imshow(err, cmap="inferno", vmin=0, vmax=64); axes[row, 2].set_title("abs. error (0-64)", fontsize=8)
                for a_ in axes[row]: a_.axis("off")
            fig.tight_layout()
            report.savefig(fig, f"camera_{mi}_panels.png", f"{m['name']}: worst / median / best held-out frames.")

        # novel-view probe
        try:
            probe_idx = [int(i) for i in np.linspace(0, len(i_eval) - 1, 3).round()]
            fig, axes = plt.subplots(len(probe_idx), len(NOVEL_OFFSETS), figsize=(3.2 * len(NOVEL_OFFSETS), 2.1 * len(probe_idx)))
            sharp_rows = []
            for rr, pi in enumerate(probe_idx):
                c2w = np.array(frames[int(i_eval[pi])]["transform_matrix"])
                base_sharp = None
                for cc, (lab, off) in enumerate(NOVEL_OFFSETS):
                    c2 = c2w.copy(); c2[:3, 3] = c2w[:3, 3] + c2w[:3, :3] @ off
                    img = r.render(c2, already_in_model_frame=False)
                    s = sharpness(img); base_sharp = base_sharp or s
                    sharp_rows.append([lab, int(i_eval[pi]), s, s / base_sharp])
                    axes[rr, cc].imshow(img); axes[rr, cc].set_title(f"{lab}  sharp {s/base_sharp:.2f}x", fontsize=7); axes[rr, cc].axis("off")
            fig.tight_layout()
            report.savefig(fig, f"camera_{mi}_novel_views.png", f"{m['name']}: renders at poses the car never visited. 'sharp' = Laplacian variance relative to the on-trajectory render: well below 1 = blur, well above 1 = high-frequency floaters/streaks; both mean the model is being used outside the poses it learned from. Judge the images.")
            agg = {}
            for lab, _, s, rel in sharp_rows:
                agg.setdefault(lab, []).append(rel)
            report.put("camera", m["name"], "novel_view_relative_sharpness", value={k: float(np.mean(v)) for k, v in agg.items()})
        except Exception as e:  # noqa: BLE001
            report.note(f"novel-view probe failed: {e}")
        del r; torch.cuda.empty_cache()

    report.h(3, "Summary (held-out frames)")
    report.table(["model", "n eval", "PSNR mean [dB]", "PSNR p10", "SSIM", "LPIPS", "PSNR train frames", "reported PSNR", "render [ms]"], summary_rows)
    report.p("Reported PSNR = number in the original package / ns-eval logs, for a consistency check. "
             "The p10 column (10th percentile) tells how bad the worst tenth of the lap looks; a large gap "
             "between train and eval PSNR would indicate overfitting to the training frames.")
    return True
