#!/usr/bin/env python
"""Visualise benchmark results: top-down view and ego-camera overlay of history, GT and planned trajectories.

    python visualize.py --run results/alpamayo15_flashdrive_20261019_104613 --idx 40 60
    python visualize.py --run results/<run> --every 5 --video          # PNG per sample + MP4 over the lap
    python visualize.py --run results/<a> --run results/<b> --idx 40   # compare plans of several runs

Left: top-down in the ego frame at t0 (car up, left = left). Right: last input frame with GT and plan
projected onto the ground plane (pinhole model from the sim scenario; distortion ignored).
Colours: blue = history, green = ground truth, red = plan of sample 0, grey = other K samples,
orange = best-of-K. Dots mark 1-s steps. Needs only numpy, matplotlib, pillow, pyyaml (no GPU).
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bench.dataset import build_samples  # noqa: E402

CAR_LEN, CAR_WID = 0.42, 0.26  # m (scenario: vehicle normalised to 42 cm; collision footprint 26 cm)
COLORS = ["#d62728", "#9467bd", "#8c564b", "#e377c2"]  # plan colours per run


# ----------------------------------------------------------------------------- camera model
class Camera:
    def __init__(self, scenario: dict, img_w: int, img_h: int, ground_z: float):
        cam, mount = scenario["camera"], scenario["sensors"]["camera"]
        s = img_w / cam["width"]
        assert abs(s - img_h / cam["height"]) < 1e-6, "image aspect differs from the scenario camera"
        self.fx, self.fy, self.cx, self.cy = cam["fx"] * s, cam["fy"] * s, cam["cx"] * s, cam["cy"] * s
        x, y, z, w = mount["quaternion"]
        self.R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                           [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                           [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])  # optical RDF -> vehicle FLU
        self.t = np.array(mount["translation"])
        self.ground_z = ground_z

    def project(self, xy: np.ndarray):
        """Ego-frame xy (N,2) on the ground plane -> pixel (N,2); NaN where behind the camera."""
        p = np.column_stack([xy, np.full(len(xy), self.ground_z)])
        pc = (p - self.t) @ self.R  # R^T (p - t)
        out = np.full((len(xy), 2), np.nan)
        ok = pc[:, 2] > 0.05
        out[ok, 0] = self.fx * pc[ok, 0] / pc[ok, 2] + self.cx
        out[ok, 1] = self.fy * pc[ok, 1] / pc[ok, 2] + self.cy
        return out


# ----------------------------------------------------------------------------- data
def load_run(run_dir: Path):
    d = np.load(run_dir / "predictions.npz")
    cfg = yaml.safe_load((run_dir / "config.snapshot.yaml").read_text())
    meta = json.loads((run_dir / "summary.json").read_text())["meta"]
    return {"dir": run_dir, "name": meta["model_name"] + (f" ({cfg['run_tag']})" if cfg.get("run_tag") else ""),
            "idx": list(map(int, d["idx"])), "t0": d["t0"], "pred": d["pred_xy"], "gt": d["gt_xy"], "hist": d["hist_xy"],
            "cfg": cfg, "period": meta["period_s"]}


def rebuild_samples(run: dict):
    dc = run["cfg"]["data"]
    rec = Path(dc["recording"])
    rec = rec if rec.is_absolute() else (HERE / rec)
    samples = build_samples(rec, num_frames=int(dc.get("num_frames", 4)), period_s=run["period"],
                            max_samples=dc.get("max_samples"), min_speed=float(dc.get("min_speed", 0.0)))
    return samples


# ----------------------------------------------------------------------------- drawing
def _line(ax, xy, color, lw, ls="-", label=None, alpha=1.0, dots=True, z=3):
    ax.plot(xy[:, 0], xy[:, 1], ls, color=color, lw=lw, alpha=alpha, label=label, zorder=z, solid_capstyle="round")
    if dots:
        ax.plot(xy[9::10, 0], xy[9::10, 1], "o", color=color, ms=3.5, alpha=alpha, zorder=z)


def plan_errors(p, gt):
    """ADE per plan sample (K,) over the waypoints both have: models plan 64 (6.4 s) or 50 (Qwen-Drive, 5 s)."""
    t = min(p.shape[1], gt.shape[0])
    return np.linalg.norm(p[:, :t] - gt[None, :t], axis=-1).mean(-1)


def draw_topdown(ax, hist, gt, plans, names, ade_txt):
    """Plot with x (forward) up and y (left) to the left, i.e. horizontal axis = -y."""
    T = lambda a: np.column_stack([-a[:, 1], a[:, 0]])
    ax.add_patch(Rectangle((-CAR_WID / 2, -CAR_LEN / 2), CAR_WID, CAR_LEN, fc="#444", ec="k", zorder=5))
    ax.annotate("", xy=(0, CAR_LEN / 2 + 0.15), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="w"), zorder=6)
    ax.plot(*T(hist).T, "-o", color="#1f77b4", lw=2, ms=3, label="history 1.5 s", zorder=3)
    _line(ax, T(gt), "#2ca02c", 2.5, label="ground truth")
    for r, (p, name) in enumerate(zip(plans, names)):
        c = COLORS[r % len(COLORS)]
        if r == 0:
            gts = plan_errors(p, gt)
            for j in range(1, p.shape[0]):
                _line(ax, T(p[j]), "grey", 1.0, alpha=0.5, dots=False, z=2, label="other samples" if j == 1 else None)
            b = int(gts.argmin())
            if b != 0:
                _line(ax, T(p[b]), "#ff7f0e", 1.8, ls="--", label="best of K", z=3, dots=False)
        _line(ax, T(p[0]), c, 2.5, label=f"plan: {name}")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlabel("left  <-  y [m]  ->  right")
    ax.set_ylabel("x forward [m]")
    ax.set_title(ade_txt, fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=7, frameon=False)
    pts = np.vstack([T(hist), T(gt)] + [T(p[0]) for p in plans])
    pad = 1.0
    ax.set_xlim(pts[:, 0].min() - pad, pts[:, 0].max() + pad)
    ax.set_ylim(min(pts[:, 1].min(), -1) - pad, pts[:, 1].max() + pad)


def draw_camera(ax, img, cam, hist, gt, plans, names):
    ax.imshow(img)
    for xy, color, lw, label in [(hist, "#1f77b4", 3, "history"), (gt, "#2ca02c", 3, "ground truth")] + \
            [(p[0], COLORS[r % len(COLORS)], 3, f"plan: {n}") for r, (p, n) in enumerate(zip(plans, names))]:
        uv = cam.project(xy)
        ok = ~np.isnan(uv[:, 0])
        inside = ok & (uv[:, 0] > -50) & (uv[:, 0] < img.shape[1] + 50) & (uv[:, 1] > -50) & (uv[:, 1] < img.shape[0] + 50)
        seg = np.where(inside, uv[:, 0], np.nan)
        ax.plot(seg, np.where(inside, uv[:, 1], np.nan), "-", color=color, lw=lw, label=label)
        ticks = np.arange(9, len(xy), 10)
        ax.plot(seg[ticks], np.where(inside, uv[:, 1], np.nan)[ticks], "o", color=color, ms=5)
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.axis("off")
    ax.set_title(f"ego camera (last input frame), ground plane z={cam.ground_z:+.3f} m", fontsize=10)
    ax.legend(loc="lower left", fontsize=7)


def render(sample, i_in_run, runs, cam, out_png: Path):
    hist, gt = sample.hist_xyz[:, :2], sample.fut_xyz[:, :2]
    plans = [r["pred"][r["pos"][i_in_run]] for r in runs]
    names = [r["name"] for r in runs]
    stats = []
    for p, n in zip(plans, names):
        d = plan_errors(p, gt)
        horizon = min(p.shape[1], gt.shape[0]) / 10
        stats.append(f"{n}: ADE {d[0]:.2f} / minADE {d.min():.2f} m" + (f" ({horizon:g} s)" if p.shape[1] < gt.shape[0] else ""))
    title = f"t0 = {sample.t0:.2f} s, speed {sample.speed:.2f} m/s | " + " | ".join(stats)
    img = np.asarray(Image.open(sample.frame_paths[-1]).convert("RGB"))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw={"width_ratios": [1, 1.5]})
    draw_topdown(a1, hist, gt, plans, names, "top-down, ego frame at t0")
    draw_camera(a2, img, cam, hist, gt, plans, names)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=90)
    plt.close(fig)


def calibrate_ground(sample, scenario, zs, out_png: Path):
    """GT path projected with several ground heights on one frame: pick the z where it lies on the floor."""
    img = np.asarray(Image.open(sample.frame_paths[-1]).convert("RGB"))
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.imshow(img)
    for z, c in zip(zs, plt.cm.viridis(np.linspace(0, 1, len(zs)))):
        uv = Camera(scenario, img.shape[1], img.shape[0], z).project(sample.fut_xyz[:, :2])
        inside = (uv[:, 0] > -50) & (uv[:, 0] < img.shape[1] + 50) & (uv[:, 1] > -50) & (uv[:, 1] < img.shape[0] + 50)
        ax.plot(np.where(inside, uv[:, 0], np.nan), np.where(inside, uv[:, 1], np.nan), "-", color=c, lw=2,
                label=f"ground z = {z:+.2f} m")
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.axis("off")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title(f"GT path for t0 = {sample.t0:.2f} s with different ground heights (vehicle frame)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=90)
    plt.close(fig)


def make_video(pngs: list[Path], out: Path, fps: int) -> Path | None:
    try:
        import imageio.v2 as imageio

        with imageio.get_writer(str(out), fps=fps, macro_block_size=1) as w:
            for p in pngs:
                w.append_data(imageio.imread(p))
        return out
    except Exception as exc:  # noqa: BLE001 - fall back to ffmpeg binary, then GIF
        if shutil.which("ffmpeg"):
            lst = out.with_suffix(".txt")
            lst.write_text("".join(f"file '{p.resolve()}'\nduration {1 / fps}\n" for p in pngs))
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                            "-pix_fmt", "yuv420p", str(out)], check=True)
            return out
        gif = out.with_suffix(".gif")
        ims = [Image.open(p).convert("P") for p in pngs]
        ims[0].save(gif, save_all=True, append_images=ims[1:], duration=int(1000 / fps), loop=0)
        print(f"No mp4 writer ({exc}); wrote GIF instead.")
        return gif


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="results/<run>; repeat to compare runs")
    sel = ap.add_mutually_exclusive_group()
    sel.add_argument("--idx", type=int, nargs="+", help="sample indices (idx column of per_sample.csv)")
    sel.add_argument("--every", type=int, default=5, help="every n-th sample (default 5)")
    ap.add_argument("--video", action="store_true", help="also write an MP4 over the selected samples")
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--ground-z", type=float, default=-0.15,
                    help="height of the trajectory plane in the vehicle frame [m]; tune until GT lies on the floor")
    ap.add_argument("--calibrate", type=float, nargs="+", metavar="Z",
                    help="instead of rendering: overlay the GT with these ground heights (e.g. -0.05 -0.12 -0.2)")
    ap.add_argument("--scenario", default=str(HERE.parent / "RoboRacer-3D-Isaac-Sim/configs/racing_142014.json"))
    ap.add_argument("--out", help="output folder (default: <first run>/viz)")
    args = ap.parse_args()

    runs = [load_run(Path(r)) for r in args.run]
    for r in runs:
        r["pos"] = {i: k for k, i in enumerate(r["idx"])}
    samples = rebuild_samples(runs[0])
    common = [i for i in runs[0]["idx"] if all(i in r["pos"] for r in runs)]
    for r in runs[1:]:
        assert abs(samples[common[0]].t0 - r["t0"][r["pos"][common[0]]]) < 1e-3, "runs use different samples"
    assert abs(samples[common[0]].t0 - runs[0]["t0"][runs[0]["pos"][common[0]]]) < 1e-3, \
        "rebuilt samples do not match the run (data folder or config changed?)"
    chosen = [i for i in args.idx if i in common] if args.idx else common[:: args.every]
    if not chosen:
        raise SystemExit("No matching samples.")

    scenario = json.loads(Path(args.scenario).read_text())
    if args.calibrate:
        out_dir = Path(args.out) if args.out else runs[0]["dir"] / "viz"
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in chosen:
            calibrate_ground(samples[i], scenario, args.calibrate, out_dir / f"ground_calibration_{i:03d}.png")
        print(f"calibration images in {out_dir}")
        return
    w, h = Image.open(samples[chosen[0]].frame_paths[-1]).size
    cam = Camera(scenario, w, h, args.ground_z)

    out_dir = Path(args.out) if args.out else runs[0]["dir"] / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    pngs = []
    for i in chosen:
        png = out_dir / f"sample_{i:03d}.png"
        render(samples[i], i, runs, cam, png)
        pngs.append(png)
    print(f"{len(pngs)} images in {out_dir}")
    if args.video:
        print("video:", make_video(pngs, out_dir / "lap.mp4", args.fps))


if __name__ == "__main__":
    main()
