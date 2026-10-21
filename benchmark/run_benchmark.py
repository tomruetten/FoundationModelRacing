#!/usr/bin/env python
"""One-shot model benchmark on a recorded F1TENTH sim run.

    python run_benchmark.py --config config.yaml
    python run_benchmark.py --config config.yaml --model configs/models/alpamayo15_flashdrive.yaml
    python run_benchmark.py --config config.yaml --set num_samples=1 --set data.max_samples=20
    python run_benchmark.py --seed 7                 # one run with a fixed seed (default: seed from config.yaml)
    python run_benchmark.py --seeds 1 2 3            # one run per seed, plus aggregate.md (mean/std over seeds)

Per sample: front camera frames + ego history + task prompt -> K trajectories, timed per stage
(vision / prefill / decode / diffusion) and scored against the driven future (ADE, minADE).
Randomness (reasoning tokens, diffusion noise) is seeded per sample from (seed, sample index).
"""

import argparse
import copy
import datetime
import json
import logging
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from bench.adapters import create_adapter  # noqa: E402
from bench.aggregate import aggregate  # noqa: E402
from bench.dataset import build_samples  # noqa: E402
from bench.metrics import trajectory_metrics  # noqa: E402
from bench.report import write_report  # noqa: E402
from bench.seeding import sample_seed, set_seed  # noqa: E402

log = logging.getLogger("benchmark")


def _resolve(path: str, base: Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (base / p)


def apply_overrides(cfg: dict, overrides: list[str]) -> None:
    for o in overrides:
        key, _, val = o.partition("=")
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(val)


def load_config(config_path: Path, model_override: str | None, overrides: list[str]) -> dict:
    cfg = yaml.safe_load(config_path.read_text())
    if model_override:
        cfg["model"] = model_override
    model_overrides = [o[len("model_cfg."):] for o in overrides if o.startswith("model_cfg.")]
    apply_overrides(cfg, [o for o in overrides if not o.startswith("model_cfg.")])
    model_file = _resolve(cfg["model"], config_path.parent)
    cfg["model_cfg"] = yaml.safe_load(model_file.read_text())
    cfg["model_cfg"].setdefault("name", model_file.stem)
    apply_overrides(cfg["model_cfg"], model_overrides)
    return cfg


def apply_adapter_defaults(cfg: dict) -> None:
    """Models without a free-text prompt (Qwen-Drive) run without `task_prompt`; the run is tagged `no_prompt`."""
    probe = create_adapter(cfg["model_cfg"], cfg.get("task_prompt"))
    if not probe.uses_task_prompt and cfg.get("task_prompt"):
        log.warning("%s takes no free-text prompt: task_prompt is ignored, run tag set to 'no_prompt'",
                    cfg["model_cfg"]["name"])
        cfg["task_prompt"], cfg["run_tag"] = None, "no_prompt"


def gpu_name() -> str:
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "unknown"


def run_seed(adapter, samples, cfg: dict, seed: int, period: float, run_name: str, out_dir: Path) -> dict:
    """One pass over all samples (chronological) with the given seed; writes the report into out_dir."""
    mcfg, dcfg = cfg["model_cfg"], cfg["data"]
    task = cfg.get("task_prompt")
    K = int(cfg.get("num_samples", 6))
    horizons = list(cfg.get("horizons_s", [2.0, 4.0, 6.4]))
    warmup = int(cfg.get("warmup", 3))

    rows, prev = [], None
    saved = {"idx": [], "t0": [], "pred_xy": [], "gt_xy": [], "hist_xy": []}
    reasoning = []
    for i, s in enumerate(samples):
        if prev is None or s.episode != prev.episode or s.t0 - prev.t0 > 1.5 * period + 0.05:
            adapter.reset_stream()
        prev = s
        inputs = adapter.prepare(s, K)
        set_seed(sample_seed(seed, i))
        pred = adapter.predict(inputs, K)
        row = {"idx": i, "episode": s.episode, "t0": round(s.t0, 3), "speed_mps": round(s.speed, 3),
               "warmup": i < warmup, "has_plan": pred.xy is not None}
        for c, v in pred.stage_ms.items():
            row[f"lat_{c}"] = round(v, 3) if v == v else math.nan
        if pred.xy is not None:
            row.update(trajectory_metrics(pred.xy, s.fut_xyz[:, :2], horizons))
            row["reasoning_chars"] = len(pred.text)
            saved["idx"].append(i)
            saved["t0"].append(s.t0)
            saved["pred_xy"].append(pred.xy.astype(np.float32))
            saved["gt_xy"].append(s.fut_xyz[:, :2])
            saved["hist_xy"].append(s.hist_xyz[:, :2])
            reasoning.append({"idx": i, "t0": round(s.t0, 3), "lat_decode_ms": row.get("lat_decode"),
                              "speed_mps": round(s.speed, 3), "text": pred.text})
        rows.append(row)
        log.info("[seed %s | %d/%d] total %.0f ms%s", seed, i + 1, len(samples), pred.stage_ms["total"],
                 f" | ADE {row['ade']:.2f} minADE {row['min_ade']:.2f} m" if pred.xy is not None else " | (no plan)")

    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    rows = [{k: r.get(k, math.nan) for k in keys} for r in rows]

    meta = {"run_name": run_name, "model_name": mcfg["name"], "checkpoint": mcfg.get("checkpoint"),
            "gpu": gpu_name(), "num_samples": K, "task_prompt": task, "period_s": period, "seed": seed,
            "recording": str(dcfg["recording"]), "warmup": warmup, "horizons_s": horizons}
    summary = write_report(out_dir, rows, meta, horizons)
    if saved["idx"]:
        # ego frame at t0 (x forward, y left), metres: pred_xy (N,K,64,2), gt_xy (N,64,2), hist_xy (N,16,2)
        np.savez_compressed(out_dir / "predictions.npz", **{k: np.array(v) for k, v in saved.items()})
        (out_dir / "reasoning.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in reasoning))
    snap = copy.deepcopy(cfg)
    snap["seed"] = seed
    snap.pop("seeds", None)
    (out_dir / "config.snapshot.yaml").write_text(yaml.safe_dump(snap, sort_keys=False, allow_unicode=True))
    log.info("Results: %s", out_dir)
    print((out_dir / "summary.md").read_text())
    return summary


def run(cfg: dict, config_dir: Path) -> dict:
    """Load the model once, then run one pass per seed. Returns the summary (1 seed) or the aggregate (several)."""
    mcfg = cfg["model_cfg"]
    seeds = [int(s) for s in cfg["seeds"]] if cfg.get("seeds") else [int(cfg.get("seed", 42))]

    adapter = create_adapter(mcfg, cfg.get("task_prompt"))
    adapter.load()

    dcfg = cfg["data"]
    period = adapter.required_period_s or float(dcfg.get("period_s", 0.5))
    if adapter.required_period_s and dcfg.get("period_s") not in (None, adapter.required_period_s):
        log.warning("%s needs consecutive windows; overriding data.period_s=%s with %s",
                    mcfg["name"], dcfg.get("period_s"), period)
    frame_dt = adapter.frame_spacing_s or float(dcfg.get("frame_dt_s", 0.1))
    samples = build_samples(_resolve(dcfg["recording"], config_dir), num_frames=int(dcfg.get("num_frames", 4)),
                            period_s=period, max_samples=dcfg.get("max_samples"),
                            min_speed=float(dcfg.get("min_speed", 0.0)), frame_dt_s=frame_dt)
    if not samples:
        raise SystemExit("No samples could be built from the recording (check frames/ and trajectory.jsonl).")
    log.info("%d samples, period %.2f s, frame spacing %.2f s, model %s, seeds %s", len(samples), period, frame_dt,
             mcfg["name"], seeds)

    tag = f"_{cfg['run_tag']}" if cfg.get("run_tag") else ""
    stamp = f"{datetime.datetime.now():%Y%m%d_%H%M%S}"
    results_dir = _resolve(cfg.get("output_dir", "results"), config_dir)

    if len(seeds) == 1:
        name = cfg.get("_run_name") or f"{mcfg['name']}{tag}_{stamp}"
        out_dir = Path(cfg["_out_dir"]) if cfg.get("_out_dir") else results_dir / name
        return run_seed(adapter, samples, cfg, seeds[0], period, name, out_dir)

    root_name = f"{mcfg['name']}{tag}_{len(seeds)}seeds_{stamp}"
    root = results_dir / root_name
    summaries = {}
    for seed in seeds:
        summaries[seed] = run_seed(adapter, samples, cfg, seed, period, f"{root_name}/seed_{seed}",
                                   root / f"seed_{seed}")
    meta = {"run_name": root_name, "model_name": mcfg["name"], "num_samples": int(cfg.get("num_samples", 6)),
            "task_prompt": cfg.get("task_prompt")}
    result = aggregate(summaries, root, meta)
    log.info("Aggregate: %s", root / "aggregate.md")
    print((root / "aggregate.md").read_text())
    return result


def run_seeds_in_processes(args, cfg: dict, config_dir: Path) -> dict | None:
    """One child process (= one model load) per seed; then aggregate. For models that cannot restart a stream.

    A failing seed (e.g. CUDA out of memory while loading on a shared GPU) does not stop the others; seeds that already
    have a summary.json in the run folder are skipped, so a failed run is completed with `--resume <run folder>`.
    """
    mcfg = cfg["model_cfg"]
    seeds = [int(s) for s in cfg["seeds"]]
    tag = f"_{cfg['run_tag']}" if cfg.get("run_tag") else ""
    if args.resume:
        root = Path(args.resume).resolve()
        root_name = root.name
    else:
        root_name = f"{mcfg['name']}{tag}_{len(seeds)}seeds_{datetime.datetime.now():%Y%m%d_%H%M%S}"
        root = _resolve(cfg.get("output_dir", "results"), config_dir) / root_name
    base = [sys.executable, str(Path(__file__).resolve()), "--config", str(Path(args.config).resolve())]
    if args.model:
        base += ["--model", args.model]
    for o in args.overrides:
        base += ["--set", o]
    failed = []
    for seed in seeds:
        if (root / f"seed_{seed}" / "summary.json").exists():
            log.info("Seed %s already done in %s, skipping", seed, root)
            continue
        cmd = base + ["--seed", str(seed), "--out-dir", str(root / f"seed_{seed}"),
                      "--run-name", f"{root_name}/seed_{seed}"]
        log.info("Seed %s in its own process: %s", seed, " ".join(cmd))
        if subprocess.run(cmd).returncode != 0:
            log.error("Seed %s failed; continuing with the other seeds", seed)
            failed.append(seed)
    summaries = {s: json.loads((root / f"seed_{s}" / "summary.json").read_text()) for s in seeds
                 if (root / f"seed_{s}" / "summary.json").exists()}
    if failed:
        log.error("Failed seeds: %s. Complete the run with: <same command> --resume %s", failed, root)
    if not summaries:
        return None
    first = next(iter(summaries.values()))["meta"]
    meta = {"run_name": root_name, "model_name": first["model_name"], "num_samples": first["num_samples"],
            "task_prompt": first["task_prompt"]}
    result = aggregate(summaries, root, meta)
    log.info("Aggregate over seeds %s: %s", sorted(summaries), root / "aggregate.md")
    print((root / "aggregate.md").read_text())
    if failed:
        sys.exit(1)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--model", help="model YAML (overrides `model:` in the config)")
    ap.add_argument("--seed", type=int, help="seed of a single run (overrides `seed:` in the config)")
    ap.add_argument("--seeds", type=int, nargs="+", help="run once per seed and write aggregate.md over all seeds")
    ap.add_argument("--resume", metavar="RUN_DIR",
                    help="complete an interrupted --seeds run of models that run one process per seed: "
                         "seeds with an existing summary.json in RUN_DIR are skipped")
    ap.add_argument("--out-dir", help=argparse.SUPPRESS)   # used by child processes of --seeds
    ap.add_argument("--run-name", help=argparse.SUPPRESS)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", dest="overrides",
                    help="override a config key, e.g. num_samples=1, data.max_samples=20, model_cfg.cuda_graph=false")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path, args.model, args.overrides)
    if args.seed is not None:
        cfg["seed"] = args.seed
        cfg.pop("seeds", None)
    if args.seeds:
        cfg["seeds"] = args.seeds
    apply_adapter_defaults(cfg)
    if args.out_dir:
        cfg["_out_dir"], cfg["_run_name"] = args.out_dir, args.run_name
    if len(cfg.get("seeds") or []) > 1 or args.resume:
        # Models that cannot restart a stream in one process (FlashDrive), or `model_cfg.separate_processes: true`
        probe = create_adapter(cfg["model_cfg"], cfg.get("task_prompt"))
        if cfg["model_cfg"].get("separate_processes") or not probe.supports_stream_restart:
            run_seeds_in_processes(args, cfg, config_path.parent)
            return
        if args.resume:
            raise SystemExit("--resume applies only to models that run one process per seed (e.g. FlashDrive).")
    run(cfg, config_path.parent)


if __name__ == "__main__":
    main()
