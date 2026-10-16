#!/usr/bin/env python
"""One-shot model benchmark on a recorded F1TENTH sim run.

    python run_benchmark.py --config config.yaml
    python run_benchmark.py --config config.yaml --model configs/models/alpamayo15_flashdrive.yaml
    python run_benchmark.py --config config.yaml --set num_samples=1 --set data.max_samples=20

Per sample: front camera frames + ego history + task prompt -> K trajectories, timed per stage
(vision / prefill / decode / diffusion) and scored against the driven future (ADE, minADE).
"""

import argparse
import copy
import datetime
import logging
import math
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from bench.adapters import create_adapter  # noqa: E402
from bench.dataset import build_samples  # noqa: E402
from bench.metrics import trajectory_metrics  # noqa: E402
from bench.report import write_report  # noqa: E402

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


def gpu_name() -> str:
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "unknown"


def run(cfg: dict, config_dir: Path) -> dict:
    mcfg = cfg["model_cfg"]
    task = cfg.get("task_prompt")
    K = int(cfg.get("num_samples", 6))
    horizons = list(cfg.get("horizons_s", [2.0, 4.0, 6.4]))
    warmup = int(cfg.get("warmup", 3))

    adapter = create_adapter(mcfg, task)
    adapter.load()

    dcfg = cfg["data"]
    period = adapter.required_period_s or float(dcfg.get("period_s", 0.5))
    if adapter.required_period_s and dcfg.get("period_s") not in (None, adapter.required_period_s):
        log.warning("%s needs consecutive windows; overriding data.period_s=%s with %s",
                    mcfg["name"], dcfg.get("period_s"), period)
    samples = build_samples(_resolve(dcfg["recording"], config_dir), num_frames=int(dcfg.get("num_frames", 4)),
                            period_s=period, max_samples=dcfg.get("max_samples"),
                            min_speed=float(dcfg.get("min_speed", 0.0)))
    if not samples:
        raise SystemExit("No samples could be built from the recording (check frames/ and trajectory.jsonl).")
    log.info("%d samples, period %.2f s, model %s", len(samples), period, mcfg["name"])

    rows, prev = [], None
    for i, s in enumerate(samples):
        if prev is None or s.episode != prev.episode or s.t0 - prev.t0 > 1.5 * period + 0.05:
            adapter.reset_stream()
        prev = s
        inputs = adapter.prepare(s, K)
        pred = adapter.predict(inputs, K)
        row = {"idx": i, "episode": s.episode, "t0": round(s.t0, 3), "speed_mps": round(s.speed, 3),
               "warmup": i < warmup, "has_plan": pred.xy is not None}
        for c, v in pred.stage_ms.items():
            row[f"lat_{c}"] = round(v, 3) if v == v else math.nan
        if pred.xy is not None:
            row.update(trajectory_metrics(pred.xy, s.fut_xyz[:, :2], horizons))
            row["reasoning_chars"] = len(pred.text)
        rows.append(row)
        log.info("[%d/%d] total %.0f ms%s", i + 1, len(samples), pred.stage_ms["total"],
                 f" | ADE {row['ade']:.2f} minADE {row['min_ade']:.2f} m" if pred.xy is not None else " | (no plan)")

    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    rows = [{k: r.get(k, math.nan) for k in keys} for r in rows]

    run_name = f"{mcfg['name']}_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    meta = {"run_name": run_name, "model_name": mcfg["name"], "checkpoint": mcfg.get("checkpoint"),
            "gpu": gpu_name(), "num_samples": K, "task_prompt": task, "period_s": period,
            "recording": str(dcfg["recording"]), "warmup": warmup, "horizons_s": horizons}
    out_dir = _resolve(cfg.get("output_dir", "results"), config_dir) / run_name
    summary = write_report(out_dir, rows, meta, horizons)
    (out_dir / "config.snapshot.yaml").write_text(
        yaml.safe_dump(copy.deepcopy(cfg), sort_keys=False, allow_unicode=True))
    log.info("Results: %s", out_dir)
    print((out_dir / "summary.md").read_text())
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--model", help="model YAML (overrides `model:` in the config)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", dest="overrides",
                    help="override a config key, e.g. num_samples=1, data.max_samples=20, model_cfg.cuda_graph=false")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config_path = Path(args.config).resolve()
    run(load_config(config_path, args.model, args.overrides), config_path.parent)


if __name__ == "__main__":
    main()
