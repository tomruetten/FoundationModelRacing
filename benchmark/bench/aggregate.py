"""Aggregate the summaries of several seeds: per-seed values, mean and standard deviation over seeds."""

import json
import math
from pathlib import Path

import numpy as np

LAT_STAGES = ["vision", "prefill", "decode", "diffusion", "other", "total"]


def _mean_std(vals: list[float]) -> tuple[float, float]:
    a = np.array([v for v in vals if v == v], dtype=np.float64)
    if a.size == 0:
        return math.nan, math.nan
    return float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else math.nan


def aggregate(summaries: dict[int, dict], out_dir: Path, meta: dict) -> dict:
    """`summaries`: seed -> summary dict from write_report. Writes aggregate.json and aggregate.md."""
    seeds = list(summaries)
    first = summaries[seeds[0]]
    rows = {}
    for key in first["accuracy_m"]:
        rows[f"{key} (m)"] = [summaries[s]["accuracy_m"][key]["mean"] for s in seeds]
    for stage in LAT_STAGES:
        rows[f"latency {stage} p50 (ms)"] = [summaries[s]["latency_ms"][stage]["p50"] for s in seeds]
    rows["latency total p90 (ms)"] = [summaries[s]["latency_ms"]["total"]["p90"] for s in seeds]
    rows["latency total mean (ms)"] = [summaries[s]["latency_ms"]["total"]["mean"] for s in seeds]

    table = {}
    for name, vals in rows.items():
        m, sd = _mean_std(vals)
        table[name] = {"per_seed": dict(zip(map(str, seeds), vals)), "mean": m, "std_over_seeds": sd}
    result = {"meta": meta, "seeds": seeds, "metrics": table}

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "aggregate.json").write_text(json.dumps(result, indent=2))
    head = "| Metric | " + " | ".join(f"seed {s}" for s in seeds) + " | mean | std over seeds |"
    md = [f"# {meta['run_name']}: {len(seeds)} seeds", "",
          f"Model: `{meta['model_name']}` · K={meta['num_samples']} · scored samples per seed: "
          f"{first['n_scored']} · prompt: `{meta.get('task_prompt')}`", "",
          "Means per seed over the scored samples; mean and std (n-1) over seeds. The std over seeds is the run-to-run "
          "noise of sampling; differences between models/prompts well below it are not reliable.", "",
          head, "|" + "---|" * (len(seeds) + 3)]
    for name, t in table.items():
        vals = " | ".join(f"{v:.3f}" if v == v else "-" for v in t["per_seed"].values())
        sd = f"{t['std_over_seeds']:.3f}" if t["std_over_seeds"] == t["std_over_seeds"] else "-"
        md.append(f"| {name} | {vals} | {t['mean']:.3f} | {sd} |")
    (out_dir / "aggregate.md").write_text("\n".join(md) + "\n")
    return result
