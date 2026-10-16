"""Write per-sample CSV, summary JSON and a Markdown table."""

import csv
import json
import math
from pathlib import Path

import numpy as np

LAT_COLS = ["vision", "prefill", "decode", "diffusion", "other", "total"]


def _stats(vals: list[float]) -> dict:
    a = np.array([v for v in vals if v == v], dtype=np.float64)
    if a.size == 0:
        return {"n": 0, "mean": math.nan, "std": math.nan, "p50": math.nan, "p90": math.nan}
    return {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std()), "p50": float(np.percentile(a, 50)),
            "p90": float(np.percentile(a, 90))}


def write_report(out_dir: Path, rows: list[dict], meta: dict, horizons: list[float]) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(out_dir / "per_sample.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    timed = [r for r in rows if not r["warmup"] and r["has_plan"]]
    summary = {
        "meta": meta,
        "n_samples_total": len(rows),
        "n_scored": len(timed),
        "latency_ms": {c: _stats([r[f"lat_{c}"] for r in timed]) for c in LAT_COLS},
        "accuracy_m": {},
    }
    acc_keys = ["ade", "fde", "min_ade", "min_fde"] + [f"{p}@{h:g}s" for h in horizons for p in ("ade", "min_ade")]
    for k in acc_keys:
        summary["accuracy_m"][k] = _stats([r[k] for r in timed])
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    L = summary["latency_ms"]
    md = [f"# {meta['run_name']}", "",
          f"Model: `{meta['model_name']}` ({meta.get('checkpoint', '-')}) · GPU: {meta.get('gpu', '-')} · "
          f"K={meta['num_samples']} · scored samples: {len(timed)} / {len(rows)}", "",
          f"Prompt: `{meta.get('task_prompt')}`", "",
          "## Latency (ms)", "", "| Stage | p50 | p90 | mean | std |", "|---|---|---|---|---|"]
    for c in LAT_COLS:
        s = L[c]
        md.append(f"| {c} | {s['p50']:.1f} | {s['p90']:.1f} | {s['mean']:.1f} | {s['std']:.1f} |")
    md += ["", "## Accuracy (m, xy, unscaled)", "", "| Metric | mean | median | p90 |", "|---|---|---|---|"]
    for k, s in summary["accuracy_m"].items():
        md.append(f"| {k} | {s['mean']:.3f} | {s['p50']:.3f} | {s['p90']:.3f} |")
    (out_dir / "summary.md").write_text("\n".join(md) + "\n")
    return summary
