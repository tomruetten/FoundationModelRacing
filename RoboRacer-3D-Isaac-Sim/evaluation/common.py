"""Shared helpers for the evaluation suite: paths, metrics, report writer."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA_LOCAL = REPO / "data" / "local"


def env(name: str, default=None):
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def load_module(path: Path, name: str):
    """Import a script file (names starting with digits are fine) as a module."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def raw_session_dir(token: str) -> Path | None:
    """Find the raw session folder on the data drive for a session id token
    (e.g. '20260827_142014') via the RAW_SESSION_* variables from scripts/env.sh,
    falling back to a search under DATA_DRIVE_ROOT."""
    for k, v in os.environ.items():
        if k.startswith("RAW_SESSION_") and token in v and Path(v).is_dir():
            return Path(v)
    root = env("DATA_DRIVE_ROOT")
    if root and Path(root).is_dir():
        for p in Path(root).glob(f"*/*{token}*"):
            if p.is_dir():
                return p
    return None


# ----------------------------------------------------------------------------- metrics
def rmse(a: np.ndarray, b: np.ndarray, axis=0) -> np.ndarray:
    return np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2, axis=axis))


def chamfer_fscore(P: np.ndarray, Q: np.ndarray, taus=(0.05, 0.10, 0.20), max_points: int = 200_000) -> dict:
    """Symmetric Chamfer distance (mean nearest-neighbour distance both ways) and
    F-scores at the given thresholds [m] between two point sets (N,3), (M,3)."""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(0)
    if len(P) > max_points:
        P = P[rng.choice(len(P), max_points, replace=False)]
    if len(Q) > max_points:
        Q = Q[rng.choice(len(Q), max_points, replace=False)]
    if len(P) == 0 or len(Q) == 0:
        return {"chamfer_m": float("nan"), "n_P": int(len(P)), "n_Q": int(len(Q))}
    d_pq, _ = cKDTree(Q).query(P, k=1)
    d_qp, _ = cKDTree(P).query(Q, k=1)
    out = {
        "chamfer_m": float(0.5 * (d_pq.mean() + d_qp.mean())),
        "d_P_to_Q_mean_m": float(d_pq.mean()),
        "d_Q_to_P_mean_m": float(d_qp.mean()),
        "d_P_to_Q_median_m": float(np.median(d_pq)),
        "n_P": int(len(P)),
        "n_Q": int(len(Q)),
    }
    for t in taus:
        prec = float((d_pq < t).mean())
        rec = float((d_qp < t).mean())
        out[f"fscore@{int(t*100)}cm"] = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
        out[f"precision@{int(t*100)}cm"] = prec
        out[f"recall@{int(t*100)}cm"] = rec
    return out


# ----------------------------------------------------------------------------- report
class Report:
    """Collects markdown sections + a JSON summary and writes them to out_dir."""

    def __init__(self, out_dir: Path, title: str):
        self.out_dir = Path(out_dir)
        self.fig_dir = self.out_dir / "figures"
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.title = title
        self.md: list[str] = []
        self.summary: dict = {"title": title, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
        self.t0 = time.time()

    # markdown ------------------------------------------------------------
    def h(self, level: int, text: str):
        self.md.append("\n" + "#" * level + " " + text + "\n")

    def p(self, text: str):
        self.md.append(text.strip() + "\n")

    def table(self, headers: list[str], rows: list[list], fmt: str = "{:.4g}"):
        def cell(x):
            if isinstance(x, float):
                return "nan" if np.isnan(x) else fmt.format(x)
            return str(x)
        self.md.append("| " + " | ".join(headers) + " |")
        self.md.append("|" + "|".join(["---"] * len(headers)) + "|")
        for r in rows:
            self.md.append("| " + " | ".join(cell(c) for c in r) + " |")
        self.md.append("")

    def figure(self, rel_path: str, caption: str):
        self.md.append(f"![{caption}]({rel_path})\n\n*{caption}*\n")

    def note(self, text: str):
        self.md.append(f"> {text}\n")

    # data ----------------------------------------------------------------
    def put(self, *keys, value):
        d = self.summary
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value

    def savefig(self, fig, name: str, caption: str, dpi: int = 110):
        path = self.fig_dir / name
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)
        self.figure(f"figures/{name}", caption)

    def write(self):
        self.summary["elapsed_s"] = round(time.time() - self.t0, 1)
        (self.out_dir / "report.md").write_text("# " + self.title + "\n" + "\n".join(self.md), encoding="utf-8")
        (self.out_dir / "summary.json").write_text(json.dumps(self.summary, indent=2, default=_json_default), encoding="utf-8")
        return self.out_dir / "report.md"


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def log(msg: str):
    print(f"[eval {time.strftime('%H:%M:%S')}] {msg}", flush=True)
