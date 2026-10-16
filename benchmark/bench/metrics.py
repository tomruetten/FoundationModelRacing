"""Trajectory accuracy metrics. All values in metres, xy only, no rescaling."""

import numpy as np

DT = 0.1  # waypoint spacing of all supported models


def _dist(pred_xy: np.ndarray, gt_xy: np.ndarray) -> np.ndarray:
    """Per-waypoint L2 error. pred (K, T, 2), gt (T, 2) -> (K, T)."""
    return np.linalg.norm(pred_xy - gt_xy[None], axis=-1)


def trajectory_metrics(pred_xy: np.ndarray, gt_xy: np.ndarray, horizons_s: list[float]) -> dict:
    """ADE/FDE of sample 0 and minADE/minFDE over all K samples, overall and per horizon."""
    pred_xy = np.asarray(pred_xy, dtype=np.float64)
    gt_xy = np.asarray(gt_xy, dtype=np.float64)
    assert pred_xy.ndim == 3 and gt_xy.ndim == 2, (pred_xy.shape, gt_xy.shape)
    t = min(pred_xy.shape[1], gt_xy.shape[0])
    d = _dist(pred_xy[:, :t], gt_xy[:t])  # (K, T)

    out = {
        "ade": float(d[0].mean()),
        "fde": float(d[0, -1]),
        "min_ade": float(d.mean(axis=1).min()),
        "min_fde": float(d[:, -1].min()),
        "num_samples": int(pred_xy.shape[0]),
    }
    for h in horizons_s:
        n = min(int(round(h / DT)), t)
        key = f"{h:g}s"
        out[f"ade@{key}"] = float(d[0, :n].mean())
        out[f"min_ade@{key}"] = float(d[:, :n].mean(axis=1).min())
    return out
