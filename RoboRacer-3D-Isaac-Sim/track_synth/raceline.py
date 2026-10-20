"""Lap-time-oriented racing line + speed profile on a closed track corridor.

The line is a bounded linear least-squares problem over lateral offsets from the centre line:
minimise  integral kappa^2 ds  +  mu * (path length term), s.t. |offset| <= half_width - margin.
mu trades a smooth line against a short one; it is chosen by evaluating the lap time of the
resulting friction-limited speed profile (a_lat / a_acc / a_dec / v_max). This is a standard
minimum-curvature / shortest-path blend, not a full minimum-time optimal control solution.
"""
import numpy as np
import scipy.sparse as sp
from scipy import ndimage as ndi
from scipy.optimize import lsq_linear

from . import vehicle

DEFAULTS = dict(v_max=vehicle.SPEED_MAX, a_lat=vehicle.A_LAT, a_acc=vehicle.A_ACC, a_dec=vehicle.A_DEC)
# tightest drivable curvature tan(max steering) / wheelbase = 0.90 1/m; keep a margin for the dynamics
KAPPA_LIMIT = .95 * vehicle.KAPPA_STEER_MAX


def arc_length(path):
    d = np.linalg.norm(np.roll(path, -1, axis=0) - path, axis=1)
    return np.r_[0., np.cumsum(d)][:-1], float(d.sum())


def resample_closed(path, spacing):
    pts = np.vstack([path, path[:1]])
    s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    t = np.linspace(0, s[-1], max(int(round(s[-1] / spacing)), 8), endpoint=False)
    return np.c_[np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])]


def curvature(path, smooth_m=0.):
    """Signed curvature (1/m) of a closed polyline, central differences in arc length."""
    s, total = arc_length(path)
    ds = total / len(path)
    p = ndi.gaussian_filter1d(path, smooth_m / ds, axis=0, mode='wrap') if smooth_m > 0 else path
    d1 = (np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)) / (2 * ds)
    d2 = (np.roll(p, -1, axis=0) - 2 * p + np.roll(p, 1, axis=0)) / ds ** 2
    return (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.maximum(np.linalg.norm(d1, axis=1) ** 3, 1e-9)


def speed_profile(path, v_max=vehicle.SPEED_MAX, a_lat=vehicle.A_LAT, a_acc=vehicle.A_ACC, a_dec=vehicle.A_DEC,
                  kappa_smooth_m=.3):
    """Friction-limited speed along a closed path: curvature limit, then forward/backward passes."""
    kappa = np.abs(curvature(path, kappa_smooth_m))
    s, total = arc_length(path)
    n = len(path); ds = np.linalg.norm(np.roll(path, -1, axis=0) - path, axis=1)
    v = np.clip(np.sqrt(a_lat / np.maximum(kappa, 1e-3)), vehicle.SPEED_MIN, v_max)
    for _ in range(3):                                             # periodic: several sweeps
        for i in range(2 * n):
            j, k = i % n, (i + 1) % n
            v[k] = min(v[k], np.sqrt(v[j] ** 2 + 2 * a_acc * ds[j]))
        for i in range(2 * n, 0, -1):
            j, k = i % n, (i - 1) % n
            v[k] = min(v[k], np.sqrt(v[j] ** 2 + 2 * a_dec * ds[k]))
    lap = float(np.sum(ds / np.maximum(.5 * (v + np.roll(v, -1)), 1e-3)))
    return v, lap


def _line(centerline, half_width, margin, mu, spacing, kappa_limit=KAPPA_LIMIT, rounds=8):
    c = resample_closed(centerline, spacing)
    n = len(c)
    # offset limits and fixed normals from the (coarse) centre line
    tangent = np.roll(c, -1, axis=0) - np.roll(c, 1, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    normal = np.c_[-tangent[:, 1], tangent[:, 0]]
    s_c, total_c = arc_length(centerline)
    hw = np.interp(np.linspace(0, total_c, n, endpoint=False), np.r_[s_c, total_c],
                   np.r_[half_width, half_width[0]])
    limit = np.maximum(hw - margin, 0.)
    eye = sp.identity(n, format='csr')
    shift_f = sp.csr_matrix(np.roll(np.eye(n), -1, axis=0)); shift_b = sp.csr_matrix(np.roll(np.eye(n), 1, axis=0))
    d1 = shift_f - eye
    d2 = shift_f - 2 * eye + shift_b
    # variables: offsets a (n); p = c + N a  (x and y stacked). Reweighted rounds push the curvature
    # peaks below the steering limit (iteratively reweighted least squares ~ minimise the maximum).
    weight = np.ones(n); a = np.zeros(n)
    for _ in range(rounds):
        blocks, rhs = [], []
        for axis in (0, 1):
            N = sp.diags(normal[:, axis])
            blocks += [sp.diags(weight) @ d2 @ N / spacing ** 1.5, np.sqrt(mu) * d1 @ N / spacing ** .5]
            rhs += [-weight * (d2 @ c[:, axis]) / spacing ** 1.5, -np.sqrt(mu) * (d1 @ c[:, axis]) / spacing ** .5]
        a = lsq_linear(sp.vstack(blocks).tocsr(), np.concatenate(rhs), bounds=(-limit, limit),
                       method='trf', lsq_solver='lsmr', max_iter=300).x
        kappa = np.abs(curvature(c + normal * a[:, None], .3))
        weight = weight * np.where(kappa > kappa_limit, 1.8, 1.)
        if kappa.max() <= kappa_limit:
            break
    return c + normal * a[:, None], a, limit


def optimise_line(centerline, half_width, margin=.4, spacing=.3, mus=(.02, .06, .2, .6),
                  kappa_limit=KAPPA_LIMIT, **limits):
    """Best line over `mus` by lap time among lines the steering can follow (kappa_max <= limit * 1.05;
    if none qualifies, the smoothest one). Returns dict(path, speed, lap_time, mu, candidates, ...)."""
    limits = {**DEFAULTS, **limits}
    best, candidates = None, []
    for mu in mus:
        path, offset, limit = _line(centerline, half_width, margin, mu, spacing, kappa_limit)
        path = resample_closed(path, .2)
        speed, lap = speed_profile(path, **limits)
        kmax = float(np.abs(curvature(path, .3)).max())
        candidates.append((mu, lap, kmax))
        feasible = kmax <= kappa_limit * 1.05
        rank = (0, lap) if feasible else (1, kmax)
        if best is None or rank < best['rank']:
            best = dict(path=path, speed=speed, lap_time=lap, mu=mu, rank=rank, feasible=feasible)
    best.pop('rank')
    center = resample_closed(centerline, .2)
    speed_c, lap_c = speed_profile(center, **limits)
    best.update(candidates=candidates, centerline_lap_time=lap_c, limits=limits,
                kappa_max=float(np.abs(curvature(best['path'], .3)).max()))
    return best
