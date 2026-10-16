"""Per-ray learned LiDAR and visibility-first composition (X forward, Y left, Z up).

The static renderer returns geometric hits before Bernoulli return sampling.
Call ``compose_ray_samples`` once, with all actors, to produce observed hits.
There is no recorded-scan lookup or checkpoint autodiscovery in this module.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class RaySamples:
    ranges: np.ndarray
    hit_mask: np.ndarray
    intensity: np.ndarray
    return_probability: np.ndarray

    def __post_init__(self):
        self.ranges = np.asarray(self.ranges, dtype=np.float32)
        self.hit_mask = np.asarray(self.hit_mask, dtype=bool)
        self.intensity = np.asarray(self.intensity, dtype=np.float32)
        self.return_probability = np.asarray(self.return_probability, dtype=np.float32)
        shape = self.ranges.shape
        if len(shape) != 1 or any(a.shape != shape for a in
                                 (self.hit_mask, self.intensity, self.return_probability)):
            raise ValueError('RaySamples fields must all have shape (N,)')
        if np.any(self.hit_mask & (~np.isfinite(self.ranges) | (self.ranges <= 0))):
            raise ValueError('Geometric hits must have finite positive ranges')
        if (not np.isfinite(self.intensity).all() or
                not np.isfinite(self.return_probability).all() or
                np.any((self.return_probability < 0) | (self.return_probability > 1))):
            raise ValueError('Intensity must be finite and return probabilities in [0, 1]')

    @classmethod
    def empty(cls, n):
        return cls(np.full(n, np.inf), np.zeros(n, bool), np.zeros(n), np.zeros(n))


def compose_ray_samples(static, dynamic=(), *, seed=None, rng=None):
    """Choose nearest opaque geometry, then sample that surface's return.

    Inputs' hit masks indicate geometry. Output's hit mask indicates observed
    returns; its ranges/intensities/probabilities still describe the nearest
    surface, including dropped returns. Do not feed this sampled output into
    another composition; pass all actors together instead.
    """
    if seed is not None and rng is not None:
        raise ValueError('Specify seed or rng, not both')
    rng = np.random.default_rng(seed) if rng is None else rng
    actors = [dynamic] if isinstance(dynamic, RaySamples) else dynamic
    result = RaySamples(static.ranges.copy(), static.hit_mask.copy(),
                        static.intensity.copy(), static.return_probability.copy())
    for actor in actors:
        if actor.ranges.shape != result.ranges.shape:
            raise ValueError('All samples must describe the same rays')
        closer = actor.hit_mask & (~result.hit_mask | (actor.ranges < result.ranges))
        for field in ('ranges', 'hit_mask', 'intensity', 'return_probability'):
            getattr(result, field)[closer] = getattr(actor, field)[closer]
    result.hit_mask &= rng.random(len(result.ranges)) < result.return_probability
    return result


# Ported from data/local/pipeline_scripts_1707/lidar_renderer_okession.py.
# Keep ordering fixed: argmax makes face ties deterministic at cube seams.
FACE_ROTATIONS = np.array([
    np.eye(3), [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
    [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
    [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
    [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
    [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
], dtype=np.float64)
R_CL_BASE = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float64)


def generate_rays(n_points=20000, frame_idx=0):
    """Latest pipeline's approximate Livox Risley-prism pattern, in sensor frame.

    This preserves the source generator exactly; its integer frame offsets
    repeat the same pattern. It is not a calibrated nonrepeating sensor model.
    """
    if not isinstance(n_points, (int, np.integer)) or n_points < 1:
        raise ValueError('n_points must be a positive integer')
    t = np.linspace(0, 1, n_points) + frame_idx
    phi1, phi2 = 2*np.pi*7*t, 2*np.pi*11*t
    dx = np.sin(np.radians(15))*np.cos(phi1) + np.sin(np.radians(12))*np.cos(phi2)
    dy = np.sin(np.radians(15))*np.sin(phi1) + np.sin(np.radians(12))*np.sin(phi2)
    azimuth = np.arctan2(dy, dx)
    elevation = np.clip(np.sqrt(dx**2 + dy**2)*np.radians(30) - np.radians(7),
                        np.radians(-7), np.radians(52))
    return np.stack([np.cos(elevation)*np.cos(azimuth),
                     np.cos(elevation)*np.sin(azimuth), np.sin(elevation)], axis=1).astype(np.float32)


def _unit_rays(rays):
    rays = np.asarray(rays, dtype=np.float64)
    if rays.ndim != 2 or rays.shape[1] != 3 or not np.isfinite(rays).all():
        raise ValueError('rays must be a finite (N, 3) array')
    lengths = np.linalg.norm(rays, axis=1)
    if np.any(lengths < 1e-12):
        raise ValueError('Ray direction cannot be zero')
    return rays / lengths[:, None]


def _pose(pose):
    pose = np.asarray(pose, dtype=np.float64)
    if (pose.shape != (4, 4) or not np.isfinite(pose).all() or
            not np.allclose(pose[3], [0, 0, 0, 1]) or
            not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5) or
            not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-5)):
        raise ValueError('pose must be a rigid sensor-to-world 4x4 transform')
    return pose


def sample_cubemap(rays, depths, alphas, colors, *, return_head='none',
                   alpha_threshold=.99, max_range=8.0):
    """Sample six rendered faces without filtering out the ray's geometry.

    ``depths`` contains gsplat expected axial depth, not Euclidean range.
    ``return`` is sigmoid(colors[...,1]), matching training's depth_gt > 0
    target; ``drop`` is its complement, for explicitly drop-trained models.
    ``none`` never uses the untrained channel and sets probability to one.
    """
    if return_head not in ('none', 'return', 'drop'):
        raise ValueError('return_head must be none, return, or drop')
    if not 0 <= alpha_threshold <= 1 or not np.isfinite(max_range) or max_range <= 0:
        raise ValueError('Invalid alpha threshold or max range')
    rays = _unit_rays(rays)
    depths, alphas, colors = map(np.asarray, (depths, alphas, colors))
    if (depths.ndim != 3 or depths.shape[0] != 6 or depths.shape[1] != depths.shape[2]
            or depths.shape[1] < 2 or alphas.shape != depths.shape or
            colors.shape != depths.shape + (3,)):
        raise ValueError('Expected six square depth/alpha faces and three color channels')
    result = RaySamples.empty(len(rays))
    camera_rays = np.stack([rays @ (face @ R_CL_BASE.T) for face in FACE_ROTATIONS])
    best = np.argmax(camera_rays[:, :, 2], axis=0)
    width = depths.shape[2]
    focal = width / 2
    for face in range(6):
        indices = np.flatnonzero(best == face)
        if not len(indices):
            continue
        cam = camera_rays[face, indices]
        # Border extension covers exact edges/corners; one winning face owns
        # each ray, so another face can neither overwrite it nor leave a seam.
        u = np.clip(focal * cam[:, 0]/cam[:, 2] + focal, 0, width-1)
        v = np.clip(focal * cam[:, 1]/cam[:, 2] + focal, 0, width-1)
        u0, v0 = u.astype(int), v.astype(int)
        u1, v1 = np.minimum(u0+1, width-1), np.minimum(v0+1, width-1)
        fu, fv = u-u0, v-v0

        def bilerp(img):
            return (img[v0, u0]*(1-fu)*(1-fv) + img[v0, u1]*fu*(1-fv) +
                    img[v1, u0]*(1-fu)*fv + img[v1, u1]*fu*fv)

        distance = bilerp(depths[face]) / cam[:, 2]
        alpha = bilerp(alphas[face])
        intensity = bilerp(colors[face, ..., 0])
        probability = np.ones(len(indices))
        if return_head != 'none':
            logit = bilerp(colors[face, ..., 1])
            probability = 1 / (1 + np.exp(-np.clip(logit, -80, 80)))
            if return_head == 'drop':
                probability = 1-probability
        hit = np.isfinite(distance) & (distance > 0) & (distance <= max_range) & (alpha >= alpha_threshold)
        result.ranges[indices] = np.where(np.isfinite(distance) & (distance > 0), distance, np.inf)
        result.hit_mask[indices] = hit
        result.intensity[indices] = np.nan_to_num(intensity, nan=0, posinf=0, neginf=0)
        result.return_probability[indices] = np.nan_to_num(probability, nan=0)
    return result


def triangle_ray_samples(pose, rays, triangles, *, intensity=.5, return_probability=1.,
                         max_range=8., ray_chunk=256, triangle_chunk=256):
    """Chunked two-sided Möller–Trumbore intersection with world-space triangles.

    Intended for diagnostics and small meshes; Isaac's batched Warp mesh
    raycast can populate the same RaySamples interface for runtime throughput.
    """
    pose, rays = _pose(pose), _unit_rays(rays)
    triangles = np.asarray(triangles, dtype=np.float64)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or not np.isfinite(triangles).all():
        raise ValueError('triangles must have shape (T, 3, 3) and be finite')
    if ray_chunk < 1 or triangle_chunk < 1:
        raise ValueError('Chunk sizes must be positive')
    directions = rays @ pose[:3, :3].T
    result = RaySamples.empty(len(rays))
    for start in range(0, len(rays), ray_chunk):
        direction = directions[start:start+ray_chunk]
        nearest = np.full(len(direction), np.inf)
        for offset in range(0, len(triangles), triangle_chunk):
            tri = triangles[offset:offset+triangle_chunk]
            e1, e2 = tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]
            pvec = np.cross(direction[:, None, :], e2)
            det = np.einsum('rtj,tj->rt', pvec, e1)
            nonparallel = np.abs(det) > 1e-10
            inv = np.divide(1., det, out=np.zeros_like(det), where=nonparallel)
            tvec = pose[:3, 3]-tri[:, 0]
            u = np.einsum('tj,rtj->rt', tvec, pvec)*inv
            qvec = np.cross(tvec, e1)
            v = np.einsum('rj,tj->rt', direction, qvec)*inv
            distance = np.einsum('tj,tj->t', e2, qvec)[None, :]*inv
            hit = nonparallel & (u >= 0) & (v >= 0) & (u+v <= 1) & (distance > 1e-5) & (distance <= max_range)
            nearest = np.minimum(nearest, np.min(np.where(hit, distance, np.inf), axis=1))
        result.ranges[start:start+len(direction)] = nearest
    result.hit_mask = np.isfinite(result.ranges)
    result.intensity[:] = intensity
    result.return_probability[:] = return_probability
    result.__post_init__()
    return result


class LearnedLidar:
    """Persistent gsplat model; instantiate only in a compatible Python runtime."""

    def __init__(self, checkpoint, device='cuda', return_head='none',
                 alpha_threshold=.99, resolution=256, max_range=8.):
        if return_head not in ('none', 'return', 'drop'):
            raise ValueError('return_head must be none, return, or drop')
        if resolution < 2 or not 0 <= alpha_threshold <= 1 or max_range <= 0:
            raise ValueError('Invalid LiDAR render configuration')
        import torch
        from gsplat import rasterization
        self.torch, self.rasterization = torch, rasterization
        self.device, self.resolution = device, int(resolution)
        self.return_head, self.alpha_threshold, self.max_range = return_head, alpha_threshold, max_range
        checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
        raw = torch.load(checkpoint, map_location=device, weights_only=True)
        required = {'means': 3, 'scales': 3, 'quats': 4, 'opacities': None, 'colors': 3}
        if not isinstance(raw, dict) or not required.keys() <= raw.keys():
            raise ValueError('Checkpoint must contain means/scales/quats/opacities/colors tensors')
        self.splats = {}
        count = len(raw['means'])
        for key, columns in required.items():
            value = raw[key]
            expected = (count,) if columns is None else (count, columns)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected:
                raise ValueError(f'Invalid checkpoint tensor {key}; expected {expected}')
            self.splats[key] = value.detach().to(device=device, dtype=torch.float32).contiguous()
        self.splats['scales'] = self.splats['scales'].exp()
        self.splats['opacities'] = self.splats['opacities'].sigmoid()
        f = self.resolution/2
        self.K = torch.tensor([[f, 0, f], [0, f, f], [0, 0, 1]], device=device).float()[None]

    def render(self, pose, rays):
        pose, rays = _pose(pose), _unit_rays(rays)
        torch = self.torch
        depths, colors, alphas = [], [], []
        with torch.inference_mode():
            for face in FACE_ROTATIONS:
                rotation = R_CL_BASE @ face.T @ pose[:3, :3].T
                view = np.eye(4, dtype=np.float32)
                view[:3, :3], view[:3, 3] = rotation, -rotation @ pose[:3, 3]
                image, alpha, _ = self.rasterization(
                    **self.splats, viewmats=torch.as_tensor(view, device=self.device)[None],
                    Ks=self.K, width=self.resolution, height=self.resolution,
                    render_mode='RGB+ED', near_plane=.1, far_plane=self.max_range+1,
                    sh_degree=None)
                image, alpha = image[0].cpu().numpy(), alpha[0, ..., 0].cpu().numpy()
                depths.append(image[..., 3])
                colors.append(image[..., :3])
                alphas.append(alpha)
        return sample_cubemap(rays, depths, alphas, colors, return_head=self.return_head,
                              alpha_threshold=self.alpha_threshold, max_range=self.max_range)
