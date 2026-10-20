"""Pure-NumPy mesh and texture generation (no USD / Isaac dependency)."""
import numpy as np


def sweep_tube(path, z, radius_h, radius_v, sides=24, uv_length=.5):
    """Swept elliptical tube with sealed ends.

    path: (N,2) XY centre line; z: scalar or (N,) centre height;
    radius_h / radius_v: scalar or (N,) horizontal / vertical radii.
    Returns dict(points, counts, indices, normals, uvs) with face-varying UVs
    (u = arc length / uv_length, v = angle fraction), triangulated caps included.
    """
    path = np.asarray(path, dtype=float); n = len(path)
    if n < 2:
        raise ValueError('tube path needs at least two points')
    tangent = np.gradient(path, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    left = np.c_[-tangent[:, 1], tangent[:, 0]]
    z = np.broadcast_to(np.asarray(z, dtype=float), (n,))
    rh = np.broadcast_to(np.asarray(radius_h, dtype=float), (n,))
    rv = np.broadcast_to(np.asarray(radius_v, dtype=float), (n,))
    theta = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    c, s = np.cos(theta), np.sin(theta)
    ring = np.empty((n, sides, 3))
    ring[..., :2] = path[:, None, :] + left[:, None, :] * (rh[:, None] * c)[..., None]
    ring[..., 2] = z[:, None] + rv[:, None] * s
    normal = np.empty_like(ring)
    normal[..., :2] = left[:, None, :] * (c / np.maximum(rh[:, None], 1e-9))[..., None]
    normal[..., 2] = s / np.maximum(rv[:, None], 1e-9)
    normal /= np.linalg.norm(normal, axis=2, keepdims=True)
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))] / uv_length

    points = ring.reshape(-1, 3); normals = normal.reshape(-1, 3)
    index = lambda i, j: i * sides + (j % sides)
    indices, uvs = [], []
    for i in range(n - 1):
        for j in range(sides):
            quad = [index(i, j), index(i, j + 1), index(i + 1, j + 1), index(i + 1, j)]
            indices += quad
            uvs += [(arc[i], j / sides), (arc[i], (j + 1) / sides),
                    (arc[i + 1], (j + 1) / sides), (arc[i + 1], j / sides)]
    counts = [4] * ((n - 1) * sides)
    extra_points, extra_normals = [], []
    for end, direction in ((0, -1.), (n - 1, 1.)):
        centre = np.array([path[end, 0], path[end, 1], z[end]])
        base = len(points) + len(extra_points)
        extra_points.append(centre); extra_normals.append(np.r_[tangent[end] * direction, 0.])
        for j in range(sides):
            a, b = index(end, j), index(end, j + 1)
            indices += [base, b, a] if direction > 0 else [base, a, b]
            uvs += [(arc[end], .5)] * 3; counts.append(3)
    # cap rim vertices keep the side normals: acceptable for a soft hose end
    points = np.vstack([points, extra_points]); normals = np.vstack([normals, extra_normals])
    return dict(points=points, counts=np.array(counts), indices=np.array(indices),
                normals=normals, uvs=np.array(uvs))


def _smooth_noise(n, rng, scale):
    """Zero-mean smooth noise with correlation length ~scale samples, unit-ish amplitude."""
    k = int(scale * 3) | 1
    kernel = np.exp(-.5 * (np.arange(k) - k // 2) ** 2 / scale ** 2)
    pad = np.r_[rng.standard_normal(n + 2 * k)]
    out = np.convolve(pad, kernel / np.linalg.norm(kernel), mode='same')[k:k + n]
    return out


def hose_profile(path_length_points, rng, radius=.15, flatten=.72):
    """Irregular soft hose: slow width changes, pinched / flattened stretches, random sag."""
    n = path_length_points
    slow = _smooth_noise(n, rng, 25); fast = _smooth_noise(n, rng, 6)
    rh = radius * (1 + .10 * slow + .03 * fast)
    pinch = np.clip(.5 * _smooth_noise(n, rng, 14) - .25, -.45, 0)      # occasional strongly flattened stretches
    rv = radius * flatten * (1 + pinch + .10 * _smooth_noise(n, rng, 20) + .03 * fast)
    return np.maximum(rh, .5 * radius), np.maximum(rv, .35 * radius * flatten)


def _stretch_mask(n, rng, fraction, spacing=.1, min_m=2., max_m=7., taper_m=.6):
    """Random stretches covering ~`fraction` of the length, with smooth tapers (1 = present)."""
    mask = np.zeros(n)
    if fraction >= 1:
        return np.ones(n)
    target = fraction * n; guard = 0
    while mask.sum() < target and guard < 200:
        guard += 1
        length = int(rng.uniform(min_m, max_m) / spacing); start = int(rng.integers(-length // 2, n))
        lo, hi = max(start, 0), min(start + length, n)
        if hi > lo:
            mask[lo:hi] = 1
    taper = max(int(taper_m / spacing), 1)
    return np.clip(np.convolve(mask, np.ones(taper) / taper, mode='same') * 1.0, 0, 1)


def stacked_hose_layers(path, rng, radius=.15, flatten=.72, layers=2, sides=24, upper_fraction=1.,
                        strap_spacing_m=1.8, spacing=.1):
    """Soft hoses along one wall line, optionally a second layer on top.

    Returns (hose_meshes, strap_meshes). The upper layer exists only along `upper_fraction` of the
    wall (random stretches, tapering to nothing at their ends), like the real boundary where the
    hose is mostly single-layer with some stacked stretches. Straps are thin brown rings around
    the hose every ~`strap_spacing_m`.
    """
    meshes, straps = [], []
    n = len(path)
    tangent = np.gradient(path, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    left = np.c_[-tangent[:, 1], tangent[:, 0]]
    wander = .03 * _smooth_noise(n, rng, 30)
    z_top = np.zeros(n)
    for layer in range(layers):
        rh, rv = hose_profile(n, rng, radius, flatten)
        if layer == 0:
            centre, z = path + left * wander[:, None], rv
        else:
            presence = _stretch_mask(n, rng, upper_fraction, spacing)
            # the upper hose is squeezed flatter by its own weight on the first
            rh, rv = rh * np.maximum(presence, 1e-3), rv * .85 * np.maximum(presence, 1e-3)
            centre = path + left * (wander + .03 * _smooth_noise(n, rng, 20))[:, None]
            z = z_top + rv * .9
        z_top = z + rv
        if layer == 0 or presence.max() > 0:
            meshes.append(sweep_tube(centre, z, rh, rv, sides=sides))
        if layer == 0:
            lower = (centre, z, rh, rv)
    # straps around the lowest layer (and over the upper one where it exists)
    step = max(int(strap_spacing_m / spacing), 3)
    for i in range(int(rng.integers(2, step)), n - 2, step):
        centre, z, rh, rv = lower
        a, b = centre[i], centre[i] + (centre[i + 1] - centre[i]) * .35
        straps.append(sweep_tube(np.vstack([a, b]), z[i], rh[i] * 1.05, rv[i] * 1.05, sides=sides, uv_length=1.))
    return meshes, straps


def _periodic_noise(size, cells_u, cells_v, rng):
    """Smooth tileable noise in [-1, 1]-ish with `cells_u` x `cells_v` lattice cells."""
    g = rng.standard_normal((cells_v, cells_u))
    iu = np.linspace(0, cells_u, size, endpoint=False); iv = np.linspace(0, cells_v, size, endpoint=False)
    u0, v0 = iu.astype(int), iv.astype(int); fu, fv = iu - u0, iv - v0
    su, sv = fu * fu * (3 - 2 * fu), fv * fv * (3 - 2 * fv)
    a = g[v0][:, u0]; b = g[v0][:, (u0 + 1) % cells_u]; c = g[(v0 + 1) % cells_v][:, u0]; d = g[(v0 + 1) % cells_v][:, (u0 + 1) % cells_u]
    top = a * (1 - su) + b * su; bottom = c * (1 - su) + d * su
    return top * (1 - sv)[:, None] + bottom * sv[:, None]


def corrugation_normal_map(size=512, ripples=20, depth=.5, seed=0):
    """Tangent-space normal map. u runs along the hose (ripples = corrugation rings), v around it.
    Ripple depth and phase wobble with smooth noise, plus random creases and dents."""
    rng = np.random.default_rng(seed)
    u = (np.arange(size) + .5) / size
    phase = .35 * _periodic_noise(size, 6, 4, rng)                       # rings are not perfectly regular
    amp = depth * (.75 + .5 * _periodic_noise(size, 4, 4, rng))
    ring = np.cos(2 * np.pi * ripples * u[None, :] + phase)
    slope_u = -amp * ring * .6
    crumple = _periodic_noise(size, 12, 8, rng) * .25 + _periodic_noise(size, 40, 24, rng) * .12
    slope_v = np.gradient(crumple, axis=0) * size / 25
    slope_u = slope_u + np.gradient(crumple, axis=1) * size / 25
    n = np.stack([-slope_u, -slope_v * .5, np.ones_like(slope_u)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return ((n * .5 + .5) * 255).astype(np.uint8)


def hose_albedo(size=512, ripples=20, base=(0.94, 0.93, 0.90), seed=0):
    """Bright white fabric hose: ring shading, dirt toward the underside (v ~ 0.75), soft stains."""
    rng = np.random.default_rng(seed + 7)
    u = (np.arange(size) + .5) / size; v = (np.arange(size) + .5) / size
    phase = .35 * _periodic_noise(size, 6, 4, np.random.default_rng(seed))
    ring = np.cos(2 * np.pi * ripples * u[None, :] + phase)
    shade = .94 + .06 * ring
    under = np.exp(-.5 * ((((v - .75) + .5) % 1 - .5) / .12) ** 2)[:, None]     # contact with the floor
    stains = _periodic_noise(size, 8, 6, rng) * .05 + _periodic_noise(size, 30, 20, rng) * .03
    img = np.array(base)[None, None, :] * (shade + stains)[..., None] * (1 - .22 * under)[..., None]
    img *= 1 + .015 * rng.standard_normal((size, size, 1))
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def concrete_albedo(size=1024, base=(0.76, 0.74, 0.69), seed=1):
    """Tileable grey epoxy floor: low-frequency mottling + fine grain + a few dark stains."""
    rng = np.random.default_rng(seed)
    def periodic(cells):
        g = rng.standard_normal((cells, cells))
        idx = np.linspace(0, cells, size, endpoint=False)
        i0 = idx.astype(int); fr = idx - i0
        sm = fr * fr * (3 - 2 * fr)
        a = g[i0][:, i0]; b = g[i0][:, (i0 + 1) % cells]
        c = g[(i0 + 1) % cells][:, i0]; d = g[(i0 + 1) % cells][:, (i0 + 1) % cells]
        top = a * (1 - sm)[None] + b * sm[None]; bottom = c * (1 - sm)[None] + d * sm[None]
        return top * (1 - sm)[:, None] + bottom * sm[:, None]
    mottling = periodic(6) * .5 + periodic(16) * .3 + periodic(64) * .2
    grain = rng.standard_normal((size, size)) * .02
    value = 1 + .07 * mottling + grain
    img = np.array(base)[None, None, :] * value[..., None]
    for _ in range(18):
        cx, cy = rng.integers(0, size, 2); r = rng.integers(6, 28)
        yy, xx = np.ogrid[:size, :size]
        d = ((xx - cx) ** 2 + (yy - cy) ** 2) / r ** 2
        img *= (1 - .2 * np.exp(-d))[..., None]
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)
