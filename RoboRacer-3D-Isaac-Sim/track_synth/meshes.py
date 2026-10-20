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


def hose_profile(path_length_points, rng, radius=.15, flatten=.82):
    """Slowly varying radii so the hose looks sagging instead of a perfect pipe."""
    n = path_length_points
    t = np.arange(n)
    phase = rng.uniform(0, 2 * np.pi, 3)
    wobble = (.04 * np.sin(2 * np.pi * t / 90 + phase[0]) + .025 * np.sin(2 * np.pi * t / 37 + phase[1]))
    rh = radius * (1 + wobble)
    rv = radius * flatten * (1 - .5 * wobble + .02 * np.sin(2 * np.pi * t / 23 + phase[2]))
    return rh, rv


def stacked_hose_layers(path, rng, radius=.15, flatten=.82, layers=2, sides=24):
    """`layers` hoses stacked on top of each other along one wall line.

    The upper hose rests on the lower one with a small low-frequency lateral drift.
    Returns a list of meshes, bottom first.
    """
    meshes = []
    n = len(path)
    tangent = np.gradient(path, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    left = np.c_[-tangent[:, 1], tangent[:, 0]]
    z_top = 0.
    for layer in range(layers):
        rh, rv = hose_profile(n, rng, radius, flatten)
        if layer == 0:
            centre, z = path, rv
        else:
            drift = .03 * np.sin(2 * np.pi * np.arange(n) / 120 + rng.uniform(0, 6.28))
            centre = path + left * drift[:, None]
            z = z_top + rv * .92          # slightly sunk into the hose below
        z_top = z + rv
        meshes.append(sweep_tube(centre, z, rh, rv, sides=sides))
    return meshes


def corrugation_normal_map(size=256, ripples=12, depth=.9):
    """Tangent-space normal map; ripples vary along u (hose axis)."""
    u = (np.arange(size) + .5) / size
    slope = depth * np.cos(2 * np.pi * ripples * u)
    nx = -slope[None, :].repeat(size, 0) * .6
    n = np.stack([nx, np.zeros_like(nx), np.ones_like(nx)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return ((n * .5 + .5) * 255).astype(np.uint8)


def hose_albedo(size=256, ripples=12, base=(0.86, 0.86, 0.83), seed=0):
    rng = np.random.default_rng(seed)
    u = (np.arange(size) + .5) / size
    shade = .93 + .07 * np.cos(2 * np.pi * ripples * u)
    img = np.ones((size, size, 3)) * np.array(base) * shade[None, :, None]
    img *= 1 + .04 * rng.standard_normal((size, size, 1))
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def concrete_albedo(size=1024, base=(0.52, 0.51, 0.48), seed=1):
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
        img *= (1 - .35 * np.exp(-d))[..., None]
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)
