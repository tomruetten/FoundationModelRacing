"""Occupancy map (ROS map_server PNG+YAML) -> wall polylines, track band, closed centerline.

World frame = map frame: x right, y up, metres, origin from the YAML (lower-left pixel).
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi


@dataclass
class TrackGeometry:
    resolution: float
    origin: tuple
    walls: list            # open/closed XY polylines (m): tube centre lines
    centerline: np.ndarray # (N,2) closed, ordered, ~uniform spacing (m)
    half_width: np.ndarray # (N,) free half width at each centerline point (m)
    outer: np.ndarray      # (M,2) outer boundary of the free band (m)
    holes: list            # boundaries of the enclosed infield regions (m)
    bounds: tuple          # xmin, ymin, xmax, ymax of the map (m)


def load_yaml(path):
    values = {}
    for line in Path(path).read_text().splitlines():
        line = line.split('#')[0].strip()
        if ':' in line:
            key, value = line.split(':', 1)
            values[key.strip()] = value.strip().strip('"\'')
    origin = [float(v) for v in values['origin'].strip('[]').split(',')]
    image = Path(values['image']); image = image if image.is_absolute() else Path(path).parent / image
    return dict(image=image, resolution=float(values['resolution']), origin=origin,
                negate=int(values.get('negate', 0)), occupied=float(values.get('occupied_thresh', .65)))


def occupancy(config):
    gray = np.asarray(Image.open(config['image']).convert('L'), dtype=float) / 255.
    occupied_prob = gray if config['negate'] else 1. - gray
    return occupied_prob > config['occupied']


def split_map(occupied, min_wall_pixels=200):
    """Drop map-grid specks, return (walls mask, free-track mask).

    The free track is the largest free component that is not the exterior; the
    exterior is the component touching the border.
    """
    labels, n = ndi.label(occupied, structure=np.ones((3, 3)))
    sizes = ndi.sum(occupied, labels, range(1, n + 1))
    walls = np.isin(labels, [i + 1 for i in range(n) if sizes[i] >= min_wall_pixels])
    free, m = ndi.label(~walls)
    border = set(np.unique(np.r_[free[0], free[-1], free[:, 0], free[:, -1]]))
    sizes = ndi.sum(~walls, free, range(1, m + 1))
    candidates = [i for i in range(1, m + 1) if i not in border]
    if not candidates:
        raise ValueError('No enclosed free region found')
    track = free == max(candidates, key=lambda i: sizes[i - 1])
    return walls, track


def _to_world(rc, config, height):
    """(row, col) pixel centres -> world XY."""
    rc = np.asarray(rc, dtype=float)
    return np.c_[config['origin'][0] + (rc[:, 1] + .5) * config['resolution'],
                 config['origin'][1] + (height - rc[:, 0] - .5) * config['resolution']]


def _smooth_open(points, sigma):
    if len(points) < 5 or sigma <= 0:
        return points
    return ndi.gaussian_filter1d(points, sigma, axis=0, mode='nearest')


def _smooth_closed(points, sigma):
    return ndi.gaussian_filter1d(points, sigma, axis=0, mode='wrap')


def resample(points, spacing, closed):
    pts = np.vstack([points, points[:1]]) if closed else np.asarray(points)
    s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    count = max(int(round(s[-1] / spacing)), 3)
    t = np.linspace(0, s[-1], count + 1)
    out = np.c_[np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])]
    return out[:-1] if closed else out


def wall_polylines(walls, config, spacing=.1, smooth_m=.3, overlap_m=.2):
    """Skeletonise the wall lines into polylines between junctions/endpoints."""
    from skimage.morphology import skeletonize
    from skan import Skeleton
    skeleton = skeletonize(walls)
    graph = Skeleton(skeleton, keep_images=False)
    height = walls.shape[0]
    lines = []
    for i in range(graph.n_paths):
        rc = graph.path_coordinates(i)
        if len(rc) < 3:
            continue
        xy = _to_world(rc, config, height)
        xy = _smooth_open(xy, smooth_m / config['resolution'])
        if np.linalg.norm(np.diff(xy, axis=0), axis=1).sum() < .3:
            continue
        xy = resample(xy, spacing, closed=False)
        # skan cuts paths at junction pixels; overlap the ends so tubes meet without gaps
        head = xy[0] - xy[min(3, len(xy) - 1)]; tail = xy[-1] - xy[max(-4, -len(xy))]
        xy = np.vstack([xy[0] + head / (np.linalg.norm(head) + 1e-9) * overlap_m, xy,
                        xy[-1] + tail / (np.linalg.norm(tail) + 1e-9) * overlap_m])
        lines.append(xy)
    return lines


def _prune(skeleton, iterations):
    """Remove short spurs (endpoint branches) by iteratively deleting endpoints."""
    sk = skeleton.copy()
    removed = []
    kernel = np.ones((3, 3))
    for _ in range(iterations):
        nb = ndi.convolve(sk.astype(int), kernel, mode='constant') - sk
        ends = sk & (nb <= 1)
        if not ends.any():
            break
        removed.append(ends)
        sk &= ~ends
    # regrow the main branch ends so the long loop is not shortened
    for ends in reversed(removed):
        grown = ndi.binary_dilation(sk, structure=kernel) & ends
        end_nb = ndi.convolve(sk.astype(int), kernel, mode='constant') - sk
        tips = sk & (end_nb <= 1)
        sk |= ndi.binary_dilation(tips, structure=kernel) & ends & grown
    return sk


def trace_centerline(track, config, spacing=.1, sigma_m=.4):
    """Medial axis of the free band as one ordered closed loop."""
    from skimage.morphology import skeletonize
    skeleton = skeletonize(track)
    skeleton = _prune(skeleton, iterations=40)
    labels, n = ndi.label(skeleton, structure=np.ones((3, 3)))
    if n > 1:
        sizes = ndi.sum(skeleton, labels, range(1, n + 1))
        skeleton = labels == (1 + int(np.argmax(sizes)))
    coords = np.argwhere(skeleton)
    index = {tuple(c): i for i, c in enumerate(coords)}
    neighbours = [[index[(r + dr, c + dc)] for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                   if (dr or dc) and (r + dr, c + dc) in index] for r, c in coords]
    # walk the loop, preferring the neighbour that continues the heading
    start = int(np.argmin(coords[:, 1] + 1e-3 * coords[:, 0]))
    order = [start]; seen = {start}
    current = start
    while True:
        options = [j for j in neighbours[current] if j not in seen]
        if not options:
            break
        if len(order) > 1:
            heading = coords[current] - coords[order[-2]]
            options.sort(key=lambda j: -np.dot(coords[j] - coords[current], heading) /
                         (np.linalg.norm(coords[j] - coords[current]) + 1e-9))
        else:
            options.sort(key=lambda j: np.linalg.norm(coords[j] - coords[current]))
        current = options[0]; order.append(current); seen.add(current)
    if len(order) < .9 * len(coords):
        raise ValueError(f'Centerline trace covers {len(order)} of {len(coords)} skeleton pixels; '
                         'inspect the map (junctions)')
    xy = _to_world(coords[order], config, track.shape[0])
    xy = _smooth_closed(xy, sigma_m / config['resolution'])
    return resample(xy, spacing, closed=True)


def boundaries(track, config, erode_m=0.):
    """Outer / hole polygons of the free band, optionally shrunk by `erode_m` (sub-pixel, via EDT)."""
    import cv2
    if erode_m > 0:
        track = ndi.distance_transform_edt(track) * config['resolution'] > erode_m
    contours, hierarchy = cv2.findContours(track.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    height = track.shape[0]
    outer, holes = None, []
    for contour, info in zip(contours, hierarchy[0]):
        rc = contour[:, 0, ::-1]
        polygon = resample(_smooth_closed(_to_world(rc, config, height), 1.5), .1, closed=True)
        if info[3] == -1:
            if outer is None or len(polygon) > len(outer):
                outer = polygon
        else:
            holes.append(polygon)
    return outer, holes


def extract(yaml_path, spacing=.1, boundary_erode_m=0.):
    config = load_yaml(yaml_path)
    occupied = occupancy(config)
    walls, track = split_map(occupied)
    centerline = trace_centerline(track, config, spacing)
    distance = ndi.distance_transform_edt(track) * config['resolution']
    height = track.shape[0]
    rows = height - 1 - ((centerline[:, 1] - config['origin'][1]) / config['resolution']).astype(int)
    cols = ((centerline[:, 0] - config['origin'][0]) / config['resolution']).astype(int)
    half_width = distance[np.clip(rows, 0, height - 1), np.clip(cols, 0, track.shape[1] - 1)]
    outer, holes = boundaries(track, config, boundary_erode_m)
    width = track.shape[1]
    bounds = (config['origin'][0], config['origin'][1],
              config['origin'][0] + width * config['resolution'], config['origin'][1] + height * config['resolution'])
    return TrackGeometry(config['resolution'], tuple(config['origin']), wall_polylines(walls, config, spacing),
                         centerline, half_width, outer, holes, bounds)
