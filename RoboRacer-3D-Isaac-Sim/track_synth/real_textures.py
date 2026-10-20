"""Floor texture from real (or 3DGS-rendered) ego camera frames.

Inverse perspective mapping of the floor in front of the car (known intrinsics, mount and
height) gives a metric top view. Small windows free of line paint / debris are collected from
all frames and quilted into a tileable texture (2 mm per pixel).

    python -m track_synth.real_textures --frames ../data/frames --out reference/iros2026/floor_real.png
"""
import argparse
from pathlib import Path
import numpy as np
import cv2
from racing.config import sensor_pose

# 1280x800 intrinsics of the OAK-D frames (2x the 640x400 racing config) and the Car02 mount
K_1280 = np.array([[566.8698, 0, 641.9587], [0, 583.7564, 388.6107], [0, 0, 1]])
MOUNT = dict(translation=[0.29267, 0.06987, 0.200],
             quaternion=[-0.4743109522221859, 0.4989799497372533, -0.5079499488336963, 0.5177209478494539])
BASE_Z = .05


def ipm_maps(K, size_m=1.4, ahead_m=.95, res=.002, scale=1.):
    """Pixel lookup (u, v) for a size_m x size_m floor patch starting ahead_m in front of base_link."""
    K = K.copy(); K[:2] *= scale
    w2c = np.linalg.inv(sensor_pose(dict(x=0., y=0., yaw=0., z=BASE_Z), MOUNT))
    xs = np.arange(ahead_m, ahead_m + size_m, res); ys = np.arange(size_m / 2, -size_m / 2, -res)
    X, Y = np.meshgrid(xs, ys)
    P = np.stack([X, Y, np.zeros_like(X), np.ones_like(X)], -1) @ w2c.T
    uv = P[..., :3] @ K.T
    return (uv[..., 0] / uv[..., 2]).astype(np.float32), (uv[..., 1] / uv[..., 2]).astype(np.float32)


def paint_mask(top):
    """Line paint (yellow/orange/red), tape and dark debris."""
    lab = cv2.cvtColor(top, cv2.COLOR_BGR2LAB).astype(np.int16)
    hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
    b, a = lab[..., 2] - 128, lab[..., 1] - 128
    yellowish = b > np.median(b) + 9
    reddish = a > np.median(a) + 9
    dark = hsv[..., 2] < np.median(hsv[..., 2]) * .55
    mask = (yellowish | reddish | dark).astype(np.uint8)
    return cv2.dilate(mask, np.ones((11, 11), np.uint8))


def clean_windows(frames, u, v, window, stride, max_paint=0.):
    """All floor windows (window x window px) without paint/debris, brightness-normalised."""
    out = []
    for path in frames:
        top = cv2.remap(cv2.imread(str(path)), u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        valid = top.sum(2) > 0
        mask = paint_mask(top)
        for r in range(0, top.shape[0] - window + 1, stride):
            for c in range(0, top.shape[1] - window + 1, stride):
                if valid[r:r + window, c:c + window].all() and mask[r:r + window, c:c + window].mean() <= max_paint:
                    out.append(top[r:r + window, c:c + window].astype(np.float32))
    if not out:
        raise RuntimeError('no clean floor windows found')
    out = np.stack(out)
    means = out.mean((1, 2), keepdims=True)
    target = np.median(means, axis=0, keepdims=True)
    # keep 30% of each window's brightness deviation: mottling without a patchwork look
    return out / means * (target + .3 * (means - target))


def quilt(windows, size, window, rng):
    """Toroidal mosaic of random windows with feathered overlaps (tileable by construction)."""
    step = window // 2
    acc = np.zeros((size, size, 3), np.float32); wsum = np.zeros((size, size, 1), np.float32)
    ramp = np.sin(np.linspace(0, np.pi, window)) ** 2
    weight = (ramp[:, None] * ramp[None, :])[..., None] + 1e-4
    for r in range(0, size, step):
        for c in range(0, size, step):
            patch = windows[rng.integers(len(windows))]
            k = rng.integers(4)
            patch = np.rot90(patch, k)            # floor texture is isotropic
            rows = (np.arange(window) + r) % size; cols = (np.arange(window) + c) % size
            acc[np.ix_(rows, cols)] += patch * weight
            wsum[np.ix_(rows, cols)] += weight
    return np.clip(acc / wsum, 0, 255).astype(np.uint8)


def floor_texture(frames, out, size_px=2048, window=160, seed=0):
    """size_px at 2 mm/px: 2048 px = 4.1 m texture tile."""
    frames = sorted(Path(frames).glob('*.png'))
    if not frames:
        raise FileNotFoundError('no PNG frames')
    scale = cv2.imread(str(frames[0])).shape[1] / 1280.
    u, v = ipm_maps(K_1280, size_m=1.6, ahead_m=.8, scale=scale)
    windows = clean_windows(frames[::2], u, v, window, stride=window // 2)
    tile = quilt(windows, size_px, window, np.random.default_rng(seed))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), tile)
    return dict(windows=len(windows), frames=len(frames), tile_m=size_px * .002)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--frames', required=True)
    parser.add_argument('--out', default='reference/iros2026/floor_real.png')
    args = parser.parse_args()
    print(floor_texture(args.frames, args.out))


if __name__ == '__main__':
    main()
