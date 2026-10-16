"""
Undistort all COLMAP input images using the OAK-D Pro W factory intrinsic
calibration (camera_intrinsics.yaml: rational_polynomial, 8 distortion params),
producing a clean PINHOLE image set for a fresh COLMAP run.

Root cause being addressed: COLMAP's self-calibration on the original images
converged to fl_x=1078.97/fl_y=1007.03 with a 4-param OPENCV model (HFOV=61deg),
which is implausible for the OAK-D Pro W "Wide" lens (factory HFOV=97deg,
matching the 95-105deg "wide" spec). The factory rational_polynomial model has
much larger high-order coefficients (k1=3.62, k4=3.98, k5=2.05), characteristic
of a strong wide-angle lens that a 4-param model cannot represent -- causing
view-direction-dependent (and thus turn-dependent) systematic pose/triangulation
errors, the likely cause of the recurring trajectory shape distortion.

Usage:
    conda run -n nerfstudio-py310 python3 undistort_with_factory_calib.py
"""
import os
from pathlib import Path
import cv2
import numpy as np

# PROBE_ROOT: root of the raw Probe capture data (COLMAP input/output trees).
# Raw data isn't included in this repo (see root README) -- point this at
# your own copy, e.g. export PROBE_ROOT=/path/to/Probe
PROBE_ROOT = Path(os.environ.get('PROBE_ROOT', './data/Probe'))
SRC_DIR = PROBE_ROOT / 'colmap_out/images'
DST_DIR = PROBE_ROOT / 'colmap_undistort/images'
DST_DIR.mkdir(parents=True, exist_ok=True)

K_factory = np.array([[566.2562, 0, 641.9587],
                       [0, 565.9493, 388.6107],
                       [0, 0, 1]], dtype=np.float64)
dist_factory = np.array([3.6221985816955566, 0.9726417660713196,
                          9.896762639982626e-05, -4.5452539779944345e-05,
                          0.009347673505544662, 3.98378849029541,
                          2.0535287857055664, 0.11342744529247284], dtype=np.float64)

files = sorted(SRC_DIR.glob('*.png'))
print(f"undistorting {len(files)} images -> {DST_DIR}")
for i, f in enumerate(files):
    img = cv2.imread(str(f))
    und = cv2.undistort(img, K_factory, dist_factory)
    cv2.imwrite(str(DST_DIR / f.name), und)
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(files)}")
print("done")
