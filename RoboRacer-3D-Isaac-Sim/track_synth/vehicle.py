"""Car02 (F1TENTH / RoboRacer) constants used by the synthetic track pipeline.

Every value is taken from the repo's own reference data (tests/test_track_synth.py re-checks them
against the files, so they cannot drift silently):

  reference/vehicle/vehicle_parameters.txt           geometry, steering range, sensor mounts
  reference/calibration/camera_intrinsics.yaml       OAK-D Pro W intrinsics ("THE FILE TO USE")
  reference/calibration/extrinsics_candidate_checked_v2.yaml   camera orientation / x,y offset
  reference/raceline_ftm_halle/reftraj_FTM_Halle_min_time.json min-time reference: acceleration limits
  simulation/closed_loop/10_v2_main_demo.py          validated Pure Pursuit settings, GRU training speed range
"""
import math

# --- geometry (vehicle_parameters.txt, section 1) ---------------------------------------------
WHEELBASE = 0.3225        # m
WIDTH = 0.26              # m, over the tyres
LENGTH = 0.42             # m, over the wheels
BASE_Z = 0.05             # m, base_link sits at axle height (tyre radius ~0.048-0.055 m, section 6)

# --- actuation (vehicle_parameters.txt, section 2/5) --------------------------------------------
MAX_STEERING = 0.29       # rad (+-16.6 deg)
KAPPA_STEER_MAX = math.tan(MAX_STEERING) / WHEELBASE      # tightest drivable curvature, 0.90 1/m
DT = 0.02                 # s

# --- speed / acceleration envelope --------------------------------------------------------------
# The Run30 GRU is only valid inside its training range: PurePursuit(train_speed_min=0.3, train_speed_max=2.5)
# in 10_v2_main_demo.py ("clamp speed to training distribution to avoid OOD GRU inputs"). The VESC command
# range itself is 0..3.5 m/s.
SPEED_MIN = 0.3           # m/s
SPEED_MAX = 2.5           # m/s
# Acceleration limits of the Car02 min-time reference raceline (FTM Halle, source: Z. Elnaggar): a_long in
# [-2.5, +2.2] m/s^2, and v^2*|kappa| peaks at 2.8 m/s^2 wherever the speed is curvature limited.
A_ACC = 2.2               # m/s^2
A_DEC = 2.5               # m/s^2
A_LAT = 2.8               # m/s^2

# --- validated Pure Pursuit (10_v2_main_demo.py, config D: 0.033 m mean tracking error with Run30) ---
LOOKAHEAD = 0.5           # m
LD_MIN, LD_MAX, LD_CURVE_K = 0.3, 2.0, 3.0   # adaptive lookahead: ld = clip(LOOKAHEAD / (1 + k*|kappa|), min, max)

# --- camera (OAK-D Pro W) -----------------------------------------------------------------------
# Intrinsics: camera_intrinsics.yaml, rgb, 1280x800, undistorted pinhole. The racing config uses 640x400.
INTRINSICS_1280 = dict(width=1280, height=800, fx=566.2562, fy=565.9493, cx=641.9587, cy=388.6107)
# Mount: extrinsics_candidate_checked_v2.yaml base_link_to_oak_rgb_camera_optical_frame (calibrated):
# x, y offset and orientation (optical RDF -> base_link FLU, quaternion xyzw). Pitch +5.3 deg, yaw -1.3, roll -1.3.
CAMERA_XY = (0.29266665, 0.06987346)
CAMERA_QUATERNION = (-0.46552296, 0.48718884, -0.52270603, 0.52220090)
# Height: the yaml composes z = 0.0656 m and vehicle_parameters.txt states 0.200 m over base_link, which
# disagree (0.12 m vs 0.25 m above the floor). The recorded scenes measure it directly: camera path Z in
# [0.15, 0.21] m, mean 0.18 m (02_camera_splat_client_isaac.py), New1 floor-plane fit 0.15 +- 0.07 m (README).
CAMERA_HEIGHT = 0.18      # m above the floor


def camera_intrinsics(scale=0.5):
    """Intrinsics at `scale` x the 1280x800 calibration (0.5 -> the 640x400 racing camera)."""
    return {k: (round(v * scale) if k in ('width', 'height') else v * scale) for k, v in INTRINSICS_1280.items()}


def camera_mount(height=CAMERA_HEIGHT):
    """Mount dict in the racing/sensor_pose format; z is relative to base_link (BASE_Z above the floor)."""
    return dict(translation=[CAMERA_XY[0], CAMERA_XY[1], height - BASE_Z], quaternion=list(CAMERA_QUATERNION))
