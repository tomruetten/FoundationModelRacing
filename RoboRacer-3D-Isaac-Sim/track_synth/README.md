# track_synth: synthetic track from an occupancy map

Turns a ROS map (PNG + YAML, e.g. `reference/iros2026/`) into an Isaac Sim USD scene with two-layer
white corrugated hoses (30 cm Ø) as the only track boundary, in a placeholder industrial hall, and
collects ego camera frames with exact poses in the format `Code/benchmark` reads.

```bash
# 1. Build scene (plain Python; needs usd-core, skan, opencv-python-headless, pillow, scipy).
#    Default background: simple placeholder hall (walls, beams, a few boxes).
python -m track_synth.build --map reference/iros2026/IROS2026.yaml --out outputs/iros2026
python -m track_synth.preview outputs/iros2026     # software sanity renders, not photoreal

# 1b. Optional: real-hall backdrop instead of the placeholder (GPU, roboracer env)
python -m track_synth.splat_panorama --config <camera_3dgs config.yml> [--data-dir <dir with transforms.json>] \
    --out reference/iros2026/hall_pano.png
python -m track_synth.build ... --hdri reference/iros2026/hall_pano.png

# 2. Collect data (Isaac Sim python, GPU)
PYTHONPATH=. python -m track_synth.collect --scene outputs/iros2026 \
    --dynamics-checkpoint dynamics/deployed/trained_models/gru_run30_best.pth \
    --episodes 20 --seconds 25 --output outputs/iros2026_data
```

Vehicle values: all constants live in [vehicle.py](vehicle.py), each taken from the repo's reference data
(wheelbase 0.3225 m, 0.26 x 0.42 m, steering +-0.29 rad, camera intrinsics/orientation from `reference/calibration`,
speed range 0.3-2.5 m/s of the GRU training data, a_long +2.2/-2.5 and a_lat 2.8 m/s^2 of the Car02 min-time
reference raceline, the validated adaptive-lookahead Pure Pursuit of `closed_loop/10_v2_main_demo.py`);
`tests/test_track_synth.py` re-checks them against those files.

Driving line: `build` also computes a lap-time-oriented racing line (bounded least squares, minimum curvature
blended with shortest path, curvature kept below the steering limit) and a friction-limited speed profile
(`--v-max 2.5 --a-lat 2.8 --a-acc 2.2 --a-dec 2.5`); `preview` plots it (`raceline.png`). `collect --driver raceline`
(default) follows it with that Pure Pursuit at 0.45-0.6x the profile speed (`--speed-scale-min/max`, about 1.0-1.5 m/s),
starts rolling at that speed with small lateral / heading offsets (`--max-lateral 0.15`, `--yaw-noise 0.05`) and
drives `--lead-in 2` s before the first frame. `--driver centerline` = constant speed on the centre line.

Pipeline: wall lines of the map are skeletonised -> hose centre lines; free band -> medial-axis
centre line (closed, 0.1 m) and collision polygons. Each episode starts at a random point on the
centre line with lateral/heading offset and a random speed; `RaceEngine` autopilot follows the
centre line with the GRU (or `--kinematic`) dynamics. Camera = the OAK-D intrinsics and mount from
`configs/racing_142014.json`.

## Assumptions to check
- Map wall lines are taken as hose **centres**. The free width is ~2.2 m between the drawn lines,
  so about 1.9 m between hose surfaces.
- Camera height 0.18 m above the floor is a choice between conflicting numbers in the reference data
  (vehicle_parameters.txt: 0.200 m over base_link = 0.25 m above the floor; extrinsics yaml: 0.0656 m = 0.12 m;
  measured in the recorded scenes: path Z 0.15-0.21 m, floor-plane fit 0.15 +- 0.07 m). `collect --camera-height`
  overrides it; the floor texture assumes the same height (`real_textures.py`).
- Hose (modelled on data/frames): fat padded tubes, 30 cm wide and ~20 cm tall (flatten 0.68), cream-white,
  fine shallow ripples, hardly any irregularity, thin dark-brown straps every ~1.8 m, two stacked layers
  everywhere by default (upper one slightly smaller). The real boundary is mostly
  single-layer: `--upper-fraction 0.5` at build time stacks only half of the wall (random stretches
  that taper out). No cones, no floor markings.
- Lighting: dome + two invisible overhead DistantLights (both with the placeholder hall and with `--hdri`,
  where the dome is the panorama = visible background + image-based light). `collect.py --dome-scale` changes the panorama only, `--fill-scale`
  the distant lights only (hoses/floor), `--light-scale` all of them. A second DomeLight must not be
  used: RTX picks a single environment and the background turned black.
- Floor: real floor of the hall, inverse-perspective-mapped from 3DGS ego frames and quilted from line-free
  0.32 m windows (`real_textures.py`, 4.1 m tile). Camera: 0.18 m above the floor, calibrated orientation
  (pitched 5.3 deg up, yaw -1.3, roll -1.3); `--camera-height` overrides.
- `--hdri` panorama = 360° render of a hall 3DGS at 1 m height, an infinitely distant dome (no parallax). If its
  horizon is tilted, rebuild with `--dome-rotate-x 90`.
- The full racing pipeline (scene = 3DGS) cannot host this track: the scans cover a ~8 x 6 m track,
  the IROS layout is 52 x 31 m.
- Speed envelope with the Run30 GRU in the loop (measured with `collect.episodes` on this track, 9 episodes of 25 s
  per setting): speed scale 0.4 and 0.6 (mean 0.9 / 1.35 m/s) all complete without contact (min hose clearance
  0.17-0.27 m); 0.75 (1.6 m/s) ends in contact in 6-7 of 9 episodes, 0.88 and 1.0 in all of them (the car snakes with
  +-1.5 rad/s yaw rate, a longer lookahead helps only a little). The repo validated this controller at 0.8-1.0 m/s.
  So the 77 s lap time of the profile is not reachable with this GRU; the achievable lap is ~130 s.
  Yaw damping (`--yaw-damping`), steering low-pass and a longer lookahead did not fix it (best: 5/9 contacts at
  1.9 m/s): the GRU's yaw-rate gain is 1.3-1.6x the bicycle model's at larger steering angles and it produces yaw
  rates of 3-5 rad/s (physical maximum ~1.8) once the loop oscillates, i.e. the closed loop leaves its training data.
  A kinematic model follows the line at full speed (8 episodes, `--kinematic --speed-scale-min 0.9 --speed-scale-max 1.0`:
  2.3-2.5 m/s mean, min hose clearance 0.36 m, no slip modelled). The vehicle itself is not the limit: VESC command
  range 0-3.5 m/s, GRU training range 0.3-2.5 m/s, Car02 min-time reference up to 2.8 m/s.
- Collision polygons sit on the hose surface (free-space edge shrunk by hose radius - half line width); the racing line
  keeps 0.28 m between the car side and the hose surface. `trajectory.jsonl` rows carry `hose_clearance_m`,
  `dataset.json` the per-episode minimum.
- The racing line is a minimum-curvature / shortest-path blend scored by lap time, not a full minimum-time
  optimal control solution; speed limits (`a_lat` etc.) are assumptions, not measured Car02 grip. The tightest
  corner needs ~0.8 1/m (radius 1.2 m) against a steering limit of ~0.9 1/m: little margin for the learned dynamics.
- The GRU was trained on other surfaces/tracks; poses are exact simulator states, not real-car predictions.
- Exposure is fixed by default (`--auto-exposure` opts in to the temporal one). Auto exposure adapts over
  time, so the same pose gets different brightness depending on the frames before, and it cancels light changes.
  `collect` prints `probe: ...` (4 renders of the first pose); a spread above 3 % means brightness still drifts.
- Light intensities are uncalibrated; tune with `--light-scale/--fill-scale/--dome-scale` at collect time.
