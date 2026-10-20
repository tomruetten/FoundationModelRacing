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

Pipeline: wall lines of the map are skeletonised -> hose centre lines; free band -> medial-axis
centre line (closed, 0.1 m) and collision polygons. Each episode starts at a random point on the
centre line with lateral/heading offset and a random speed; `RaceEngine` autopilot follows the
centre line with the GRU (or `--kinematic`) dynamics. Camera = the OAK-D intrinsics and mount from
`configs/racing_142014.json`.

## Assumptions to check
- Map wall lines are taken as hose **centres**. The free width is ~2.2 m between the drawn lines,
  so about 1.9 m between hose surfaces.
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
  0.32 m windows (`real_textures.py`, 4.1 m tile). Camera: 0.25 m above the floor (vehicle_parameters.txt
  §6b), pitched ~3 deg up; `--camera-height` overrides.
- `--hdri` panorama = 360° render of a hall 3DGS at 1 m height, an infinitely distant dome (no parallax). If its
  horizon is tilted, rebuild with `--dome-rotate-x 90`.
- The full racing pipeline (scene = 3DGS) cannot host this track: the scans cover a ~8 x 6 m track,
  the IROS layout is 52 x 31 m.
- The GRU was trained on other surfaces/tracks; poses are exact simulator states, not real-car predictions.
- Light intensities are uncalibrated; tune with `--light-scale/--fill-scale/--dome-scale` at collect time.
