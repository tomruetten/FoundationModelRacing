# track_synth: synthetic track from an occupancy map

Turns a ROS map (PNG + YAML, e.g. `reference/iros2026/`) into an Isaac Sim USD scene with two-layer
white corrugated hoses (30 cm Ø) as the only track boundary, in a placeholder industrial hall, and
collects ego camera frames with exact poses in the format `Code/benchmark` reads.

```bash
# 0. Real-hall backdrop from the camera 3DGS (roboracer env, GPU), once
python -m track_synth.splat_panorama --config data/local/New1/checkpoints/camera_3dgs/config.yml \
    --out reference/iros2026/hall_pano.png
#    (floor texture reference/iros2026/floor_real.jpg is already in git; rebuild it with
#     python -m track_synth.real_textures --frames <ego frames dir> --out reference/iros2026/floor_real.jpg)

# 1. Build scene (plain Python; needs usd-core, skan, opencv-python-headless, pillow, scipy)
python -m track_synth.build --map reference/iros2026/IROS2026.yaml --out outputs/iros2026 \
    --hdri reference/iros2026/hall_pano.png
python -m track_synth.preview outputs/iros2026     # software sanity renders, not photoreal

# 2. Collect data (Isaac Sim 6 python, GPU)
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
- Hose: 30 cm diameter, soft (flattened to 0.72, irregular width, pinched stretches), brown straps every
  ~1.8 m. A second stacked layer covers `--upper-fraction` of the wall (default 0.5, random stretches,
  tapering at the ends) because the real boundary is mostly single-layer; `--upper-fraction 1` stacks everywhere.
  No cones, no floor markings.
- Lighting (with `--hdri`): `collect.py --dome-scale` changes only the visible panorama, `--fill-scale`
  (default 2) only the light on hoses and floor, `--light-scale` both. The fill dome relies on the
  Omniverse attribute `visibleInPrimaryRay=False`; if the background gets brighter with `--fill-scale`,
  that attribute is not honoured and the fill dome has to be dropped.
- With `--hdri` the hall is a 360° render of the New1 3DGS at 1 m height, used as an infinitely
  distant dome (no parallax) and as image-based light. Without it, a placeholder hall is generated.
  If the panorama horizon is tilted or vertical, rebuild with `--dome-rotate-x 90` (USD dome
  orientation on Z-up stages); `--dome-rotate-z` only turns the background.
- Floor: real floor of the hall, inverse-perspective-mapped from 3DGS ego frames and quilted from
  line-free 0.32 m windows (4.1 m tile). Camera: 0.25 m above the floor (vehicle_parameters.txt §6b).
- The full racing pipeline (scene = 3DGS) cannot host this track: the scans cover a ~8 x 6 m track,
  the IROS layout is 52 x 31 m.
- The GRU was trained on other surfaces/tracks; poses are exact simulator states, not real-car predictions.
- `collect.py` has not been run in Isaac Sim yet (no GPU where it was written).
