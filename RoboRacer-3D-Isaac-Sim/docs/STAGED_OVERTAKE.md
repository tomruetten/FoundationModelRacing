# Staged Porsche overtake

[Watch the local 19-second 4K demo](../outputs/staged_overtake/staged_overtake_4k.mp4).
The dark F1Tenth ego follows the miniature Porsche, moves left, accelerates past,
returns to its line and continues around the New1 track. The longitudinal order
changes at 5.1 seconds. The ego travels 29.50 m and returns within 8 cm of its
starting position at 16.02 seconds before continuing.

## Reproduce

Use the `roboracer` environment with CUDA, gsplat and ffmpeg. Prepare the
[Porsche and HD hall assets](PORSCHE_HALL_DEMO.md) first. This replay also needs
`data/local/objects/f1tenth.npz`, the New1 dataset transforms and floor plane,
`checkpoints/camera_3dgs_flat/config.yml` and `checkpoints/lidar_3dgs/best_flat.pt`
under `data/local/New1`, and the 50 scan templates in
`outputs/learned_demo_5s_livox_black/scans/`. These learned checkpoints, source
recordings and generated outputs are local data, not included in Git.

```bash
conda activate roboracer
export PYTHONNOUSERSITE=1
python scripts/render_staged_overtake.py --preview
python scripts/render_staged_overtake.py
```

The preview saves eight milestone images in `outputs/staged_overtake/preview`.
The full run writes `staged_overtake_4k.mp4`, `trajectory.json`,
`render_metrics.json`, milestone images, synthetic audio and 190 LiDAR scans
under `outputs/staged_overtake`. Use `--out PATH` for a separate run.
The output is 3840×2160 at 30 fps; the main camera is rendered at 2560×1600.
`--blur-samples 1` disables temporal integration; the default is three samples
over `--exposure-ms 8`, averaged in linear light.

## Scene and motion

The replay uses the floor-corrected camera and LiDAR checkpoints, places the
cars on the measured floor, and adds contact shadows. Distance drives wheel
rotation; steering follows path curvature. Acceleration drives restrained body
pitch/roll and camera-mount response. HD wooden props stay outside the planned
driving paths. Silver Porsche paint and the original dark ego colours are retained.

The inset is a rendered rear camera, switching to a right-facing camera during
the side-by-side phase. The minimap and signed gap use the same world trajectories
as the camera and LiDAR. Electric motor and rolling sounds are synthesized from
the vehicle speeds; they are not recorded audio.

`scripts/plan_staged_overtake.py` creates the shared trajectories at 100 Hz from
the smoothed recorded track, including position, heading, speed, steering,
acceleration, odometer and progress. Quintic transitions control the lateral
manoeuvre and lead change. Both vehicles keep moving throughout the sequence.

The planner checks finite motion, speed/acceleration/steering bounds and oriented
vehicle footprints every 10 ms. This run has a minimum separating-axis gap of
0.169 m between the two footprint proxies and a conservative 0.352 m clearance
from inserted props. Ego speed is 1.15–1.94 m/s; Porsche speed is 0.74–2.00 m/s.
These checks do not validate collisions against the entire reconstructed hall.

## Interpretation and credits

This is an explicitly staged kinematic demonstration, not reactive opponent AI
or a closed-loop learned-dynamics race. It does not change the interactive racing
application. Thirty fps describes offline video playback, not real-time performance.

LiDAR comes from the separate LiDAR Gaussian map, with the moving Porsche and
fixed props inserted. Its 3D and BEV panels show the same sensor-frame points.
Return directions are taken from the August recording and repeat every five
seconds; they are not New1 ground truth or a complete Livox firing pattern.
Ego self returns are excluded. Learned-map geometry streaks and poorly observed
camera regions remain; people in the captured hall are static reconstruction.

Porsche: **Lionsharp Studios**, [Porsche 911 Carrera 4S](https://sketchfab.com/3d-models/free-porsche-911-carrera-4s-d01b254483794de3819786d93e0e1ebf),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), modified for scale,
Gaussian conversion, materials and articulated wheels. The hall props are CC0;
full sources and conversion details are in [the asset documentation](PORSCHE_HALL_DEMO.md).
