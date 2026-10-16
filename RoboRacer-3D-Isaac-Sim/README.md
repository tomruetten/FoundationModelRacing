# RoboRacer 3D · Isaac Sim

Digital twin of the F1TENTH/RoboRacer **Car02** for closed-loop simulation in NVIDIA Isaac Sim.
Three learned components are combined:

| Component | Method | Input | Deployed result |
|---|---|---|---|
| **Camera sensor** | 3D Gaussian Splatting (nerfstudio `splatfacto`) | OAK-D Pro W video, COLMAP poses | PSNR 27.2 dB |
| **LiDAR sensor** | Custom 3DGS with ray-drop head (gsplat) | Livox Mid-360 scans, COLMAP poses | Chamfer 26.3 cm, F@10cm 46 % |
| **Vehicle dynamics** | GRU · Deep Dynamics (Pacejka) · MS-NN (nnodely) | 20 ms telemetry (MoCap, IMU, VESC) | GRU 3ch CL-RMSE 0.445 rad/s · MS-NN yaw-rate RMSE 0.116 rad/s |

Isaac Sim drives the car along a trajectory (replay or Pure-Pursuit closed loop), and at every
step the splat renderers produce the camera image and LiDAR scan the car "sees", while the
learned dynamics model predicts the next state.

```
                 ┌──────────────────────────────┐
                 │  Isaac Sim (env_isaaclab)    │
                 │  scene · car mesh · control  │
                 └──────┬──────────┬────────┬───┘
      c2w pose (json)   │          │        │   state + cmd (json)
                        ▼          ▼        ▼
   ┌────────────────┐ ┌────────────────┐ ┌──────────────────────┐
   │ camera splat   │ │ LiDAR splat    │ │ dynamics server      │
   │ render server  │ │ render server  │ │ GRU / MS-NN / DD     │
   └────────────────┘ └────────────────┘ └──────────────────────┘
            all three run in the single `roboracer` conda env
            IPC = file queue (request_N.json → response_N.png/json)
```

---

## Contents

- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data](#data)
- [Quick start](#quick-start)
  - [1. Camera 3DGS](#1-camera-3dgs)
  - [2. LiDAR 3DGS](#2-lidar-3dgs)
  - [3. Vehicle dynamics](#3-vehicle-dynamics)
  - [4. Isaac Sim integration](#4-isaac-sim-integration)
    - [Opponent placement, vehicles, and demo video](#opponent-vehicles-as-gaussian-splats)
- [Evaluation](#evaluation)
- [Vehicle and sensors](#vehicle-and-sensors)
- [Data-split rule](#data-split-rule)
- [Troubleshooting](#troubleshooting)
- [Legacy environment names](#legacy-environment-names)

---

## Repository layout

```
.
├── README.md                 ← this file
├── environment.yml           ← unified conda env (documentation; use scripts/setup_env.sh)
├── scripts/                  ← setup, data linking, Isaac Sim demo, splat converters, 3DGRUT pipeline, evaluation
├── evaluation/               ← camera / LiDAR / dynamics evaluation suite (scripts/run_evaluation.sh)
├── perception/
│   ├── camera_gs/            undistort → COLMAP → splatfacto (camera 3DGS)
│   └── lidar_gs/             T_cl calibration → sync → point cloud → LiDAR 3DGS
├── dynamics/
│   ├── training/             bag → CSV → GRU / Deep Dynamics / MS-NN training
│   ├── msnn/                 MS-NN reference code (src/), paper notebooks, evaluation
│   ├── deployed/             deployed checkpoints + minimal inference API (GRU, MS-NN)
│   └── retraining_20260815/  retraining experiments
├── simulation/
│   ├── isaac_sim/            render servers, dynamics servers, Isaac Sim clients (replay)
│   └── closed_loop/          Pure-Pursuit closed loop with GRU/MS-NN plant
├── racing/, configs/, tests/ ← racing runtime package (see docs/F1TENTH_RACING.md)
├── reference/
│   ├── calibration/          camera intrinsics/extrinsics, TF usage
│   ├── raceline_ftm_halle/   track + min-time raceline
│   └── vehicle/              measured vehicle parameters + photos
├── envs/legacy/              ← the original per-tool conda env files (superseded by environment.yml)
├── docs/                     ← legacy package overview (German), racing docs, opponents (OPPONENTS.md), evaluation report
├── data/local/               ← git-ignored local copies of checkpoints, poses, CSVs, scenes (see "Data")
└── third_party/3dgrut/       ← git-ignored clone of NVIDIA 3DGRUT + its uv venv
```

Every pipeline folder has its own `README.md` with step-by-step instructions and expected
numbers. This top-level README tells you how to set up the machine and how the pieces fit;
the folder READMEs are the detailed protocol.

---

## Requirements

| | Tested |
|---|---|
| OS | Ubuntu 22.04 LTS |
| GPU | NVIDIA RTX 4090 / 4080 SUPER (≥ 16 GB VRAM recommended, ≥ 8 GB minimum) |
| Driver | ≥ 550 (CUDA 12.4 runtime wheels); 570 tested |
| conda | Miniconda ≥ 24 |
| Disk | ~12 GB for the `roboracer` env, ~30 GB for Isaac Sim, plus your data |
| Isaac Sim | **6.1** (pip, `env_isaac6`, `scripts/setup_isaac6.sh`) for the native gaussian-splat scene. 5.1 (`env_isaaclab`) also works but only with the baked point-cloud scene. Clients use the `isaacsim.core.*` API with a legacy fallback. |

ROS 2 Humble is only needed to record new bags. Reading existing `.db3` bags is done with
`sqlite3` directly, no ROS install required.

---

## Installation

### 1. Unified `roboracer` environment (camera GS + LiDAR GS + all dynamics models)

```bash
git clone <this repo> RoboRacer-3D-Isaac-Sim
cd RoboRacer-3D-Isaac-Sim
bash scripts/setup_env.sh        # ~10-15 min, ~12 GB
conda activate roboracer
python scripts/check_env.py      # all checks should print [ok]
bash   scripts/smoke_test.sh     # loads the shipped GRU + MS-NN checkpoints, runs a few steps
```

What the script installs, in this order (order matters):

1. Python 3.10, ffmpeg (conda-forge)
2. `torch 2.6.0+cu124`, `torchvision 0.21.0+cu124`
3. `nerfstudio 1.1.5` (camera 3DGS)
4. `gsplat 1.5.3+pt24cu124` prebuilt wheel (LiDAR 3DGS; skips a 20-min CUDA build)
5. `open3d 0.19`, scientific stack, `casadi`
6. `nnodely 1.5.4` **with `--no-deps`** (MS-NN). nnodely pins `torch==2.2.2` on
   Python 3.10 x86-64, which is the only reason the original project used a separate
   CPU-only env. nnodely runs fine on torch 2.6, so we skip its pin.
7. `colmap` 3.13 from conda-forge (no sudo needed). If you have sudo, `apt install colmap`
   (3.7) also works. All scripts call COLMAP with `--use_gpu 0`, so either build is fine.

The script also sets `PYTHONNOUSERSITE=1` on the env so that packages under
`~/.local/lib/python3.10/site-packages` (from earlier `pip install --user`) cannot shadow
the env. Re-running the script is safe; it only installs what is missing.

Changes made to the original code for this environment (torch 2.6 / nnodely 1.5.4):

- `torch.load(..., weights_only=False)` added where checkpoints contain numpy objects
  (torch 2.6 changed the default and refused to load the GRU/LiDAR checkpoints).
- `dynamics/deployed/13_inference_msnn.py` synced with the fixed version from
  `dynamics/msnn/` (unique nnodely dataset names per step; the old copy crashed on the
  second `step()` call).

### 2. Isaac Sim

Isaac Sim ships its own Python and CUDA stack and cannot share a process with
nerfstudio/gsplat. It is therefore **not** part of `roboracer`; it talks to the render and
dynamics servers over a file queue.

**Isaac Sim 6.1** (recommended, renders the trained 3DGS scene natively):

```bash
bash scripts/setup_isaac6.sh        # conda env env_isaac6, ~26 GB, ~15 min
```

**Isaac Sim 5.1** is already installed on the lab PC in `env_isaaclab` (Python 3.11, with
Isaac Lab 0.48). Fine for everything except the native splat scene:

```bash
conda create -n env_isaaclab python=3.11 -y && conda activate env_isaaclab
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
pip install numpy pandas matplotlib scipy torch
```

The Isaac Sim client scripts (`simulation/isaac_sim/*_client_isaac.py`, `07_dynamics_isaac_demo.py`)
run inside one of these envs. Everything else runs in `roboracer`.

### 3. Data paths

```bash
cp scripts/env.sh.example scripts/env.sh   # edit DATA_DRIVE_ROOT etc. for your machine
source scripts/env.sh                      # in every shell
bash scripts/link_data.sh                  # once: symlinks into the repo (see "Data")
```

---

## Data

Raw data is **not** in git. Two tiers:

- **`data/local/`** (git-ignored, ~14 GB): fast local copies of everything the scripts need
  at run time. Checkpoints, COLMAP poses, USD scenes, dynamics CSVs, trained runs, the
  newest pipeline scripts. Keeps the pipelines usable when the drive is unplugged.
- **External Samsung T7 drive** `/media/alienware_felix/T7/roboracer_data/` (~65 GB): the full
  project working directory with raw ROS 2 bags, videos and frame dumps. Only needed to re-run
  COLMAP / training from scratch.

`scripts/env.sh` (git-ignored, machine-specific) points every variable at the local tier and
keeps `DATA_DRIVE_ROOT` / `RAW_SESSION_*` for the drive. `scripts/link_data.sh` creates the
symlinks the scripts expect:

| Symlink in repo | Target | Used by |
|---|---|---|
| `{dynamics/training,dynamics/msnn,dynamics/msnn/src,simulation/closed_loop}/processed_data` and `data/processed_data` | `data/local/processed_data/` (97 CSVs) | all dynamics scripts |
| `data/New1`, `data/Probe`, `data/drive` | `data/local/New1`, `data/local/Probe`, T7 drive | convenience |
| `dynamics/msnn/trained_models` | `../dynamics/deployed/trained_models` (relative, in git) | `14_eval_and_plot.py`, `15_gru_inference.py` |

```bash
cp scripts/env.sh.example scripts/env.sh   # adjust DATA_DRIVE_ROOT if the drive is mounted elsewhere
source scripts/env.sh && bash scripts/link_data.sh
```

### data/local layout

```
data/local/
├── New1/                         2026-07-14: checkpoints/camera_3dgs (27.61 dB), checkpoints/lidar_3dgs, dataset/transforms.json, scene.usda
├── Probe/                        vocab_tree.bin, experiments/isaac_sim_export (track meshes), old LiDAR checkpoints
├── processed_data/               dynamics CSVs at 20 ms (see below)
├── dynamics_trained_models/      every GRU / Deep Dynamics / MS-NN run from ma_jonathanmohr-main
├── sessions/
│   ├── session_20260827_142014_noflare/   COLMAP poses, splatfacto ckpt (27.0 dB), LiDAR GS, USD scenes
│   ├── session_20260827_143212_noflare/   same for the dynamics track (27.4 dB)
│   └── session_20260904_161841_zhuitong/  splatfacto ckpts (27.5 dB, 150k steps), USD scenes with car
├── pipeline_scripts_1707/        the current session pipeline (paths still hard-coded to the training machine's home)
└── calibration_summary/          camera intrinsics, T_cl, base_link↔livox/mocap reports
```

Re-populate from the drive with the `rsync` calls in `scripts/sync_local_data.sh`.

### What is on the drive

| Drive folder | Content | State |
|---|---|---|
| `Zusammenfassung/` | the code package (= the pipeline folders here, slightly older) + `5_MSNN_for_Lab/Code von Felix/ma_jonathanmohr-main/` with **all dynamics CSVs** and **all GRU / Deep Dynamics / MS-NN training runs** (`trained_models/`) | complete |
| `New1/` | 2026-07-14 recording: camera 3DGS checkpoint (27.61 dB, `checkpoints/camera_3dgs/config.yml`), LiDAR 3DGS checkpoint (13.44 cm, `checkpoints/lidar_3dgs/best.pt`), COLMAP `dataset/transforms.json` (1034 frames, images in `dataset/images_real/`), `scene.usda`, track meshes | checkpoints + poses, no bag/video |
| `27.08新赛道和Dynamik/` | 2026-08-27 new track, two sessions `142014` (3DGS) and `143212` (dynamics), each with `all_data` ROS 2 bag, `camera.mp4`, `odometry_bag`, COLMAP, splatfacto checkpoints (`*_noflare` = warm-up frames removed, PSNR 27.0 / 27.4 dB), LiDAR 3DGS experiments, Isaac Sim 6 USD scenes | complete raw + results |
| `07,09,2026 New 3DGS/zhui tong session_20260904_161841/` | 2026-09-04 recording at ~60 fps, 1.4 GB bag, splatfacto 150k steps (27.5 dB), USD scenes with car | complete raw + results |
| `1707-Test/` | 2026-07-17 recording with MoCap (`OKsession_20260717_153729`, bag + video + checkpoints) and **`scripts/`: the current Aug/Sep 2026 pipeline** (`extract_undistort_frames_full.py`, `colmap_to_transforms_full.py`, `colmap_points_to_ply.py`, `train_lidar_3dgs_okession_v2.py`, `isaac6_*.py`, `bake_full_scene.py`, ...) | complete |
| `Probe/` | first project phase (layout02). Only `colmap_undistort/vocab_tree.bin`, LiDAR checkpoints and `experiments/isaac_sim_export/` meshes remain; **raw frames and the March LiDAR bag are gone** | partial |
| `calibration_summary/`, `reference_repos/`, `ros2_ws/`, `tools/` | calibration reports, AutoDRIVE F1TENTH model, Livox ROS 2 workspace, node/npm | reference |

Consequence: the scripts in `perception/camera_gs/` and `perception/lidar_gs/` (Probe layout,
`$PROBE_ROOT/colmap_out/images`, `run_20260327_105224_0.db3`) cannot be re-run from raw data
any more. New recordings should use the session-based pipeline in `$DATA_DRIVE_ROOT/1707-Test/scripts/`
(entry points: `run_full_pipeline_20260827.sh` and `run_full_pipeline_20260907.sh` inside the
session folders, all paths hard-coded to the training machine's home directory). Porting those
scripts into this repo is the natural next step.

### Dynamics CSV layout

```
$DATA_DIR/                              20 ms CSVs, 34 columns (MoCap pose/vel/acc, odom, VESC, IMU, cmds)
├── session_20260721_20ms.csv           Day 1 — TEST ONLY
├── session_20260722_*_20ms.csv         Day 2 — train/val (3 sessions, segmented)
├── session_20260728_*_20ms.csv         Day 3 — train/val (exclude 124427, NaN IMU)
└── session_20260827_143212_{train,val}_20ms.csv   new track, 2026-08-27
```

Small reference data **is** in git: camera intrinsics/extrinsics
(`reference/calibration/`), the FTM-Halle track and min-time raceline,
the deployed GRU and MS-NN checkpoints (`dynamics/deployed/trained_models/`), and the
measured vehicle parameters.

---

## Quick start

All commands below assume `conda activate roboracer` and `source scripts/env.sh`.
Scripts are run directly by path (folders start with digits, so they are not importable
packages).

### 1. Camera 3DGS

Detailed protocol: `perception/camera_gs/README.md`.

```bash
cd perception/camera_gs
python 01_undistort_frames.py          # OAK-D rational-polynomial → pinhole, crop to 1280×600
bash   03_colmap_commands.sh           # SIFT (CPU) + sequential + vocab-tree loop closure
bash   04_train_and_eval.sh            # ns-train splatfacto, 80k it, then ns-eval
```

Skip the first ~1 s of every new recording (auto-exposure warm-up frames cost ~0.3 dB).
Expected: ≥ 900 registered images, PSNR ≈ 27 dB after ~2 h on an RTX 3090-class GPU.

### 2. LiDAR 3DGS

Detailed protocol: `perception/lidar_gs/README.md`. Requires the camera COLMAP
`transforms.json` from step 1.

```bash
cd perception/lidar_gs
python 02_sync_lidar_camera.py                                   # → camera_sync.csv
python 04_train_lidar_3dgs.py --use-colmap --steps 30000 \
       --densify --raydrop --exp lidar_3dgs_colmap_v1            # ~45 min
python 04_train_lidar_3dgs.py --use-colmap --densify --raydrop \
       --render --exp lidar_3dgs_colmap_v1                       # Chamfer / F-score
```

The camera→LiDAR extrinsic `T_cl` is baked into the script (AprilGrid calibration
2026-07-14, ROS child-frame convention, transposed internally). Re-calibrate with
`01_calibrate_tcl.py` only if the sensor mount changed.

### 3. Vehicle dynamics

Detailed protocol: `dynamics/training/README.md` and
`dynamics/msnn/README.md`. Shared modules (`Dataset_Classes.py`,
`RNN_architectures.py`, `training.py`) live in `dynamics/msnn/src/`.

```bash
cd dynamics/training
python 01_extract_bag.py               # ROS 2 bag → CSV
python 02_data_preparation.py          # v_x_local, a_x_local, ... at 20 ms
python 03_mirror_augmentation.py       # δ→−δ, v_y→−v_y, ψ̇→−ψ̇  (GRU ONLY, never MS-NN)

python 06_train_gru_run30_best.py      # GRU, 7 in / 5 out, ~15-20 min GPU
python 10_train_deep_dynamics_v8_best.py   # Pacejka Deep Dynamics
python 12_train_msnn_exp9_best.py      # MS-NN exp9, Phase 1+2 only, ~35 min (CPU-bound)

python 08_eval_closed_loop.py          # GRU / DD closed-loop RMSE on Day 1
cd ../dynamics/msnn && python 14_eval_and_plot.py --data-dir "$DATA_DIR"   # MS-NN vs GRU
```

Inference APIs for use in a simulator loop (both in `dynamics/deployed/`):

```python
from gru_inference import GRUInference          # torch
gru = GRUInference(checkpoint='trained_models/gru_run30_best.pth'); gru.reset()
out = gru.step(vx=2.0, vy=0.0, yr=0.1, ax=0.2, ay=-0.1, speed_cmd=2.0, steering_cmd=0.05)

from inference_msnn import MSNNInference        # nnodely
msnn = MSNNInference(model_dir='trained_models/msnn_exp9_phase2_backup'); msnn.reset()
yr, ax = msnn.step(delta=0.05, vel=2.0, acc=0.2, accy=-0.1, crt=3.0)
```

### 4. Isaac Sim integration

Detailed protocol: `simulation/isaac_sim/README.md` (V1 replay) and
`simulation/closed_loop/README.md` (V2 closed loop).

**What you see vs. what the algorithms see.** The Isaac Sim viewport shows the trained 3DGS
scene itself (native gaussian-splat prim in Isaac Sim 6, colored point cloud in 5.1) plus the
car. That is only the visual layer. The sensor data an algorithm would consume comes from the
external models over the file queue: camera images from the splat render server, LiDAR scans
from the LiDAR-3DGS server, and the vehicle state from the GRU / MS-NN dynamics servers.
Isaac Sim's own renderer and physics are never used as ground truth.

Scene files for the 2026-07-14 recording (both git-ignored, rebuild with the converters):

| Scene | Built by | Isaac Sim | Look |
|---|---|---|---|
| `data/local/New1/scene_splat_gsplat.usd` | `scripts/splat_to_usd_gsplat.py` (runs in `env_isaac6`) | ≥ 6.0 | photoreal from car height, blurry from far novel views |
| `data/local/New1/scene_splat_points.usda` | `scripts/splat_to_usd_points.py` (runs in `roboracer`) | any | colored point cloud |
| `data/local/New1/scene.usda` | earlier LiDAR floor mesh | any | grey mesh |

Both converters take `--to-world` with the checkpoint's `dataparser_transforms.json` and the
dataset `transforms.json`, so the gaussians land in the same world frame as the recorded
trajectory. The gsplat converter keeps the gaussians untouched in `*_raw.usd` and puts the
world transform on a wrapper Xform: rotating the gaussians themselves would also require
rotating their spherical-harmonics coefficients, otherwise white surfaces get green/purple
casts. Filters (`--max-scale`, `--z-range`, `--around-trajectory`) remove floaters. Keep
them mild for the onboard view (`--max-scale 3 --margin 25`, no `--z-range`): aggressive
cropping removes the ceiling and far walls and the viewport turns grey. Isaac's splat renderer (ray-traced, RealTimePathTracing mode) keeps a fine grain that no
RTX setting removes (tested AA/DLSS modes, path tracing up to 256 spp, gaussian accumulation
caps, denoisers); geometry and colour match the gsplat render server.

**One command** (V1 replay: car follows the recorded 2026-07-14 trajectory, every pose is
rendered by the camera 3DGS model; picks `env_isaac6` + splat scene when installed):

```bash
source scripts/env.sh
bash scripts/run_isaac_demo.sh --gui --stride 5     # Isaac Sim window, onboard (F1TENTH camera) view
bash scripts/run_isaac_demo.sh --gui --view chase   # low third-person view behind the car
bash scripts/run_isaac_demo.sh                      # headless, all frames to outputs/isaac_demo_*/frames
```

With `--gui` two windows open: the Isaac Sim viewport (3DGS scene + blue car marker; `--view
onboard` = a USD camera with the OAK-D intrinsics posed exactly like the recorded camera, default;
`chase` = low view behind the car; `overview`) and a live viewer showing the 3DGS render the car sees.
The Isaac viewport never shows the splat itself: that is by design of the V1 file-queue
architecture. The render server is ready after ~5 s, Isaac Sim needs 1–2 min. Frames land in
`outputs/isaac_demo_<timestamp>/frames/`. Set `TRANSFORMS=` / `SCENE=` to use another
session's `transforms.json` and USD scene.

**Manual**, terminal 1–3 in `roboracer` (start these first, wait for "watching ..."):

```bash
cd simulation/isaac_sim
python 01_camera_render_server.py --queue-dir /tmp/isaac_queue/camera --config "$CAMERA_3DGS_CONFIG"
python 03_lidar_render_server.py  --queue-dir /tmp/isaac_queue/lidar --checkpoint "$LIDAR_3DGS_CKPT" \
                                  --no-raydrop --alpha-thresh 0.99
python 08_gru_mocap_dynamics_server.py --checkpoint ../dynamics/deployed/trained_models/gru_run30_best.pth \
                                       --queue-dir /tmp/isaac_queue/gru
python 05_msnn_dynamics_server.py --model ../dynamics/deployed/trained_models/msnn_exp9_phase2_backup \
                                  --queue-dir /tmp/isaac_queue/msnn
```

`--config` may point at any nerfstudio splatfacto `config.yml`; the renderer relocates the
training-machine paths stored inside it (see `scripts/env.sh`, `DATA_DRIVE_ROOT`). The New1 LiDAR
checkpoint has no ray-drop head, hence `--no-raydrop --alpha-thresh 0.99`; the Probe-era
`lidar_3dgs_v7_raydrop` checkpoints use the default ray-drop mask instead.

Use `08_gru_mocap_dynamics_server.py` for the shipped `gru_run30_best.pth` (7 inputs / 5
outputs). `06_gru_dynamics_server.py` expects the older onboard-only 6-in/4-out checkpoint,
which is not in this repo. Render servers use `request_N.json` → `response_N.png`; dynamics
servers use `req_NNNNNN.json` → `res_NNNNNN.json`.

Terminal 4, `env_isaaclab`:

```bash
conda activate env_isaaclab
cd simulation/isaac_sim
OMNI_KIT_ACCEPT_EULA=yes python 02_camera_splat_client_isaac.py --gui --stride 5 \
    --transforms "$NEW1_ROOT/dataset/transforms.json" --scene "$NEW1_ROOT/scene.usda" \
    --queue-dir /tmp/isaac_queue/camera --output-dir outputs/camera_splat_demo
```

Camera + LiDAR together: `04_lidar_splat_client_isaac.py --gui --camera-queue-dir ... --lidar-queue-dir ...`.
Without `--gui` the clients run headless and only write frames.

**Just look at a scene in the Isaac Sim GUI** (no servers): `conda activate env_isaaclab && isaacsim`,
then File → Open one of the ready USD scenes on the drive, e.g.
`$SESSION_ZHUITONG/isaac_sim_output/camera3dgs_zhuitong_ONGROUND_WITH_CAR.usd` (3DGS map baked
to points + car) or `$NEW1_ROOT/scene.usda` (track mesh).

Known Isaac Sim 6.0 issue (observed 2026-08-29): a `ParticleField3DGaussianSplat` prim hid all other
geometry in the stage. Not reproducible in 6.1: car, trajectory markers and splat render together.
Reference splat USDs onto a *typeless* prim (the clients do this); `add_reference_to_stage` authors an
Xform that overrides the splat prim type and nothing shows.

**NuRec-quality scene via 3DGRUT (NVIDIA ray-traced Gaussians).** NVIDIA's own Isaac demo
scenes are trained with [3DGRUT](https://github.com/nv-tlabs/3dgrut) and carry PPISP (learned
exposure/colour post-processing), rendered through the `omni.rtx.spg` graph. The same path works
for our COLMAP sessions and matches Isaac's ray-traced renderer better than a splatfacto model:

```bash
# one-time: builds third_party/3dgrut/.venv (uv, CUDA 12.8, ~8 GB)
cd third_party/3dgrut && UV_PYTHON_PREFERENCE=only-managed CUDA_HOME=/usr ./install_env_uv.sh && cd ../..
source scripts/env.sh
bash scripts/train_3dgrut.sh 142014_noflare post_processing.method=ppisp   # ~30k it, exports ParticleField USDZ
bash scripts/build_3dgrut_scene.sh "$RUNS_DIR/<experiment>"                   # -> scene_3dgrut.usda + trajectory
CAMERA_3DGS_CONFIG="$SESSION_142014/outputs/colmap_dataset_full/splatfacto/2026-08-28_234736/config.yml" \
SCENE="$ISAAC_SCENE_3DGRUT" TRANSFORMS="$TRAJ_3DGRUT" bash scripts/run_isaac_demo.sh --gui --nurec --stride 5
```

Result on session 142014_noflare (2026-09-10, 30k iterations, 44 min on the RTX 4090, 1M gaussians,
held-out PSNR 25.2 dB over 187 frames vs 27.0 dB for the 80k-iteration splatfacto):
the Isaac viewport now matches the offline gsplat renders and the photos in geometry, tone and
sharpness (see `outputs/` comparison sheets). The trajectory JSON written by the wrapper keeps the
original pose as `transform_matrix_source`, which the client sends to the render server, so the
"what the algorithm sees" image still comes from the splatfacto checkpoint in its own frame.

Datasets live in `data/local/3dgrut/<name>/{images,sparse/0}` (symlinks to the session's COLMAP
model and frames on the drive; `142014_noflare` and `zhuitong` are prepared). 3DGRUT exports in
the raw COLMAP frame; `wrap_3dgrut_usd.py` applies nerfstudio's dataparser transform times the
session's odometry-fitted metres-per-unit (3.9868 for 142014_noflare) so the scene lands in the
metric Z-up frame the earlier Isaac scenes use, and converts the trajectory JSON accordingly.
`--nurec` on the client enables `omni.rtx.spg` + `isaacsim.replicator.nurec_utils` and applies
NVIDIA's pre-Hydra render overrides. Runs are written to `$RUNS_DIR` on the T7 drive.

PPISP is exported *folded into the SH coefficients* (`--ppisp-integration-mode sh-optimized`):
the runtime SPG variant only post-processes the dataset cameras it exports, not a free Isaac
camera. 3DGRUT commit a37ef72 has a key mismatch in that bake (`pred_rgb` vs `pred_features`);
apply `scripts/patches/3dgrut_sh_bake_pred_features.patch` after cloning:
`cd third_party/3dgrut && git apply ../../scripts/patches/3dgrut_sh_bake_pred_features.patch`.
The onboard camera also applies the principal-point offset (cx, cy) from `transforms.json`; without
it the Isaac view sits ~20 px off against the photos for this session.

**The newest three-window demos** (`start_zhuitong_viewers.sh`, `launch_live_viewers.sh` on the
drive) use the `1707-Test/scripts` pipeline with hard-coded training-machine paths and old env names. They are not runnable here without porting; see "Data".

Stop a server cleanly with `touch /tmp/isaac_queue/<name>/STOP`.

### Opponent vehicles as Gaussian splats

Each opponent or obstacle is a separate Gaussian object in metres, with +X forward,
+Y left and +Z up. The camera renderer inserts its posed Gaussians into the scene;
the LiDAR renderer composites the nearest solid object surface. Both use the same
object pose. Isaac Sim displays the corresponding Gaussian USD asset.

Run these commands from the repository root. Runtime assets live in
`data/local/objects` (`$OBJECTS_DIR`) and are git-ignored; obtain/import them first
on a fresh checkout. Available local assets are `f1tenth` (detailed CAD-derived car, also named `f1tenth_detailed`),
`cone_photoscan` (photographed cone), `cone_lowpoly` (CAD cone), and `cone`
(extracted from the learned New1 scene).

**Run one moving opponent in Isaac Sim:**

```bash
conda activate roboracer
source scripts/env.sh
bash scripts/run_isaac_demo.sh --gui --opponent=f1tenth
# Place it 3 m ahead along the path and 0.3 m to the left:
bash scripts/run_isaac_demo.sh --gui --opponent=f1tenth --opponent-gap 3 --opponent-lateral 0.3
# Use another imported asset by its filename stem:
bash scripts/run_isaac_demo.sh --gui --opponent=car
```

`--opponent` alone selects `f1tenth`, 2.5 m ahead. `--opponent-gap` is distance
along the recorded trajectory, not straight-line distance. Positive
`--opponent-lateral` moves left; negative moves right. `--opponent-lead 30`
instead selects a pose 30 trajectory frames ahead (before `--stride`).
Objects stand on the scene's floor plane when `floor_plane.json` exists next to
`transforms.json` (`python scripts/build_floor_plane.py --ckpt ... --dataparser-json ... --transforms ...`,
once per scene); otherwise `--ground-offset` is the camera height above the floor (New1: 0.15 m).
The New1 maps in `scripts/env.sh` are the floor-flattened fine-tunes (`scripts/flatten_floor.py`,
PSNR 27.4 -> 27.2 dB, floor band +7.5 -> +0.6 cm); opponents animate wheels, steering and
pitch/roll and can follow a GRU-driven trajectory (`scripts/simulate_opponent_path.py`,
`--opponent-trajectory`). See docs/OPPONENTS.md.
The launcher loads `<name>.npz` for camera rendering and `<name>.usd` for Isaac.
It runs camera replay with **one** opponent; that opponent follows recorded poses.
Using `--opponent=cone_photoscan` also moves the cone along the path—use the
multi-object example below for stationary cones.

**Detailed F1TENTH appearance (default local opponent):** the updated asset uses
component-specific materials, smooth normals, 2 mm surface sampling, baked contact
occlusion, black wheel rings, silver hardware, dark brackets and LiDAR housing,
black cables, and a compact stereo camera. The oversized solid rear tower is
replaced with an open support frame. It is CAD-derived, guided by
[official F1TENTH build photos](https://f1tenth-doc.readthedocs.io/en/latest/getting_started/build_car/index.html),
[RAS Pittsburgh's car](https://www.raspitt.org/projects/f1tenth), and our lab photos.
It is not a photo-trained Gaussian reconstruction.

**Miniature Porsche + changing hall:** a licensed 46 cm Porsche 911 opponent,
rotating wheels, and CC0 pallet/crate insertions are available in the
[Porsche hall demo](docs/PORSCHE_HALL_DEMO.md). It includes a five-second camera,
3D LiDAR and BEV clip, matched before/after views, asset download/build commands,
and measured rendering timings. This is a scripted replay, not reactive racing.
Its default is now 4K output with high-resolution `_hd` obstacles: 2 mm pallet/crate
sampling, 1 mm cones, and the original 4K plastic-crate texture replacing a black
export. The earlier assets remain available for lower-cost rendering.

**Staged overtake:** [run the 19-second Porsche racing demo](docs/STAGED_OVERTAKE.md)
with `python scripts/render_staged_overtake.py`. Ego follows, passes the Porsche
and continues around the track. It includes grounded cars, steering and wheel
animation, short-exposure motion blur, rear/side views, synthetic RC audio and
matching 3D/BEV LiDAR panels. The saved trajectories are checked for vehicle
footprint separation; this is a scripted demonstration, not reactive racing AI.

The local `f1tenth.npz/.usd` defaults now contain this version; `f1tenth_legacy`
preserves the original. Explicitly select it with `--opponent=f1tenth_detailed`.
The earlier [before/after views](outputs/vehicle_asset_review/before_after.png)
show the previous colourful version. Rebuild from the original AutoDRIVE USD (path recorded in
`data/local/objects/f1tenth.json`, or pass `--source /path/to/F1TENTH.usd`):

```bash
conda activate roboracer
source scripts/env.sh
PYTHONNOUSERSITE=1 python scripts/build_f1tenth_appearance.py --preview
conda activate env_isaac6
OMNI_KIT_ACCEPT_EULA=yes python scripts/splat_to_usd_gsplat.py \
  --object-npz "$OBJECTS_DIR/f1tenth_detailed.npz" \
  --out-usd "$OBJECTS_DIR/f1tenth_detailed.usd" --opacity-thresh 0
bash scripts/run_isaac_demo.sh --gui --opponent=f1tenth_detailed
```

The builder writes the named detailed asset; it does not overwrite the default
or legacy files. To render it explicitly in the five-second video, pass
`--car data/local/objects/f1tenth_detailed.npz` to `scripts/render_learned_demo.py`.
The new model has about 633k Gaussians versus 168k previously, so it uses more
rendering memory. Baked lighting follows the rigid object; wheel animation and
new shadows cast onto the track are not modeled.
The current palette uses black and dark-grey housings, mounts and cables with
metallic hardware; the earlier bright blue, red and orange accents were removed.
The refreshed [camera + 3D/BEV video](outputs/learned_demo_5s_livox_black/roboracer_camera_lidar_3d_bev_5s.mp4)
shows this darker appearance. Earlier colourful local assets are backed up as
`f1tenth_colorful` and `f1tenth_detailed_colorful`.

**Import another car or cone, export it for Isaac, then run it:**

```bash
conda activate roboracer
source scripts/env.sh
# Replace capture.ply with your Gaussian capture; axes describe the source asset.
python scripts/import_object_splat.py --ply capture.ply --up Y --forward -Z \
  --height 0.24 --name car --out "$OBJECTS_DIR/car.npz" --preview
# Alternatively import a textured GLB/OBJ with --mesh, or a USD mesh with --usd.
conda activate env_isaac6
OMNI_KIT_ACCEPT_EULA=yes python scripts/splat_to_usd_gsplat.py \
  --object-npz "$OBJECTS_DIR/car.npz" --out-usd "$OBJECTS_DIR/car.usd" --opacity-thresh 0
bash scripts/run_isaac_demo.sh --gui --opponent=car --opponent-gap 2
```

For camera **and LiDAR** replay in Isaac, start the two render servers using the
manual commands above, then run this client in `env_isaac6` from the repo root:

```bash
source scripts/env.sh
OMNI_KIT_ACCEPT_EULA=yes python simulation/isaac_sim/04_lidar_splat_client_isaac.py \
  --gui --transforms "$NEW1_ROOT/dataset/transforms.json" --scene "$ISAAC_SCENE_GSPLAT" \
  --camera-queue-dir /tmp/isaac_queue/camera --lidar-queue-dir /tmp/isaac_queue/lidar \
  --opponent-splat "$OBJECTS_DIR/f1tenth.npz" --opponent-usd "$OBJECTS_DIR/f1tenth.usd" \
  --opponent-gap 2.5 --opponent-lateral 0 --output-dir outputs/opponent_camera_lidar
```

**Reproduce the five-second demo with two cars and four stationary cones:**

```bash
conda activate roboracer
PYTHONNOUSERSITE=1 python scripts/render_learned_demo.py --preview
PYTHONNOUSERSITE=1 python scripts/render_learned_demo.py
```

Output: [camera + LiDAR MP4](outputs/learned_demo_5s_detailed/roboracer_camera_lidar_5s.mp4),
1920 × 900, 30 fps, exactly 5 seconds. This directly renders the learned sensors;
no Isaac GUI or queue servers are needed. The script requires the local New1
checkpoints, `f1tenth.npz`, and `cone_photoscan.npz`. Use `--start 560` to select
the starting COLMAP pose index and `--out outputs/my_demo` for another output folder.

For **3D + BEV LiDAR views with recorded Livox angular sampling**, run this
after creating the demo above:

```bash
PYTHONNOUSERSITE=1 python scripts/render_livox_demo.py \
  --bag '/media/alienware_felix/T7/roboracer_data/27.08新赛道和Dynamik/3dgs session_20260827_142014/all_data/all_data_0.db3'
```

This writes [camera + 3D + BEV](outputs/learned_demo_5s_livox/roboracer_camera_lidar_3d_bev_5s.mp4)
and a separate [raw Livox reference](outputs/learned_demo_5s_livox/raw_livox_reference_5s.mp4).
Both are five seconds at 30 fps; LiDAR updates follow the recorded scan timestamps
(approximately 10 Hz), with no accumulated scans. The two scan views use identical
XYZ, a sensor-centred metre grid, an 8 m radial limit and height colours.
Saved NPZ scans and `provenance.json` accompany the videos.

The original demo queries every cubemap pixel, which **does not reproduce Livox
sampling**. The new version queries the LiDAR-specific New1 `best.pt` along
consecutive recorded return directions from the August session. These are angular
templates, not New1 ground truth or the complete sensor firing pattern. The raw
reference is explicitly from a different track/time. New1's original recording is
unavailable locally, and its known streak artifacts remain; this visualization
does not establish LiDAR accuracy. Motion distortion and physical reflectance of
inserted objects are not simulated. Both versions use shared object Gaussians for
the added cars and cones, rather than separately trained object LiDAR models.

To change placements, edit the `cones` and `objects` lists in
[scripts/render_learned_demo.py](scripts/render_learned_demo.py). `ahead(t, gap,
lateral)` returns an object-to-world pose: the two cars currently use gaps of
1.7 m and 3.3 m. Cones use gaps/lateral offsets `(1.4, -0.5)`, `(2.5, 0.55)`,
`(3.8, -0.5)`, `(5.0, 0.55)`. Cone poses are computed once at `start`, keeping
them stationary; car poses are recomputed each frame. Update the video captions
and the LiDAR display-label assignment if changing the number/type of objects.
For arbitrary placement, supply a 4×4 `T_wo` (object → world, translation in metres)
in each `(ObjectSplat, T_wo)` tuple and pass the same list to both renderers.
Use the scene's coordinate frame; New1's COLMAP floor is tilted, which the
`ahead` helper handles automatically.

The replacement `cone_photoscan` is **Traffic Cone (Photoscanned)** by Nik Kottmann,
[available from BlenderKit under CC0](https://www.blenderkit.com/get-blenderkit/6a3dbebd-990f-4495-b023-cdcfe0b68431/).
Its photographed texture and wear are converted from a scanned mesh into 71,764
Gaussians; it is not a pretrained splat capture. The source GLB and metadata are
in `data/local/objects/sources/`. To rebuild from the downloaded GLB:

```bash
conda activate roboracer
source scripts/env.sh
python -m pip install DracoPy  # decoder for this compressed GLB
python scripts/import_object_splat.py --mesh "$OBJECTS_DIR/sources/cone_photoscan.glb" \
  --up Y --forward Z --height 0.33 --spacing-mm 2 --footprint 1.25 \
  --name cone_photoscan --out "$OBJECTS_DIR/cone_photoscan.npz" --preview
```

Export it with the same `env_isaac6` command above, replacing `car` with
`cone_photoscan`. The cars in this clip remain mesh-derived. The clip uses
recorded-trajectory sensor replay. To drive ego and opponent with independent
learned GRU dynamics, run `./scripts/run_racing.sh --mode autopilot --autostart`;
that separate racing application uses mesh opponents. Configure their initial
poses in `spawns` and their speed in `opponent_speed` in
`configs/racing_142014.json`; see [racing setup and controls](docs/F1TENTH_RACING.md).

One simulation step (camera + Livox LiDAR + GRU, six objects) takes ~75 ms in-process and
~100 ms through the Isaac file queue (`python scripts/benchmark_sim_step.py`; the camera server
now writes BMP instead of PNG, the LiDAR server uncompressed npz and a 1/4 object proxy).
The staged overtake takes its cars as arguments:
`python scripts/render_staged_overtake.py --opponent-asset f1tenth.npz` renders the pass against
a second F1TENTH instead of the Porsche.

For capture instructions, custom object requests, coordinate conventions, and
limitations, see [docs/OPPONENTS.md](docs/OPPONENTS.md). Clip provenance and
reproduction notes are also saved in `outputs/learned_demo_5s_detailed/README.md`.

V2 closed loop (Pure Pursuit + GRU plant, no Isaac Sim GUI required):

```bash
cd simulation/closed_loop
python 09_v2_reference_extract.py --csv "$DATA_DIR/session_20260728_163324_seg1_20ms.csv" \
                                  --output reference_path_day3.csv
python 10_v2_main_demo.py --gru ../dynamics/deployed/trained_models/gru_run30_best.pth \
                          --ref-csv "$DATA_DIR/session_20260728_163324_seg1_20ms.csv" \
                          --duration 60 --target-speed 1.5
```

---

## Evaluation

`scripts/run_evaluation.sh` builds a dedicated conda environment (`roboracer-eval`, same stack as
`roboracer` plus `rosbags` for reading the bags), runs every evaluation unattended and writes
`outputs/evaluation_<timestamp>/report.md` with `summary.json`, per-frame/per-scan CSVs and figures.

```bash
bash scripts/run_evaluation.sh                    # fresh env + full run (~1 h incl. env build)
bash scripts/run_evaluation.sh --quick            # subsampled smoke run
ENV_NAME=roboracer bash scripts/run_evaluation.sh --reuse-env --only dynamics
```

| Part | What is measured | Ground truth |
|---|---|---|
| Camera 3DGS (`evaluation/eval_camera.py`) | PSNR / SSIM / LPIPS on the frames nerfstudio held out (deterministic 90/10 split), PSNR along the lap, worst/median/best panels with error maps, novel-view sharpness at poses the car never visited | recorded frames (2026-07-14 frames are undistorted with the factory calibration first) |
| LiDAR 3DGS (`evaluation/eval_lidar.py`) | for every 10th scan of the 2026-08-27 sessions: hit rate, per-ray range error, Chamfer distance, F-score@10/20 cm, error vs range, overlay | real Livox scans from the bag at the FAST-LIO odometry pose |
| Dynamics (`evaluation/eval_dynamics.py`) | GRU family, Deep Dynamics v8, MS-NN variants: one-step RMSE vs persistence, autoregressive rollout error vs horizon (0.1–2 s), full closed-loop rollout (60/125 s tiers), pose drift after integration, error by speed bin | MoCap (GRU/DD) and IMU/VESC (MS-NN) on same-day, Day-2, cross-day (Day 1) and new-track (2026-08-27) segments |

Latest report: `docs/evaluation/2026-09-10/report.md` (fresh `roboracer-eval` env, 4 min runtime after the env build).

Predictions are aligned to the next sample everywhere (a prediction made at *t* is scored against
*t + 20 ms*). All models are loaded through the same code paths the simulators use, so the numbers
describe what Isaac Sim actually gets. Two bugs were found while building the suite and fixed in the
production code: the LiDAR ray-cast returned z-depth instead of range along the ray (up to 42 % short
at cubemap face corners), and the 2026-07-14 "images" are raw frames that must be undistorted before
any comparison.

## Vehicle and sensors

| Item | Value |
|---|---|
| Car | F1TENTH Car02, mass 3.788 kg, wheelbase 0.3225 m (lf 0.1584 / lr 0.1641 m) |
| Footprint | 0.42 m × 0.26 m, tyre width 0.04 m |
| Steering | ±0.29 rad (±16.6°) |
| Speed | 0–3.5 m/s; `ERPM_TO_MS = 1/7614.0` (calibrated VESC value) |
| Camera | OAK-D Pro W, RGB CAM_A 1280×800, HFOV 97°, fx 566.26 fy 565.95 cx 641.96 cy 388.61, rational-polynomial distortion |
| LiDAR | Livox Mid-360 (also provides IMU at 200 Hz, downsampled to 50 Hz) |
| MoCap | Marker offsets relative to `base_link`, see `reference/vehicle/vehicle_parameters.txt` |

Full parameter sheet with measurement photos: `reference/vehicle/`.
Frame conventions and TF usage: `reference/calibration/README_sensor_tf_usage.md`.

---

## Data-split rule

Never violate this when training or evaluating dynamics models:

| Role | Files |
|---|---|
| **TEST ONLY** | `session_20260721_20ms.csv` (Day 1). Final evaluation only. Never a reference path, never in train/val. |
| Train / val | `session_20260722_*.csv` (Day 2), `session_20260728_*.csv` (Day 3) |
| Excluded | `session_20260728_124427_*.csv` (100 % NaN IMU) |

Mirror augmentation is for the GRU only. It breaks the MS-NN local-model fuzzy centres.

---

## Troubleshooting

**`python scripts/check_env.py` fails on `gsplat kernel`.**
The prebuilt wheel did not match torch. Check `python -c "import torch; print(torch.__version__)"`
prints `2.6.0+cu124`, then `pip install --force-reinstall --no-deps --index-url https://docs.gsplat.studio/whl/pt24cu124 gsplat==1.5.3+pt24cu124`.

**`nnodely` import error about torch version.**
nnodely was installed with its pins. Fix: `pip install --no-deps --force-reinstall nnodely==1.5.4`.

**`NameError: 'yaw_rate' already used as NeuObj`.**
nnodely keeps a global name registry. Build one MS-NN model per Python process, or clear
`NeuObj.names` at the top of the script (see `dynamics/msnn/TECHNICAL_NOTES.md`).

**COLMAP registers < 500 images.**
Check the vocab-tree path in `03_colmap_commands.sh`, raise `overlap` to 30. Make sure
the undistorted images show straight lines.

**Isaac Sim client hangs waiting for `response_N.png`.**
Render server crashed or was started in the wrong env. Check its terminal and
`ls /tmp/isaac_queue/<name>/`. Remove a stale `STOP` file.

**Two CUDA processes fight for the GPU.**
Isaac Sim and the render servers share one GPU by design. If VRAM runs out, lower the
render resolution or run the servers with `CUDA_VISIBLE_DEVICES` on a second GPU.

**`conda install colmap` fails.**
Use the system package instead: `sudo apt install colmap` (3.7, CPU build; all scripts use
`--use_gpu 0`).

---

## Legacy environment names

The pipeline READMEs were written when the project used four separate
conda envs. They all map to the two envs described here:

| Name in old READMEs | Use now |
|---|---|
| `nerfstudio-py310`, `f1tenth_dynamics`, `probe` | `roboracer` |
| `nnodely-env` | `roboracer` |
| `env_isaac6` | `env_isaaclab` |
| `autodrive-bridge` (xdotool control of the AutoDRIVE simulator) | not needed for the Isaac Sim pipeline |

Where a README says `conda run -n nerfstudio-py310 python3 X.py`, run
`conda activate roboracer && python X.py`. The original per-env YAML files are kept in
`envs/legacy/` for reference.

---

## Hybrid F1Tenth racing (Isaac Sim 6)

The new `racing/` application combines native Gaussian camera rendering, learned
Gaussian LiDAR with mesh opponents, and independent learned GRU vehicle dynamics.
Run `./scripts/run_racing.sh`; see [setup, controls, ROS topics and validation](docs/F1TENTH_RACING.md).
The included 142014 calibration is provisional (0.41 m mean alignment residual),
its collision boundaries are corridor proxies, and native mesh/Gaussian compositing
still produces washed-out opponents in this recording. The racing app uses the separate
`env_isaac6` environment; existing `env_isaaclab` workflows remain separate.
