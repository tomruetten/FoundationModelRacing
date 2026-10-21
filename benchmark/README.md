# One-shot benchmark (F1TENTH)

Offline model comparison on a recorded sim run: per sample the model gets the front-camera frames
(4 @ 10 Hz), the ego history (16 poses @ 10 Hz) and a task prompt, and plans 64 waypoints (6.4 s; Qwen-Drive 50 = 5 s).
Output: latency per stage (vision / prefill / decode / diffusion / other) and ADE / minADE vs. the driven future.

## 1. Record data (Isaac Sim, GPU machine)
```bash
cd ../RoboRacer-3D-Isaac-Sim
./scripts/run_racing.sh --headless --mode autopilot --autostart --steps 2000 \
  --save-frames --output outputs/racing_bench
```
Writes `trajectory.jsonl` and `frames/ep<E>_<step>.png`. Point `data.recording` in `config.yaml` at this folder.

## 2. Run
```bash
pip install pyyaml pillow numpy       # plus the alpamayo1_5 / flashdrive environments for the GPU adapters
python run_benchmark.py                                   # model from config.yaml
python run_benchmark.py --model configs/models/alpamayo15_flashdrive.yaml
python run_benchmark.py --model configs/models/constant_velocity.yaml   # no GPU, sanity baseline
python run_benchmark.py --set num_samples=1 --set data.max_samples=20
```
Results land in `results/<model>_<timestamp>/` (`per_sample.csv`, `summary.md/json`, `config.snapshot.yaml`).

## Qwen-Drive-1.0
Own environment (needs `transformers==5.14.x`, `torch>=2.8`, which the Alpamayo environments do not have):
```bash
cd ../Qwen-Drive-1.0 && conda create -n qwen-drive python=3.12 && conda activate qwen-drive
pip install -r requirements.txt --no-build-isolation      # flash-attn, causal-conv1d, flash-linear-attention
pip install pyyaml huggingface_hub                        # benchmark side
hf auth login                                             # if the weights are gated
cd ../benchmark
python run_benchmark.py --model configs/models/qwen_drive_direct.yaml
python run_benchmark.py --model configs/models/qwen_drive_reasoning.yaml
python run_benchmark.py --model configs/models/qwen_drive_reasoning_rl.yaml   # optional: RL expert, reasoning only
```
The weights (`Qwen/Qwen-Drive-1.0-4B`, 9 GB VLM + `planner-sft/`, `planner-rl/`) are fetched from the Hub into the normal
cache on first use (or point `checkpoint:` at a local folder); the code is imported from `../Qwen-Drive-1.0/src`
if the package is not pip-installed. `causal-conv1d` and `flash-linear-attention` speed up the Qwen3.5 linear-attention
layers; without them transformers falls back to slow PyTorch kernels and the latencies are not representative.

What differs from the Alpamayo runs (keep this in mind when comparing and say it in the thesis):
- **Frames**: 4 per view *0.5 s apart* (t-1.5 .. t0) as in training, not 4 @ 10 Hz (`frame_spacing_s`, set by the adapter).
  Poses stay at 10 Hz. Samples therefore start 1.5 s into an episode (the pose history needs that anyway).
- **Views**: the model reads FRONT / FRONT LEFT / FRONT RIGHT. The car has one camera, so `views: repeat_front` feeds the front
  image into all three (3x image tokens; `views: black` is the alternative). Latency is higher than for a true single view.
- **Prompt**: no free-text input. `task_prompt` is ignored, the run is tagged `no_prompt` (navigation command: always GO STRAIGHT).
- **Plan**: 50 waypoints = 5 s. `ade`/`fde` cover 5 s (column `pred_horizon_s`); `ade@6.4s` is NaN. Compare models at `@2s/@4s/@5s`.
- **Image size**: the model's own resize (history frames <= 174k px, current <= 922k px, 32 px grid), not the sim resolution.
- **Randomness**: reasoning is greedy; only the expert noise is seeded (`torch.initial_seed()` of the sample, +k per sample).
- **Timing**: direct = no decode stage; reasoning decode includes the forward that closes the assistant turn. `prepare` (resize,
  tokenisation, copy to GPU) is outside the latency like for Alpamayo; no CUDA graph / `torch.compile` for Qwen-Drive, Alpamayo
  runs with the expert CUDA graph, so the diffusion stage is not like for like.
- `planner-sft` serves both modes with the same weights (clean mode comparison); `planner-rl` was optimised on reasoning rollouts only.
- Hook module paths (`vlm.model.visual`, `vlm.model.language_model`) are untested on GPU: if a lookup fails the adapter raises with a hint.

## Seeds and repeated runs
Reasoning tokens and diffusion noise are random, so two runs of the same configuration differ (about 0.4 m ADE in our
first runs). `seed` in `config.yaml` (default 42) seeds every sample from (seed, sample index); the order of the
samples is always chronological.
```bash
python run_benchmark.py --seed 7              # single run, fixed seed
python run_benchmark.py --seeds 1 2 3         # one pass per seed (model loaded once; FlashDrive: one process per seed)
```
FlashDrive cannot start a second stream in the same process (shape caches and compiled graphs are bound to the first
one), so `--seeds` starts one child process per seed for it (and for any model with `separate_processes: true`); the
model is loaded and compiled once per seed. A failed seed (e.g. CUDA out of memory while loading on a shared GPU)
does not stop the others; complete the run with the same command plus `--resume results/<run folder>` (finished seeds
are skipped, the aggregate covers all seeds). The same limit applies to recordings with several episodes.
With `--seeds` the results go to `results/<model>_<tag>_3seeds_<time>/seed_<s>/` (full report per seed) plus
`aggregate.md/json`: per-seed means, mean and std over seeds for every metric. Use the std over seeds as the noise
level when comparing models or prompts. Bit-exact reproducibility is not guaranteed on GPU (CUDA graphs,
quantisation, speculative decoding): check it once by running the same seed twice and comparing `predictions.npz`.

## Visualise (no GPU, runs on the Mac)
```bash
python visualize.py --run results/<run> --idx 40 100            # selected samples
python visualize.py --run results/<run> --every 5 --video       # PNG per 5th sample + lap.mp4
python visualize.py --run results/<a> --run results/<b> --idx 40  # compare plans of several runs
python visualize.py --run results/<run> --idx 40 --calibrate -0.05 -0.15 -0.25   # choose --ground-z
```
Left: top-down in the ego frame (history, GT, plan, other K samples, best of K). Right: last input frame with
GT and plan projected onto the ground plane (pinhole from `racing_142014.json`, scaled to the frame size, no
distortion). The ground height `--ground-z` (default -0.15 m in the vehicle frame) is estimated from the images;
check it with `--calibrate`. Output: `results/<run>/viz/`. Needs matplotlib, imageio, imageio-ffmpeg.

## Add a model
New class in `bench/adapters/` implementing `ModelAdapter` (`load`, `prepare`, `predict`), register it in
`bench/adapters/__init__.py`, add `configs/models/<name>.yaml`.

## Definitions
- ADE: mean L2 error of the xy waypoints, sample 0. minADE: best of K samples. Also `@2s/@4s/@6.4s`, FDE. Metres, unscaled.
- Stage times are CUDA-synchronised hooks; `other` = total - sum(stages). `NaN` = model has no such stage.
- Alpamayo: vision = vision tower, prefill = first LLM forward, decode = following LLM forwards, diffusion = `diffusion.sample`.
- FlashDrive: encode / DFlash prefill / DFlash decode loop / action expert mapped to the same columns. Needs 0.1 s
  consecutive windows (set automatically); the first window of a stream only primes the cache and is not scored.

## Caveats
- Zero-shot on a 1:10 car with one camera: poor minADE is expected.
- `num_views: 1` for FlashDrive and the nav-text prompt route are untested on GPU (marked TODO in the adapters).
- GT is the autopilot's driven path, not a human expert.
