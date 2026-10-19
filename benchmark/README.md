# One-shot benchmark (F1TENTH)

Offline model comparison on a recorded sim run: per sample the model gets the front-camera frames
(4 @ 10 Hz), the ego history (16 poses @ 10 Hz) and a task prompt, and plans 64 waypoints (6.4 s).
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
