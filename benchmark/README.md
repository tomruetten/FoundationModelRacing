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
