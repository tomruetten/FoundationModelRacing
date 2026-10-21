"""Qwen-Drive-1.0 (Qwen3.5-4B VLM + flow-matching Planning Expert), direct or reasoning planning.

Differences to the Alpamayo adapters that matter for the comparison:
    * frames: 4 per view at 0.5 s spacing (t-1.5 .. t0), not 4 @ 10 Hz -> `frame_spacing_s = 0.5`
    * views: trained on FRONT / FRONT LEFT / FRONT RIGHT, the car has one camera. `views: repeat_front` feeds the
      front image to all three views (3x the image tokens), `views: black` fills the side views with black images
    * no free-text prompt: the instruction is generated from history + navigation command (always GO STRAIGHT)
    * plan: 50 waypoints (5 s) of (x, y, heading); pose history needs heading, velocity and acceleration (`ego_inputs`)
    * reasoning (mode `reasoning_planning`) is greedy, one sentence; `direct_planning` skips the decode stage

Stage mapping (forward hooks on the actual submodules, CUDA-synchronised; module paths: TODO(GPU) verify):
    vision    = VLM vision tower
    prefill   = first LLM forward (direct: the only one; reasoning: prompt prefill inside `generate`)
    decode    = every further LLM forward (reasoning tokens + the forward that closes the assistant turn)
    diffusion = `planning_expert.sample` (10 Euler steps over 32 layers)
    other     = rest (rope positions, cache hand-over, denormalisation, copy to host)
"""

import sys
import time
from pathlib import Path

import numpy as np

from .base import ModelAdapter, Prediction
from ..timing import StageTimer

DT = 0.1
VIEW_NAMES = ("<FRONT VIEW>", "<FRONT LEFT VIEW>", "<FRONT RIGHT VIEW>")
DRIVING_COMMAND_STRAIGHT = [0, 1, 0, 0]  # left / straight / right / unknown (docs/data.md)
REPO_DEFAULT = Path(__file__).resolve().parents[3] / "Qwen-Drive-1.0"


def ego_inputs(hist_xyz: np.ndarray, hist_rot: np.ndarray):
    """Qwen-Drive history inputs from the benchmark's pose history (ego frame at t0, 10 Hz, oldest first).

    Returns pose (H, 3) = (x, y, heading), velocity (H, 2) and acceleration (H, 2). As in the training data, velocity
    and acceleration are expressed in the body frame of each timestamp. They are finite differences of the poses and
    use only poses up to t0 (second-order backward difference at the last row).
    """
    xy = np.asarray(hist_xyz, dtype=np.float64)[:, :2]
    yaw = np.arctan2(hist_rot[:, 1, 0], hist_rot[:, 0, 0])
    vel = np.gradient(xy, DT, axis=0, edge_order=2)
    acc = np.gradient(vel, DT, axis=0, edge_order=2)
    c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]

    def body(v):
        return np.concatenate([c * v[:, :1] + s * v[:, 1:], -s * v[:, :1] + c * v[:, 1:]], axis=1)

    pose = np.concatenate([xy, yaw[:, None]], axis=1)
    return pose.astype(np.float32), body(vel).astype(np.float32), body(acc).astype(np.float32)


def _resolve_attr(root, *paths):
    for path in paths:
        node = root
        try:
            for part in path.split("."):
                node = getattr(node, part)
            return node
        except AttributeError:
            continue
    raise AttributeError(f"none of {paths} found on {type(root).__name__}; adapt QwenDriveAdapter.load")


class QwenDriveAdapter(ModelAdapter):
    frame_spacing_s = 0.5
    uses_task_prompt = False

    def load(self) -> None:
        import torch

        try:
            import qwen_drive  # noqa: F401
        except ImportError:  # not pip-installed: use the checkout next to the benchmark
            repo = Path(self.cfg.get("repo") or REPO_DEFAULT).expanduser()
            if not (repo / "src" / "qwen_drive").is_dir():
                raise SystemExit(f"qwen_drive not importable and no checkout at {repo}; set model_cfg.repo")
            sys.path.insert(0, str(repo / "src"))
        from qwen_drive import InferenceMode, QwenDriveForPlanning

        self.torch = torch
        self.device = self.cfg.get("device", "cuda")
        self.mode = InferenceMode(self.cfg.get("mode", "direct_planning"))
        if self.mode is InferenceMode.VQA:
            raise ValueError("mode must be direct_planning or reasoning_planning")
        self.views = self.cfg.get("views", "repeat_front")
        if self.views not in ("repeat_front", "black"):
            raise ValueError(f"views must be repeat_front or black, got {self.views!r}")

        checkpoint = self.cfg["checkpoint"]
        local = Path(checkpoint).expanduser()
        if not local.is_dir():  # Hub id: the planner heads are sub-folders, so fetch the whole snapshot
            from huggingface_hub import snapshot_download

            local = Path(snapshot_download(checkpoint))
        planner = Path(self.cfg.get("planner", "planner-sft")).expanduser()
        planner = planner if planner.is_dir() else local / planner
        self.model = QwenDriveForPlanning.from_pretrained(
            str(local), planner=str(planner), dtype=torch.bfloat16,
            attn_implementation=self.cfg.get("attn_implementation", "sdpa")).to(self.device).eval()

        vlm = self.model.vlm
        self.timer = StageTimer()
        self.timer.hook_module(_resolve_attr(vlm, "model.visual", "visual"), "vision")
        self.timer.hook_prefill_decode(_resolve_attr(vlm, "model.language_model", "language_model"))
        self.timer.wrap_method(self.model.planning_expert, "sample", "diffusion")

    def prepare(self, sample, num_samples: int):
        """Scene -> model tensors on the GPU (image resize/patchify, tokenisation, host-to-device copy)."""
        from PIL import Image

        from qwen_drive import CameraFrame, DrivingScene

        front = [CameraFrame(Image.fromarray(f)) for f in sample.load_frames()]
        if self.views == "repeat_front":
            views = {name: front for name in VIEW_NAMES}
        else:
            black = [CameraFrame(Image.new("RGB", f.image.size)) for f in front]
            views = {VIEW_NAMES[0]: front, VIEW_NAMES[1]: black, VIEW_NAMES[2]: black}
        pose, vel, acc = ego_inputs(sample.hist_xyz, sample.hist_rot)
        scene = DrivingScene(
            views=views, history=pose, history_velocity=vel, history_acceleration=acc,
            ego_velocity=vel[-1], ego_acceleration=acc[-1], driving_command=DRIVING_COMMAND_STRAIGHT,
            nav_command=0, token=f"ep{sample.episode}_{sample.step}")
        inputs = self.model.processor(scene, with_reasoning=self.mode.value == "reasoning_planning", device="cpu")
        return {k: v.to(self.device) if self.torch.is_tensor(v) else v for k, v in inputs.items()}

    @property
    def _noise_seed(self) -> int:
        # the runner seeds torch per sample from (seed, sample index); sample k of the expert then uses seed + k
        return int(self.torch.initial_seed() % (2**31 - 1))

    def predict(self, inputs, num_samples: int) -> Prediction:
        torch = self.torch
        self.timer.reset()
        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.inference_mode():
            out = self.model.plan_from_inputs(inputs, mode=self.mode, num_samples=num_samples,
                                              seed=self._noise_seed, max_new_tokens=self.cfg.get("max_new_tokens"))
        torch.cuda.synchronize()
        total = (time.perf_counter() - t) * 1000.0
        xy = np.asarray(out.trajectories, dtype=np.float32)[:, :, :2]  # (K, 50, 2)
        return Prediction(xy, self.timer.snapshot(total), out.reasoning or "")
