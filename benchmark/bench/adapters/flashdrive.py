"""Alpamayo 1.5 + FlashDrive (z-lab checkpoint) or the original model through the FlashDrive loader.

Streaming model: each window reuses the KV cache of the previous one, so windows must be
consecutive (10 Hz). The first window of every stream only prefills and returns no plan
(`Prediction.xy is None`); the runner skips it in the metrics.

Stage mapping onto the benchmark columns (FlashDrive names -> benchmark names):
    vision    = `_encode`                       (FlashDrive "encode")
    prefill   = `_dflash_prefill`               (DFlash prefill of the window)
    decode    = `_dflash_decode_loop`           (speculative draft/verify loop)
    diffusion = `_sample_actions_and_format`    (FlashDrive "action": expert + trajectory formatting)

TODO(GPU): FlashDrive is written for 4 views x 4 frames. `num_views` here re-initialises the
streaming state for fewer views; verify on the 3090 that compile/CUDA-graph shapes accept it.
"""

import time

import torch

from .alpamayo import CAMERA_INDEX_FRONT
from .base import ModelAdapter, Prediction
from ..timing import StageTimer

MAX_NEW_TOKENS = 128
DIFFUSION_STEPS = 8
CACHE_STEPS = [3, 4, 5, 6]


class FlashDriveAdapter(ModelAdapter):
    @property
    def required_period_s(self) -> float | None:
        return 0.1 if getattr(self, "optimized", False) else None

    def load(self) -> None:
        import flashdrive
        import importlib

        path = self.cfg["checkpoint"]
        package = flashdrive.resolve_model_class(path).__module__.split(".")[0]
        self.helper = importlib.import_module(f"{package}.helper")
        self.device = self.cfg.get("device", "cuda")
        self.model = flashdrive.from_pretrained(path, device=self.device)
        self.optimized = isinstance(self.model, flashdrive.FlashDriveMixin)
        self.processor = self.helper.get_processor(self.model.tokenizer)

        self.num_views = self.cfg.get("num_views", 1)
        self.num_frames = self.cfg.get("num_frames", 4)
        if self.optimized:
            from flashdrive import _EXTENDED_VISION_RANGES

            self.model.setup_streaming(extended_vision_ranges=_EXTENDED_VISION_RANGES[flashdrive.resolve_model_class(path)],
                                       num_views=self.num_views, num_frames_per_view=self.num_frames)
            self.vs_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
            self.ve_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")

        self.timer = StageTimer()
        if self.optimized:
            for name, stage in (("_encode", "vision"), ("_dflash_prefill", "prefill"),
                                ("_dflash_decode_loop", "decode"), ("_sample_actions_and_format", "diffusion")):
                self.timer.wrap_method(self.model, name, stage)
        else:
            self.timer.hook_module(self.model.vlm.model.visual, "vision")
            self.timer.hook_prefill_decode(self.model.vlm.model.language_model)
            self.timer.wrap_method(self.model.diffusion, "sample", "diffusion")
        self._streaming_started = False

    def reset_stream(self) -> None:
        if self.optimized:
            self.model._past_key_values = None
            # The first window of a stream has a different input shape than the following streaming windows.
            # Vision tower and language model keep shape-specific caches from the previous stream's last window;
            # FlashDrive clears them only at the end of the first prefill, so clear them before a new stream
            # (needed for several seeds / episodes in one process).
            self.model.vlm.model.visual.reset_shape_caches()
            self.model.vlm.model.language_model.reset_shape_caches()
        self._streaming_started = False

    def prepare(self, sample, num_samples: int):
        frames = torch.from_numpy(sample.load_frames()).permute(0, 3, 1, 2).contiguous()
        kwargs = {"camera_indices": torch.tensor([CAMERA_INDEX_FRONT])}
        if self.task_prompt and self.cfg.get("task_as_nav_text", True):
            kwargs["nav_text"] = self.task_prompt
        messages = self.helper.create_message(frames, num_frames_per_camera=frames.shape[0], **kwargs)
        tok = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, continue_final_message=True,
            return_dict=True, return_tensors="pt")
        data = {"tokenized_data": tok,
                "ego_history_xyz": torch.from_numpy(sample.hist_xyz)[None, None],
                "ego_history_rot": torch.from_numpy(sample.hist_rot)[None, None]}
        if self.optimized and self._streaming_started:
            from flashdrive.streaming import convert_to_streaming_window

            data = convert_to_streaming_window(data, self.vs_id, self.ve_id, num_views=self.num_views,
                                               num_frames_per_view=self.num_frames)
        return self.helper.to_device(data, self.device)

    @torch.inference_mode()
    def predict(self, inputs, num_samples: int) -> Prediction:
        self.timer.reset()
        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if self.optimized:
                out = self.model.sample_trajectories_streaming(
                    data=inputs, max_new_tokens=MAX_NEW_TOKENS, num_traj_samples=num_samples, return_extra=True,
                    diffusion_kwargs={"inference_step": DIFFUSION_STEPS, "cache_steps": CACHE_STEPS,
                                      "int_method": "euler_with_cache"})
            else:
                out = self.model.sample_trajectories_from_data_with_vlm_rollout(
                    data=inputs, num_traj_samples=num_samples, max_generation_length=256, return_extra=True)
        torch.cuda.synchronize()
        total = (time.perf_counter() - t) * 1000.0
        self._streaming_started = True  # next window is converted to the streaming form
        pred_xyz, _, extra = out
        if pred_xyz is None:  # streaming primer window
            return Prediction(None, self.timer.snapshot(total))
        return Prediction(pred_xyz[0, 0, :, :, :2].float().cpu().numpy(), self.timer.snapshot(total),
                          str(extra["cot"][0][0][0]))
