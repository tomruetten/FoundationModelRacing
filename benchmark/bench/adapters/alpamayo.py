"""Alpamayo 1.5 (nvidia checkpoint, original HF-style rollout).

Stage mapping (forward hooks on the actual submodules, CUDA-synchronised):
    vision    = Qwen3-VL vision tower          `model.vlm.model.visual`
    prefill   = first LLM forward of a rollout `model.vlm.model.language_model`
    decode    = every further LLM forward (one per generated token)
    diffusion = `model.diffusion.sample` (flow-matching loop incl. all expert forwards)
    other     = rest (tokenizer fusion, KV-cache handling, action->trajectory, ...)
"""

import time

import numpy as np
import torch

from .base import ModelAdapter, Prediction
from ..hf_utils import resolve_repo_relative_vlm
from ..timing import StageTimer

CAMERA_INDEX_FRONT = 1  # helper.CAMERA_DISPLAY_NAMES[1] == "Front camera"


class AlpamayoAdapter(ModelAdapter):
    def load(self) -> None:
        from alpamayo1_5 import helper
        from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

        self.helper = helper
        self.device = self.cfg.get("device", "cuda")
        # The checkpoint defaults to flash_attention_2, which needs the flash_attn package (not installed
        # in the FlashDrive env). Like flashdrive.from_pretrained, override the config explicitly.
        attn = self.cfg.get("attn_implementation", "sdpa")
        checkpoint = self.cfg["checkpoint"]
        # Fix a repository-relative vlm_name_or_path (e.g. SIMFORGE: "vlm") in the raw config dict first: building
        # the config object already loads a processor from that path.
        config_dict, _ = Alpamayo1_5.config_class.get_config_dict(checkpoint)
        checkpoint = resolve_repo_relative_vlm(config_dict, checkpoint)
        config = Alpamayo1_5.config_class.from_dict(config_dict)
        config.attn_implementation = config._attn_implementation = attn
        if attn == "sdpa":
            Alpamayo1_5._supports_sdpa = True  # wrapper class does not declare it, submodels support it
        self.model = Alpamayo1_5.from_pretrained(checkpoint, config=config, dtype=torch.bfloat16)
        self.model = self.model.to(self.device).eval()
        self.processor = helper.get_processor(self.model.tokenizer)
        if self.cfg.get("cuda_graph", False):
            if hasattr(self.model, "enable_diffusion_expert_cuda_graph"):
                self.model.enable_diffusion_expert_cuda_graph(max_batch_size=self.cfg.get("num_samples", 6))
            else:
                print("WARNING: installed alpamayo1_5 has no expert CUDA graph; running eager", flush=True)

        self.timer = StageTimer()
        self.timer.hook_module(self.model.vlm.model.visual, "vision")
        self.timer.hook_prefill_decode(self.model.vlm.model.language_model)
        self.timer.wrap_method(self.model.diffusion, "sample", "diffusion")

    def prepare(self, sample, num_samples: int):
        frames = torch.from_numpy(sample.load_frames()).permute(0, 3, 1, 2).contiguous()  # (T,3,H,W) uint8
        n = frames.shape[0]
        kwargs = {}
        if self.task_prompt and self.cfg.get("task_as_nav_text", True):
            kwargs["nav_text"] = self.task_prompt
        elif self.cfg.get("use_nav_prompt", False):
            kwargs["use_nav_prompt"] = True
        messages = self.helper.create_message(
            frames, camera_indices=torch.tensor([CAMERA_INDEX_FRONT]), num_frames_per_camera=n, **kwargs)
        tok = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, continue_final_message=True,
            return_dict=True, return_tensors="pt")
        data = {
            "tokenized_data": tok,
            "ego_history_xyz": torch.from_numpy(sample.hist_xyz)[None, None],
            "ego_history_rot": torch.from_numpy(sample.hist_rot)[None, None],
        }
        return self.helper.to_device(data, self.device)

    @torch.inference_mode()
    def predict(self, inputs, num_samples: int) -> Prediction:
        self.timer.reset()
        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, _, extra = self.model.sample_trajectories_from_data_with_vlm_rollout(
                data=inputs, top_p=self.cfg.get("top_p", 0.98), temperature=self.cfg.get("temperature", 0.6),
                num_traj_samples=num_samples, max_generation_length=self.cfg.get("max_new_tokens", 256),
                return_extra=True)
        torch.cuda.synchronize()
        total = (time.perf_counter() - t) * 1000.0
        xy = pred_xyz[0, 0, :, :, :2].float().cpu().numpy()  # (K, 64, 2)
        text = str(extra["cot"][0][0][0])
        return Prediction(xy, self.timer.snapshot(total), text)
