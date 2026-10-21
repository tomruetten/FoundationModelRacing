"""Model adapter interface. One adapter per model family; the checkpoint comes from the model YAML."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Prediction:
    xy: np.ndarray | None  # (K, T, 2) ego frame at t0, metres; None if the model produced no plan
    stage_ms: dict[str, float]  # vision, prefill, decode, diffusion, other, total (NaN = stage n/a)
    text: str = ""  # reasoning text, if the model produces any
    extra: dict = field(default_factory=dict)


class ModelAdapter(ABC):
    def __init__(self, cfg: dict, task_prompt: str | None):
        self.cfg = cfg
        self.task_prompt = task_prompt

    @abstractmethod
    def load(self) -> None:
        """Load weights (not part of any measurement)."""

    @abstractmethod
    def prepare(self, sample, num_samples: int):
        """CPU-side preprocessing (image decode, tokenisation). Not part of the latency."""

    @abstractmethod
    def predict(self, inputs, num_samples: int) -> Prediction:
        """One timed forward: model inputs -> K trajectories + stage times."""

    # False if one process can only run a single stream (e.g. FlashDrive): several seeds then run in separate processes.
    supports_stream_restart = True

    def reset_stream(self) -> None:
        """Called at every episode start (stateful models, e.g. streaming KV caches)."""

    # Spacing of the input frames in seconds; None = 0.1 s (10 Hz) like the pose history. Qwen-Drive needs 0.5 s.
    frame_spacing_s: float | None = None

    # False if the model has no free-text prompt: the runner then drops `task_prompt` and tags the run `no_prompt`.
    uses_task_prompt = True

    @property
    def required_period_s(self) -> float | None:
        """Sample spacing the model needs (streaming models: 0.1 s consecutive windows), else None."""
        return None
