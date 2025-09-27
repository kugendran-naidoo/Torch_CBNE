from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class RuntimeConfig:
    """Holds runtime configuration options for the Torch CBNE implementation."""

    epsilon: float = 0.1
    iter_limit: int = -1
    deg_limit: int = -1
    output_shot_count: bool = False
    output_step_count: bool = False
    use_one_norm: bool = True
    num_data_points: int = 1
    out_path: Optional[str] = None
    cbne_version: str = "cbne"
    device: str = "cuda"
    seed: Optional[int] = None
    time_limit: int = -1

    def torch_device(self) -> torch.device:
        if self.device == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(self.device)

    def should_output_counts(self) -> bool:
        return self.output_shot_count or self.output_step_count

    def validate(self) -> None:
        if self.cbne_version not in {"cbne", "cbneCheby", "cbneMusco", "cbneCompressed"}:
            raise ValueError(
                "cbne_version must be one of {cbne, cbneCheby, cbneMusco, cbneCompressed}"
            )
        if self.num_data_points < 1:
            raise ValueError("num_data_points must be >= 1")
        if self.iter_limit < -1:
            raise ValueError("iter_limit must be >= -1")
        if self.deg_limit < -1:
            raise ValueError("deg_limit must be >= -1")
        if self.time_limit < -1:
            raise ValueError("time_limit must be >= -1")

