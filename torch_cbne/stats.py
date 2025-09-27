from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Statistics:
    """Runtime statistics mirroring the original C++ structure."""

    walk_length: int = 0
    sample_count: int = 0
    number_of_cycles: int = 0

    def set_sample_count(self, value: int) -> None:
        self.sample_count = int(value)

    def set_walk_length(self, value: int) -> None:
        self.walk_length = int(value)

    def incr_number_of_cycles(self) -> None:
        self.number_of_cycles += 1

    def summary(self) -> str:
        return (
            "Statistics:\n"
            f"Max length of walk {self.walk_length}\n"
            f"Number of samples: {self.sample_count}\n"
            f"Number of cycles: {self.number_of_cycles}\n"
            f"Number of 0s: {self.sample_count - self.number_of_cycles}\n"
        )
