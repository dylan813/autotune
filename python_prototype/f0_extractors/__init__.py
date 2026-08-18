"""F0 (fundamental frequency) estimators with a shared interface.

All detectors expose:

    detect(audio: np.ndarray, sample_rate: int, hop_size: int)
        -> F0Result(times, f0_hz, voicing)

so the rest of the pipeline can stay detector-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class F0Result:
    """Framewise pitch estimation result."""
    times: np.ndarray       # shape (n_frames,), seconds, frame centers
    f0_hz: np.ndarray       # shape (n_frames,), 0 or NaN for unvoiced
    voicing: np.ndarray     # shape (n_frames,), in [0, 1] confidence/probability


class F0Detector(Protocol):
    name: str

    def detect(
        self,
        audio: np.ndarray,
        sample_rate: int,
        hop_size: int,
    ) -> F0Result: ...


def get_detector(name: str) -> F0Detector:
    """Factory used by autotune_offline.py.

    TODO(milestone-2): wire up the three implementations and surface options
    (model size, GPU/CPU, etc.) through a config object.
    """
    name = name.lower()
    if name == "pyin":
        from .pyin_detector import PyinDetector
        return PyinDetector()
    if name == "torchcrepe":
        from .torchcrepe_detector import TorchcrepeDetector
        return TorchcrepeDetector()
    if name == "rmvpe":
        from .rmvpe_detector import RmvpeDetector
        return RmvpeDetector()
    raise ValueError(f"Unknown detector: {name!r}")
