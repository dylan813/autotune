"""pYIN F0 detector (via ``librosa.pyin``).

Classical DSP baseline. Cheap and real-time friendly, but jittery on
breathy/noisy vocals and prone to octave errors. Default detector for the
Milestone-1 offline pipeline because it has no GPU / model-download
requirements.
"""

from __future__ import annotations

import numpy as np

from . import F0Result


class PyinDetector:
    name = "pyin"

    def __init__(self, fmin_hz: float = 60.0, fmax_hz: float = 1000.0) -> None:
        self.fmin_hz = fmin_hz
        self.fmax_hz = fmax_hz

    def detect(
        self,
        audio: np.ndarray,
        sample_rate: int,
        hop_size: int,
    ) -> F0Result:
        import librosa  # local import keeps module import cheap

        audio_f32 = np.ascontiguousarray(audio, dtype=np.float32)

        # frame_length defaults to ~2 pitch periods at fmin; librosa picks one
        # if we leave it None. Be explicit so latency budgets are predictable.
        frame_length = 2048
        if hop_size >= frame_length:
            frame_length = hop_size * 2

        f0, voiced_flag, voiced_prob = librosa.pyin(
            audio_f32,
            fmin=self.fmin_hz,
            fmax=self.fmax_hz,
            sr=sample_rate,
            hop_length=hop_size,
            frame_length=frame_length,
        )

        # librosa returns NaN on unvoiced frames; keep that convention.
        f0_clean = np.where(np.isfinite(f0), f0, np.nan).astype(float)
        voicing  = np.nan_to_num(voiced_prob, nan=0.0).astype(float)
        times    = librosa.times_like(f0, sr=sample_rate, hop_length=hop_size)

        return F0Result(times=times, f0_hz=f0_clean, voicing=voicing)
