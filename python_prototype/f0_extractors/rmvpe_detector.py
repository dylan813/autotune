"""RMVPE F0 detector (Robust Model for Vocal Pitch Estimation).

Stronger than pYIN/CREPE on breathy or mixed-source vocals, but heavier
and requires a manually-downloaded checkpoint.

Setup:
    1. pip install rmvpe   (or clone https://github.com/Dream-High/RMVPE)
    2. Download `rmvpe.pt` from the release page.
    3. Set the path via the RMVPE_MODEL_PATH env var, or pass it to the
       constructor.

This detector is treated as an offline / "high quality" reference for
Phase 2 comparison. Real-time use is not the goal here.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from . import F0Result


# Reference sample rate the model was trained on.
RMVPE_NATIVE_SR = 16000


class RmvpeDetector:
    name = "rmvpe"

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
        voicing_threshold: float = 0.03,
    ) -> None:
        if model_path is None:
            model_path = os.environ.get("RMVPE_MODEL_PATH")
        if not model_path:
            raise FileNotFoundError(
                "RMVPE model path not provided. Set RMVPE_MODEL_PATH or pass "
                "model_path=... to RmvpeDetector(). See "
                "docs/research_notes.md for setup."
            )
        self.model_path = model_path
        self.device = device
        self.voicing_threshold = voicing_threshold
        self._model = None  # lazy-loaded on first detect()

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _load_model(self):
        if self._model is not None:
            return self._model
        # The official RMVPE repo exposes `RMVPE` directly when installed
        # via pip. If the user vendored the source tree, the import path
        # is the same: `rmvpe.rmvpe.RMVPE`.
        try:
            from rmvpe import RMVPE  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "RMVPE package not installed. See docs/research_notes.md."
            ) from e
        device = self._resolve_device()
        self._model = RMVPE(self.model_path, hop_length=160, device=device)
        return self._model

    def detect(
        self,
        audio: np.ndarray,
        sample_rate: int,
        hop_size: int,
    ) -> F0Result:
        model = self._load_model()

        # RMVPE expects 16 kHz mono float32 audio.
        audio_f32 = np.ascontiguousarray(audio, dtype=np.float32)
        if sample_rate != RMVPE_NATIVE_SR:
            import librosa
            audio_f32 = librosa.resample(
                y=audio_f32,
                orig_sr=sample_rate,
                target_sr=RMVPE_NATIVE_SR,
            ).astype(np.float32)

        # `infer_from_audio` returns per-frame F0 at RMVPE's internal hop
        # (10 ms by default). We resample the F0 contour to the caller's
        # requested hop afterwards.
        f0_native = np.asarray(
            model.infer_from_audio(audio_f32, thred=self.voicing_threshold),
            dtype=float,
        )
        # Unvoiced -> 0 in RMVPE; convert to our NaN convention.
        f0_native = np.where(f0_native > 0, f0_native, np.nan)

        # RMVPE native hop length (samples at 16 kHz).
        native_hop_sec = 160 / RMVPE_NATIVE_SR  # 0.01 s
        native_times   = np.arange(f0_native.size, dtype=float) * native_hop_sec

        # Resample contour to caller's hop grid at the *input* sample rate.
        n_target = int(np.ceil(audio.size / hop_size))
        target_times = np.arange(n_target, dtype=float) * (hop_size / sample_rate)

        # Interpolate F0 in log domain, treating NaN segments as gaps.
        finite = np.isfinite(f0_native)
        if finite.sum() < 2:
            f0_resampled = np.full(n_target, np.nan)
        else:
            log_f0_native = np.log2(np.where(finite, f0_native, 1.0))
            log_f0_native[~finite] = np.nan
            # Linear interp on log F0; preserve NaN gaps.
            f0_resampled = _interp_with_gaps(target_times, native_times, log_f0_native)
            voiced_target = np.isfinite(f0_resampled)
            f0_resampled[voiced_target] = 2.0 ** f0_resampled[voiced_target]

        voicing = (~np.isnan(f0_resampled)).astype(float)
        return F0Result(times=target_times, f0_hz=f0_resampled, voicing=voicing)


def _interp_with_gaps(x_new: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Linear interpolation that returns NaN inside NaN gaps of ``y``.

    Treats every NaN in y as an unvoiced region. Output at x_new[i] is NaN
    if the nearest non-NaN samples in x bracket a NaN (i.e. the gap spans
    the query point).
    """
    out = np.full_like(x_new, np.nan, dtype=float)
    valid = np.isfinite(y)
    if valid.sum() < 2:
        return out
    xs = x[valid]
    ys = y[valid]

    # Quick linear interp for points inside [xs[0], xs[-1]].
    interp_vals = np.interp(x_new, xs, ys, left=np.nan, right=np.nan)

    # Reject any query point that falls inside a NaN gap of the original y.
    # We check by finding the bracketing original indices and seeing if any
    # sample inside the bracket was NaN.
    orig_idx = np.searchsorted(x, x_new)
    for i, q in enumerate(x_new):
        if not np.isfinite(interp_vals[i]):
            continue
        lo = max(0, orig_idx[i] - 1)
        hi = min(y.size - 1, orig_idx[i])
        if not (np.isfinite(y[lo]) and np.isfinite(y[hi])):
            interp_vals[i] = np.nan
    out[:] = interp_vals
    return out
