"""On-the-fly synthetic vocal-like training data.

Generates monophonic excerpts where:
    - F0 follows a piecewise-constant melody with smooth note transitions.
    - Vibrato is added on sustained notes with random rate (4-7 Hz) and
      depth (10-40 cents).
    - A few harmonics roll off with random spectral tilts.
    - A small amount of broadband noise + occasional unvoiced gaps.
    - Output: (audio_samples, f0_per_frame_hz, voiced_mask_per_frame).

This is the same family of signal as test_signals/generate_test_signal.py
but randomized for training. Ground truth F0 is sampled at the model's
hop_size, so loss is per-frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


@dataclass
class SyntheticDataConfig:
    sample_rate: int     = 16000
    duration_sec: float  = 2.0
    hop_size: int        = 256
    min_f0_hz: float     = 80.0
    max_f0_hz: float     = 600.0
    p_unvoiced: float    = 0.15      # probability a "note" is replaced by silence/noise
    vibrato_rate_hz: tuple[float, float] = (4.0, 7.0)
    vibrato_cents:   tuple[float, float] = (10.0, 40.0)
    n_harmonics:     int   = 6
    noise_level:     float = 0.01


def _rand_uniform(lo: float, hi: float, rng: np.random.Generator) -> float:
    return float(rng.uniform(lo, hi))


def synth_one(cfg: SyntheticDataConfig,
              rng: np.random.Generator | None = None
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a single (audio, f0_per_frame, voiced_per_frame) triple."""
    if rng is None:
        rng = np.random.default_rng()
    sr = cfg.sample_rate
    n  = int(cfg.duration_sec * sr)
    t  = np.arange(n, dtype=np.float64) / sr

    n_frames = n // cfg.hop_size
    f0_frames     = np.zeros(n_frames, dtype=np.float64)
    voiced_frames = np.zeros(n_frames, dtype=bool)

    # Build a piecewise-constant melody at the frame rate.
    cur_frame = 0
    while cur_frame < n_frames:
        note_len = int(rng.integers(20, 80))  # frames
        end = min(cur_frame + note_len, n_frames)

        if rng.uniform() < cfg.p_unvoiced:
            f0_frames[cur_frame:end]     = 0.0
            voiced_frames[cur_frame:end] = False
        else:
            midi = float(rng.uniform(
                12.0 * np.log2(cfg.min_f0_hz / 16.351597831287414),
                12.0 * np.log2(cfg.max_f0_hz / 16.351597831287414),
            ))
            note_hz = 16.351597831287414 * (2.0 ** (midi / 12.0))
            f0_frames[cur_frame:end]     = note_hz
            voiced_frames[cur_frame:end] = True
        cur_frame = end

    # Smooth pitch transitions across boundaries by linear interp in cents.
    # (Random small portamento gives the model continuous F0 trajectories.)
    cents = np.where(f0_frames > 0,
                     1200.0 * np.log2(np.clip(f0_frames, 1.0, None) / 16.351597831287414),
                     np.nan)
    # Interpolate over short voiced runs only.
    valid = np.isfinite(cents)
    if valid.sum() > 4:
        # Smooth voiced regions with a tiny moving average.
        k = 5
        kernel = np.ones(k) / k
        smoothed = cents.copy()
        smoothed[~valid] = 0.0
        smoothed = np.convolve(smoothed, kernel, mode="same")
        # Re-weight by the convolution of the validity mask to compensate edges.
        weights = np.convolve(valid.astype(float), kernel, mode="same")
        with np.errstate(invalid="ignore", divide="ignore"):
            smoothed = np.where(weights > 0, smoothed / weights, np.nan)
        cents = np.where(valid, smoothed, np.nan)
    f0_frames = np.where(np.isfinite(cents),
                         16.351597831287414 * np.power(2.0, cents / 1200.0),
                         0.0)

    # Add vibrato on each sustained voiced segment.
    f0_per_sample = np.zeros(n, dtype=np.float64)
    for f_idx in range(n_frames):
        if not voiced_frames[f_idx]:
            continue
        start = f_idx * cfg.hop_size
        end   = min(start + cfg.hop_size, n)
        f0_per_sample[start:end] = f0_frames[f_idx]

    if voiced_frames.any():
        rate  = _rand_uniform(*cfg.vibrato_rate_hz, rng=rng)
        depth = _rand_uniform(*cfg.vibrato_cents,   rng=rng)
        vibrato = depth * np.sin(2.0 * np.pi * rate * t)
        f0_per_sample = np.where(
            f0_per_sample > 0,
            f0_per_sample * np.power(2.0, vibrato / 1200.0),
            0.0,
        )

    # Synthesize. Integrate instantaneous frequency to get phase.
    phase = 2.0 * np.pi * np.cumsum(f0_per_sample) / sr
    audio = np.zeros(n, dtype=np.float64)
    # Random spectral tilt -> harmonic gains.
    tilt = rng.uniform(0.6, 0.95)
    gains = np.power(tilt, np.arange(1, cfg.n_harmonics + 1))
    gains /= gains.sum()
    for k, g in enumerate(gains, start=1):
        audio += g * np.sin(k * phase)
    # Voiced/unvoiced gating in the time domain.
    voiced_per_sample = f0_per_sample > 0
    audio = audio * voiced_per_sample.astype(np.float64)

    # Noise / breath.
    audio = audio + cfg.noise_level * rng.standard_normal(n)

    # Peak normalize-ish.
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio * (0.9 / peak)

    return audio.astype(np.float32), f0_per_sample.astype(np.float32), voiced_frames


def make_batch(batch_size: int,
               cfg: SyntheticDataConfig,
               rng: np.random.Generator | None = None,
               device: str = "cpu"
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack `batch_size` examples into a training batch.

    Returns:
        audio:       (B, T_samples) float32
        f0_frames:   (B, T_frames)  float32  -- per-frame F0 in Hz, 0 if unvoiced
        voiced_mask: (B, T_frames)  bool
    """
    if rng is None:
        rng = np.random.default_rng()
    audios, f0s, voiced = [], [], []
    for _ in range(batch_size):
        a, f0_per_sample, v_frames = synth_one(cfg, rng=rng)
        # Down-sample per-sample F0 to per-frame: mean over the hop.
        n = a.size
        n_frames = n // cfg.hop_size
        f_frames = np.zeros(n_frames, dtype=np.float32)
        for fi in range(n_frames):
            chunk = f0_per_sample[fi * cfg.hop_size:(fi + 1) * cfg.hop_size]
            voiced_chunk = chunk > 0
            f_frames[fi] = float(chunk[voiced_chunk].mean()) if voiced_chunk.any() else 0.0
        audios.append(a)
        f0s.append(f_frames)
        voiced.append(v_frames)
    audio_t   = torch.from_numpy(np.stack(audios)).to(device)
    f0_t      = torch.from_numpy(np.stack(f0s)).to(device)
    voiced_t  = torch.from_numpy(np.stack(voiced)).to(device)
    return audio_t, f0_t, voiced_t
