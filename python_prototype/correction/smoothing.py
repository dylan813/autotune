"""Retune-speed smoothing + hysteresis for the correction trajectory."""

from __future__ import annotations

import math

import numpy as np


def _retune_speed_to_alpha(retune_speed: float, frame_rate_hz: float) -> float:
    """Map ``retune_speed`` in [0, 1] to a per-frame mixing coefficient.

    Interpret ``retune_speed`` as a time constant. Speed=1.0 -> ~5 ms tau
    (essentially instant), speed=0.0 -> ~250 ms tau (very slow correction).
    Then alpha = 1 - exp(-1 / (tau * frame_rate)).
    """
    retune_speed = float(np.clip(retune_speed, 0.0, 1.0))
    if retune_speed >= 0.999:
        return 1.0
    tau_ms = math.exp(
        math.log(5.0)   * retune_speed +
        math.log(250.0) * (1.0 - retune_speed)
    )
    tau_sec = tau_ms / 1000.0
    return 1.0 - math.exp(-1.0 / max(tau_sec * frame_rate_hz, 1e-6))


def apply_retune_speed(
    target_hz: np.ndarray,
    retune_speed: float,
    frame_rate_hz: float,
) -> np.ndarray:
    """Low-pass the target pitch trajectory.

    Reference recurrence:

        corrected[t] = alpha * target[t] + (1 - alpha) * corrected[t-1]

    NaN frames keep NaN AND reset the smoother state, so the first voiced
    frame after a silence isn't pulled toward the last note we sang.
    """
    target = np.asarray(target_hz, dtype=float)
    out    = np.full_like(target, np.nan)
    alpha  = _retune_speed_to_alpha(retune_speed, frame_rate_hz)

    prev: float = float("nan")
    for i in range(target.size):
        x = target[i]
        if not np.isfinite(x):
            prev = float("nan")
            continue
        if not np.isfinite(prev):
            prev = x
        else:
            prev = alpha * x + (1.0 - alpha) * prev
        out[i] = prev
    return out


def apply_hysteresis(
    target_hz: np.ndarray,
    min_dwell_frames: int = 3,
    switch_cents: float = 40.0,
) -> np.ndarray:
    """Prevent flicker between two adjacent target notes.

    The input is the *quantized* target trajectory (one MIDI note per voiced
    frame, NaN on unvoiced frames). We refuse to switch the held note unless:

      1. The candidate note differs from the held one, AND
      2. The detected target has been at the candidate for at least
         ``min_dwell_frames`` consecutive voiced frames, AND
      3. The cents distance from the held note to the candidate exceeds
         ``switch_cents`` (typically ~half a semitone).

    Args:
        target_hz: shape (n_frames,), already snapped to scale (Hz),
            NaN on unvoiced.
        min_dwell_frames: minimum consecutive frames before switching.
        switch_cents: required cents distance to allow a switch.

    Returns:
        Hysteresis-stabilised target (Hz), same shape as input.
    """
    f = np.asarray(target_hz, dtype=float).copy()
    out = np.full_like(f, np.nan)

    held: float = float("nan")
    candidate: float = float("nan")
    candidate_count: int = 0

    for i in range(f.size):
        x = f[i]
        if not np.isfinite(x):
            # Unvoiced: reset everything so we don't carry stale state.
            held = float("nan")
            candidate = float("nan")
            candidate_count = 0
            continue

        if not np.isfinite(held):
            held = x
            candidate = x
            candidate_count = 1
        elif abs(_cents_between(x, held)) < 1.0:
            # Same note as held.
            candidate = held
            candidate_count = 0
        else:
            if np.isfinite(candidate) and abs(_cents_between(x, candidate)) < 1.0:
                candidate_count += 1
            else:
                candidate = x
                candidate_count = 1

            cents_distance = abs(_cents_between(candidate, held))
            if candidate_count >= min_dwell_frames and cents_distance >= switch_cents:
                held = candidate
                candidate_count = 0

        out[i] = held
    return out


def _cents_between(a_hz: float, b_hz: float) -> float:
    if not (np.isfinite(a_hz) and np.isfinite(b_hz)) or a_hz <= 0 or b_hz <= 0:
        return float("nan")
    return 1200.0 * math.log2(a_hz / b_hz)
