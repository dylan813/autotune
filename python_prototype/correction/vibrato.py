"""Vibrato preservation.

Split the pitch contour into a slow trend (the note being sung) and a fast
residual (vibrato / expressive movement). Correct the trend, optionally add
back a fraction of the residual.

All decomposition happens in cents space (12 * log2(f / ref)) because that
makes the split linear and frequency-independent. We use a Butterworth
low-pass for the trend; the residual is the original minus the trend.

NaN-aware: unvoiced gaps are preserved.
"""

from __future__ import annotations

import numpy as np


def _hz_to_cents(f0_hz: np.ndarray, ref_hz: float = 440.0) -> np.ndarray:
    f0 = np.asarray(f0_hz, dtype=float)
    out = np.full_like(f0, np.nan)
    mask = np.isfinite(f0) & (f0 > 0)
    out[mask] = 1200.0 * np.log2(f0[mask] / ref_hz)
    return out


def _cents_to_hz(cents: np.ndarray, ref_hz: float = 440.0) -> np.ndarray:
    cents = np.asarray(cents, dtype=float)
    out = np.full_like(cents, np.nan)
    mask = np.isfinite(cents)
    out[mask] = ref_hz * np.power(2.0, cents[mask] / 1200.0)
    return out


def _butter_lowpass(cents_contour: np.ndarray,
                    frame_rate_hz: float,
                    cutoff_hz: float) -> np.ndarray:
    """Butterworth zero-phase low-pass, NaN-aware.

    NaN segments are bridged by linear interpolation, filtered, then the
    output is masked back to NaN where input was NaN.
    """
    from scipy.signal import butter, filtfilt

    n = cents_contour.size
    if n == 0:
        return cents_contour.copy()

    finite = np.isfinite(cents_contour)
    if finite.sum() < 8:
        return cents_contour.copy()

    # Linear bridge across NaN gaps for the filter input.
    x = cents_contour.copy()
    xs = np.arange(n)
    x[~finite] = np.interp(xs[~finite], xs[finite], cents_contour[finite])

    nyq = 0.5 * frame_rate_hz
    wn  = float(np.clip(cutoff_hz / nyq, 1e-4, 0.99))
    b, a = butter(4, wn, btype="low")
    y = filtfilt(b, a, x, method="gust")

    y[~finite] = np.nan
    return y


def decompose_pitch(
    f0_hz: np.ndarray,
    frame_rate_hz: float,
    trend_cutoff_hz: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (slow_trend_hz, fast_residual_cents).

    ``slow_trend_hz`` is the same shape as ``f0_hz`` (NaN on unvoiced).
    ``fast_residual_cents`` is the original-minus-trend in cents (NaN on unvoiced).
    """
    cents = _hz_to_cents(f0_hz)
    trend_cents = _butter_lowpass(cents, frame_rate_hz, trend_cutoff_hz)
    residual_cents = cents - trend_cents
    trend_hz = _cents_to_hz(trend_cents)
    return trend_hz, residual_cents


def recombine_with_vibrato(
    corrected_trend_hz: np.ndarray,
    residual_cents: np.ndarray,
    vibrato_amount: float = 1.0,
) -> np.ndarray:
    """Add ``vibrato_amount`` * residual back onto the corrected trend.

    ``vibrato_amount`` = 0   -> dead-flat correction (no vibrato preserved)
    ``vibrato_amount`` = 1.0 -> preserve original expressive movement
    """
    vibrato_amount = float(np.clip(vibrato_amount, 0.0, 4.0))
    trend = np.asarray(corrected_trend_hz, dtype=float)
    res   = np.asarray(residual_cents, dtype=float)

    out = np.full_like(trend, np.nan)
    mask = np.isfinite(trend) & np.isfinite(res)
    out[mask] = trend[mask] * np.power(2.0, vibrato_amount * res[mask] / 1200.0)
    # If we have trend but no residual (start of voiced region), pass trend through.
    only_trend = np.isfinite(trend) & ~np.isfinite(res)
    out[only_trend] = trend[only_trend]
    return out
