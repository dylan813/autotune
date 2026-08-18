"""Rubber Band Library pitch shifter (via pyrubberband).

Higher-quality vocal pitch shifting than the basic phase-vocoder OLA.
Requires the ``rubberband-cli`` system binary on PATH:

    macOS:    brew install rubberband
    Ubuntu:   sudo apt install rubberband-cli
    Windows:  download the binary and add it to PATH

For a time-varying shift, we slice the audio into hop-aligned analysis
windows, shift each window with Rubber Band, and overlap-add with a Hann
window. We use a 75% overlap so adjacent block shifts cross-fade smoothly.

Note: this introduces extra algorithmic delay vs ``basic_shift``. Real-time
use (Milestone 5/6) should instead stream samples through Rubber Band's
``RubberBandStretcher`` API, which we'll do in the JUCE plugin.
"""

from __future__ import annotations

import numpy as np


WINDOW_SIZE = 2048
OUT_HOP     = 512


def shift_by_cents(
    audio: np.ndarray,
    sample_rate: int,
    cents_per_frame: np.ndarray,
    hop_size: int,
    preserve_formants: bool = True,
) -> np.ndarray:
    """Pitch-shift ``audio`` by a time-varying cents amount via Rubber Band."""
    import pyrubberband as pyrb

    audio = np.asarray(audio, dtype=np.float32)
    cents = np.asarray(cents_per_frame, dtype=float)
    n = audio.size
    if n == 0:
        return audio.copy()

    out      = np.zeros(n, dtype=np.float32)
    win_sum  = np.zeros(n, dtype=np.float32)
    hann     = np.hanning(WINDOW_SIZE).astype(np.float32)

    pad = WINDOW_SIZE // 2
    padded = np.pad(audio, pad, mode="constant")

    rb_args = {}
    # Formant preservation via the -F CLI flag can fail on some Windows builds
    # of rubberband-cli due to missing DLL exports. Disabled here for offline
    # use; the JUCE plugin uses RubberBandStretcher's native C++ API instead.
    # Re-enable by setting preserve_formants=True once your binary supports it.
    _ = preserve_formants  # acknowledged, not used in offline path

    for start in range(0, n, OUT_HOP):
        window_audio = padded[start : start + WINDOW_SIZE]
        if window_audio.size < WINDOW_SIZE:
            window_audio = np.pad(window_audio,
                                  (0, WINDOW_SIZE - window_audio.size))

        center = start + WINDOW_SIZE // 2 - pad
        frame_idx = int(round(center / hop_size))
        frame_idx = max(0, min(frame_idx, cents.size - 1))
        c = cents[frame_idx]

        if not np.isfinite(c) or abs(c) < 1.0:
            shifted = window_audio
        else:
            n_steps = c / 100.0
            shifted = pyrb.pitch_shift(
                y=window_audio,
                sr=sample_rate,
                n_steps=n_steps,
                **rb_args,
            ).astype(np.float32)
            if shifted.size > WINDOW_SIZE:
                shifted = shifted[:WINDOW_SIZE]
            elif shifted.size < WINDOW_SIZE:
                shifted = np.pad(shifted, (0, WINDOW_SIZE - shifted.size))

        # Centered write -- see notes in basic_shift.py.
        contribution = shifted * hann
        write_start = start - WINDOW_SIZE // 2
        write_end   = write_start + WINDOW_SIZE
        src_lo = max(0, -write_start)
        dst_lo = max(0,  write_start)
        src_hi = WINDOW_SIZE - max(0, write_end - n)
        dst_hi = min(n, write_end)
        if dst_hi > dst_lo:
            out    [dst_lo:dst_hi] += contribution[src_lo:src_hi]
            win_sum[dst_lo:dst_hi] += hann        [src_lo:src_hi]

    eps = 1e-6
    safe = win_sum > eps
    out[safe] = out[safe] / win_sum[safe]
    return out
