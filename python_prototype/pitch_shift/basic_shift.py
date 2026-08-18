"""Basic baseline pitch shifter (Milestone 1).

Goal: produce *something* you can listen to, even if it's a phase vocoder
that smears transients. Real vocal-quality work lives in
``rubberband_shift.py`` and (eventually) a PSOLA implementation.

Strategy: overlap-add of pitch-shifted analysis windows.

    - Analysis window length = 2048 samples (~46 ms @ 44.1 kHz).
    - Output hop = 512 samples (75% overlap).
    - For each window, pick the cents value at the window's center sample,
      shift with `librosa.effects.pitch_shift`, multiply by a Hann window,
      and overlap-add into the output buffer. The Hann's square-sum at
      75% overlap is constant, so no extra normalisation is needed.

This is intentionally simple. Milestone 4 swaps in Rubber Band for real
vocal quality.
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
) -> np.ndarray:
    """Pitch-shift ``audio`` by a time-varying cents amount.

    Args:
        audio: mono float32/float64 in [-1, 1].
        sample_rate: Hz.
        cents_per_frame: shape (n_frames,), shift in cents per analysis frame.
            NaN means "no shift" (unvoiced frame -> pass through).
        hop_size: samples between consecutive ``cents_per_frame`` entries.

    Returns:
        Shifted audio of the same length as ``audio``.
    """
    import librosa

    audio = np.asarray(audio, dtype=np.float32)
    cents = np.asarray(cents_per_frame, dtype=float)
    n = audio.size

    if n == 0:
        return audio.copy()

    out      = np.zeros(n, dtype=np.float32)
    win_sum  = np.zeros(n, dtype=np.float32)   # normalisation accumulator
    hann     = np.hanning(WINDOW_SIZE).astype(np.float32)

    # Pad both sides so the first/last windows have audio to work with.
    pad = WINDOW_SIZE // 2
    padded = np.pad(audio, pad, mode="constant")

    for start in range(0, n, OUT_HOP):
        window_audio = padded[start : start + WINDOW_SIZE]
        if window_audio.size < WINDOW_SIZE:
            # Final window may be short; pad to full length.
            window_audio = np.pad(
                window_audio, (0, WINDOW_SIZE - window_audio.size)
            )

        # Cents value to use for this window: look up the analysis frame
        # closest to the window's center in the *original* (unpadded) timeline.
        center = start + WINDOW_SIZE // 2 - pad
        frame_idx = int(round(center / hop_size))
        frame_idx = max(0, min(frame_idx, cents.size - 1))
        c = cents[frame_idx]

        if not np.isfinite(c) or abs(c) < 1.0:
            # Unvoiced or basically-no-shift: copy straight through.
            shifted = window_audio
        else:
            n_steps = c / 100.0  # semitones
            shifted = librosa.effects.pitch_shift(
                y=window_audio,
                sr=sample_rate,
                n_steps=n_steps,
            ).astype(np.float32)

            # `pitch_shift` should preserve length but occasionally returns
            # +/- 1 sample due to internal STFT rounding; guard against that.
            if shifted.size > WINDOW_SIZE:
                shifted = shifted[:WINDOW_SIZE]
            elif shifted.size < WINDOW_SIZE:
                shifted = np.pad(shifted, (0, WINDOW_SIZE - shifted.size))

        # The analysis window read from padded[start:start+W] is centered at
        # original sample `start` (because pad == W/2). Write it back centered
        # there too, otherwise the wet output is delayed by half a window
        # relative to the dry signal -- audible when wet_dry < 1 and just
        # added latency at wet_dry == 1.
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

    # Window-sum normalisation. Anywhere we accumulated near-zero window
    # weight (the very edges) just falls back to silence.
    eps = 1e-6
    safe = win_sum > eps
    out[safe] = out[safe] / win_sum[safe]
    return out
