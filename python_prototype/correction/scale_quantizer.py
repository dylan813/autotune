"""Snap detected pitches to the nearest note in a selected musical scale."""

from __future__ import annotations

from typing import Iterable

import numpy as np


# Scale degree offsets (semitones from tonic) for the common scales we ship.
SCALE_INTERVALS: dict[str, tuple[int, ...]] = {
    "major":      (0, 2, 4, 5, 7, 9, 11),
    "minor":      (0, 2, 3, 5, 7, 8, 10),
    "chromatic":  tuple(range(12)),
}

KEY_PC: dict[str, int] = {
    "C": 0,  "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4,
    "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8,
    "A": 9, "A#": 10, "Bb": 10, "B": 11,
}


def hz_to_midi(f0_hz: np.ndarray) -> np.ndarray:
    """Convert a Hz array to MIDI note number (float). 0 / NaN inputs -> NaN."""
    f0 = np.asarray(f0_hz, dtype=float)
    out = np.full_like(f0, np.nan, dtype=float)
    mask = np.isfinite(f0) & (f0 > 0)
    out[mask] = 69.0 + 12.0 * np.log2(f0[mask] / 440.0)
    return out


def midi_to_hz(midi: np.ndarray) -> np.ndarray:
    """Convert MIDI -> Hz. NaN passes through."""
    midi = np.asarray(midi, dtype=float)
    return 440.0 * np.power(2.0, (midi - 69.0) / 12.0)


def allowed_pitch_classes(key: str, scale: str,
                          custom: Iterable[int] | None = None) -> set[int]:
    """Return the set of pitch classes (0-11) allowed by ``key`` + ``scale``."""
    if scale == "custom":
        if custom is None:
            raise ValueError("scale='custom' requires `custom` pitch-class list")
        return {int(pc) % 12 for pc in custom}
    if scale not in SCALE_INTERVALS:
        raise ValueError(f"Unknown scale: {scale!r}")
    if key not in KEY_PC:
        raise ValueError(f"Unknown key: {key!r}")
    tonic = KEY_PC[key]
    return {(tonic + i) % 12 for i in SCALE_INTERVALS[scale]}


def _nearest_allowed_midi(midi_value: float, allowed_pcs: np.ndarray) -> float:
    """Vector-free nearest-allowed-MIDI helper for a single frame.

    Search candidates: each allowed pitch class placed in the nearest octave,
    plus its neighbours above and below. That always covers the nearest note
    because the worst case (allowed PCs separated by an octave) still gives
    a closest candidate within 6 semitones.
    """
    base_octave = int(np.floor(midi_value / 12.0))
    # Three octaves of candidates, flattened.
    candidates = np.concatenate([
        (base_octave - 1) * 12 + allowed_pcs,
        (base_octave    ) * 12 + allowed_pcs,
        (base_octave + 1) * 12 + allowed_pcs,
    ])
    idx = int(np.argmin(np.abs(candidates - midi_value)))
    return float(candidates[idx])


def snap_to_scale(
    f0_hz: np.ndarray,
    key: str,
    scale: str,
    voicing: np.ndarray | None = None,
    voiced_threshold: float = 0.5,
    custom_pitch_classes: Iterable[int] | None = None,
) -> np.ndarray:
    """Snap each framewise F0 to the nearest allowed MIDI note in the scale.

    Unvoiced frames (low voicing or non-positive / NaN F0) are returned as NaN
    so downstream stages can leave them alone.

    Args:
        f0_hz: shape (n_frames,), detected fundamental in Hz; NaN/0 = unvoiced.
        key: tonic (e.g. "C", "F#").
        scale: "major" | "minor" | "chromatic" | "custom".
        voicing: optional voicing/confidence in [0, 1], shape (n_frames,).
        voiced_threshold: frames with voicing < threshold treated as unvoiced.
        custom_pitch_classes: required when scale == "custom".

    Returns:
        target_hz: shape (n_frames,), nearest allowed-scale Hz; NaN on unvoiced.
    """
    f0 = np.asarray(f0_hz, dtype=float).copy()

    # Build the voiced mask.
    voiced_mask = np.isfinite(f0) & (f0 > 0)
    if voicing is not None:
        v = np.asarray(voicing, dtype=float)
        if v.shape != f0.shape:
            raise ValueError(
                f"voicing shape {v.shape} != f0 shape {f0.shape}"
            )
        voiced_mask &= (v >= voiced_threshold)

    allowed_pcs = np.array(
        sorted(allowed_pitch_classes(key, scale, custom_pitch_classes)),
        dtype=float,
    )

    midi = hz_to_midi(f0)
    target_midi = np.full_like(midi, np.nan)
    for i in np.flatnonzero(voiced_mask):
        target_midi[i] = _nearest_allowed_midi(midi[i], allowed_pcs)

    target_hz = np.full_like(f0, np.nan)
    voiced_target = np.isfinite(target_midi)
    target_hz[voiced_target] = midi_to_hz(target_midi[voiced_target])
    return target_hz
