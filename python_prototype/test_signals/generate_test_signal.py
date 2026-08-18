"""Generate a synthetic vocal-like test signal for the autotune pipeline.

Why synthetic first?
    - Deterministic ground truth for F0 (useful for jitter/octave-error eval).
    - Lets us run the pipeline end-to-end without a real microphone.
    - Forces the snap-to-scale logic to actually do something: notes are
      intentionally detuned off-grid by +/- 30 cents.

Signal contents:
    - 8 seconds @ 44.1 kHz, mono, 16-bit PCM.
    - Five sustained "notes" with 100 ms silent gaps between them
      (gives the detector real voiced/unvoiced transitions to handle).
    - Each note has a fundamental + a few harmonics with decreasing
      amplitude (rough vocal-like spectrum).
    - Vibrato: ~5 Hz, +/- 25 cents on sustained portion of each note.
    - Onset/offset envelope: short attack + release to avoid clicks.
    - Intentional detuning per note: chosen pitches sit between scale
      notes so the autotune has something to correct.

Run:
    python python_prototype/test_signals/generate_test_signal.py
    -> writes:   python_prototype/test_signals/audio/test_vocal_synth.wav
                 python_prototype/test_signals/audio/test_vocal_synth_groundtruth.csv

Uses only stdlib + numpy + a tiny standard-library WAV writer (the `wave`
module) so it runs even before the full requirements.txt is installed.
"""

from __future__ import annotations

import csv
import math
import os
import wave
from pathlib import Path

try:
    import numpy as np
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "numpy is required. Install it first:  pip install numpy"
    ) from e


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SAMPLE_RATE   = 44_100
DURATION_SEC  = 8.0
OUT_DIR       = Path(__file__).resolve().parent / "audio"
OUT_WAV       = OUT_DIR / "test_vocal_synth.wav"
OUT_GT_CSV    = OUT_DIR / "test_vocal_synth_groundtruth.csv"

# Notes are MIDI numbers; the cents offsets push them OFF the C-major grid on
# purpose so the autotune has something real to do. Final F0 = midi_to_hz(midi)
# multiplied by 2 ** (cents_offset / 1200).
NOTES = [
    # (midi, cents_offset, note_duration_sec)
    (60,  +30, 1.4),   # C4 sharp by 30 cents
    (64,  -25, 1.4),   # E4 flat by 25 cents
    (67,  +18, 1.4),   # G4 sharp by 18 cents
    (65,  -33, 1.4),   # F4 flat by 33 cents
    (62,  +12, 1.4),   # D4 sharp by 12 cents
]
GAP_SEC          = 0.10           # silence between notes
ATTACK_SEC       = 0.025
RELEASE_SEC      = 0.050
VIBRATO_HZ       = 5.0
VIBRATO_CENTS    = 25.0
# Harmonic amplitudes (relative to fundamental). Five harmonics with mild
# roll-off gives a rough "voicey" spectrum without sounding like a buzzy saw.
HARMONIC_GAINS   = (1.00, 0.55, 0.32, 0.18, 0.10)
NOISE_LEVEL      = 0.003          # tiny breath/noise floor
PEAK_NORMALIZE   = 0.9            # leave a little headroom


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def midi_to_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def adsr_envelope(num_samples: int,
                  sample_rate: int,
                  attack_sec: float,
                  release_sec: float) -> np.ndarray:
    """Linear attack + sustain at 1.0 + linear release. No decay or sustain dip."""
    env = np.ones(num_samples, dtype=np.float64)
    n_attack  = max(1, int(attack_sec  * sample_rate))
    n_release = max(1, int(release_sec * sample_rate))
    n_attack  = min(n_attack,  num_samples // 2)
    n_release = min(n_release, num_samples // 2)
    env[:n_attack]            = np.linspace(0.0, 1.0, n_attack, endpoint=True)
    env[-n_release:]          = np.linspace(1.0, 0.0, n_release, endpoint=True)
    return env


def synth_note(midi: float,
               cents_offset: float,
               duration_sec: float,
               sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (audio, f0_per_sample) for a single sustained note with vibrato."""
    n = int(round(duration_sec * sample_rate))
    t = np.arange(n, dtype=np.float64) / sample_rate

    # Base frequency, detuned per note.
    f0_base = midi_to_hz(midi) * (2.0 ** (cents_offset / 1200.0))

    # Vibrato is in cents, modulated by 5 Hz. Apply once the attack has had
    # a moment to settle so the vibrato doesn't dominate the onset.
    vibrato_gain = np.clip((t - 0.10) / 0.15, 0.0, 1.0)
    vibrato_cents_curve = vibrato_gain * VIBRATO_CENTS * np.sin(
        2.0 * math.pi * VIBRATO_HZ * t
    )
    f0_per_sample = f0_base * (2.0 ** (vibrato_cents_curve / 1200.0))

    # Integrate instantaneous frequency to get phase for the fundamental.
    phase = 2.0 * math.pi * np.cumsum(f0_per_sample) / sample_rate

    # Sum harmonics on the same phase to avoid de-tuned partials.
    audio = np.zeros(n, dtype=np.float64)
    for k, gain in enumerate(HARMONIC_GAINS, start=1):
        audio += gain * np.sin(k * phase)
    audio /= sum(HARMONIC_GAINS)

    # Apply onset/offset envelope.
    audio *= adsr_envelope(n, sample_rate, ATTACK_SEC, RELEASE_SEC)

    return audio, f0_per_sample


def build_signal() -> tuple[np.ndarray, np.ndarray]:
    """Concatenate notes with silent gaps. Returns (audio, f0_per_sample).
    Unvoiced samples have f0 = 0.0 in the ground-truth array."""
    pieces_audio: list[np.ndarray] = []
    pieces_f0:    list[np.ndarray] = []

    gap_samples = int(round(GAP_SEC * SAMPLE_RATE))
    silence_audio = np.zeros(gap_samples, dtype=np.float64)
    silence_f0    = np.zeros(gap_samples, dtype=np.float64)

    for i, (midi, cents, dur) in enumerate(NOTES):
        a, f = synth_note(midi, cents, dur, SAMPLE_RATE)
        pieces_audio.append(a)
        pieces_f0.append(f)
        if i != len(NOTES) - 1:
            pieces_audio.append(silence_audio)
            pieces_f0.append(silence_f0)

    audio = np.concatenate(pieces_audio)
    f0    = np.concatenate(pieces_f0)

    # Trim or pad to DURATION_SEC.
    target_n = int(round(DURATION_SEC * SAMPLE_RATE))
    if audio.size < target_n:
        pad = target_n - audio.size
        audio = np.concatenate([audio, np.zeros(pad, dtype=np.float64)])
        f0    = np.concatenate([f0,    np.zeros(pad, dtype=np.float64)])
    elif audio.size > target_n:
        audio = audio[:target_n]
        f0    = f0[:target_n]

    # Sprinkle a tiny noise floor so unvoiced regions aren't bit-exactly zero;
    # closer to a real recording.
    rng = np.random.default_rng(0xA1701)
    audio = audio + NOISE_LEVEL * rng.standard_normal(audio.size)

    # Peak normalize.
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio * (PEAK_NORMALIZE / peak)

    return audio.astype(np.float32), f0.astype(np.float32)


def write_wav_pcm16(path: Path, audio_float: np.ndarray, sample_rate: int) -> None:
    """Write a 16-bit PCM mono WAV using the stdlib `wave` module.

    Avoids a hard dependency on soundfile/scipy so this script runs from a
    fresh venv with only numpy installed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio_float, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)         # 16-bit
        w.setframerate(sample_rate)
        w.writeframes(pcm16.tobytes())


def write_groundtruth_csv(path: Path,
                          f0_per_sample: np.ndarray,
                          sample_rate: int,
                          hop_size: int = 256) -> None:
    """Down-sample the per-sample F0 to per-hop frames and dump as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = f0_per_sample.size
    rows = []
    for start in range(0, n, hop_size):
        end = min(start + hop_size, n)
        chunk = f0_per_sample[start:end]
        # Mean F0 over the hop; zero stays zero (unvoiced).
        if np.any(chunk > 0):
            voiced_mean = float(chunk[chunk > 0].mean())
            voiced_frac = float((chunk > 0).mean())
        else:
            voiced_mean = 0.0
            voiced_frac = 0.0
        time_s = (start + (end - start) / 2.0) / sample_rate
        rows.append((time_s, voiced_mean, voiced_frac))

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_sec", "f0_hz", "voiced_fraction"])
        for row in rows:
            writer.writerow([f"{row[0]:.6f}", f"{row[1]:.4f}", f"{row[2]:.4f}"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Generating synthetic vocal-like test signal ({DURATION_SEC:.1f} s @ "
          f"{SAMPLE_RATE} Hz)...")
    audio, f0_per_sample = build_signal()

    write_wav_pcm16(OUT_WAV, audio, SAMPLE_RATE)
    write_groundtruth_csv(OUT_GT_CSV, f0_per_sample, SAMPLE_RATE, hop_size=256)

    rel_wav = os.path.relpath(OUT_WAV, Path.cwd())
    rel_csv = os.path.relpath(OUT_GT_CSV, Path.cwd())
    print(f"Wrote audio        : {rel_wav}")
    print(f"Wrote ground-truth : {rel_csv}")

    # Sanity check / summary.
    voiced = f0_per_sample[f0_per_sample > 0]
    if voiced.size:
        print(
            f"Voiced fraction    : {voiced.size / f0_per_sample.size:.2%}  "
            f"F0 range: {voiced.min():.2f} Hz .. {voiced.max():.2f} Hz"
        )
    print("Done.")


if __name__ == "__main__":
    main()
