"""Offline autotune pipeline.

End-to-end:
    vocal.wav
    -> framewise F0 estimation (pyin / torchcrepe / rmvpe)
    -> voiced/unvoiced detection
    -> snap to selected musical scale
    -> hysteresis (anti-flicker)
    -> retune-speed smoothing
    -> optional vibrato preservation
    -> pitch shift (basic phase-vocoder OLA / Rubber Band)
    -> tuned_vocal.wav

Run:
    python -m python_prototype.autotune_offline \
        --input  path/to/vocal.wav \
        --output path/to/tuned.wav \
        --key C --scale major --retune-speed 0.8 --vibrato 0.6
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Make the file runnable as a script *and* as a module. When invoked as
# `python python_prototype/autotune_offline.py`, __package__ is empty and
# the imports below would fail; adding the project root to sys.path fixes it.
if __package__ in (None, ""):
    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

import numpy as np

from python_prototype.correction.scale_quantizer import snap_to_scale      # noqa: E402
from python_prototype.correction.smoothing import (                          # noqa: E402
    apply_retune_speed, apply_hysteresis,
)
from python_prototype.correction.vibrato import (                            # noqa: E402
    decompose_pitch, recombine_with_vibrato,
)
from python_prototype.f0_extractors import get_detector                      # noqa: E402


@dataclass
class AutotuneConfig:
    key: str = "C"
    scale: str = "major"
    retune_speed: float = 0.5     # 0 = no correction, 1 = hard tune
    wet_dry: float = 1.0          # 0 = dry only, 1 = wet only
    detector: str = "pyin"        # "pyin" | "torchcrepe" | "rmvpe"
    shifter: str = "basic"        # "basic" | "rubberband"
    hop_size: int = 256
    sample_rate: int = 44100
    voiced_threshold: float = 0.5
    hysteresis_min_dwell_frames: int = 3
    hysteresis_switch_cents: float = 40.0
    vibrato_amount: float = 0.0      # 0 = dead flat, 1 = preserve original
    vibrato_trend_cutoff_hz: float = 4.0
    preserve_formants: bool = True   # only used by rubberband shifter


def _load_audio(path: Path, target_sr: int) -> np.ndarray:
    """Load mono float32 audio at ``target_sr``. Uses soundfile + numpy.

    Resamples with librosa if the file's sample rate doesn't match.
    """
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        # Down-mix to mono.
        data = np.mean(data, axis=1).astype(np.float32)
    if sr != target_sr:
        import librosa
        data = librosa.resample(
            y=data.astype(np.float32),
            orig_sr=sr,
            target_sr=target_sr,
        ).astype(np.float32)
    return data


def _write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    # 16-bit PCM keeps things compatible with everything; we're not in a
    # critical-listening pipeline yet.
    sf.write(str(path), np.clip(audio, -1.0, 1.0), sample_rate, subtype="PCM_16")


def _compute_cents_shift(detected_hz: np.ndarray, target_hz: np.ndarray) -> np.ndarray:
    """Cents to shift each frame's audio by. Unvoiced frames -> NaN -> no shift."""
    detected = np.asarray(detected_hz, dtype=float)
    target   = np.asarray(target_hz,   dtype=float)
    out      = np.full_like(detected, np.nan)
    mask = (
        np.isfinite(detected) & (detected > 0)
        & np.isfinite(target) & (target > 0)
    )
    out[mask] = 1200.0 * np.log2(target[mask] / detected[mask])
    return out


def run_offline_autotune(
    input_wav: Path,
    output_wav: Path,
    config: AutotuneConfig,
    verbose: bool = True,
) -> None:
    """Run the offline autotune pipeline on ``input_wav`` and write ``output_wav``."""
    if verbose:
        print(f"[autotune] load    : {input_wav}")
    audio = _load_audio(Path(input_wav), config.sample_rate)
    if verbose:
        print(f"[autotune]            sr={config.sample_rate}, "
              f"len={audio.size / config.sample_rate:.2f} s")

    if verbose:
        print(f"[autotune] detect  : {config.detector}")
    detector = get_detector(config.detector)
    result = detector.detect(audio, config.sample_rate, config.hop_size)

    if verbose:
        n_voiced = int(np.sum(np.isfinite(result.f0_hz) & (result.f0_hz > 0)))
        print(f"[autotune]            frames={result.f0_hz.size}, "
              f"voiced={n_voiced} ({n_voiced / max(result.f0_hz.size,1):.0%})")

    if verbose:
        print(f"[autotune] snap    : {config.key} {config.scale}")
    target_hz = snap_to_scale(
        f0_hz=result.f0_hz,
        key=config.key,
        scale=config.scale,
        voicing=result.voicing,
        voiced_threshold=config.voiced_threshold,
    )

    if verbose:
        print(f"[autotune] hyst    : "
              f"min_dwell={config.hysteresis_min_dwell_frames} "
              f"switch_cents={config.hysteresis_switch_cents:.0f}")
    target_hz = apply_hysteresis(
        target_hz=target_hz,
        min_dwell_frames=config.hysteresis_min_dwell_frames,
        switch_cents=config.hysteresis_switch_cents,
    )

    frame_rate = config.sample_rate / config.hop_size
    if verbose:
        print(f"[autotune] smooth  : retune_speed={config.retune_speed:.2f} "
              f"(frame_rate={frame_rate:.1f} Hz)")
    smoothed_target = apply_retune_speed(
        target_hz=target_hz,
        retune_speed=config.retune_speed,
        frame_rate_hz=frame_rate,
    )

    # Optional vibrato preservation: pull the *detected* vibrato residual
    # back onto the corrected trend.
    if config.vibrato_amount > 0.0:
        if verbose:
            print(f"[autotune] vibrato : amount={config.vibrato_amount:.2f} "
                  f"trend_cutoff={config.vibrato_trend_cutoff_hz:.1f} Hz")
        _trend_hz, residual_cents = decompose_pitch(
            result.f0_hz,
            frame_rate_hz=frame_rate,
            trend_cutoff_hz=config.vibrato_trend_cutoff_hz,
        )
        smoothed_target = recombine_with_vibrato(
            corrected_trend_hz=smoothed_target,
            residual_cents=residual_cents,
            vibrato_amount=config.vibrato_amount,
        )

    cents_per_frame = _compute_cents_shift(result.f0_hz, smoothed_target)

    if verbose:
        finite = cents_per_frame[np.isfinite(cents_per_frame)]
        if finite.size:
            print(f"[autotune]            cents shift: "
                  f"min={finite.min():.1f} median={np.median(finite):.1f} "
                  f"max={finite.max():.1f}")

    if verbose:
        print(f"[autotune] shift   : {config.shifter}")
    if config.shifter == "basic":
        from python_prototype.pitch_shift.basic_shift import shift_by_cents
    elif config.shifter == "rubberband":
        from python_prototype.pitch_shift.rubberband_shift import shift_by_cents  # type: ignore
    else:
        raise ValueError(f"Unknown shifter: {config.shifter!r}")

    if config.shifter == "rubberband":
        wet = shift_by_cents(
            audio=audio,
            sample_rate=config.sample_rate,
            cents_per_frame=cents_per_frame,
            hop_size=config.hop_size,
            preserve_formants=config.preserve_formants,
        )
    else:
        wet = shift_by_cents(
            audio=audio,
            sample_rate=config.sample_rate,
            cents_per_frame=cents_per_frame,
            hop_size=config.hop_size,
        )

    # Length-align in case the shifter returned a slightly different length.
    if wet.size > audio.size:
        wet = wet[: audio.size]
    elif wet.size < audio.size:
        wet = np.pad(wet, (0, audio.size - wet.size))

    mix = config.wet_dry * wet + (1.0 - config.wet_dry) * audio

    if verbose:
        print(f"[autotune] write   : {output_wav}")
    _write_audio(Path(output_wav), mix.astype(np.float32), config.sample_rate)

    if verbose:
        print("[autotune] done.")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Offline autotune (Milestone 1).")
    p.add_argument("--input",  required=True, type=Path, help="Input vocal WAV.")
    p.add_argument("--output", required=True, type=Path, help="Output tuned WAV.")
    p.add_argument("--key",    default="C")
    p.add_argument("--scale",  default="major",
                   choices=["major", "minor", "chromatic", "custom"])
    p.add_argument("--retune-speed", type=float, default=0.5)
    p.add_argument("--wet-dry",      type=float, default=1.0)
    p.add_argument("--detector",     default="pyin",
                   choices=["pyin", "torchcrepe", "rmvpe"])
    p.add_argument("--shifter",      default="basic",
                   choices=["basic", "rubberband"])
    p.add_argument("--hop-size",     type=int, default=256)
    p.add_argument("--sample-rate",  type=int, default=44100)
    p.add_argument("--vibrato",      type=float, default=0.0,
                   help="Vibrato preservation amount in [0, 1]. "
                        "0 = dead flat, 1 = preserve original.")
    p.add_argument("--no-formants",  action="store_true",
                   help="Disable formant preservation in the rubberband shifter.")
    p.add_argument("--hysteresis-dwell",  type=int, default=3)
    p.add_argument("--hysteresis-cents",  type=float, default=40.0)
    p.add_argument("--quiet",        action="store_true",
                   help="Suppress progress logging.")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    config = AutotuneConfig(
        key=args.key,
        scale=args.scale,
        retune_speed=args.retune_speed,
        wet_dry=args.wet_dry,
        detector=args.detector,
        shifter=args.shifter,
        hop_size=args.hop_size,
        sample_rate=args.sample_rate,
        hysteresis_min_dwell_frames=args.hysteresis_dwell,
        hysteresis_switch_cents=args.hysteresis_cents,
        vibrato_amount=args.vibrato,
        preserve_formants=not args.no_formants,
    )
    run_offline_autotune(
        input_wav=args.input,
        output_wav=args.output,
        config=config,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
