"""Plot F0 contours for visual inspection.

Produces a single PNG with:
  - Raw detected F0 (per detector)
  - Allowed scale grid (horizontal dashed lines per MIDI note in the scale)
  - Optional snapped + smoothed target contour
  - Optional ground-truth F0 overlay

Usage:

    python -m python_prototype.evaluation.plot_f0 \
        --input python_prototype/test_signals/audio/test_vocal_synth.wav \
        --key C --scale major \
        --groundtruth python_prototype/test_signals/audio/test_vocal_synth_groundtruth.csv \
        --output python_prototype/evaluation/evaluation_results/plot_f0.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )

from python_prototype.f0_extractors import get_detector                # noqa: E402
from python_prototype.correction.scale_quantizer import (              # noqa: E402
    snap_to_scale, hz_to_midi, midi_to_hz, allowed_pitch_classes,
)
from python_prototype.correction.smoothing import apply_retune_speed   # noqa: E402


def _load_mono(path: Path, target_sr: int) -> np.ndarray:
    import soundfile as sf
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1).astype(np.float32)
    if sr != target_sr:
        import librosa
        data = librosa.resample(y=data, orig_sr=sr, target_sr=target_sr)
    return data.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--detectors", nargs="+", default=["pyin", "torchcrepe"])
    parser.add_argument("--key", default="C")
    parser.add_argument("--scale", default="major")
    parser.add_argument("--retune-speed", type=float, default=0.8)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--hop-size", type=int, default=256)
    parser.add_argument("--groundtruth", type=Path, default=None)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    audio = _load_mono(args.input, args.sample_rate)
    fig, ax = plt.subplots(figsize=(12, 5))

    # Scale grid as horizontal dashed lines between MIDI 36 (C2) and 84 (C6).
    allowed_pcs = allowed_pitch_classes(args.key, args.scale)
    for m in range(36, 85):
        if (m % 12) in allowed_pcs:
            ax.axhline(midi_to_hz(np.asarray(m, dtype=float)),
                       linestyle="--", linewidth=0.5, alpha=0.35)

    # Plot each detector's raw F0.
    for det_name in args.detectors:
        try:
            det = get_detector(det_name)
            res = det.detect(audio, args.sample_rate, args.hop_size)
            ax.plot(res.times, res.f0_hz, label=f"{det_name} raw", linewidth=1.0)

            target_hz = snap_to_scale(res.f0_hz, args.key, args.scale,
                                      voicing=res.voicing)
            smoothed  = apply_retune_speed(
                target_hz, args.retune_speed,
                frame_rate_hz=args.sample_rate / args.hop_size,
            )
            ax.plot(res.times, smoothed, label=f"{det_name} target",
                    linewidth=1.4, alpha=0.85)
        except Exception as e:
            print(f"[plot_f0] skipping {det_name}: {e}")

    # Optional ground truth.
    if args.groundtruth is not None:
        import csv
        gt_t, gt_f = [], []
        with args.groundtruth.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                gt_t.append(float(row["time_sec"]))
                gt_f.append(float(row["f0_hz"]))
        gt_t_arr = np.asarray(gt_t)
        gt_f_arr = np.where(np.asarray(gt_f) > 0, np.asarray(gt_f), np.nan)
        ax.plot(gt_t_arr, gt_f_arr, label="ground truth",
                color="black", linewidth=1.0, alpha=0.5)

    ax.set_yscale("log")
    ax.set_ylim(80, 1000)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("F0 (Hz)")
    ax.set_title(f"F0 vs scale ({args.key} {args.scale})")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, which="both", alpha=0.25)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=120)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
