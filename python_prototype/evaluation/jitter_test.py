"""F0 jitter, voiced/unvoiced error rate, and octave-jump rate per detector.

Frame-to-frame jitter is measured in cents and reported as RMS + p95.
Octave jumps are counted as consecutive-frame |delta cents| > 600.

VUV error compares each detector's voicing decision to a ground-truth
mask. When a ground-truth CSV is supplied (e.g. produced by the synthetic
signal generator), VUV errors are computed; otherwise they are skipped.

Usage:

    python -m python_prototype.evaluation.jitter_test \
        --input python_prototype/test_signals/audio/test_vocal_synth.wav \
        --groundtruth python_prototype/test_signals/audio/test_vocal_synth_groundtruth.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )

from python_prototype.f0_extractors import get_detector  # noqa: E402


DETECTORS = ("pyin", "torchcrepe")


def _load_mono(path: Path, target_sr: int) -> np.ndarray:
    import soundfile as sf
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1).astype(np.float32)
    if sr != target_sr:
        import librosa
        data = librosa.resample(y=data, orig_sr=sr, target_sr=target_sr)
    return data.astype(np.float32)


def _load_groundtruth(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (times_sec, f0_hz) where unvoiced rows have f0_hz == 0."""
    times: list[float] = []
    f0s:   list[float] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row["time_sec"]))
            f0s.append(float(row["f0_hz"]))
    return np.asarray(times), np.asarray(f0s)


def _align_to_groundtruth(
    det_times: np.ndarray,
    det_f0: np.ndarray,
    gt_times: np.ndarray,
    gt_f0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample det_f0 onto gt_times via nearest-neighbour. NaN preserved."""
    aligned = np.full_like(gt_times, np.nan, dtype=float)
    idx = np.searchsorted(det_times, gt_times)
    idx = np.clip(idx, 0, det_times.size - 1)
    aligned = det_f0[idx]
    return gt_f0, aligned


def _stats_for(detector_name: str,
               audio: np.ndarray,
               sample_rate: int,
               hop_size: int,
               gt: tuple[np.ndarray, np.ndarray] | None) -> dict:
    detector = get_detector(detector_name)
    res = detector.detect(audio, sample_rate, hop_size)
    f0 = res.f0_hz

    # Voiced-to-voiced frame deltas in cents.
    voiced_idx = np.where(np.isfinite(f0) & (f0 > 0))[0]
    cent_deltas = []
    octave_jumps = 0
    for j in range(1, voiced_idx.size):
        i_prev, i_cur = voiced_idx[j - 1], voiced_idx[j]
        if i_cur - i_prev != 1:
            continue  # crossed an unvoiced gap; skip
        d = 1200.0 * np.log2(f0[i_cur] / f0[i_prev])
        cent_deltas.append(d)
        if abs(d) > 600.0:
            octave_jumps += 1
    cent_deltas_arr = np.asarray(cent_deltas, dtype=float)
    jitter_rms = float(np.sqrt(np.mean(cent_deltas_arr ** 2))) if cent_deltas_arr.size else 0.0
    jitter_p95 = float(np.percentile(np.abs(cent_deltas_arr), 95)) if cent_deltas_arr.size else 0.0

    out: dict = dict(
        detector=detector_name,
        frames=int(f0.size),
        voiced_frames=int(voiced_idx.size),
        jitter_rms_cents=jitter_rms,
        jitter_p95_cents=jitter_p95,
        octave_jumps=octave_jumps,
    )

    if gt is not None:
        gt_times, gt_f0 = gt
        det_times = res.times
        gt_f0_a, det_f0_a = _align_to_groundtruth(det_times, f0, gt_times, gt_f0)
        gt_voiced  = gt_f0_a > 0
        det_voiced = np.isfinite(det_f0_a) & (det_f0_a > 0)
        vuv_err    = float(np.mean(gt_voiced != det_voiced))

        common = gt_voiced & det_voiced
        if common.sum():
            err_cents = 1200.0 * np.log2(det_f0_a[common] / gt_f0_a[common])
            out["pitch_mae_cents"]   = float(np.mean(np.abs(err_cents)))
            out["pitch_rmse_cents"]  = float(np.sqrt(np.mean(err_cents ** 2)))
        else:
            out["pitch_mae_cents"]   = float("nan")
            out["pitch_rmse_cents"]  = float("nan")
        out["vuv_error_rate"] = vuv_err
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--groundtruth", type=Path, default=None)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--hop-size", type=int, default=256)
    parser.add_argument("--detectors", nargs="+", default=list(DETECTORS))
    args = parser.parse_args()

    audio = _load_mono(args.input, args.sample_rate)
    gt = _load_groundtruth(args.groundtruth) if args.groundtruth else None

    out_rows: list[dict] = []
    header = (
        f"{'detector':<12} {'voiced':>7} {'jitter_rms':>11} "
        f"{'jitter_p95':>11} {'oct_jumps':>10}"
    )
    if gt is not None:
        header += f" {'mae_c':>7} {'rmse_c':>7} {'vuv_err':>8}"
    print(header)
    print("-" * len(header))
    for det in args.detectors:
        stats = _stats_for(det, audio, args.sample_rate, args.hop_size, gt)
        line = (
            f"{stats['detector']:<12} "
            f"{stats['voiced_frames']:>7} "
            f"{stats['jitter_rms_cents']:>11.2f} "
            f"{stats['jitter_p95_cents']:>11.2f} "
            f"{stats['octave_jumps']:>10}"
        )
        if gt is not None:
            line += (
                f" {stats.get('pitch_mae_cents', float('nan')):>7.2f} "
                f"{stats.get('pitch_rmse_cents', float('nan')):>7.2f} "
                f"{stats.get('vuv_error_rate', float('nan')):>8.3f}"
            )
        print(line)
        out_rows.append(stats)

    out_dir = Path(__file__).resolve().parent / "evaluation_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "jitter.csv"
    fieldnames = sorted({k for r in out_rows for k in r})
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
