"""Per-frame inference latency for each detector.

Reports mean / p50 / p95 / p99 wall-clock latency for each
(detector, chunk_size) combination on a vocal-like buffer.

Usage (from the project root):

    python -m python_prototype.evaluation.latency_test \
        --input python_prototype/test_signals/audio/test_vocal_synth.wav \
        --iters 30

Outputs:
    - Pretty table to stdout
    - python_prototype/evaluation/evaluation_results/latency.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )

from python_prototype.f0_extractors import get_detector  # noqa: E402


CHUNK_SIZES = (256, 512, 1024, 2048, 4096)
DETECTORS   = ("pyin", "torchcrepe")  # RMVPE only if the user has installed the model


def _load_mono(path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    import soundfile as sf
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1).astype(np.float32)
    if sr != target_sr:
        import librosa
        data = librosa.resample(y=data, orig_sr=sr, target_sr=target_sr)
    return data.astype(np.float32), target_sr


def _percentiles(samples: Iterable[float]) -> dict[str, float]:
    arr = np.asarray(list(samples), dtype=float)
    return dict(
        mean=float(arr.mean()),
        p50=float(np.percentile(arr, 50)),
        p95=float(np.percentile(arr, 95)),
        p99=float(np.percentile(arr, 99)),
    )


def measure(
    audio: np.ndarray,
    sample_rate: int,
    detector_name: str,
    chunk_size: int,
    iters: int,
) -> dict[str, float]:
    """Time `iters` independent calls of `detector.detect` on `chunk_size` samples."""
    detector = get_detector(detector_name)

    # Warm-up: many models lazy-load / JIT on first call.
    n = min(chunk_size, audio.size)
    warm_chunk = audio[:n]
    try:
        detector.detect(warm_chunk, sample_rate, max(1, chunk_size // 4))
    except Exception as e:
        return dict(error=str(e))

    timings_ms: list[float] = []
    for k in range(iters):
        # Slide the analysis window through the audio so we hit different
        # content each iteration; wrap if necessary.
        start = (k * chunk_size) % max(1, audio.size - chunk_size + 1)
        chunk = audio[start:start + chunk_size]
        if chunk.size < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - chunk.size))
        t0 = time.perf_counter()
        detector.detect(chunk, sample_rate, max(1, chunk_size // 4))
        t1 = time.perf_counter()
        timings_ms.append((t1 - t0) * 1000.0)
    return _percentiles(timings_ms)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--detectors", nargs="+", default=list(DETECTORS))
    parser.add_argument("--chunk-sizes", nargs="+", type=int, default=list(CHUNK_SIZES))
    args = parser.parse_args()

    audio, sr = _load_mono(args.input, args.sample_rate)

    rows: list[dict] = []
    print(f"{'detector':<12} {'chunk':>6} {'mean ms':>9} {'p50':>7} {'p95':>7} {'p99':>7}")
    print("-" * 56)
    for det in args.detectors:
        for ch in args.chunk_sizes:
            stats = measure(audio, sr, det, ch, args.iters)
            if "error" in stats:
                print(f"{det:<12} {ch:>6} ERROR: {stats['error']}")
                continue
            print(f"{det:<12} {ch:>6} {stats['mean']:>9.2f} "
                  f"{stats['p50']:>7.2f} {stats['p95']:>7.2f} {stats['p99']:>7.2f}")
            rows.append(dict(detector=det, chunk=ch, **stats))

    out_dir = Path(__file__).resolve().parent / "evaluation_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "latency.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["detector", "chunk", "mean", "p50", "p95", "p99"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
