"""Real-time standalone prototype (Milestone 5).

Streams audio from the default input device through the autotune pipeline
to the default output device using ``sounddevice``. Logs end-to-end
latency every ~1 s.

Architecture (two-thread design):

    Audio thread (callback):
        - copies incoming block into a ring buffer
        - applies last-known cents shift (pure resample, no allocation)
        - writes output — must finish in < block_period ms

    Detection thread (background):
        - drains the ring buffer
        - runs pYIN on the accumulated analysis window
        - updates shared cents shift via a lock
        - target update rate ~ block_size / sample_rate * analysis_frames

This keeps the audio callback allocation-free and well under its deadline
even though pYIN takes ~50 ms.

Run:

    python -m python_prototype.realtime_prototype \
        --key C --scale major --retune-speed 0.8 \
        --block-size 512 --sample-rate 44100

List audio devices:

    python -m python_prototype.realtime_prototype --list-devices
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

from python_prototype.correction.scale_quantizer import snap_to_scale  # noqa: E402
from python_prototype.correction.smoothing import (                    # noqa: E402
    apply_retune_speed, _retune_speed_to_alpha,
)
from python_prototype.f0_extractors import get_detector                # noqa: E402


# ---------------------------------------------------------------------------
# Shared state between audio thread and detection thread
# ---------------------------------------------------------------------------

class _SharedState:
    """Lock-protected state written by the detection thread,
    read by the audio callback."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._detected_hz: float = 0.0
        self._smoothed_target_hz: float = 0.0
        self._cents: float = 0.0

    def update(self, detected_hz: float, smoothed_target_hz: float) -> None:
        cents = 0.0
        if detected_hz > 0.0 and smoothed_target_hz > 0.0:
            cents = 1200.0 * float(np.log2(smoothed_target_hz / detected_hz))
        with self._lock:
            self._detected_hz        = detected_hz
            self._smoothed_target_hz = smoothed_target_hz
            self._cents              = cents

    def get_cents(self) -> float:
        with self._lock:
            return self._cents


# ---------------------------------------------------------------------------
# Detection worker
# ---------------------------------------------------------------------------

class _DetectionWorker(threading.Thread):
    """Background thread: pulls audio chunks from a deque, accumulates an
    analysis buffer, runs pYIN, updates the shared state."""

    def __init__(self,
                 state: _SharedState,
                 sample_rate: int,
                 block_size: int,
                 key: str,
                 scale: str,
                 retune_speed: float,
                 detector_name: str = "pyin",
                 analysis_frames: int = 8) -> None:
        super().__init__(daemon=True, name="DetectionWorker")
        self.state         = state
        self.sample_rate   = sample_rate
        self.block_size    = block_size
        self.key           = key
        self.scale         = scale
        self.detector      = get_detector(detector_name)
        self.analysis_size = analysis_frames * block_size

        self.frame_rate_hz = sample_rate / block_size
        self.alpha = _retune_speed_to_alpha(retune_speed, self.frame_rate_hz)

        self.analysis_buf  = np.zeros(self.analysis_size, dtype=np.float32)
        self.smoothed_hz: float = 0.0

        # Audio chunks pushed by the callback.
        self.queue: deque[np.ndarray] = deque(maxlen=64)
        self._stop_event = threading.Event()

    def push(self, block: np.ndarray) -> None:
        """Called from the audio thread — must be fast."""
        self.queue.append(block.copy())

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        while not self._stop_event.is_set():
            if not self.queue:
                time.sleep(0.001)
                continue

            chunk = self.queue.popleft()

            # Roll the analysis buffer and append the new chunk.
            n = chunk.size
            self.analysis_buf = np.roll(self.analysis_buf, -n)
            self.analysis_buf[-n:] = chunk

            try:
                res = self.detector.detect(
                    self.analysis_buf, self.sample_rate, self.block_size
                )
            except Exception:
                continue

            if res.f0_hz.size == 0:
                continue

            f0_now      = float(res.f0_hz[-1])
            voicing_now = float(res.voicing[-1]) if res.voicing.size else 0.0

            if not np.isfinite(f0_now) or f0_now <= 0 or voicing_now < 0.5:
                self.smoothed_hz = 0.0
                self.state.update(0.0, 0.0)
                continue

            snapped_arr = snap_to_scale(
                np.array([f0_now]), self.key, self.scale,
                voicing=np.array([voicing_now]),
            )
            target_hz = float(snapped_arr[0])
            if not np.isfinite(target_hz):
                continue

            if self.smoothed_hz <= 0.0:
                self.smoothed_hz = target_hz
            else:
                self.smoothed_hz = (
                    self.alpha * target_hz +
                    (1.0 - self.alpha) * self.smoothed_hz
                )

            self.state.update(f0_now, self.smoothed_hz)


# ---------------------------------------------------------------------------
# Main processor (audio-thread-only operations)
# ---------------------------------------------------------------------------

class RealtimeAutotune:
    """Owns the shared state and detection worker.
    The audio callback calls `process()` only — no heavy work inside."""

    def __init__(self,
                 sample_rate: int,
                 block_size: int,
                 key: str,
                 scale: str,
                 retune_speed: float,
                 detector_name: str = "pyin") -> None:
        self.state   = _SharedState()
        self.worker  = _DetectionWorker(
            state=self.state,
            sample_rate=sample_rate,
            block_size=block_size,
            key=key,
            scale=scale,
            retune_speed=retune_speed,
            detector_name=detector_name,
        )
        self.worker.start()
        self.block_times_ms: deque[float] = deque(maxlen=200)

    def stop(self) -> None:
        self.worker.stop()

    # ------------------------------------------------------------------
    # Audio callback — runs in the sounddevice audio thread.
    # Must complete well within one block period.
    # ------------------------------------------------------------------

    def process(self, indata: np.ndarray, outdata: np.ndarray) -> None:
        t0 = time.perf_counter()
        mono_in = (indata[:, 0] if indata.ndim > 1 else indata).astype(np.float32)

        # Hand a copy to the detection worker (non-blocking).
        self.worker.push(mono_in)

        # Apply the last-known pitch shift.
        cents = self.state.get_cents()
        out   = _resample_shift(mono_in, cents)

        if outdata.ndim > 1:
            outdata[:, 0] = out
            if outdata.shape[1] > 1:
                outdata[:, 1] = out
        else:
            outdata[:] = out

        self.block_times_ms.append((time.perf_counter() - t0) * 1000.0)


# ---------------------------------------------------------------------------
# Cheap resample pitch shifter (audio-thread safe, allocation-free paths)
# ---------------------------------------------------------------------------

def _resample_shift(block: np.ndarray, cents: float) -> np.ndarray:
    """Two-pass linear-interpolation pitch shift. Rough but fast."""
    if not np.isfinite(cents) or abs(cents) < 1.0:
        return block

    ratio = 2.0 ** (cents / 1200.0)
    n     = block.size
    new_n = max(1, int(round(n / ratio)))

    # Pass 1: resample to new_n.
    xs    = np.linspace(0, n - 1, new_n)
    idx   = np.clip(xs.astype(int), 0, n - 2)
    frac  = xs - idx
    tmp   = (1.0 - frac) * block[idx] + frac * block[idx + 1]

    # Pass 2: stretch back to n.
    xs2   = np.linspace(0, new_n - 1, n)
    idx2  = np.clip(xs2.astype(int), 0, new_n - 2)
    frac2 = xs2 - idx2
    out   = (1.0 - frac2) * tmp[idx2] + frac2 * tmp[idx2 + 1]
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key",         default="C")
    parser.add_argument("--scale",       default="major",
                        choices=["major", "minor", "chromatic"])
    parser.add_argument("--retune-speed", type=float, default=0.8)
    parser.add_argument("--block-size",  type=int,   default=1024)
    parser.add_argument("--sample-rate", type=int,   default=44100)
    parser.add_argument("--detector",    default="pyin",
                        choices=["pyin", "torchcrepe"])
    parser.add_argument("--input-device",  type=int, default=None,
                        help="Input device index (see --list-devices).")
    parser.add_argument("--output-device", type=int, default=None,
                        help="Output device index (see --list-devices).")
    parser.add_argument("--seconds",     type=float, default=0.0,
                        help="Stop after N seconds (0 = run until Ctrl+C).")
    parser.add_argument("--list-devices", action="store_true",
                        help="Print available audio devices and exit.")
    args = parser.parse_args()

    import sounddevice as sd

    if args.list_devices:
        print(sd.query_devices())
        return

    rt = RealtimeAutotune(
        sample_rate=args.sample_rate,
        block_size=args.block_size,
        key=args.key,
        scale=args.scale,
        retune_speed=args.retune_speed,
        detector_name=args.detector,
    )

    block_period_ms = 1000.0 * args.block_size / args.sample_rate
    print(f"[realtime] block={args.block_size}  sr={args.sample_rate}  "
          f"detector={args.detector}  key={args.key} {args.scale}")
    print(f"[realtime] block period: {block_period_ms:.1f} ms  "
          f"(callback must finish well under this)")
    print(f"[realtime] Ctrl+C to stop")

    def callback(indata, outdata, frames, time_info, status):
        if status:
            print("[realtime] xrun:", status, file=sys.stderr)
        rt.process(indata, outdata)

    stream_kwargs: dict = dict(
        samplerate=args.sample_rate,
        blocksize=args.block_size,
        channels=(1, 1),
        dtype="float32",
        callback=callback,
    )
    if args.input_device is not None or args.output_device is not None:
        stream_kwargs["device"] = (args.input_device, args.output_device)

    with sd.Stream(**stream_kwargs):
        start = time.perf_counter()
        try:
            while True:
                time.sleep(1.0)
                if rt.block_times_ms:
                    arr = np.asarray(rt.block_times_ms)
                    print(f"[realtime] callback ms  "
                          f"mean={arr.mean():.2f}  "
                          f"p95={np.percentile(arr, 95):.2f}  "
                          f"max={arr.max():.2f}  "
                          f"(budget={block_period_ms:.1f} ms)")
                if args.seconds and (time.perf_counter() - start) >= args.seconds:
                    break
        except KeyboardInterrupt:
            pass

    rt.stop()
    print("[realtime] done.")


if __name__ == "__main__":
    main()
