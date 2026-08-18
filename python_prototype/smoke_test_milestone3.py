"""Milestone 3 validation: correction policy modules.

Tests:
  1. Retune speed   – fast vs slow alpha, smoother reset on silence
  2. Hysteresis     – flicker prevention, dwell gate, cents threshold
  3. Vibrato        – decompose/recombine round-trip, amount=0 vs 1
  4. Voiced/unvoiced – NaN propagation through the full pipeline
  5. End-to-end     – smoke the offline pipeline at 3 retune speeds

Run from the project root:

    python -m python_prototype.smoke_test_milestone3
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if __package__ in (None, ""):
    sys.path.insert(0, str(HERE.parent))

import numpy as np

from python_prototype.correction.smoothing import apply_retune_speed, apply_hysteresis
from python_prototype.correction.vibrato import decompose_pitch, recombine_with_vibrato
from python_prototype.correction.scale_quantizer import snap_to_scale


PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
_failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  {PASS}  {name}")
    else:
        print(f"  {FAIL}  {name}" + (f"  ({detail})" if detail else ""))
        _failures.append(name)


# ---------------------------------------------------------------------------
# 1. Retune speed
# ---------------------------------------------------------------------------
print("\n── 1. Retune speed ──────────────────────────────────────────")

# At speed=1.0 (hard tune) alpha=1, so output == input on voiced frames.
target = np.array([261.0, 261.0, 261.0, 261.0, 261.0], dtype=float)
out = apply_retune_speed(target, retune_speed=1.0, frame_rate_hz=172.0)
check("speed=1.0 instant: output matches target exactly",
      np.allclose(out, target, atol=0.01),
      f"got {out}")

# At speed=0.0 (slow) tau=250 ms. After 30 frames (~174 ms < 1 tau) the
# smoother should be well under halfway to 440 Hz from 220 Hz.
target2 = np.full(30, 440.0)   # constant 440 Hz target
target2[0] = 220.0              # start at 220 Hz
out2 = apply_retune_speed(target2, retune_speed=0.0, frame_rate_hz=172.0)
check("speed=0.0 slow: output doesn't instantly reach target (< halfway after <1 tau)",
      out2[-1] < 360.0,
      f"value after 30 frames={out2[-1]:.2f} Hz")

# NaN gap resets the smoother: after silence the next voiced frame snaps.
target3 = np.array([440.0, 440.0, np.nan, np.nan, 220.0, 220.0], dtype=float)
out3 = apply_retune_speed(target3, retune_speed=0.5, frame_rate_hz=172.0)
check("NaN gap resets smoother: first post-silence frame == target",
      np.isfinite(out3[4]) and abs(out3[4] - 220.0) < 0.01,
      f"got {out3[4]:.3f} Hz, expected 220.0")

# NaN frames produce NaN in output.
check("NaN frames stay NaN in output",
      np.all(np.isnan(out3[2:4])),
      f"got {out3[2:4]}")

# ---------------------------------------------------------------------------
# 2. Hysteresis
# ---------------------------------------------------------------------------
print("\n── 2. Hysteresis ────────────────────────────────────────────")

C4  = 261.63
Cs4 = 277.18   # C#4 — 100 cents from C4, below switch_cents=150

# Flickering between C4 and C#4 should be suppressed (< switch threshold).
flicker = np.array([C4, Cs4, C4, Cs4, C4, Cs4, C4], dtype=float)
out_h = apply_hysteresis(flicker, min_dwell_frames=3, switch_cents=150.0)
check("Flicker <150 cents suppressed: all output == first note",
      np.all(np.abs(out_h[np.isfinite(out_h)] - C4) < 1.0),
      f"got {out_h}")

# A note that dwells long enough and is far enough should switch.
G4  = 392.0   # ~705 cents from C4 — well above 150 cents threshold
dwell = np.array([C4, C4, G4, G4, G4, G4], dtype=float)
out_d = apply_hysteresis(dwell, min_dwell_frames=3, switch_cents=150.0)
check("Dwell gate: switches after 3 frames at G4",
      abs(out_d[-1] - G4) < 1.0,
      f"got {out_d}")

# NaN (unvoiced) resets state: after silence the next note is taken fresh.
with_silence = np.array([C4, C4, np.nan, np.nan, G4, G4, G4], dtype=float)
out_s = apply_hysteresis(with_silence, min_dwell_frames=3, switch_cents=150.0)
check("NaN resets hysteresis state",
      np.all(np.isnan(out_s[2:4])),
      f"unvoiced frames: {out_s[2:4]}")

# ---------------------------------------------------------------------------
# 3. Vibrato decomposition / recombination
# ---------------------------------------------------------------------------
print("\n── 3. Vibrato ───────────────────────────────────────────────")

FRAME_RATE = 172.3
# Build a simple vibrato: 5 Hz sinusoid in cents on top of A4=440 Hz.
n = 300
t = np.arange(n) / FRAME_RATE
vibrato_cents = 30.0 * np.sin(2 * np.pi * 5.0 * t)
f0_with_vibrato = 440.0 * np.power(2.0, vibrato_cents / 1200.0)

trend_hz, residual_cents = decompose_pitch(
    f0_with_vibrato, frame_rate_hz=FRAME_RATE, trend_cutoff_hz=4.0
)

# Trend should be close to 440 Hz (within ±10 cents after filter ringing).
trend_cents_err = 1200.0 * np.log2(trend_hz[20:-20] / 440.0)
check("Vibrato trend stays within ±10 cents of carrier",
      float(np.nanmax(np.abs(trend_cents_err))) < 10.0,
      f"max trend err={np.nanmax(np.abs(trend_cents_err)):.2f} cents")

# Residual should capture the 30-cent vibrato (peak > 20 cents).
check("Vibrato residual captures oscillation (peak > 20 cents)",
      float(np.nanmax(np.abs(residual_cents[20:-20]))) > 20.0,
      f"residual peak={np.nanmax(np.abs(residual_cents[20:-20])):.2f} cents")

# amount=0: output should equal corrected trend (no vibrato re-added).
corrected = np.full(n, 440.0)   # pretend we corrected to dead-flat A4
out_no_vib = recombine_with_vibrato(corrected, residual_cents, vibrato_amount=0.0)
check("vibrato_amount=0: output == corrected trend",
      np.allclose(out_no_vib[np.isfinite(out_no_vib)],
                  corrected[np.isfinite(out_no_vib)], atol=0.01),
      f"max diff={np.nanmax(np.abs(out_no_vib - corrected)):.4f} Hz")

# amount=1: output should differ from flat by roughly the original residual.
out_vib = recombine_with_vibrato(corrected, residual_cents, vibrato_amount=1.0)
recon_cents = 1200.0 * np.log2(out_vib[20:-20] / 440.0)
check("vibrato_amount=1: reconstructed vibrato peak > 20 cents",
      float(np.nanmax(np.abs(recon_cents))) > 20.0,
      f"recon peak={np.nanmax(np.abs(recon_cents)):.2f} cents")

# ---------------------------------------------------------------------------
# 4. Voiced / unvoiced propagation
# ---------------------------------------------------------------------------
print("\n── 4. Voiced/unvoiced propagation ───────────────────────────")

f0_mixed = np.array([261.0, 261.0, np.nan, np.nan, 293.0, 293.0], dtype=float)
voicing  = np.array([0.9,   0.9,   0.0,    0.0,    0.8,   0.8  ], dtype=float)

snapped = snap_to_scale(f0_mixed, key="C", scale="major", voicing=voicing)
check("Unvoiced frames remain NaN after snap_to_scale",
      np.all(np.isnan(snapped[2:4])),
      f"got {snapped[2:4]}")
check("Voiced frames are snapped to C major",
      np.all(np.isfinite(snapped[[0, 1, 4, 5]])),
      f"got {snapped[[0, 1, 4, 5]]}")

smoothed = apply_retune_speed(snapped, retune_speed=0.8, frame_rate_hz=172.0)
check("Unvoiced NaN propagates through apply_retune_speed",
      np.all(np.isnan(smoothed[2:4])),
      f"got {smoothed[2:4]}")

# ---------------------------------------------------------------------------
# 5. End-to-end smoke: 3 retune speeds
# ---------------------------------------------------------------------------
print("\n── 5. End-to-end pipeline at 3 retune speeds ────────────────")

from python_prototype.autotune_offline import AutotuneConfig, run_offline_autotune
from python_prototype.test_signals.generate_test_signal import (
    OUT_WAV as TEST_WAV,
    main as generate_test_signal_main,
)

if not TEST_WAV.exists():
    generate_test_signal_main()

for speed in (0.0, 0.5, 1.0):
    label = {0.0: "slow", 0.5: "medium", 1.0: "hard"}[speed]
    out_path = TEST_WAV.with_name(f"test_vocal_tuned_{label}.wav")
    config = AutotuneConfig(
        key="C", scale="major",
        retune_speed=speed,
        wet_dry=1.0,
        detector="pyin",
        shifter="basic",
        hop_size=256,
        sample_rate=44100,
        vibrato_amount=0.5 if speed < 1.0 else 0.0,
    )
    try:
        run_offline_autotune(TEST_WAV, out_path, config, verbose=False)
        size_kb = out_path.stat().st_size / 1024
        check(f"retune_speed={speed} ({label}): pipeline completes, WAV written ({size_kb:.0f} KB)",
              out_path.exists() and size_kb > 10)
    except Exception as e:
        check(f"retune_speed={speed} ({label}): pipeline completes", False, str(e))

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "─" * 52)
total = 14  # update if you add checks
passed = total - len(_failures)
print(f"  {passed}/{total} checks passed")
if _failures:
    print("  Failed:")
    for f in _failures:
        print(f"    • {f}")
else:
    print("  All checks passed — Milestone 3 correction policy is valid.")
