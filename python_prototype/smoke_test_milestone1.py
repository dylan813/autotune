"""Milestone 1 smoke test.

Generates the synthetic vocal-like test signal (if missing) and runs the
offline autotune pipeline on it.

Outputs land in python_prototype/test_signals/audio/:
    test_vocal_synth.wav   (input)
    test_vocal_tuned.wav   (output)

Run from the repository root:

    python -m python_prototype.smoke_test_milestone1

Or as a script:

    python python_prototype/smoke_test_milestone1.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the script runnable both ways.
HERE = Path(__file__).resolve().parent
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(HERE.parent))

from python_prototype.autotune_offline import AutotuneConfig, run_offline_autotune
from python_prototype.test_signals.generate_test_signal import (
    OUT_WAV as TEST_WAV,
    main as generate_test_signal_main,
)


def main() -> None:
    if not TEST_WAV.exists():
        print(f"[smoke] {TEST_WAV.name} missing; generating it now.")
        generate_test_signal_main()
    else:
        print(f"[smoke] reusing existing {TEST_WAV}")

    out_path = TEST_WAV.with_name("test_vocal_tuned.wav")

    config = AutotuneConfig(
        key="C",
        scale="major",
        retune_speed=0.8,
        wet_dry=1.0,
        detector="pyin",
        shifter="basic",
        hop_size=256,
        sample_rate=44100,
    )

    run_offline_autotune(
        input_wav=TEST_WAV,
        output_wav=out_path,
        config=config,
        verbose=True,
    )
    print(f"\n[smoke] tuned audio -> {out_path}")


if __name__ == "__main__":
    main()
