# Autotune Project

Custom real-time autotune-style vocal plugin. Long-term target: VST3 loadable in Ableton Live (Intro / Standard / Suite).

Strategy: strong DSP engineering for pitch modification, modern AI/deep learning where it actually helps (mainly pitch detection at first, optionally a learned correction policy later). See `docs/architecture.md` for the full reasoning.

## Project Layout

```
autotune-project/
|-- python_prototype/
|   |-- autotune_offline.py     End-to-end CLI: WAV in -> tuned WAV out
|   |-- smoke_test_milestone1.py  Generates synthetic audio + runs the pipeline
|   |-- realtime_prototype.py   Mic-in/out streaming via sounddevice (M5)
|   |-- f0_extractors/          pYIN, torchcrepe, RMVPE behind a shared interface
|   |-- correction/             Scale quantizer, smoothing+hysteresis, vibrato
|   |-- pitch_shift/            basic OLA + Rubber Band wrappers
|   |-- evaluation/             latency_test, jitter_test, plot_f0
|   |-- training/               Causal CNN F0 trainer + ONNX export (M7)
|   `-- test_signals/           Synthetic vocal-like test audio generator
|
|-- juce_plugin/                JUCE VST3 audio effect (M6)
|   |-- CMakeLists.txt          FetchContent-based JUCE build
|   |-- Source/                 Plugin processor + DSP modules
|   `-- Builds/                 Build artefacts (gitignored)
|
`-- docs/                       Architecture, latency notes, research notes
```

## Quick Start

```bash
cd autotune-project
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Unix/Mac: source .venv/bin/activate
pip install -r requirements.txt

# Generate a synthetic vocal-like test signal
python python_prototype/test_signals/generate_test_signal.py

# Milestone 1 offline pipeline (single command)
python -m python_prototype.smoke_test_milestone1

# Equivalent explicit run
python -m python_prototype.autotune_offline \
    --input  python_prototype/test_signals/audio/test_vocal_synth.wav \
    --output python_prototype/test_signals/audio/test_vocal_tuned.wav \
    --key C --scale major --retune-speed 0.8 --vibrato 0.5
```

## Per-Milestone Commands

```bash
# M2 -- detector latency comparison (pYIN vs torchcrepe; RMVPE if installed)
python -m python_prototype.evaluation.latency_test \
    --input python_prototype/test_signals/audio/test_vocal_synth.wav

# M2 -- pitch accuracy / jitter / octave-jump comparison
python -m python_prototype.evaluation.jitter_test \
    --input       python_prototype/test_signals/audio/test_vocal_synth.wav \
    --groundtruth python_prototype/test_signals/audio/test_vocal_synth_groundtruth.csv

# M2 -- diagnostic plot
python -m python_prototype.evaluation.plot_f0 \
    --input       python_prototype/test_signals/audio/test_vocal_synth.wav \
    --groundtruth python_prototype/test_signals/audio/test_vocal_synth_groundtruth.csv \
    --output      python_prototype/evaluation/evaluation_results/plot_f0.png

# M4 -- offline run with Rubber Band shifter (needs rubberband-cli on PATH)
python -m python_prototype.autotune_offline \
    --input  ... --output ... --shifter rubberband --vibrato 0.6

# M5 -- live mic prototype (Ctrl-C to stop)
python -m python_prototype.realtime_prototype \
    --block-size 512 --retune-speed 0.8

# M6 -- build the VST3 plugin
cd juce_plugin && cmake -S . -B Builds && cmake --build Builds --config Release

# M7 -- train the causal F0 model on synthetic data
python -m python_prototype.training.train --steps 5000 --device auto
python -m python_prototype.training.export_onnx \
    --checkpoint python_prototype/training/checkpoints/causal_f0_best.pt \
    --output     python_prototype/training/checkpoints/causal_f0.onnx
```

## Status

- **Milestone 1 -- offline baseline**: pYIN F0 -> snap to scale -> retune smoothing -> overlap-add pitch shift -> WAV out. Rough by design.
- **Milestone 2 -- detector comparison**: pYIN / torchcrepe / RMVPE behind a shared interface + latency/jitter/plot harness.
- **Milestone 3 -- musical correction**: hysteresis (anti-flicker) + vibrato preservation wired through the offline pipeline.
- **Milestone 4 -- Rubber Band shifter**: pyrubberband-backed block-OLA shifter with optional formant preservation.
- **Milestone 5 -- real-time standalone**: `realtime_prototype.py` streams mic in/out via sounddevice and reports per-block latency.
- **Milestone 6 -- JUCE VST3 plugin**: CMake project with a working YIN detector, correction policy (scale + hysteresis + smoothing), resample-based pitch shifter, and a minimal UI bound to the APVTS.
- **Milestone 7 -- causal F0 model**: PyTorch causal CNN, synthetic-data training loop with bin-classification head, and ONNX export.

The lowest-quality parts are intentional baselines: the offline overlap-add shifter (replace with Rubber Band or PSOLA) and the plugin's resample-based shifter (swap for a Rubber Band live wrapper). Both are isolated behind interfaces so the swap is local.

## Hardware Notes

- Target GPU: RTX 2060 (sufficient for torchcrepe / RMVPE inference and for training the causal F0 model on synthetic data).
- Latency goal in the plugin: under ~10-20 ms end-to-end where achievable.
- Plugin must NOT depend on Max for Live (Ableton Intro/Standard compatibility).
