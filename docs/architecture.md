# Architecture

High-level signal flow (Phase 6 target):

```
Ableton audio input
  -> JUCE input buffer
  -> frame collector (ring buffer at analysis hop)
  -> F0 detector (torchcrepe / RMVPE / pYIN behind a common interface)
  -> voicing/confidence filter
  -> scale quantizer
  -> smoothing + hysteresis
  -> pitch-shift-amount curve (cents per analysis hop)
  -> Rubber Band / SoundTouch / PSOLA pitch shifter
  -> wet/dry mix
  -> Ableton output
```

## Module Boundaries

The Python prototype and the C++ plugin intentionally share the same module
boundaries so that work in one ports cleanly to the other.

| Concern                | Python                                       | C++ (JUCE)                  |
|------------------------|----------------------------------------------|-----------------------------|
| F0 estimation          | `f0_extractors/` (PyinDetector, ...)         | `PitchDetector.h` interface |
| Scale / smoothing      | `correction/`                                | `CorrectionPolicy.h`        |
| Pitch shifting         | `pitch_shift/`                               | `PitchShifter.h` interface  |
| Pipeline orchestration | `autotune_offline.py`                        | `PluginProcessor.cpp`       |

## Why This Split

Autotune quality is **not** just pitch-detection accuracy. The four-way
split (detection / temporal stabilization / shifting / latency engineering)
mirrors the most important factors observed in commercial tools. Each box
above is independently swappable, which lets us:

- Replace the F0 detector without re-validating the rest of the pipeline.
- A/B Rubber Band vs PSOLA without touching correction logic.
- Reuse the correction policy as a learned model later (Phase 7, Option B)
  by replacing the implementation behind the same interface.

## Threading Model (Plugin)

`processBlock` runs on the audio thread and must be allocation-free and
lock-free. The expected layout once detector latency forces it:

```
[Audio thread]                            [Worker thread]
  push hop -> SPSC FIFO  ----------------> pop hop
                                            run detector
  pop latest F0  <- SPSC FIFO  ---------- push (timeSeconds, f0, voicing)
  CorrectionPolicy.updateAndGetTargetHz
  PitchShifter.process
```

If the detector is fast enough (e.g. pYIN at small hops), it can run
inline and the worker thread can be skipped.

## Open Questions

- Which detector wins the latency/accuracy tradeoff on the RTX 2060?
  Run `python -m python_prototype.evaluation.latency_test` and
  `python_prototype.evaluation.jitter_test`. The plot script overlays
  detected vs ground-truth contours per detector.
- Is Rubber Band's licensing acceptable for the intended distribution?
  Re-check before shipping the plugin (the offline path uses it
  unconditionally; the plugin will need either the commercial license or
  to fall back to ResamplePitchShifter / PSOLA).
- PSOLA vs Rubber Band for the live plugin: Rubber Band live is the
  default if licensing is settled; otherwise port the offline OLA shifter
  or implement TD-PSOLA.
- Whether the M7 causal F0 net beats a quantized torchcrepe tiny on the
  same RTX 2060 in streaming mode. Both can be exported to ONNX and
  benchmarked from the same harness.
