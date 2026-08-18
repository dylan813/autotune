# JUCE VST3 Plugin (Phase 6)

This directory holds the JUCE audio-effect plugin. It is intentionally a
scaffold: the source files declare the structure the plugin will take, but
none of the DSP is wired up yet. Most of that work should not start until
the Python prototype (Milestones 1-5) is producing music we like.

## Getting Started

1. Install JUCE: https://juce.com/get-juce (use the framework, not Projucer-only).
2. Create a new "Audio Plug-In" project (preferred: CMake template) targeting
   **VST3** and (optionally) **Standalone** for headless testing.
3. Add the files in `Source/` to the project.
4. Set the plugin name to `Autotune` (or whatever you like) and the
   manufacturer/code so Ableton scans it cleanly.

## Folder Structure

```
juce_plugin/
|-- Source/
|   |-- PluginProcessor.h / .cpp   AudioProcessor: parameter tree, prepareToPlay, processBlock
|   |-- PluginEditor.h   / .cpp    AudioProcessorEditor: minimal UI for the parameters
|   |-- PitchDetector.h            Causal F0 detector interface (ports the Python interface)
|   |-- CorrectionPolicy.h         Scale snap + smoothing + hysteresis + vibrato preserve
|   `-- PitchShifter.h             Pitch shifter interface (Rubber Band / PSOLA backends)
`-- Builds/                        IDE / CMake output (gitignored)
```

## Real-Time Rules (do not break these inside `processBlock`)

- No allocations.
- No locks. Use lock-free FIFOs for any cross-thread data passing.
- No file I/O.
- No Python runtime.
- No blocking GPU calls if avoidable. If torchcrepe-style inference is
  needed, run it on a worker thread with a lookahead buffer; the main audio
  thread reads the latest target trajectory through a lock-free queue.

## Latency Budget

Aim for ~10-20 ms end-to-end including the host's own buffer. Document the
breakdown in `../docs/latency_notes.md` as we measure.

## Ableton Notes

- Target VST3 only (LV2 / AU are not needed for the stated goal).
- The plugin must work in Ableton **Intro** and **Standard**, so it cannot
  depend on Max for Live.
- Verify scanning from Ableton's Preferences -> Plug-Ins.
