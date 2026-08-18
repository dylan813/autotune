# Latency Notes

Target: total end-to-end latency under ~10-20 ms while feeling musical.

## Sample-size cheat sheet (at 44.1 kHz)

| Samples | ms     |
|---------|--------|
| 256     | 5.8    |
| 512     | 11.6   |
| 1024    | 23.2   |
| 2048    | 46.4   |
| 4096    | 92.9   |

## Sources of latency to budget

- Host buffer (Ableton): determined by user's audio interface, often 64-256
  samples on a competent interface.
- Detector frame size: most pitch detectors need a full pitch period. At
  the lowest expected vocal F0 of ~80 Hz, one period is ~12.5 ms.
- Detector model lookahead (CREPE / RMVPE windows).
- Pitch-shifter algorithmic delay (Rubber Band > PSOLA usually).
- Worker-thread crossing (FIFO + scheduling) if we offload the detector.

## Measurement protocol

1. **Detector cost (CPU/GPU)**: `python_prototype/evaluation/latency_test.py`
   reports per-frame wall-clock at each candidate chunk size.
2. **Detector intrinsic delay**: count the number of input samples needed
   before the *first* valid F0 is produced.
3. **Shifter delay**: Rubber Band reports its own; PSOLA depends on the
   analysis window.
4. **End-to-end (real-time prototype)**: loopback a click track through
   `sounddevice`, measure click-to-click delay in the captured output.
5. **In-plugin**: use a JUCE `AudioPluginHost` with a loopback file player
   and a click stimulus; align input vs output with cross-correlation.

## Likely-best initial settings (to validate)

- Analysis hop: 256 samples.
- Analysis frame: 1024 samples (gives ~23 ms window for pitch detection).
- Detector: torchcrepe `tiny` running on GPU; fall back to pYIN if GPU
  unavailable.
- Shifter: Rubber Band R3, formant-preserving on.
- Worker-thread detection with a 1-hop FIFO lookahead.

Replace these once we have real numbers from Milestone 2 and Milestone 5.
