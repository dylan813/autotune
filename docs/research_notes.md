# Research Notes

A scratchpad for references, decisions, and experiment outcomes. Cite
specific commits / paper versions where it matters.

## Pitch Detection

- **YIN** (de Cheveigne, Kawahara 2002): classical autocorrelation-based.
- **pYIN** (Mauch, Dixon 2014): probabilistic YIN with HMM smoothing.
  Available via `librosa.pyin`.
- **CREPE** (Kim et al. 2018): CNN over raw waveform, framewise pitch
  classification. Strong baseline.
- **torchcrepe** (Maxwell Morrison): PyTorch port with weight management.
  Default detector for the prototype.
- **RMVPE** (Wei et al. 2023): designed for noisy / mixed-source vocals.
  Heavier; offline first, real-time later.

## Pitch / Time Modification

- Phase vocoder: classic, but transient smearing and phasiness on vocals.
- TD-PSOLA (Moulines, Charpentier 1990): grain-based, very strong on
  monophonic voice when pitch marks are reliable.
- Rubber Band Library (Cannam): general-purpose, R3 engine, optional
  formant preservation. Check license terms before redistributing the VST.
- SoundTouch (Parviainen): MIT, lighter, generally lower quality than
  Rubber Band on vocals but easier to embed.

## Formant Preservation

- Spectral envelope decoupling (LP or cepstral lifter) before pitch shift
  and re-application after.
- Rubber Band's `--formant` option is a good starting target.

## Hybrid / Modern Directions

- Causal neural F0 (e.g. distilled CREPE with streaming inference).
- Learned correction policy that maps (raw F0, voicing, audio features,
  key/scale) -> target F0 trajectory. Could be trained on detuned/correct
  pairs synthesized from pristine vocal data.
- End-to-end neural pitch correction (vocal-to-vocal). Expensive, requires
  paired data, deferred to research phase.

## Tooling Decisions (to revisit)

| Decision                       | Current pick           | Rationale                                        |
|--------------------------------|------------------------|--------------------------------------------------|
| Default detector (prototype)   | torchcrepe tiny        | Speed-quality balance on RTX 2060                |
| Default shifter (prototype)    | Rubber Band            | High quality, easy via pyrubberband              |
| Plugin framework               | JUCE (VST3)            | Compatible with Ableton Intro/Standard/Suite     |
| Real-time prototype audio I/O  | sounddevice / PortAudio| Cross-platform, easy to wire to NumPy pipeline   |
| Build system (plugin)          | JUCE + CMake           | Reproducible, IDE-agnostic                       |

## Experiment Log

Add entries as `YYYY-MM-DD short title -> outcome` so the history is grep-able.

- 2026-05-16 scaffold + M1-M7 baselines committed -> end-to-end pipeline works on
  the synthetic test signal; plugin compiles via CMake/FetchContent; causal F0
  model trains on synthetic data; all module interfaces in place for targeted
  improvements (Rubber Band live in the plugin; PSOLA shifter; richer eval set).
