# AI-Assisted Autotune Build Plan for Claude

## Project Goal

Build a custom real-time autotune-style vocal plugin that can run inside Ableton Live as a VST3 plugin. The system should use modern AI/deep learning where it is useful, but should rely on strong DSP engineering for the actual pitch modification.

The long-term goal is not only to make a working autotune, but to understand the full technical stack behind tools like Antares Auto-Tune:

```text
vocal audio input
→ pitch / voicing estimation
→ confidence + temporal smoothing
→ musical correction policy
→ formant-aware pitch shifting
→ low-latency VST3 output
```

The user has deep learning experience and owns an RTX 2060 GPU. They are not getting Ableton Suite, so Max for Live should not be the main path. The plugin should be compatible with Ableton Intro, Standard, and Suite through VST3.

---

## Core Insight

Do not treat this as only a pitch-detection problem.

A high-quality autotune system depends on four major pieces:

1. **Pitch detection / F0 estimation**
2. **Temporal stabilization and musical correction logic**
3. **Pitch shifting / vocal resynthesis DSP**
4. **Low-latency plugin engineering**

Pitch detection is very strong today, but not fully solved for real-time use. For autotune, the hard part is not only estimating pitch accurately. The system must estimate pitch with low latency, avoid jitter, preserve intentional vibrato/slides, handle consonants and unvoiced regions, and produce a musical correction trajectory.

---

## Recommended Development Roadmap

### Phase 0 — Define Scope

Start with a monophonic vocal autotune effect.

Initial constraints:

- Single vocal input
- Offline Python prototype first
- Then real-time C++/JUCE VST3
- Key/scale selection
- Retune speed knob
- Wet/dry mix
- No polyphonic correction
- No end-to-end neural audio generation at first

Avoid starting with a fully neural audio-to-audio model. That is a research project by itself and will make real-time performance much harder.

---

## Phase 1 — Offline Python Prototype

### Goal

Build the simplest possible autotune pipeline offline before touching plugin development.

### Pipeline

```text
input vocal.wav
→ estimate F0 over frames
→ convert F0 to MIDI note values
→ snap notes to selected scale
→ smooth correction trajectory
→ pitch shift audio
→ export tuned_vocal.wav
```

### Recommended Libraries

- `librosa` for initial audio loading and baseline pitch detection
- `soundfile` for export
- `numpy` for pitch/math operations
- `torchcrepe` for neural F0 estimation
- Optional: RMVPE for offline comparison
- Optional: Rubber Band CLI/library for better pitch shifting

### First Baseline

Use pYIN or torchcrepe for pitch detection, then use a basic pitch shifter.

Claude should help implement:

1. Load vocal file
2. Estimate framewise pitch
3. Detect voiced/unvoiced frames
4. Convert F0 to MIDI:

```text
midi = 69 + 12 * log2(f0 / 440)
```

5. Snap to nearest note in selected key
6. Smooth correction amount
7. Apply pitch shift
8. Export result

### Important Notes

The first version may sound rough. That is acceptable. The purpose is to verify the logic.

---

## Phase 2 — Pitch Detection Comparison

### Goal

Compare several pitch detectors for accuracy, stability, and latency.

### Methods to Compare

#### 1. YIN / pYIN

Classical DSP baseline.

Pros:

- Lightweight
- Easy to understand
- Real-time friendly

Cons:

- Jittery on noisy vocals
- Can struggle with breathy vocals, vibrato, and octave errors

#### 2. CREPE / torchcrepe

Deep learning pitch detection.

Pros:

- Strong accuracy
- Good practical choice
- torchcrepe is more modern and GPU-friendly than original CREPE

Cons:

- Still requires buffering
- Not automatically optimized for low-latency VST use

#### 3. RMVPE

Robust vocal pitch extractor.

Pros:

- Strong on vocals
- Stable F0 contours
- Better with noisy or mixed audio

Cons:

- Heavier
- Better as an offline/high-quality option first
- More complicated for low-latency plugin use

### Evaluation Metrics

Claude should help build scripts that measure:

- Inference time per audio frame/chunk
- Pitch jitter frame-to-frame
- Voiced/unvoiced errors
- Octave jumps
- Subjective correction quality
- End-to-end latency estimate

### Latency Experiment

Test chunk sizes:

```text
256 samples
512 samples
1024 samples
2048 samples
4096 samples
```

At 44.1 kHz:

```text
512 samples ≈ 11.6 ms
1024 samples ≈ 23.2 ms
2048 samples ≈ 46.4 ms
```

The final plugin should aim for low enough latency to feel usable while recording, ideally under roughly 10–20 ms if possible. This may require compromises.

---

## Phase 3 — Correction Policy

### Goal

Build the part that makes the system musical instead of just mathematically correct.

Naive correction:

```text
detected pitch → nearest scale note
```

Better correction:

```text
raw F0
→ confidence filtering
→ voiced/unvoiced masking
→ hysteresis
→ retune speed smoothing
→ vibrato preservation
→ target pitch trajectory
```

### Features to Implement

#### 1. Scale Quantization

User selects:

- Key: C, C#, D, etc.
- Scale: major, minor, chromatic, custom

The system maps detected pitch to nearest allowed scale degree.

#### 2. Retune Speed

Controls how fast the vocal moves toward the target pitch.

Fast retune:

- Hard autotune effect
- Robotic / modern rap effect

Slow retune:

- Natural correction
- Preserves more expression

Possible smoothing:

```text
corrected_pitch[t] = alpha * target_pitch[t] + (1 - alpha) * corrected_pitch[t-1]
```

Where alpha increases with retune speed.

#### 3. Hysteresis

Prevents the target note from flickering between two nearby notes.

Example:

- Do not switch target notes unless the detected pitch moves far enough toward the new note
- Add a minimum time before note switching

#### 4. Vibrato Preservation

Separate pitch into:

```text
slow pitch trend + fast vibrato residual
```

Then correct the slow trend while optionally preserving some fast vibrato.

#### 5. Voiced/Unvoiced Handling

Do not pitch-correct consonants, breaths, or noisy/unvoiced regions.

Use:

- pitch confidence
- RMS energy
- voicing probability

---

## Phase 4 — Pitch Shifting DSP

### Goal

Replace crude pitch shifting with a higher-quality vocal pitch shifter.

This is the part that actually changes the sound.

### Options

#### 1. Phase Vocoder

Pros:

- Conceptually common
- Works for general audio

Cons:

- Can smear transients
- Can sound metallic or phasey
- Not always best for vocals

#### 2. PSOLA

Pros:

- Strong for monophonic vocals
- Lower latency potential
- Preserves vocal character better when done well

Cons:

- Harder to implement correctly
- Requires reliable pitch period detection

#### 3. Rubber Band Library

Pros:

- High-quality pitch/time processing
- Useful for prototyping and possible integration

Cons:

- Licensing/integration must be checked
- May add latency

#### 4. SoundTouch

Pros:

- Easier to integrate
- Good for pitch/time modification

Cons:

- May not sound as polished as commercial vocal-specific tools

### Recommended Path

1. Start with library-based pitch shifting.
2. Use Rubber Band or SoundTouch for proof of concept.
3. Later investigate PSOLA for lower latency and better vocal-specific behavior.

---

## Phase 5 — Real-Time Prototype Outside Ableton

### Goal

Before building a VST3, make a standalone real-time test app.

Use Python or C++ to stream audio through the system.

Possible tools:

- Python: `sounddevice`, `pyaudio`
- C++: RtAudio, PortAudio, JUCE standalone app

### Test Pipeline

```text
microphone input
→ frame buffer
→ pitch detector
→ correction policy
→ pitch shifter
→ audio output
```

### What to Measure

- Total latency
- CPU/GPU load
- Dropouts
- Pitch stability
- Subjective feel while singing/rapping

---

## Phase 6 — JUCE VST3 Plugin

### Goal

Build a VST3 audio effect that Ableton can load.

### Framework

Use JUCE.

Create an audio plugin project:

```text
Plugin type: Audio Effect
Format: VST3
Optional: Standalone build for testing
```

### Plugin Parameters

Minimum controls:

- Key
- Scale
- Retune Speed
- Amount / Correction Strength
- Wet/Dry Mix
- Formant Preserve toggle or amount
- Input Gain
- Output Gain

Later controls:

- Humanize
- Vibrato Preserve
- Note Transition Time
- Detection Mode: Fast / High Quality
- F0 model selection

### Plugin Processing Structure

```text
processBlock(audioBuffer)
→ collect incoming samples into analysis buffer
→ run pitch detection at analysis hop size
→ update target pitch trajectory
→ apply pitch shifting to output buffer
→ return processed audio
```

### Important Real-Time Rules

Inside `processBlock()`:

- Do not allocate memory repeatedly
- Do not block on GPU calls if avoidable
- Avoid file I/O
- Avoid Python runtime
- Avoid heavy model loading
- Use lock-free queues if passing data between threads

---

## Phase 7 — AI Integration Strategy

### Option A — Practical First Version

Use AI only for pitch detection.

```text
neural F0 detector
→ rule-based correction policy
→ DSP pitch shifter
```

This is the recommended starting architecture.

### Option B — Learned Correction Policy

Train a model to predict a corrected target pitch trajectory.

Input:

```text
raw F0 contour
voicing/confidence
audio features
selected key/scale
```

Output:

```text
corrected F0 target
correction strength
vibrato preservation mask
```

This could be more novel than simply replacing the pitch detector.

### Option C — End-to-End Neural Pitch Correction

Input:

```text
off-key vocal + target pitch contour
```

Output:

```text
corrected vocal audio
```

This is ambitious and not recommended for the first version because it requires:

- paired or synthetically detuned data
- neural vocoder/resynthesis
- formant preservation
- very low-latency inference
- careful artifact control

---

## Suggested Research Directions

### Foundational Pitch Detection

- YIN: classical monophonic pitch tracking
- pYIN: probabilistic YIN with temporal smoothing
- CREPE: CNN-based pitch estimation from raw waveform
- torchcrepe: practical PyTorch implementation of CREPE-style pitch detection
- RMVPE: robust vocal pitch extraction, especially useful in voice conversion/vocal AI pipelines

### Pitch Modification

- Phase vocoder methods
- PSOLA / TD-PSOLA
- Formant preservation methods
- Time-scale modification literature

### Modern Research Direction

The most interesting research direction for this project is probably not simply “use a better pitch detector.” A better direction is:

```text
low-latency neural F0 + learned correction policy + high-quality DSP pitch shifting
```

Possible research contribution:

- Causal neural F0 estimator optimized for real-time autotune
- Learned pitch correction policy that preserves musical expression
- Hybrid DSP/neural autotune with explicit latency constraints

---

## Milestone Checklist

### Milestone 1 — Offline Baseline

- [ ] Load a vocal file
- [ ] Estimate F0
- [ ] Snap to key
- [ ] Apply pitch shift
- [ ] Export tuned file

### Milestone 2 — Better F0

- [ ] Add torchcrepe
- [ ] Add RMVPE offline comparison
- [ ] Compare jitter, latency, octave errors

### Milestone 3 — Musical Correction

- [ ] Add retune speed
- [ ] Add hysteresis
- [ ] Add voiced/unvoiced handling
- [ ] Add vibrato preservation

### Milestone 4 — Better Pitch Shifting

- [ ] Integrate Rubber Band or SoundTouch
- [ ] Compare with basic pitch shifting
- [ ] Investigate PSOLA

### Milestone 5 — Real-Time Standalone

- [ ] Stream microphone input
- [ ] Process in real time
- [ ] Measure latency
- [ ] Reduce dropouts

### Milestone 6 — JUCE Plugin

- [ ] Create JUCE VST3 audio effect
- [ ] Add UI controls
- [ ] Add real-time audio processing
- [ ] Load in Ableton
- [ ] Test on vocal track

### Milestone 7 — Advanced AI

- [ ] Train/finetune causal F0 model
- [ ] Export model to ONNX/TensorRT if needed
- [ ] Consider learned correction policy

---

## Suggested Architecture for First Serious Version

```text
Ableton audio input
→ JUCE input buffer
→ frame collector
→ torchcrepe or lightweight F0 model
→ confidence/voicing filter
→ scale quantizer
→ smoothing + hysteresis
→ pitch-shift amount curve
→ Rubber Band / SoundTouch / PSOLA pitch shifter
→ wet/dry mix
→ Ableton output
```

---

## What Makes Antares Good: Lessons to Copy

Antares is good because it is not just a pitch detector.

Important qualities to reproduce:

1. Stable pitch tracking over time
2. Very polished pitch shifting DSP
3. Low latency
4. Musical correction behavior
5. Edge-case handling for real vocals

The most important lesson:

```text
Autotune quality = temporal behavior + DSP quality + musical heuristics + latency engineering
```

Not just neural pitch accuracy.

---

## What Claude Should Help With First

Claude should start by helping implement the offline prototype.

First task:

> Create a Python script that loads a monophonic vocal WAV file, estimates F0 with torchcrepe, snaps voiced frames to a selected musical scale, smooths the correction trajectory with retune speed, and exports diagnostic plots/data. Do not worry about perfect pitch shifting yet.

Second task:

> Add a simple pitch shifting stage and export a tuned WAV file.

Third task:

> Build an evaluation script that compares pYIN, torchcrepe, and RMVPE on latency, pitch jitter, and octave jumps.

Fourth task:

> Convert the validated pipeline into a JUCE VST3 architecture plan.

---

## Recommended Initial File Structure

```text
autotune-project/
├── python_prototype/
│   ├── autotune_offline.py
│   ├── f0_extractors/
│   │   ├── pyin_detector.py
│   │   ├── torchcrepe_detector.py
│   │   └── rmvpe_detector.py
│   ├── correction/
│   │   ├── scale_quantizer.py
│   │   ├── smoothing.py
│   │   └── vibrato.py
│   ├── pitch_shift/
│   │   ├── basic_shift.py
│   │   └── rubberband_shift.py
│   └── evaluation/
│       ├── latency_test.py
│       ├── jitter_test.py
│       └── plot_f0.py
│
├── juce_plugin/
│   ├── Source/
│   │   ├── PluginProcessor.cpp
│   │   ├── PluginProcessor.h
│   │   ├── PluginEditor.cpp
│   │   ├── PluginEditor.h
│   │   ├── PitchDetector.h
│   │   ├── CorrectionPolicy.h
│   │   └── PitchShifter.h
│   └── Builds/
│
└── docs/
    ├── architecture.md
    ├── latency_notes.md
    └── research_notes.md
```

---

## Final Recommendation

Build the project in this order:

```text
Python offline prototype
→ detector comparison
→ correction policy
→ high-quality pitch shifting
→ real-time standalone
→ JUCE VST3
→ custom causal AI model
```

The strongest first version is:

```text
torchcrepe + musical smoothing/hysteresis + Rubber Band/SoundTouch + JUCE VST3
```

The most interesting advanced version is:

```text
custom causal F0 estimator + learned correction policy + PSOLA/formant-preserving pitch shifter
```
