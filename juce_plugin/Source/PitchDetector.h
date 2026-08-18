// PitchDetector.h
//
// Abstract real-time F0 detector. Mirrors the Python F0Detector protocol so
// we can port detectors from the Python prototype with minimal interface
// drift.
//
// All implementations MUST be allocation-free per call once prepared.

#pragma once

#include <JuceHeader.h>

struct F0Frame
{
    double timeSeconds = 0.0;  // center of analysis frame
    float  f0Hz        = 0.0f; // 0.0 means unvoiced
    float  voicing     = 0.0f; // in [0, 1]
};

class PitchDetector
{
public:
    virtual ~PitchDetector() = default;

    // Allocate / load model weights / reset state. May allocate. Called from
    // prepareToPlay, not from the audio thread.
    virtual void prepare (double sampleRate, int hopSize, int frameSize) = 0;

    // Push one analysis hop worth of audio. Returns true and fills `out`
    // when a new F0 estimate is available; returns false otherwise.
    virtual bool pushHop (const float* hopAudio, int numSamples, F0Frame& out) = 0;

    // Latency reported to the host (samples). Detectors are free to be
    // honest about their internal lookahead.
    virtual int getLatencySamples() const = 0;
};

// Concrete implementations to add in Milestone 6:
//   class PyinPitchDetector       : public PitchDetector { ... };
//   class TorchcrepePitchDetector : public PitchDetector { ... };   // ONNX/TensorRT
//   class CausalNeuralPitchDetector : public PitchDetector { ... }; // Milestone 7
