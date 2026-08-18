// PitchShifter.h
//
// Abstract pitch shifter. Implementations:
//   - RubberBandPitchShifter  (Milestone 6 default, links to Rubber Band)
//   - PsolaPitchShifter       (Milestone 7, custom, lower latency)
//
// All implementations MUST be allocation-free in process().

#pragma once

#include <JuceHeader.h>

class PitchShifter
{
public:
    virtual ~PitchShifter() = default;

    // Allocate buffers / configure backend. Called from prepareToPlay.
    virtual void prepare (double sampleRate, int maxBlockSize) = 0;

    // Shift `input` by `cents` and write to `output`. Both buffers are
    // `numSamples` long. `cents` is a single value for the block; ramps
    // happen at the analysis-hop granularity above this layer.
    virtual void process (const float* input,
                          float*       output,
                          int          numSamples,
                          float        cents,
                          bool         preserveFormants) = 0;

    virtual int getLatencySamples() const = 0;
};
