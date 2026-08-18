// ResamplePitchShifter.h
//
// Cheap real-time pitch shifter using resample + time-stretch via two
// linear-interpolation passes. Same approach as the Python realtime
// prototype: faster and lower-latency than a phase vocoder, sounds OK
// for moderate shifts (< ~3 semitones), audibly modulates beyond that.
//
// This is the M6 baseline. Swap in a JUCE wrapper around Rubber Band's
// live API (RubberBandStretcher) for production-quality shifting.

#pragma once

#include <JuceHeader.h>
#include <vector>
#include <cmath>

#include "PitchShifter.h"

class ResamplePitchShifter : public PitchShifter
{
public:
    void prepare (double sampleRate, int maxBlockSize) override
    {
        sampleRate_ = sampleRate;
        scratchA_.assign (maxBlockSize * 4, 0.0f);
        scratchB_.assign (maxBlockSize,     0.0f);
    }

    void process (const float* input,
                  float*       output,
                  int          numSamples,
                  float        cents,
                  bool         /*preserveFormants*/) override
    {
        if (! std::isfinite (cents) || std::abs (cents) < 1.0f)
        {
            for (int i = 0; i < numSamples; ++i) output[i] = input[i];
            return;
        }

        const double ratio = std::pow (2.0, (double) cents / 1200.0);
        if (ratio <= 0.0)
        {
            for (int i = 0; i < numSamples; ++i) output[i] = input[i];
            return;
        }

        const int n = numSamples;
        const int newN = juce::jmax (1, (int) std::round (n / ratio));

        if ((int) scratchA_.size() < newN) scratchA_.assign ((size_t) newN, 0.0f);

        // Pass 1: resample input -> length newN.
        for (int i = 0; i < newN; ++i)
        {
            const double pos = (double) i * (double) (n - 1) / (double) (newN - 1 > 0 ? newN - 1 : 1);
            const int    idx = juce::jlimit (0, n - 2, (int) pos);
            const double frac = pos - (double) idx;
            scratchA_[(size_t) i] = (float) ((1.0 - frac) * input[idx]
                                             +     frac  * input[idx + 1]);
        }

        // Pass 2: stretch length newN -> length n.
        for (int i = 0; i < n; ++i)
        {
            const double pos = (double) i * (double) (newN - 1) / (double) (n - 1 > 0 ? n - 1 : 1);
            const int    idx = juce::jlimit (0, newN - 2, (int) pos);
            const double frac = pos - (double) idx;
            output[i] = (float) ((1.0 - frac) * scratchA_[(size_t) idx]
                                 +     frac  * scratchA_[(size_t) (idx + 1)]);
        }
    }

    int getLatencySamples() const override { return 0; }

private:
    double             sampleRate_ = 44100.0;
    std::vector<float> scratchA_;
    std::vector<float> scratchB_;
};
