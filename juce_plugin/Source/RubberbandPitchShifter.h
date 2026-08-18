// RubberbandPitchShifter.h
//
// Real-time pitch shifter wrapping RubberBand::RubberBandStretcher (v3).
//
// Design notes:
//   - Single mono stretcher instance.  PluginProcessor feeds channel 0 and
//     reuses the wet buffer for any additional channels.
//   - OptionProcessRealTime + OptionPitchHighQuality + OptionThreadingNever:
//     no background threads, high-quality R3 pitch engine.
//   - Output is drained into a pre-allocated circular ring buffer each block.
//     During startup (until latency pre-roll fills), dry signal passes through.
//   - Formant toggle recreates the stretcher (rare event, fine to do).
//   - No heap allocation in the steady-state process() path.

#pragma once

#include <JuceHeader.h>
#include <rubberband/RubberBandStretcher.h>
#include <vector>
#include <cmath>
#include <memory>

#include "PitchShifter.h"

class RubberbandPitchShifter : public PitchShifter
{
public:
    // -----------------------------------------------------------------------
    void prepare (double sampleRate, int maxBlockSize) override
    {
        sampleRate_   = sampleRate;
        maxBlockSize_ = maxBlockSize;

        // Temp retrieve buffer: Rubber Band may occasionally return slightly
        // more than maxBlockSize samples (latency padding at startup).
        retrieveBuf_.assign ((size_t) maxBlockSize * 8, 0.0f);

        // Ring buffer: sized to hold ~2 seconds worth; in steady state it will
        // hold only ~latency samples (~512–4096 at 44 kHz).
        const int ringCap = maxBlockSize * 128;
        ringBuf_.assign ((size_t) ringCap, 0.0f);
        ringHead_ = ringTail_ = ringCount_ = 0;
        ringCap_  = ringCap;

        createStretcher (false);
    }

    // -----------------------------------------------------------------------
    void process (const float* input,
                  float*       output,
                  int          numSamples,
                  float        cents,
                  bool         preserveFormants) override
    {
        // Recreate on formant toggle (rare — no need to be cheap).
        if (preserveFormants != lastFormant_)
        {
            createStretcher (preserveFormants);
            lastFormant_ = preserveFormants;
        }

        // Pass-through for negligible or invalid shift.
        if (! std::isfinite (cents) || std::abs (cents) < 0.5f)
        {
            for (int i = 0; i < numSamples; ++i) output[i] = input[i];
            return;
        }

        const double ratio = std::pow (2.0, (double) cents / 1200.0);
        stretcher_->setPitchScale (ratio);

        // Feed input to the stretcher.
        const float* inPtrs[1] = { input };
        stretcher_->process (inPtrs, (size_t) numSamples, false);

        // Drain all available output into ring buffer.
        int avail = (int) stretcher_->available();
        if (avail < 0) avail = 0;

        if (avail > 0)
        {
            // Grow retrieve buffer only if absolutely necessary (shouldn't happen).
            if (avail > (int) retrieveBuf_.size())
                retrieveBuf_.assign ((size_t) avail, 0.0f);

            float* outPtrs[1] = { retrieveBuf_.data() };
            stretcher_->retrieve (outPtrs, (size_t) avail);

            // Push into ring buffer — drop oldest if overflow (shouldn't happen).
            for (int i = 0; i < avail; ++i)
            {
                if (ringCount_ < ringCap_)
                {
                    ringBuf_[(size_t) ringTail_] = retrieveBuf_[(size_t) i];
                    ringTail_ = (ringTail_ + 1) % ringCap_;
                    ++ringCount_;
                }
            }
        }

        // If ring buffer is filled past startup latency, copy to output.
        if (ringCount_ >= numSamples)
        {
            for (int i = 0; i < numSamples; ++i)
            {
                output[i] = ringBuf_[(size_t) ringHead_];
                ringHead_ = (ringHead_ + 1) % ringCap_;
                --ringCount_;
            }
        }
        else
        {
            // Startup: not enough output yet — pass through dry signal.
            for (int i = 0; i < numSamples; ++i) output[i] = input[i];
        }
    }

    // -----------------------------------------------------------------------
    int getLatencySamples() const override
    {
        return stretcher_ ? (int) stretcher_->getLatency() : 0;
    }

private:
    // -----------------------------------------------------------------------
    void createStretcher (bool preserveFormants)
    {
        using RB = RubberBand::RubberBandStretcher;

        RB::Options opts = RB::OptionProcessRealTime
                         | RB::OptionPitchHighQuality
                         | RB::OptionThreadingNever;

        if (preserveFormants)
            opts |= RB::OptionFormantPreserved;

        stretcher_ = std::make_unique<RB> (
            (size_t) sampleRate_,
            1u,          // mono — processor mixes to mono before calling us
            opts,
            1.0,         // timeRatio: no time-stretch
            1.0);        // pitchScale: updated per-block via setPitchScale()

        // Reset ring buffer state so dry pass-through re-triggers on warm-up.
        ringHead_ = ringTail_ = ringCount_ = 0;
    }

    // -----------------------------------------------------------------------
    double sampleRate_   = 44100.0;
    int    maxBlockSize_ = 512;
    bool   lastFormant_  = false;

    std::unique_ptr<RubberBand::RubberBandStretcher> stretcher_;

    std::vector<float> retrieveBuf_;   // temp: holds one retrieve() result

    // Pre-allocated circular ring buffer for stretcher output.
    std::vector<float> ringBuf_;
    int ringHead_ = 0;
    int ringTail_ = 0;
    int ringCount_ = 0;
    int ringCap_   = 0;
};
