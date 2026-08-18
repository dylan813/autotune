// YinPitchDetector.h
//
// Minimal real-time YIN pitch detector.
//
// Reference: de Cheveigne, Kawahara 2002 -- "YIN, a fundamental frequency
// estimator for speech and music."
//
// Implements steps 1-4 of YIN (autocorrelation -> difference function
// -> cumulative mean normalised difference -> absolute threshold pick),
// followed by parabolic interpolation around the chosen lag. Skips the
// "best local estimate" step from the paper; for autotune the cumulative
// thresholding gets us close enough and is cheap.
//
// Buffers a single analysis frame (frameSize_) and emits one F0 estimate
// per `pushHop` call once enough audio has been seen.

#pragma once

#include <JuceHeader.h>
#include <vector>
#include <cmath>

#include "PitchDetector.h"

class YinPitchDetector : public PitchDetector
{
public:
    void prepare (double sampleRate, int hopSize, int frameSize) override
    {
        sampleRate_  = sampleRate;
        hopSize_     = hopSize;
        frameSize_   = frameSize;
        ringBuffer_.assign (frameSize_, 0.0f);
        writePos_    = 0;
        samplesSinceLastEstimate_ = 0;
        diff_.assign (frameSize_ / 2, 0.0);
        cmnd_.assign (frameSize_ / 2, 0.0);
    }

    int getLatencySamples() const override { return frameSize_; }

    bool pushHop (const float* hopAudio, int numSamples, F0Frame& out) override
    {
        for (int i = 0; i < numSamples; ++i)
        {
            ringBuffer_[writePos_] = hopAudio[i];
            writePos_ = (writePos_ + 1) % frameSize_;
        }
        samplesSinceLastEstimate_ += numSamples;
        if (samplesSinceLastEstimate_ < hopSize_)
            return false;
        samplesSinceLastEstimate_ = 0;

        // Copy ring buffer into a linear analysis buffer.
        static thread_local std::vector<float> linear;
        if ((int) linear.size() != frameSize_) linear.assign (frameSize_, 0.0f);
        for (int i = 0; i < frameSize_; ++i)
            linear[i] = ringBuffer_[(writePos_ + i) % frameSize_];

        const int W = frameSize_ / 2;

        // YIN step 2: difference function.
        for (int tau = 0; tau < W; ++tau)
        {
            double s = 0.0;
            for (int j = 0; j < W; ++j)
            {
                const double d = (double) linear[j] - (double) linear[j + tau];
                s += d * d;
            }
            diff_[tau] = s;
        }

        // YIN step 3: cumulative mean normalised difference.
        cmnd_[0] = 1.0;
        double runningSum = 0.0;
        for (int tau = 1; tau < W; ++tau)
        {
            runningSum += diff_[tau];
            cmnd_[tau] = (runningSum > 1e-12) ? diff_[tau] * tau / runningSum : 1.0;
        }

        // Step 4: absolute threshold. Pick the first local minimum below
        // `threshold_`, otherwise the global minimum.
        constexpr double threshold = 0.20;
        int tauEstimate = -1;
        int minTau = (int) std::round (sampleRate_ / 600.0);  // 600 Hz ceiling (vocal range)
        int maxTau = (int) std::round (sampleRate_ / 80.0);   // 80 Hz floor  (vocal range)
        minTau = juce::jmax (2, minTau);
        maxTau = juce::jmin (W - 1, maxTau);

        for (int tau = minTau; tau < maxTau; ++tau)
        {
            if (cmnd_[tau] < threshold)
            {
                while (tau + 1 < maxTau && cmnd_[tau + 1] < cmnd_[tau])
                    ++tau;
                tauEstimate = tau;
                break;
            }
        }
        if (tauEstimate < 0)
        {
            // Fall back to argmin.
            int best = minTau;
            for (int tau = minTau + 1; tau < maxTau; ++tau)
                if (cmnd_[tau] < cmnd_[best])
                    best = tau;
            tauEstimate = best;
        }

        // Parabolic interpolation for sub-sample accuracy.
        const int t  = tauEstimate;
        double betterTau = (double) t;
        if (t > 0 && t < W - 1)
        {
            const double s0 = cmnd_[t - 1];
            const double s1 = cmnd_[t];
            const double s2 = cmnd_[t + 1];
            const double denom = (2.0 * (2.0 * s1 - s2 - s0));
            if (std::abs (denom) > 1e-12)
                betterTau = t + (s2 - s0) / denom;
        }

        const double f0Hz = (betterTau > 0.0) ? sampleRate_ / betterTau : 0.0;
        const float  voicing = (cmnd_[tauEstimate] < threshold)
                             ? (float) (1.0 - cmnd_[tauEstimate])
                             : (float) std::max (0.0, 1.0 - cmnd_[tauEstimate]);

        out.f0Hz   = (f0Hz >= 80.0 && f0Hz <= 600.0) ? (float) f0Hz : 0.0f;
        out.voicing = voicing;
        out.timeSeconds = 0.0;  // caller stamps this if it cares
        return true;
    }

private:
    double sampleRate_ = 44100.0;
    int    hopSize_    = 256;
    int    frameSize_  = 1024;

    std::vector<float>  ringBuffer_;
    int                 writePos_ = 0;
    int                 samplesSinceLastEstimate_ = 0;

    std::vector<double> diff_;
    std::vector<double> cmnd_;
};
