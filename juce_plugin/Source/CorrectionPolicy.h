// CorrectionPolicy.h
//
// Maps detected F0 -> target F0 using scale, retune speed, hysteresis, and
// optional vibrato preservation. Mirrors python_prototype/correction/.

#pragma once

#include <JuceHeader.h>
#include <cmath>

enum class Scale { Major, Minor, Chromatic };

struct CorrectionConfig
{
    int    keyPc          = 0;      // 0 = C, 1 = C#/Db, ...
    Scale  scale          = Scale::Major;
    float  retuneSpeed    = 0.5f;   // [0, 1]
    float  amount         = 1.0f;   // [0, 1] correction strength
    int    minDwellHops   = 3;
    float  hysteresisCents = 40.0f;
};

class CorrectionPolicy
{
public:
    void prepare (double sampleRate, int hopSize)
    {
        sampleRate_ = sampleRate;
        hopSize_    = hopSize;
        frameRateHz_ = (float) (sampleRate / hopSize);
        reset();
    }

    void setConfig (const CorrectionConfig& c)
    {
        config_ = c;
        updateAlpha();
    }

    void reset()
    {
        smoothedTargetHz_ = 0.0f;
        heldTargetHz_     = 0.0f;
        candidateHz_      = 0.0f;
        candidateCount_   = 0;
        unvoicedCount_    = 0;
        updateAlpha();
    }

    // Convert a detected F0 (Hz) + voicing into a desired output F0 (Hz).
    // Returns 0 when the frame is unvoiced or when correction is disabled.
    float updateAndGetTargetHz (float detectedHz, float voicing)
    {
        if (!std::isfinite (detectedHz) || detectedHz <= 0.0f || voicing < 0.25f)
        {
            // Hold note-locking state through brief unvoiced gaps (consonants,
            // breath noise, etc.) — only fully reset after kResetHops consecutive
            // unvoiced frames (~120 ms), which indicates a genuine phrase break.
            if (++unvoicedCount_ >= kResetHops)
            {
                heldTargetHz_     = 0.0f;
                candidateHz_      = 0.0f;
                candidateCount_   = 0;
                smoothedTargetHz_ = 0.0f;
            }
            return 0.0f;
        }
        unvoicedCount_ = 0;

        // 1. Snap to scale.
        const float snapped = snapToScaleHz (detectedHz);

        // 2. Hysteresis.
        float held = applyHysteresis (snapped);

        // 3. Retune-speed smoothing.
        if (smoothedTargetHz_ <= 0.0f)
            smoothedTargetHz_ = held;
        else
            smoothedTargetHz_ = alpha_ * held + (1.0f - alpha_) * smoothedTargetHz_;

        // 4. Correction amount: blend toward detected.
        const float corrected =
            config_.amount * smoothedTargetHz_ + (1.0f - config_.amount) * detectedHz;
        return corrected;
    }

private:
    static int scaleSize (Scale s)
    {
        switch (s) {
            case Scale::Major:     return 7;
            case Scale::Minor:     return 7;
            case Scale::Chromatic: return 12;
        }
        return 7;
    }

    static const int* scaleIntervals (Scale s)
    {
        static constexpr int major[]     = { 0, 2, 4, 5, 7, 9, 11 };
        static constexpr int minor[]     = { 0, 2, 3, 5, 7, 8, 10 };
        static constexpr int chromatic[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        switch (s) {
            case Scale::Major:     return major;
            case Scale::Minor:     return minor;
            case Scale::Chromatic: return chromatic;
        }
        return major;
    }

    float snapToScaleHz (float hz) const
    {
        const double midi = 69.0 + 12.0 * std::log2 ((double) hz / 440.0);
        const int    baseOct = (int) std::floor (midi / 12.0);
        const int    n = scaleSize (config_.scale);
        const int*   iv = scaleIntervals (config_.scale);

        double bestErr = 1e9;
        double bestMidi = midi;
        for (int oct = baseOct - 1; oct <= baseOct + 1; ++oct)
        {
            for (int k = 0; k < n; ++k)
            {
                const int cand = oct * 12 + ((config_.keyPc + iv[k]) % 12);
                const double err = std::abs ((double) cand - midi);
                if (err < bestErr) { bestErr = err; bestMidi = (double) cand; }
            }
        }
        return (float) (440.0 * std::pow (2.0, (bestMidi - 69.0) / 12.0));
    }

    float applyHysteresis (float candidateNewHz)
    {
        if (heldTargetHz_ <= 0.0f)
        {
            heldTargetHz_ = candidateNewHz;
            candidateHz_  = candidateNewHz;
            candidateCount_ = 1;
            return heldTargetHz_;
        }

        const float distCents = 1200.0f * std::log2 (candidateNewHz / heldTargetHz_);
        if (std::abs (distCents) < 1.0f)
        {
            candidateHz_ = heldTargetHz_;
            candidateCount_ = 0;
            return heldTargetHz_;
        }

        if (candidateHz_ > 0.0f &&
            std::abs (1200.0f * std::log2 (candidateNewHz / candidateHz_)) < 1.0f)
        {
            ++candidateCount_;
        }
        else
        {
            candidateHz_   = candidateNewHz;
            candidateCount_ = 1;
        }

        if (candidateCount_ >= config_.minDwellHops &&
            std::abs (1200.0f * std::log2 (candidateHz_ / heldTargetHz_))
                >= config_.hysteresisCents)
        {
            heldTargetHz_   = candidateHz_;
            candidateCount_ = 0;
        }
        return heldTargetHz_;
    }

    void updateAlpha()
    {
        const float s = juce::jlimit (0.0f, 1.0f, config_.retuneSpeed);
        if (s >= 0.999f) { alpha_ = 1.0f; return; }
        // tau_ms: log-interpolate from 250 ms (s=0) to 5 ms (s=1).
        const float tauMs = std::exp (std::log (5.0f) * s +
                                       std::log (250.0f) * (1.0f - s));
        const float tauSec = tauMs * 0.001f;
        alpha_ = 1.0f - std::exp (-1.0f / std::max (tauSec * frameRateHz_, 1e-6f));
    }

    double  sampleRate_  = 44100.0;
    int     hopSize_     = 256;
    float   frameRateHz_ = 44100.0f / 256.0f;
    float   alpha_       = 1.0f;

    CorrectionConfig config_;
    float heldTargetHz_     = 0.0f;
    float candidateHz_      = 0.0f;
    int   candidateCount_   = 0;
    float smoothedTargetHz_ = 0.0f;

    // Consecutive unvoiced-frame counter for deferred state reset.
    int unvoicedCount_ = 0;
    static constexpr int kResetHops = 20;  // ~120 ms before full state wipe
};
