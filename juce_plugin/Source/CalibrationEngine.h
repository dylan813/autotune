// CalibrationEngine.h
//
// Analyses a short window of live singing and computes personalised plugin
// parameters for that specific voice.
//
// Data collected (one sample per YIN hop while calibrating):
//   f0Hz    — detected fundamental (0 when unvoiced)
//   voicing — YIN voicing confidence [0,1]
//   rms     — smoothed envelope RMS
//
// Parameters derived:
//   retuneSpeed     [0,1]  — slow for accurate singers, fast for less accurate
//   gateThreshold   float  — set to 20 % of the singer's median voiced RMS
//                            so quiet passages still trigger correction
//
// Thread safety:
//   start() and pushFrame() are called from the AUDIO thread only.
//   getState() / getProgress() / getResult() are safe to poll from any thread
//   because they read atomics / already-complete data.

#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <atomic>

struct CalibrationResult
{
    float retuneSpeed      = 0.50f;  // APVTS "retuneSpeed" parameter value
    float gateThreshold    = 0.02f;  // internal envelopeRms_ gate
    bool  valid            = false;
};

class CalibrationEngine
{
public:
    enum class State { Idle, Calibrating, Complete };

    // -----------------------------------------------------------------------
    // Called from audio thread.
    // -----------------------------------------------------------------------

    void start (int durationHops)
    {
        targetHops_ = durationHops;
        collected_  = 0;
        f0_.clear();      f0_.reserve      ((size_t) durationHops);
        voicing_.clear(); voicing_.reserve ((size_t) durationHops);
        rms_.clear();     rms_.reserve     ((size_t) durationHops);
        result_ = {};
        state_.store ((int) State::Calibrating);
    }

    // Push one frame; call once per YIN hop while calibrating.
    void pushFrame (float f0Hz, float voicing, float rms)
    {
        if (getState() != State::Calibrating) return;

        f0_.push_back      (f0Hz);
        voicing_.push_back (voicing);
        rms_.push_back     (rms);
        ++collected_;

        if (collected_ >= targetHops_)
        {
            result_ = analyze();
            state_.store ((int) State::Complete);
        }
    }

    // -----------------------------------------------------------------------
    // Safe to call from any thread.
    // -----------------------------------------------------------------------

    State getState() const { return (State) state_.load(); }

    float getProgress() const
    {
        switch (getState())
        {
            case State::Idle:       return 0.0f;
            case State::Complete:   return 1.0f;
            case State::Calibrating:
                return (targetHops_ > 0)
                     ? (float) collected_ / (float) targetHops_
                     : 0.0f;
        }
        return 0.0f;
    }

    const CalibrationResult& getResult() const { return result_; }

    void reset()
    {
        state_.store ((int) State::Idle);
        collected_ = 0;
        result_ = {};
    }

private:
    // -----------------------------------------------------------------------
    CalibrationResult analyze() const
    {
        CalibrationResult r;
        r.valid = false;

        // -------------------------------------------------------------------
        // Separate voiced from unvoiced frames.
        // -------------------------------------------------------------------
        std::vector<float> voicedF0, voicedRms;
        voicedF0.reserve  (f0_.size());
        voicedRms.reserve (f0_.size());

        for (size_t i = 0; i < f0_.size(); ++i)
        {
            if (voicing_[i] > 0.25f && f0_[i] > 60.0f)
            {
                voicedF0.push_back  (f0_[i]);
                voicedRms.push_back (rms_[i]);
            }
        }

        // Need at least one second of voiced audio to calibrate reliably.
        if ((int) voicedF0.size() < 20) return r;

        // -------------------------------------------------------------------
        // Gate threshold
        // Set to 20 % of the median voiced RMS so soft consonants pass through
        // while background noise is still rejected.
        // -------------------------------------------------------------------
        {
            std::vector<float> s (voicedRms);
            std::sort (s.begin(), s.end());
            const float med = s[s.size() / 2];
            r.gateThreshold = med * 0.20f;
            r.gateThreshold = std::max (0.005f, std::min (0.06f, r.gateThreshold));
        }

        // -------------------------------------------------------------------
        // Retune speed
        // Measure how far the singer typically sits from exact pitch.
        // We use cents deviation from the nearest chromatic semitone (scale-
        // agnostic) to get a pure intonation quality metric.
        //
        // Mapping:
        //   60th-percentile deviation < 15 cents  →  speed 0.35 (slow, natural)
        //   60th-percentile deviation ~ 35 cents  →  speed 0.50 (medium, typical)
        //   60th-percentile deviation > 55 cents  →  speed 0.65 (fast, needs help)
        //
        // This intentionally caps at 0.65 to stay in "natural" territory even
        // for less accurate singers — purely hard effect is always available via
        // the manual Retune Speed knob.
        // -------------------------------------------------------------------
        {
            std::vector<float> dev;
            dev.reserve (voicedF0.size());
            for (float hz : voicedF0)
            {
                const float midi  = 69.0f + 12.0f * std::log2 (hz / 440.0f);
                const float devC  = std::abs (midi - std::round (midi)) * 100.0f;
                dev.push_back (devC);
            }
            std::sort (dev.begin(), dev.end());
            const float p60 = dev[(size_t) ((float) dev.size() * 0.60f)];

            const float norm  = std::max (0.0f, std::min (1.0f, (p60 - 15.0f) / 40.0f));
            r.retuneSpeed     = 0.35f + norm * 0.30f;
        }

        r.valid = true;
        return r;
    }

    // -----------------------------------------------------------------------
    std::atomic<int> state_ { (int) State::Idle };
    int   targetHops_ = 0;
    int   collected_  = 0;

    std::vector<float> f0_;
    std::vector<float> voicing_;
    std::vector<float> rms_;

    CalibrationResult result_;
};
