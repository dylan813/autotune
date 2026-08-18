// ReferenceAnalyser.h
//
// Analyses a reference audio snippet of a processed (auto-tuned) vocal to
// reverse-engineer the correction style used, so you can match an artist's
// settings.
//
// Algorithm:
//   1. YIN pitch extraction over the whole file
//   2. Key + scale via Krumhansl-Schmuckler pitch-class profile correlation
//   3. Retune speed via lag-1 autocorrelation of snap-deviation
//      (low correlation → pitch snaps instantly → fast retune)
//      (high correlation → pitch glides smoothly → slow retune)
//   4. Correction amount via mean deviation from nearest scale note
//
// Call analyse() from a background thread; it is fully self-contained and
// does not touch any JUCE message-thread state.

#pragma once

#include <JuceHeader.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <atomic>

#include "YinPitchDetector.h"

// ============================================================================
struct RefAnalysisResult
{
    int   keyPc       = 0;      // 0 = C, 1 = C#, ..., 11 = B
    int   scaleIdx    = 0;      // 0 = Major, 1 = Minor, 2 = Chromatic
    float retuneSpeed = 0.5f;   // APVTS "retuneSpeed" 0-1
    float amount      = 1.0f;   // APVTS "amount"      0-1
    bool  valid       = false;
    juce::String summary;       // human-readable one-liner shown in UI
};

// ============================================================================
class ReferenceAnalyser
{
public:
    // -----------------------------------------------------------------------
    // Run the full analysis on a mono audio buffer.
    // progressOut — optional atomic updated 0→1 during processing.
    // shouldCancel — set to true externally to abort early.
    // -----------------------------------------------------------------------
    RefAnalysisResult analyse (const juce::AudioBuffer<float>& monoAudio,
                               double                           sampleRate,
                               std::atomic<float>*              progressOut  = nullptr,
                               std::atomic<bool>*               shouldCancel = nullptr)
    {
        RefAnalysisResult result;

        const int numSamples = monoAudio.getNumSamples();
        if (numSamples < 2048) return result;

        // ---- 1. Pitch extraction via YIN ----------------------------------
        constexpr int hopSize   = 256;
        constexpr int frameSize = 1024;

        YinPitchDetector yin;
        yin.prepare (sampleRate, hopSize, frameSize);

        std::vector<float> voicedF0;
        voicedF0.reserve ((size_t) numSamples / hopSize);

        const float* src      = monoAudio.getReadPointer (0);
        const int    totalHops = numSamples / hopSize;
        int          hopsDone  = 0;

        for (int pos = 0; pos + hopSize <= numSamples; pos += hopSize)
        {
            if (shouldCancel && shouldCancel->load()) return result;

            F0Frame frame{};
            if (yin.pushHop (src + pos, hopSize, frame))
                if (frame.f0Hz > 0.0f && frame.voicing > 0.25f)
                    voicedF0.push_back (frame.f0Hz);

            if (progressOut)
                progressOut->store (0.80f * (float) ++hopsDone / (float) totalHops);
        }

        // Need at least ~1 second of voiced audio.
        if ((int) voicedF0.size() < 30) return result;

        // ---- 2. Key + scale -----------------------------------------------
        detectKeyAndScale (voicedF0, result.keyPc, result.scaleIdx);
        if (progressOut) progressOut->store (0.88f);
        if (shouldCancel && shouldCancel->load()) return result;

        // ---- 3. Retune speed (lag-1 autocorrelation of deviation) ---------
        result.retuneSpeed = estimateRetuneSpeed (voicedF0, result.keyPc, result.scaleIdx);
        if (progressOut) progressOut->store (0.94f);

        // ---- 4. Correction amount (mean deviation from scale notes) -------
        result.amount = estimateAmount (voicedF0, result.keyPc, result.scaleIdx);
        if (progressOut) progressOut->store (1.00f);

        // ---- 5. Summary string --------------------------------------------
        static const char* notes[]  = { "C","C#","D","D#","E","F","F#","G","G#","A","A#","B" };
        static const char* scales[] = { "Major","Minor","Chromatic" };
        const char* style =
              (result.retuneSpeed > 0.78f) ? "hard effect (T-Pain / Travis)"
            : (result.retuneSpeed > 0.58f) ? "noticeable (modern pop)"
            : (result.retuneSpeed > 0.42f) ? "natural (pop/R&B)"
            :                                 "transparent (light correction)";

        result.summary = juce::String::formatted (
            "%s %s  |  retune %.2f  |  amount %.0f%%  |  %s",
            notes[result.keyPc], scales[result.scaleIdx],
            result.retuneSpeed, result.amount * 100.0f, style);

        result.valid = true;
        return result;
    }

private:
    // -----------------------------------------------------------------------
    static float snapToScaleMidi (float midi, int keyPc, int scaleIdx)
    {
        static constexpr int major[]     = { 0, 2, 4, 5, 7, 9, 11 };
        static constexpr int minor[]     = { 0, 2, 3, 5, 7, 8, 10 };
        static constexpr int chromatic[] = { 0,1,2,3,4,5,6,7,8,9,10,11 };

        const int* ivs  = (scaleIdx == 0) ? major
                        : (scaleIdx == 1) ? minor : chromatic;
        const int  nIvs = (scaleIdx == 2) ? 12 : 7;

        const int baseOct = (int) std::floor (midi / 12.0f);
        float bestErr = 1e9f, bestMidi = midi;

        for (int oct = baseOct - 1; oct <= baseOct + 1; ++oct)
            for (int k = 0; k < nIvs; ++k)
            {
                const float cand = (float)(oct * 12 + ((keyPc + ivs[k]) % 12));
                const float err  = std::abs (cand - midi);
                if (err < bestErr) { bestErr = err; bestMidi = cand; }
            }
        return bestMidi;
    }

    // -----------------------------------------------------------------------
    // Krumhansl-Schmuckler key-finding: correlate a pitch-class histogram
    // against rotations of the major and minor tonal profiles.
    // -----------------------------------------------------------------------
    static void detectKeyAndScale (const std::vector<float>& f0Hz,
                                   int& keyPcOut, int& scaleIdxOut)
    {
        // Build pitch-class histogram.
        float hist[12] = {};
        for (float hz : f0Hz)
        {
            const float midi = 69.0f + 12.0f * std::log2 (hz / 440.0f);
            const int   pc   = ((int) std::round (midi) % 12 + 12) % 12;
            hist[pc] += 1.0f;
        }

        // Krumhansl (1990) tonal hierarchy profiles.
        static constexpr float ksMajor[12] = {
            6.35f, 2.23f, 3.48f, 2.33f, 4.38f, 4.09f,
            2.52f, 5.19f, 2.39f, 3.66f, 2.29f, 2.88f };
        static constexpr float ksMinor[12] = {
            6.33f, 2.68f, 3.52f, 5.38f, 2.60f, 3.53f,
            2.54f, 4.75f, 3.98f, 2.69f, 3.34f, 3.17f };

        // Pearson correlation between histogram and a profile rotated to root.
        auto correlate = [](const float* h, const float* prof, int root) -> float
        {
            float sh = 0, sp = 0, sh2 = 0, sp2 = 0, cross = 0;
            for (int i = 0; i < 12; ++i)
            {
                const float hi = h[i];
                const float pi = prof[(i - root + 12) % 12];
                sh += hi; sp += pi; sh2 += hi*hi; sp2 += pi*pi; cross += hi*pi;
            }
            const float num   = 12.0f * cross - sh * sp;
            const float denom = std::sqrt ((12.0f*sh2 - sh*sh) * (12.0f*sp2 - sp*sp));
            return (denom > 1e-6f) ? num / denom : 0.0f;
        };

        float bestScore = -1e9f;
        keyPcOut    = 0;
        scaleIdxOut = 0;

        for (int root = 0; root < 12; ++root)
        {
            float sM = correlate (hist, ksMajor, root);
            float sm = correlate (hist, ksMinor, root);
            if (sM > bestScore) { bestScore = sM; keyPcOut = root; scaleIdxOut = 0; }
            if (sm > bestScore) { bestScore = sm; keyPcOut = root; scaleIdxOut = 1; }
        }
    }

    // -----------------------------------------------------------------------
    // Retune speed via lag-1 autocorrelation of pitch deviation from the
    // nearest scale note.
    //
    // Intuition: a fast retune snaps the pitch immediately — adjacent frames
    // have nearly uncorrelated small residuals (r1 ≈ 0).  A slow retune lets
    // the pitch glide smoothly toward the target — adjacent frames share a
    // consistent trend (r1 ≈ 0.8–0.9).
    //
    // Mapping:  speed = 0.90 − r1 × 0.65,  clamped [0.20, 0.95]
    // -----------------------------------------------------------------------
    static float estimateRetuneSpeed (const std::vector<float>& f0Hz,
                                      int keyPc, int scaleIdx)
    {
        if ((int) f0Hz.size() < 4) return 0.5f;

        std::vector<float> dev;
        dev.reserve (f0Hz.size());
        for (float hz : f0Hz)
        {
            const float midi    = 69.0f + 12.0f * std::log2 (hz / 440.0f);
            const float snapped = snapToScaleMidi (midi, keyPc, scaleIdx);
            dev.push_back ((midi - snapped) * 100.0f);  // cents
        }

        float mean = 0.0f;
        for (float d : dev) mean += d;
        mean /= (float) dev.size();

        float num = 0.0f, den = 0.0f;
        for (int i = 1; i < (int) dev.size(); ++i)
        {
            const float a = dev[(size_t)(i-1)] - mean;
            const float b = dev[(size_t) i   ] - mean;
            num += a * b;
            den += b * b;
        }
        const float r1    = (den > 1e-6f) ? std::max (-1.0f, std::min (1.0f, num / den)) : 0.0f;
        const float speed = 0.90f - r1 * 0.65f;
        return std::max (0.20f, std::min (0.95f, speed));
    }

    // -----------------------------------------------------------------------
    // Correction amount: if pitch stays tightly on scale notes (< 5 cents mean
    // deviation) the track was heavily corrected (amount → 1.0).  More
    // deviation implies lighter correction.
    // -----------------------------------------------------------------------
    static float estimateAmount (const std::vector<float>& f0Hz,
                                 int keyPc, int scaleIdx)
    {
        if (f0Hz.empty()) return 1.0f;

        float sumDev = 0.0f;
        for (float hz : f0Hz)
        {
            const float midi    = 69.0f + 12.0f * std::log2 (hz / 440.0f);
            const float snapped = snapToScaleMidi (midi, keyPc, scaleIdx);
            sumDev += std::abs (midi - snapped) * 100.0f;
        }
        const float meanDev = sumDev / (float) f0Hz.size();

        // 0–5 cents → 1.0;   45+ cents → 0.2
        return 1.0f - std::max (0.0f, std::min (0.8f, (meanDev - 5.0f) / 50.0f));
    }
};
