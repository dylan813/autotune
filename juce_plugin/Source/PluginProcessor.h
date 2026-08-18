// PluginProcessor.h
//
// JUCE AudioProcessor for the autotune plugin.
// Owns the parameter tree, the YIN-based pitch detector, the correction
// policy, and the resample pitch shifter. processBlock pulls one frame
// from each.

#pragma once

#include <JuceHeader.h>
#include <memory>
#include <vector>

#include "PitchDetector.h"
#include "CorrectionPolicy.h"
#include "PitchShifter.h"
#include "CalibrationEngine.h"

class AutotuneAudioProcessor : public juce::AudioProcessor
{
public:
    AutotuneAudioProcessor();
    ~AutotuneAudioProcessor() override;

    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;
    void processBlock (juce::AudioBuffer<float>& buffer,
                       juce::MidiBuffer& midi) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return "Autotune"; }
    bool acceptsMidi() const override  { return false; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }

    int  getNumPrograms() override          { return 1; }
    int  getCurrentProgram() override       { return 0; }
    void setCurrentProgram (int) override   {}
    const juce::String getProgramName (int) override         { return {}; }
    void changeProgramName (int, const juce::String&) override {}

    void getStateInformation (juce::MemoryBlock& dest) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

    // Start a calibration window of `durationSecs` seconds.
    // Safe to call from the message thread at any time.
    void requestCalibration (float durationSecs);

    juce::AudioProcessorValueTreeState apvts;

    // Read by the editor on the message thread — atomic so no lock needed.
    std::atomic<float> displayHz    { 0.0f };
    std::atomic<float> displayCents { 0.0f };

    // Calibration: set by audio thread, consumed by message thread (editor).
    std::atomic<bool>  calibrationReady_ { false };
    CalibrationResult  calibrationLatestResult_;   // valid when calibrationReady_ is true

    // Calibration progress [0,1] for the progress bar in the editor.
    std::atomic<float> calibrationProgress_ { 0.0f };

    // Envelope follower for smooth energy gating (no flickering).
    float envelopeRms_ = 0.0f;

    // Smoothed cents shift — ramps to the correction target over ~20 ms
    // so block-boundary jumps don't cause audible discontinuities.
    float smoothedCentsShift_ = 0.0f;

private:
    static juce::AudioProcessorValueTreeState::ParameterLayout createLayout();

    void pullParameters();

    std::unique_ptr<PitchDetector>   detector_;
    std::unique_ptr<PitchShifter>    shifter_;
    CorrectionPolicy                 correction_;
    CalibrationEngine                calibration_;

    // Atomic request from message thread → audio thread.
    std::atomic<int> calibrationRequestedHops_ { 0 };

    // Gate threshold — starts at default, updated after calibration.
    float dynamicGateThreshold_ = 0.02f;

    // Last computed cents shift; held across processBlock calls so we
    // can keep shifting between detector updates.
    float currentCentsShift_ = 0.0f;
    float lastDetectedHz_    = 0.0f;
    float lastTargetHz_      = 0.0f;

    // How many consecutive hops without a voiced detection.
    // Correction is held for kCorrectionHoldHops before being zeroed,
    // avoiding audible glitches on brief unvoiced frames mid-phrase.
    int unvoicedHops_ = 0;
    static constexpr int kCorrectionHoldHops = 15;  // ~87 ms — holds correction through consonants/breath noise

    int   hopSize_   = 256;
    int   frameSize_ = 1024;

    // Cached parameter snapshots.
    int   cachedKeyPc_   = 0;
    int   cachedScale_   = 0;
    float cachedRetune_  = 0.5f;
    float cachedAmount_  = 1.0f;
    float cachedWetDry_  = 1.0f;
    bool  cachedFormant_ = true;
    float cachedInGain_  = 0.0f;
    float cachedOutGain_ = 0.0f;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AutotuneAudioProcessor)
};
