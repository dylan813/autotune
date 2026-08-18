// PluginProcessor.cpp

#include "PluginProcessor.h"
#include "PluginEditor.h"
#include "YinPitchDetector.h"
#include "RubberbandPitchShifter.h"

namespace
{
    inline float dbToGain (float db) { return std::pow (10.0f, db * 0.05f); }
}

AutotuneAudioProcessor::AutotuneAudioProcessor()
    : juce::AudioProcessor (BusesProperties()
        .withInput  ("Input",  juce::AudioChannelSet::stereo(), true)
        .withOutput ("Output", juce::AudioChannelSet::stereo(), true)),
      apvts (*this, nullptr, "PARAMS", createLayout())
{
    detector_ = std::make_unique<YinPitchDetector>();
    shifter_  = std::make_unique<RubberbandPitchShifter>();
}

AutotuneAudioProcessor::~AutotuneAudioProcessor() = default;

juce::AudioProcessorValueTreeState::ParameterLayout
AutotuneAudioProcessor::createLayout()
{
    juce::AudioProcessorValueTreeState::ParameterLayout layout;

    layout.add (std::make_unique<juce::AudioParameterChoice>(
        "key", "Key",
        juce::StringArray{ "C","C#","D","D#","E","F","F#","G","G#","A","A#","B" }, 0));

    layout.add (std::make_unique<juce::AudioParameterChoice>(
        "scale", "Scale",
        juce::StringArray{ "Major", "Minor", "Chromatic" }, 0));

    layout.add (std::make_unique<juce::AudioParameterFloat>(
        "retuneSpeed", "Retune Speed",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.001f), 0.5f));  // ~35ms tau — natural pop/R&B default

    layout.add (std::make_unique<juce::AudioParameterFloat>(
        "amount", "Amount",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.001f), 1.0f));

    layout.add (std::make_unique<juce::AudioParameterFloat>(
        "wetDry", "Wet/Dry",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.001f), 1.0f));

    layout.add (std::make_unique<juce::AudioParameterBool>(
        "preserveFormants", "Preserve Formants", true));

    layout.add (std::make_unique<juce::AudioParameterFloat>(
        "inputGainDb",  "Input Gain",
        juce::NormalisableRange<float>(-24.0f, 24.0f, 0.1f), 0.0f));

    layout.add (std::make_unique<juce::AudioParameterFloat>(
        "outputGainDb", "Output Gain",
        juce::NormalisableRange<float>(-24.0f, 24.0f, 0.1f), 0.0f));

    return layout;
}

void AutotuneAudioProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    detector_->prepare (sampleRate, hopSize_, frameSize_);
    correction_.prepare (sampleRate, hopSize_);
    shifter_->prepare (sampleRate, samplesPerBlock);
    currentCentsShift_ = 0.0f;
    lastDetectedHz_    = 0.0f;
    lastTargetHz_      = 0.0f;
    unvoicedHops_      = 0;

    // Tell the DAW how many samples of latency we introduce so it can
    // delay other tracks and keep everything time-aligned.
    setLatencySamples (shifter_->getLatencySamples());
}

void AutotuneAudioProcessor::releaseResources() {}

bool AutotuneAudioProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
    // Accept mono or stereo, as long as in/out channel counts match.
    const auto& in  = layouts.getMainInputChannelSet();
    const auto& out = layouts.getMainOutputChannelSet();
    if (in != out) return false;
    return in == juce::AudioChannelSet::mono()
        || in == juce::AudioChannelSet::stereo();
}

void AutotuneAudioProcessor::pullParameters()
{
    cachedKeyPc_   = (int)  apvts.getRawParameterValue ("key")              ->load();
    cachedScale_   = (int)  apvts.getRawParameterValue ("scale")            ->load();
    cachedRetune_  = (float) apvts.getRawParameterValue ("retuneSpeed")     ->load();
    cachedAmount_  = (float) apvts.getRawParameterValue ("amount")          ->load();
    cachedWetDry_  = (float) apvts.getRawParameterValue ("wetDry")          ->load();
    cachedFormant_ = (bool)  apvts.getRawParameterValue ("preserveFormants")->load();
    cachedInGain_  = (float) apvts.getRawParameterValue ("inputGainDb")     ->load();
    cachedOutGain_ = (float) apvts.getRawParameterValue ("outputGainDb")    ->load();

    CorrectionConfig cfg;
    cfg.keyPc          = cachedKeyPc_;
    cfg.scale          = static_cast<Scale> (cachedScale_);
    cfg.retuneSpeed    = cachedRetune_;
    cfg.amount         = cachedAmount_;
    cfg.minDwellHops    = 8;     // ~46 ms — matches Antares' note transition window
    cfg.hysteresisCents = 50.0f; // exactly half-semitone: catches off-pitch, ignores normal vibrato (±30–40 cents)
    correction_.setConfig (cfg);
}

void AutotuneAudioProcessor::processBlock (juce::AudioBuffer<float>& buffer,
                                           juce::MidiBuffer&)
{
    juce::ScopedNoDenormals noDenormals;
    pullParameters();

    const int numSamples  = buffer.getNumSamples();
    const int numChannels = buffer.getNumChannels();
    const float inGain    = dbToGain (cachedInGain_);
    const float outGain   = dbToGain (cachedOutGain_);
    const float w         = cachedWetDry_;

    // Apply input gain to all channels.
    for (int ch = 0; ch < numChannels; ++ch)
    {
        auto* d = buffer.getWritePointer (ch);
        for (int i = 0; i < numSamples; ++i)
            d[i] *= inGain;
    }

    // Mix channels to mono for pitch detection.
    static thread_local std::vector<float> mono;
    if ((int) mono.size() < numSamples) mono.assign ((size_t) numSamples, 0.0f);
    mono.assign (buffer.getReadPointer (0), buffer.getReadPointer (0) + numSamples);
    if (numChannels > 1)
    {
        auto* ch1 = buffer.getReadPointer (1);
        for (int i = 0; i < numSamples; ++i)
            mono[(size_t) i] = 0.5f * (mono[(size_t) i] + ch1[i]);
    }

    // Envelope follower: fast attack (~5 ms), slow release (~150 ms).
    // Smooths over breath variations so the gate doesn't flicker.
    float rmsSum = 0.0f;
    for (int i = 0; i < numSamples; ++i)
        rmsSum += mono[(size_t)i] * mono[(size_t)i];
    const float instantRms = std::sqrt (rmsSum / (float) numSamples);
    // Coefficients must be per-block, not per-sample.
    const float sr          = (float) getSampleRate();
    const float attackCoef  = 1.0f - std::exp (-(float) numSamples / (0.005f * sr));
    const float releaseCoef = 1.0f - std::exp (-(float) numSamples / (0.150f * sr));
    const float coef = instantRms > envelopeRms_ ? attackCoef : releaseCoef;
    envelopeRms_ = coef * instantRms + (1.0f - coef) * envelopeRms_;
    const bool hasSignal = envelopeRms_ > dynamicGateThreshold_;

    // -----------------------------------------------------------------------
    // Calibration: start if the editor requested one.
    // -----------------------------------------------------------------------
    {
        const int requested = calibrationRequestedHops_.exchange (0);
        if (requested > 0)
            calibration_.start (requested);
    }

    // Run pitch detection on the mono mix and update correction.
    // The correction is held for kCorrectionHoldHops consecutive hops
    // without a voiced detection before being zeroed — this prevents
    // audible glitches from brief unvoiced frames mid-phrase.
    if (hasSignal)
    {
        F0Frame frame{};
        if (detector_->pushHop (mono.data(), numSamples, frame))
        {
            lastDetectedHz_ = frame.f0Hz;
            const float corrected =
                correction_.updateAndGetTargetHz (frame.f0Hz, frame.voicing);

            if (corrected > 0.0f && lastDetectedHz_ > 0.0f)
            {
                lastTargetHz_      = corrected;
                currentCentsShift_ = 1200.0f * std::log2 (corrected / lastDetectedHz_);
                unvoicedHops_      = 0;
            }
            else
            {
                // Unvoiced frame — hold before zeroing.
                if (++unvoicedHops_ > kCorrectionHoldHops)
                    currentCentsShift_ = 0.0f;
            }

            displayHz.store (lastDetectedHz_);
            displayCents.store (currentCentsShift_);

            // Feed the calibration engine (audio thread only).
            if (calibration_.getState() == CalibrationEngine::State::Calibrating)
                calibration_.pushFrame (lastDetectedHz_, frame.voicing, envelopeRms_);
        }
        // If pushHop returns false (hop not full yet), hold everything as-is.
    }
    else
    {
        // No signal — hold briefly then zero to avoid mid-phrase glitches.
        if (++unvoicedHops_ > kCorrectionHoldHops)
        {
            currentCentsShift_ = 0.0f;
            displayHz.store (0.0f);
            displayCents.store (0.0f);
        }
    }

    // Publish calibration progress and completion for the editor's poll timer.
    calibrationProgress_.store (calibration_.getProgress());
    if (calibration_.getState() == CalibrationEngine::State::Complete
        && ! calibrationReady_.load())
    {
        const auto& res = calibration_.getResult();
        if (res.valid)
        {
            calibrationLatestResult_ = res;
            dynamicGateThreshold_    = res.gateThreshold;
            calibrationReady_.store (true);
        }
        calibration_.reset();
    }

    // Smooth the cents shift with a ~20 ms ramp so block-boundary
    // jumps in the correction target don't create audible discontinuities.
    {
        const float centsAlpha = 1.0f - std::exp (-(float) numSamples / (0.025f * (float) getSampleRate()));
        smoothedCentsShift_ = centsAlpha * currentCentsShift_
                            + (1.0f - centsAlpha) * smoothedCentsShift_;
    }

    // Apply pitch shift.
    // The stretcher is a stateful mono stream — call it once on channel 0,
    // then reuse the same wet buffer for all channels.  The dry component
    // is still taken per-channel so the stereo field is preserved.
    static thread_local std::vector<float> wet;
    if ((int) wet.size() < numSamples) wet.assign ((size_t) numSamples, 0.0f);

    {
        auto* ch0 = buffer.getWritePointer (0);
        shifter_->process (ch0, wet.data(), numSamples,
                           smoothedCentsShift_, cachedFormant_);
    }

    for (int ch = 0; ch < numChannels; ++ch)
    {
        auto* data = buffer.getWritePointer (ch);
        for (int i = 0; i < numSamples; ++i)
            data[i] = outGain * (w * wet[(size_t) i] + (1.0f - w) * data[i]);
    }
}

void AutotuneAudioProcessor::requestCalibration (float durationSecs)
{
    // Called from the message thread; hands off to the audio thread via atomic.
    const int hops = (int) (durationSecs * getSampleRate() / (double) hopSize_);
    calibrationReady_.store (false);
    calibrationRequestedHops_.store (hops > 0 ? hops : 1);
}

juce::AudioProcessorEditor* AutotuneAudioProcessor::createEditor()
{
    return new AutotuneAudioProcessorEditor (*this);
}

void AutotuneAudioProcessor::getStateInformation (juce::MemoryBlock& dest)
{
    if (auto xml = apvts.copyState().createXml())
        copyXmlToBinary (*xml, dest);
}

void AutotuneAudioProcessor::setStateInformation (const void* data, int size)
{
    if (auto xml = getXmlFromBinary (data, size))
        apvts.replaceState (juce::ValueTree::fromXml (*xml));
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new AutotuneAudioProcessor();
}
