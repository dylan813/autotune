// PluginEditor.h
//
// Minimal UI: combo boxes for key / scale, sliders for everything else.
// All controls attach to the processor's APVTS, so state save/restore
// and host automation work out of the box.

#pragma once

#include <JuceHeader.h>
#include <thread>
#include <atomic>
#include "PluginProcessor.h"
#include "ReferenceAnalyser.h"

class AutotuneAudioProcessorEditor : public juce::AudioProcessorEditor
{
public:
    explicit AutotuneAudioProcessorEditor (AutotuneAudioProcessor&);
    ~AutotuneAudioProcessorEditor() override;

    void paint (juce::Graphics&) override;
    void resized() override;

private:
    AutotuneAudioProcessor& processor_;

    juce::ComboBox keyBox_, scaleBox_;
    juce::Slider   retuneSlider_, amountSlider_, wetDrySlider_;
    juce::Slider   inGainSlider_, outGainSlider_;
    juce::ToggleButton formantsToggle_ { "Preserve Formants" };

    using APVTS = juce::AudioProcessorValueTreeState;
    std::unique_ptr<APVTS::ComboBoxAttachment>  keyAttachment_;
    std::unique_ptr<APVTS::ComboBoxAttachment>  scaleAttachment_;
    std::unique_ptr<APVTS::SliderAttachment>    retuneAttachment_;
    std::unique_ptr<APVTS::SliderAttachment>    amountAttachment_;
    std::unique_ptr<APVTS::SliderAttachment>    wetDryAttachment_;
    std::unique_ptr<APVTS::SliderAttachment>    inGainAttachment_;
    std::unique_ptr<APVTS::SliderAttachment>    outGainAttachment_;
    std::unique_ptr<APVTS::ButtonAttachment>    formantsAttachment_;

    void addLabelledRotary (juce::Slider& s, const juce::String& name);

    juce::Label detectionLabel_;
    juce::Timer* timer_ = nullptr;

    // Calibration UI --------------------------------------------------------
    juce::TextButton calibrateBtn_  { "Calibrate" };
    juce::ComboBox   calibDurationBox_;
    juce::ProgressBar calibProgress_;
    juce::Label       calibStatusLabel_;
    double calibProgressValue_ = 0.0;  // ProgressBar needs a double&

    // Reference import UI ---------------------------------------------------
    juce::TextButton  importRefBtn_    { "Import Reference..." };
    juce::Label       importRefNameLabel_;
    juce::ProgressBar importRefProgress_;
    juce::Label       importRefStatusLabel_;
    double            importRefProgressValue_ = 0.0;

    std::atomic<float> importRefProgressAtomic_ { 0.0f };
    std::atomic<bool>  importRefComplete_        { false };
    std::atomic<bool>  analysisShouldCancel_     { false };
    RefAnalysisResult  pendingRefResult_;

    std::shared_ptr<juce::FileChooser> fileChooser_;
    std::unique_ptr<std::thread>       analysisThread_;

    void startReferenceAnalysis (const juce::File& file);

    struct PollTimer : public juce::Timer
    {
        AutotuneAudioProcessorEditor& owner;
        PollTimer (AutotuneAudioProcessorEditor& o) : owner (o) {}
        void timerCallback() override
        {
            owner.updateDetectionLabel();
            owner.updateCalibrationUI();
        }
    };
    std::unique_ptr<PollTimer> pollTimer_;

    void updateDetectionLabel();
    void updateCalibrationUI();

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AutotuneAudioProcessorEditor)
};
