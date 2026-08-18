// PluginEditor.cpp

#include "PluginEditor.h"

AutotuneAudioProcessorEditor::AutotuneAudioProcessorEditor (AutotuneAudioProcessor& p)
    : juce::AudioProcessorEditor (&p), processor_ (p),
      calibProgress_      (calibProgressValue_),
      importRefProgress_  (importRefProgressValue_)
{
    setSize (560, 465);

    keyBox_.addItemList ({ "C","C#","D","D#","E","F","F#","G","G#","A","A#","B" }, 1);
    scaleBox_.addItemList ({ "Major", "Minor", "Chromatic" }, 1);
    addAndMakeVisible (keyBox_);
    addAndMakeVisible (scaleBox_);

    addLabelledRotary (retuneSlider_, "Retune");
    addLabelledRotary (amountSlider_, "Amount");
    addLabelledRotary (wetDrySlider_, "Wet/Dry");
    addLabelledRotary (inGainSlider_, "In dB");
    addLabelledRotary (outGainSlider_, "Out dB");
    addAndMakeVisible (formantsToggle_);

    using APVTS = juce::AudioProcessorValueTreeState;
    keyAttachment_      = std::make_unique<APVTS::ComboBoxAttachment>(processor_.apvts, "key",              keyBox_);
    scaleAttachment_    = std::make_unique<APVTS::ComboBoxAttachment>(processor_.apvts, "scale",            scaleBox_);
    retuneAttachment_   = std::make_unique<APVTS::SliderAttachment>  (processor_.apvts, "retuneSpeed",      retuneSlider_);
    amountAttachment_   = std::make_unique<APVTS::SliderAttachment>  (processor_.apvts, "amount",           amountSlider_);
    wetDryAttachment_   = std::make_unique<APVTS::SliderAttachment>  (processor_.apvts, "wetDry",           wetDrySlider_);
    inGainAttachment_   = std::make_unique<APVTS::SliderAttachment>  (processor_.apvts, "inputGainDb",      inGainSlider_);
    outGainAttachment_  = std::make_unique<APVTS::SliderAttachment>  (processor_.apvts, "outputGainDb",     outGainSlider_);
    formantsAttachment_ = std::make_unique<APVTS::ButtonAttachment>  (processor_.apvts, "preserveFormants", formantsToggle_);

    detectionLabel_.setJustificationType (juce::Justification::centred);
    detectionLabel_.setFont (juce::Font (13.0f));
    detectionLabel_.setColour (juce::Label::textColourId, juce::Colours::limegreen);
    detectionLabel_.setText ("No pitch detected", juce::dontSendNotification);
    addAndMakeVisible (detectionLabel_);

    // -----------------------------------------------------------------------
    // Voice calibration section
    // -----------------------------------------------------------------------
    calibDurationBox_.addItem ("10 sec", 1);
    calibDurationBox_.addItem ("20 sec", 2);
    calibDurationBox_.addItem ("30 sec", 3);
    calibDurationBox_.setSelectedId (2, juce::dontSendNotification);
    addAndMakeVisible (calibDurationBox_);

    calibrateBtn_.setColour (juce::TextButton::buttonColourId,   juce::Colour (0xff2a5fad));
    calibrateBtn_.setColour (juce::TextButton::buttonOnColourId, juce::Colour (0xff1d4278));
    calibrateBtn_.onClick = [this]
    {
        const int id = calibDurationBox_.getSelectedId();
        const float secs = (id == 1) ? 10.0f : (id == 3) ? 30.0f : 20.0f;
        processor_.requestCalibration (secs);
        calibrateBtn_.setEnabled (false);
        calibStatusLabel_.setColour (juce::Label::textColourId, juce::Colours::yellow);
        calibStatusLabel_.setText ("Sing naturally — calibrating...", juce::dontSendNotification);
    };
    addAndMakeVisible (calibrateBtn_);

    calibProgress_.setColour (juce::ProgressBar::backgroundColourId, juce::Colour (0xff1e2228));
    calibProgress_.setColour (juce::ProgressBar::foregroundColourId, juce::Colour (0xff2a5fad));
    addAndMakeVisible (calibProgress_);

    calibStatusLabel_.setFont (juce::Font (12.0f));
    calibStatusLabel_.setColour (juce::Label::textColourId, juce::Colours::grey);
    calibStatusLabel_.setJustificationType (juce::Justification::centredLeft);
    calibStatusLabel_.setText ("Sing for a few seconds, then press Calibrate.", juce::dontSendNotification);
    addAndMakeVisible (calibStatusLabel_);

    // -----------------------------------------------------------------------
    // Reference import section
    // -----------------------------------------------------------------------
    importRefBtn_.setColour (juce::TextButton::buttonColourId,   juce::Colour (0xff3a6b3a));
    importRefBtn_.setColour (juce::TextButton::buttonOnColourId, juce::Colour (0xff274d27));
    importRefBtn_.onClick = [this]
    {
        fileChooser_ = std::make_shared<juce::FileChooser> (
            "Select reference vocal track",
            juce::File::getSpecialLocation (juce::File::userMusicDirectory),
            "*.wav;*.aif;*.aiff;*.flac;*.ogg;*.mp3");

        fileChooser_->launchAsync (
            juce::FileBrowserComponent::openMode |
            juce::FileBrowserComponent::canSelectFiles,
            [this] (const juce::FileChooser& fc)
            {
                const auto file = fc.getResult();
                if (file.existsAsFile())
                    startReferenceAnalysis (file);
            });
    };
    addAndMakeVisible (importRefBtn_);

    importRefNameLabel_.setFont (juce::Font (11.0f));
    importRefNameLabel_.setColour (juce::Label::textColourId, juce::Colours::grey);
    importRefNameLabel_.setJustificationType (juce::Justification::centredLeft);
    importRefNameLabel_.setText ("No file loaded", juce::dontSendNotification);
    addAndMakeVisible (importRefNameLabel_);

    importRefProgress_.setColour (juce::ProgressBar::backgroundColourId, juce::Colour (0xff1e2228));
    importRefProgress_.setColour (juce::ProgressBar::foregroundColourId, juce::Colour (0xff3a6b3a));
    addAndMakeVisible (importRefProgress_);

    importRefStatusLabel_.setFont (juce::Font (12.0f));
    importRefStatusLabel_.setColour (juce::Label::textColourId, juce::Colours::grey);
    importRefStatusLabel_.setJustificationType (juce::Justification::centredLeft);
    importRefStatusLabel_.setText ("Import a vocal snippet to match an artist's style.", juce::dontSendNotification);
    addAndMakeVisible (importRefStatusLabel_);

    pollTimer_ = std::make_unique<PollTimer> (*this);
    pollTimer_->startTimerHz (10);
}

AutotuneAudioProcessorEditor::~AutotuneAudioProcessorEditor()
{
    if (pollTimer_) pollTimer_->stopTimer();

    // Cancel and join any running analysis thread before member atomics go away.
    analysisShouldCancel_.store (true);
    if (analysisThread_ && analysisThread_->joinable())
        analysisThread_->join();
}

// ---------------------------------------------------------------------------
void AutotuneAudioProcessorEditor::startReferenceAnalysis (const juce::File& file)
{
    // Cancel any previous analysis.
    analysisShouldCancel_.store (true);
    if (analysisThread_ && analysisThread_->joinable())
        analysisThread_->join();

    // Read the audio file on the message thread (fast — just opens the reader).
    juce::AudioFormatManager fmgr;
    fmgr.registerBasicFormats();

    std::unique_ptr<juce::AudioFormatReader> reader (fmgr.createReaderFor (file));
    if (! reader)
    {
        importRefStatusLabel_.setColour (juce::Label::textColourId, juce::Colours::red);
        importRefStatusLabel_.setText ("Could not read file. Try WAV, AIFF, FLAC or OGG.",
                                       juce::dontSendNotification);
        return;
    }

    // Limit to first 60 seconds so memory stays bounded.
    const int maxSamples = (int) (reader->sampleRate * 60.0);
    const int numSamples = std::min ((int) reader->lengthInSamples, maxSamples);
    const double sr      = reader->sampleRate;

    // Load all channels then mix to mono.
    juce::AudioBuffer<float> raw ((int) reader->numChannels, numSamples);
    reader->read (&raw, 0, numSamples, 0, true, true);

    juce::AudioBuffer<float> mono (1, numSamples);
    mono.copyFrom (0, 0, raw, 0, 0, numSamples);
    if (raw.getNumChannels() > 1)
    {
        for (int i = 0; i < numSamples; ++i)
            mono.setSample (0, i, 0.5f * (raw.getSample (0, i) + raw.getSample (1, i)));
    }

    // Update UI.
    importRefNameLabel_.setText (file.getFileNameWithoutExtension(), juce::dontSendNotification);
    importRefProgressValue_ = 0.0;
    importRefProgressAtomic_.store (0.0f);
    importRefComplete_.store (false);
    importRefBtn_.setEnabled (false);
    importRefStatusLabel_.setColour (juce::Label::textColourId, juce::Colours::yellow);
    importRefStatusLabel_.setText ("Analysing — this takes a few seconds...",
                                   juce::dontSendNotification);

    // Kick off background analysis.
    analysisShouldCancel_.store (false);
    analysisThread_ = std::make_unique<std::thread> (
        [this, buf = std::move (mono), sampleRate = sr] () mutable
        {
            ReferenceAnalyser analyser;
            pendingRefResult_ = analyser.analyse (buf, sampleRate,
                                                  &importRefProgressAtomic_,
                                                  &analysisShouldCancel_);
            importRefComplete_.store (true);
        });
}

// ---------------------------------------------------------------------------
void AutotuneAudioProcessorEditor::updateDetectionLabel()
{
    const float hz    = processor_.displayHz.load();
    const float cents = processor_.displayCents.load();
    if (hz > 0.0f)
    {
        juce::Colour c = std::abs (cents) < 1.0f
            ? juce::Colours::limegreen
            : juce::Colours::orange;
        detectionLabel_.setColour (juce::Label::textColourId, c);
        detectionLabel_.setText (
            juce::String::formatted ("detected: %.1f Hz   correction: %+.1f cents", hz, cents),
            juce::dontSendNotification);
    }
    else
    {
        detectionLabel_.setColour (juce::Label::textColourId, juce::Colours::grey);
        detectionLabel_.setText ("No pitch detected", juce::dontSendNotification);
    }
}

// ---------------------------------------------------------------------------
void AutotuneAudioProcessorEditor::updateCalibrationUI()
{
    // --- Live calibration progress -----------------------------------------
    const float calibProg = processor_.calibrationProgress_.load();
    calibProgressValue_ = (double) calibProg;
    calibProgress_.repaint();

    if (processor_.calibrationReady_.exchange (false))
    {
        const auto& res = processor_.calibrationLatestResult_;
        if (auto* p = processor_.apvts.getParameter ("retuneSpeed"))
            p->setValueNotifyingHost (res.retuneSpeed);

        calibrateBtn_.setEnabled (true);
        calibStatusLabel_.setColour (juce::Label::textColourId, juce::Colours::limegreen);
        calibStatusLabel_.setText (
            juce::String::formatted ("Done — retune set to %.2f, gate %.3f",
                res.retuneSpeed, res.gateThreshold),
            juce::dontSendNotification);
    }

    // --- Reference import progress -----------------------------------------
    const float refProg = importRefProgressAtomic_.load();
    importRefProgressValue_ = (double) refProg;
    importRefProgress_.repaint();

    if (importRefComplete_.exchange (false))
    {
        importRefBtn_.setEnabled (true);

        if (! pendingRefResult_.valid)
        {
            importRefStatusLabel_.setColour (juce::Label::textColourId, juce::Colours::red);
            importRefStatusLabel_.setText (
                "Analysis failed — not enough voiced audio detected. Try a longer clip.",
                juce::dontSendNotification);
            return;
        }

        // Write key, scale, retune speed, amount into APVTS — knobs visibly move.
        if (auto* p = processor_.apvts.getParameter ("key"))
            p->setValueNotifyingHost ((float) pendingRefResult_.keyPc / 11.0f);

        if (auto* p = processor_.apvts.getParameter ("scale"))
            p->setValueNotifyingHost ((float) pendingRefResult_.scaleIdx / 2.0f);

        if (auto* p = processor_.apvts.getParameter ("retuneSpeed"))
            p->setValueNotifyingHost (pendingRefResult_.retuneSpeed);

        if (auto* p = processor_.apvts.getParameter ("amount"))
            p->setValueNotifyingHost (pendingRefResult_.amount);

        importRefStatusLabel_.setColour (juce::Label::textColourId, juce::Colours::limegreen);
        importRefStatusLabel_.setText (pendingRefResult_.summary, juce::dontSendNotification);
    }
}

// ---------------------------------------------------------------------------
void AutotuneAudioProcessorEditor::addLabelledRotary (juce::Slider& s,
                                                      const juce::String& name)
{
    s.setSliderStyle (juce::Slider::RotaryHorizontalVerticalDrag);
    s.setTextBoxStyle (juce::Slider::TextBoxBelow, false, 72, 18);
    s.setName (name);
    addAndMakeVisible (s);
}

// ---------------------------------------------------------------------------
void AutotuneAudioProcessorEditor::paint (juce::Graphics& g)
{
    g.fillAll (juce::Colour (0xff121417));

    g.setColour (juce::Colours::white);
    g.setFont (20.0f);
    g.drawFittedText ("Autotune", { 16, 12, 200, 24 }, juce::Justification::left, 1);

    g.setFont (12.0f);
    auto labelRow = [&] (const juce::String& s, int x, int y, int w)
    {
        g.drawFittedText (s, { x, y, w, 14 }, juce::Justification::centred, 1);
    };
    labelRow ("Key",     16,  48, 80);
    labelRow ("Scale",  104,  48, 100);
    labelRow ("Retune",  16, 108, 96);
    labelRow ("Amount", 120, 108, 96);
    labelRow ("Wet/Dry",224, 108, 96);
    labelRow ("In dB",  328, 108, 96);
    labelRow ("Out dB", 432, 108, 96);

    // Section dividers
    auto drawSection = [&] (int y, const char* title)
    {
        g.setColour (juce::Colour (0xff2a2d32));
        g.drawHorizontalLine (y, 16.0f, 544.0f);
        g.setColour (juce::Colours::darkgrey);
        g.setFont (10.5f);
        g.drawFittedText (title, { 16, y + 3, 220, 13 }, juce::Justification::left, 1);
    };
    drawSection (316, "VOICE CALIBRATION");
    drawSection (394, "REFERENCE IMPORT");
}

// ---------------------------------------------------------------------------
void AutotuneAudioProcessorEditor::resized()
{
    keyBox_.setBounds   (16,  64, 80,  24);
    scaleBox_.setBounds (104, 64, 100, 24);

    int x = 16;
    const int rotaryY = 124, rotaryW = 96, rotaryH = 110;
    for (auto* s : { &retuneSlider_, &amountSlider_, &wetDrySlider_,
                     &inGainSlider_, &outGainSlider_ })
    {
        s->setBounds (x, rotaryY, rotaryW, rotaryH);
        x += rotaryW + 8;
    }
    formantsToggle_.setBounds (16, 256, 200, 24);
    detectionLabel_.setBounds (16, 286, 528, 22);

    // Voice calibration row (y=334)
    calibrateBtn_.setBounds     (16,  334,  90, 24);
    calibDurationBox_.setBounds (114, 334,  80, 24);
    calibProgress_.setBounds    (202, 334, 358, 24);
    calibStatusLabel_.setBounds ( 16, 362, 528, 20);

    // Reference import row (y=408)
    importRefBtn_.setBounds       ( 16, 408, 140, 24);
    importRefNameLabel_.setBounds (164, 408, 218, 24);
    importRefProgress_.setBounds  (390, 408, 170, 24);
    importRefStatusLabel_.setBounds(16, 436, 528, 20);
}
