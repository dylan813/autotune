cmake_minimum_required(VERSION 3.22)

project(AutotunePlugin
        VERSION 0.1.0
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# ---------------------------------------------------------------------------
# JUCE
#
# Two options:
#   A) Have JUCE installed locally and point at it with -DJUCE_DIR=/path/to/JUCE
#   B) Let CMake fetch it via FetchContent (slow first build, but reproducible).
# ---------------------------------------------------------------------------
option(AUTOTUNE_FETCH_JUCE "Use FetchContent to download JUCE" ON)

if (AUTOTUNE_FETCH_JUCE)
    FetchContent_Declare(
        JUCE
        GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
        GIT_TAG        7.0.12     # pin a known-good tag; bump as needed
    )
    FetchContent_MakeAvailable(JUCE)
else()
    find_package(JUCE CONFIG REQUIRED)
endif()

# ---------------------------------------------------------------------------
# Rubber Band library  (single-file build — no extra deps on Windows)
# ---------------------------------------------------------------------------
FetchContent_Declare(
    rubberband
    GIT_REPOSITORY https://github.com/breakfastquay/rubberband.git
    GIT_TAG        default          # default branch (repo uses Mercurial-style naming)
)
FetchContent_MakeAvailable(rubberband)

add_library(rubberband_single STATIC
    ${rubberband_SOURCE_DIR}/single/RubberBandSingle.cpp)

target_include_directories(rubberband_single PUBLIC
    ${rubberband_SOURCE_DIR})

target_compile_definitions(rubberband_single PRIVATE
    NOMINMAX
    _USE_MATH_DEFINES
    $<$<CXX_COMPILER_ID:MSVC>:_CRT_SECURE_NO_WARNINGS>)

set_target_properties(rubberband_single PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Plugin target
# ---------------------------------------------------------------------------
juce_add_plugin(Autotune
    COMPANY_NAME            "DylanAudio"
    PLUGIN_MANUFACTURER_CODE "DyAu"
    PLUGIN_CODE             "Atun"
    FORMATS                 VST3 Standalone
    PRODUCT_NAME            "Autotune"
    IS_SYNTH                FALSE
    NEEDS_MIDI_INPUT        FALSE
    NEEDS_MIDI_OUTPUT       FALSE
    IS_MIDI_EFFECT          FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE
    COPY_PLUGIN_AFTER_BUILD FALSE
)

juce_generate_juce_header(Autotune)

target_sources(Autotune PRIVATE
    Source/PluginProcessor.cpp
    Source/PluginProcessor.h
    Source/PluginEditor.cpp
    Source/PluginEditor.h
    Source/PitchDetector.h
    Source/YinPitchDetector.h
    Source/CorrectionPolicy.h
    Source/PitchShifter.h
    Source/ResamplePitchShifter.h
    Source/RubberbandPitchShifter.h
    Source/CalibrationEngine.h
    Source/ReferenceAnalyser.h
)

target_compile_definitions(Autotune PRIVATE
    JUCE_VST3_CAN_REPLACE_VST2=0
    JUCE_USE_CURL=0
    JUCE_WEB_BROWSER=0
    JUCE_USE_MP3AUDIOFORMAT=1
)

target_link_libraries(Autotune PRIVATE
    juce::juce_audio_utils
    juce::juce_dsp
    rubberband_single
)

# Warnings-as-errors locally; relax on CI as needed.
target_compile_options(Autotune PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>
)
