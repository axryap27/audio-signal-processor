#pragma once

#include <Arduino.h>
#include <math.h>

/*
 * dsp_features.h — real-time audio feature extraction for ESP32
 *
 * Provides spectral analysis (centroid, flatness, rolloff, mel bands),
 * time-domain features (ZCR, RMS, peak), peak detection with hysteresis,
 * and the Goertzel algorithm for single-frequency detection.
 */

typedef float dsp_real_t;

// ── Data Structures ─────────────────────────────────────────────────

struct SpectralFeatures {
    dsp_real_t centroid;     // centre of mass of the spectrum (Hz)
    dsp_real_t flatness;     // 0 = tonal, 1 = flat / noise-like
    dsp_real_t spread;       // bandwidth around the centroid (Hz)
    dsp_real_t rolloff_85;   // frequency below which 85 % of energy sits (Hz)
};

struct TemporalFeatures {
    dsp_real_t zcr;            // zero-crossing rate, normalised 0–1
    dsp_real_t rms_energy;     // root-mean-square energy
    dsp_real_t peak_amplitude; // highest absolute sample, normalised 0–1
};

struct PeakDetectionConfig {
    dsp_real_t threshold;   // minimum magnitude to qualify as a peak
    dsp_real_t hysteresis;  // prevents jitter near the threshold
    uint16_t   min_width;   // minimum bins between successive peaks
};

struct GoertzelState {
    dsp_real_t target_freq;  // frequency we're looking for (Hz)
    dsp_real_t sample_rate;  // sampling rate (Hz)

    dsp_real_t coeff;        // pre-computed 2*cos(2*pi*k/N)
    dsp_real_t s0, s1, s2;   // IIR delay line
    uint32_t   sample_count; // samples fed so far

    dsp_real_t magnitude;    // result: detected magnitude
    dsp_real_t phase;        // result: detected phase (radians)
};

// ── Feature Extractor ───────────────────────────────────────────────

class DSPFeatureExtractor {
public:
    // Spectral -----------------------------------------------------------

    static SpectralFeatures compute_spectral_centroid(
        const dsp_real_t* spectrum,
        uint16_t spectrum_len,
        dsp_real_t freq_res);

    static dsp_real_t compute_spectral_flatness(
        const dsp_real_t* spectrum,
        uint16_t spectrum_len);

    // Temporal -----------------------------------------------------------

    static dsp_real_t compute_zcr(
        const int16_t* signal,
        uint16_t signal_len);

    static dsp_real_t compute_rms_energy(
        const int16_t* signal,
        uint16_t signal_len);

    static TemporalFeatures compute_temporal_features(
        const int16_t* signal,
        uint16_t signal_len);

    // Peak detection -----------------------------------------------------

    static uint16_t detect_peaks(
        const dsp_real_t* spectrum,
        uint16_t spectrum_len,
        const PeakDetectionConfig& config,
        uint16_t* peaks_out,
        uint16_t peaks_out_len);

    // Goertzel single-frequency detector ---------------------------------

    static void goertzel_init(
        GoertzelState& state,
        dsp_real_t target_freq,
        dsp_real_t sample_rate,
        uint32_t sample_window);

    static void goertzel_process(GoertzelState& state, int16_t sample);
    static void goertzel_finalize(GoertzelState& state);
    static void goertzel_reset(GoertzelState& state);

    // Mel-scale spectrum -------------------------------------------------

    static void compute_mel_spectrum(
        const dsp_real_t* spectrum,
        uint16_t spectrum_len,
        dsp_real_t freq_res,
        dsp_real_t* mel_bands,
        uint8_t num_bands);
};
