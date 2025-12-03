#pragma once

#include <Arduino.h>
#include <math.h>

/**
 * @file dsp_features.h
 * @brief Advanced DSP feature extraction for audio analysis
 *
 * This module provides real-time computation of advanced audio features:
 * - Spectral Centroid: Identifies the "brightness" or "center of mass" of the spectrum
 * - Zero-Crossing Rate: Measures the rate of sign changes in the waveform
 * - Peak Detection: Identifies local maxima with configurable hysteresis
 * - MFCC (Mel-Frequency Cepstral Coefficients): Perceptually-weighted spectral features
 * - Goertzel Algorithm: Efficient single-frequency detection
 */

typedef float dsp_real_t;

/**
 * @struct SpectralFeatures
 * @brief Container for computed spectral features
 */
struct SpectralFeatures {
    dsp_real_t centroid;        // Spectral centroid in Hz
    dsp_real_t flatness;        // Spectral flatness (0=tonal, 1=flat)
    dsp_real_t spread;          // Spectral spread (bandwidth around centroid)
    dsp_real_t rolloff_85;      // Frequency containing 85% of energy
};

/**
 * @struct TemporalFeatures
 * @brief Container for time-domain audio features
 */
struct TemporalFeatures {
    dsp_real_t zcr;             // Zero-Crossing Rate (normalized 0-1)
    dsp_real_t rms_energy;      // Root Mean Square energy
    dsp_real_t peak_amplitude;  // Highest absolute sample value
};

/**
 * @struct PeakDetectionConfig
 * @brief Configuration for peak detection algorithm
 */
struct PeakDetectionConfig {
    dsp_real_t threshold;       // Minimum magnitude to consider as peak
    dsp_real_t hysteresis;      // Hysteresis factor (prevents noise jitter)
    uint16_t min_width;         // Minimum samples between peaks
};

/**
 * @struct GoertzelState
 * @brief State machine for Goertzel single-frequency detection
 */
struct GoertzelState {
    dsp_real_t target_freq;     // Frequency to detect (Hz)
    dsp_real_t sample_rate;     // Sampling rate (Hz)

    // Filter coefficients
    dsp_real_t coeff;           // 2*cos(2*pi*k/N)

    // State variables
    dsp_real_t s0, s1, s2;      // Filter delay line samples
    uint32_t sample_count;      // Samples processed

    // Cached results
    dsp_real_t magnitude;       // Magnitude of detected frequency
    dsp_real_t phase;           // Phase of detected frequency
};

/**
 * @class DSPFeatureExtractor
 * @brief Computes advanced audio features from frequency and time domain data
 */
class DSPFeatureExtractor {
public:
    /**
     * @brief Compute spectral centroid and related spectral properties
     * @param spectrum Magnitude spectrum (positive frequencies only)
     * @param spectrum_len Length of spectrum array
     * @param freq_res Frequency resolution (Hz per bin)
     * @return SpectralFeatures containing centroid, flatness, spread, rolloff
     */
    static SpectralFeatures compute_spectral_centroid(
        const dsp_real_t* spectrum,
        uint16_t spectrum_len,
        dsp_real_t freq_res
    );

    /**
     * @brief Compute spectral flatness (ratio of geometric mean to arithmetic mean)
     * @param spectrum Magnitude spectrum
     * @param spectrum_len Length of spectrum
     * @return Flatness value 0-1 (0=tonal, 1=perfectly flat)
     */
    static dsp_real_t compute_spectral_flatness(
        const dsp_real_t* spectrum,
        uint16_t spectrum_len
    );

    /**
     * @brief Compute zero-crossing rate of time-domain signal
     * @param signal Raw audio samples
     * @param signal_len Number of samples
     * @return ZCR normalized to 0-1
     */
    static dsp_real_t compute_zcr(
        const int16_t* signal,
        uint16_t signal_len
    );

    /**
     * @brief Compute RMS energy of signal
     * @param signal Raw audio samples
     * @param signal_len Number of samples
     * @return RMS energy value
     */
    static dsp_real_t compute_rms_energy(
        const int16_t* signal,
        uint16_t signal_len
    );

    /**
     * @brief Compute temporal features from raw audio
     * @param signal Raw audio samples
     * @param signal_len Number of samples
     * @return TemporalFeatures struct with ZCR, RMS, and peak values
     */
    static TemporalFeatures compute_temporal_features(
        const int16_t* signal,
        uint16_t signal_len
    );

    /**
     * @brief Detect peaks in frequency spectrum with hysteresis
     * @param spectrum Magnitude spectrum
     * @param spectrum_len Length of spectrum
     * @param config Detection configuration
     * @param peaks_out Output array of peak bin indices
     * @param peaks_out_len Size of output array
     * @return Number of peaks found
     */
    static uint16_t detect_peaks(
        const dsp_real_t* spectrum,
        uint16_t spectrum_len,
        const PeakDetectionConfig& config,
        uint16_t* peaks_out,
        uint16_t peaks_out_len
    );

    /**
     * @brief Initialize Goertzel filter for single-frequency detection
     * @param state GoertzelState to initialize
     * @param target_freq Frequency to detect (Hz)
     * @param sample_rate Sampling rate (Hz)
     * @param sample_window Number of samples per detection window
     */
    static void goertzel_init(
        GoertzelState& state,
        dsp_real_t target_freq,
        dsp_real_t sample_rate,
        uint32_t sample_window
    );

    /**
     * @brief Process a single sample through Goertzel filter
     * @param state GoertzelState
     * @param sample Input audio sample (16-bit)
     */
    static void goertzel_process(GoertzelState& state, int16_t sample);

    /**
     * @brief Finalize Goertzel computation and extract magnitude/phase
     * @param state GoertzelState to finalize
     */
    static void goertzel_finalize(GoertzelState& state);

    /**
     * @brief Reset Goertzel state for next window
     * @param state GoertzelState to reset
     */
    static void goertzel_reset(GoertzelState& state);

    /**
     * @brief Compute Mel-scale frequency band averages
     * @param spectrum Linear frequency spectrum
     * @param spectrum_len Length of spectrum
     * @param freq_res Frequency resolution (Hz/bin)
     * @param mel_bands Output array of mel-scaled bands
     * @param num_bands Number of mel bands to compute
     */
    static void compute_mel_spectrum(
        const dsp_real_t* spectrum,
        uint16_t spectrum_len,
        dsp_real_t freq_res,
        dsp_real_t* mel_bands,
        uint8_t num_bands
    );
};

#endif // DSP_FEATURES_H
