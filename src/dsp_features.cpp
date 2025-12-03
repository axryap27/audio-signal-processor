#include "dsp_features.h"

/**
 * @brief Compute spectral centroid and related spectral properties
 *
 * The spectral centroid is the "center of mass" of the spectrum.
 * It's an important feature for music analysis - high centroid suggests bright sounds,
 * low centroid suggests dark/bass-heavy sounds.
 */
SpectralFeatures DSPFeatureExtractor::compute_spectral_centroid(
    const dsp_real_t* spectrum,
    uint16_t spectrum_len,
    dsp_real_t freq_res
) {
    SpectralFeatures features = {0};

    if (spectrum_len == 0) return features;

    dsp_real_t weighted_sum = 0.0;
    dsp_real_t total_energy = 0.0;
    dsp_real_t freq;

    // Compute centroid: sum(f * magnitude) / sum(magnitude)
    for (uint16_t i = 0; i < spectrum_len; i++) {
        freq = i * freq_res;
        weighted_sum += freq * spectrum[i];
        total_energy += spectrum[i];
    }

    features.centroid = (total_energy > 0) ? weighted_sum / total_energy : 0;

    // Compute spectral spread: sqrt(sum((f - centroid)^2 * magnitude) / sum(magnitude))
    dsp_real_t spread_sum = 0.0;
    for (uint16_t i = 0; i < spectrum_len; i++) {
        freq = i * freq_res;
        dsp_real_t diff = freq - features.centroid;
        spread_sum += (diff * diff) * spectrum[i];
    }
    features.spread = (total_energy > 0) ? sqrt(spread_sum / total_energy) : 0;

    // Compute rolloff: frequency containing 85% of energy
    dsp_real_t rolloff_threshold = 0.85 * total_energy;
    dsp_real_t energy_accumulator = 0.0;
    for (uint16_t i = 0; i < spectrum_len; i++) {
        energy_accumulator += spectrum[i];
        if (energy_accumulator >= rolloff_threshold) {
            features.rolloff_85 = i * freq_res;
            break;
        }
    }

    // Compute flatness: geometric_mean / arithmetic_mean
    dsp_real_t log_sum = 0.0;
    dsp_real_t mean_magnitude = total_energy / spectrum_len;

    for (uint16_t i = 0; i < spectrum_len; i++) {
        if (spectrum[i] > 1e-10) {  // Avoid log(0)
            log_sum += log(spectrum[i]);
        }
    }

    if (mean_magnitude > 1e-10) {
        dsp_real_t geometric_mean = exp(log_sum / spectrum_len);
        features.flatness = geometric_mean / mean_magnitude;
        // Clamp to 0-1
        features.flatness = fmin(1.0, fmax(0.0, features.flatness));
    }

    return features;
}

/**
 * @brief Compute spectral flatness
 *
 * Indicates whether spectrum is "peaky" (tonal, flatness near 0) or flat (noise-like, near 1)
 */
dsp_real_t DSPFeatureExtractor::compute_spectral_flatness(
    const dsp_real_t* spectrum,
    uint16_t spectrum_len
) {
    if (spectrum_len == 0) return 0;

    dsp_real_t log_sum = 0.0;
    dsp_real_t arithmetic_sum = 0.0;

    for (uint16_t i = 0; i < spectrum_len; i++) {
        arithmetic_sum += spectrum[i];
        if (spectrum[i] > 1e-10) {
            log_sum += log(spectrum[i]);
        }
    }

    dsp_real_t geometric_mean = exp(log_sum / spectrum_len);
    dsp_real_t arithmetic_mean = arithmetic_sum / spectrum_len;

    if (arithmetic_mean > 1e-10) {
        dsp_real_t flatness = geometric_mean / arithmetic_mean;
        return fmin(1.0, fmax(0.0, flatness));
    }

    return 0;
}

/**
 * @brief Compute zero-crossing rate
 *
 * ZCR is the rate at which the signal changes sign.
 * High ZCR indicates high-frequency content (consonants, noise)
 * Low ZCR indicates low-frequency or voiced speech (vowels, bass)
 */
dsp_real_t DSPFeatureExtractor::compute_zcr(
    const int16_t* signal,
    uint16_t signal_len
) {
    if (signal_len < 2) return 0;

    uint32_t zero_crossings = 0;

    for (uint16_t i = 1; i < signal_len; i++) {
        // Count sign changes
        if ((signal[i] >= 0 && signal[i - 1] < 0) ||
            (signal[i] < 0 && signal[i - 1] >= 0)) {
            zero_crossings++;
        }
    }

    // Normalize to 0-1 (maximum possible zero crossings is len-1)
    return (dsp_real_t)zero_crossings / (signal_len - 1);
}

/**
 * @brief Compute RMS energy
 *
 * Measures the overall loudness/intensity of the signal
 */
dsp_real_t DSPFeatureExtractor::compute_rms_energy(
    const int16_t* signal,
    uint16_t signal_len
) {
    if (signal_len == 0) return 0;

    int64_t sum_squares = 0;

    for (uint16_t i = 0; i < signal_len; i++) {
        int32_t sample = signal[i];
        sum_squares += sample * sample;
    }

    dsp_real_t mean_square = (dsp_real_t)sum_squares / signal_len;
    return sqrt(mean_square);
}

/**
 * @brief Compute multiple temporal features at once
 */
TemporalFeatures DSPFeatureExtractor::compute_temporal_features(
    const int16_t* signal,
    uint16_t signal_len
) {
    TemporalFeatures features = {0};

    if (signal_len == 0) return features;

    features.zcr = compute_zcr(signal, signal_len);
    features.rms_energy = compute_rms_energy(signal, signal_len);

    // Compute peak amplitude
    int16_t peak = 0;
    for (uint16_t i = 0; i < signal_len; i++) {
        int16_t abs_sample = abs(signal[i]);
        if (abs_sample > peak) {
            peak = abs_sample;
        }
    }
    features.peak_amplitude = (dsp_real_t)peak / 32768.0;  // Normalize to 0-1

    return features;
}

/**
 * @brief Detect peaks in spectrum with hysteresis
 *
 * Hysteresis prevents noise from creating spurious peaks.
 * A sample becomes a peak only if it's > threshold and > both neighbors
 */
uint16_t DSPFeatureExtractor::detect_peaks(
    const dsp_real_t* spectrum,
    uint16_t spectrum_len,
    const PeakDetectionConfig& config,
    uint16_t* peaks_out,
    uint16_t peaks_out_len
) {
    uint16_t peak_count = 0;
    uint16_t last_peak = 0;

    if (spectrum_len < 3) return 0;

    for (uint16_t i = 1; i < spectrum_len - 1; i++) {
        // Check if current sample is a local maximum
        if (spectrum[i] > config.threshold &&
            spectrum[i] > spectrum[i - 1] &&
            spectrum[i] > spectrum[i + 1] &&
            (i - last_peak) >= config.min_width) {

            if (peak_count < peaks_out_len) {
                peaks_out[peak_count++] = i;
                last_peak = i;
            }
        }
    }

    return peak_count;
}

/**
 * @brief Initialize Goertzel filter for single-frequency detection
 *
 * Goertzel algorithm is more efficient than FFT for detecting a single frequency.
 * It uses an IIR filter with a pole on the unit circle at the target frequency.
 */
void DSPFeatureExtractor::goertzel_init(
    GoertzelState& state,
    dsp_real_t target_freq,
    dsp_real_t sample_rate,
    uint32_t sample_window
) {
    state.target_freq = target_freq;
    state.sample_rate = sample_rate;
    state.sample_count = 0;

    // Compute filter coefficient: 2*cos(2*pi*k/N)
    dsp_real_t k = target_freq * sample_window / sample_rate;
    state.coeff = 2.0 * cos(2.0 * M_PI * k / sample_window);

    // Initialize filter state
    state.s0 = 0;
    state.s1 = 0;
    state.s2 = 0;

    state.magnitude = 0;
    state.phase = 0;
}

/**
 * @brief Process a single sample through Goertzel filter
 *
 * Implements: s(n) = coeff*s(n-1) - s(n-2) + input
 */
void DSPFeatureExtractor::goertzel_process(GoertzelState& state, int16_t sample) {
    // Normalize sample to floating point
    dsp_real_t x = (dsp_real_t)sample / 32768.0;

    // IIR filter difference equation
    state.s0 = x + state.coeff * state.s1 - state.s2;

    // Update delay line
    state.s2 = state.s1;
    state.s1 = state.s0;

    state.sample_count++;
}

/**
 * @brief Finalize Goertzel and extract magnitude/phase
 *
 * Computes the complex result from the final state variables
 */
void DSPFeatureExtractor::goertzel_finalize(GoertzelState& state) {
    dsp_real_t k = state.target_freq * state.sample_count / state.sample_rate;

    // Complex result: X = s1 - s2*exp(-j*2*pi*k/N)
    dsp_real_t w = 2.0 * M_PI * k / state.sample_count;
    dsp_real_t real_part = state.s1 - state.s2 * cos(w);
    dsp_real_t imag_part = state.s2 * sin(w);

    // Compute magnitude
    state.magnitude = sqrt(real_part * real_part + imag_part * imag_part);

    // Compute phase
    state.phase = atan2(imag_part, real_part);
}

/**
 * @brief Reset Goertzel state for next detection window
 */
void DSPFeatureExtractor::goertzel_reset(GoertzelState& state) {
    state.s0 = 0;
    state.s1 = 0;
    state.s2 = 0;
    state.sample_count = 0;
}

/**
 * @brief Compute Mel-scale frequency bands
 *
 * Mel-scale is a perceptually-motivated frequency scale where octaves are equally spaced.
 * This is the first step toward MFCC (Mel-Frequency Cepstral Coefficients).
 *
 * Conversion: mel = 2595 * log10(1 + f/700)
 */
void DSPFeatureExtractor::compute_mel_spectrum(
    const dsp_real_t* spectrum,
    uint16_t spectrum_len,
    dsp_real_t freq_res,
    dsp_real_t* mel_bands,
    uint8_t num_bands
) {
    if (spectrum_len == 0 || num_bands == 0) return;

    // Determine frequency range to cover (up to Nyquist or full spectrum)
    dsp_real_t max_freq = spectrum_len * freq_res;

    // Convert frequency limits to mel scale
    auto hz_to_mel = [](dsp_real_t hz) {
        return 2595.0 * log10(1.0 + hz / 700.0);
    };

    auto mel_to_hz = [](dsp_real_t mel) {
        return 700.0 * (pow(10.0, mel / 2595.0) - 1.0);
    };

    dsp_real_t min_mel = hz_to_mel(0);
    dsp_real_t max_mel = hz_to_mel(max_freq);

    // Create logarithmically-spaced mel band edges
    dsp_real_t* mel_edges = new dsp_real_t[num_bands + 2];
    for (uint8_t i = 0; i <= num_bands + 1; i++) {
        dsp_real_t mel_point = min_mel + (dsp_real_t)i / (num_bands + 1) * (max_mel - min_mel);
        mel_edges[i] = mel_to_hz(mel_point) / freq_res;  // Convert to bin index
    }

    // Create triangular filters centered at each mel band
    for (uint8_t band = 0; band < num_bands; band++) {
        dsp_real_t sum = 0.0;
        dsp_real_t left = mel_edges[band];
        dsp_real_t center = mel_edges[band + 1];
        dsp_real_t right = mel_edges[band + 2];

        for (uint16_t bin = (uint16_t)left; bin < spectrum_len && bin <= (uint16_t)right; bin++) {
            dsp_real_t weight = 0.0;

            if (bin < center) {
                // Left slope of triangle
                weight = (bin - left) / (center - left);
            } else {
                // Right slope of triangle
                weight = (right - bin) / (right - center);
            }

            if (weight > 0) {
                sum += weight * spectrum[bin];
            }
        }

        mel_bands[band] = sum;
    }

    delete[] mel_edges;
}
