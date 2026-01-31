#include "dsp_features.h"

// ── Spectral Analysis ───────────────────────────────────────────────

/*
 * Spectral centroid is the "center of mass" of the spectrum — a single
 * number that captures brightness.  While we're looping through the bins
 * we also grab spread, 85 % rolloff, and flatness so we only pay for
 * one pass over the data.
 */
SpectralFeatures DSPFeatureExtractor::compute_spectral_centroid(
    const dsp_real_t* spectrum,
    uint16_t spectrum_len,
    dsp_real_t freq_res)
{
    SpectralFeatures features = {0};
    if (spectrum_len == 0) return features;

    // First pass — accumulate weighted energy for centroid + total energy
    dsp_real_t weighted_sum = 0.0;
    dsp_real_t total_energy = 0.0;

    for (uint16_t i = 0; i < spectrum_len; i++) {
        dsp_real_t freq = i * freq_res;
        weighted_sum += freq * spectrum[i];
        total_energy += spectrum[i];
    }

    features.centroid = (total_energy > 0) ? weighted_sum / total_energy : 0;

    // Second pass — spread (std-dev around centroid) and flatness (geo/arith mean)
    dsp_real_t spread_sum = 0.0;
    dsp_real_t log_sum    = 0.0;

    for (uint16_t i = 0; i < spectrum_len; i++) {
        dsp_real_t freq = i * freq_res;
        dsp_real_t diff = freq - features.centroid;
        spread_sum += (diff * diff) * spectrum[i];

        if (spectrum[i] > 1e-10)
            log_sum += log(spectrum[i]);
    }

    features.spread = (total_energy > 0)
        ? sqrt(spread_sum / total_energy)
        : 0;

    // Flatness: geometric_mean / arithmetic_mean, clamped to [0, 1]
    dsp_real_t arith_mean = total_energy / spectrum_len;
    if (arith_mean > 1e-10) {
        dsp_real_t geo_mean = exp(log_sum / spectrum_len);
        features.flatness = fmin(1.0, fmax(0.0, geo_mean / arith_mean));
    }

    // Rolloff — the bin at which 85 % of the total energy has been reached
    dsp_real_t threshold   = 0.85 * total_energy;
    dsp_real_t accumulated = 0.0;

    for (uint16_t i = 0; i < spectrum_len; i++) {
        accumulated += spectrum[i];
        if (accumulated >= threshold) {
            features.rolloff_85 = i * freq_res;
            break;
        }
    }

    return features;
}

/*
 * Standalone spectral flatness — useful when you don't need the full
 * SpectralFeatures struct and just want a quick tonal-vs-noise number.
 * Returns 0 for strongly tonal signals, ~1 for white noise.
 */
dsp_real_t DSPFeatureExtractor::compute_spectral_flatness(
    const dsp_real_t* spectrum,
    uint16_t spectrum_len)
{
    if (spectrum_len == 0) return 0;

    dsp_real_t log_sum   = 0.0;
    dsp_real_t arith_sum = 0.0;

    for (uint16_t i = 0; i < spectrum_len; i++) {
        arith_sum += spectrum[i];
        if (spectrum[i] > 1e-10)
            log_sum += log(spectrum[i]);
    }

    dsp_real_t arith_mean = arith_sum / spectrum_len;
    if (arith_mean > 1e-10) {
        dsp_real_t geo_mean = exp(log_sum / spectrum_len);
        return fmin(1.0, fmax(0.0, geo_mean / arith_mean));
    }

    return 0;
}

// ── Time-Domain Features ────────────────────────────────────────────

/*
 * Zero-crossing rate — how often the waveform flips sign.
 * Speech consonants and noise sit high, bass and vowels sit low.
 */
dsp_real_t DSPFeatureExtractor::compute_zcr(
    const int16_t* signal,
    uint16_t signal_len)
{
    if (signal_len < 2) return 0;

    uint32_t crossings = 0;

    for (uint16_t i = 1; i < signal_len; i++) {
        bool curr_neg = signal[i] < 0;
        bool prev_neg = signal[i - 1] < 0;
        if (curr_neg != prev_neg)
            crossings++;
    }

    return (dsp_real_t)crossings / (signal_len - 1);
}

/*
 * RMS energy — overall loudness of the frame.
 * We accumulate in int64 to avoid overflow on 16-bit samples.
 */
dsp_real_t DSPFeatureExtractor::compute_rms_energy(
    const int16_t* signal,
    uint16_t signal_len)
{
    if (signal_len == 0) return 0;

    int64_t sum_sq = 0;
    for (uint16_t i = 0; i < signal_len; i++) {
        int32_t s = signal[i];
        sum_sq += s * s;
    }

    return sqrt((dsp_real_t)sum_sq / signal_len);
}

/*
 * Bundle ZCR, RMS, and peak amplitude into one call so the caller
 * doesn't have to iterate the buffer three separate times.
 */
TemporalFeatures DSPFeatureExtractor::compute_temporal_features(
    const int16_t* signal,
    uint16_t signal_len)
{
    TemporalFeatures features = {0};
    if (signal_len == 0) return features;

    features.zcr        = compute_zcr(signal, signal_len);
    features.rms_energy  = compute_rms_energy(signal, signal_len);

    // Peak amplitude — use int32 so abs(INT16_MIN) doesn't overflow
    int32_t peak = 0;
    for (uint16_t i = 0; i < signal_len; i++) {
        int32_t magnitude = abs((int32_t)signal[i]);
        if (magnitude > peak)
            peak = magnitude;
    }
    features.peak_amplitude = (dsp_real_t)peak / 32768.0;

    return features;
}

// ── Peak Detection ──────────────────────────────────────────────────

/*
 * Simple local-maximum finder with a minimum-spacing guard.
 * A bin counts as a peak when it exceeds the threshold and is
 * strictly greater than both of its immediate neighbours.
 */
uint16_t DSPFeatureExtractor::detect_peaks(
    const dsp_real_t* spectrum,
    uint16_t spectrum_len,
    const PeakDetectionConfig& config,
    uint16_t* peaks_out,
    uint16_t peaks_out_len)
{
    if (spectrum_len < 3) return 0;

    uint16_t count     = 0;
    uint16_t last_peak = 0;

    for (uint16_t i = 1; i < spectrum_len - 1; i++) {
        bool above_threshold = spectrum[i] > config.threshold;
        bool local_max       = spectrum[i] > spectrum[i - 1]
                            && spectrum[i] > spectrum[i + 1];
        bool far_enough      = (i - last_peak) >= config.min_width;

        if (above_threshold && local_max && far_enough) {
            if (count >= peaks_out_len) break;
            peaks_out[count++] = i;
            last_peak = i;
        }
    }

    return count;
}

// ── Goertzel Single-Frequency Detector ──────────────────────────────

/*
 * The Goertzel algorithm detects a single frequency with O(N) work,
 * whereas a full FFT costs O(N log N).  Handy when you only care
 * about one tone (e.g. a tuning reference or DTMF digit).
 *
 * We pre-compute the IIR coefficient 2*cos(2*pi*k/N) where k is the
 * fractional bin index of the target frequency.
 */
void DSPFeatureExtractor::goertzel_init(
    GoertzelState& state,
    dsp_real_t target_freq,
    dsp_real_t sample_rate,
    uint32_t sample_window)
{
    state.target_freq  = target_freq;
    state.sample_rate  = sample_rate;
    state.sample_count = 0;

    dsp_real_t k = target_freq * sample_window / sample_rate;
    state.coeff  = 2.0 * cos(2.0 * M_PI * k / sample_window);

    state.s0 = 0;
    state.s1 = 0;
    state.s2 = 0;
    state.magnitude = 0;
    state.phase     = 0;
}

/*
 * Feed one sample into the IIR: s[n] = x + coeff*s[n-1] - s[n-2]
 */
void DSPFeatureExtractor::goertzel_process(GoertzelState& state, int16_t sample) {
    dsp_real_t x = (dsp_real_t)sample / 32768.0;

    state.s0 = x + state.coeff * state.s1 - state.s2;
    state.s2 = state.s1;
    state.s1 = state.s0;
    state.sample_count++;
}

/*
 * After all samples have been fed in, recover the magnitude and phase
 * from the final filter state.
 */
void DSPFeatureExtractor::goertzel_finalize(GoertzelState& state) {
    dsp_real_t w = 2.0 * M_PI * state.target_freq
                 * state.sample_count / state.sample_rate;
    w /= state.sample_count;  // normalise to per-sample angular freq

    dsp_real_t re = state.s1 - state.s2 * cos(w);
    dsp_real_t im = state.s2 * sin(w);

    state.magnitude = sqrt(re * re + im * im);
    state.phase     = atan2(im, re);
}

void DSPFeatureExtractor::goertzel_reset(GoertzelState& state) {
    state.s0 = 0;
    state.s1 = 0;
    state.s2 = 0;
    state.sample_count = 0;
}

// ── Mel-Scale Spectrum ──────────────────────────────────────────────

/*
 * Map a linear FFT spectrum onto perceptually-spaced mel bands using
 * overlapping triangular filters.  This is the standard first step
 * toward MFCCs.
 *
 * mel  = 2595 * log10(1 + f/700)
 * f    = 700 * (10^(mel/2595) - 1)
 *
 * We keep the mel-edge array on the stack (num_bands is a uint8_t,
 * so worst case is 257 floats ≈ 1 KB — well within ESP32 stack).
 */
void DSPFeatureExtractor::compute_mel_spectrum(
    const dsp_real_t* spectrum,
    uint16_t spectrum_len,
    dsp_real_t freq_res,
    dsp_real_t* mel_bands,
    uint8_t num_bands)
{
    if (spectrum_len == 0 || num_bands == 0) return;

    dsp_real_t max_freq = spectrum_len * freq_res;

    // Hz ↔ mel conversions
    auto hz_to_mel = [](dsp_real_t hz) {
        return 2595.0 * log10(1.0 + hz / 700.0);
    };
    auto mel_to_hz = [](dsp_real_t mel) {
        return 700.0 * (pow(10.0, mel / 2595.0) - 1.0);
    };

    dsp_real_t min_mel = hz_to_mel(0);
    dsp_real_t max_mel = hz_to_mel(max_freq);

    // Build triangular filter edges on the stack instead of the heap
    constexpr uint16_t MAX_EDGES = 257;  // uint8_t max + 2
    dsp_real_t mel_edges[MAX_EDGES];
    uint16_t edge_count = (uint16_t)num_bands + 2;

    for (uint16_t i = 0; i < edge_count; i++) {
        dsp_real_t mel_pt = min_mel + (dsp_real_t)i / (num_bands + 1) * (max_mel - min_mel);
        mel_edges[i] = mel_to_hz(mel_pt) / freq_res;  // convert to bin index
    }

    // Apply each triangular filter to the spectrum
    for (uint8_t band = 0; band < num_bands; band++) {
        dsp_real_t left   = mel_edges[band];
        dsp_real_t center = mel_edges[band + 1];
        dsp_real_t right  = mel_edges[band + 2];
        dsp_real_t sum    = 0.0;

        for (uint16_t bin = (uint16_t)left; bin < spectrum_len && bin <= (uint16_t)right; bin++) {
            dsp_real_t weight = (bin < center)
                ? (bin - left)  / (center - left)   // rising slope
                : (right - bin) / (right - center); // falling slope

            if (weight > 0)
                sum += weight * spectrum[bin];
        }

        mel_bands[band] = sum;
    }
}

// ── MFCCs ───────────────────────────────────────────────────────────

/*
 * Compute Mel-Frequency Cepstral Coefficients:
 *   1. Run compute_mel_spectrum to get mel-scaled energy bands
 *   2. Take the log of each band (with a floor to avoid log(0))
 *   3. Apply a Type-II DCT to compress into cepstral coefficients
 *
 * The first ~13 coefficients capture the overall shape of the
 * spectrum, which is what most audio classifiers care about.
 */
void DSPFeatureExtractor::compute_mfcc(
    const dsp_real_t* spectrum,
    uint16_t spectrum_len,
    dsp_real_t freq_res,
    dsp_real_t* mfcc_out,
    uint8_t num_coefficients,
    uint8_t num_mel_bands)
{
    if (num_coefficients == 0 || num_mel_bands == 0) return;

    // Step 1: compute mel spectrum into a stack buffer
    constexpr uint8_t MAX_MEL_BANDS = 40;
    uint8_t bands = (num_mel_bands > MAX_MEL_BANDS) ? MAX_MEL_BANDS : num_mel_bands;
    dsp_real_t mel_bands[MAX_MEL_BANDS];

    compute_mel_spectrum(spectrum, spectrum_len, freq_res, mel_bands, bands);

    // Step 2: take the log of each mel band
    for (uint8_t i = 0; i < bands; i++) {
        mel_bands[i] = log(fmax(mel_bands[i], 1e-10));
    }

    // Step 3: DCT-II to get cepstral coefficients
    for (uint8_t k = 0; k < num_coefficients; k++) {
        dsp_real_t sum = 0.0;
        for (uint8_t n = 0; n < bands; n++) {
            sum += mel_bands[n] * cos(M_PI * k * (2.0 * n + 1.0) / (2.0 * bands));
        }
        mfcc_out[k] = sum;
    }
}
