#include <Arduino.h>
#include <driver/i2s.h>
#include <arduinoFFT.h>
#include "dsp_features.h"
#include <ArduinoJson.h>

// ============================================================================
// CONFIGURATION
// ============================================================================

// I2S Microphone Pin Configuration
#define I2S_WS 32
#define I2S_SD 35
#define I2S_SCK 33

// Sample rate and FFT configuration
const uint16_t sample_rate = 44100;  // Hz
const uint8_t fft_sample_count_log2 = 11;
const uint16_t fft_sample_count = 1 << fft_sample_count_log2;  // 2048 samples
const dsp_real_t fft_sampling_freq = (dsp_real_t)sample_rate;
const uint16_t fft_freq_bin_count = fft_sample_count / 2;
const float fft_freq_step = fft_sampling_freq / fft_sample_count;

// I2S configuration constants
const i2s_port_t i2s_port = I2S_NUM_1;
const i2s_bits_per_sample_t i2s_bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT;
const uint8_t i2s_bytes_per_sample = i2s_bits_per_sample / 8;
const uint16_t i2s_read_size_bytes = fft_sample_count * i2s_bytes_per_sample;
const uint16_t i2s_buffer_size_samples = 1024;
const uint16_t i2s_buffer_count = 3;
const int i2s_queue_len = 16;

// Frequency band configuration (16 logarithmic bands for visualization)
const uint8_t freq_band_count = 16;
const float freq_min_hz = 60.0;
const float freq_max_hz = 16000.0;

// Goertzel detection for musical notes
const uint8_t num_note_detectors = 12;  // Chromatic scale (C, C#, D, D#, E, F, F#, G, G#, A, A#, B)
const float base_note_freq = 440.0;  // A4 reference frequency

// Smoothing configuration
const uint16_t smoothing_buf_len = 5;
const unsigned long display_update_interval = 100;  // milliseconds

// ============================================================================
// DATA STRUCTURES & BUFFERS
// ============================================================================

// I2S audio buffers
int16_t mic_read_buffer[fft_sample_count] = {0};
QueueHandle_t i2s_queue = nullptr;

// FFT computation buffers
typedef float fft_real_t;
fft_real_t fft_data_real[fft_sample_count] = {0.0};
fft_real_t fft_data_imag[fft_sample_count] = {0.0};
ArduinoFFT<fft_real_t> fft(fft_data_real, fft_data_imag, fft_sample_count, fft_sampling_freq);

// Frequency band processing
float freq_band_end_hz[freq_band_count];
float freq_band_amp[freq_band_count];
float freq_band_avg[freq_band_count];
float freq_band_buf[freq_band_count][smoothing_buf_len];
float freq_band_smoothed[freq_band_count];
uint16_t buf_counter = 0;

// Peak detection configuration
PeakDetectionConfig peak_config = {
    .threshold = 0.05,
    .hysteresis = 0.02,
    .min_width = 5
};
uint16_t detected_peaks[64];

// Goertzel state for note detection
GoertzelState note_detectors[num_note_detectors];

// Feature containers
SpectralFeatures spectral_features;
TemporalFeatures temporal_features;

// Timing
unsigned long last_display_update = 0;
unsigned long frame_count = 0;
unsigned long total_processing_time = 0;

// ============================================================================
// FUNCTION PROTOTYPES
// ============================================================================

bool setup_i2s_mic();
void apply_hanning_window(fft_real_t* data, uint16_t len);
float calc_buf_mean(float* buf, uint16_t len);
void initialize_frequency_bands();
void initialize_note_detectors();
float get_note_frequency(int semitone_from_a4);
void transmit_data_json();
void handle_serial_commands();

// ============================================================================
// I2S MICROPHONE SETUP
// ============================================================================

bool setup_i2s_mic() {
    esp_err_t i2s_error;

    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = sample_rate,
        .bits_per_sample = i2s_bits_per_sample,
        .channel_format = I2S_CHANNEL_FMT_ONLY_RIGHT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = i2s_buffer_count,
        .dma_buf_len = i2s_buffer_size_samples,
        .use_apll = false
    };

    i2s_error = i2s_driver_install(i2s_port, &i2s_config, i2s_queue_len, &i2s_queue);
    if (i2s_error) {
        log_e("Failed to start i2s driver. ESP error: %s (%x)",
              esp_err_to_name(i2s_error), i2s_error);
        return false;
    }

    if (i2s_queue == nullptr) {
        log_e("Failed to setup i2s event queue.");
        return false;
    }

    i2s_pin_config_t i2s_pin_config = {
        .bck_io_num = I2S_SCK,
        .ws_io_num = I2S_WS,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = I2S_SD
    };

    i2s_error = i2s_set_pin(i2s_port, &i2s_pin_config);
    if (i2s_error) {
        log_e("Failed to set i2s pins. ESP error: %s (%x)",
              esp_err_to_name(i2s_error), i2s_error);
        return false;
    }

    return true;
}

// ============================================================================
// WINDOWING AND FEATURE EXTRACTION
// ============================================================================

void apply_hanning_window(fft_real_t* data, uint16_t len) {
    for (uint16_t i = 0; i < len; i++) {
        float window = 0.5 * (1.0 - cos(2.0 * PI * i / (len - 1)));
        data[i] *= window;
    }
}

float calc_buf_mean(float* buf, uint16_t len) {
    float total = 0.0;
    for (uint16_t i = 0; i < len; i++) {
        total += buf[i];
    }
    return total / ((float)len);
}

// ============================================================================
// FREQUENCY BAND INITIALIZATION
// ============================================================================

void initialize_frequency_bands() {
    // Create logarithmically-spaced frequency bands
    for (int i = 0; i < freq_band_count; i++) {
        freq_band_end_hz[i] = freq_min_hz * pow(freq_max_hz / freq_min_hz,
                                                 (float)(i + 1) / freq_band_count);

        // Frequency-dependent amplification (compensate for human hearing curve)
        if (i < 2) {
            freq_band_amp[i] = 80.0;   // Bass boost
        } else if (i < 6) {
            freq_band_amp[i] = 120.0;  // Low-mid
        } else if (i < 10) {
            freq_band_amp[i] = 150.0;  // Mid
        } else if (i < 13) {
            freq_band_amp[i] = 180.0;  // High-mid
        } else {
            freq_band_amp[i] = 200.0;  // Treble
        }
    }

    // Initialize smoothing buffers
    for (uint8_t i = 0; i < freq_band_count; i++) {
        for (uint8_t j = 0; j < smoothing_buf_len; j++) {
            freq_band_buf[i][j] = 0.0;
        }
    }

    log_d("Initialized %d frequency bands (%.0f - %.0f Hz)",
          freq_band_count, freq_min_hz, freq_max_hz);
}

// ============================================================================
// NOTE DETECTION INITIALIZATION
// ============================================================================

float get_note_frequency(int semitone_from_a4) {
    // A4 = 440 Hz is the reference
    // Each semitone is 2^(1/12) times the frequency
    return base_note_freq * pow(2.0, semitone_from_a4 / 12.0);
}

void initialize_note_detectors() {
    // Notes relative to A4 (index 0 = A3, index 9 = A4, etc)
    // Cover 2 octaves: A3 to G#5
    int semitone_offsets[] = {-12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1};

    for (int i = 0; i < num_note_detectors; i++) {
        float freq = get_note_frequency(semitone_offsets[i]);
        DSPFeatureExtractor::goertzel_init(note_detectors[i], freq, sample_rate, fft_sample_count);
    }

    log_d("Initialized %d note detectors", num_note_detectors);
}

// ============================================================================
// DATA TRANSMISSION
// ============================================================================

void transmit_data_json() {
    // Calculate processing time
    unsigned long frame_time = millis() - last_display_update;

    // Create JSON document
    StaticJsonDocument<2048> doc;

    // Add timestamp and frame info
    doc["timestamp_ms"] = millis();
    doc["frame_time_ms"] = frame_time;
    doc["frame_count"] = frame_count;

    // Add spectral features
    JsonObject spectral = doc.createNestedObject("spectral");
    spectral["centroid_hz"] = spectral_features.centroid;
    spectral["spread_hz"] = spectral_features.spread;
    spectral["flatness"] = spectral_features.flatness;
    spectral["rolloff_85_hz"] = spectral_features.rolloff_85;

    // Add temporal features
    JsonObject temporal = doc.createNestedObject("temporal");
    temporal["zcr"] = temporal_features.zcr;
    temporal["rms_energy"] = temporal_features.rms_energy;
    temporal["peak_amplitude"] = temporal_features.peak_amplitude;

    // Add frequency bands
    JsonArray bands = doc.createNestedArray("freq_bands");
    for (uint8_t i = 0; i < freq_band_count; i++) {
        JsonObject band = bands.createNestedObject();
        band["index"] = i;
        band["freq_min"] = (i == 0) ? freq_min_hz : freq_band_end_hz[i - 1];
        band["freq_max"] = freq_band_end_hz[i];
        band["magnitude"] = freq_band_smoothed[i];
    }

    // Add detected peaks
    uint16_t peak_count = DSPFeatureExtractor::detect_peaks(
        fft_data_real, fft_freq_bin_count, peak_config, detected_peaks, 64);

    JsonArray peaks = doc.createNestedArray("peaks");
    for (uint16_t i = 0; i < peak_count && i < 10; i++) {  // Limit to top 10 peaks
        JsonObject peak = peaks.createNestedObject();
        peak["bin"] = detected_peaks[i];
        peak["frequency_hz"] = detected_peaks[i] * fft_freq_step;
        peak["magnitude"] = fft_data_real[detected_peaks[i]];
    }

    // Add note detection results
    JsonArray notes = doc.createNestedArray("note_detection");
    const char* note_names[] = {"A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"};
    for (uint8_t i = 0; i < num_note_detectors; i++) {
        JsonObject note = notes.createNestedObject();
        note["note"] = note_names[i];
        note["magnitude"] = note_detectors[i].magnitude;
        note["confidence"] = note_detectors[i].magnitude;  // Simplified confidence metric
    }

    // Serialize and transmit
    serializeJson(doc, Serial);
    Serial.println();
}

void handle_serial_commands() {
    if (Serial.available()) {
        String cmd = Serial.readStringUntil('\n');
        cmd.trim();

        if (cmd == "CONFIG") {
            // Return configuration info
            StaticJsonDocument<512> config;
            config["sample_rate"] = sample_rate;
            config["fft_size"] = fft_sample_count;
            config["freq_bins"] = fft_freq_bin_count;
            config["freq_resolution"] = fft_freq_step;
            config["freq_bands"] = freq_band_count;
            serializeJson(config, Serial);
            Serial.println();
        } else if (cmd == "HELP") {
            Serial.println("Commands: CONFIG, HELP");
        }
    }
}

// ============================================================================
// SETUP AND MAIN LOOP
// ============================================================================

void setup() {
    Serial.begin(115200);
    delay(1000);  // Wait for serial to stabilize

    log_i("Audio Signal Processor starting...");
    log_i("Sample Rate: %d Hz", sample_rate);
    log_i("FFT Size: %d", fft_sample_count);

    if (!setup_i2s_mic()) {
        log_e("I2S setup failed!");
        while (1) delay(1000);  // Halt
    }

    initialize_frequency_bands();
    initialize_note_detectors();

    log_i("Setup complete. Streaming data as JSON...");
}

void loop() {
    // Handle serial commands
    handle_serial_commands();

    // Read audio data from I2S
    esp_err_t i2s_error = ESP_OK;
    size_t i2s_bytes_read = 0;

    i2s_error = i2s_read(i2s_port, mic_read_buffer, i2s_read_size_bytes,
                         &i2s_bytes_read, portMAX_DELAY);

    if (i2s_error) {
        log_e("i2s_read failure: %s", esp_err_to_name(i2s_error));
        return;
    }

    if (i2s_bytes_read != i2s_read_size_bytes) {
        log_w("i2s_read: unexpected byte count %d (expected %d)",
              i2s_bytes_read, i2s_read_size_bytes);
        return;
    }

    // Check if we have valid signal (not all zeros)
    bool has_signal = false;
    for (uint16_t i = 0; i < fft_sample_count; i += 100) {
        if (abs(mic_read_buffer[i]) > 100) {
            has_signal = true;
            break;
        }
    }

    if (has_signal) {
        // Compute DC offset
        int32_t block_sum = 0;
        for (uint16_t i = 0; i < fft_sample_count; i++) {
            block_sum += mic_read_buffer[i];
        }
        int16_t block_avg = block_sum / fft_sample_count;

        // Prepare FFT input (DC offset removal + windowing)
        const fft_real_t int16_max_inv = 1.0f / 32768.0f;
        for (uint32_t i = 0; i < fft_sample_count; i++) {
            int16_t v = mic_read_buffer[i] - block_avg;
            fft_data_real[i] = int16_max_inv * v;
            fft_data_imag[i] = 0.0f;
        }

        apply_hanning_window(fft_data_real, fft_sample_count);

        // Compute FFT
        unsigned long start_time = micros();
        fft.compute(FFTDirection::Forward);
        fft.complexToMagnitude();
        unsigned long fft_time = micros() - start_time;

        // Compute spectral features
        spectral_features = DSPFeatureExtractor::compute_spectral_centroid(
            fft_data_real, fft_freq_bin_count, fft_freq_step);

        // Compute temporal features
        temporal_features = DSPFeatureExtractor::compute_temporal_features(
            mic_read_buffer, fft_sample_count);

        // Compute frequency band averages
        uint32_t counter = max(1, (int)(freq_min_hz / fft_freq_step));
        for (uint8_t bin_num = 0; bin_num < freq_band_count; bin_num++) {
            freq_band_avg[bin_num] = 0;
            uint32_t bin_count = 0;

            while (counter * fft_freq_step < freq_band_end_hz[bin_num] &&
                   counter < fft_freq_bin_count) {
                freq_band_avg[bin_num] += fft_data_real[counter];
                counter++;
                bin_count++;
            }

            if (bin_count > 0) {
                freq_band_avg[bin_num] /= (float)bin_count;
            }
        }

        // Update smoothing buffers
        for (uint8_t i = 0; i < freq_band_count; i++) {
            freq_band_buf[i][buf_counter] = freq_band_avg[i];
        }
        buf_counter = (buf_counter + 1) % smoothing_buf_len;

        // Calculate smoothed values
        for (uint8_t i = 0; i < freq_band_count; i++) {
            freq_band_smoothed[i] = calc_buf_mean(freq_band_buf[i], smoothing_buf_len);
        }

        // Process audio through note detectors
        for (uint8_t i = 0; i < num_note_detectors; i++) {
            DSPFeatureExtractor::goertzel_reset(note_detectors[i]);
        }

        for (uint16_t i = 0; i < fft_sample_count; i++) {
            for (uint8_t j = 0; j < num_note_detectors; j++) {
                DSPFeatureExtractor::goertzel_process(note_detectors[j], mic_read_buffer[i]);
            }
        }

        for (uint8_t i = 0; i < num_note_detectors; i++) {
            DSPFeatureExtractor::goertzel_finalize(note_detectors[i]);
        }

    } else {
        // Fade out visualization if no signal
        for (uint8_t i = 0; i < freq_band_count; i++) {
            freq_band_smoothed[i] *= 0.9;
        }
    }

    // Transmit data periodically
    unsigned long current_time = millis();
    if (current_time - last_display_update >= display_update_interval) {
        transmit_data_json();
        last_display_update = current_time;
        frame_count++;
    }
}
