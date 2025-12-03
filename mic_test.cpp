#include <Arduino.h>
#include <driver/i2s.h>

// sample rate
const uint16_t sample_rate = 44100; // Unit: Hz

// i2s hardware constants
const i2s_port_t i2s_port = I2S_NUM_1;
const int kI2S_PinClk = 0;
const int kI2S_PinData = 34;

// Connections to INMP441 I2S microphone
#define I2S_WS 32
#define I2S_SD 35
#define I2S_SCK 33

// i2s constants
const uint16_t i2s_buf_len = 2048;
const i2s_bits_per_sample_t i2s_bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT;
const uint8_t i2s_bytes_per_sample = i2s_bits_per_sample / 8;
const uint16_t i2s_read_size_bytes = i2s_buf_len * i2s_bytes_per_sample;
const uint16_t i2s_buffer_size_samples = 1024;
const uint16_t i2s_buffer_count = 3;
const int i2s_queue_len = 16;

// i2s variables
int16_t mic_read_buffer[i2s_buf_len] = {0};
QueueHandle_t i2s_queue = nullptr;

bool setup_i2s_mic()
{
    esp_err_t i2s_error;

    // i2s configuration for sampling 16 bit mono audio data
    //
    // Notes related to i2s.c:
    // - 'dma_buf_len', i.e. the number of samples in each DMA buffer, is limited to 1024
    // - 'dma_buf_len' * 'bytes_per_sample' is limted to 4092
    // - 'I2S_CHANNEL_FMT_ONLY_RIGHT' means "mono", i.e. only one channel to be received via i2s (must use RIGHT!!!)

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

    if (i2s_error)
    {
        log_e("Failed to start i2s driver. ESP error: %s (%x)", esp_err_to_name(i2s_error), i2s_error);
        return false;
    }

    if (i2s_queue == nullptr)
    {
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

    if (i2s_error)
    {
        log_e("Failed to set i2s pins. ESP error: %s (%x)", esp_err_to_name(i2s_error), i2s_error);
        return false;
    }

    return true;
}

void setup() {
    Serial.begin(115200);
    if (!setup_i2s_mic()) {
        log_e("I2S setup failed!");
    }

    log_d("Setup successfully completed.");
}

void loop() {
    esp_err_t i2s_error = ESP_OK;
    size_t i2s_bytes_read = 0;
    i2s_error = i2s_read(i2s_port, mic_read_buffer, i2s_read_size_bytes, &i2s_bytes_read, portMAX_DELAY);

    // Check i2s error state after reading
    if (i2s_error) {
        log_e("i2s_read failure. ESP error: %s (%x)", esp_err_to_name(i2s_error), i2s_error);
    }

    // Check whether right number of bytes has been read
    if (i2s_bytes_read != i2s_read_size_bytes) {
        log_w("i2s_read unexpected number of bytes: %d", i2s_bytes_read);
    }

    // Compute sum of the current sample block
    int32_t block_sum = 0;
    for (uint16_t ii = 0; ii < i2s_buf_len; ii++) {
        block_sum += abs(mic_read_buffer[ii]);
    }
    // Compute average value for the current sample block
    int16_t block_avg = block_sum / i2s_buf_len;
    Serial.println(block_avg);
}

