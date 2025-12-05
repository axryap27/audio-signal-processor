# Audio Signal Processor - Architecture & Design

## System Overview

Real-time DSP audio analysis on ESP32 microcontroller with:
- **I2S Audio Input**: 44.1 kHz, 16-bit mono from INMP441 microphone
- **FFT Analysis**: 2048-point FFT (21.5 Hz frequency resolution)
- **Feature Extraction**: Spectral & temporal features + Goertzel note detection
- **Data Streaming**: JSON over serial @ 115200 baud

## Signal Processing Pipeline

```
I2S Microphone (44.1 kHz, 16-bit)
     ↓ DMA Buffer
Audio Frame (2048 samples = 46.4 ms)
     ↓
DC Removal + Hanning Window
     ↓
FFT (Cooley-Tukey, ~15-20 ms)
     ↓ Magnitude Spectrum
  ┌──┴──┐
  ↓     ↓
Spectral  Temporal
Features  Features
  ↓     ↓
  └──┬──┘
     ↓
Frequency Bands (16 logarithmic)
Peak Detection (local maxima)
Note Detection (Goertzel algorithm)
     ↓
JSON Serialization
     ↓
Serial Output (~1.5 KB per frame, 10 Hz rate)
```

## Key Components

### 1. I2S Audio Capture
- **Protocol**: I2S (Inter-IC Sound)
- **Microphone**: INMP441 digital microphone
- **Sample Rate**: 44.1 kHz (Nyquist: 22.05 kHz)
- **Bit Depth**: 16-bit signed
- **DMA Buffering**: 3 buffers × 1024 samples (prevents dropout during processing)
- **Pin Configuration**:
  - WS (L/R): GPIO 32
  - SD (DATA): GPIO 35
  - SCK (Clock): GPIO 33

### 2. Preprocessing
**DC Offset Removal**:
```
x_centered[n] = x[n] - mean(x)
```
Removes bias that skews frequency analysis.

**Hanning Window**:
```
w[n] = 0.5 * (1 - cos(2π*n/(N-1)))
x_windowed[n] = x_centered[n] * w[n]
```
Reduces spectral leakage by tapering signal at edges.

### 3. FFT Analysis
- **Algorithm**: Cooley-Tukey Radix-2 FFT (ArduinoFFT library)
- **Size**: 2048 points (log2 = 11)
- **Frequency Resolution**: 44100 / 2048 = 21.5 Hz/bin
- **Processing Time**: ~15-20 ms on ESP32 @ 240 MHz
- **Output**: 1024 positive frequency bins + magnitude spectrum

### 4. Spectral Features

#### Spectral Centroid
"Center of mass" of spectrum - indicates brightness:
```
centroid = Σ(f_i * M_i) / Σ(M_i)

< 2 kHz: Dark/bass-heavy
2-5 kHz: Warm/normal
> 5 kHz: Bright/presence
```

#### Spectral Spread
Bandwidth around centroid (like standard deviation):
```
spread = sqrt(Σ((f_i - centroid)² * M_i) / Σ(M_i))

< 500 Hz: Tonal (pure tone)
500-2000 Hz: Harmonic (instruments)
> 2000 Hz: Noisy (percussion)
```

#### Spectral Flatness
Tonal vs noise (Wiener entropy):
```
flatness = geometric_mean(M) / arithmetic_mean(M)

0.0-0.1: Pure tone
0.1-0.5: Musical
0.5-1.0: Noise
```

#### Spectral Rolloff (85%)
Frequency containing 85% of energy.

### 5. Temporal Features

#### Zero-Crossing Rate (ZCR)
Frequency of sign changes:
```
ZCR = (1/(N-1)) * Σ|sign(x[n]) - sign(x[n-1])|/2

< 0.1: Low-frequency, smooth (vowels)
0.1-0.2: Normal speech
0.2-0.4: High-frequency, consonants
> 0.4: Very rough (noise)
```

#### RMS Energy
Overall loudness:
```
E_rms = sqrt(mean(x[n]²))
0-1 normalized
```

#### Peak Amplitude
Maximum absolute sample value (headroom indicator).

### 6. Frequency Band Analysis
- **16 Logarithmic Bands** (60 Hz - 16 kHz)
- **Why Logarithmic?** Human hearing is logarithmic
- **Per-Band Processing**: Average magnitudes of FFT bins in band range
- **Smoothing**: 5-frame moving average (reduces noise jitter)

### 7. Peak Detection
Local maxima with hysteresis:
```cpp
if (spectrum[i] > threshold &&
    spectrum[i] > spectrum[i-1] &&
    spectrum[i] > spectrum[i+1] &&
    (i - last_peak) >= min_width)
    detect_peak();
```
- Finds dominant frequencies (harmonics, formants)
- Robust to noise floor

### 8. Goertzel Algorithm
Efficient single-frequency detection:
```
Complexity: O(N) vs FFT O(N log N)
Memory: O(1) vs FFT O(N)
```

**Implementation**:
- 12 chromatic note detectors (A3-G#5, semitone resolution)
- Each detector processes all 2048 samples in ~1-2 ms
- Returns magnitude and phase for each note
- Formula: `f = 440 * 2^(n/12)` Hz

**Why This Matters**: For detecting 12 specific frequencies, Goertzel is 10-100x faster than FFT!

### 9. Mel-Scale Spectrum
Perceptually-weighted frequency bands:
```
mel = 2595 * log10(1 + f/700)
```
Foundation for MFCC (Mel-Frequency Cepstral Coefficients).

## Performance Metrics

### Real-Time Processing (per frame)

| Component | Time | CPU Load |
|-----------|------|----------|
| I2S read | < 1 ms | DMA (0%) |
| FFT | 15-20 ms | 25% |
| Features | 2-3 ms | 5% |
| JSON | 1-2 ms | 2% |
| **Total** | **~25 ms** | **~32%** |

### Latency Analysis
```
Audio capture:      0-46 ms (sliding window)
Processing:         ~25 ms
Transmission:       ~10 ms (serial)
Total:             ~81 ms
```
Excellent for real-time interactive applications.

### Memory Usage
```
FFT buffers:       16 KB (fft_data_real/imag)
Audio buffer:      4 KB (mic_read_buffer)
Features:          ~1 KB (freq_bands, etc)
JSON doc:          ~2 KB (dynamic)
─────────────────────
Total:            ~23 KB / 160 KB available (14%)
```

## Data Format: JSON

Complete analysis snapshot per frame:
```json
{
  "timestamp_ms": 12345678,
  "frame_time_ms": 46,
  "spectral": {
    "centroid_hz": 2450.5,
    "spread_hz": 1200.3,
    "flatness": 0.45,
    "rolloff_85_hz": 4500.0
  },
  "temporal": {
    "zcr": 0.15,
    "rms_energy": 0.35,
    "peak_amplitude": 0.89
  },
  "freq_bands": [...],
  "peaks": [...],
  "note_detection": [...]
}
```

## Design Decisions

### Sample Rate: 44.1 kHz
Standard audio (CD quality). Nyquist: 22.05 kHz covers all audible frequencies.

### FFT Size: 2048
21.5 Hz resolution, ~46 ms latency. Sweet spot between frequency resolution and latency.

### Logarithmic Bands: 16 bands
Matches human hearing, good balance between detail and visualization.

### Goertzel for Notes
Only 12 frequencies → much faster than FFT. Real-time note tracking with low latency.

### 5-Frame Smoothing
Reduces noise jitter. ~50 ms latency (acceptable for visualization).

## Extension Points

1. **MFCC**: Apply DCT to mel spectrum for ML features
2. **Chromagram**: Map spectrum to musical notes
3. **Onset Detection**: Identify transients for BPM
4. **BPM Estimation**: Autocorrelation of beat energy
5. **WiFi Streaming**: Replace serial with UDP/WebSocket
6. **TensorFlow Lite**: Real-time genre/voice classification
