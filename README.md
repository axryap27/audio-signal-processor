# Audio Signal Processor

Real-time DSP audio analysis system for ESP32 with advanced feature extraction and JSON data streaming.

## Features

- **Real-time FFT**: 2048-point, 21.5 Hz resolution
- **Spectral Features**: Centroid, spread, flatness, rolloff
- **Temporal Features**: Zero-crossing rate, RMS energy, peak detection
- **Note Detection**: 12-semitone chromatic scale (Goertzel algorithm)
- **Frequency Bands**: 16 logarithmic bands (60 Hz - 16 kHz)
- **JSON Streaming**: ~10 Hz updates over serial

## Hardware

- ESP32 DevKit V1
- INMP441 I2S digital microphone
- USB cable (power + serial)

## Quick Start

### Pinout
```
INMP441    ESP32
WS    �    GPIO 32
SD    �    GPIO 35
SCK   �    GPIO 33
GND   �    GND
3V3   �    3V3
```

### Build
```bash
cd audio-signal-processor
pio run --target upload
```

### Monitor
```bash
pio device monitor --baud 115200
```

## Output Format

JSON with spectral, temporal, and frequency band features:

```json
{
  "timestamp_ms": 12345678,
  "spectral": {"centroid_hz": 2450.5, ...},
  "temporal": {"zcr": 0.15, "rms_energy": 0.35, ...},
  "freq_bands": [...],
  "peaks": [...],
  "note_detection": [...]
}
```

## System Architecture

```
Audio Input (44.1 kHz)
    ↓
Audio Frame Buffer (2048 samples)
    ↓
Preprocessing (DC removal, windowing)
    ↓
FFT Analysis (2048-point)
    ↓
Feature Extraction (48 features)
    ├─ Spectral: centroid, spread, flatness, rolloff
    ├─ Temporal: ZCR, RMS energy, peak amplitude
    ├─ Frequency: 16 logarithmic bands (60 Hz - 16 kHz)
    ├─ MFCC: 13 coefficients
    └─ Chroma: 12 note bins
    ↓
ML Classification (TensorFlow Lite)
    ├─ Normalize features
    └─ Genre prediction (10 genres)
    ↓
JSON Output (serial @ 115200 baud)
```

### ML Pipeline & Training

1. **Feature Extraction** (`ml/feature_extraction.py`)
   - Extract 48 audio features from audio files
   - Output: CSV with feature vectors

2. **Model Training** (`ml/train_model.py`)
   - Train neural network on extracted features
   - Split: 64% train, 16% validation, 20% test (subject to change)
   - Output: TensorFlow Lite model for ESP32

3. **Deployment**
   - Quantized model runs on ESP32 in real-time (might change)
   - Classifies audio into 10 music genres
     
4. **Training Results and Initial Data**
   - Correlation Matrix & Training History:
   <img width="750" height="600" alt="image" src="https://github.com/user-attachments/assets/ef1503b2-3940-4337-aa76-f16736c97428" />
   <img width="750" height="260" alt="image" src="https://github.com/user-attachments/assets/aded7aa8-3372-4c81-901d-357fef3bc644" />


## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - Technical deep-dive
- [FEATURES_REFERENCE.md](FEATURES_REFERENCE.md) - Feature explanations
- [DSP Features API](src/dsp_features.h) - API documentation

## License

MIT
