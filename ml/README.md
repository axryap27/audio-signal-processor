# Genre Classification - ML Pipeline

Real-time music genre classification using neural networks deployed on ESP32 via TensorFlow Lite.

## Overview

```
GTZAN Dataset (1000 songs, 10 genres)
    ↓
Extract Features (feature_extraction.py)
  - Spectral: centroid, spread, flatness, rolloff
  - Temporal: ZCR, RMS energy, peak amplitude
  - Frequency: 16 logarithmic bands
  - MFCC: 13 coefficients
  - Chroma: 12 chromatic bins
    ↓ (~58 features per song)
Train Neural Network (train_model.py)
  - 256→128→64→32 neurons
  - BatchNorm + Dropout
  - Adam optimizer
    ↓
TensorFlow Lite Model (~500 KB)
    ↓
Deploy on ESP32 + Real-time Classification
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download GTZAN Dataset

Option A: Manual download
```bash
wget http://opihi.cs.uvic.ca/sound/genres.tar.gz
tar -xzf genres.tar.gz
mv genres gtzan
```

Option B: Automated (Python)
```python
# See download_gtzan.py if available
```

### 3. Extract Features
```bash
python feature_extraction.py
```

This will:
- Process all 1000 GTZAN audio files
- Extract 58 features per song
- Save to `gtzan_features.csv`

### 4. Train Model
```bash
python train_model.py
```

This will:
- Load features from CSV
- Split into train/val/test (60/20/20)
- Train neural network (100 epochs max)
- Evaluate on test set (~80-85% accuracy expected)
- Save model files:
  - `genre_model.h5` - Keras model
  - `genre_model.tflite` - TensorFlow Lite (for ESP32)
  - `model_metadata.json` - Feature scaling params
  - `plots/` - Training curves and confusion matrix

## File Structure

```
ml/
├── feature_extraction.py    # Extract audio features from GTZAN
├── train_model.py          # Train neural network
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── gtzan/                 # GTZAN dataset (download separately)
├── gtzan_features.csv     # Extracted features (generated)
├── genre_model.h5         # Trained Keras model (generated)
├── genre_model.tflite     # TFLite for ESP32 (generated)
└── model_metadata.json    # Feature scaling params (generated)
```

## Feature Engineering

### Spectral Features
- **Centroid**: "Brightness" of sound (Hz)
- **Spread**: Bandwidth around centroid (Hz)
- **Flatness**: Tonal vs noise (0-1)
- **Rolloff**: Frequency at 85% energy (Hz)

### Temporal Features
- **ZCR**: Zero-crossing rate (0-1)
- **RMS Energy**: Loudness (0-1)
- **Peak Amplitude**: Maximum sample (0-1)

### Frequency Domain
- **16 Logarithmic Bands**: Matches human hearing
- **13 MFCC Coefficients**: Mel-frequency perception
- **12 Chroma Bins**: Musical note distribution

### Total: 58 Features
Best balance between feature richness and model simplicity for ESP32 deployment.

## Model Architecture

```
Input (58 features)
    ↓
Dense(256) + BatchNorm + Dropout(0.3)
    ↓
Dense(128) + BatchNorm + Dropout(0.3)
    ↓
Dense(64) + BatchNorm + Dropout(0.2)
    ↓
Dense(32) + BatchNorm + Dropout(0.2)
    ↓
Dense(10, softmax) - Genre probabilities
```

### Why This Architecture?
- **58→256**: Expand feature space to capture interactions
- **BatchNorm**: Stabilize training, reduce internal covariate shift
- **Dropout**: Reduce overfitting (critical with 1000 samples)
- **256→128→64→32**: Gradually compress to bottleneck
- **Output (softmax)**: 10-way genre classification

## Genres Classified

1. Blues
2. Classical
3. Country
4. Disco
5. Hip-hop
6. Jazz
7. Metal
8. Pop
9. Reggae
10. Rock


## Troubleshooting

**Issue: "GTZAN directory not found"**
- Download GTZAN dataset first
- Ensure it's in `ml/gtzan/` directory

**Issue: "Out of memory" during feature extraction**
- Feature extraction is memory-efficient
- Each song loads one at a time
- Should work on any machine with 4GB+ RAM

**Issue: Low test accuracy (< 70%)**
- Check feature scaling (should be zero mean, unit variance)
- Verify GTZAN files are not corrupted
- Try training longer (increase epochs)

**Issue: TFLite model too large for ESP32**
- Already using int8 quantization (minimal size)
- Consider removing some features if needed
- ESP32 has ~320 KB of SRAM, model fits

## References

- [GTZAN Dataset](http://marsyas.info/downloads/datasets.html)
- [LibROSA Audio Features](https://librosa.org/doc/main/feature.html)
- [TensorFlow Lite ESP32](https://www.tensorflow.org/lite/microcontrollers)
- [MFCC Audio Features](https://en.wikipedia.org/wiki/Mel-frequency_cepstral_coefficient)
