# Audio Features - Quick Reference

## Spectral Features

### Spectral Centroid
The "center of mass" of the frequency spectrum - indicates brightness of sound.

**Interpretation**:
- `< 2 kHz`: Dark, bass-heavy (classical, low synths)
- `2-4 kHz`: Warm, normal (human voice, most instruments)
- `> 5 kHz`: Bright, presence (pop, high frequencies)

**Use Case**: Genre classification, timbre analysis, audio quality assessment

### Spectral Spread
Bandwidth around the centroid. Measures how tonal vs dispersed the spectrum is.

**Interpretation**:
- `< 500 Hz`: Very tonal (pure sine wave)
- `500-2000 Hz`: Harmonic (musical instruments)
- `> 2000 Hz`: Noisy (percussion, white noise)

### Spectral Flatness
Ratio of geometric to arithmetic mean. Indicates tonal vs noise-like.

**Interpretation**:
- `0.0-0.2`: Very tonal
- `0.2-0.5`: Musical
- `0.5-1.0`: Noisy

**Use Case**: Voice activity detection, silence detection

### Spectral Rolloff
Frequency containing 85% of the energy.

**Interpretation**: Where the spectrum "cuts off" - audio bandwidth indicator

## Temporal Features

### Zero-Crossing Rate (ZCR)
Rate at which signal changes sign. Indicates frequency content.

**Interpretation**:
- `< 0.05`: Silent or very low-frequency
- `0.05-0.15`: Low-frequency, smooth (vowels, bass)
- `0.15-0.25`: Normal speech
- `0.25-0.40`: High-frequency content (consonants)
- `> 0.40`: Very high-frequency, rough (noise)

**Use Case**: Voice activity detection, voiced/unvoiced classification, noise detection

### RMS Energy
Overall loudness/intensity of the signal (0-1 normalized).

**Interpretation**:
- `< 0.05`: Silent
- `0.05-0.2`: Quiet (whisper)
- `0.2-0.5`: Normal
- `0.5-0.8`: Loud
- `> 0.8`: Very loud (clipping risk)

**Use Case**: Microphone level checking, gain control, activity detection

### Peak Amplitude
Highest absolute sample value in frame (0-1 normalized).

**Interpretation**: Headroom indicator, clipping warning
- `> 0.95`: Risk of clipping

## Frequency Bands

### 16 Logarithmic Bands (60 Hz - 16 kHz)
Spectrum divided into perceptually-spaced bands matching human hearing.

**Examples**:
- Bands 0-2 (60-211 Hz): Bass
- Bands 4-6 (322-1147 Hz): Low-mid, warmth
- Bands 7-9 (1147-4084 Hz): Presence (critical for speech)
- Bands 10-13 (4084-14557 Hz): Treble, air

**Use Case**: Real-time visualization, genre analysis, EQ

### Genre Characteristic Profiles
```
Classical:  Strong low end (bands 0-2)
Pop:        Boosted presence (bands 7-9)
Rock:       Strong mids + treble (bands 5-13)
Electronic: Varied (depends on synth content)
```

## Peak Detection

### Local Maxima with Hysteresis
Finds dominant frequencies in spectrum.

**Output**: Top peaks by magnitude with frequencies

**Use Case**: Harmonic analysis, formant detection, dominant frequency tracking

## Note Detection (Goertzel)

### 12 Chromatic Detectors (A3-G#5)
Efficient single-frequency detection for musical notes.

**Notes Detected**:
- A3 (110 Hz), A#3 (116.5 Hz), B3 (123.5 Hz)
- C4-B4 (one octave)
- C5-G#5 (partial octave)

**Output**: Magnitude and phase for each note

**Use Case**: Real-time tuner, musical key detection, note tracking

## Combined Feature Applications

### Voice Activity Detection (VAD)
```
Silence:   ZCR < 0.05, Energy < 0.05
Speech:    ZCR 0.1-0.2, Energy > 0.1
Noise:     Flatness > 0.5
```

### Music Genre Classification
```
Analyze: Centroid, Spread, ZCR, Energy, Frequency Bands
Classify: Train ML model on features
```

### Real-Time Visualizer
```
Display: 16 frequency bands (bar chart)
Mark: Spectral centroid line
Show: Top 5-10 peaks as markers
Indicate: ZCR + Energy activity level
```

## Typical Values (Reference)

### Speech
- Centroid: 1.5-3.5 kHz
- Spread: 1-2 kHz
- Flatness: 0.2-0.4
- ZCR: 0.1-0.25
- Energy: 0.2-0.6

### Music (complex)
- Centroid: 2-5 kHz (varies by genre)
- Spread: 3-5 kHz
- Flatness: 0.1-0.5
- ZCR: 0.05-0.2
- Energy: 0.3-0.9

### Noise/Silence
- Centroid: Undefined
- Spread: High (if noise)
- Flatness: > 0.5 (noise), < 0.1 (silence)
- ZCR: < 0.05 (silence), > 0.3 (noise)
- Energy: < 0.05 (silence)

## API Reference

```cpp
// Spectral features
SpectralFeatures DSPFeatureExtractor::compute_spectral_centroid(
    const dsp_real_t* spectrum, uint16_t spectrum_len, dsp_real_t freq_res);

// Temporal features
TemporalFeatures DSPFeatureExtractor::compute_temporal_features(
    const int16_t* signal, uint16_t signal_len);

// Peak detection
uint16_t DSPFeatureExtractor::detect_peaks(
    const dsp_real_t* spectrum, uint16_t spectrum_len,
    const PeakDetectionConfig& config, uint16_t* peaks_out, uint16_t peaks_out_len);

// Note detection
void DSPFeatureExtractor::goertzel_init(GoertzelState& state, ...);
void DSPFeatureExtractor::goertzel_process(GoertzelState& state, int16_t sample);
void DSPFeatureExtractor::goertzel_finalize(GoertzelState& state);

// Mel spectrum
void DSPFeatureExtractor::compute_mel_spectrum(
    const dsp_real_t* spectrum, uint16_t spectrum_len, dsp_real_t freq_res,
    dsp_real_t* mel_bands, uint8_t num_bands);
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for implementation details and mathematical foundations.
