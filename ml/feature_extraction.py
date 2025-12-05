#!/usr/bin/env python3
"""
GTZAN Genre Classification - Feature Extraction Script

Extracts audio features from GTZAN dataset for neural network training.
Features include:
- Spectral: centroid, spread, flatness, rolloff
- Temporal: ZCR, RMS energy, peak amplitude
- Frequency bands: 16 logarithmic bands
- MFCC: 13 coefficients
"""

import os
import numpy as np
import pandas as pd
import librosa
import librosa.feature
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Configuration
GTZAN_PATH = "./gtzan"  # Download from: http://marsyas.info/downloads/datasets.html
OUTPUT_CSV = "./gtzan_features.csv"
SAMPLE_RATE = 44100
DURATION = 30  # seconds (GTZAN clips are 30s)

# Genre labels (10 genres in GTZAN)
GENRES = ['blues', 'classical', 'country', 'disco', 'hip-hop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']

class AudioFeatureExtractor:
    """Extract audio features from WAV/MP3 files"""

    def __init__(self, sr=SAMPLE_RATE):
        self.sr = sr

    def compute_spectral_centroid(self, y, sr):
        """Compute spectral centroid - "brightness" indicator"""
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        return np.mean(centroid)

    def compute_spectral_rolloff(self, y, sr):
        """Compute spectral rolloff - frequency at 85% energy"""
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        return np.mean(rolloff)

    def compute_spectral_spread(self, y, sr):
        """Compute spectral spread - bandwidth around centroid"""
        spec = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)
        mag_spec = np.mean(spec, axis=1)
        centroid = np.sum(freqs * mag_spec) / np.sum(mag_spec)
        spread = np.sqrt(np.sum(((freqs - centroid) ** 2) * mag_spec) / np.sum(mag_spec))
        return spread

    def compute_spectral_flatness(self, y, sr):
        """Compute spectral flatness - tonal vs noise"""
        spec = np.abs(librosa.stft(y))
        mag_spec = np.mean(spec, axis=1)

        # Geometric mean / Arithmetic mean
        geo_mean = np.exp(np.mean(np.log(mag_spec + 1e-10)))
        arith_mean = np.mean(mag_spec)
        flatness = geo_mean / (arith_mean + 1e-10)
        return np.clip(flatness, 0, 1)

    def compute_zcr(self, y):
        """Compute zero-crossing rate - frequency content"""
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        return np.mean(zcr)

    def compute_rms_energy(self, y):
        """Compute RMS energy - loudness"""
        rms = librosa.feature.rms(y=y)[0]
        return np.mean(rms)

    def compute_peak_amplitude(self, y):
        """Compute peak amplitude (normalized to -1 to 1)"""
        return np.max(np.abs(y))

    def compute_mfcc(self, y, sr, n_mfcc=13):
        """Compute MFCC - perceptual features for ML"""
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfcc, axis=1)  # Average across time

    def compute_chroma(self, y, sr):
        """Compute chromatic features - musical note distribution"""
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        return np.mean(chroma, axis=1)  # 12 chroma bins

    def compute_freq_bands(self, y, sr, n_bands=16):
        """
        Compute 16 logarithmic frequency bands
        Similar to the DSP implementation on ESP32
        """
        spec = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)
        mag_spec = np.mean(spec, axis=1)

        freq_min = 60.0
        freq_max = 16000.0

        # Create logarithmically-spaced band edges
        band_edges = np.logspace(np.log10(freq_min), np.log10(freq_max), n_bands + 1)

        bands = []
        for i in range(n_bands):
            mask = (freqs >= band_edges[i]) & (freqs < band_edges[i + 1])
            band_energy = np.mean(mag_spec[mask]) if np.any(mask) else 0
            bands.append(band_energy)

        return np.array(bands)

    def extract_features(self, audio_path):
        """Extract all features from audio file"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sr, duration=DURATION)

            # Initialize feature dict
            features = {}

            # Spectral features
            features['centroid'] = self.compute_spectral_centroid(y, sr)
            features['spread'] = self.compute_spectral_spread(y, sr)
            features['flatness'] = self.compute_spectral_flatness(y, sr)
            features['rolloff'] = self.compute_spectral_rolloff(y, sr)

            # Temporal features
            features['zcr'] = self.compute_zcr(y)
            features['rms_energy'] = self.compute_rms_energy(y)
            features['peak_amplitude'] = self.compute_peak_amplitude(y)

            # Frequency bands (16)
            freq_bands = self.compute_freq_bands(y, sr, n_bands=16)
            for i, band_energy in enumerate(freq_bands):
                features[f'band_{i}'] = band_energy

            # MFCC (13 coefficients)
            mfcc = self.compute_mfcc(y, sr, n_mfcc=13)
            for i, coeff in enumerate(mfcc):
                features[f'mfcc_{i}'] = coeff

            # Chroma features (12 bins)
            chroma = self.compute_chroma(y, sr)
            for i, chroma_val in enumerate(chroma):
                features[f'chroma_{i}'] = chroma_val

            return features, True

        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None, False

def extract_gtzan_features(gtzan_path=GTZAN_PATH, output_csv=OUTPUT_CSV):
    """
    Extract features from all GTZAN audio files

    GTZAN structure:
    gtzan/
    ├── blues/
    │   ├── blues.00000.au
    │   ├── blues.00001.au
    │   └── ...
    ├── classical/
    ├── country/
    └── ...
    """

    print("=" * 60)
    print("GTZAN Feature Extraction")
    print("=" * 60)

    # Check if GTZAN directory exists
    if not os.path.exists(gtzan_path):
        print(f"\n❌ GTZAN directory not found at: {gtzan_path}")
        print("\nTo download GTZAN:")
        print("1. Visit: http://marsyas.info/downloads/datasets.html")
        print("2. Download: GTZAN Genre Collection")
        print("3. Extract to: ./gtzan/")
        print("\nOr download programmatically:")
        print("  wget http://opihi.cs.uvic.ca/sound/genres.tar.gz")
        print("  tar -xzf genres.tar.gz")
        return None

    # Initialize extractor
    extractor = AudioFeatureExtractor(sr=SAMPLE_RATE)

    # Collect all features
    all_features = []
    total_files = 0
    successful_files = 0

    # Iterate through genres
    for genre in GENRES:
        genre_path = os.path.join(gtzan_path, genre)

        if not os.path.exists(genre_path):
            print(f"⚠️  Genre directory not found: {genre_path}")
            continue

        # Get all audio files (WAV or AU format)
        audio_files = list(Path(genre_path).glob('*.wav')) + list(Path(genre_path).glob('*.au'))

        print(f"\n🎵 Processing genre: {genre.upper()} ({len(audio_files)} files)")

        # Process each file
        pbar = tqdm(audio_files, desc=f"  {genre}")
        for audio_file in pbar:
            total_files += 1

            features, success = extractor.extract_features(str(audio_file))

            if success:
                features['genre'] = genre
                features['filename'] = audio_file.name
                all_features.append(features)
                successful_files += 1

            pbar.set_postfix({"success": successful_files, "failed": total_files - successful_files})

    # Create DataFrame
    df = pd.DataFrame(all_features)

    # Print summary
    print("\n" + "=" * 60)
    print(f"✅ Extraction Complete!")
    print(f"   Total files: {total_files}")
    print(f"   Successfully processed: {successful_files}")
    print(f"   Failed: {total_files - successful_files}")
    print(f"   Features per file: {len(df.columns) - 2}")  # -2 for genre and filename
    print(f"   Total samples: {len(df)}")
    print("=" * 60)

    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"\n💾 Features saved to: {output_csv}")

    # Print feature summary
    print("\n📊 Feature Summary:")
    print(df.describe())

    print("\n🏷️  Genre Distribution:")
    print(df['genre'].value_counts().sort_index())

    return df

def load_features(csv_path=OUTPUT_CSV):
    """Load pre-extracted features from CSV"""
    return pd.read_csv(csv_path)

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--load":
        # Load existing features
        print("Loading features from CSV...")
        df = load_features()
        print(f"Loaded {len(df)} samples with {len(df.columns)} columns")
    else:
        # Extract features from GTZAN
        df = extract_gtzan_features()
