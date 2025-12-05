#!/usr/bin/env python3
"""
Genre Classifier - Neural Network Training

Trains a neural network on extracted audio features to classify music genre.
Outputs TensorFlow Lite model for deployment on ESP32.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import seaborn as sns
from pathlib import Path

# Configuration
FEATURES_CSV = "./gtzan_features.csv"
MODEL_OUTPUT = "./genre_model.h5"
TFLITE_OUTPUT = "./genre_model.tflite"
SCALER_OUTPUT = "./feature_scaler.pkl"

GENRES = ['blues', 'classical', 'country', 'disco', 'hip-hop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']

class GenreClassifier:
    """Train and evaluate genre classification model"""

    def __init__(self, features_csv=FEATURES_CSV):
        self.features_csv = features_csv
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None

    def load_data(self):
        """Load features from CSV"""
        print("\n📂 Loading features from CSV...")

        if not os.path.exists(self.features_csv):
            print(f" Features file not found: {self.features_csv}")
            print("   Run feature_extraction.py first!")
            return None, None, None, None

        df = pd.read_csv(self.features_csv)
        print(f" Loaded {len(df)} samples with {len(df.columns)} features")

        # Separate features and labels
        X = df.drop(['genre', 'filename'], axis=1)
        y = df['genre']

        self.feature_names = X.columns.tolist()
        print(f"   Features: {len(self.feature_names)}")
        print(f"   Genres: {len(y.unique())}")

        return X, y, df

    def preprocess_data(self, X, y):
        """Normalize features and encode labels"""
        print("\n⚙️  Preprocessing data...")

        # Normalize features (mean=0, std=1)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Encode genre labels (0-9)
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        print(f" Features scaled to zero mean, unit variance")
        print(f" Genre labels encoded: {dict(zip(GENRES, self.label_encoder.transform(GENRES)))}")

        return X_scaled, y_encoded

    def split_data(self, X_scaled, y_encoded, test_size=0.2, val_size=0.2):
        """Split into train/val/test sets"""
        print(f"\n📊 Splitting data ({100-int(test_size*100)}% train, {int(val_size*100)}% val, {int(test_size*100)}% test)...")

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )

        # Train/val split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=42, stratify=y_train
        )

        print(f"   Train: {len(X_train)} samples")
        print(f"   Val:   {len(X_val)} samples")
        print(f"   Test:  {len(X_test)} samples")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def build_model(self, input_dim, num_classes=10):
        """Build neural network architecture"""
        print(f"\n Building neural network...")

        model = models.Sequential([
            # Input layer
            layers.Input(shape=(input_dim,)),

            # First dense block
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            # Second dense block
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            # Third dense block
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),

            # Fourth dense block
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),

            # Output layer
            layers.Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print(" Model architecture:")
        model.summary()

        self.model = model
        return model

    def train_model(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Train the model"""
        print(f"\n🚀 Training model ({epochs} epochs, batch_size={batch_size})...")

        # Callbacks
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )

        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )

        return history

    def evaluate_model(self, X_test, y_test):
        """Evaluate model on test set"""
        print(f"\n Evaluating model...")

        y_pred = self.model.predict(X_test)
        y_pred_labels = np.argmax(y_pred, axis=1)

        accuracy = accuracy_score(y_test, y_pred_labels)
        print(f" Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

        print("\n Classification Report:")
        print(classification_report(
            y_test, y_pred_labels,
            target_names=self.label_encoder.classes_
        ))

        return y_pred, y_pred_labels

    def plot_results(self, history, y_test, y_pred_labels, output_dir="./plots"):
        """Plot training history and confusion matrix"""
        print(f"\n Generating plots...")

        os.makedirs(output_dir, exist_ok=True)

        # Training history
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(history.history['loss'], label='Train Loss')
        ax1.plot(history.history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Model Loss')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(history.history['accuracy'], label='Train Accuracy')
        ax2.plot(history.history['val_accuracy'], label='Val Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=150)
        print(f"   Saved: training_history.png")
        plt.close()

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
        print(f"   Saved: confusion_matrix.png")
        plt.close()

    def save_model(self, h5_path=MODEL_OUTPUT):
        """Save Keras model"""
        self.model.save(h5_path)
        print(f"\n💾 Model saved: {h5_path}")

    def convert_to_tflite(self, tflite_path=TFLITE_OUTPUT, quantize=True):
        """Convert Keras model to TensorFlow Lite"""
        print(f"\n🔄 Converting to TensorFlow Lite...")

        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        if quantize:
            # Quantize to int8 for smaller model size and faster inference
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            print("   Using int8 quantization for ESP32")

        tflite_model = converter.convert()

        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)

        file_size = os.path.getsize(tflite_path) / 1024  # KB
        print(f"✅ TFLite model saved: {tflite_path}")
        print(f"   Model size: {file_size:.1f} KB")

    def save_metadata(self, metadata_path="./model_metadata.json"):
        """Save model metadata for inference"""
        import json

        metadata = {
            'feature_names': self.feature_names,
            'genres': self.label_encoder.classes_.tolist(),
            'num_features': len(self.feature_names),
            'num_genres': len(self.label_encoder.classes_),
            'scaler_mean': self.scaler.mean_.tolist(),
            'scaler_scale': self.scaler.scale_.tolist(),
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Metadata saved: {metadata_path}")

def main():
    """Main training pipeline"""
    print("=" * 70)
    print("Genre Classification - Neural Network Training")
    print("=" * 70)

    classifier = GenreClassifier()

    # 1. Load data
    X, y, df = classifier.load_data()
    if X is None:
        return

    # 2. Preprocess
    X_scaled, y_encoded = classifier.preprocess_data(X, y)

    # 3. Split data
    X_train, X_val, X_test, y_train, y_val, y_test = classifier.split_data(
        X_scaled, y_encoded
    )

    # 4. Build model
    classifier.build_model(input_dim=X_train.shape[1], num_classes=len(GENRES))

    # 5. Train model
    history = classifier.train_model(X_train, y_train, X_val, y_val, epochs=100)

    # 6. Evaluate
    y_pred, y_pred_labels = classifier.evaluate_model(X_test, y_test)

    # 7. Plot results
    classifier.plot_results(history, y_test, y_pred_labels)

    # 8. Save models
    classifier.save_model()
    classifier.convert_to_tflite(quantize=True)
    classifier.save_metadata()

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"\nGenerated files:")
    print(f"  - {MODEL_OUTPUT} (Keras model)")
    print(f"  - {TFLITE_OUTPUT} (TensorFlow Lite for ESP32)")
    print(f"  - model_metadata.json (Feature names, genres, scaling params)")
    print(f"  - plots/training_history.png (Training curves)")
    print(f"  - plots/confusion_matrix.png (Prediction accuracy)")

if __name__ == "__main__":
    main()
