"""
Scoring Model Module
ML model for scoring image quality based on extracted features and deep learning.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from typing import Dict, Optional, Tuple
import pickle
import os


class ImageScoringModel:
    """
    Hybrid ML model for image quality scoring.
    Combines deep learning (CNN features) with traditional features.
    """

    def __init__(self, feature_dim: int = 30, input_shape: Tuple[int, int, int] = (512, 512, 3)):
        """
        Initialize the scoring model.

        Args:
            feature_dim: Dimension of traditional feature vector
            input_shape: Shape of input images for CNN
        """
        self.feature_dim = feature_dim
        self.input_shape = input_shape
        self.model = None
        self.is_trained = False

    def build_model(self):
        """
        Build the hybrid scoring model.

        Architecture:
        - CNN branch: Pre-trained MobileNetV2 for visual features
        - Feature branch: Dense layers for traditional features
        - Combined: Merged and processed for final score
        """
        # Image input branch (CNN)
        image_input = layers.Input(shape=self.input_shape, name='image_input')

        # Use pre-trained MobileNetV2 as feature extractor
        base_model = MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        # Freeze base model layers for faster training with small datasets
        base_model.trainable = False

        # Extract features from CNN
        x = base_model(image_input, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        cnn_features = layers.Dense(128, activation='relu', name='cnn_dense')(x)

        # Traditional features input branch
        feature_input = layers.Input(shape=(self.feature_dim,), name='feature_input')
        y = layers.Dense(64, activation='relu')(feature_input)
        y = layers.Dropout(0.2)(y)
        y = layers.Dense(32, activation='relu')(y)
        traditional_features = layers.Dropout(0.2)(y)

        # Combine both branches
        combined = layers.Concatenate()([cnn_features, traditional_features])

        # Final processing
        z = layers.Dense(64, activation='relu')(combined)
        z = layers.Dropout(0.3)(z)
        z = layers.Dense(32, activation='relu')(z)

        # Output layers for different aspects
        overall_score = layers.Dense(1, activation='sigmoid', name='overall_score')(z)
        composition_score = layers.Dense(1, activation='sigmoid', name='composition_score')(z)
        color_score = layers.Dense(1, activation='sigmoid', name='color_score')(z)
        technical_score = layers.Dense(1, activation='sigmoid', name='technical_score')(z)

        # Create model
        self.model = models.Model(
            inputs=[image_input, feature_input],
            outputs=[overall_score, composition_score, color_score, technical_score],
            name='image_quality_scorer'
        )

        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'overall_score': 'mse',
                'composition_score': 'mse',
                'color_score': 'mse',
                'technical_score': 'mse'
            },
            loss_weights={
                'overall_score': 2.0,
                'composition_score': 1.0,
                'color_score': 1.0,
                'technical_score': 1.0
            },
            metrics=['mae']
        )

        return self.model

    def build_lightweight_model(self):
        """
        Build a lightweight model that works without pre-training.
        Useful when no training data is available - uses heuristic scoring.
        """
        # Traditional features input
        feature_input = layers.Input(shape=(self.feature_dim,), name='feature_input')

        # Feature processing
        x = layers.Dense(128, activation='relu')(feature_input)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)

        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Dense(32, activation='relu')(x)

        # Output layer
        overall_score = layers.Dense(1, activation='sigmoid', name='overall_score')(x)

        # Create model
        self.model = models.Model(
            inputs=feature_input,
            outputs=overall_score,
            name='lightweight_quality_scorer'
        )

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        return self.model

    def predict_score(self, image: np.ndarray, features: np.ndarray) -> Dict[str, float]:
        """
        Predict quality scores for an image.

        Args:
            image: Preprocessed image (normalized to 0-1)
            features: Feature vector from feature extractor

        Returns:
            Dictionary of scores
        """
        if self.model is None:
            raise RuntimeError("Model not built. Call build_model() first.")

        # Prepare inputs
        image_batch = np.expand_dims(image, axis=0)
        feature_batch = np.expand_dims(features, axis=0)

        # Get model input names to determine model type
        input_names = [inp.name for inp in self.model.inputs]

        if 'image_input' in input_names[0]:
            # Full hybrid model
            predictions = self.model.predict(
                [image_batch, feature_batch],
                verbose=0
            )

            scores = {
                'overall_score': float(predictions[0][0, 0]),
                'composition_score': float(predictions[1][0, 0]),
                'color_score': float(predictions[2][0, 0]),
                'technical_score': float(predictions[3][0, 0])
            }
        else:
            # Lightweight model
            prediction = self.model.predict(feature_batch, verbose=0)
            scores = {
                'overall_score': float(prediction[0, 0])
            }

        return scores

    def heuristic_score(self, features_dict: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate heuristic-based scores when no trained model is available.
        Uses rules and thresholds based on photographic principles.

        Args:
            features_dict: Dictionary of extracted features

        Returns:
            Dictionary of scores
        """
        scores = {}

        # Composition score
        composition_factors = []
        composition_factors.append(min(features_dict.get('rule_of_thirds_score', 0) * 2, 1.0))
        composition_factors.append(features_dict.get('center_focus', 0.5))
        composition_factors.append(min(features_dict.get('complexity', 0) * 3, 1.0))
        scores['composition_score'] = np.mean(composition_factors)

        # Color score
        color_factors = []
        color_factors.append(features_dict.get('color_vibrancy', 0.5))
        color_factors.append(min(features_dict.get('color_diversity', 0) / 2.5, 1.0))
        color_factors.append(1.0 - abs(features_dict.get('color_balance', 0.5) - 0.5) * 2)
        scores['color_score'] = np.mean(color_factors)

        # Technical score (contrast, sharpness, exposure, noise)
        technical_factors = []
        technical_factors.append(min(features_dict.get('rms_contrast', 0) * 3, 1.0))
        technical_factors.append(min(features_dict.get('laplacian_variance', 0) * 2, 1.0))

        # Exposure - penalize over/underexposure
        exposure_penalty = (
            features_dict.get('overexposure', 0) +
            features_dict.get('underexposure', 0)
        )
        exposure_score = max(0, 1.0 - exposure_penalty * 5)
        technical_factors.append(exposure_score)

        # Noise - higher SNR is better
        technical_factors.append(features_dict.get('snr_estimate', 0.5))

        scores['technical_score'] = np.mean(technical_factors)

        # Overall score - weighted combination
        scores['overall_score'] = (
            scores['composition_score'] * 0.35 +
            scores['color_score'] * 0.25 +
            scores['technical_score'] * 0.40
        )

        # Normalize to 0-100 scale for user-friendly output
        for key in scores:
            scores[key] = min(max(scores[key], 0), 1.0)  # Clamp to [0, 1]

        return scores

    def save_model(self, path: str):
        """Save model to disk."""
        if self.model is None:
            raise RuntimeError("No model to save.")
        self.model.save(path)

    def load_model(self, path: str):
        """Load model from disk."""
        self.model = keras.models.load_model(path)
        self.is_trained = True

    def get_model_summary(self) -> str:
        """Get model architecture summary."""
        if self.model is None:
            return "Model not built yet."

        from io import StringIO
        stream = StringIO()
        self.model.summary(print_fn=lambda x: stream.write(x + '\n'))
        return stream.getvalue()
