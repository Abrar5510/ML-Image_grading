"""
Model Training Script
Trains the image scoring model using datasets like Adobe FiveK.
"""

import os
import sys
import numpy as np
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import json
from datetime import datetime

from image_loader import CR2ImageLoader
from feature_extractor import ImageFeatureExtractor
from scoring_model import ImageScoringModel
from dataset_loader import AdobeFiveKLoader, CustomDatasetLoader, FeedbackDataset


class ModelTrainer:
    """Handles model training with various datasets."""

    def __init__(
        self,
        model: ImageScoringModel,
        feature_extractor: ImageFeatureExtractor,
        image_loader: CR2ImageLoader
    ):
        """
        Initialize the trainer.

        Args:
            model: Image scoring model
            feature_extractor: Feature extractor
            image_loader: Image loader
        """
        self.model = model
        self.feature_extractor = feature_extractor
        self.image_loader = image_loader

    def train_on_adobe_fivek(
        self,
        dataset_path: str,
        expert: str = 'c',
        num_samples: int = None,
        epochs: int = 20,
        batch_size: int = 8,
        validation_split: float = 0.2
    ) -> Dict:
        """
        Train model on Adobe FiveK dataset.

        Args:
            dataset_path: Path to Adobe FiveK dataset
            expert: Which expert's edits to use
            num_samples: Number of samples to use (None for all)
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Fraction of data for validation

        Returns:
            Training history
        """
        print("Loading Adobe FiveK dataset...")
        loader = AdobeFiveKLoader(dataset_path, expert=expert)
        image_pairs = loader.load_image_pairs(limit=num_samples)

        if not image_pairs:
            raise ValueError("No image pairs loaded!")

        print(f"Loaded {len(image_pairs)} image pairs")

        # Prepare training data
        X_images = []
        X_features = []
        y_scores = []

        print("Extracting features...")
        for i, (original, edited, filename) in enumerate(image_pairs):
            if i % 50 == 0:
                print(f"  Processing {i}/{len(image_pairs)}...")

            # Extract features from original
            features = self.feature_extractor.get_feature_vector(original)

            # Calculate quality score from edited version
            # Here we assume edited versions are high quality
            # In practice, you might want more sophisticated scoring
            quality_score = self._calculate_quality_from_edit(original, edited)

            X_images.append(original)
            X_features.append(features)
            y_scores.append(quality_score)

        # Convert to numpy arrays
        X_images = np.array(X_images)
        X_features = np.array(X_features)
        y_scores = np.array(y_scores)

        print(f"Training data shape: Images={X_images.shape}, Features={X_features.shape}")

        # Build model if not already built
        if self.model.model is None:
            print("Building model...")
            self.model.feature_dim = X_features.shape[1]
            self.model.build_model()

        # Prepare targets for multi-output model
        # For Adobe FiveK, we mainly have overall quality
        # We'll estimate other scores based on features
        y_overall = y_scores
        y_composition = self._estimate_composition_score(X_features)
        y_color = self._estimate_color_score(X_features)
        y_technical = self._estimate_technical_score(X_features)

        # Train the model
        print(f"\nTraining for {epochs} epochs...")
        history = self.model.model.fit(
            [X_images, X_features],
            [y_overall, y_composition, y_color, y_technical],
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )

        self.model.is_trained = True

        return history.history

    def train_on_custom_dataset(
        self,
        dataset_path: str,
        epochs: int = 20,
        batch_size: int = 8,
        validation_split: float = 0.2
    ) -> Dict:
        """
        Train model on custom dataset with ratings.

        Args:
            dataset_path: Path to custom dataset
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Fraction of data for validation

        Returns:
            Training history
        """
        print("Loading custom dataset...")
        loader = CustomDatasetLoader(dataset_path)
        data = loader.load_images_with_ratings()

        if not data:
            raise ValueError("No data loaded!")

        print(f"Loaded {len(data)} images with ratings")

        # Prepare training data
        X_images = []
        X_features = []
        y_overall = []
        y_composition = []
        y_color = []
        y_technical = []

        print("Extracting features...")
        for i, (image, ratings, filename) in enumerate(data):
            if i % 50 == 0:
                print(f"  Processing {i}/{len(data)}...")

            features = self.feature_extractor.get_feature_vector(image)

            X_images.append(image)
            X_features.append(features)
            y_overall.append(ratings.get('overall_score', 0.5))
            y_composition.append(ratings.get('composition_score', 0.5))
            y_color.append(ratings.get('color_score', 0.5))
            y_technical.append(ratings.get('technical_score', 0.5))

        # Convert to numpy arrays
        X_images = np.array(X_images)
        X_features = np.array(X_features)
        y_overall = np.array(y_overall)
        y_composition = np.array(y_composition)
        y_color = np.array(y_color)
        y_technical = np.array(y_technical)

        # Build model if not already built
        if self.model.model is None:
            print("Building model...")
            self.model.feature_dim = X_features.shape[1]
            self.model.build_model()

        # Train the model
        print(f"\nTraining for {epochs} epochs...")
        history = self.model.model.fit(
            [X_images, X_features],
            [y_overall, y_composition, y_color, y_technical],
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )

        self.model.is_trained = True

        return history.history

    def train_on_feedback(
        self,
        feedback_path: str,
        epochs: int = 10,
        batch_size: int = 4
    ) -> Dict:
        """
        Fine-tune model on user feedback data.

        Args:
            feedback_path: Path to feedback database
            epochs: Number of training epochs
            batch_size: Batch size

        Returns:
            Training history
        """
        print("Loading feedback data...")
        feedback_dataset = FeedbackDataset(feedback_path)
        training_data = feedback_dataset.get_training_data()

        if len(training_data) < 10:
            print(f"Warning: Only {len(training_data)} feedback samples available.")
            print("Consider collecting more feedback for better fine-tuning.")
            return {}

        print(f"Loaded {len(training_data)} feedback samples")

        # Note: Feedback contains features only, not images
        # We'll use the lightweight model for feedback-based training
        X_features = []
        y_overall = []

        for features_dict, scores in training_data:
            # Convert features dict to array
            feature_vector = np.array([features_dict[k] for k in sorted(features_dict.keys())])
            X_features.append(feature_vector)
            y_overall.append(scores.get('overall_score', 0.5))

        X_features = np.array(X_features)
        y_overall = np.array(y_overall)

        # Build lightweight model if needed
        if self.model.model is None:
            print("Building lightweight model for feedback training...")
            self.model.feature_dim = X_features.shape[1]
            self.model.build_lightweight_model()

        # Fine-tune the model
        print(f"\nFine-tuning for {epochs} epochs...")
        history = self.model.model.fit(
            X_features,
            y_overall,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )

        return history.history

    def _calculate_quality_from_edit(
        self,
        original: np.ndarray,
        edited: np.ndarray
    ) -> float:
        """
        Calculate quality score by comparing original to expert edit.

        For now, we assign high scores to edited versions.
        In future, this could be more sophisticated.
        """
        # Assume edited versions are high quality
        return 0.85

    def _estimate_composition_score(self, features: np.ndarray) -> np.ndarray:
        """Estimate composition scores from features."""
        # Use composition-related feature columns
        # This is a rough estimate
        scores = []
        for feat in features:
            # Assuming certain indices correspond to composition features
            # This would need to be adapted based on actual feature order
            score = np.clip(np.mean(feat[0:5]), 0, 1)
            scores.append(score)
        return np.array(scores)

    def _estimate_color_score(self, features: np.ndarray) -> np.ndarray:
        """Estimate color scores from features."""
        scores = []
        for feat in features:
            score = np.clip(np.mean(feat[5:11]), 0, 1)
            scores.append(score)
        return np.array(scores)

    def _estimate_technical_score(self, features: np.ndarray) -> np.ndarray:
        """Estimate technical scores from features."""
        scores = []
        for feat in features:
            score = np.clip(np.mean(feat[11:]), 0, 1)
            scores.append(score)
        return np.array(scores)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train image quality scoring model')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset')
    parser.add_argument('--dataset-type', type=str, default='adobe_fivek',
                       choices=['adobe_fivek', 'custom', 'feedback'],
                       help='Type of dataset')
    parser.add_argument('--expert', type=str, default='c',
                       choices=['a', 'b', 'c', 'd', 'e'],
                       help='Adobe FiveK expert (default: c)')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--num-samples', type=int, default=None,
                       help='Number of samples to use (None for all)')
    parser.add_argument('--output', type=str, default='models/trained_model.h5',
                       help='Output model path')

    args = parser.parse_args()

    # Initialize components
    print("Initializing model components...")
    image_loader = CR2ImageLoader()
    feature_extractor = ImageFeatureExtractor()
    model = ImageScoringModel()

    trainer = ModelTrainer(model, feature_extractor, image_loader)

    # Train based on dataset type
    if args.dataset_type == 'adobe_fivek':
        history = trainer.train_on_adobe_fivek(
            args.dataset,
            expert=args.expert,
            num_samples=args.num_samples,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
    elif args.dataset_type == 'custom':
        history = trainer.train_on_custom_dataset(
            args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
    elif args.dataset_type == 'feedback':
        history = trainer.train_on_feedback(
            args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size
        )

    # Save model
    print(f"\nSaving model to {args.output}...")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    model.save_model(args.output)

    # Save training history
    history_path = args.output.replace('.h5', '_history.json')
    with open(history_path, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        history_serializable = {
            k: [float(v) for v in vals] for k, vals in history.items()
        }
        json.dump(history_serializable, f, indent=2)

    print(f"Training complete! Model saved to {args.output}")
    print(f"Training history saved to {history_path}")


if __name__ == '__main__':
    main()
