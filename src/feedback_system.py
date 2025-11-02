"""
Feedback System
Interactive system for collecting user feedback and continuous model improvement.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Optional
import json

from dataset_loader import FeedbackDataset
from train_model import ModelTrainer
from image_loader import CR2ImageLoader
from feature_extractor import ImageFeatureExtractor
from scoring_model import ImageScoringModel


class FeedbackCollector:
    """Collects and manages user feedback for model improvement."""

    def __init__(self, feedback_path: str = 'data/feedback'):
        """
        Initialize feedback collector.

        Args:
            feedback_path: Path to store feedback data
        """
        self.feedback_dataset = FeedbackDataset(feedback_path)

    def collect_feedback_interactive(
        self,
        image_path: str,
        predicted_scores: Dict[str, float],
        features: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Collect feedback interactively from the user.

        Args:
            image_path: Path to the image
            predicted_scores: Scores predicted by the model
            features: Extracted features

        Returns:
            User-provided scores
        """
        print("\n" + "=" * 60)
        print("FEEDBACK COLLECTION")
        print("=" * 60)
        print(f"\nImage: {image_path}")
        print("\nModel's predicted scores:")
        for key, value in predicted_scores.items():
            print(f"  {key}: {value:.2f}")

        print("\n" + "-" * 60)
        print("Please rate the accuracy of these predictions (0-1 scale):")
        print("Enter your own scores for this image.")
        print("Press Enter to accept model's prediction for any score.")
        print("-" * 60)

        user_scores = {}

        # Collect overall score
        user_scores['overall_score'] = self._get_score_input(
            "Overall Quality",
            predicted_scores.get('overall_score', 0.5)
        )

        # Collect composition score
        user_scores['composition_score'] = self._get_score_input(
            "Composition",
            predicted_scores.get('composition_score', 0.5)
        )

        # Collect color score
        user_scores['color_score'] = self._get_score_input(
            "Color Quality",
            predicted_scores.get('color_score', 0.5)
        )

        # Collect technical score
        user_scores['technical_score'] = self._get_score_input(
            "Technical Quality",
            predicted_scores.get('technical_score', 0.5)
        )

        # Add feedback to dataset
        self.feedback_dataset.add_feedback(
            image_path=image_path,
            predicted_scores=predicted_scores,
            user_scores=user_scores,
            features=features
        )

        print("\n✓ Feedback recorded successfully!")

        return user_scores

    def collect_feedback_batch(
        self,
        feedback_data: Dict[str, Dict]
    ):
        """
        Collect feedback in batch mode from a JSON file.

        Args:
            feedback_data: Dictionary with feedback entries
        """
        for entry in feedback_data.get('entries', []):
            self.feedback_dataset.add_feedback(
                image_path=entry['image_path'],
                predicted_scores=entry['predicted_scores'],
                user_scores=entry['user_scores'],
                features=entry['features'],
                timestamp=entry.get('timestamp')
            )

        print(f"✓ Loaded {len(feedback_data.get('entries', []))} feedback entries")

    def _get_score_input(self, label: str, default: float) -> float:
        """Get score input from user with validation."""
        while True:
            try:
                user_input = input(f"\n  {label} (0-1, default={default:.2f}): ").strip()

                if not user_input:
                    return default

                score = float(user_input)

                if 0 <= score <= 1:
                    return score
                else:
                    print("    ⚠ Score must be between 0 and 1")

            except ValueError:
                print("    ⚠ Please enter a valid number")
            except KeyboardInterrupt:
                print("\n\nFeedback collection cancelled.")
                sys.exit(0)

    def get_statistics(self) -> Dict:
        """Get feedback statistics."""
        return self.feedback_dataset.get_statistics()

    def export_for_training(self, output_path: str) -> bool:
        """Export feedback for model retraining."""
        return self.feedback_dataset.export_for_training(output_path)


class ContinuousLearner:
    """Manages continuous learning from user feedback."""

    def __init__(
        self,
        model_path: str,
        feedback_path: str = 'data/feedback',
        min_feedback_samples: int = 50
    ):
        """
        Initialize continuous learner.

        Args:
            model_path: Path to the model
            feedback_path: Path to feedback data
            min_feedback_samples: Minimum samples needed for retraining
        """
        self.model_path = model_path
        self.feedback_path = feedback_path
        self.min_feedback_samples = min_feedback_samples
        self.feedback_dataset = FeedbackDataset(feedback_path)

    def should_retrain(self) -> bool:
        """Check if model should be retrained based on available feedback."""
        stats = self.feedback_dataset.get_statistics()
        total_feedback = stats.get('total_feedback', 0)

        return total_feedback >= self.min_feedback_samples

    def retrain_model(
        self,
        epochs: int = 10,
        backup: bool = True
    ) -> bool:
        """
        Retrain model with accumulated feedback.

        Args:
            epochs: Number of training epochs
            backup: Whether to backup the old model

        Returns:
            True if retraining successful
        """
        if not self.should_retrain():
            stats = self.feedback_dataset.get_statistics()
            total = stats.get('total_feedback', 0)
            print(f"Not enough feedback samples ({total}/{self.min_feedback_samples})")
            return False

        print("Starting model retraining with user feedback...")

        # Backup existing model
        if backup and os.path.exists(self.model_path):
            backup_path = self.model_path.replace('.h5', '_backup.h5')
            print(f"Backing up model to {backup_path}")
            os.rename(self.model_path, backup_path)

        # Initialize components
        image_loader = CR2ImageLoader()
        feature_extractor = ImageFeatureExtractor()
        model = ImageScoringModel()

        # Load existing model if available
        if os.path.exists(self.model_path.replace('.h5', '_backup.h5')):
            try:
                model.load_model(self.model_path.replace('.h5', '_backup.h5'))
                print("Loaded existing model for fine-tuning")
            except:
                print("Could not load existing model, training from scratch")

        # Train on feedback
        trainer = ModelTrainer(model, feature_extractor, image_loader)
        history = trainer.train_on_feedback(
            self.feedback_path,
            epochs=epochs
        )

        # Save retrained model
        model.save_model(self.model_path)
        print(f"✓ Model retrained and saved to {self.model_path}")

        # Clear old feedback (optional - you might want to keep it)
        # self._archive_feedback()

        return True

    def get_improvement_report(self) -> str:
        """Generate a report on model improvements from feedback."""
        stats = self.feedback_dataset.get_statistics()

        report = [
            "\n" + "=" * 60,
            "MODEL IMPROVEMENT REPORT",
            "=" * 60,
            f"\nTotal feedback samples: {stats.get('total_feedback', 0)}",
            f"Minimum required for retraining: {self.min_feedback_samples}",
            "\nAverage prediction errors:"
        ]

        errors = stats.get('average_errors', {})
        for score_type, error in errors.items():
            report.append(f"  {score_type}: {error:.3f}")

        if self.should_retrain():
            report.append("\n✓ Ready for retraining!")
        else:
            needed = self.min_feedback_samples - stats.get('total_feedback', 0)
            report.append(f"\n⏳ Need {needed} more samples for retraining")

        return "\n".join(report)


def main():
    """Main feedback system interface."""
    parser = argparse.ArgumentParser(description='Feedback and continuous learning system')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['stats', 'retrain', 'export'],
                       help='Operation mode')
    parser.add_argument('--feedback-path', type=str, default='data/feedback',
                       help='Path to feedback data')
    parser.add_argument('--model-path', type=str, default='models/trained_model.h5',
                       help='Path to model')
    parser.add_argument('--export-path', type=str, default='data/feedback_export.json',
                       help='Path to export feedback')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Training epochs for retraining')
    parser.add_argument('--min-samples', type=int, default=50,
                       help='Minimum feedback samples for retraining')

    args = parser.parse_args()

    if args.mode == 'stats':
        # Show feedback statistics
        learner = ContinuousLearner(
            args.model_path,
            args.feedback_path,
            args.min_samples
        )
        print(learner.get_improvement_report())

    elif args.mode == 'retrain':
        # Retrain model with feedback
        learner = ContinuousLearner(
            args.model_path,
            args.feedback_path,
            args.min_samples
        )
        success = learner.retrain_model(epochs=args.epochs)

        if success:
            print("\n✓ Retraining complete!")
        else:
            print("\n⚠ Retraining not performed")

    elif args.mode == 'export':
        # Export feedback data
        collector = FeedbackCollector(args.feedback_path)
        success = collector.export_for_training(args.export_path)

        if success:
            print(f"✓ Feedback exported to {args.export_path}")
        else:
            print("⚠ Not enough feedback to export")


if __name__ == '__main__':
    main()
