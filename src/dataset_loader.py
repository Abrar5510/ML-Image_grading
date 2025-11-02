"""
Dataset Loader Module
Loads and processes datasets like Adobe FiveK for model training.
"""

import os
import numpy as np
from typing import List, Tuple, Dict, Optional
import glob
from pathlib import Path
import json
from PIL import Image
import cv2


class AdobeFiveKLoader:
    """
    Loader for Adobe FiveK dataset.

    Adobe FiveK contains 5000 photos with expert retouching from 5 different photographers.
    Expected structure:
    - original/: Original RAW or JPEG images
    - expert_a/, expert_b/, expert_c/, expert_d/, expert_e/: Expert-edited versions
    """

    def __init__(self, dataset_path: str, expert: str = 'c'):
        """
        Initialize the Adobe FiveK loader.

        Args:
            dataset_path: Path to the Adobe FiveK dataset
            expert: Which expert's edits to use ('a', 'b', 'c', 'd', 'e')
                   Expert C is most commonly used as it's considered most aesthetically pleasing
        """
        self.dataset_path = Path(dataset_path)
        self.expert = expert.lower()
        self.original_path = self.dataset_path / 'original'
        self.edited_path = self.dataset_path / f'expert_{self.expert}'

        if not self.original_path.exists():
            raise ValueError(f"Original images path not found: {self.original_path}")
        if not self.edited_path.exists():
            raise ValueError(f"Expert {self.expert} edited images path not found: {self.edited_path}")

    def load_image_pairs(
        self,
        limit: Optional[int] = None,
        image_size: Tuple[int, int] = (512, 512)
    ) -> List[Tuple[np.ndarray, np.ndarray, str]]:
        """
        Load pairs of original and expert-edited images.

        Args:
            limit: Maximum number of image pairs to load (None for all)
            image_size: Target size for images

        Returns:
            List of (original_image, edited_image, filename) tuples
        """
        # Find all original images
        original_files = sorted(glob.glob(str(self.original_path / '*.*')))
        if limit:
            original_files = original_files[:limit]

        pairs = []
        for orig_path in original_files:
            filename = Path(orig_path).name
            edited_path = self.edited_path / filename

            # Try different extensions if exact match not found
            if not edited_path.exists():
                base_name = Path(orig_path).stem
                edited_candidates = list(self.edited_path.glob(f'{base_name}.*'))
                if edited_candidates:
                    edited_path = edited_candidates[0]
                else:
                    continue

            try:
                # Load images
                original = self._load_and_preprocess(orig_path, image_size)
                edited = self._load_and_preprocess(str(edited_path), image_size)

                pairs.append((original, edited, filename))
            except Exception as e:
                print(f"Warning: Failed to load {filename}: {e}")
                continue

        return pairs

    def _load_and_preprocess(
        self,
        image_path: str,
        target_size: Tuple[int, int]
    ) -> np.ndarray:
        """Load and preprocess an image."""
        img = Image.open(image_path).convert('RGB')

        # Resize
        img = img.resize(target_size, Image.LANCZOS)

        # Convert to numpy and normalize
        img_array = np.array(img).astype(np.float32) / 255.0

        return img_array

    def calculate_quality_score(
        self,
        original: np.ndarray,
        edited: np.ndarray
    ) -> float:
        """
        Calculate a quality score by comparing edited version to original.

        The edited version is assumed to be higher quality, so we can use
        the difference to infer what makes a good image.

        Args:
            original: Original image
            edited: Expert-edited image

        Returns:
            Quality score (0-1)
        """
        # For training purposes, assume edited versions have high quality
        # We can use various metrics to quantify the "improvement"

        # For now, assign edited images a high base score
        # In practice, this would be more sophisticated
        return 0.85  # Base quality score for professionally edited images


class CustomDatasetLoader:
    """
    Loader for custom datasets with quality ratings.

    Expected format:
    - images/: Image files
    - ratings.json: JSON file with ratings
      {
        "filename.jpg": {
          "overall_score": 0.85,
          "composition_score": 0.9,
          "color_score": 0.8,
          "technical_score": 0.85
        },
        ...
      }
    """

    def __init__(self, dataset_path: str):
        """
        Initialize custom dataset loader.

        Args:
            dataset_path: Path to dataset directory
        """
        self.dataset_path = Path(dataset_path)
        self.images_path = self.dataset_path / 'images'
        self.ratings_path = self.dataset_path / 'ratings.json'

        if not self.images_path.exists():
            raise ValueError(f"Images path not found: {self.images_path}")

        # Load ratings if available
        self.ratings = {}
        if self.ratings_path.exists():
            with open(self.ratings_path, 'r') as f:
                self.ratings = json.load(f)

    def load_images_with_ratings(
        self,
        image_size: Tuple[int, int] = (512, 512),
        limit: Optional[int] = None
    ) -> List[Tuple[np.ndarray, Dict[str, float], str]]:
        """
        Load images with their quality ratings.

        Args:
            image_size: Target size for images
            limit: Maximum number of images to load

        Returns:
            List of (image, ratings_dict, filename) tuples
        """
        image_files = sorted(glob.glob(str(self.images_path / '*.*')))
        if limit:
            image_files = image_files[:limit]

        data = []
        for img_path in image_files:
            filename = Path(img_path).name

            try:
                # Load image
                img = Image.open(img_path).convert('RGB')
                img = img.resize(image_size, Image.LANCZOS)
                img_array = np.array(img).astype(np.float32) / 255.0

                # Get ratings (use defaults if not available)
                ratings = self.ratings.get(filename, {
                    'overall_score': 0.5,
                    'composition_score': 0.5,
                    'color_score': 0.5,
                    'technical_score': 0.5
                })

                data.append((img_array, ratings, filename))
            except Exception as e:
                print(f"Warning: Failed to load {filename}: {e}")
                continue

        return data

    def save_ratings(self, ratings: Dict[str, Dict[str, float]]):
        """
        Save ratings to JSON file.

        Args:
            ratings: Dictionary mapping filenames to rating dictionaries
        """
        with open(self.ratings_path, 'w') as f:
            json.dump(ratings, indent=2, fp=f)


class FeedbackDataset:
    """
    Manages user feedback data for continuous learning.

    Stores user feedback on predictions and uses it for model improvement.
    """

    def __init__(self, feedback_path: str):
        """
        Initialize feedback dataset.

        Args:
            feedback_path: Path to feedback database
        """
        self.feedback_path = Path(feedback_path)
        self.feedback_file = self.feedback_path / 'feedback.json'

        # Create directory if it doesn't exist
        self.feedback_path.mkdir(parents=True, exist_ok=True)

        # Load existing feedback
        self.feedback = self._load_feedback()

    def _load_feedback(self) -> Dict:
        """Load feedback from disk."""
        if self.feedback_file.exists():
            with open(self.feedback_file, 'r') as f:
                return json.load(f)
        return {'entries': [], 'statistics': {}}

    def _save_feedback(self):
        """Save feedback to disk."""
        with open(self.feedback_file, 'w') as f:
            json.dump(self.feedback, indent=2, fp=f)

    def add_feedback(
        self,
        image_path: str,
        predicted_scores: Dict[str, float],
        user_scores: Dict[str, float],
        features: Dict[str, float],
        timestamp: Optional[str] = None
    ):
        """
        Add user feedback entry.

        Args:
            image_path: Path to the image
            predicted_scores: Scores predicted by the model
            user_scores: Scores provided by the user
            features: Extracted features from the image
            timestamp: Timestamp of feedback (auto-generated if None)
        """
        from datetime import datetime

        if timestamp is None:
            timestamp = datetime.now().isoformat()

        entry = {
            'timestamp': timestamp,
            'image_path': image_path,
            'predicted_scores': predicted_scores,
            'user_scores': user_scores,
            'features': features
        }

        self.feedback['entries'].append(entry)
        self._update_statistics()
        self._save_feedback()

    def _update_statistics(self):
        """Update feedback statistics."""
        if not self.feedback['entries']:
            return

        # Calculate average errors
        errors = {
            'overall_score': [],
            'composition_score': [],
            'color_score': [],
            'technical_score': []
        }

        for entry in self.feedback['entries']:
            pred = entry['predicted_scores']
            user = entry['user_scores']

            for key in errors.keys():
                if key in pred and key in user:
                    errors[key].append(abs(pred[key] - user[key]))

        # Calculate statistics
        self.feedback['statistics'] = {
            'total_feedback': len(self.feedback['entries']),
            'average_errors': {
                key: np.mean(vals) if vals else 0
                for key, vals in errors.items()
            }
        }

    def get_training_data(self) -> List[Tuple[Dict[str, float], Dict[str, float]]]:
        """
        Get training data from feedback.

        Returns:
            List of (features, user_scores) tuples
        """
        training_data = []

        for entry in self.feedback['entries']:
            features = entry['features']
            scores = entry['user_scores']
            training_data.append((features, scores))

        return training_data

    def get_statistics(self) -> Dict:
        """Get feedback statistics."""
        return self.feedback.get('statistics', {})

    def export_for_training(
        self,
        output_path: str,
        min_samples: int = 10
    ) -> bool:
        """
        Export feedback data in a format suitable for model retraining.

        Args:
            output_path: Path to save training data
            min_samples: Minimum number of samples required

        Returns:
            True if export successful, False otherwise
        """
        if len(self.feedback['entries']) < min_samples:
            return False

        training_data = {
            'samples': self.feedback['entries'],
            'count': len(self.feedback['entries']),
            'statistics': self.feedback['statistics']
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(training_data, indent=2, fp=f)

        return True
