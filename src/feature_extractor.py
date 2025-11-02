"""
Feature Extractor Module
Extracts image characteristics including composition, colors, contrast, and other attributes.
"""

import numpy as np
import cv2
from scipy import ndimage
from typing import Dict, Tuple
import matplotlib.pyplot as plt


class ImageFeatureExtractor:
    """Extracts features from images for quality assessment."""

    def __init__(self):
        """Initialize the feature extractor."""
        pass

    def extract_all_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract all features from an image.

        Args:
            image: Input image (float32, normalized to 0-1, RGB)

        Returns:
            Dictionary of feature names and values
        """
        features = {}

        # Composition features
        features.update(self.extract_composition_features(image))

        # Color features
        features.update(self.extract_color_features(image))

        # Contrast features
        features.update(self.extract_contrast_features(image))

        # Sharpness features
        features.update(self.extract_sharpness_features(image))

        # Exposure features
        features.update(self.extract_exposure_features(image))

        # Noise features
        features.update(self.extract_noise_features(image))

        return features

    def extract_composition_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract composition-related features.

        Features include:
        - Rule of thirds alignment
        - Edge density
        - Symmetry
        - Complexity
        """
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        h, w = gray.shape

        features = {}

        # Edge detection for composition analysis
        edges = cv2.Canny(gray, 50, 150)

        # Rule of thirds - check edge density at intersection points
        third_h, third_w = h // 3, w // 3
        roi_size = min(h, w) // 10  # Region of interest size

        thirds_points = [
            (third_w, third_h), (2 * third_w, third_h),
            (third_w, 2 * third_h), (2 * third_w, 2 * third_h)
        ]

        thirds_density = []
        for x, y in thirds_points:
            roi = edges[max(0, y - roi_size):min(h, y + roi_size),
                       max(0, x - roi_size):min(w, x + roi_size)]
            thirds_density.append(np.mean(roi) / 255.0)

        features['rule_of_thirds_score'] = np.mean(thirds_density)

        # Overall edge density
        features['edge_density'] = np.mean(edges) / 255.0

        # Symmetry (vertical)
        left_half = gray[:, :w // 2]
        right_half = cv2.flip(gray[:, w // 2:], 1)
        min_width = min(left_half.shape[1], right_half.shape[1])
        symmetry_diff = np.abs(left_half[:, :min_width].astype(float) -
                              right_half[:, :min_width].astype(float))
        features['vertical_symmetry'] = 1.0 - (np.mean(symmetry_diff) / 255.0)

        # Complexity (using edge count)
        features['complexity'] = np.sum(edges > 0) / (h * w)

        # Center focus (edge density in center vs edges)
        center_h, center_w = h // 2, w // 2
        center_size = min(h, w) // 3
        center_region = edges[center_h - center_size:center_h + center_size,
                             center_w - center_size:center_w + center_size]
        features['center_focus'] = np.mean(center_region) / 255.0

        return features

    def extract_color_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract color-related features.

        Features include:
        - Color vibrancy
        - Color harmony
        - Dominant colors
        - Color distribution
        """
        features = {}

        # Convert to HSV
        hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        # Color vibrancy (saturation)
        features['color_vibrancy'] = np.mean(s) / 255.0
        features['saturation_std'] = np.std(s) / 255.0

        # Color diversity (histogram spread)
        hist, _ = np.histogram(h, bins=16, range=(0, 180))
        hist = hist / np.sum(hist)
        features['color_diversity'] = -np.sum(hist * np.log(hist + 1e-10))  # Entropy

        # Dominant color strength
        features['dominant_color_strength'] = np.max(hist)

        # Warm vs cool colors
        warm_mask = ((h >= 0) & (h < 30)) | (h >= 150)
        cool_mask = (h >= 60) & (h < 150)
        features['warm_cool_ratio'] = np.sum(warm_mask) / (np.sum(cool_mask) + 1)

        # Color balance (RGB channels)
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        features['color_balance'] = 1.0 - (np.std([np.mean(r), np.mean(g), np.mean(b)]))

        return features

    def extract_contrast_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract contrast-related features.
        """
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

        features = {}

        # RMS contrast
        features['rms_contrast'] = np.std(gray) / 255.0

        # Michelson contrast
        max_lum = np.max(gray)
        min_lum = np.min(gray)
        if max_lum + min_lum > 0:
            features['michelson_contrast'] = (max_lum - min_lum) / (max_lum + min_lum)
        else:
            features['michelson_contrast'] = 0.0

        # Dynamic range
        features['dynamic_range'] = (max_lum - min_lum) / 255.0

        # Local contrast (using standard deviation in local patches)
        kernel_size = 15
        local_std = ndimage.generic_filter(gray.astype(float), np.std, size=kernel_size)
        features['local_contrast'] = np.mean(local_std) / 255.0

        return features

    def extract_sharpness_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract sharpness-related features.
        """
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

        features = {}

        # Laplacian variance (focus measure)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        features['laplacian_variance'] = np.var(laplacian) / 1000.0  # Normalize

        # Tenengrad (gradient-based sharpness)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gx ** 2 + gy ** 2)
        features['tenengrad'] = np.mean(gradient_magnitude) / 255.0

        # Edge sharpness (percentage of strong edges)
        edges = cv2.Canny(gray, 100, 200)
        features['edge_sharpness'] = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])

        return features

    def extract_exposure_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract exposure-related features.
        """
        # Convert to grayscale for luminance
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

        features = {}

        # Average brightness
        features['average_brightness'] = np.mean(gray) / 255.0

        # Brightness distribution
        features['brightness_std'] = np.std(gray) / 255.0

        # Overexposure (percentage of near-white pixels)
        features['overexposure'] = np.sum(gray > 240) / gray.size

        # Underexposure (percentage of near-black pixels)
        features['underexposure'] = np.sum(gray < 15) / gray.size

        # Histogram peaks
        hist, bins = np.histogram(gray, bins=256, range=(0, 256))
        hist = hist / np.sum(hist)

        # Check for proper exposure distribution
        mid_range = hist[64:192]
        features['mid_tone_distribution'] = np.sum(mid_range)

        return features

    def extract_noise_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract noise-related features.
        """
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

        features = {}

        # Estimate noise using high-frequency components
        # Median filter to get noise estimate
        median_filtered = cv2.medianBlur(gray, 5)
        noise = gray.astype(float) - median_filtered.astype(float)

        features['noise_estimate'] = np.std(noise) / 255.0

        # Signal-to-noise ratio estimate
        signal_power = np.mean(gray ** 2)
        noise_power = np.mean(noise ** 2)
        if noise_power > 0:
            features['snr_estimate'] = 10 * np.log10(signal_power / (noise_power + 1e-10))
        else:
            features['snr_estimate'] = 100.0  # Very high SNR

        features['snr_estimate'] = min(features['snr_estimate'] / 50.0, 1.0)  # Normalize

        return features

    def get_feature_vector(self, image: np.ndarray) -> np.ndarray:
        """
        Get feature vector as a numpy array.

        Args:
            image: Input image

        Returns:
            Feature vector as numpy array
        """
        features = self.extract_all_features(image)
        # Sort by key for consistent ordering
        sorted_keys = sorted(features.keys())
        return np.array([features[k] for k in sorted_keys])

    def get_feature_names(self, image: np.ndarray) -> list:
        """
        Get feature names in the same order as get_feature_vector.

        Args:
            image: Input image (used to extract features once)

        Returns:
            List of feature names
        """
        features = self.extract_all_features(image)
        return sorted(features.keys())
