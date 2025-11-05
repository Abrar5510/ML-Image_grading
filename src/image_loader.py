"""
Image Loader Module
Handles loading CR2 (Canon Raw) images and converting them to processable formats.
"""

import imageio
import numpy as np
from PIL import Image
import cv2
from typing import Tuple, Optional


class CR2ImageLoader:
    """Loads and preprocesses CR2 images."""

    def __init__(self, target_size: Tuple[int, int] = (512, 512)):
        """
        Initialize the image loader.

        Args:
            target_size: Target size for resizing images (width, height)
        """
        self.target_size = target_size

    def load_cr2(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a CR2 image file using OpenCV.

        Args:
            file_path: Path to the CR2 file

        Returns:
            Tuple of (original_image, preprocessed_image)
            - original_image: Full resolution RGB image
            - preprocessed_image: Resized and normalized image for ML processing

        Note:
            CR2 support in OpenCV depends on system configuration.
            If CR2 loading fails, consider converting files to JPEG/PNG first.
        """
        try:
            # Read CR2 file using OpenCV
            # IMREAD_COLOR reads as BGR, we'll convert to RGB
            img_bgr = cv2.imread(file_path, cv2.IMREAD_COLOR)

            if img_bgr is None:
                raise RuntimeError(
                    f"Failed to load CR2 file. OpenCV couldn't read the file. "
                    f"Consider converting CR2 files to JPEG or PNG format first."
                )

            # Convert BGR to RGB
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Convert to float32 for processing
            original = rgb.astype(np.float32) / 255.0

            # Create preprocessed version
            preprocessed = self._preprocess_image(original)

            return original, preprocessed

        except Exception as e:
            raise RuntimeError(f"Error loading CR2 file {file_path}: {str(e)}")

    def load_image(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load an image file (CR2, JPEG, PNG, etc.).

        Args:
            file_path: Path to the image file

        Returns:
            Tuple of (original_image, preprocessed_image)
        """
        file_path_lower = file_path.lower()

        # Handle CR2 files
        if file_path_lower.endswith('.cr2'):
            return self.load_cr2(file_path)

        # Handle standard image formats
        try:
            img = Image.open(file_path).convert('RGB')
            original = np.array(img).astype(np.float32) / 255.0
            preprocessed = self._preprocess_image(original)
            return original, preprocessed

        except Exception as e:
            raise RuntimeError(f"Error loading image file {file_path}: {str(e)}")

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for ML model.

        Args:
            image: Input image (float32, normalized to 0-1)

        Returns:
            Preprocessed image
        """
        # Convert to uint8 for cv2 operations
        img_uint8 = (image * 255).astype(np.uint8)

        # Resize while maintaining aspect ratio
        h, w = img_uint8.shape[:2]
        target_w, target_h = self.target_size

        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        # Resize
        resized = cv2.resize(img_uint8, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Pad to target size
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        padded = cv2.copyMakeBorder(
            resized,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )

        # Convert back to float32 and normalize
        preprocessed = padded.astype(np.float32) / 255.0

        return preprocessed

    def save_image(self, image: np.ndarray, output_path: str, quality: int = 95):
        """
        Save an image to disk.

        Args:
            image: Image array (float32, normalized to 0-1)
            output_path: Path to save the image
            quality: JPEG quality (1-100)
        """
        # Convert to uint8
        img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)

        # Save using PIL
        img_pil = Image.fromarray(img_uint8)
        img_pil.save(output_path, quality=quality)
