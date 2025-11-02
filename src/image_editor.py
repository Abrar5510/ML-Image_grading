"""
Image Editor Module
Applies automatic edits to images based on suggestions.
"""

import numpy as np
import cv2
from typing import Dict, Tuple
from PIL import Image, ImageEnhance, ImageFilter
import scipy.ndimage as ndimage


class ImageEditor:
    """Applies automatic edits to improve image quality."""

    def __init__(self):
        """Initialize the image editor."""
        pass

    def apply_edits(
        self,
        image: np.ndarray,
        edit_params: Dict[str, float]
    ) -> np.ndarray:
        """
        Apply all edits to an image.

        Args:
            image: Input image (float32, normalized to 0-1, RGB)
            edit_params: Dictionary of editing parameters

        Returns:
            Edited image (float32, normalized to 0-1, RGB)
        """
        edited = image.copy()

        # Apply edits in optimal order
        # 1. Noise reduction (if needed)
        if edit_params.get('denoise', 0) > 0:
            edited = self.apply_denoising(edited, edit_params['denoise'])

        # 2. Exposure/brightness
        if edit_params.get('brightness', 0) != 0:
            edited = self.adjust_brightness(edited, edit_params['brightness'])

        # 3. Contrast
        if edit_params.get('contrast', 0) != 0:
            edited = self.adjust_contrast(edited, edit_params['contrast'])

        # 4. Saturation
        if edit_params.get('saturation', 0) != 0:
            edited = self.adjust_saturation(edited, edit_params['saturation'])

        # 5. Temperature/white balance
        if edit_params.get('temperature', 0) != 0:
            edited = self.adjust_temperature(edited, edit_params['temperature'])

        # 6. Sharpening
        if edit_params.get('sharpness', 0) > 0:
            edited = self.apply_sharpening(edited, edit_params['sharpness'])

        # 7. Vignette (if needed)
        if edit_params.get('vignette', 0) > 0:
            edited = self.apply_vignette(edited, edit_params['vignette'])

        return edited

    def adjust_brightness(self, image: np.ndarray, adjustment: float) -> np.ndarray:
        """
        Adjust image brightness.

        Args:
            image: Input image
            adjustment: Brightness adjustment (-1 to 1)

        Returns:
            Adjusted image
        """
        # Convert to HSV
        img_uint8 = (image * 255).astype(np.uint8)
        hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)

        # Adjust V channel
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1.0 + adjustment), 0, 255)

        # Convert back
        hsv_uint8 = hsv.astype(np.uint8)
        rgb = cv2.cvtColor(hsv_uint8, cv2.COLOR_HSV2RGB)

        return rgb.astype(np.float32) / 255.0

    def adjust_contrast(self, image: np.ndarray, adjustment: float) -> np.ndarray:
        """
        Adjust image contrast.

        Args:
            image: Input image
            adjustment: Contrast adjustment (-1 to 1)

        Returns:
            Adjusted image
        """
        # Contrast adjustment using PIL
        img_pil = Image.fromarray((image * 255).astype(np.uint8))
        enhancer = ImageEnhance.Contrast(img_pil)

        # Map adjustment to factor (0.5 to 1.5)
        factor = 1.0 + adjustment
        enhanced = enhancer.enhance(factor)

        return np.array(enhanced).astype(np.float32) / 255.0

    def adjust_saturation(self, image: np.ndarray, adjustment: float) -> np.ndarray:
        """
        Adjust color saturation.

        Args:
            image: Input image
            adjustment: Saturation adjustment (-1 to 1)

        Returns:
            Adjusted image
        """
        # Convert to HSV
        img_uint8 = (image * 255).astype(np.uint8)
        hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)

        # Adjust S channel
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + adjustment), 0, 255)

        # Convert back
        hsv_uint8 = hsv.astype(np.uint8)
        rgb = cv2.cvtColor(hsv_uint8, cv2.COLOR_HSV2RGB)

        return rgb.astype(np.float32) / 255.0

    def adjust_temperature(self, image: np.ndarray, kelvin_shift: float) -> np.ndarray:
        """
        Adjust color temperature.

        Args:
            image: Input image
            kelvin_shift: Temperature shift in kelvin (-100 to 100)
                         Positive = warmer, Negative = cooler

        Returns:
            Adjusted image
        """
        result = image.copy()

        # Simple temperature adjustment
        # Warm: increase red, decrease blue
        # Cool: decrease red, increase blue
        factor = kelvin_shift / 100.0

        if factor > 0:  # Warm
            result[:, :, 0] = np.clip(result[:, :, 0] * (1 + factor * 0.1), 0, 1)  # R
            result[:, :, 2] = np.clip(result[:, :, 2] * (1 - factor * 0.1), 0, 1)  # B
        else:  # Cool
            result[:, :, 0] = np.clip(result[:, :, 0] * (1 + factor * 0.1), 0, 1)  # R
            result[:, :, 2] = np.clip(result[:, :, 2] * (1 - factor * 0.1), 0, 1)  # B

        return result

    def apply_sharpening(self, image: np.ndarray, amount: float) -> np.ndarray:
        """
        Apply sharpening to image.

        Args:
            image: Input image
            amount: Sharpening amount (0 to 1)

        Returns:
            Sharpened image
        """
        # Convert to uint8 for OpenCV
        img_uint8 = (image * 255).astype(np.uint8)

        # Create sharpening kernel
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ]) * amount

        # Adjust center
        kernel[1, 1] = 1 + 4 * amount

        # Apply sharpening
        sharpened = cv2.filter2D(img_uint8, -1, kernel)

        return sharpened.astype(np.float32) / 255.0

    def apply_denoising(self, image: np.ndarray, strength: float) -> np.ndarray:
        """
        Apply noise reduction.

        Args:
            image: Input image
            strength: Denoising strength (0 to 1)

        Returns:
            Denoised image
        """
        # Convert to uint8
        img_uint8 = (image * 255).astype(np.uint8)

        # Apply Non-Local Means Denoising
        h = int(strength * 20)  # Filter strength
        denoised = cv2.fastNlMeansDenoisingColored(
            img_uint8,
            None,
            h=h,
            hColor=h,
            templateWindowSize=7,
            searchWindowSize=21
        )

        return denoised.astype(np.float32) / 255.0

    def apply_vignette(self, image: np.ndarray, strength: float) -> np.ndarray:
        """
        Apply vignette effect to focus attention on center.

        Args:
            image: Input image
            strength: Vignette strength (0 to 1)

        Returns:
            Image with vignette
        """
        h, w = image.shape[:2]

        # Create radial gradient
        Y, X = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2

        # Calculate distance from center (normalized)
        dist_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
        max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
        dist_from_center = dist_from_center / max_dist

        # Create vignette mask
        vignette = 1 - (dist_from_center ** 2) * strength
        vignette = np.clip(vignette, 0, 1)

        # Apply vignette
        result = image.copy()
        for i in range(3):  # Apply to all channels
            result[:, :, i] = result[:, :, i] * vignette

        return result

    def auto_enhance(self, image: np.ndarray) -> np.ndarray:
        """
        Apply automatic enhancement (histogram equalization).

        Args:
            image: Input image

        Returns:
            Enhanced image
        """
        # Convert to LAB color space
        img_uint8 = (image * 255).astype(np.uint8)
        lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])

        # Convert back to RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        return enhanced.astype(np.float32) / 255.0

    def create_comparison(
        self,
        original: np.ndarray,
        edited: np.ndarray
    ) -> np.ndarray:
        """
        Create side-by-side comparison image.

        Args:
            original: Original image
            edited: Edited image

        Returns:
            Comparison image (side by side)
        """
        # Ensure same size
        h, w = original.shape[:2]
        edited_resized = cv2.resize(
            (edited * 255).astype(np.uint8),
            (w, h)
        ).astype(np.float32) / 255.0

        # Add labels
        original_labeled = self._add_label(original, "ORIGINAL")
        edited_labeled = self._add_label(edited_resized, "EDITED")

        # Concatenate horizontally
        comparison = np.concatenate([original_labeled, edited_labeled], axis=1)

        return comparison

    def _add_label(self, image: np.ndarray, text: str) -> np.ndarray:
        """Add text label to image."""
        img_uint8 = (image * 255).astype(np.uint8).copy()

        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        color = (255, 255, 255)

        # Get text size
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

        # Position at top center
        x = (img_uint8.shape[1] - text_width) // 2
        y = 30

        # Add background rectangle
        cv2.rectangle(
            img_uint8,
            (x - 10, y - text_height - 10),
            (x + text_width + 10, y + 10),
            (0, 0, 0),
            -1
        )

        # Add text
        cv2.putText(img_uint8, text, (x, y), font, font_scale, color, thickness)

        return img_uint8.astype(np.float32) / 255.0
