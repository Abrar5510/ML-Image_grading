"""
Suggestion Engine Module
Generates improvement suggestions based on image analysis and scores.
"""

import numpy as np
from typing import Dict, List, Tuple


class ImprovementSuggestion:
    """Represents a single improvement suggestion."""

    def __init__(self, category: str, description: str, priority: str, parameters: Dict = None):
        """
        Initialize a suggestion.

        Args:
            category: Category of improvement (composition, color, technical)
            description: Human-readable description
            priority: Priority level (high, medium, low)
            parameters: Parameters for automatic editing
        """
        self.category = category
        self.description = description
        self.priority = priority
        self.parameters = parameters or {}

    def __repr__(self):
        return f"[{self.priority.upper()}] {self.category}: {self.description}"


class SuggestionEngine:
    """Generates improvement suggestions based on image features and scores."""

    def __init__(self):
        """Initialize the suggestion engine."""
        self.thresholds = {
            'low_contrast': 0.15,
            'low_sharpness': 0.3,
            'low_vibrancy': 0.3,
            'overexposure': 0.05,
            'underexposure': 0.05,
            'high_noise': 0.15,
            'low_thirds_score': 0.2,
            'poor_exposure': 0.4,
        }

    def generate_suggestions(
        self,
        features: Dict[str, float],
        scores: Dict[str, float]
    ) -> List[ImprovementSuggestion]:
        """
        Generate improvement suggestions.

        Args:
            features: Dictionary of extracted features
            scores: Dictionary of quality scores

        Returns:
            List of improvement suggestions
        """
        suggestions = []

        # Composition suggestions
        suggestions.extend(self._analyze_composition(features, scores))

        # Color suggestions
        suggestions.extend(self._analyze_color(features, scores))

        # Technical suggestions (exposure, contrast, sharpness, noise)
        suggestions.extend(self._analyze_technical(features, scores))

        # Sort by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        suggestions.sort(key=lambda x: priority_order[x.priority])

        return suggestions

    def _analyze_composition(
        self,
        features: Dict[str, float],
        scores: Dict[str, float]
    ) -> List[ImprovementSuggestion]:
        """Analyze composition and generate suggestions."""
        suggestions = []

        # Rule of thirds
        thirds_score = features.get('rule_of_thirds_score', 0)
        if thirds_score < self.thresholds['low_thirds_score']:
            suggestions.append(ImprovementSuggestion(
                category='composition',
                description='Consider positioning key elements along rule-of-thirds lines for better composition',
                priority='medium',
                parameters={'type': 'crop', 'rule_of_thirds': True}
            ))

        # Center focus
        center_focus = features.get('center_focus', 0)
        if center_focus < 0.2:
            suggestions.append(ImprovementSuggestion(
                category='composition',
                description='Main subject appears to lack prominence. Consider cropping to emphasize the focal point',
                priority='medium',
                parameters={'type': 'crop', 'center_emphasis': True}
            ))

        # Complexity
        complexity = features.get('complexity', 0)
        if complexity < 0.05:
            suggestions.append(ImprovementSuggestion(
                category='composition',
                description='Image appears simple. Consider adding visual interest or adjusting framing',
                priority='low',
                parameters={}
            ))
        elif complexity > 0.4:
            suggestions.append(ImprovementSuggestion(
                category='composition',
                description='Image appears cluttered. Consider simplifying composition or using selective focus',
                priority='medium',
                parameters={'type': 'vignette', 'strength': 0.3}
            ))

        return suggestions

    def _analyze_color(
        self,
        features: Dict[str, float],
        scores: Dict[str, float]
    ) -> List[ImprovementSuggestion]:
        """Analyze color and generate suggestions."""
        suggestions = []

        # Color vibrancy
        vibrancy = features.get('color_vibrancy', 0)
        if vibrancy < self.thresholds['low_vibrancy']:
            suggestions.append(ImprovementSuggestion(
                category='color',
                description='Increase saturation to make colors more vibrant and appealing',
                priority='high',
                parameters={'type': 'saturation', 'adjustment': 0.3}
            ))
        elif vibrancy > 0.8:
            suggestions.append(ImprovementSuggestion(
                category='color',
                description='Colors appear oversaturated. Consider reducing saturation for a more natural look',
                priority='medium',
                parameters={'type': 'saturation', 'adjustment': -0.2}
            ))

        # Color diversity
        diversity = features.get('color_diversity', 0)
        if diversity < 1.5:
            suggestions.append(ImprovementSuggestion(
                category='color',
                description='Limited color palette. This can be good for minimalism or consider enhancing color variety',
                priority='low',
                parameters={}
            ))

        # Warm/cool balance
        warm_cool_ratio = features.get('warm_cool_ratio', 1.0)
        if warm_cool_ratio < 0.5:
            suggestions.append(ImprovementSuggestion(
                category='color',
                description='Image has cool tone. Consider warming up for more inviting feel',
                priority='low',
                parameters={'type': 'temperature', 'adjustment': 15}
            ))
        elif warm_cool_ratio > 2.0:
            suggestions.append(ImprovementSuggestion(
                category='color',
                description='Image has very warm tone. Consider cooling down if needed',
                priority='low',
                parameters={'type': 'temperature', 'adjustment': -15}
            ))

        return suggestions

    def _analyze_technical(
        self,
        features: Dict[str, float],
        scores: Dict[str, float]
    ) -> List[ImprovementSuggestion]:
        """Analyze technical aspects and generate suggestions."""
        suggestions = []

        # Contrast
        contrast = features.get('rms_contrast', 0)
        if contrast < self.thresholds['low_contrast']:
            suggestions.append(ImprovementSuggestion(
                category='technical',
                description='Increase contrast to add depth and make the image pop',
                priority='high',
                parameters={'type': 'contrast', 'adjustment': 0.3}
            ))
        elif contrast > 0.4:
            suggestions.append(ImprovementSuggestion(
                category='technical',
                description='Reduce contrast to recover details in highlights and shadows',
                priority='medium',
                parameters={'type': 'contrast', 'adjustment': -0.2}
            ))

        # Sharpness
        sharpness = features.get('laplacian_variance', 0)
        if sharpness < self.thresholds['low_sharpness']:
            suggestions.append(ImprovementSuggestion(
                category='technical',
                description='Apply sharpening to enhance image clarity and detail',
                priority='high',
                parameters={'type': 'sharpen', 'amount': 0.5}
            ))

        # Exposure issues
        overexposure = features.get('overexposure', 0)
        underexposure = features.get('underexposure', 0)

        if overexposure > self.thresholds['overexposure']:
            suggestions.append(ImprovementSuggestion(
                category='technical',
                description=f'Reduce exposure to recover highlights ({overexposure * 100:.1f}% overexposed)',
                priority='high',
                parameters={'type': 'exposure', 'adjustment': -0.3}
            ))

        if underexposure > self.thresholds['underexposure']:
            suggestions.append(ImprovementSuggestion(
                category='technical',
                description=f'Increase exposure to reveal shadow details ({underexposure * 100:.1f}% underexposed)',
                priority='high',
                parameters={'type': 'exposure', 'adjustment': 0.3}
            ))

        # Overall brightness
        brightness = features.get('average_brightness', 0.5)
        if brightness < 0.3:
            suggestions.append(ImprovementSuggestion(
                category='technical',
                description='Image is too dark overall. Increase brightness',
                priority='high',
                parameters={'type': 'brightness', 'adjustment': 0.2}
            ))
        elif brightness > 0.7:
            suggestions.append(ImprovementSuggestion(
                category='technical',
                description='Image is too bright overall. Decrease brightness',
                priority='medium',
                parameters={'type': 'brightness', 'adjustment': -0.2}
            ))

        # Noise
        noise = features.get('noise_estimate', 0)
        if noise > self.thresholds['high_noise']:
            suggestions.append(ImprovementSuggestion(
                category='technical',
                description='Apply noise reduction to clean up the image',
                priority='medium',
                parameters={'type': 'denoise', 'strength': 0.5}
            ))

        return suggestions

    def format_suggestions(self, suggestions: List[ImprovementSuggestion]) -> str:
        """
        Format suggestions as a readable string.

        Args:
            suggestions: List of suggestions

        Returns:
            Formatted string
        """
        if not suggestions:
            return "No improvements needed - image quality is excellent!"

        output = ["Image Improvement Suggestions:", "=" * 50, ""]

        # Group by category
        categories = {}
        for sug in suggestions:
            if sug.category not in categories:
                categories[sug.category] = []
            categories[sug.category].append(sug)

        for category, sugs in categories.items():
            output.append(f"\n{category.upper()}:")
            output.append("-" * 30)
            for sug in sugs:
                priority_symbol = {
                    'high': '🔴',
                    'medium': '🟡',
                    'low': '🔵'
                }.get(sug.priority, '')
                output.append(f"  {priority_symbol} [{sug.priority}] {sug.description}")

        return "\n".join(output)

    def get_edit_parameters(
        self,
        suggestions: List[ImprovementSuggestion]
    ) -> Dict[str, any]:
        """
        Extract editing parameters from suggestions.

        Args:
            suggestions: List of suggestions

        Returns:
            Dictionary of editing parameters
        """
        edit_params = {
            'brightness': 0,
            'contrast': 0,
            'saturation': 0,
            'sharpness': 0,
            'temperature': 0,
            'denoise': 0,
            'vignette': 0,
        }

        for sug in suggestions:
            if not sug.parameters:
                continue

            param_type = sug.parameters.get('type')

            if param_type == 'brightness':
                edit_params['brightness'] += sug.parameters.get('adjustment', 0)
            elif param_type == 'contrast':
                edit_params['contrast'] += sug.parameters.get('adjustment', 0)
            elif param_type == 'saturation':
                edit_params['saturation'] += sug.parameters.get('adjustment', 0)
            elif param_type == 'sharpen':
                edit_params['sharpness'] = max(edit_params['sharpness'], sug.parameters.get('amount', 0))
            elif param_type == 'temperature':
                edit_params['temperature'] += sug.parameters.get('adjustment', 0)
            elif param_type == 'denoise':
                edit_params['denoise'] = max(edit_params['denoise'], sug.parameters.get('strength', 0))
            elif param_type == 'vignette':
                edit_params['vignette'] = max(edit_params['vignette'], sug.parameters.get('strength', 0))
            elif param_type == 'exposure':
                # Exposure affects brightness
                edit_params['brightness'] += sug.parameters.get('adjustment', 0)

        return edit_params
