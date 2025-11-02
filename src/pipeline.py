"""
Main Pipeline
Complete pipeline for image quality assessment, scoring, and improvement.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Dict
import numpy as np

from image_loader import CR2ImageLoader
from feature_extractor import ImageFeatureExtractor
from scoring_model import ImageScoringModel
from suggestion_engine import SuggestionEngine
from image_editor import ImageEditor
from feedback_system import FeedbackCollector


class ImageGradingPipeline:
    """
    Complete pipeline for CR2 image grading and improvement.

    Workflow:
    1. Load CR2 image
    2. Extract features
    3. Score the image
    4. Generate improvement suggestions
    5. Apply automatic edits
    6. Save results
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        use_heuristic: bool = True,
        feedback_path: str = 'data/feedback'
    ):
        """
        Initialize the pipeline.

        Args:
            model_path: Path to trained model (None to use heuristic scoring)
            use_heuristic: Whether to use heuristic scoring when model unavailable
            feedback_path: Path to feedback data directory
        """
        self.image_loader = CR2ImageLoader()
        self.feature_extractor = ImageFeatureExtractor()
        self.suggestion_engine = SuggestionEngine()
        self.image_editor = ImageEditor()
        self.feedback_collector = FeedbackCollector(feedback_path)

        # Initialize model
        self.model = ImageScoringModel()
        self.use_heuristic = use_heuristic

        if model_path and os.path.exists(model_path):
            print(f"Loading trained model from {model_path}...")
            try:
                self.model.load_model(model_path)
                print("✓ Model loaded successfully")
            except Exception as e:
                print(f"⚠ Failed to load model: {e}")
                if not use_heuristic:
                    raise
                print("Falling back to heuristic scoring")
        elif use_heuristic:
            print("Using heuristic-based scoring (no trained model)")
        else:
            raise ValueError("No model available and heuristic mode disabled")

    def process_image(
        self,
        image_path: str,
        output_dir: str = 'output',
        apply_edits: bool = True,
        collect_feedback: bool = False,
        save_comparison: bool = True
    ) -> Dict:
        """
        Process a single image through the complete pipeline.

        Args:
            image_path: Path to input image
            output_dir: Directory to save output files
            apply_edits: Whether to apply automatic edits
            collect_feedback: Whether to collect user feedback
            save_comparison: Whether to save before/after comparison

        Returns:
            Dictionary with results
        """
        print("\n" + "=" * 70)
        print(f"PROCESSING IMAGE: {Path(image_path).name}")
        print("=" * 70)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # 1. Load image
        print("\n[1/6] Loading image...")
        original, preprocessed = self.image_loader.load_image(image_path)
        print(f"  ✓ Image loaded: {original.shape}")

        # 2. Extract features
        print("\n[2/6] Extracting features...")
        features_dict = self.feature_extractor.extract_all_features(original)
        feature_vector = self.feature_extractor.get_feature_vector(original)
        print(f"  ✓ Extracted {len(features_dict)} features")

        # 3. Score the image
        print("\n[3/6] Scoring image quality...")
        if self.model.is_trained:
            scores = self.model.predict_score(preprocessed, feature_vector)
        else:
            scores = self.model.heuristic_score(features_dict)

        print("\n  Image Quality Scores:")
        print("  " + "-" * 40)
        for key, value in scores.items():
            score_out_of_100 = value * 100
            bar = "█" * int(score_out_of_100 / 5) + "░" * (20 - int(score_out_of_100 / 5))
            print(f"  {key:20s}: {score_out_of_100:5.1f}/100 [{bar}]")

        # 4. Generate suggestions
        print("\n[4/6] Generating improvement suggestions...")
        suggestions = self.suggestion_engine.generate_suggestions(features_dict, scores)

        if suggestions:
            print(f"  ✓ Generated {len(suggestions)} suggestions\n")
            print(self.suggestion_engine.format_suggestions(suggestions))
        else:
            print("  ✓ No improvements needed - excellent image quality!")

        # 5. Apply automatic edits (if requested)
        edited_image = None
        if apply_edits and suggestions:
            print("\n[5/6] Applying automatic edits...")
            edit_params = self.suggestion_engine.get_edit_parameters(suggestions)

            # Show what will be edited
            print("  Edit parameters:")
            for param, value in edit_params.items():
                if value != 0:
                    print(f"    {param}: {value:+.2f}")

            edited_image = self.image_editor.apply_edits(original, edit_params)
            print("  ✓ Edits applied")
        else:
            print("\n[5/6] Skipping automatic edits")
            edited_image = original

        # 6. Save results
        print("\n[6/6] Saving results...")
        base_name = Path(image_path).stem
        results = {
            'image_path': image_path,
            'scores': scores,
            'features': features_dict,
            'suggestions': [str(s) for s in suggestions]
        }

        # Save edited image
        if apply_edits:
            edited_path = os.path.join(output_dir, f"{base_name}_edited.jpg")
            self.image_loader.save_image(edited_image, edited_path)
            results['edited_path'] = edited_path
            print(f"  ✓ Edited image: {edited_path}")

        # Save comparison
        if save_comparison and apply_edits:
            comparison_path = os.path.join(output_dir, f"{base_name}_comparison.jpg")
            comparison = self.image_editor.create_comparison(original, edited_image)
            self.image_loader.save_image(comparison, comparison_path)
            results['comparison_path'] = comparison_path
            print(f"  ✓ Comparison image: {comparison_path}")

        # Save report
        report_path = os.path.join(output_dir, f"{base_name}_report.txt")
        self._save_report(report_path, results)
        results['report_path'] = report_path
        print(f"  ✓ Report: {report_path}")

        # Collect feedback (if requested)
        if collect_feedback:
            print("\n" + "=" * 70)
            user_scores = self.feedback_collector.collect_feedback_interactive(
                image_path=image_path,
                predicted_scores=scores,
                features=features_dict
            )
            results['user_scores'] = user_scores

        print("\n" + "=" * 70)
        print("✓ PROCESSING COMPLETE")
        print("=" * 70 + "\n")

        return results

    def process_batch(
        self,
        input_dir: str,
        output_dir: str = 'output',
        apply_edits: bool = True,
        pattern: str = '*.CR2'
    ):
        """
        Process multiple images in batch mode.

        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save outputs
            apply_edits: Whether to apply automatic edits
            pattern: File pattern to match (e.g., '*.CR2', '*.jpg')
        """
        import glob

        image_paths = glob.glob(os.path.join(input_dir, pattern))

        if not image_paths:
            print(f"No images found matching {pattern} in {input_dir}")
            return

        print(f"\nProcessing {len(image_paths)} images...")

        results = []
        for i, image_path in enumerate(image_paths, 1):
            print(f"\n{'=' * 70}")
            print(f"Image {i}/{len(image_paths)}")
            print(f"{'=' * 70}")

            try:
                result = self.process_image(
                    image_path,
                    output_dir=output_dir,
                    apply_edits=apply_edits,
                    collect_feedback=False,
                    save_comparison=True
                )
                results.append(result)
            except Exception as e:
                print(f"⚠ Error processing {image_path}: {e}")
                continue

        # Save batch summary
        self._save_batch_summary(results, output_dir)

        print(f"\n✓ Batch processing complete! Processed {len(results)}/{len(image_paths)} images")

    def _save_report(self, report_path: str, results: Dict):
        """Save detailed report to file."""
        with open(report_path, 'w') as f:
            f.write("IMAGE QUALITY ASSESSMENT REPORT\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Image: {results['image_path']}\n\n")

            f.write("QUALITY SCORES:\n")
            f.write("-" * 70 + "\n")
            for key, value in results['scores'].items():
                f.write(f"{key:25s}: {value * 100:6.2f}/100\n")

            f.write("\n\nIMPROVEMENT SUGGESTIONS:\n")
            f.write("-" * 70 + "\n")
            if results['suggestions']:
                for suggestion in results['suggestions']:
                    f.write(f"  • {suggestion}\n")
            else:
                f.write("  No improvements needed - excellent image quality!\n")

            f.write("\n\nFEATURE ANALYSIS:\n")
            f.write("-" * 70 + "\n")
            for key, value in sorted(results['features'].items()):
                f.write(f"{key:30s}: {value:8.4f}\n")

    def _save_batch_summary(self, results: list, output_dir: str):
        """Save batch processing summary."""
        summary_path = os.path.join(output_dir, 'batch_summary.txt')

        with open(summary_path, 'w') as f:
            f.write("BATCH PROCESSING SUMMARY\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Total images processed: {len(results)}\n\n")

            # Calculate statistics
            if results:
                overall_scores = [r['scores'].get('overall_score', 0) for r in results]
                avg_score = np.mean(overall_scores) * 100
                min_score = np.min(overall_scores) * 100
                max_score = np.max(overall_scores) * 100

                f.write("QUALITY STATISTICS:\n")
                f.write("-" * 70 + "\n")
                f.write(f"Average Quality Score: {avg_score:.2f}/100\n")
                f.write(f"Minimum Quality Score: {min_score:.2f}/100\n")
                f.write(f"Maximum Quality Score: {max_score:.2f}/100\n\n")

                f.write("INDIVIDUAL RESULTS:\n")
                f.write("-" * 70 + "\n")
                for result in results:
                    name = Path(result['image_path']).name
                    score = result['scores'].get('overall_score', 0) * 100
                    f.write(f"{name:40s}: {score:6.2f}/100\n")

        print(f"\n✓ Batch summary saved to {summary_path}")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description='ML Image Grading Pipeline - CR2 image quality assessment and improvement'
    )

    parser.add_argument('input', type=str, help='Input image path or directory')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    parser.add_argument('--model', type=str, default=None, help='Path to trained model')
    parser.add_argument('--batch', action='store_true', help='Batch processing mode')
    parser.add_argument('--pattern', type=str, default='*.CR2', help='File pattern for batch mode')
    parser.add_argument('--no-edit', action='store_true', help='Skip automatic editing')
    parser.add_argument('--feedback', action='store_true', help='Collect user feedback')
    parser.add_argument('--feedback-path', type=str, default='data/feedback',
                       help='Path to feedback directory')

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = ImageGradingPipeline(
        model_path=args.model,
        use_heuristic=True,
        feedback_path=args.feedback_path
    )

    # Process image(s)
    if args.batch:
        pipeline.process_batch(
            input_dir=args.input,
            output_dir=args.output,
            apply_edits=not args.no_edit,
            pattern=args.pattern
        )
    else:
        pipeline.process_image(
            image_path=args.input,
            output_dir=args.output,
            apply_edits=not args.no_edit,
            collect_feedback=args.feedback
        )


if __name__ == '__main__':
    main()
