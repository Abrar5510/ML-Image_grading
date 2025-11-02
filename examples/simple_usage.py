"""
Simple Usage Example
Demonstrates basic usage of the ML Image Grading system.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pipeline import ImageGradingPipeline


def main():
    """Simple example of processing an image."""

    # Path to your image
    image_path = 'path/to/your/image.CR2'  # Change this to your image path

    # Path to trained model (optional - will use heuristic if not available)
    model_path = 'models/trained_model.h5'

    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = ImageGradingPipeline(
        model_path=model_path if os.path.exists(model_path) else None,
        use_heuristic=True  # Fall back to heuristic if no model
    )

    # Process image
    print(f"\nProcessing image: {image_path}")
    results = pipeline.process_image(
        image_path=image_path,
        output_dir='output',
        apply_edits=True,
        collect_feedback=False,
        save_comparison=True
    )

    # Display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print("\nQuality Scores:")
    for key, value in results['scores'].items():
        print(f"  {key:25s}: {value * 100:6.2f}/100")

    print(f"\nSuggestions: {len(results['suggestions'])}")
    for suggestion in results['suggestions']:
        print(f"  • {suggestion}")

    print(f"\nOutput files:")
    if 'edited_path' in results:
        print(f"  Edited image: {results['edited_path']}")
    if 'comparison_path' in results:
        print(f"  Comparison: {results['comparison_path']}")
    if 'report_path' in results:
        print(f"  Report: {results['report_path']}")

    print("\n✓ Done!")


if __name__ == '__main__':
    main()
