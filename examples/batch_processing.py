"""
Batch Processing Example
Demonstrates how to process multiple images in batch mode.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pipeline import ImageGradingPipeline


def main():
    """Batch processing example."""

    # Directory containing images
    input_dir = 'path/to/images'  # Change this to your images directory

    # Path to trained model (optional)
    model_path = 'models/trained_model.h5'

    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = ImageGradingPipeline(
        model_path=model_path if os.path.exists(model_path) else None,
        use_heuristic=True
    )

    # Process batch
    print(f"\nProcessing all images in: {input_dir}")
    pipeline.process_batch(
        input_dir=input_dir,
        output_dir='output/batch',
        apply_edits=True,
        pattern='*.CR2'  # Can also use '*.jpg', '*.png', etc.
    )

    print("\n✓ Batch processing complete!")
    print("Check output/batch/ for results and batch_summary.txt")


if __name__ == '__main__':
    main()
