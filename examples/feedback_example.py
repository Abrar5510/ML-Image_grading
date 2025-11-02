"""
Feedback Collection Example
Demonstrates how to collect user feedback for continuous learning.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pipeline import ImageGradingPipeline
from feedback_system import ContinuousLearner


def process_with_feedback():
    """Process image and collect feedback."""

    image_path = 'path/to/your/image.CR2'  # Change this
    model_path = 'models/trained_model.h5'

    # Initialize pipeline
    pipeline = ImageGradingPipeline(
        model_path=model_path if os.path.exists(model_path) else None,
        use_heuristic=True,
        feedback_path='data/feedback'
    )

    # Process with feedback collection
    print("Processing image with feedback collection...")
    results = pipeline.process_image(
        image_path=image_path,
        output_dir='output',
        apply_edits=True,
        collect_feedback=True  # Enable feedback collection
    )

    print("\n✓ Feedback recorded!")


def check_feedback_status():
    """Check feedback statistics and retraining readiness."""

    model_path = 'models/trained_model.h5'

    # Initialize continuous learner
    learner = ContinuousLearner(
        model_path=model_path,
        feedback_path='data/feedback',
        min_feedback_samples=50
    )

    # Display report
    print(learner.get_improvement_report())

    # Check if ready for retraining
    if learner.should_retrain():
        print("\n" + "=" * 70)
        response = input("Model is ready for retraining. Proceed? (y/n): ")

        if response.lower() == 'y':
            print("\nRetraining model with user feedback...")
            success = learner.retrain_model(epochs=10, backup=True)

            if success:
                print("\n✓ Model successfully retrained!")
                print("The model will now incorporate your feedback.")
            else:
                print("\n⚠ Retraining failed")


def main():
    """Main function."""
    print("=" * 70)
    print("FEEDBACK SYSTEM EXAMPLE")
    print("=" * 70)

    print("\n1. Process image with feedback")
    print("2. Check feedback status and retrain")
    print("3. Both")

    choice = input("\nSelect option (1-3): ").strip()

    if choice == '1':
        process_with_feedback()
    elif choice == '2':
        check_feedback_status()
    elif choice == '3':
        process_with_feedback()
        print("\n\n")
        check_feedback_status()
    else:
        print("Invalid choice")


if __name__ == '__main__':
    main()
