# ML Image Grading System

An intelligent ML-powered system that analyzes CR2 (Canon Raw) images, scores them based on composition, colors, contrast, and other attributes, suggests improvements, and automatically edits photos accordingly.

## Features

- **CR2 Image Support**: Native support for Canon Raw format (CR2) and standard formats (JPEG, PNG)
- **Comprehensive Analysis**: Evaluates composition, color quality, contrast, sharpness, exposure, and noise
- **ML Scoring**: Hybrid neural network combining CNN features and traditional image metrics
- **Smart Suggestions**: Generates actionable improvement recommendations
- **Automatic Editing**: Applies professional-grade edits based on analysis
- **Continuous Learning**: Feedback system for model improvement over time
- **Kaggle Training**: Train on large datasets using Kaggle's GPU resources
- **Mac Compatible**: Optimized for local usage on macOS

## Project Structure

```
ML-Image_grading/
├── src/
│   ├── image_loader.py          # CR2 and image loading
│   ├── feature_extractor.py     # Feature extraction (30+ features)
│   ├── scoring_model.py         # ML scoring model
│   ├── suggestion_engine.py     # Improvement suggestions
│   ├── image_editor.py          # Automatic image editing
│   ├── dataset_loader.py        # Dataset loaders (Adobe FiveK, custom)
│   ├── train_model.py           # Model training script
│   ├── feedback_system.py       # Feedback and continuous learning
│   └── pipeline.py              # Main pipeline
├── models/                       # Trained models
├── output/                       # Output images and reports
├── data/                         # Training data and feedback
├── examples/                     # Example scripts
├── kaggle_training_notebook.ipynb
├── requirements.txt
└── README.md
```

## Installation

### macOS Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/ML-Image_grading.git
cd ML-Image_grading
```

2. **Create a virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download pre-trained model** (optional):
- Train your own on Kaggle (see below), or
- Download a pre-trained model and place it in `models/`

## Usage

### Quick Start

Process a single CR2 image:

```bash
python src/pipeline.py path/to/image.CR2 --output output/
```

### With Trained Model

```bash
python src/pipeline.py path/to/image.CR2 --model models/trained_model.h5 --output output/
```

### Batch Processing

Process multiple images:

```bash
python src/pipeline.py path/to/images/ --batch --pattern "*.CR2" --output output/
```

### With Feedback Collection

Collect feedback for continuous learning:

```bash
python src/pipeline.py image.CR2 --feedback --output output/
```

### Skip Automatic Editing

Only analyze without editing:

```bash
python src/pipeline.py image.CR2 --no-edit --output output/
```

## Training the Model

### Option 1: Train on Kaggle (Recommended for large datasets)

1. **Upload the Kaggle notebook:**
   - Go to [Kaggle](https://www.kaggle.com)
   - Create new notebook
   - Upload `kaggle_training_notebook.ipynb`

2. **Add Adobe FiveK dataset:**
   - Download from [MIT CSAIL](https://data.csail.mit.edu/graphics/fivek/)
   - Or search "Adobe FiveK" on Kaggle Datasets
   - Add to your notebook

3. **Enable GPU:**
   - Settings > Accelerator > GPU

4. **Run the notebook:**
   - Execute all cells
   - Wait for training to complete (~1-2 hours)

5. **Download the model:**
   - Download `models/image_quality_scorer.h5` from output
   - Place in your local `models/` directory

### Option 2: Train Locally

```bash
python src/train_model.py \
  --dataset path/to/adobe_fivek \
  --dataset-type adobe_fivek \
  --expert c \
  --epochs 30 \
  --batch-size 8 \
  --output models/trained_model.h5
```

### Training on Custom Dataset

Create a custom dataset:

```
data/custom_dataset/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── ratings.json
```

`ratings.json` format:
```json
{
  "image1.jpg": {
    "overall_score": 0.85,
    "composition_score": 0.9,
    "color_score": 0.8,
    "technical_score": 0.85
  }
}
```

Train:
```bash
python src/train_model.py \
  --dataset data/custom_dataset \
  --dataset-type custom \
  --epochs 20 \
  --output models/custom_model.h5
```

## Continuous Learning with Feedback

### Collect Feedback

```bash
python src/pipeline.py image.CR2 --feedback
```

The system will ask you to rate its predictions, helping it improve over time.

### View Feedback Statistics

```bash
python src/feedback_system.py --mode stats --feedback-path data/feedback
```

### Retrain with Feedback

After collecting enough feedback (50+ samples):

```bash
python src/feedback_system.py \
  --mode retrain \
  --model-path models/trained_model.h5 \
  --feedback-path data/feedback \
  --epochs 10
```

### Export Feedback Data

```bash
python src/feedback_system.py \
  --mode export \
  --feedback-path data/feedback \
  --export-path data/feedback_export.json
```

## Image Analysis Features

The system analyzes 30+ features across multiple categories:

### Composition Features
- Rule of thirds alignment
- Edge density and distribution
- Symmetry (vertical/horizontal)
- Complexity and center focus

### Color Features
- Color vibrancy and saturation
- Color diversity and harmony
- Dominant colors
- Warm/cool balance

### Technical Features
- **Contrast**: RMS contrast, Michelson contrast, dynamic range
- **Sharpness**: Laplacian variance, Tenengrad, edge sharpness
- **Exposure**: Brightness distribution, over/underexposure
- **Noise**: Noise estimation, signal-to-noise ratio

## Output

For each processed image, the system generates:

1. **Edited Image**: `<filename>_edited.jpg`
2. **Comparison Image**: Side-by-side before/after
3. **Report**: Detailed analysis and scores
4. **Suggestions**: Prioritized improvement recommendations

### Example Report

```
IMAGE QUALITY ASSESSMENT REPORT
======================================================================

Image: DSC_1234.CR2

QUALITY SCORES:
----------------------------------------------------------------------
overall_score            :  78.50/100
composition_score        :  82.00/100
color_score              :  75.00/100
technical_score          :  79.00/100


IMPROVEMENT SUGGESTIONS:
----------------------------------------------------------------------
  🔴 [high] Increase saturation to make colors more vibrant
  🔴 [high] Apply sharpening to enhance image clarity
  🟡 [medium] Consider positioning key elements along rule-of-thirds
  🔵 [low] Image has cool tone. Consider warming up
```

## Example Scripts

### Basic Usage Example

```python
from src.pipeline import ImageGradingPipeline

# Initialize pipeline
pipeline = ImageGradingPipeline(
    model_path='models/trained_model.h5',
    use_heuristic=True
)

# Process image
results = pipeline.process_image(
    image_path='photo.CR2',
    output_dir='output/',
    apply_edits=True
)

# Print scores
for key, value in results['scores'].items():
    print(f"{key}: {value * 100:.1f}/100")
```

### Batch Processing Example

See `examples/batch_process.py` for a complete example.

## Technical Details

### Model Architecture

The scoring model uses a hybrid architecture:

- **CNN Branch**: MobileNetV2 (pre-trained on ImageNet) for visual features
- **Feature Branch**: Dense layers for traditional metrics
- **Multi-Output**: Separate scores for overall, composition, color, and technical quality

### Image Processing Pipeline

1. **Load**: CR2 → RGB conversion with camera white balance
2. **Preprocess**: Resize (512×512), normalize
3. **Extract**: 30+ feature metrics
4. **Score**: ML model + heuristic scoring
5. **Suggest**: Rule-based improvement generation
6. **Edit**: Apply adjustments (brightness, contrast, saturation, etc.)
7. **Save**: Edited image, comparison, report

### Editing Capabilities

- Brightness adjustment
- Contrast enhancement
- Saturation control
- Temperature/white balance
- Sharpening (Laplacian-based)
- Noise reduction (Non-local means)
- Vignette effect
- Auto-enhancement (CLAHE)

## Datasets

### Supported Datasets

1. **Adobe FiveK**
   - 5,000 photos with expert retouching
   - 5 expert photographers (A, B, C, D, E)
   - Expert C most commonly used
   - Download: https://data.csail.mit.edu/graphics/fivek/

2. **Custom Datasets**
   - Your own images with quality ratings
   - JSON-based rating format

3. **Feedback Dataset**
   - User feedback for continuous learning
   - Automatically managed

## Requirements

- Python 3.8+
- TensorFlow 2.10+
- OpenCV
- rawpy (for CR2 support)
- NumPy, SciPy, Pillow
- See `requirements.txt` for complete list

## Performance

### Training
- Kaggle GPU: ~1-2 hours for 1000 images (30 epochs)
- Local (CPU): ~8-12 hours for same dataset
- Recommended: Use Kaggle for initial training

### Inference
- macOS (CPU): ~2-5 seconds per image
- macOS (M1/M2): ~1-2 seconds per image
- Batch processing: Efficient for multiple images

## Troubleshooting

### CR2 Files Not Loading

Install rawpy dependencies:
```bash
brew install libraw  # macOS
pip install rawpy --no-binary rawpy
```

### Out of Memory

Reduce batch size or image size:
```python
image_loader = CR2ImageLoader(target_size=(256, 256))
```

### Model Not Loading

Ensure TensorFlow version compatibility:
```bash
pip install tensorflow==2.10.0
```

## Contributing

Contributions are welcome! Areas for improvement:

- [ ] Support for more RAW formats (NEF, ARW, DNG)
- [ ] Advanced composition analysis
- [ ] Style transfer capabilities
- [ ] Web interface
- [ ] Mobile app

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Adobe FiveK dataset creators
- TensorFlow and Keras teams
- rawpy and OpenCV communities

## Citation

If you use this project in your research, please cite:

```bibtex
@software{ml_image_grading,
  author = {Your Name},
  title = {ML Image Grading System},
  year = {2024},
  url = {https://github.com/yourusername/ML-Image_grading}
}
```

## Contact

- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

## Roadmap

- [x] CR2 image loading
- [x] Feature extraction
- [x] ML scoring model
- [x] Automatic editing
- [x] Feedback system
- [x] Kaggle training support
- [ ] Web interface
- [ ] Real-time preview
- [ ] Batch editing GUI
- [ ] Cloud deployment
- [ ] Mobile app

---

Made with ❤️ for photographers and ML enthusiasts
