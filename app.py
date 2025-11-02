"""
ML Image Grading Web Application
Flask web server with REST API for image grading and editing
"""

import os
import sys
import json
import uuid
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from flask import Flask, request, jsonify, send_file, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import io
import base64

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pipeline import ImageGradingPipeline
from image_loader import CR2ImageLoader
from feedback_system import FeedbackSystem

# Initialize Flask app
app = Flask(__name__,
            static_folder='web/static',
            template_folder='web/templates')
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['MODEL_PATH'] = 'models/image_quality_scorer.h5'
app.config['ALLOWED_EXTENSIONS'] = {'cr2', 'jpg', 'jpeg', 'png', 'tiff', 'bmp'}

# Ensure directories exist
for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

# Initialize pipeline
pipeline = None

def get_pipeline():
    """Lazy load pipeline to avoid startup delays"""
    global pipeline
    if pipeline is None:
        model_path = app.config['MODEL_PATH'] if os.path.exists(app.config['MODEL_PATH']) else None
        pipeline = ImageGradingPipeline(model_path=model_path, use_heuristic=True)
    return pipeline


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def encode_image_to_base64(image_path: str) -> str:
    """Encode image to base64 for web display"""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


@app.route('/')
def index():
    """Serve main web interface"""
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': pipeline is not None,
        'version': '1.0.0'
    })


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """
    Upload and process a single image
    Returns: Analysis results with scores and suggestions
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400

    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        upload_id = str(uuid.uuid4())
        upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], upload_id)
        os.makedirs(upload_dir, exist_ok=True)

        filepath = os.path.join(upload_dir, filename)
        file.save(filepath)

        # Get processing options
        apply_edits = request.form.get('apply_edits', 'true').lower() == 'true'

        # Process image
        output_dir = os.path.join(app.config['OUTPUT_FOLDER'], upload_id)
        os.makedirs(output_dir, exist_ok=True)

        p = get_pipeline()
        results = p.process_image(
            image_path=filepath,
            output_dir=output_dir,
            apply_edits=apply_edits
        )

        # Prepare response
        response = {
            'upload_id': upload_id,
            'filename': filename,
            'scores': {k: float(v) * 100 for k, v in results['scores'].items()},
            'suggestions': results['suggestions'],
            'original_image': f'/api/image/{upload_id}/original',
        }

        if apply_edits and 'edited_path' in results:
            response['edited_image'] = f'/api/image/{upload_id}/edited'
            response['comparison_image'] = f'/api/image/{upload_id}/comparison'

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/batch-upload', methods=['POST'])
def batch_upload():
    """
    Upload and process multiple images
    Returns: Batch processing job ID
    """
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400

    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'Empty file list'}), 400

    try:
        batch_id = str(uuid.uuid4())
        batch_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'batch', batch_id)
        os.makedirs(batch_dir, exist_ok=True)

        uploaded_files = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(batch_dir, filename)
                file.save(filepath)
                uploaded_files.append({
                    'filename': filename,
                    'path': filepath
                })

        return jsonify({
            'batch_id': batch_id,
            'files_uploaded': len(uploaded_files),
            'files': [f['filename'] for f in uploaded_files]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/batch-process/<batch_id>', methods=['POST'])
def batch_process(batch_id: str):
    """
    Process all images in a batch
    Returns: Results for all images
    """
    try:
        batch_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'batch', batch_id)
        if not os.path.exists(batch_dir):
            return jsonify({'error': 'Batch not found'}), 404

        apply_edits = request.json.get('apply_edits', True) if request.json else True

        # Process all images
        results = []
        p = get_pipeline()

        for filename in os.listdir(batch_dir):
            if allowed_file(filename):
                filepath = os.path.join(batch_dir, filename)
                output_dir = os.path.join(app.config['OUTPUT_FOLDER'], 'batch', batch_id,
                                         os.path.splitext(filename)[0])
                os.makedirs(output_dir, exist_ok=True)

                try:
                    result = p.process_image(
                        image_path=filepath,
                        output_dir=output_dir,
                        apply_edits=apply_edits
                    )

                    results.append({
                        'filename': filename,
                        'status': 'success',
                        'scores': {k: float(v) * 100 for k, v in result['scores'].items()},
                        'suggestions': result['suggestions'],
                        'image_id': os.path.splitext(filename)[0]
                    })
                except Exception as e:
                    results.append({
                        'filename': filename,
                        'status': 'error',
                        'error': str(e)
                    })

        return jsonify({
            'batch_id': batch_id,
            'total_images': len(results),
            'results': results
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/image/<upload_id>/<image_type>', methods=['GET'])
def get_image(upload_id: str, image_type: str):
    """
    Retrieve processed images
    image_type: original, edited, comparison
    """
    try:
        if image_type == 'original':
            upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], upload_id)
            if os.path.exists(upload_dir):
                files = os.listdir(upload_dir)
                if files:
                    return send_from_directory(upload_dir, files[0])

        elif image_type in ['edited', 'comparison']:
            output_dir = os.path.join(app.config['OUTPUT_FOLDER'], upload_id)
            if os.path.exists(output_dir):
                for filename in os.listdir(output_dir):
                    if image_type == 'edited' and '_edited' in filename:
                        return send_from_directory(output_dir, filename)
                    elif image_type == 'comparison' and '_comparison' in filename:
                        return send_from_directory(output_dir, filename)

        return jsonify({'error': 'Image not found'}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/batch-image/<batch_id>/<image_id>/<image_type>', methods=['GET'])
def get_batch_image(batch_id: str, image_id: str, image_type: str):
    """
    Retrieve batch processed images
    """
    try:
        if image_type == 'original':
            batch_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'batch', batch_id)
            for filename in os.listdir(batch_dir):
                if os.path.splitext(filename)[0] == image_id:
                    return send_from_directory(batch_dir, filename)

        elif image_type in ['edited', 'comparison']:
            output_dir = os.path.join(app.config['OUTPUT_FOLDER'], 'batch', batch_id, image_id)
            if os.path.exists(output_dir):
                for filename in os.listdir(output_dir):
                    if image_type == 'edited' and '_edited' in filename:
                        return send_from_directory(output_dir, filename)
                    elif image_type == 'comparison' and '_comparison' in filename:
                        return send_from_directory(output_dir, filename)

        return jsonify({'error': 'Image not found'}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """
    Submit feedback for continuous learning
    """
    try:
        data = request.json
        upload_id = data.get('upload_id')
        scores = data.get('scores')
        comments = data.get('comments', '')

        if not upload_id or not scores:
            return jsonify({'error': 'Missing required fields'}), 400

        # Get original image path
        upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], upload_id)
        if not os.path.exists(upload_dir):
            return jsonify({'error': 'Upload not found'}), 404

        files = os.listdir(upload_dir)
        if not files:
            return jsonify({'error': 'Image not found'}), 404

        image_path = os.path.join(upload_dir, files[0])

        # Save feedback
        feedback_system = FeedbackSystem(feedback_dir='data/feedback')
        feedback_system.collect_feedback(
            image_path=image_path,
            predicted_scores=scores,
            user_scores=scores,
            comments=comments
        )

        return jsonify({'status': 'success', 'message': 'Feedback submitted'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """
    Get processing statistics and model performance
    """
    try:
        feedback_system = FeedbackSystem(feedback_dir='data/feedback')
        stats = feedback_system.get_statistics()

        return jsonify({
            'total_feedback': stats.get('total_samples', 0),
            'average_scores': stats.get('average_scores', {}),
            'model_accuracy': stats.get('model_accuracy', {})
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/preview', methods=['POST'])
def preview_edits():
    """
    Generate real-time preview of edits without saving
    """
    try:
        data = request.json
        upload_id = data.get('upload_id')
        adjustments = data.get('adjustments', {})

        if not upload_id:
            return jsonify({'error': 'Missing upload_id'}), 400

        # Get original image
        upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], upload_id)
        if not os.path.exists(upload_dir):
            return jsonify({'error': 'Upload not found'}), 404

        files = os.listdir(upload_dir)
        if not files:
            return jsonify({'error': 'Image not found'}), 404

        image_path = os.path.join(upload_dir, files[0])

        # Load and apply edits
        from image_editor import ImageEditor

        # Load image
        if image_path.lower().endswith('.cr2'):
            loader = CR2ImageLoader()
            img_array = loader.load_image(image_path)
            img = Image.fromarray((img_array * 255).astype(np.uint8))
        else:
            img = Image.open(image_path)
            img_array = np.array(img).astype(np.float32) / 255.0

        editor = ImageEditor()

        # Apply custom adjustments
        edited_array = img_array.copy()

        if 'brightness' in adjustments:
            edited_array = editor.adjust_brightness(edited_array, adjustments['brightness'])
        if 'contrast' in adjustments:
            edited_array = editor.adjust_contrast(edited_array, adjustments['contrast'])
        if 'saturation' in adjustments:
            edited_array = editor.adjust_saturation(edited_array, adjustments['saturation'])
        if 'temperature' in adjustments:
            edited_array = editor.adjust_temperature(edited_array, adjustments['temperature'])
        if 'sharpness' in adjustments and adjustments['sharpness'] > 0:
            edited_array = editor.apply_sharpening(edited_array, adjustments['sharpness'])

        # Convert to base64
        edited_img = Image.fromarray((edited_array * 255).astype(np.uint8))
        buffer = io.BytesIO()
        edited_img.save(buffer, format='JPEG', quality=90)
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return jsonify({
            'preview_image': f'data:image/jpeg;base64,{img_base64}'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/download/<upload_id>', methods=['GET'])
def download_edited(upload_id: str):
    """
    Download edited image
    """
    try:
        output_dir = os.path.join(app.config['OUTPUT_FOLDER'], upload_id)
        if not os.path.exists(output_dir):
            return jsonify({'error': 'Output not found'}), 404

        for filename in os.listdir(output_dir):
            if '_edited' in filename and not '_comparison' in filename:
                return send_from_directory(
                    output_dir,
                    filename,
                    as_attachment=True,
                    download_name=filename
                )

        return jsonify({'error': 'Edited image not found'}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Development server
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'True').lower() == 'true'

    print(f"Starting ML Image Grading Web Server on port {port}")
    print(f"Debug mode: {debug}")
    print(f"Model path: {app.config['MODEL_PATH']}")

    app.run(host='0.0.0.0', port=port, debug=debug)
