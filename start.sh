#!/bin/bash

# ML Image Grading - Startup Script
# This script helps you get started quickly

echo "==================================="
echo "ML Image Grading System"
echo "==================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p uploads output models data/feedback web/static web/templates

# Check if model exists
if [ ! -f "models/image_quality_scorer.h5" ]; then
    echo ""
    echo "⚠️  Warning: No trained model found"
    echo "The system will use heuristic scoring."
    echo "To train a model, see README.md"
    echo ""
fi

# Start the application
echo ""
echo "Starting ML Image Grading web server..."
echo "Access the application at: http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

export FLASK_APP=app.py
export FLASK_ENV=development
export DEBUG=True

python app.py
