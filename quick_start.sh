#!/bin/bash
# Quick start script for N-gram spelling checker
# This script demonstrates the complete workflow from data download to web interface
# Downloads sample data (~2GB) for quick testing
#
# For full datasets, run: bash scripts/data_preparation/download_unsupervised_data.sh

set -e

# Save the initial directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "========================================"
echo "N-gram Spelling Checker - Quick Start"
echo "========================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not found"
    exit 1
fi

# Install dependencies
echo "Step 1: Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "Step 2: Creating directories..."
mkdir -p data/raw/unsupervised
mkdir -p data/processed/unsupervised
mkdir -p models/ngram

echo ""
echo "Step 3: Downloading unsupervised data (sample mode ~2GB)..."
echo ""

# Use the standard download script with --sample flag for limited data
bash scripts/data_preparation/download_unsupervised_data.sh --sample

echo ""
echo "Step 4: Processing unsupervised data..."
python scripts/data_preparation/process_unsupervised_data.py --all || echo "Some processing failed, continuing..."

echo ""
echo "Step 5: Training N-gram models..."
python scripts/train_ngram_model.py \
    --data "data/processed/unsupervised/*.txt" \
    --output models/ngram

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Data sources downloaded (sample mode):"
echo "  - Wikipedia: 5 articles (via API)"
echo "  - CC-News: 50K articles (~500MB)"
echo "  - BookCorpus: 50K passages (~500MB)"
echo ""
echo "Total: ~1GB of training data"
echo ""
echo "To start the web interface, run:"
echo "  python app.py --models models/ngram --port 5001"
echo ""
echo "Then open your browser to: http://localhost:5001"
echo ""
echo "Note: For full datasets, run:"
echo "  bash scripts/data_preparation/download_unsupervised_data.sh"
echo ""
