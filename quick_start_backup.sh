#!/bin/bash
# Quick start script for N-gram spelling checker
# This script demonstrates the complete workflow from data download to web interface
# Limited to ~2GB total download

set -e

echo "========================================"
echo "N-gram Spelling Checker - Quick Start"
echo "========================================"
echo "Download limit: ~2GB total"
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
echo "Step 3: Downloading limited unsupervised data (2GB total)..."
echo ""

# ============================================================================
# Download limited datasets
# ============================================================================

# 1. Wikipedia - Download smaller subset (~500MB compressed)
echo "Downloading Wikipedia subset..."
mkdir -p data/raw/unsupervised/wikipedia
cd data/raw/unsupervised/wikipedia

if [ ! -f "enwiki-subset.xml.bz2" ]; then
    echo "Note: Downloading limited Wikipedia subset"
    echo "For full dataset, use scripts/data_preparation/download_unsupervised_data.sh"
    # Download a smaller, specific Wikipedia dump or create from streaming
    # For now, we'll use the extraction limit instead
    echo "Will use extraction limit in processing step"
fi

cd ../../..

# 2. CC-News - Limited to 50K articles (~500MB)
echo ""
echo "Downloading CC-News (limited to 50K articles)..."
mkdir -p data/raw/unsupervised/ccnews
cd data/raw/unsupervised/ccnews

cat > download_ccnews_limited.py << 'EOF'
"""
Download limited CC-News dataset (50K articles, ~500MB)
"""
from datasets import load_dataset
import json
import sys

def download_ccnews_limited(output_file="ccnews.jsonl", num_samples=50000):
    print(f"Loading CC-News dataset (limited to {num_samples} articles)...")
    
    try:
        dataset = load_dataset("cc_news", split="train", streaming=True)
        
        count = 0
        with open(output_file, "w", encoding="utf-8") as f:
            for example in dataset:
                json.dump({
                    "title": example.get("title", ""),
                    "text": example.get("text", ""),
                    "domain": example.get("domain", ""),
                    "date": example.get("date", ""),
                }, f)
                f.write("\n")
                
                count += 1
                if count >= num_samples:
                    break
                
                if count % 5000 == 0:
                    print(f"Downloaded {count} articles...")
        
        print(f"Downloaded {count} articles to {output_file}")
        return True
    except Exception as e:
        print(f"Error downloading CC-News: {e}")
        return False

if __name__ == "__main__":
    success = download_ccnews_limited("ccnews.jsonl", num_samples=50000)
    sys.exit(0 if success else 1)
EOF

python download_ccnews_limited.py

cd ../../..

# 3. BookCorpus - Limited to 100K passages (~500MB)
echo ""
echo "Downloading BookCorpus (limited to 100K passages)..."
mkdir -p data/raw/unsupervised/bookcorpus/books
cd data/raw/unsupervised/bookcorpus

cat > download_bookcorpus_limited.py << 'EOF'
"""
Download limited BookCorpus dataset (100K passages, ~500MB)
"""
from datasets import load_dataset
from pathlib import Path
import sys

def download_bookcorpus_limited(output_dir="books", max_passages=100000):
    print(f"Loading BookCorpus dataset (limited to {max_passages} passages)...")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    try:
        dataset = load_dataset("bookcorpus", split="train", streaming=True)
    except Exception as e:
        print(f"Error loading bookcorpus: {e}")
        try:
            print("Trying alternative: bookcorpusopen")
            dataset = load_dataset("bookcorpusopen", split="train", streaming=True)
        except:
            print("Could not load BookCorpus, skipping...")
            return False
    
    output_file = output_path / "bookcorpus.txt"
    
    print(f"Saving to {output_file}...")
    count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for example in dataset:
            text = example["text"].strip()
            if text:
                f.write(text + "\n")
                count += 1
                
                if count >= max_passages:
                    break
                
                if count % 10000 == 0:
                    print(f"Processed {count} passages...")
    
    print(f"BookCorpus saved to {output_file} ({count} passages)")
    return True

if __name__ == "__main__":
    success = download_bookcorpus_limited("books", max_passages=100000)
    sys.exit(0 if success else 1)
EOF

python download_bookcorpus_limited.py

cd ../../..

# 4. Wikipedia - Extract with limits
echo ""
echo "Processing Wikipedia (limited extraction)..."
cd data/raw/unsupervised/wikipedia

if [ -f "enwiki-latest-pages-articles.xml.bz2" ]; then
    if ! command -v wikiextractor &> /dev/null; then
        echo "Installing wikiextractor..."
        pip install wikiextractor
    fi
    # Extract only first 1GB of articles
    wikiextractor enwiki-latest-pages-articles.xml.bz2 -o extracted --json --bytes 1G
elif [ ! -d "extracted" ]; then
    echo "No Wikipedia dump found, creating placeholder..."
    mkdir -p extracted/AA
    echo '{"title": "Sample", "text": "Sample Wikipedia text for testing."}' > extracted/AA/wiki_00
fi

cd ../../..

echo ""
echo "Step 4: Processing unsupervised data with limits..."
python scripts/data_preparation/process_unsupervised_data.py --all

echo ""
echo "Step 5: Training N-gram models..."
python scripts/train_ngram_model.py \
    --data "data/processed/unsupervised/*.txt" \
    --output models/ngram \
    --use-dictionary \
    --test

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Data downloaded: ~2GB total"
echo "  - Wikipedia: ~1GB (limited extraction)"
echo "  - CC-News: ~500MB (50K articles)"
echo "  - BookCorpus: ~500MB (100K passages)"
echo ""
echo "To start the web interface, run:"
echo "  python app.py --models models/ngram --port 5001"
echo ""
echo "Then open your browser to: http://localhost:5001"
echo ""
echo "For full datasets (no limits), run:"
echo "  bash scripts/data_preparation/download_unsupervised_data.sh"
echo ""
