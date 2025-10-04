#!/bin/bash
# scripts/data_preparation/download_unsupervised_data.sh
#
# Download script for unsupervised text corpora used for n-gram language model training.
# Datasets: Wikipedia, CC-News, BookCorpus

set -e

# Create data/raw/unsupervised directory
mkdir -p data/raw/unsupervised
cd data/raw/unsupervised

echo "================================================"
echo "Downloading Unsupervised Text Corpora"
echo "================================================"

# ============================================================================
# 1. Wikipedia Dumps
# ============================================================================
echo ""
echo "📚 Downloading Wikipedia Dump (English)..."
echo "Note: This downloads a recent Wikipedia dump. You can modify the date below."

# You can change this date to get different Wikipedia versions
WIKI_DATE="20231001"
WIKI_LANG="en"

# Create Wikipedia directory
mkdir -p wikipedia
cd wikipedia

# Download the latest Wikipedia dump (articles only, no full history)
# Using the 'latest' symlink for convenience
echo "Downloading English Wikipedia articles dump..."
wget -c "https://dumps.wikimedia.org/${WIKI_LANG}wiki/latest/${WIKI_LANG}wiki-latest-pages-articles.xml.bz2" \
    -O enwiki-latest-pages-articles.xml.bz2

echo "✅ Wikipedia dump downloaded"
echo "Note: You need to extract text using WikiExtractor:"
echo "  pip install wikiextractor"
echo "  wikiextractor enwiki-latest-pages-articles.xml.bz2 -o extracted --json"

cd ..

# ============================================================================
# 2. CC-News (Common Crawl News)
# ============================================================================
echo ""
echo "📰 Downloading CC-News (Common Crawl News)..."

mkdir -p ccnews
cd ccnews

# CC-News is available via Hugging Face datasets or direct download
# For simplicity, we'll provide instructions to use the datasets library
# Alternatively, download a subset from Common Crawl directly

echo "CC-News is best accessed via the Hugging Face datasets library."
echo "To download programmatically, use the Python script provided."
echo ""
echo "Creating a placeholder download script..."

cat > download_ccnews.py << 'EOF'
"""
Download CC-News dataset using Hugging Face datasets library.

Install: pip install datasets
"""

from datasets import load_dataset
import json

def download_ccnews(output_file="ccnews.jsonl", num_samples=100000):
    """
    Download CC-News dataset and save as JSONL.
    
    Args:
        output_file: Output file path
        num_samples: Number of samples to download (None for all)
    """
    print("Loading CC-News dataset from Hugging Face...")
    
    # Load dataset
    # Note: Full dataset is very large (~70GB), so we limit samples
    dataset = load_dataset("cc_news", split="train", streaming=True)
    
    count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for example in dataset:
            # Write as JSONL
            json.dump({
                "title": example.get("title", ""),
                "text": example.get("text", ""),
                "domain": example.get("domain", ""),
                "date": example.get("date", ""),
            }, f)
            f.write("\n")
            
            count += 1
            if num_samples and count >= num_samples:
                break
            
            if count % 10000 == 0:
                print(f"Downloaded {count} articles...")
    
    print(f"✅ Downloaded {count} articles to {output_file}")

if __name__ == "__main__":
    # Download 100k articles (adjust as needed)
    download_ccnews("ccnews.jsonl", num_samples=100000)
EOF

echo "✅ CC-News download script created: download_ccnews.py"
echo "Run: python download_ccnews.py"

cd ..

# ============================================================================
# 3. BookCorpus
# ============================================================================
echo ""
echo "📖 Downloading BookCorpus..."

mkdir -p bookcorpus
cd bookcorpus

# BookCorpus is no longer officially available due to copyright issues
# However, it's available through Hugging Face datasets
# We'll create a download script

cat > download_bookcorpus.py << 'EOF'
"""
Download BookCorpus dataset using Hugging Face datasets library.

Install: pip install datasets
"""

from datasets import load_dataset
from pathlib import Path

def download_bookcorpus(output_dir="books"):
    """
    Download BookCorpus dataset and save as text files.
    
    Args:
        output_dir: Output directory for book files
    """
    print("Loading BookCorpus dataset from Hugging Face...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load dataset
    try:
        dataset = load_dataset("bookcorpus", split="train")
    except Exception as e:
        print(f"Error loading bookcorpus: {e}")
        print("Trying alternative: bookcorpusopen")
        dataset = load_dataset("bookcorpusopen", split="train")
    
    # Save as single file (one sentence per line)
    output_file = output_path / "bookcorpus.txt"
    
    print(f"Saving to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        for i, example in enumerate(dataset):
            text = example["text"].strip()
            if text:
                f.write(text + "\n")
            
            if (i + 1) % 100000 == 0:
                print(f"Processed {i + 1} passages...")
    
    print(f"✅ BookCorpus saved to {output_file}")

if __name__ == "__main__":
    download_bookcorpus("books")
EOF

echo "✅ BookCorpus download script created: download_bookcorpus.py"
echo "Run: python download_bookcorpus.py"

cd ..

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "================================================"
echo "✅ Download scripts prepared"
echo "================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Wikipedia:"
echo "   - Install wikiextractor: pip install wikiextractor"
echo "   - Extract text: wikiextractor wikipedia/enwiki-latest-pages-articles.xml.bz2 -o wikipedia/extracted"
echo ""
echo "2. CC-News:"
echo "   - Install dependencies: pip install datasets"
echo "   - Run: cd ccnews && python download_ccnews.py"
echo ""
echo "3. BookCorpus:"
echo "   - Install dependencies: pip install datasets"
echo "   - Run: cd bookcorpus && python download_bookcorpus.py"
echo ""
echo "After downloading, use the parsers in src/spellchecker/data/parsers/unsupervised_parser.py"
echo "to process the data for n-gram training."
echo ""


