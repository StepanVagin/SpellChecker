#!/bin/bash
# scripts/data_preparation/download_unsupervised_data.sh
#
# Download script for unsupervised text corpora used for n-gram language model training.
# Datasets: Wikipedia, CC-News, BookCorpus
#
# Usage:
#   ./download_unsupervised_data.sh           # Download full datasets
#   ./download_unsupervised_data.sh --sample  # Download limited samples (~2GB)

set -e

# Parse arguments
SAMPLE_MODE=false
if [ "$1" == "--sample" ]; then
    SAMPLE_MODE=true
fi

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Create data/raw/unsupervised directory relative to project root
mkdir -p "$PROJECT_ROOT/data/raw/unsupervised"
cd "$PROJECT_ROOT/data/raw/unsupervised"

echo "================================================"
echo "Downloading Unsupervised Text Corpora"
if [ "$SAMPLE_MODE" = true ]; then
    echo "MODE: Sample (Limited to ~2GB)"
else
    echo "MODE: Full datasets"
fi
echo "================================================"

# ============================================================================
# 1. Wikipedia Dumps
# ============================================================================
echo ""
echo "📚 Wikipedia Setup..."

# Create Wikipedia directory
mkdir -p wikipedia
cd wikipedia

if [ "$SAMPLE_MODE" = true ]; then
    # Sample mode: Download sample articles via Wikipedia API
    echo "Sample mode: Downloading sample Wikipedia articles via API..."
    mkdir -p extracted/AA
    
    cat > download_wiki_sample.py << 'EOFPYTHON'
import urllib.request
import json

articles = [
    "Python_(programming_language)",
    "Machine_learning",
    "Natural_language_processing",
    "Artificial_intelligence",
    "Data_science"
]

wiki_texts = []

for article_title in articles:
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{article_title}"
        req = urllib.request.Request(
            url,
            headers={
                'User-Agent': 'SpellCheckerBot/1.0 (Educational Project)',
                'Accept': 'application/json'
            }
        )
        
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read())
            title = data.get("title", article_title.replace("_", " "))
            extract = data.get("extract", "")
            
            wiki_text = f'<doc id="{len(wiki_texts)+1}" title="{title}">\n{extract}\n</doc>'
            wiki_texts.append(wiki_text)
            print(f"  ✓ Downloaded: {title}")
    
    except Exception as e:
        print(f"  ✗ Failed: {article_title}: {e}")

if wiki_texts:
    with open("extracted/AA/wiki_00", "w", encoding="utf-8") as f:
        f.write("\n".join(wiki_texts))
    print(f"\n✅ Downloaded {len(wiki_texts)} Wikipedia articles")
else:
    print("\n❌ Failed to download any Wikipedia articles")
    exit(1)
EOFPYTHON

    python download_wiki_sample.py || echo "⚠️  Wikipedia sample download failed"
else
    # Full mode: Download complete Wikipedia dump
    echo "Downloading English Wikipedia articles dump..."
    echo "Note: This is a large download (~20GB compressed)"
    
    if [ ! -f "enwiki-latest-pages-articles.xml.bz2" ]; then
        wget -c "https://dumps.wikimedia.org/${WIKI_LANG}wiki/latest/${WIKI_LANG}wiki-latest-pages-articles.xml.bz2" \
            -O enwiki-latest-pages-articles.xml.bz2
        
        echo "✅ Wikipedia dump downloaded"
        echo "Note: You need to extract text using WikiExtractor:"
        echo "  pip install wikiextractor"
        echo "  wikiextractor enwiki-latest-pages-articles.xml.bz2 -o extracted"
    else
        echo "Wikipedia dump already exists, skipping download..."
    fi
fi

cd ..

# ============================================================================
# 2. CC-News (Common Crawl News)
# ============================================================================
echo ""
echo "📰 CC-News Setup..."

mkdir -p ccnews
cd ccnews

if [ ! -f "ccnews.jsonl" ]; then
    echo "Downloading CC-News articles..."
    
    cat > download_ccnews.py << EOF
"""
Download CC-News dataset using Hugging Face datasets library.

Install: pip install datasets
"""

from datasets import load_dataset
import json
import sys

def download_ccnews(output_file="ccnews.jsonl", num_samples=None):
    """
    Download CC-News dataset and save as JSONL.
    
    Args:
        output_file: Output file path
        num_samples: Number of samples to download (None for all)
    """
    print(f"Loading CC-News dataset from Hugging Face...")
    if num_samples:
        print(f"Limited to {num_samples} articles")
    
    try:
        # Load dataset
        # Note: Full dataset is very large (~70GB), so we can limit samples
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
        return True
    except Exception as e:
        print(f"Error downloading CC-News: {e}")
        return False

if __name__ == "__main__":
    # Get num_samples from command line or use default
    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else None
    success = download_ccnews("ccnews.jsonl", num_samples=num_samples)
    sys.exit(0 if success else 1)
EOF

    # Run with appropriate sample size
    if [ "$SAMPLE_MODE" = true ]; then
        python download_ccnews.py 50000 || echo "CC-News download failed, continuing..."
    else
        python download_ccnews.py 100000 || echo "CC-News download failed, continuing..."
    fi
else
    echo "CC-News already downloaded, skipping..."
fi

cd ..

# ============================================================================
# 3. BookCorpus
# ============================================================================
echo ""
echo "📖 BookCorpus Setup..."

mkdir -p bookcorpus/books
cd bookcorpus

if [ ! -f "books/bookcorpus.txt" ]; then
    echo "Downloading BookCorpus passages..."
    
    cat > download_bookcorpus.py << EOF
"""
Download BookCorpus dataset using Hugging Face datasets library.

Install: pip install datasets
"""

from datasets import load_dataset
from pathlib import Path
import sys

def download_bookcorpus(output_dir="books", max_passages=None):
    """
    Download BookCorpus dataset and save as text files.
    
    Args:
        output_dir: Output directory for book files
        max_passages: Maximum number of passages to download (None for all)
    """
    print("Loading BookCorpus dataset from Hugging Face...")
    if max_passages:
        print(f"Limited to {max_passages} passages")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load dataset using the working bookcorpusopen dataset
    try:
        dataset = load_dataset("lucadiliello/bookcorpusopen", split="train", streaming=True)
    except Exception as e:
        print(f"Error loading BookCorpus: {e}")
        return False
    
    # Save as single file (one sentence per line)
    output_file = output_path / "bookcorpus.txt"
    
    print(f"Saving to {output_file}...")
    count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for example in dataset:
            text = example["text"].strip()
            if text:
                f.write(text + "\n")
                count += 1
                
                if max_passages and count >= max_passages:
                    break
                
                if count % 10000 == 0:
                    print(f"Processed {count} passages...")
    
    print(f"✅ BookCorpus saved to {output_file} ({count} passages)")
    return True

if __name__ == "__main__":
    # Get max_passages from command line or use default
    max_passages = int(sys.argv[1]) if len(sys.argv) > 1 else None
    success = download_bookcorpus("books", max_passages=max_passages)
    sys.exit(0 if success else 1)
EOF

    # Run with appropriate sample size
    if [ "$SAMPLE_MODE" = true ]; then
        python download_bookcorpus.py 50000 || echo "BookCorpus download failed, continuing..."
    else
        python download_bookcorpus.py || echo "BookCorpus download failed, continuing..."
    fi
else
    echo "BookCorpus already downloaded, skipping..."
fi

cd ..

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "================================================"
echo "✅ Data Download Complete"
echo "================================================"
echo ""
if [ "$SAMPLE_MODE" = true ]; then
    echo "Sample data downloaded (~1GB):"
    echo "  - Wikipedia: 5 articles (via API)"
    echo "  - CC-News: 50K articles (~500MB)"
    echo "  - BookCorpus: 50K passages (~500MB)"
else
    echo "Full datasets downloaded:"
    echo "  - Wikipedia: Full dump (requires extraction)"
    echo "  - CC-News: 100K articles"
    echo "  - BookCorpus: Full corpus"
    echo ""
    echo "Note: If Wikipedia dump was downloaded, extract it with:"
    echo "  pip install wikiextractor"
    echo "  cd data/raw/unsupervised/wikipedia"
    echo "  wikiextractor enwiki-latest-pages-articles.xml.bz2 -o extracted"
fi
echo ""
echo "Next: Process the data with:"
echo "  python scripts/data_preparation/process_unsupervised_data.py --all"
echo ""


