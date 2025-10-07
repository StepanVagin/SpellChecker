#!/bin/bash
# scripts/data_preparation/download_unsupervised_data.sh
#
# Download script for unsupervised text corpora used for n-gram language model training.
# Datasets: Wikipedia, CC-News, BookCorpus
#
# Usage:
#   ./download_unsupervised_data.sh           # Download full datasets
#   ./download_unsupervised_data.sh --sample  # Download limited samples (~2GB)

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
echo "Wikipedia Setup..."

# Create Wikipedia directory
WIKIPEDIA_DIR="$PROJECT_ROOT/data/raw/unsupervised/wikipedia"
mkdir -p "$WIKIPEDIA_DIR"

if [ "$SAMPLE_MODE" = true ]; then
    # Sample mode: Download sample articles via Wikipedia API
    echo "Sample mode: Downloading sample Wikipedia articles via API..."
    mkdir -p "$WIKIPEDIA_DIR/extracted/AA"
    
    cd "$WIKIPEDIA_DIR"
    python "$SCRIPT_DIR/data/raw/unsupervised/download_wiki_sample.py" || {
        echo "Wikipedia sample download failed"
    }
else
    # Full mode: Download complete Wikipedia dump
    echo "Downloading English Wikipedia articles dump..."
    echo "Note: This is a large download (~20GB compressed)"
    
    if [ ! -f "$WIKIPEDIA_DIR/enwiki-latest-pages-articles.xml.bz2" ]; then
        cd "$WIKIPEDIA_DIR"
        wget -c "https://dumps.wikimedia.org/${WIKI_LANG}wiki/latest/${WIKI_LANG}wiki-latest-pages-articles.xml.bz2" \
            -O enwiki-latest-pages-articles.xml.bz2
        
        echo "Wikipedia dump downloaded"
        echo "Note: You need to extract text using WikiExtractor:"
        echo "  pip install wikiextractor"
        echo "  wikiextractor enwiki-latest-pages-articles.xml.bz2 -o extracted"
    else
        echo "Wikipedia dump already exists, skipping download..."
    fi
fi

# ============================================================================
# 2. CC-News (Common Crawl News)
# ============================================================================
echo ""
echo "CC-News Setup..."

CCNEWS_DIR="$PROJECT_ROOT/data/raw/unsupervised/ccnews"
mkdir -p "$CCNEWS_DIR"

echo "Downloading and processing CC-News (compressed pickle)..."
cd "$CCNEWS_DIR"
# Run with appropriate sample size, write directly to cleaned output
if [ "$SAMPLE_MODE" = true ]; then
    python "$SCRIPT_DIR/data/raw/unsupervised/download_ccnews.py" 50000 ccnews.pkl.gz || {
        echo "CC-News download failed, continuing..."
    }
else
    python "$SCRIPT_DIR/data/raw/unsupervised/download_ccnews.py" "" ccnews.pkl.gz || {
        echo "CC-News download failed, continuing..."
    }
fi

# =============================================================================
# 3. BookCorpus (disabled for debugging)
# =============================================================================
# echo ""
# echo "BookCorpus Setup..."
#
# BOOKCORPUS_DIR="$PROJECT_ROOT/data/raw/unsupervised/bookcorpus"
# mkdir -p "$BOOKCORPUS_DIR/books"
#
# echo "Downloading and processing BookCorpus (compressed pickle)..."
# cd "$BOOKCORPUS_DIR"
# # Run with appropriate sample size, write directly to cleaned output in books/ for compatibility
# if [ "$SAMPLE_MODE" = true ]; then
#     python "$SCRIPT_DIR/data/raw/unsupervised/download_bookcorpus.py" 50000 books/bookcorpus.pkl.gz --stream10k || {
#         echo "BookCorpus download failed, continuing..."
#     }
# else
#     python "$SCRIPT_DIR/data/raw/unsupervised/download_bookcorpus.py" "" books/bookcorpus.pkl.gz || {
#         echo "BookCorpus download failed, continuing..."
#     }
# fi

# ============================================================================
# Processing downloaded corpora
# ============================================================================
echo ""
echo "Processing downloaded corpora..."

# Ensure processed directory exists and return to project root before Python
mkdir -p "$PROJECT_ROOT/data/processed/unsupervised"
cd "$PROJECT_ROOT"

PYTHONPATH="$PROJECT_ROOT/src" python - <<'PY'
from pathlib import Path
from spellchecker.data.parsers.unsupervised_parser import (
    UniversalTextCleaner,
    process_unsupervised_corpus,
)

base_raw = Path("data/raw/unsupervised")
base_out = Path("data/processed/unsupervised")
base_out.mkdir(parents=True, exist_ok=True)

cleaner = UniversalTextCleaner()

tasks = [
    ("wikipedia", base_raw / "wikipedia" / "extracted", base_out / "wikipedia.txt"),
    # For ccnews and bookcorpus, the downloaders already wrote cleaned text (compressed)
    ("ccnews", base_raw / "ccnews" / "ccnews.pkl.gz", base_out / "ccnews.txt"),
    # ("bookcorpus", base_raw / "bookcorpus" / "books" / "bookcorpus.pkl.gz", base_out / "bookcorpus.txt"),
]

for corpus, input_path, output_path in tasks:
    try:
        print(f"Processing {corpus}...")
        if corpus == "wikipedia":
            process_unsupervised_corpus(corpus, str(input_path), str(output_path), cleaner)
        else:
            # For ccnews and bookcorpus, decompress pickle and write to processed .txt
            import gzip, pickle
            src = Path(input_path)
            dst = Path(output_path)
            if src.exists() and src.stat().st_size > 0:
                try:
                    with gzip.open(src, "rb") as f:
                        texts = pickle.load(f)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    with open(dst, "w", encoding="utf-8") as out_f:
                        for t in texts:
                            out_f.write(t + "\n")
                except Exception as e:
                    print(f"Failed to decompress {src}: {e}")
            else:
                print(f"Source not found, skipping copy: {src}")
    except Exception as e:
        print(f"Processing failed for {corpus}: {e}")
PY

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "================================================"
echo "Data Download and Processing Complete"
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
    # echo "  - BookCorpus: Full corpus"
fi
echo ""


