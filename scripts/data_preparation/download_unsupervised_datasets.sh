#!/bin/bash
# Download unsupervised datasets for n-gram training
# Total: ~11 GB compressed, ~51 GB uncompressed
# Datasets: WikiText-103, BookCorpus+, USENET, 1 Billion Word, OpenWebText (sample)

set -e

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$PROJECT_ROOT"

# Create directories
mkdir -p data/unsupervised
mkdir -p data/processed/unsupervised

echo "================================================"
echo "Downloading Unsupervised Datasets"
echo "Total size: ~4 GB compressed, ~11 GB uncompressed"
echo "(OpenWebText excluded to reduce size)"
echo "================================================"
echo ""

# Check Python dependencies
echo "Checking Python dependencies..."
python3 -c "import datasets" 2>/dev/null || {
    echo "Installing datasets library..."
    pip install datasets
}

# ============================================================================
# 1. WikiText-103 (~0.1 GB compressed)
# ============================================================================
echo ""
echo "1/5: Downloading WikiText-103 (~0.1 GB)..."
python scripts/data_preparation/download_wikitext.py \
    --output data/processed/unsupervised/wikitext103.txt || {
    echo "Warning: WikiText-103 download failed, continuing..."
}

# ============================================================================
# 2. BookCorpus+ (~2 GB compressed)
# ============================================================================
echo ""
echo "2/5: Downloading BookCorpus+ (~2 GB)..."
BOOKCORPUS_OUTPUT="data/processed/unsupervised/bookcorpus.txt"
python3 << 'PYEOF'
from datasets import load_dataset
import os

print("Downloading BookCorpus (this may take 10-20 minutes)...")
output_file = "data/processed/unsupervised/bookcorpus.txt"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

try:
    dataset = load_dataset("bookcorpus", split="train", streaming=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, item in enumerate(dataset):
            if i % 10000 == 0:
                print(f"  Processed {i:,} samples...", end='\r')
            if 'text' in item:
                f.write(item['text'] + '\n')
            elif 'content' in item:
                f.write(item['content'] + '\n')
            # Limit to reasonable size
            if i >= 500000:  # ~500K samples
                break
    print(f"\n✓ BookCorpus downloaded: {i:,} samples")
except Exception as e:
    print(f"✗ BookCorpus download failed: {e}")
    print("Note: BookCorpus may require special access. Skipping...")
PYEOF

# ============================================================================
# 3. USENET Corpus (~0.8 GB compressed)
# ============================================================================
echo ""
echo "3/5: Downloading USENET Corpus (~0.8 GB)..."
USENET_OUTPUT="data/processed/unsupervised/usenet.txt"
python3 << 'PYEOF'
from datasets import load_dataset
import os

print("Downloading USENET Corpus...")
output_file = "data/processed/unsupervised/usenet.txt"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

try:
    dataset = load_dataset("usenet", split="train", streaming=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, item in enumerate(dataset):
            if i % 10000 == 0:
                print(f"  Processed {i:,} samples...", end='\r')
            if 'text' in item:
                f.write(item['text'] + '\n')
            elif 'content' in item:
                f.write(item['content'] + '\n')
            # Limit to reasonable size
            if i >= 200000:  # ~200K samples
                break
    print(f"\n✓ USENET downloaded: {i:,} samples")
except Exception as e:
    print(f"✗ USENET download failed: {e}")
    print("Note: USENET dataset may not be available. Skipping...")
PYEOF

# ============================================================================
# 4. 1 Billion Word Benchmark (~1.5 GB compressed)
# ============================================================================
echo ""
echo "4/5: Downloading 1 Billion Word Benchmark (~1.5 GB)..."
ONEBILLION_DIR="data/unsupervised/1billion"
mkdir -p "$ONEBILLION_DIR"
cd "$ONEBILLION_DIR"

if [ ! -f "1-billion-word-language-modeling-benchmark-r13output.tar.gz" ]; then
    echo "Downloading 1 Billion Word Benchmark..."
    wget -c "http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz" || {
        echo "Warning: 1 Billion Word download failed. You may need to download manually."
        cd "$PROJECT_ROOT"
    }
else
    echo "1 Billion Word archive already exists."
fi

if [ -f "1-billion-word-language-modeling-benchmark-r13output.tar.gz" ]; then
    if [ ! -d "training-monolingual.tokenized.shuffled" ]; then
        echo "Extracting 1 Billion Word Benchmark..."
        tar -xzf 1-billion-word-language-modeling-benchmark-r13output.tar.gz || {
            echo "Warning: Extraction failed"
        }
    fi
    
    if [ -d "training-monolingual.tokenized.shuffled" ]; then
        echo "Combining 1 Billion Word files..."
        cd "$PROJECT_ROOT"
        ONEBILLION_OUTPUT="data/processed/unsupervised/1billion_words.txt"
        # Combine first 10 files (to limit size)
        cat "$ONEBILLION_DIR/training-monolingual.tokenized.shuffled/news.en-0000"*of-00100 | head -n 1000000 > "$ONEBILLION_OUTPUT" || {
            echo "Warning: Failed to combine files"
        }
        echo "✓ 1 Billion Word processed"
    fi
fi

cd "$PROJECT_ROOT"

# ============================================================================
# 5. OpenWebText Sample - REMOVED (too large: ~7 GB compressed, ~40 GB uncompressed)
# ============================================================================
# Skipping OpenWebText to reduce download size
echo ""
echo "5/5: Skipping OpenWebText (too large for this setup)"
echo "     If needed, you can download it separately later"

# ============================================================================
# Preprocess all downloaded datasets
# ============================================================================
echo ""
echo "================================================"
echo "Preprocessing downloaded datasets..."
echo "================================================"

for dataset_file in data/processed/unsupervised/*.txt; do
    if [ -f "$dataset_file" ]; then
        echo ""
        echo "Preprocessing $(basename $dataset_file)..."
        python scripts/data_preparation/preprocess_corpus.py \
            --input "$dataset_file" \
            --output "${dataset_file%.txt}_cleaned.txt" || {
            echo "Warning: Preprocessing failed for $(basename $dataset_file)"
        }
    fi
done

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "================================================"
echo "Unsupervised Download Complete!"
echo "================================================"
echo ""
echo "Downloaded datasets:"
ls -lh data/processed/unsupervised/*.txt 2>/dev/null | awk '{print "  -", $9, "(" $5 ")"}' || echo "  (check processed files)"
echo ""
echo "Next steps:"
echo "  1. Merge datasets:"
echo "     python scripts/data_preparation/merge_corpora.py \\"
echo "         --new-files data/processed/unsupervised/*_cleaned.txt \\"
echo "         --output data/processed/unsupervised/combined.txt"
echo ""
echo "  2. Train models:"
echo "     python scripts/train_ngram_model.py \\"
echo "         --data data/processed/unsupervised/combined.txt \\"
echo "         --output models/ngram \\"
echo "         --use-dictionary"
echo ""
echo "  3. Evaluate:"
echo "     python scripts/evaluate_ngram.py"
echo ""

