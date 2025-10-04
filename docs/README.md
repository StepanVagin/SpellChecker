# SpellChecker: Quick Setup & Test Guide

Get the spell checker running with unsupervised data in minutes.

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install pandas wikiextractor datasets kenlm

# 2. Download and setup data (choose one option)
# Option A: Small dataset (fastest, ~10 minutes)
bash scripts/data_preparation/download_unsupervised_data.sh
cd data/raw/unsupervised/ccnews
# Edit download_ccnews.py: change num_samples=100000 to num_samples=10000
python download_ccnews.py
cd ../../..

# Option B: Full Wikipedia (best quality, ~2 hours)
bash scripts/data_preparation/download_unsupervised_data.sh
cd data/raw/unsupervised/wikipedia
wikiextractor enwiki-latest-pages-articles.xml.bz2 -o extracted
cd ../../..

# 3. Process the data
python scripts/data_preparation/process_unsupervised_data.py --all

# 4. Test it works
python scripts/data_preparation/example_ngram_usage.py
```

## 📊 Dataset Options

| Dataset | Size | Download Time | Best For |
|---------|------|---------------|----------|
| CC-News (10k) | 100MB | 2 min | Quick testing |
| CC-News (100k) | 1GB | 10 min | Development |
| Wikipedia | 20GB | 60 min | Production quality |

## 🔧 Installation

### Prerequisites
- Python 3.8+
- 30GB free disk space (for full setup)
- Stable internet connection

### Install Dependencies
```bash
pip install pandas wikiextractor datasets kenlm
```

## 📥 Download & Process Data

### Option 1: Quick Test (Recommended for first try)
```bash
# Download small CC-News dataset
mkdir -p data/raw/unsupervised/ccnews
cd data/raw/unsupervised/ccnews

# Create download script with small sample
cat > download_ccnews.py << 'EOF'
from datasets import load_dataset
import json

dataset = load_dataset("cc_news", split="train", streaming=True)
with open("ccnews.jsonl", "w", encoding="utf-8") as f:
    for i, example in enumerate(dataset):
        if i >= 10000:  # Small sample for testing
            break
        json.dump({
            "title": example.get("title", ""),
            "text": example.get("text", ""),
        }, f)
        f.write("\n")
        if i % 1000 == 0:
            print(f"Downloaded {i} articles...")
print(f"✅ Downloaded {i+1} articles")
EOF

python download_ccnews.py
cd ../../..

# Process the data
python scripts/data_preparation/process_unsupervised_data.py \
    --corpus ccnews \
    --input data/raw/unsupervised/ccnews/ccnews.jsonl \
    --output data/processed/unsupervised/ccnews.txt
```

### Option 2: Full Wikipedia (Production Quality)
```bash
# Download Wikipedia
bash scripts/data_preparation/download_unsupervised_data.sh
cd data/raw/unsupervised/wikipedia
wikiextractor enwiki-latest-pages-articles.xml.bz2 -o extracted
cd ../../..

# Process Wikipedia
python scripts/data_preparation/process_unsupervised_data.py \
    --corpus wikipedia \
    --input data/raw/unsupervised/wikipedia/extracted \
    --output data/processed/unsupervised/wikipedia.txt
```

## 🧪 Test the Setup

### Basic Test
```bash
python scripts/data_preparation/example_ngram_usage.py
```

Expected output:
```
✅ Training trigram model...
✅ Model trained on X sentences
Perplexity on test sentences:
- "This is a test sentence.": 45.2
- "The quick brown fox jumps.": 38.7
✅ Setup working correctly!
```

### Train Production Model
```bash
# Train 5-gram model with KenLM
lmplz -o 5 < data/processed/unsupervised/ccnews.txt > model.arpa
build_binary model.arpa model.klm

# Test the model
python -c "
import kenlm
model = kenlm.Model('model.klm')
score = model.score('This is a test sentence.', bos=True, eos=True)
print(f'Model score: {score}')
print('✅ Production model working!')
"
```

## 🔍 Verify Installation

Check your directory structure:
```bash
tree data/ -L 3
```

Should look like:
```
data/
├── processed/
│   └── unsupervised/
│       └── ccnews.txt (or wikipedia.txt)
└── raw/
    └── unsupervised/
        └── ccnews/ (or wikipedia/)
```

Check processed data:
```bash
# Count lines
wc -l data/processed/unsupervised/*.txt

# Preview content
head -5 data/processed/unsupervised/*.txt
```

## ❗ Troubleshooting

**Problem**: `wikiextractor` command not found  
**Solution**: `pip install wikiextractor` or use `python -m wikiextractor.WikiExtractor`

**Problem**: Out of memory during processing  
**Solution**: Use smaller dataset or process in chunks

**Problem**: Datasets library fails to download  
**Solution**: Check internet connection, try with VPN

**Problem**: Processed file is empty  
**Solution**: Check if raw data exists: `ls -la data/raw/unsupervised/*/`

**Problem**: Permission denied on scripts  
**Solution**: `chmod +x scripts/data_preparation/*.sh`

## 🎯 Next Steps

1. **Integrate with spell checker**: Use trained n-gram models for scoring corrections
2. **Experiment with parameters**: Try different n-gram orders (3, 4, 5)
3. **Combine datasets**: Merge multiple corpora for better coverage
4. **Evaluate performance**: Test on spell checking benchmarks

## 📚 File Structure

```
SpellChecker/
├── data/
│   ├── processed/unsupervised/    # Clean text for training
│   └── raw/unsupervised/          # Downloaded datasets
├── scripts/data_preparation/      # Download & processing scripts
└── src/spellchecker/             # Main spell checker code
```

---

**Total setup time**: 10 minutes (small dataset) to 2 hours (full Wikipedia)  
**Disk space needed**: 1GB (small) to 30GB (full)

🎉 **You're ready to go!** Run the test script to verify everything works.
