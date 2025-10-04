# Data Preparation

This directory contains scripts for downloading and preparing datasets for spell checking model training.

## Supervised Datasets

To download supervised datasets (for training supervised spell checkers), execute:

```bash
bash scripts/data_preparation/download_raw_data.sh
```

This downloads:
- NUCLE corpus
- CoNLL-2014 test data
- Birkbeck Spelling Corpus (1986)
- GitHub Typo Corpus

## Unsupervised Datasets

For n-gram language model training, download unsupervised text corpora:

```bash
bash scripts/data_preparation/download_unsupervised_data.sh
```

This prepares download scripts for:
- **Wikipedia Dumps** - Regularly updated text dumps of Wikipedia articles
- **CC-News** - News articles from Common Crawl
- **BookCorpus** - Collection of novels from unpublished authors

### Processing Unsupervised Data

After downloading the raw unsupervised data, process it for n-gram training:

```bash
# Process individual corpus
python scripts/data_preparation/process_unsupervised_data.py \
    --corpus wikipedia \
    --input data/raw/unsupervised/wikipedia/extracted \
    --output data/processed/unsupervised/wikipedia.txt

# Or process all corpora at once with default paths
python scripts/data_preparation/process_unsupervised_data.py --all
```

The processed files will be saved in `data/processed/unsupervised/` with one cleaned text passage per line, ready for n-gram model training.

### Detailed Steps

1. **Download Wikipedia**:
   ```bash
   cd data/raw/unsupervised/wikipedia
   # The download script will fetch the latest English Wikipedia dump
   # Install WikiExtractor
   pip install wikiextractor
   # Extract text from XML
   wikiextractor enwiki-latest-pages-articles.xml.bz2 -o extracted
   ```

2. **Download CC-News**:
   ```bash
   cd data/raw/unsupervised/ccnews
   pip install datasets
   python download_ccnews.py
   ```

3. **Download BookCorpus**:
   ```bash
   cd data/raw/unsupervised/bookcorpus
   pip install datasets
   python download_bookcorpus.py
   ```

4. **Process all datasets**:
   ```bash
   cd ../../..  # Back to project root
   python scripts/data_preparation/process_unsupervised_data.py --all
   ```
