#!/bin/bash
# scripts/data_preparation/download_raw_data.sh

set -e

# Create data/raw directory
mkdir -p data/raw
cd data/raw

echo "Downloading NUCLE corpus..."
wget http://www.comp.nus.edu.sg/~nlp/conll13st/release2.3.1.tar.gz
tar -xzf release2.3.1.tar.gz
mv release2.3.1 nucle
rm release2.3.1.tar.gz

echo "Downloading CoNLL-2014 test data..."
wget https://www.comp.nus.edu.sg/~nlp/conll14st/conll14st-test-data.tar.gz
tar -xzf conll14st-test-data.tar.gz
mv conll14st-test-data conll2014
rm conll14st-test-data.tar.gz

echo "Downloading Birkbeck Spelling Corpus (1986)..."
wget https://llds.ling-phil.ox.ac.uk/llds/xmlui/bitstream/handle/20.500.14106/0643/0643.zip
unzip 0643.zip -d birkbeck_spelling_1986
rm 0643.zip

echo "Downloading GitHub Typo Corpus..."
wget https://github-typo-corpus.s3.amazonaws.com/data/github-typo-corpus.v1.0.0.jsonl.gz
gunzip github-typo-corpus.v1.0.0.jsonl.gz
mv github-typo-corpus.v1.0.0.jsonl github_typo_corpus.jsonl

echo "✅ Raw data downloaded successfully"
