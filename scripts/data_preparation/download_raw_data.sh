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

echo "✅ Raw data downloaded successfully"
