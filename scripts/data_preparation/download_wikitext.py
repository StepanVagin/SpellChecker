#!/usr/bin/env python3
"""
Download and prepare WikiText-103 dataset for n-gram training.

WikiText-103 is a clean, high-quality dataset of Wikipedia articles.
Size: ~515MB uncompressed, ~103 million tokens.

Usage:
    python scripts/data_preparation/download_wikitext.py \
        --output data/processed/unsupervised/wikitext103.txt
"""

import argparse
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from spellchecker.data.parsers.unsupervised_parser import UniversalTextCleaner


def download_wikitext(output_file: str = "data/processed/unsupervised/wikitext103.txt"):
    """
    Download WikiText-103 dataset and prepare for n-gram training.
    
    Args:
        output_file: Path to save processed corpus
    """
    # URLs for WikiText-103
    base_url = "https://s3.amazonaws.com/research.metamind.io/wikitext"
    zip_file = "wikitext-103-v1.zip"
    zip_url = f"{base_url}/{zip_file}"
    
    # Create directories
    temp_dir = Path("data/wikitext")
    temp_dir.mkdir(parents=True, exist_ok=True)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    zip_path = temp_dir / zip_file
    
    # Download if not exists
    if not zip_path.exists():
        print(f"Downloading {zip_file}...")
        print(f"URL: {zip_url}")
        print("This may take a few minutes (~100MB download)...")
        
        try:
            # Handle redirects properly
            request = urllib.request.Request(zip_url)
            request.add_header('User-Agent', 'Mozilla/5.0')
            
            with urllib.request.urlopen(request, timeout=60) as response:
                # Follow redirects
                final_url = response.geturl()
                if final_url != zip_url:
                    print(f"Following redirect to: {final_url}")
                
                # Download with progress
                total_size = int(response.headers.get('Content-Length', 0))
                downloaded = 0
                chunk_size = 8192
                
                with open(zip_path, 'wb') as out_file:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        out_file.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"  Progress: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='\r')
                
                print(f"\n✓ Download complete: {zip_path}")
        except Exception as e:
            print(f"✗ Download failed: {e}")
            # Try alternative URL
            alt_url = "https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/"
            print(f"\nAlternative download methods:")
            print(f"  1. Try manually: wget {zip_url}")
            print(f"  2. Or visit: {alt_url}")
            print(f"  3. Or use Hugging Face: datasets.load_dataset('wikitext', 'wikitext-103-v1')")
            return False
    else:
        print(f"✓ Zip file already exists: {zip_path}")
    
    # Extract training data
    train_file = temp_dir / "wikitext-103" / "wiki.train.tokens"
    
    if not train_file.exists():
        print(f"\nExtracting {zip_file}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            print(f"✓ Extraction complete")
        except Exception as e:
            print(f"✗ Extraction failed: {e}")
            return False
    else:
        print(f"✓ Extracted files already exist")
    
    # Process the training data
    print(f"\nProcessing WikiText-103 training data...")
    print(f"Input:  {train_file}")
    print(f"Output: {output_file}")
    
    cleaner = UniversalTextCleaner(
        min_length=10,
        max_length=1000,
        remove_urls=True,
        remove_emails=True,
        normalize_whitespace=True
    )
    
    lines_written = 0
    lines_skipped = 0
    
    with open(train_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            if line_num % 10000 == 0:
                print(f"  Processed {line_num:,} lines...", end='\r')
            
            cleaned = cleaner.clean(line.strip())
            if cleaned:
                outfile.write(cleaned + '\n')
                lines_written += 1
            else:
                lines_skipped += 1
    
    print(f"\n{'='*60}")
    print(f"WikiText-103 processing complete:")
    print(f"  Lines written:        {lines_written:,}")
    print(f"  Lines skipped:        {lines_skipped:,}")
    print(f"  Output saved to:      {output_file}")
    print(f"{'='*60}")
    print(f"\nYou can now use this corpus for training:")
    print(f"  python scripts/train_ngram_model.py \\")
    print(f"      --data {output_file} \\")
    print(f"      --output models/ngram")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare WikiText-103 dataset"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/unsupervised/wikitext103.txt",
        help="Output file path (default: data/processed/unsupervised/wikitext103.txt)"
    )
    
    args = parser.parse_args()
    
    download_wikitext(output_file=args.output)


if __name__ == "__main__":
    main()

