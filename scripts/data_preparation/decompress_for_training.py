#!/usr/bin/env python3
"""
Decompress .pkl.gz files for training.
Converts compressed pickle files to .txt format that training scripts expect.

Usage:
    python scripts/data_preparation/decompress_for_training.py \
        --input data/processed/unsupervised/*.pkl.gz \
        --output data/processed/unsupervised/training/
"""

import argparse
import gzip
import pickle
import os
from pathlib import Path
from glob import glob


def decompress_file(input_file: str, output_file: str):
    """Decompress a pickle file to text file"""
    print(f"Decompressing {Path(input_file).name}...")
    
    try:
        # Load compressed data
        with gzip.open(input_file, 'rb') as f:
            texts = pickle.load(f)
        
        # Write to text file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as out_f:
            for text in texts:
                out_f.write(text + '\n')
        
        size_mb = os.path.getsize(output_file) / 1024 / 1024
        print(f"  ✓ Decompressed: {len(texts):,} lines, {size_mb:.1f} MB")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Decompress compressed pickle files for training"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input file pattern (e.g., 'data/processed/unsupervised/*.pkl.gz')"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/unsupervised/training",
        help="Output directory for decompressed .txt files"
    )
    
    args = parser.parse_args()
    
    # Find all matching files
    input_files = glob(args.input)
    
    if not input_files:
        print(f"No files found matching: {args.input}")
        return
    
    print(f"Found {len(input_files)} compressed file(s)")
    print(f"Output directory: {args.output}")
    print()
    
    os.makedirs(args.output, exist_ok=True)
    
    decompressed = 0
    for input_file in input_files:
        # Create output filename
        base_name = Path(input_file).stem.replace('.pkl', '')  # Remove .pkl from .pkl.gz
        output_file = os.path.join(args.output, f"{base_name}.txt")
        
        if decompress_file(input_file, output_file):
            decompressed += 1
    
    print("\n" + "="*60)
    print(f"Decompression complete: {decompressed}/{len(input_files)} files")
    print("="*60)
    print(f"\nDecompressed files are in: {args.output}")
    print("\nYou can now train using:")
    print(f"  python scripts/train_ngram_model.py \\")
    print(f"      --data \"{args.output}/*.txt\" \\")
    print(f"      --output models/ngram")


if __name__ == "__main__":
    main()

