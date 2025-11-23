#!/usr/bin/env python3
"""
Preprocess corpus files with consistent cleaning for n-gram training.

Usage:
    python scripts/data_preparation/preprocess_corpus.py \
        --input data/new_corpus.txt \
        --output data/processed/unsupervised/new_corpus.txt
    
    # With custom parameters
    python scripts/data_preparation/preprocess_corpus.py \
        --input data/new_corpus.txt \
        --output data/processed/unsupervised/new_corpus.txt \
        --min-length 20 \
        --max-length 500
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from spellchecker.data.parsers.unsupervised_parser import UniversalTextCleaner


def preprocess_corpus(
    input_file: str,
    output_file: str,
    min_length: int = 10,
    max_length: int = 1000,
    remove_urls: bool = True,
    remove_emails: bool = True,
    normalize_whitespace: bool = True
):
    """
    Preprocess corpus with consistent cleaning.
    
    Args:
        input_file: Path to raw corpus file
        output_file: Path to save cleaned corpus
        min_length: Minimum sentence length
        max_length: Maximum sentence length
        remove_urls: Whether to remove URLs
        remove_emails: Whether to remove email addresses
        normalize_whitespace: Whether to normalize whitespace
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return
    
    cleaner = UniversalTextCleaner(
        min_length=min_length,
        max_length=max_length,
        remove_urls=remove_urls,
        remove_emails=remove_emails,
        normalize_whitespace=normalize_whitespace
    )
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    lines_written = 0
    lines_skipped = 0
    total_lines = 0
    
    print(f"Preprocessing {input_file}...")
    print(f"Output: {output_file}")
    print(f"Parameters: min_length={min_length}, max_length={max_length}")
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            total_lines += 1
            
            if line_num % 10000 == 0:
                print(f"  Processed {line_num:,} lines...", end='\r')
            
            cleaned = cleaner.clean(line.strip())
            if cleaned:
                outfile.write(cleaned + '\n')
                lines_written += 1
            else:
                lines_skipped += 1
    
    print(f"\n{'='*60}")
    print(f"Preprocessing complete:")
    print(f"  Total lines read:     {total_lines:,}")
    print(f"  Lines written:        {lines_written:,}")
    print(f"  Lines skipped:        {lines_skipped:,}")
    print(f"  Output saved to:     {output_file}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess corpus files for n-gram training"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input corpus file path"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path"
    )
    
    parser.add_argument(
        "--min-length",
        type=int,
        default=10,
        help="Minimum sentence length (default: 10)"
    )
    
    parser.add_argument(
        "--max-length",
        type=int,
        default=1000,
        help="Maximum sentence length (default: 1000)"
    )
    
    parser.add_argument(
        "--keep-urls",
        action="store_true",
        help="Keep URLs in text (default: remove)"
    )
    
    parser.add_argument(
        "--keep-emails",
        action="store_true",
        help="Keep email addresses in text (default: remove)"
    )
    
    args = parser.parse_args()
    
    preprocess_corpus(
        input_file=args.input,
        output_file=args.output,
        min_length=args.min_length,
        max_length=args.max_length,
        remove_urls=not args.keep_urls,
        remove_emails=not args.keep_emails,
        normalize_whitespace=True
    )


if __name__ == "__main__":
    main()

