#!/usr/bin/env python3
"""
Merge new corpus files with existing processed data for n-gram training.

Usage:
    python scripts/data_preparation/merge_corpora.py \
        --new-files data/new_corpus1.txt data/new_corpus2.txt \
        --output data/processed/unsupervised/combined_corpus.txt
    
    # Include existing data
    python scripts/data_preparation/merge_corpora.py \
        --new-files data/new_corpus.txt \
        --existing-dir data/processed/unsupervised \
        --output data/processed/unsupervised/combined_corpus.txt
"""

import argparse
import os
from pathlib import Path
from typing import List


def merge_corpora(
    new_data_files: List[str],
    existing_data_dir: str = None,
    output_file: str = "data/processed/unsupervised/combined_corpus.txt"
):
    """
    Merge new corpus files with existing processed data.
    
    Args:
        new_data_files: List of paths to new corpus files
        existing_data_dir: Directory containing existing processed files (optional)
        output_file: Path to save merged corpus
    """
    all_files = []
    
    # Add existing files if directory specified
    if existing_data_dir and os.path.exists(existing_data_dir):
        existing_files = list(Path(existing_data_dir).glob("*.txt"))
        all_files.extend(existing_files)
        print(f"Found {len(existing_files)} existing corpus files")
    
    # Add new files
    new_paths = [Path(f) for f in new_data_files]
    all_files.extend(new_paths)
    
    print(f"\nMerging {len(all_files)} corpus files...")
    print(f"Output: {output_file}")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    total_lines = 0
    files_processed = 0
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for filepath in all_files:
            if not filepath.exists():
                print(f"Warning: {filepath} not found, skipping...")
                continue
                
            print(f"Processing {filepath}...", end=" ")
            file_lines = 0
            
            try:
                with open(filepath, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        line = line.strip()
                        if line:  # Skip empty lines
                            outfile.write(line + '\n')
                            total_lines += 1
                            file_lines += 1
                
                print(f"✓ ({file_lines:,} lines)")
                files_processed += 1
            except Exception as e:
                print(f"✗ Error: {e}")
    
    print(f"\n{'='*60}")
    print(f"Merged corpus saved to: {output_file}")
    print(f"Files processed: {files_processed}/{len(all_files)}")
    print(f"Total lines: {total_lines:,}")
    print(f"{'='*60}")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Merge corpus files for n-gram training"
    )
    
    parser.add_argument(
        "--new-files",
        nargs="+",
        required=True,
        help="New corpus files to merge"
    )
    
    parser.add_argument(
        "--existing-dir",
        type=str,
        default=None,
        help="Directory containing existing processed corpus files (optional)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/unsupervised/combined_corpus.txt",
        help="Output file path (default: data/processed/unsupervised/combined_corpus.txt)"
    )
    
    args = parser.parse_args()
    
    merge_corpora(
        new_data_files=args.new_files,
        existing_data_dir=args.existing_dir,
        output_file=args.output
    )


if __name__ == "__main__":
    main()

