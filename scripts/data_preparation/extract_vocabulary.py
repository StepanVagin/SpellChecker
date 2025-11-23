#!/usr/bin/env python3
"""
Extract vocabulary from corpus and optionally merge with existing vocabulary.

Usage:
    # Extract vocabulary from corpus
    python scripts/data_preparation/extract_vocabulary.py \
        --input data/processed/unsupervised/combined_corpus.txt \
        --output data/vocabulary.txt
    
    # Extract with minimum frequency filter
    python scripts/data_preparation/extract_vocabulary.py \
        --input data/processed/unsupervised/combined_corpus.txt \
        --output data/vocabulary.txt \
        --min-frequency 3
    
    # Merge with existing vocabulary
    python scripts/data_preparation/extract_vocabulary.py \
        --input data/processed/unsupervised/new_corpus.txt \
        --existing-vocab data/vocabulary.txt \
        --output data/vocabulary_updated.txt
"""

import argparse
import os
from collections import Counter, defaultdict
from pathlib import Path


def extract_vocabulary(
    corpus_file: str,
    output_vocab_file: str = "data/vocabulary.txt",
    min_frequency: int = 1
):
    """
    Extract vocabulary from corpus and save frequent words.
    
    Args:
        corpus_file: Path to processed corpus file
        output_vocab_file: Path to save vocabulary
        min_frequency: Minimum word frequency to include
    """
    if not os.path.exists(corpus_file):
        print(f"Error: Corpus file not found: {corpus_file}")
        return
    
    print(f"Extracting vocabulary from {corpus_file}...")
    print(f"Minimum frequency: {min_frequency}")
    
    word_counts = Counter()
    total_words = 0
    total_lines = 0
    
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            total_lines += 1
            
            if line_num % 100000 == 0:
                print(f"  Processed {line_num:,} lines...", end='\r')
            
            words = line.lower().strip().split()
            word_counts.update(words)
            total_words += len(words)
    
    print(f"\n  Processed {total_lines:,} lines")
    
    # Filter by minimum frequency
    vocabulary = {
        word: count 
        for word, count in word_counts.items() 
        if count >= min_frequency
    }
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_vocab_file), exist_ok=True)
    
    # Save vocabulary (sorted by frequency)
    with open(output_vocab_file, 'w', encoding='utf-8') as f:
        for word, count in sorted(vocabulary.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{word}\t{count}\n")
    
    print(f"\n{'='*60}")
    print(f"Vocabulary extracted:")
    print(f"  Total words processed:    {total_words:,}")
    print(f"  Unique words:             {len(word_counts):,}")
    print(f"  Vocabulary size (freq >= {min_frequency}): {len(vocabulary):,}")
    print(f"  Saved to:                 {output_vocab_file}")
    print(f"{'='*60}")
    
    return vocabulary


def merge_vocabularies(
    existing_vocab_file: str,
    new_vocab_file: str,
    output_file: str
):
    """Merge vocabulary files, combining counts."""
    vocab = defaultdict(int)
    
    # Load existing vocabulary
    if os.path.exists(existing_vocab_file):
        print(f"Loading existing vocabulary from {existing_vocab_file}...")
        with open(existing_vocab_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    vocab[parts[0]] += int(parts[1])
        print(f"  Loaded {len(vocab):,} words")
    
    # Add new vocabulary
    print(f"Loading new vocabulary from {new_vocab_file}...")
    new_words = 0
    with open(new_vocab_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                word = parts[0]
                count = int(parts[1])
                if word not in vocab:
                    new_words += 1
                vocab[word] += count
    
    print(f"  Added {new_words:,} new words")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save merged vocabulary
    with open(output_file, 'w', encoding='utf-8') as f:
        for word, count in sorted(vocab.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{word}\t{count}\n")
    
    print(f"\n{'='*60}")
    print(f"Merged vocabulary saved to: {output_file}")
    print(f"Total unique words: {len(vocab):,}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract vocabulary from corpus files"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input corpus file"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="data/vocabulary.txt",
        help="Output vocabulary file (default: data/vocabulary.txt)"
    )
    
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=1,
        help="Minimum word frequency to include (default: 1)"
    )
    
    parser.add_argument(
        "--existing-vocab",
        type=str,
        default=None,
        help="Existing vocabulary file to merge with (optional)"
    )
    
    args = parser.parse_args()
    
    # Extract vocabulary
    vocabulary = extract_vocabulary(
        corpus_file=args.input,
        output_vocab_file=args.output,
        min_frequency=args.min_frequency
    )
    
    # Merge with existing if specified
    if args.existing_vocab and os.path.exists(args.existing_vocab):
        temp_output = args.output + ".new"
        os.rename(args.output, temp_output)
        merge_vocabularies(
            existing_vocab_file=args.existing_vocab,
            new_vocab_file=temp_output,
            output_file=args.output
        )
        os.remove(temp_output)


if __name__ == "__main__":
    main()

