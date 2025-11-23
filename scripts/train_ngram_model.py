#!/usr/bin/env python3
"""
Train N-gram language models on unsupervised data for spelling correction.

This script integrates the unsupervised data processing pipeline with n-gram training.
It can use data from Wikipedia, CC-News, and BookCorpus that has been downloaded
and processed using the data preparation scripts.

Trains 1-gram, 2-gram, and 3-gram models.
Keeps all words including stopwords (important for spell correction).
Default: trains on up to 10 million lines.

Usage:
    # Train using processed unsupervised data (default: 10M lines max, stopwords preserved)
    python scripts/train_ngram_model.py --data data/processed/unsupervised/wikipedia.txt --output models/ngram
    
    # Train using multiple data sources
    python scripts/train_ngram_model.py --data data/processed/unsupervised/*.txt --output models/ngram
    
    # Train with custom max lines
    python scripts/train_ngram_model.py --data data/processed/unsupervised/*.txt --output models/ngram --max-lines 5000000
"""

import argparse
import sys
import os
from pathlib import Path
from glob import glob

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spellchecker.models.ngram_model import (
    NGramModel,
    SpellingChecker,
    load_training_corpus,
    load_english_dictionary,
)


def setup_argparse() -> argparse.ArgumentParser:
    """Setup command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Train N-gram language models for spelling correction"
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to training data file(s). Supports wildcards like data/processed/unsupervised/*.txt",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="models/ngram",
        help="Directory to save trained models (default: models/ngram)",
    )

    parser.add_argument(
        "--use-dictionary",
        action="store_true",
        help="Include English dictionary words in training",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.000001,
        help="Probability threshold for corrections (default: 0.000001)",
    )

    parser.add_argument(
        "--max-lines",
        type=int,
        default=10000000,
        help="Maximum number of lines to load total across all files (default: 10,000,000). Stopwords are kept for spell correction.",
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test examples after training",
    )

    return parser


def load_corpus_files(data_pattern: str, max_lines: int = 10000000) -> list:
    """
    Load corpus from file(s) matching the pattern.
    
    Args:
        data_pattern: File path or glob pattern
        max_lines: Maximum total lines to load across all files (default: 10M)
    
    Returns:
        List of text lines (max max_lines total)
    """
    matching_files = glob(data_pattern)
    
    if not matching_files:
        print(f"Error: No files found matching pattern: {data_pattern}")
        return []
    
    print(f"\nFound {len(matching_files)} file(s) to process:")
    for f in matching_files:
        print(f"  - {f}")
    
    print(f"\nLoading up to {max_lines:,} lines total (stopwords preserved)...")
    
    corpus = []
    total_loaded = 0
    
    for filepath in matching_files:
        if total_loaded >= max_lines:
            print(f"\nReached maximum of {max_lines:,} lines. Stopping.")
            break
            
        print(f"\nLoading {filepath}...")
        try:
            remaining = max_lines - total_loaded
            with open(filepath, 'r', encoding='utf-8') as f:
                lines_to_read = []
                for i, line in enumerate(f):
                    if i >= remaining:
                        break
                    stripped = line.strip()
                    if stripped:  # Skip empty lines
                        lines_to_read.append(stripped)
                
                corpus.extend(lines_to_read)
                total_loaded += len(lines_to_read)
                print(f"  Loaded {len(lines_to_read):,} lines (total: {total_loaded:,}/{max_lines:,})")
        except Exception as e:
            print(f"  Error loading {filepath}: {e}")
            continue
    
    print(f"\nTotal corpus loaded: {len(corpus):,} lines")
    return corpus


def train_models(corpus: list, output_dir: str) -> list:
    """
    Train n-gram models (unigram, bigram, trigram).
    All words including stopwords are preserved for spell correction.
    
    Args:
        corpus: List of training texts (stopwords included, not filtered)
        output_dir: Directory to save models
    
    Returns:
        List of trained models
    """
    os.makedirs(output_dir, exist_ok=True)
    
    ngram_models = []
    for n in [1, 2, 3]:  # Train 1-gram, 2-gram, and 3-gram models
        print(f"\n{'='*60}")
        print(f"Training {n}-gram model")
        print(f"{'='*60}")
        print(f"Note: All words preserved (stopwords included for spell correction)")
        
        model = NGramModel(n=n)
        model.train(corpus)
        
        model_path = os.path.join(output_dir, f'{n}gram_model.json')
        model.save_model(model_path)
        
        ngram_models.append(model)
    
    return ngram_models


def test_spelling_checker(checker: SpellingChecker):
    """Run test examples on the spelling checker"""
    test_sentences = [
        "I love this prodct very much",
        "The weather is beutiful today",
        "She went to the libary yesterday",
        "This is a wonderfull day",
        "The algoritm works very well",
        "He is studyng computer scince",
        "Please chek your spellig carefully",
        "The experince was amazeing",
    ]
    
    print("\n" + "="*60)
    print("TESTING SPELLING CHECKER")
    print("="*60)
    
    for sentence in test_sentences:
        corrected, corrections = checker.correct_text(sentence)
        print(f"\nOriginal:  {sentence}")
        print(f"Corrected: {corrected}")
        
        corrections_made = [c for c in corrections if c.original_word != c.corrected_word]
        if corrections_made:
            print("Corrections:")
            for corr in corrections_made:
                print(f"  '{corr.original_word}' -> '{corr.corrected_word}' (confidence: {corr.confidence:.3f})")


def main():
    """Main function"""
    parser = setup_argparse()
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("N-gram Language Model Training")
    print("="*60)
    print(f"Data source: {args.data}")
    print(f"Output directory: {args.output}")
    print(f"Max lines: {args.max_lines:,} (stopwords preserved)")
    print(f"Use dictionary: {args.use_dictionary}")
    print(f"Probability threshold: {args.threshold}")
    
    print("\n" + "="*60)
    print("Loading Training Data")
    print("="*60)
    
    corpus = load_corpus_files(args.data, args.max_lines)
    
    if not corpus:
        print("\nError: No data loaded. Exiting.")
        sys.exit(1)
    
    print(f"\nTotal corpus size: {len(corpus)} sentences")
    
    if args.use_dictionary:
        print("\nAdding English dictionary words...")
        dictionary_sentences = load_english_dictionary()
        corpus.extend(dictionary_sentences)
        print(f"Total training data: {len(corpus)} sentences")
    
    print("\n" + "="*60)
    print("Training N-gram Models")
    print("="*60)
    
    ngram_models = train_models(corpus, args.output)
    
    print("\n" + "="*60)
    print("Creating Spelling Checker")
    print("="*60)
    
    checker = SpellingChecker(ngram_models, args.threshold)
    print(f"Spelling checker initialized with {len(checker.vocabulary)} words in vocabulary")
    
    if args.test:
        test_spelling_checker(checker)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nModels saved to: {args.output}/")
    print("  - 1gram_model.json")
    print("  - 2gram_model.json")
    print("  - 3gram_model.json")
    print("\nNote: All words including stopwords are preserved in vocabulary")
    print("\nYou can now use these models in the web interface or load them programmatically.")
    print()


if __name__ == "__main__":
    main()

