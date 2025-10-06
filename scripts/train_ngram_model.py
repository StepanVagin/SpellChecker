#!/usr/bin/env python3
"""
Train N-gram language models on unsupervised data for spelling correction.

This script integrates the unsupervised data processing pipeline with n-gram training.
It can use data from Wikipedia, CC-News, and BookCorpus that has been downloaded
and processed using the data preparation scripts.

Usage:
    # Train using processed unsupervised data
    python scripts/train_ngram_model.py --data data/processed/unsupervised/wikipedia.txt --output models/ngram
    
    # Train using multiple data sources
    python scripts/train_ngram_model.py --data data/processed/unsupervised/*.txt --output models/ngram
    
    # Train with custom settings
    python scripts/train_ngram_model.py --data data/processed/unsupervised/wikipedia.txt --output models/ngram --use-dictionary --threshold 0.00001
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
        default=None,
        help="Maximum number of lines to load from each file (for testing)",
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test examples after training",
    )

    return parser


def load_corpus_files(data_pattern: str, max_lines: int = None) -> list:
    """
    Load corpus from file(s) matching the pattern.
    
    Args:
        data_pattern: File path or glob pattern
        max_lines: Maximum lines to load (None for all)
    
    Returns:
        List of text lines
    """
    matching_files = glob(data_pattern)
    
    if not matching_files:
        print(f"Error: No files found matching pattern: {data_pattern}")
        return []
    
    print(f"\nFound {len(matching_files)} file(s) to process:")
    for f in matching_files:
        print(f"  - {f}")
    
    corpus = []
    for filepath in matching_files:
        print(f"\nLoading {filepath}...")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if max_lines:
                    lines = lines[:max_lines]
                corpus.extend([line.strip() for line in lines if line.strip()])
            print(f"  Loaded {len(lines)} lines")
        except Exception as e:
            print(f"  Error loading {filepath}: {e}")
            continue
    
    return corpus


def train_models(corpus: list, output_dir: str) -> list:
    """
    Train n-gram models (unigram, bigram, trigram).
    
    Args:
        corpus: List of training texts
        output_dir: Directory to save models
    
    Returns:
        List of trained models
    """
    os.makedirs(output_dir, exist_ok=True)
    
    ngram_models = []
    for n in [1, 2, 3]:
        print(f"\n{'='*60}")
        print(f"Training {n}-gram model")
        print(f"{'='*60}")
        
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
    print("\nYou can now use these models in the web interface or load them programmatically.")
    print()


if __name__ == "__main__":
    main()

