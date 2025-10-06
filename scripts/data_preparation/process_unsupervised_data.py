#!/usr/bin/env python3
"""
Process downloaded unsupervised corpora for n-gram language model training.

This script:
1. Loads raw unsupervised text data (Wikipedia, CC-News, BookCorpus)
2. Cleans and normalizes the text
3. Saves processed text ready for n-gram training

Usage:
    python process_unsupervised_data.py --corpus wikipedia --input data/raw/unsupervised/wikipedia/extracted --output data/processed/wikipedia.txt
    python process_unsupervised_data.py --corpus ccnews --input data/raw/unsupervised/ccnews/ccnews.jsonl --output data/processed/ccnews.txt
    python process_unsupervised_data.py --corpus bookcorpus --input data/raw/unsupervised/bookcorpus/books/bookcorpus.txt --output data/processed/bookcorpus.txt
    
    # Process all at once
    python process_unsupervised_data.py --all
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from spellchecker.data.parsers.unsupervised_parser import (
    WikipediaParser,
    CCNewsParser,
    BookCorpusParser,
    UniversalTextCleaner,
    process_unsupervised_corpus,
)


def setup_argparse() -> argparse.ArgumentParser:
    """Setup command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Process unsupervised corpora for n-gram training"
    )

    parser.add_argument(
        "--corpus",
        type=str,
        choices=["wikipedia", "ccnews", "bookcorpus"],
        help="Type of corpus to process",
    )

    parser.add_argument(
        "--input",
        type=str,
        help="Path to input data (file or directory)",
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Path to output processed text file",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all corpora with default paths",
    )

    parser.add_argument(
        "--min-length",
        type=int,
        default=10,
        help="Minimum text length in characters (default: 10)",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=1000,
        help="Maximum text length in characters (default: 1000)",
    )

    parser.add_argument(
        "--keep-urls",
        action="store_true",
        help="Keep URLs in text (default: remove)",
    )

    parser.add_argument(
        "--keep-emails",
        action="store_true",
        help="Keep email addresses in text (default: remove)",
    )

    return parser


def process_single_corpus(
    corpus: str,
    input_path: str,
    output_path: str,
    cleaner: UniversalTextCleaner,
) -> None:
    """
    Process a single corpus.

    Args:
        corpus: Corpus type
        input_path: Input file or directory path
        output_path: Output file path
        cleaner: Text cleaner instance
    """
    print(f"\n{'=' * 60}")
    print(f"Processing {corpus.upper()}")
    print(f"{'=' * 60}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print()

    # Check if input exists
    if not Path(input_path).exists():
        print(f"⚠️  Warning: Input path does not exist: {input_path}")
        print(f"Skipping {corpus} - you can download it using download_unsupervised_data.sh")
        return

    # Process corpus
    try:
        count = process_unsupervised_corpus(
            corpus_type=corpus,
            input_path=input_path,
            output_file=output_path,
            cleaner=cleaner,
        )
        print(f"✅ Successfully processed {count} passages from {corpus}")
    except Exception as e:
        print(f"❌ Error processing {corpus}: {e}")
        import traceback
        traceback.print_exc()


def process_all_corpora(cleaner: UniversalTextCleaner) -> None:
    """
    Process all corpora with default paths.

    Args:
        cleaner: Text cleaner instance
    """
    # Define default paths
    base_raw_dir = Path("data/raw/unsupervised")
    base_output_dir = Path("data/processed/unsupervised")

    # Create output directory
    base_output_dir.mkdir(parents=True, exist_ok=True)

    corpora_config = [
        {
            "corpus": "wikipedia",
            "input": base_raw_dir / "wikipedia" / "extracted",
            "output": base_output_dir / "wikipedia.txt",
        },
        {
            "corpus": "ccnews",
            "input": base_raw_dir / "ccnews" / "ccnews.jsonl",
            "output": base_output_dir / "ccnews.txt",
        },
        {
            "corpus": "bookcorpus",
            "input": base_raw_dir / "bookcorpus" / "books" / "bookcorpus.txt",
            "output": base_output_dir / "bookcorpus.txt",
        },
    ]

    for config in corpora_config:
        process_single_corpus(
            corpus=config["corpus"],
            input_path=str(config["input"]),
            output_path=str(config["output"]),
            cleaner=cleaner,
        )


def main():
    """Main function"""
    parser = setup_argparse()
    args = parser.parse_args()

    # Create text cleaner
    cleaner = UniversalTextCleaner(
        min_length=args.min_length,
        max_length=args.max_length,
        remove_urls=not args.keep_urls,
        remove_emails=not args.keep_emails,
        normalize_whitespace=True,
    )

    print("\n" + "=" * 60)
    print("Unsupervised Corpus Processing")
    print("=" * 60)
    print(f"Text Cleaner Settings:")
    print(f"  - Min length: {args.min_length} chars")
    print(f"  - Max length: {args.max_length} chars")
    print(f"  - Remove URLs: {not args.keep_urls}")
    print(f"  - Remove emails: {not args.keep_emails}")
    print()

    if args.all:
        # Process all corpora
        process_all_corpora(cleaner)
    elif args.corpus and args.input and args.output:
        # Process single corpus
        process_single_corpus(
            corpus=args.corpus,
            input_path=args.input,
            output_path=args.output,
            cleaner=cleaner,
        )
    else:
        parser.print_help()
        print("\n❌ Error: Either use --all or provide --corpus, --input, and --output")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("✅ Processing complete!")
    print("=" * 60)
    print("\nThe processed files are ready for n-gram language model training.")
    print("Each line contains a cleaned text passage suitable for training.")
    print()


if __name__ == "__main__":
    main()


