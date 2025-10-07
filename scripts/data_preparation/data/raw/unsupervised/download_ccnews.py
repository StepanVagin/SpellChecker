"""
Download CC-News dataset using Hugging Face datasets library.

Install: pip install datasets
"""

from datasets import load_dataset
from pathlib import Path
import json
import sys
import os
import gzip
import pickle

# Make project src importable when invoked from shell script
try:
    from spellchecker.data.parsers.unsupervised_parser import UniversalTextCleaner
except Exception:
    # Allow running directly by adding src relative to this file
    repo_root = Path(__file__).resolve().parents[5]
    sys.path.insert(0, str(repo_root / "src"))
    from spellchecker.data.parsers.unsupervised_parser import UniversalTextCleaner

def download_ccnews(output_file="ccnews.pkl.gz", num_samples=None):
    """
    Download CC-News dataset and save as JSONL.
    
    Args:
        output_file: Output file path
        num_samples: Number of samples to download (None for all)
    """
    print(f"Loading CC-News dataset from Hugging Face...")
    if num_samples:
        print(f"Limited to {num_samples} articles")
    
    # Configure HF Hub to be more resilient
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("HF_HUB_TIMEOUT", "60")

    # If file already exists and is non-empty, skip
    out_path = Path(output_file)
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"Output already exists, skipping: {out_path}")
        return True

    try:
        # Non-streaming load; use slicing when num_samples is provided
        split = f"train[:{num_samples}]" if num_samples else "train"
        dataset = load_dataset("cc_news", split=split, streaming=False)
        cleaner = UniversalTextCleaner()
        count = 0
        cleaned_texts = []
        for example in dataset:
            title = example.get("title", "")
            text = example.get("text", "")
            full_text = f"{title}. {text}" if title else text
            cleaned = cleaner.clean(full_text)
            if cleaned:
                cleaned_texts.append(cleaned)
                count += 1
                if count % 10000 == 0:
                    print(f"Processed {count} articles...")

        # Save compressed pickle
        with gzip.open(out_path, "wb") as f:
            pickle.dump(cleaned_texts, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Saved {count} cleaned CC-News articles to {out_path}")
        try:
            del dataset  # type: ignore[name-defined]
        except Exception:
            pass
        return count > 0
    except Exception as e:
        print(f"Error downloading CC-News: {e}")
        return False

if __name__ == "__main__":
    # Args: [num_samples] [output_file]
    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else None
    output_file = sys.argv[2] if len(sys.argv) > 2 else "ccnews.pkl.gz"
    success = download_ccnews(output_file, num_samples=num_samples)
    sys.exit(0 if success else 1)

