"""
Download BookCorpus dataset using Hugging Face datasets library.

Install: pip install datasets
"""

from datasets import load_dataset
from pathlib import Path
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

def download_bookcorpus(output_file="bookcorpus.pkl.gz", max_passages=None, stream_sample_10k: bool = False):
    """
    Download BookCorpus dataset and save as text files.
    
    Args:
        output_dir: Output directory for book files
        max_passages: Maximum number of passages to download (None for all)
    """
    print("Loading BookCorpus dataset from Hugging Face...")
    if max_passages:
        print(f"Limited to {max_passages} passages")
    
    # Prepare output path (directory will be created later based on output_file)
    
    # Configure HF Hub for more reliable downloads
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("HF_HUB_TIMEOUT", "60")

    # Choose dataset loading strategy
    if stream_sample_10k:
        try:
            dataset = load_dataset("rojagtap/bookcorpus", split="train", streaming=True)
        except Exception as e:
            print(f"Error loading BookCorpus (streaming): {e}")
            return False
    else:
        # Non-streaming load; use slicing when max_passages is provided
        split = f"train[:{max_passages}]" if max_passages else "train"
        try:
            dataset = load_dataset("rojagtap/bookcorpus", split=split, streaming=False)
        except Exception as e:
            print(f"Error loading BookCorpus: {e}")
            return False
    
    # Save processed text directly (no raw file) to save disk space
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"Output already exists, skipping: {output_path}")
        return True

    cleaner = UniversalTextCleaner()
    print(f"Saving cleaned text to {output_path}...")
    count = 0
    cleaned_texts = []
    if stream_sample_10k:
        for example in dataset:
            text = (example.get("text") if isinstance(example, dict) else example["text"]) or ""
            text = text.strip()
            cleaned = cleaner.clean(text)
            if cleaned:
                cleaned_texts.append(cleaned)
                count += 1
                if count % 1000 == 0:
                    print(f"Processed {count} passages...")
                if count >= 10000:
                    break
    else:
        for example in dataset:
            text = (example.get("text") if isinstance(example, dict) else example["text"]) or ""
            text = text.strip()
            cleaned = cleaner.clean(text)
            if cleaned:
                cleaned_texts.append(cleaned)
                count += 1
                if count % 10000 == 0:
                    print(f"Processed {count} passages...")

    # Save compressed pickle
    with gzip.open(output_path, "wb") as f:
        pickle.dump(cleaned_texts, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"BookCorpus saved to {output_path} ({count} passages)")
    # Encourage cleanup of streaming resources before exiting
    try:
        del dataset  # type: ignore[name-defined]
    except Exception:
        pass
    return count > 0

if __name__ == "__main__":
    # Args: [max_passages] [output_file] [--stream10k]
    max_passages = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else None
    output_file = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "bookcorpus.pkl.gz"
    stream_sample_10k = any(arg == "--stream10k" for arg in sys.argv[1:])
    success = download_bookcorpus(output_file, max_passages=max_passages, stream_sample_10k=stream_sample_10k)
    sys.exit(0 if success else 1)

