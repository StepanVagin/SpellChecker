"""
Download BookCorpus dataset using Hugging Face datasets library.

Install: pip install datasets
"""

from datasets import load_dataset
from pathlib import Path
import sys

def download_bookcorpus(output_dir="books", max_examples=50):
    """
    Download BookCorpus dataset and save as text files.
    
    Note: Each example in the dataset is an entire book chapter with embedded newlines.
    The default downloads only 50 examples to keep the file size manageable (~18 MB).
    
    Args:
        output_dir: Output directory for book files
        max_examples: Maximum number of book chapters to download (default: 50)
    """
    print("Loading BookCorpus dataset from Hugging Face...")
    print(f"Will download {max_examples} book chapters")
    print("(Note: Each chapter contains multiple lines of text)")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load dataset using the working bookcorpusopen dataset
    try:
        dataset = load_dataset("lucadiliello/bookcorpusopen", split="train", streaming=True)
    except Exception as e:
        print(f"Error loading BookCorpus: {e}")
        return False
    
    # Save as single file (one chapter per write, preserving newlines)
    output_file = output_path / "bookcorpus.txt"
    
    print(f"Saving to {output_file}...")
    count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for i, example in enumerate(dataset):
            if i >= max_examples:
                break
                
            text = example["text"].strip()
            if text:
                f.write(text + "\n")
                count += 1
                
                if count % 10 == 0:
                    print(f"Downloaded {count} book chapters...")
    
    print(f"✅ BookCorpus saved to {output_file} ({count} book chapters)")
    return True

if __name__ == "__main__":
    # Get max_examples from command line or use default (50 chapters = ~18MB)
    max_examples = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    success = download_bookcorpus("books", max_examples=max_examples)
    sys.exit(0 if success else 1)
