"""
Download CC-News dataset using Hugging Face datasets library.

Install: pip install datasets
"""

from datasets import load_dataset
import json
import sys

def download_ccnews(output_file="ccnews.jsonl", num_samples=None):
    """
    Download CC-News dataset and save as JSONL.
    
    Args:
        output_file: Output file path
        num_samples: Number of samples to download (None for all)
    """
    print(f"Loading CC-News dataset from Hugging Face...")
    if num_samples:
        print(f"Limited to {num_samples} articles")
    
    try:
        # Load dataset
        # Note: Full dataset is very large (~70GB), so we can limit samples
        dataset = load_dataset("cc_news", split="train", streaming=True)
        
        count = 0
        with open(output_file, "w", encoding="utf-8") as f:
            for example in dataset:
                # Write as JSONL
                json.dump({
                    "title": example.get("title", ""),
                    "text": example.get("text", ""),
                    "domain": example.get("domain", ""),
                    "date": example.get("date", ""),
                }, f)
                f.write("\n")
                
                count += 1
                if num_samples and count >= num_samples:
                    break
                
                if count % 10000 == 0:
                    print(f"Downloaded {count} articles...")
        
        print(f"✅ Downloaded {count} articles to {output_file}")
        return True
    except Exception as e:
        print(f"Error downloading CC-News: {e}")
        return False

if __name__ == "__main__":
    # Get num_samples from command line or use default
    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else None
    success = download_ccnews("ccnews.jsonl", num_samples=num_samples)
    sys.exit(0 if success else 1)
