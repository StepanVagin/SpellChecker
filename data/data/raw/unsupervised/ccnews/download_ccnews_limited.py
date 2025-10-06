"""
Download limited CC-News dataset (50K articles, ~500MB)
"""
from datasets import load_dataset
import json
import sys

def download_ccnews_limited(output_file="ccnews.jsonl", num_samples=50000):
    print(f"Loading CC-News dataset (limited to {num_samples} articles)...")
    
    try:
        dataset = load_dataset("cc_news", split="train", streaming=True)
        
        count = 0
        with open(output_file, "w", encoding="utf-8") as f:
            for example in dataset:
                json.dump({
                    "title": example.get("title", ""),
                    "text": example.get("text", ""),
                    "domain": example.get("domain", ""),
                    "date": example.get("date", ""),
                }, f)
                f.write("\n")
                
                count += 1
                if count >= num_samples:
                    break
                
                if count % 5000 == 0:
                    print(f"Downloaded {count} articles...")
        
        print(f"Downloaded {count} articles to {output_file}")
        return True
    except Exception as e:
        print(f"Error downloading CC-News: {e}")
        return False

if __name__ == "__main__":
    success = download_ccnews_limited("ccnews.jsonl", num_samples=50000)
    sys.exit(0 if success else 1)
