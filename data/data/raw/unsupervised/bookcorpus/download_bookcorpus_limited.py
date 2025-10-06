"""
Download limited BookCorpus dataset (100K passages, ~500MB)
"""
from datasets import load_dataset
from pathlib import Path
import sys
import signal

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Download timed out")

def download_bookcorpus_limited(output_dir="books", max_passages=100000):
    print(f"Loading BookCorpus dataset (limited to {max_passages} passages)...")
    print("Timeout: 10 minutes")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    output_file = output_path / "bookcorpus.txt"
    
    count = 0
    try:
        # Set 10 minute timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(600)
        
        try:
            dataset = load_dataset("bookcorpus", split="train", streaming=True)
        except Exception as e:
            print(f"Error loading bookcorpus: {e}")
            try:
                print("Trying alternative: bookcorpusopen")
                dataset = load_dataset("bookcorpusopen", split="train", streaming=True)
            except:
                print("Could not load BookCorpus, skipping...")
                signal.alarm(0)
                return False
        
        print(f"Saving to {output_file}...")
        with open(output_file, "w", encoding="utf-8") as f:
            for example in dataset:
                text = example["text"].strip()
                if text:
                    f.write(text + "\n")
                    count += 1
                    
                    if count >= max_passages:
                        break
                    
                    if count % 10000 == 0:
                        print(f"Processed {count} passages...")
        
        signal.alarm(0)  # Cancel timeout
        print(f"BookCorpus saved to {output_file} ({count} passages)")
        return True
        
    except TimeoutError:
        signal.alarm(0)
        print(f"\nDownload timed out after 10 minutes.")
        print(f"Saved {count} passages. Continuing with available data...")
        return count > 1000
    except KeyboardInterrupt:
        signal.alarm(0)
        print(f"\nInterrupted. Saved {count} passages. Continuing...")
        return count > 1000
    except Exception as e:
        signal.alarm(0)
        print(f"Error: {e}")
        if count > 1000:
            print(f"Saved {count} passages before error. Continuing...")
            return True
        return False

if __name__ == "__main__":
    success = download_bookcorpus_limited("books", max_passages=100000)
    sys.exit(0 if success else 1)
