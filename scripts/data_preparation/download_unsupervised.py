#!/usr/bin/env python3
"""
Download unsupervised datasets for n-gram training.
Total: ~4 GB compressed, ~11 GB uncompressed

Datasets:
- WikiText-103: ~0.1 GB
- BookCorpus: Full dataset (if <= 5GB uncompressed)
- Reddit: ~0.8 GB
- 1 Billion Word: ~1.5 GB

Usage:
    python scripts/data_preparation/download_unsupervised.py
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from datasets import load_dataset
except ImportError:
    print("Installing datasets library...")
    os.system("pip install datasets")
    from datasets import load_dataset

from spellchecker.data.parsers.unsupervised_parser import UniversalTextCleaner


def safe_delete_file(filepath: str):
    """Safely delete a file if it exists"""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"  Deleted partial/corrupted file: {os.path.basename(filepath)}")
    except Exception as e:
        print(f"  Warning: Could not delete {filepath}: {e}")


def download_wikitext(output_file: str):
    """Download WikiText-103"""
    print("\n" + "="*60)
    print("1/4: Downloading WikiText-103 (~0.1 GB)...")
    print("="*60)
    
    # Check if file already exists and is valid
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        print(f"✓ File already exists: {os.path.basename(output_file)}")
        return True
    
    # Create directory structure
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    cleaner = UniversalTextCleaner()
    
    try:
        print("Trying Hugging Face wikitext dataset...")
        dataset = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True)
        
        count = 0
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in dataset:
                if count % 10000 == 0:
                    print(f"  Processed {count:,} lines...", end='\r')
                
                text = item.get('text', '')
                if text:
                    cleaned = cleaner.clean(text)
                    if cleaned:
                        f.write(cleaned + '\n')
                        count += 1
        
        print(f"\n✓ WikiText-103 downloaded: {count:,} lines")
        return True
    except Exception as e:
        print(f"✗ Download failed: {e}")
        safe_delete_file(output_file)
        return False


def download_bookcorpus(output_file: str):
    """Download BookCorpus full dataset (if <= 5GB uncompressed)"""
    print("\n" + "="*60)
    print("2/4: Downloading BookCorpus (full dataset, max 5GB uncompressed)...")
    print("="*60)
    
    # Check if file already exists and is valid
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        size_gb = os.path.getsize(output_file) / 1024 / 1024 / 1024
        print(f"✓ File already exists: {os.path.basename(output_file)} ({size_gb:.2f} GB)")
        return True
    
    # Create directory structure
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    cleaner = UniversalTextCleaner()
    
    # Configure HF Hub for more reliable downloads
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("HF_HUB_TIMEOUT", "120")
    
    try:
        print("Downloading BookCorpus from rojagtap/bookcorpus...")
        print("This may take 20-40 minutes for full dataset...")
        dataset = load_dataset("rojagtap/bookcorpus", split="train", streaming=True)
        
        count = 0
        total_size_bytes = 0
        max_size_bytes = 5 * 1024 * 1024 * 1024  # 5GB limit
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in dataset:
                if count % 10000 == 0:
                    size_gb = total_size_bytes / 1024 / 1024 / 1024
                    print(f"  Processed {count:,} samples ({size_gb:.2f} GB)...", end='\r')
                
                text = item.get('text', item.get('content', ''))
                if text:
                    cleaned = cleaner.clean(text)
                    if cleaned:
                        line = cleaned + '\n'
                        line_size = len(line.encode('utf-8'))
                        
                        # Check if adding this line would exceed 5GB
                        if total_size_bytes + line_size > max_size_bytes:
                            print(f"\n  Reached 5GB limit at {count:,} samples")
                            break
                        
                        f.write(line)
                        total_size_bytes += line_size
                        count += 1
        
        final_size_gb = total_size_bytes / 1024 / 1024 / 1024
        print(f"\n✓ BookCorpus downloaded: {count:,} samples ({final_size_gb:.2f} GB)")
        return True
    except Exception as e:
        print(f"✗ BookCorpus download failed: {e}")
        safe_delete_file(output_file)
        return False


def download_reddit(output_file: str, max_samples: int = 200000):
    """Download Reddit/conversational text dataset"""
    print("\n" + "="*60)
    print(f"3/4: Downloading Reddit dataset (~0.8 GB, max {max_samples:,} samples)...")
    print("="*60)
    
    # Check if file already exists and is valid
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        print(f"✓ File already exists: {os.path.basename(output_file)}")
        return True
    
    # Create directory structure
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    cleaner = UniversalTextCleaner()
    
    # Configure HF Hub
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("HF_HUB_TIMEOUT", "120")
    
    # Try multiple Reddit/conversational text sources
    alternatives = [
        # Try various Reddit datasets
        ("reddit", None, "train"),
        ("tristan22/reddit", None, "train"),
        ("jamescalam/reddit-top-posts", None, "train"),
        # Try C4 as fallback (contains conversational text)
        ("c4", "en", "train"),
    ]
    
    dataset = None
    dataset_name = None
    
    for alt in alternatives:
        try:
            if alt[1] is None:
                # Two parameter version
                print(f"Trying {alt[0]} dataset...")
                dataset = load_dataset(alt[0], split=alt[2], streaming=True)
            else:
                # Three parameter version
                print(f"Trying {alt[0]} ({alt[1]}) dataset...")
                dataset = load_dataset(alt[0], alt[1], split=alt[2], streaming=True)
            
            # Test if we can read from it
            try:
                next(iter(dataset))
                dataset_name = alt[0]
                print(f"✓ Successfully loaded {dataset_name}")
                break
            except StopIteration:
                continue
            except Exception as e:
                print(f"  Error reading from {alt[0]}: {e}")
                continue
        except Exception as e:
            print(f"  Failed to load {alt[0]}: {e}")
            continue
    
    if dataset is None:
        print("✗ No working Reddit/conversational dataset found")
        print("Note: Skipping this dataset - not critical for training.")
        return False
    
    try:
        count = 0
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in dataset:
                if count % 10000 == 0:
                    print(f"  Processed {count:,} samples...", end='\r')
                
                # Try different text fields based on dataset
                text = None
                if dataset_name and 'reddit' in dataset_name.lower():
                    # Reddit-specific fields
                    text = item.get('content', item.get('text', item.get('body', 
                                item.get('selftext', item.get('title', '')))))
                else:
                    # Generic fields
                    text = item.get('text', item.get('content', ''))
                
                if text:
                    cleaned = cleaner.clean(text)
                    if cleaned:
                        f.write(cleaned + '\n')
                        count += 1
                
                if count >= max_samples:
                    break
        
        print(f"\n✓ Reddit dataset downloaded: {count:,} samples")
        return True
    except Exception as e:
        print(f"✗ Download failed: {e}")
        safe_delete_file(output_file)
        return False


def download_1billion_words(output_file: str):
    """Download 1 Billion Word Benchmark"""
    print("\n" + "="*60)
    print("4/4: Downloading 1 Billion Word Benchmark (~1.5 GB)...")
    print("="*60)
    
    # Check if file already exists and is valid
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        print(f"✓ File already exists: {os.path.basename(output_file)}")
        return True
    
    import urllib.request
    import tarfile
    
    url = "http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz"
    temp_dir = Path("data/unsupervised/1billion")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    tar_file = temp_dir / "1-billion-word-language-modeling-benchmark-r13output.tar.gz"
    
    try:
        # Check if archive exists and is valid
        if tar_file.exists():
            # Try to verify it's not corrupted
            try:
                with tarfile.open(tar_file, 'r:gz') as tar:
                    tar.getmembers()  # Try to read members
                print("✓ Archive already exists and appears valid")
            except Exception as e:
                print(f"  Archive appears corrupted: {e}")
                print("  Re-downloading...")
                tar_file.unlink()
        
        if not tar_file.exists():
            print("Downloading 1 Billion Word Benchmark...")
            print("This may take 10-15 minutes...")
            
            # Handle redirects and show progress
            request = urllib.request.Request(url)
            request.add_header('User-Agent', 'Mozilla/5.0')
            
            with urllib.request.urlopen(request, timeout=300) as response:
                total_size = int(response.headers.get('Content-Length', 0))
                downloaded = 0
                chunk_size = 8192
                
                with open(tar_file, 'wb') as out_file:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        out_file.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"  Progress: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='\r')
            
            print(f"\n✓ Download complete")
        
        extracted_dir = temp_dir / "training-monolingual.tokenized.shuffled"
        
        if not extracted_dir.exists():
            print("Extracting archive (this may take a few minutes)...")
            try:
                with tarfile.open(tar_file, 'r:gz') as tar:
                    tar.extractall(temp_dir)
                print("✓ Extraction complete")
            except Exception as e:
                print(f"✗ Extraction failed: {e}")
                # Delete corrupted archive
                if tar_file.exists():
                    tar_file.unlink()
                return False
        
        if extracted_dir.exists():
            print("Processing files (first 10 files, 1M lines)...")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            cleaner = UniversalTextCleaner()
            
            count = 0
            with open(output_file, 'w', encoding='utf-8') as outfile:
                # Process first 10 files
                for i in range(10):
                    file_path = extracted_dir / f"news.en-{i:04d}-of-00100"
                    if file_path.exists():
                        print(f"  Processing file {i+1}/10...", end='\r')
                        try:
                            with open(file_path, 'r', encoding='utf-8') as infile:
                                for line in infile:
                                    cleaned = cleaner.clean(line.strip())
                                    if cleaned:
                                        outfile.write(cleaned + '\n')
                                        count += 1
                                        if count >= 1000000:
                                            break
                        except Exception as e:
                            print(f"\n  Error reading file {i+1}: {e}")
                            continue
                        if count >= 1000000:
                            break
            
            print(f"\n✓ 1 Billion Word processed: {count:,} lines")
            return True
        else:
            print("✗ Extracted directory not found")
            return False
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.")
        safe_delete_file(output_file)
        if tar_file.exists():
            tar_file.unlink()
        return False
    except Exception as e:
        print(f"✗ 1 Billion Word download failed: {e}")
        safe_delete_file(output_file)
        # Delete corrupted archive if extraction failed
        if tar_file.exists() and not extracted_dir.exists():
            try:
                tar_file.unlink()
            except:
                pass
        return False


def main():
    """Download all unsupervised datasets"""
    # Create main download directory
    base_dir = Path("data/processed/unsupervised")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    datasets = [
        ("wikitext103.txt", download_wikitext),
        ("bookcorpus.txt", download_bookcorpus),  # Full dataset, no sample
        ("reddit.txt", lambda f: download_reddit(f, 200000)),
        ("1billion_words.txt", download_1billion_words),
    ]
    
    print("="*60)
    print("UNSUPERVISED DATASET DOWNLOAD")
    print("="*60)
    print(f"Download directory: {base_dir}")
    print("Total size: ~4 GB compressed, ~11 GB uncompressed")
    print("="*60)
    
    downloaded = []
    failed = []
    
    for filename, download_func in datasets:
        output_file = str(base_dir / filename)
        try:
            if download_func(output_file):
                downloaded.append(filename)
            else:
                failed.append(filename)
        except KeyboardInterrupt:
            print("\n\nDownload interrupted by user.")
            # Clean up partial file
            safe_delete_file(output_file)
            break
        except Exception as e:
            print(f"\n✗ Error downloading {filename}: {e}")
            failed.append(filename)
            # Clean up partial file
            safe_delete_file(output_file)
    
    # Summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    print(f"Successfully downloaded: {len(downloaded)}/{len(datasets)}")
    for f in downloaded:
        filepath = base_dir / f
        size_mb = filepath.stat().st_size / 1024 / 1024 if filepath.exists() else 0
        print(f"  ✓ {f} ({size_mb:.1f} MB)")
    
    if failed:
        print(f"\nFailed or skipped: {len(failed)}")
        for f in failed:
            print(f"  ✗ {f}")
    
    if downloaded:
        print("\n" + "="*60)
        print("NEXT STEPS")
        print("="*60)
        print("\n1. Merge datasets:")
        print(f"   python scripts/data_preparation/merge_corpora.py \\")
        print(f"       --new-files {base_dir}/*.txt \\")
        print(f"       --output data/processed/unsupervised/combined.txt")
        print("\n2. Train models:")
        print("   python scripts/train_ngram_model.py \\")
        print("       --data data/processed/unsupervised/combined.txt \\")
        print("       --output models/ngram \\")
        print("       --use-dictionary")
        print("\n3. Evaluate:")
        print("   python scripts/evaluate_ngram.py")


if __name__ == "__main__":
    main()

