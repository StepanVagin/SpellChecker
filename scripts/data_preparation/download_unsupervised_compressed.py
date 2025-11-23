#!/usr/bin/env python3
"""
Download unsupervised datasets with COMPRESSED storage.
Saves ~70-80% disk space compared to uncompressed version.

Usage:
    python scripts/data_preparation/download_unsupervised_compressed.py
"""

import os
import sys
import gzip
import pickle
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


def save_compressed(texts: list, output_file: str):
    """Save list of texts as compressed pickle file"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with gzip.open(output_file, 'wb') as f:
        pickle.dump(texts, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✓ Saved compressed: {output_file} ({os.path.getsize(output_file) / 1024 / 1024:.1f} MB)")


def load_compressed(input_file: str) -> list:
    """Load compressed pickle file"""
    with gzip.open(input_file, 'rb') as f:
        texts = pickle.load(f)
    return texts


def download_wikitext_compressed(output_file: str):
    """Download WikiText-103 in compressed format"""
    print("\n" + "="*60)
    print("1/5: Downloading WikiText-103 (compressed)...")
    print("="*60)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    cleaner = UniversalTextCleaner()
    
    try:
        print("Downloading from Hugging Face...")
        dataset = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True)
        
        texts = []
        count = 0
        for item in dataset:
            if count % 10000 == 0:
                print(f"  Processed {count:,} lines...", end='\r')
            
            text = item.get('text', '')
            if text:
                cleaned = cleaner.clean(text)
                if cleaned:
                    texts.append(cleaned)
                    count += 1
        
        # Save compressed
        save_compressed(texts, output_file)
        print(f"\n✓ WikiText-103: {count:,} lines")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def download_bookcorpus_compressed(output_file: str, max_samples: int = 500000):
    """Download BookCorpus in compressed format"""
    print("\n" + "="*60)
    print(f"2/5: Downloading BookCorpus (compressed, max {max_samples:,})...")
    print("="*60)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    cleaner = UniversalTextCleaner()
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("HF_HUB_TIMEOUT", "120")
    
    try:
        print("Downloading from rojagtap/bookcorpus...")
        dataset = load_dataset("rojagtap/bookcorpus", split="train", streaming=True)
        
        texts = []
        count = 0
        for item in dataset:
            if count % 10000 == 0:
                print(f"  Processed {count:,} samples...", end='\r')
            
            text = item.get('text', item.get('content', ''))
            if text:
                cleaned = cleaner.clean(text)
                if cleaned:
                    texts.append(cleaned)
                    count += 1
            
            if count >= max_samples:
                break
        
        # Save compressed
        save_compressed(texts, output_file)
        print(f"\n✓ BookCorpus: {count:,} samples")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def download_webtext_compressed(output_file: str, max_samples: int = 1000000):
    """Download web text (OpenWebText/C4) in compressed format"""
    print("\n" + "="*60)
    print(f"3/5: Downloading web text (compressed, max {max_samples:,})...")
    print("="*60)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    cleaner = UniversalTextCleaner()
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("HF_HUB_TIMEOUT", "120")
    
    try:
        # Try OpenWebText, fallback to C4
        try:
            dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
        except:
            dataset = load_dataset("c4", "en", split="train", streaming=True)
        
        texts = []
        count = 0
        for item in dataset:
            if count % 10000 == 0:
                print(f"  Processed {count:,} samples...", end='\r')
            
            text = item.get('text', item.get('content', ''))
            if text:
                cleaned = cleaner.clean(text)
                if cleaned:
                    texts.append(cleaned)
                    count += 1
            
            if count >= max_samples:
                break
        
        # Save compressed
        save_compressed(texts, output_file)
        print(f"\n✓ Web text: {count:,} samples")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def download_1billion_compressed(output_file: str):
    """Download 1 Billion Word in compressed format"""
    print("\n" + "="*60)
    print("4/5: Downloading 1 Billion Word (compressed)...")
    print("="*60)
    
    import urllib.request
    import tarfile
    
    url = "http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz"
    temp_dir = Path("data/unsupervised/1billion")
    temp_dir.mkdir(parents=True, exist_ok=True)
    tar_file = temp_dir / "1-billion-word-language-modeling-benchmark-r13output.tar.gz"
    
    try:
        if not tar_file.exists():
            print("Downloading archive...")
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
                            print(f"  Progress: {percent:.1f}%", end='\r')
            
            print(f"\n✓ Download complete")
        
        extracted_dir = temp_dir / "training-monolingual.tokenized.shuffled"
        
        if not extracted_dir.exists():
            print("Extracting...")
            with tarfile.open(tar_file, 'r:gz') as tar:
                tar.extractall(temp_dir)
        
        if extracted_dir.exists():
            print("Processing and compressing...")
            cleaner = UniversalTextCleaner()
            texts = []
            count = 0
            
            for i in range(10):
                file_path = extracted_dir / f"news.en-{i:04d}-of-00100"
                if file_path.exists():
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        for line in infile:
                            cleaned = cleaner.clean(line.strip())
                            if cleaned:
                                texts.append(cleaned)
                                count += 1
                                if count >= 1000000:
                                    break
                    if count >= 1000000:
                        break
            
            # Save compressed
            save_compressed(texts, output_file)
            print(f"\n✓ 1 Billion Word: {count:,} lines")
            return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def main():
    """Download all unsupervised datasets in COMPRESSED format"""
    base_dir = Path("data/processed/unsupervised")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("UNSUPERVISED DATASET DOWNLOAD (COMPRESSED)")
    print("="*60)
    print("Storage format: Compressed pickle (.pkl.gz)")
    print("Estimated total: ~4-5 GB (vs ~11 GB uncompressed)")
    print("(OpenWebText excluded to reduce size)")
    print("="*60)
    
    datasets = [
        ("wikitext103.pkl.gz", download_wikitext_compressed),
        ("bookcorpus_sample.pkl.gz", lambda f: download_bookcorpus_compressed(f, 500000)),
        # OpenWebText removed - too large (~7 GB compressed, ~40 GB uncompressed)
        # ("openwebtext_sample.pkl.gz", lambda f: download_webtext_compressed(f, 1000000)),
        ("1billion_words.pkl.gz", download_1billion_compressed),
    ]
    
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
            break
        except Exception as e:
            print(f"\n✗ Error downloading {filename}: {e}")
            failed.append(filename)
    
    # Summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    print(f"Successfully downloaded: {len(downloaded)}/{len(datasets)}")
    for f in downloaded:
        size_mb = os.path.getsize(base_dir / f) / 1024 / 1024
        print(f"  ✓ {f} ({size_mb:.1f} MB)")
    
    if failed:
        print(f"\nFailed: {len(failed)}")
        for f in failed:
            print(f"  ✗ {f}")
    
    if downloaded:
        print("\n" + "="*60)
        print("USING COMPRESSED DATA")
        print("="*60)
        print("\nTo use compressed data for training, create a script that:")
        print("  1. Loads compressed files")
        print("  2. Decompresses them")
        print("  3. Writes to temporary .txt files")
        print("  4. Trains on those files")
        print("\nOr use the helper script:")
        print("  python scripts/data_preparation/decompress_for_training.py")


if __name__ == "__main__":
    main()

