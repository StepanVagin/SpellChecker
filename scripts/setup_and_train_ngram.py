#!/usr/bin/env python3
"""
Master script to download, process, and train n-gram models.

This script automates the entire pipeline:
1. Downloads unsupervised datasets (compressed format)
2. Decompresses the data
3. Trains n-gram models (1-gram, 2-gram, 3-gram)

Usage:
    # Full pipeline with default settings
    python scripts/setup_and_train_ngram.py

    # Custom output directory
    python scripts/setup_and_train_ngram.py --output models/ngram_custom

    # Skip download if data already exists
    python scripts/setup_and_train_ngram.py --skip-download

    # Skip decompression if already done
    python scripts/setup_and_train_ngram.py --skip-decompress

    # Limit training data size
    python scripts/setup_and_train_ngram.py --max-lines 5000000

    # Use dictionary augmentation
    python scripts/setup_and_train_ngram.py --use-dictionary
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
from glob import glob

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def check_dependencies():
    """Check if required dependencies are installed"""
    print("\n" + "="*60)
    print("Checking Dependencies")
    print("="*60)
    
    required_packages = {
        "datasets": "pip install datasets",
        "nltk": "pip install nltk",
    }
    
    missing = []
    for package, install_cmd in required_packages.items():
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (missing)")
            missing.append((package, install_cmd))
    
    if missing:
        print("\nMissing dependencies. Install with:")
        for package, install_cmd in missing:
            print(f"  {install_cmd}")
        response = input("\nInstall missing dependencies now? (y/n): ")
        if response.lower() == 'y':
            for package, install_cmd in missing:
                print(f"\nInstalling {package}...")
                subprocess.run(install_cmd.split(), check=True)
        else:
            print("\nPlease install missing dependencies and run again.")
            sys.exit(1)
    
    print("\n✓ All dependencies available")


def download_data(data_dir: Path, skip: bool = False):
    """Download compressed unsupervised datasets"""
    if skip:
        print("\n" + "="*60)
        print("Skipping Download (--skip-download flag)")
        print("="*60)
        return True
    
    print("\n" + "="*60)
    print("Step 1: Downloading Unsupervised Datasets (Compressed)")
    print("="*60)
    
    download_script = Path(__file__).parent / "data_preparation" / "download_unsupervised_compressed.py"
    
    if not download_script.exists():
        print(f"✗ Error: Download script not found at {download_script}")
        return False
    
    print(f"Running: {download_script}")
    print()
    
    try:
        result = subprocess.run(
            [sys.executable, str(download_script)],
            cwd=Path(__file__).parent.parent,
            check=True
        )
        print("\n✓ Download complete")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Download failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user")
        return False


def decompress_data(data_dir: Path, training_dir: Path, skip: bool = False):
    """Decompress downloaded data for training"""
    if skip:
        print("\n" + "="*60)
        print("Skipping Decompression (--skip-decompress flag)")
        print("="*60)
        return True
    
    print("\n" + "="*60)
    print("Step 2: Decompressing Data for Training")
    print("="*60)
    
    # Check if compressed files exist
    compressed_files = glob(str(data_dir / "*.pkl.gz"))
    
    if not compressed_files:
        print(f"✗ No compressed files found in {data_dir}")
        print("  Run download step first or use --skip-decompress if data is already decompressed")
        return False
    
    print(f"Found {len(compressed_files)} compressed file(s):")
    for f in compressed_files:
        size_mb = os.path.getsize(f) / 1024 / 1024
        print(f"  - {Path(f).name} ({size_mb:.1f} MB)")
    
    decompress_script = Path(__file__).parent / "data_preparation" / "decompress_for_training.py"
    
    if not decompress_script.exists():
        print(f"✗ Error: Decompress script not found at {decompress_script}")
        return False
    
    print(f"\nRunning: {decompress_script}")
    print(f"Output directory: {training_dir}")
    print()
    
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(decompress_script),
                "--input",
                str(data_dir / "*.pkl.gz"),
                "--output",
                str(training_dir)
            ],
            cwd=Path(__file__).parent.parent,
            check=True
        )
        print("\n✓ Decompression complete")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Decompression failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n\nDecompression interrupted by user")
        return False


def train_models(training_dir: Path, output_dir: Path, max_lines: int, use_dictionary: bool):
    """Train n-gram models"""
    print("\n" + "="*60)
    print("Step 3: Training N-gram Models")
    print("="*60)
    
    # Check if training files exist
    training_files = glob(str(training_dir / "*.txt"))
    
    if not training_files:
        print(f"✗ No training files found in {training_dir}")
        print("  Run decompression step first")
        return False
    
    print(f"Found {len(training_files)} training file(s):")
    for f in training_files:
        size_mb = os.path.getsize(f) / 1024 / 1024
        print(f"  - {Path(f).name} ({size_mb:.1f} MB)")
    
    train_script = Path(__file__).parent / "train_ngram_model.py"
    
    if not train_script.exists():
        print(f"✗ Error: Training script not found at {train_script}")
        return False
    
    print(f"\nRunning: {train_script}")
    print(f"Output directory: {output_dir}")
    print(f"Max lines: {max_lines:,}")
    print(f"Use dictionary: {use_dictionary}")
    print()
    
    cmd = [
        sys.executable,
        str(train_script),
        "--data",
        str(training_dir / "*.txt"),
        "--output",
        str(output_dir),
        "--max-lines",
        str(max_lines),
    ]
    
    if use_dictionary:
        cmd.append("--use-dictionary")
    
    cmd.append("--test")  # Run test examples after training
    
    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent,
            check=True
        )
        print("\n✓ Training complete")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Training failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Download, process, and train n-gram models for spelling correction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with defaults
  python scripts/setup_and_train_ngram.py

  # Custom output directory
  python scripts/setup_and_train_ngram.py --output models/ngram_custom

  # Skip download if already done
  python scripts/setup_and_train_ngram.py --skip-download

  # Limit training data
  python scripts/setup_and_train_ngram.py --max-lines 5000000
        """
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="models/ngram",
        help="Directory to save trained models (default: models/ngram)"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed/unsupervised",
        help="Directory for downloaded compressed data (default: data/processed/unsupervised)"
    )
    
    parser.add_argument(
        "--training-dir",
        type=str,
        default="data/processed/unsupervised/training",
        help="Directory for decompressed training data (default: data/processed/unsupervised/training)"
    )
    
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download step (use if data already downloaded)"
    )
    
    parser.add_argument(
        "--skip-decompress",
        action="store_true",
        help="Skip decompression step (use if data already decompressed)"
    )
    
    parser.add_argument(
        "--max-lines",
        type=int,
        default=10000000,
        help="Maximum number of lines to use for training (default: 10,000,000)"
    )
    
    parser.add_argument(
        "--use-dictionary",
        action="store_true",
        help="Include English dictionary words in training"
    )
    
    parser.add_argument(
        "--skip-deps-check",
        action="store_true",
        help="Skip dependency check"
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    project_root = Path(__file__).parent.parent
    data_dir = project_root / args.data_dir
    training_dir = project_root / args.training_dir
    output_dir = project_root / args.output
    
    # Create directories
    data_dir.mkdir(parents=True, exist_ok=True)
    training_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("N-gram Model Setup and Training Pipeline")
    print("="*60)
    print(f"Project root: {project_root}")
    print(f"Data directory: {data_dir}")
    print(f"Training directory: {training_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Max training lines: {args.max_lines:,}")
    print(f"Use dictionary: {args.use_dictionary}")
    print("="*60)
    
    # Check dependencies
    if not args.skip_deps_check:
        check_dependencies()
    
    # Step 1: Download
    if not download_data(data_dir, skip=args.skip_download):
        print("\n✗ Pipeline failed at download step")
        sys.exit(1)
    
    # Step 2: Decompress
    if not decompress_data(data_dir, training_dir, skip=args.skip_decompress):
        print("\n✗ Pipeline failed at decompression step")
        sys.exit(1)
    
    # Step 3: Train
    if not train_models(training_dir, output_dir, args.max_lines, args.use_dictionary):
        print("\n✗ Pipeline failed at training step")
        sys.exit(1)
    
    # Success summary
    print("\n" + "="*60)
    print("✅ Pipeline Complete!")
    print("="*60)
    print(f"\nTrained models saved to: {output_dir}/")
    print("  - 1gram_model.json")
    print("  - 2gram_model.json")
    print("  - 3gram_model.json")
    print("\nYou can now use these models:")
    print(f"  - Web interface: python app.py --models {output_dir}")
    print(f"  - Deploy API: cd deploy && NGRAM_ONLY=true uvicorn app:app --reload")
    print()


if __name__ == "__main__":
    main()

