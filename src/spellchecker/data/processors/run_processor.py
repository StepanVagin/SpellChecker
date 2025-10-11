import argparse
from pathlib import Path
import typing as tp

from spellchecker.data.processors.validation_processor import Validator
from spellchecker.data.processors.corruption_processor import Corruptor

PROCESSORS = {
    "validate": Validator,
    "corrupt": Corruptor,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run LLM processor on dataset")

    parser.add_argument(
        "processor", choices=PROCESSORS.keys(), help="Type of processor to run"
    )

    parser.add_argument(
        "input", type=Path, help="Input CSV file or directory with CSV files"
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output file path (default: input_processed.csv)",
    )

    parser.add_argument(
        "--prompt", "-p", type=Path, help="Path to prompt template file"
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="gemma-3-27b-it/latest",
        help="Model name",
    )

    parser.add_argument(
        "--batch-size", "-b", type=int, default=64, help="Batch size for processing"
    )

    parser.add_argument(
        "--checkpoint-dir",
        "-c",
        type=Path,
        default=Path("checkpoints"),
        help="Directory for checkpoints",
    )

    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=0.0,
        help="Temperature for generation",
    )

    parser.add_argument(
        "--max-tokens", type=int, default=500, help="Max tokens for generation"
    )

    return parser.parse_args()


def get_input_files(input_path: Path) -> tp.List[Path]:
    """Get list of CSV files from input path."""
    if input_path.is_file():
        return [input_path]
    elif input_path.is_dir():
        return list(input_path.glob("*.csv"))
    else:
        raise ValueError(f"Input path {input_path} does not exist")


def main():
    args = parse_args()

    # Get processor class
    processor_class = PROCESSORS[args.processor]

    # Build processor kwargs
    processor_kwargs = {
        "batch_size": args.batch_size,
        "checkpoint_dir": args.checkpoint_dir,
    }

    if args.prompt:
        processor_kwargs["prompt_path"] = args.prompt

    if args.model:
        processor_kwargs["model_name"] = args.model

    if args.temperature is not None:
        processor_kwargs["temperature"] = args.temperature

    if args.max_tokens is not None:
        processor_kwargs["max_tokens"] = args.max_tokens

    # Initialize processor
    processor = processor_class(**processor_kwargs)

    # Get input files
    input_files = get_input_files(args.input)
    print(f"Processing {len(input_files)} file(s)")

    # Process each file
    for input_file in input_files:
        print(f"\nProcessing: {input_file}")

        # Process dataframe
        result = processor.process_dataframe(input_file)

        # Validation processor
        if args.processor == "validate":
            valid_count = result["is_valid"].sum()
            result = result[result["is_valid"]]
            print(f"Filtered: {valid_count}/{len(result)} valid pairs")

        # Save output
        if args.output:
            output_path = args.output
        else:
            output_path = input_file.parent / f"{input_file.stem}_{args.processor}d.csv"

        result.to_csv(output_path, index=False)
        print(f"Saved to: {output_path}")

    print("\n✅ Processing completed!")


if __name__ == "__main__":
    main()

# python -m spellchecker.data.processors.run_processor validate path_to_unfiltered_sft_corpus.csv -o path_to_save.csv --model model_name
