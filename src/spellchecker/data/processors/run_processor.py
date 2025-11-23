import argparse
import typing as tp
from pathlib import Path

from spellchecker.data.processors.corruption_processor import Corruptor
from spellchecker.data.processors.validation_processor import Validator
from spellchecker.data.processors.heuristic_processor import HeuristicValidator

PROCESSORS = {
    "validate": Validator,
    "corrupt": Corruptor,
    "heuristic_validate": HeuristicValidator,
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
        help="Model name",
    )

    parser.add_argument(
        "--batch-size", "-b", type=int, help="Batch size for processing"
    )

    parser.add_argument(
        "--checkpoint-dir",
        "-c",
        type=Path,
        help="Directory for checkpoints",
    )

    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        help="Temperature for generation",
    )

    parser.add_argument("--max-tokens", type=int, help="Max tokens for generation")

    # For corruption processor
    parser.add_argument("--corruption-mode", type=str, help="Either llm or heuristic")

    # For heuristic processor
    parser.add_argument(
        "--max-edit-ratio",
        type=float,
        help="Maximum allowed edit distance ratio",
    )
    parser.add_argument(
        "--min-word-similarity",
        type=float,
        help="Minimum word-level similarity",
    )
    parser.add_argument(
        "--max-word-diff",
        type=int,
        help="Maximum difference in word count between source and target",
    )
    parser.add_argument(
        "--allow-case-normalization",
        type=bool,
        help="Allow case normalization as valid change",
    )
    parser.add_argument(
        "--invalid-chars-pattern",
        type=str,
        help="Regex pattern for invalid characters",
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

    processor_kwargs = {}

    if args.batch_size is not None:
        processor_kwargs["batch_size"] = args.batch_size

    if args.checkpoint_dir is not None:
        processor_kwargs["checkpoint_dir"] = args.checkpoint_dir

    if args.prompt:
        processor_kwargs["prompt_path"] = args.prompt

    if args.model:
        processor_kwargs["model_name"] = args.model

    if args.temperature is not None:
        processor_kwargs["temperature"] = args.temperature

    if args.max_tokens is not None:
        processor_kwargs["max_tokens"] = args.max_tokens

    if args.processor == "corrupt" and hasattr(args, "corruption_mode"):
        processor_kwargs["corruption_mode"] = args.corruption_mode

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
            output_path = input_file.parent / f"{input_file.stem}_{args.processor}.csv"

        result.to_csv(output_path, index=False)
        print(f"Saved to: {output_path}")

    print("\n✅ Processing completed!")


if __name__ == "__main__":
    main()

# python -m spellchecker.data.processors.run_processor validate path_to_unfiltered_sft_corpus.csv -o path_to_save.csv --model model_name
# python -m spellchecker.data.processors.run_processor heuristic_validate /Users/chrnegor/Documents/study/MLDL/SpellChecker/temp/train.csv \
#     -o filtered_train.csv \
