import argparse
from pathlib import Path

from transformers import T5Tokenizer

from spellchecker.data.datasets.csv_dataset import CSVDataset
from spellchecker.models.train_t5.args import Seq2SeqTrainingConfig
from spellchecker.models.train_t5.trainer import T5Seq2SeqTrainer


def parse_args() -> tuple[str, Seq2SeqTrainingConfig]:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv_folder",
        type=str,
        default="./data/csvs",
        help="Path to folder containing CSV files.",
    )
    parser.add_argument(
        "--model_name", type=str, default="t5-small", help="Hugging Face model name."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save model checkpoints.",
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=3, help="Number of training epochs."
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Per-device train batch size."
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=8, help="Per-device eval batch size."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-4, help="Learning rate."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay."
    )
    parser.add_argument(
        "--fp16", action="store_true", help="Use mixed precision (fp16) training."
    )

    args = parser.parse_args()

    csv_folder = Path(args.csv_folder)
    if not csv_folder.exists():
        raise FileNotFoundError(f"CSV folder not found: {csv_folder}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = Seq2SeqTrainingConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
    )

    return str(csv_folder), config


def main():
    csv_folder, config = parse_args()

    # Prepare dataset
    dataset_obj = CSVDataset(
        csv_folder=csv_folder, input_column="source_text", target_column="target_text"
    )
    dataset_obj.save_to_csv()

    # Load tokenized Hugging Face DatasetDict
    tokenizer = T5Tokenizer.from_pretrained(config.model_name)
    hf_dataset = dataset_obj.to_hf_dataset(tokenizer=tokenizer)

    # Configure and initialize trainer
    trainer = T5Seq2SeqTrainer(config)
    trainer.setup(hf_dataset)

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()

# python -m spellchecker.models.train_t5.train --csv_folder ../data/training --model_name /root/t5-small --num_train_epochs 5 --train_batch_size 16 --fp16
