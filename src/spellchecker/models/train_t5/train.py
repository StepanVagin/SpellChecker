import argparse
from dataclasses import fields
from pathlib import Path

from transformers import T5Tokenizer

from spellchecker.data.datasets.csv_dataset import CSVDataset
from spellchecker.models.train_t5.args import Seq2SeqTrainingConfig
from spellchecker.models.train_t5.trainer import T5Seq2SeqTrainer


def parse_args() -> tuple[str, Seq2SeqTrainingConfig]:
    parser = argparse.ArgumentParser(
        description="Train a T5 seq2seq model on CSV datasets."
    )

    # CSV folder
    parser.add_argument(
        "--csv_folder",
        type=str,
        default="./data/csvs",
        help="Path to folder containing CSV files.",
    )

    # Main model params
    parser.add_argument(
        "--model_name", type=str, default="t5-small", help="Model name."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save model checkpoints.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--predict_with_generate", action="store_true")
    parser.add_argument("--fp16", action="store_true")

    # Logging & checkpoints
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--load_best_model_at_end", action="store_true")
    parser.add_argument("--metric_for_best_model", type=str, default="loss")
    parser.add_argument("--greater_is_better", action="store_true")

    # Misc
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--remove_unused_columns", action="store_true")
    parser.add_argument("--report_to", type=str, default="all")

    args = parser.parse_args()

    csv_folder = Path(args.csv_folder)
    if not csv_folder.exists():
        raise FileNotFoundError(f"CSV folder not found: {csv_folder}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataclass_fields = {f.name for f in fields(Seq2SeqTrainingConfig)}
    config_kwargs = {k: v for k, v in vars(args).items() if k in dataclass_fields}

    config = Seq2SeqTrainingConfig(**config_kwargs)

    return str(csv_folder), config


def main():
    csv_folder, config = parse_args()

    print("[INFO] CSV folder:", csv_folder)
    print("[INFO] Training config:")
    for f in fields(config):
        print(f"  {f.name}: {getattr(config, f.name)}")
    print("[INFO] Training will use TensorBoard logging at:", config.output_dir)

    # Prepare dataset
    print("[INFO] Preparing dataset...")
    dataset_obj = CSVDataset(
        csv_folder=csv_folder, input_column="source_text", target_column="target_text"
    )
    dataset_obj.save_to_csv()

    print("[INFO] Tokenizing dataset...")
    tokenizer = T5Tokenizer.from_pretrained(config.model_name)
    hf_dataset = dataset_obj.to_hf_dataset(tokenizer=tokenizer)

    print("[INFO] Initializing trainer...")
    trainer = T5Seq2SeqTrainer(config)
    trainer.setup(hf_dataset)

    print("[INFO] Starting training...")
    trainer.train()
    print("[INFO] Training finished! Checkpoints and logs saved in:", config.output_dir)


if __name__ == "__main__":
    main()

# python -m spellchecker.models.train_t5.train --csv_folder ../data/training --model_name /root/t5-small --num_train_epochs 3 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --fp16 --learning_rate 3e-4 --weight_decay 1e-3 --logging_steps 50 --save_steps 500 --eval_steps 50 --seed 42
