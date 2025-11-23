import argparse
import typing as tp

import pandas as pd
import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer


def load_model_and_tokenizer(model_dir: str):
    """
    Load trained model and tokenizer from directory.
    """
    print(f"[INFO] Loading model and tokenizer from: {model_dir}")
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    print(f"[INFO] Model and tokenizer successfully loaded.")
    return model, tokenizer


def predict(
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    texts: tp.List[str],
    max_length: int = 128,
    batch_size: int = 8,
    device: str = "cpu",
):
    """
    Generate predictions for a list of texts with progress bar.
    """
    model.to(device)
    model.eval()
    predictions = []

    print(
        f"[INFO] Starting inference on {len(texts)} samples "
        f"(batch_size={batch_size}, device={device})"
    )

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="[INFO] Inference progress"):
            batch_texts = texts[i : i + batch_size]
            inputs = tokenizer(
                batch_texts, return_tensors="pt", padding=True, truncation=True
            ).to(device)
            outputs = model.generate(**inputs, max_length=max_length)
            batch_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions.extend(batch_preds)

    print(f"[INFO] Inference completed.")
    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="Run inference using a trained T5 model."
    )
    parser.add_argument(
        "--model_dir", type=str, required=True, help="Directory of the trained model."
    )
    parser.add_argument(
        "--input_csv", type=str, required=True, help="CSV file with input texts."
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Name of the text column in CSV.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="predictions.csv",
        help="CSV file to save predictions.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for inference."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum sequence length for generation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run inference on (cpu or cuda).",
    )

    args = parser.parse_args()

    # Load model & tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_dir)

    # Load data
    print(f"[INFO] Loading input CSV: {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    if args.text_column not in df.columns:
        raise ValueError(f"Column '{args.text_column}' not found in CSV.")
    print(f"[INFO] Loaded {len(df)} rows from {args.input_csv}")

    # Run predictions
    texts = df[args.text_column].tolist()
    preds = predict(
        model,
        tokenizer,
        texts,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=args.device,
    )

    # Save results
    df["prediction"] = preds
    df.to_csv(args.output_csv, index=False)
    print(f"[INFO] Predictions saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
