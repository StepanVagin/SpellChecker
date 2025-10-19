import argparse
import typing as tp

import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer


def load_model_and_tokenizer(model_dir: str):
    """
    Load trained model and tokenizer from directory.
    """
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
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
    Generate predictions for a list of texts.
    """
    model.to(device)
    predictions = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        outputs = model.generate(**inputs, max_length=max_length)
        batch_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(batch_preds)

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

    model, tokenizer = load_model_and_tokenizer(args.model_dir)

    df = pd.read_csv(args.input_csv)
    if args.text_column not in df.columns:
        raise ValueError(f"Column '{args.text_column}' not found in CSV.")

    texts = df[args.text_column].tolist()
    preds = predict(
        model,
        tokenizer,
        texts,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=args.device,
    )

    df["prediction"] = preds
    df.to_csv(args.output_csv, index=False)
    print(f"Predictions saved to {args.output_csv}")


if __name__ == "__main__":
    main()
