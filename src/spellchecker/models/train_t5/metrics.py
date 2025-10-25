import argparse
import pandas as pd
import typing as tp
import os


def compute_token_level_sets(
    source: str, target: str, prediction: str
) -> tp.Dict[str, tp.Set[int]]:
    source_tokens = source.split()
    target_tokens = target.split()
    pred_tokens = prediction.split()

    max_len = max(len(source_tokens), len(target_tokens), len(pred_tokens))
    source_tokens += [""] * (max_len - len(source_tokens))
    target_tokens += [""] * (max_len - len(target_tokens))
    pred_tokens += [""] * (max_len - len(pred_tokens))

    real_errors = {
        i for i, (s, t) in enumerate(zip(source_tokens, target_tokens)) if s != t
    }
    model_changes = {
        i for i, (s, p) in enumerate(zip(source_tokens, pred_tokens)) if s != p
    }
    true_positives = {
        i
        for i in real_errors
        if i in model_changes and pred_tokens[i] == target_tokens[i]
    }

    return {
        "real_errors": real_errors,
        "model_changes": model_changes,
        "true_positives": true_positives,
    }


def compute_metrics_for_row(
    source: str, target: str, prediction: str
) -> tp.Dict[str, float]:
    """
    Computes EM, Precision, Recall and F1 score.
    """
    em = float(prediction.strip() == target.strip())
    sets = compute_token_level_sets(source, target, prediction)

    TP = len(sets["true_positives"])
    FP = len(sets["model_changes"] - sets["real_errors"])
    FN = len(sets["real_errors"] - sets["model_changes"])

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {"EM": em, "Precision": precision, "Recall": recall, "F1": f1}


def evaluate_metrics(df: pd.DataFrame) -> tp.Dict[str, float]:
    metrics_list = []
    for _, row in df.iterrows():
        source, target, pred = row["source_text"], row["target_text"], row["prediction"]
        if not isinstance(target, str):
            continue
        metrics = compute_metrics_for_row(source, target, pred)
        metrics_list.append(metrics)

    metrics_df = pd.DataFrame(metrics_list)
    return metrics_df.mean().to_dict()


def main():
    parser = argparse.ArgumentParser(
        description="Compute Spelling Correction metrics (EM, Precision, Recall, F1)."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="CSV file with source_text, target_text, and prediction columns.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=False,
        default="metrics.csv",
        help="Path to save computed metrics.",
    )

    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input_csv)
    required_cols = {"source_text", "target_text", "prediction"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    # Compute metrics
    results = evaluate_metrics(df)

    # Print to console
    print("\n===== Evaluation Results =====")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
    print("================================\n")

    # Save to CSV
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    results_df = pd.DataFrame([results])
    results_df.to_csv(args.output_csv, index=False)
    print(f"[INFO] Metrics saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
