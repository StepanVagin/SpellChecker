import subprocess
import uuid
from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig


def get_latest_checkpoint(run_dir: Path) -> Path:
    checkpoints = list(run_dir.glob("checkpoint-*"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {run_dir}")
    # sort by numeric suffix
    latest_ckpt = max(checkpoints, key=lambda x: int(x.name.split("-")[-1]))
    return latest_ckpt

@hydra.main(config_path="../conf", config_name="t5_train_cfg", version_base=None)
def main(cfg: DictConfig):
    # Create unique directory for an experiment
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    unique_id = str(uuid.uuid4())[:8]
    run_dir = Path(cfg.paths.output_dir) / f"{timestamp}_{unique_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Hydra Pipeline] Run directory: {run_dir}")

    # Train the model
    train_cmd = [
        "python",
        "-m",
        "spellchecker.models.train_t5.train",
        f"--csv_folder={cfg.paths.csv_folder}",
        f"--model_name={cfg.paths.model_name}",
        f"--output_dir={run_dir}",
        f"--num_train_epochs={cfg.train.num_train_epochs}",
        f"--per_device_train_batch_size={cfg.train.per_device_train_batch_size}",
        f"--per_device_eval_batch_size={cfg.train.per_device_eval_batch_size}",
        f"--learning_rate={cfg.train.learning_rate}",
        f"--weight_decay={cfg.train.weight_decay}",
        f"--logging_steps={cfg.train.logging_steps}",
        f"--save_steps={cfg.train.save_steps}",
        f"--eval_steps={cfg.train.eval_steps}",
        f"--save_total_limit={cfg.train.save_total_limit}",
        f"--seed={cfg.train.seed}",
        f"--evaluation_strategy={cfg.train.evaluation_strategy}",
        f"--gradient_accumulation_steps={cfg.train.gradient_accumulation_steps}",
        f"--max_grad_norm={cfg.train.max_grad_norm}",
        f"--max_length={cfg.train.max_length}",
        f"--warmup_ratio={cfg.train.warmup_ratio}",
        f"--metric_for_best_model={cfg.train.metric_for_best_model}",
        f"--report_to={cfg.train.report_to}",
    ]
    if cfg.train.fp16:
        train_cmd.append("--fp16")
    if cfg.train.predict_with_generate:
        train_cmd.append("--predict_with_generate")
    if cfg.train.remove_unused_columns:
        train_cmd.append("--remove_unused_columns")
    if cfg.train.load_best_model_at_end:
        train_cmd.append("--load_best_model_at_end")
    if cfg.train.greater_is_better:
        train_cmd.append("--greater_is_better")

    print("[Hydra Pipeline] Training stage...")
    subprocess.run(train_cmd, check=True)

    # Inference the model on test set
    pred_csv = run_dir / "predictions.csv"
    infer_cmd = [
        "python",
        "-m",
        "spellchecker.models.train_t5.inference",
        f"--model_dir={get_latest_checkpoint(run_dir)}",
        f"--input_csv={cfg.paths.test_csv}",
        f"--output_csv={pred_csv}",
        f"--batch_size={cfg.inference.batch_size}",
        f"--max_length={cfg.inference.max_length}",
        f"--device={cfg.inference.device}",
    ]
    print("[Hydra Pipeline] Inference stage...")
    subprocess.run(infer_cmd, check=True)

    # Calculate metrics on the test set
    metrics_csv = run_dir / "metrics.csv"
    metrics_cmd = [
        "python",
        "-m",
        "spellchecker.models.train_t5.metrics",
        f"--input_csv={pred_csv}",
        f"--output_csv={metrics_csv}",
    ]
    print("[Hydra Pipeline] Metrics stage...")
    subprocess.run(metrics_cmd, check=True)

    print(f"[Hydra Pipeline] Done. Results in {run_dir}")


if __name__ == "__main__":
    main()
