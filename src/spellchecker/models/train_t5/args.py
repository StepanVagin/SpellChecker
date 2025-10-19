from dataclasses import dataclass


@dataclass
class Seq2SeqTrainingConfig:
    # Main model params
    model_name: str = "t5-small"
    output_dir: str = "./results"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    learning_rate: float = 5e-6
    weight_decay: float = 0.01
    predict_with_generate: bool = True
    fp16: bool = True

    # Logging & Checkpoints
    logging_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "loss"
    greater_is_better: bool = False

    gradient_accumulation_steps: int = 1
    warmup_steps: int = 0
    seed: int = 42
    remove_unused_columns: bool = True
    report_to: str = "all"
