from dataclasses import dataclass


@dataclass
class Seq2SeqTrainingConfig:
    model_name: str = "t5-small"
    output_dir: str = "./results"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    learning_rate: float = 5e-6
    weight_decay: float = 0.01
    predict_with_generate: bool = True
    fp16: bool = True
    logging_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 2
