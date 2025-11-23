import logging

from transformers.trainer_callback import TrainerCallback

logger = logging.getLogger(__name__)
logging.basicConfig(format="[INFO] %(message)s", level=logging.INFO)


class MetricsLoggerCallback(TrainerCallback):
    """
    Custom callback to log training and validation losses
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        msg_parts = []
        if "loss" in logs:
            msg_parts.append(f"train_loss: {logs['loss']:.4f}")
        if "eval_loss" in logs:
            msg_parts.append(f"val_loss: {logs['eval_loss']:.4f}")
        if "epoch" in logs:
            msg_parts.append(f"epoch: {logs['epoch']:.2f}")
        if "learning_rate" in logs:
            msg_parts.append(f"lr: {logs['learning_rate']:.6f}")
        if msg_parts:
            logger.info(" | ".join(msg_parts))
