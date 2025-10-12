import typing as tp

from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

from datasets import DatasetDict
from spellchecker.models.train_t5.args import Seq2SeqTrainingConfig


class T5Seq2SeqTrainer:
    """
    High-level wrapper for training a T5 model using Hugging Face Seq2SeqTrainer.
    """

    def __init__(self, config: Seq2SeqTrainingConfig):
        self.config = config
        self.tokenizer: tp.Optional[T5Tokenizer] = None
        self.model: tp.Optional[T5ForConditionalGeneration] = None
        self.data_collator: tp.Optional[DataCollatorForSeq2Seq] = None
        self.trainer: tp.Optional[Seq2SeqTrainer] = None

    def setup(self, dataset: DatasetDict):
        """
        Load model, tokenizer, and initialize Seq2SeqTrainer.
        """
        self.tokenizer = T5Tokenizer.from_pretrained(self.config.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.config.model_name)

        self.data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer, model=self.model
        )

        training_args = Seq2SeqTrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            predict_with_generate=self.config.predict_with_generate,
            fp16=self.config.fp16,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            save_total_limit=self.config.save_total_limit,
        )

        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )

    def train(self):
        """
        Start training.
        """
        if self.trainer is None:
            raise ValueError("Trainer not initialized. Call setup() first.")
        self.trainer.train()
