import torch
import typing as tp
from transformers import T5ForConditionalGeneration, T5Tokenizer


class T5Model:
    def __init__(
        self,
        model_dir: str,
        device: str = "cpu",
        batch_size: int = 8,
        max_length: int = 128,
    ):
        print(f"[INFO] Loading model from {model_dir} on {device}")
        self.tokenizer = T5Tokenizer.from_pretrained(model_dir)
        self.model = T5ForConditionalGeneration.from_pretrained(model_dir)
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.model.to(self.device)
        self.model.eval()
        print("[INFO] Model loaded successfully.")


    def predict(self, texts: tp.List[str]) -> tp.List[str]:
        preds = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i : i + self.batch_size]
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                ).to(self.device)
                outputs = self.model.generate(**inputs, max_length=self.max_length)
                batch_preds = self.tokenizer.batch_decode(
                    outputs, skip_special_tokens=True
                )
                preds.extend(batch_preds)
        return preds
