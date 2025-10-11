import json
import random
import re
import string
import typing as tp
from pathlib import Path

import pandas as pd

from spellchecker.data.processors.llm_processor import LLMProcessor
from spellchecker.data.processors.utils import query_llm


class Corruptor(LLMProcessor):
    """Text corruption processor."""

    def __init__(
        self,
        corruption_mode: str,
        prompt_path: Path = Path("spellchecker/data/prompts/corruptor.txt"),
        **kwargs,
    ):
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_template = f.read()

        super().__init__(
            prompt_template=prompt_template, output_column="source_text", **kwargs
        )

        self.corruption_mode: tp.Literal["llm", "heuristic"] = corruption_mode
        self.punctuations: tp.List[str] = list(",.;:!?")
        self.letters: tp.List[str] = string.ascii_letters

    def process_row(
        self,
        row: pd.Series,
    ) -> str:
        if self.corruption_mode == "llm":
            return self._corrupt_with_llm(row)
        else:
            return self._corrupt_with_heuristics(row)

    def _corrupt_with_heuristics(self, row: pd.Series) -> str:
        text = row["target_text"]
        words = text.split()
        num_words = len(words)

        # Estimate num of errors
        num_errors = max(1, int(0.15 * num_words))

        corrupted_words = words.copy()
        for _ in range(num_errors):
            # Choose random word to corrupt it
            idx = random.randint(0, num_words - 1)
            word = corrupted_words[idx]

            # Choose error type
            error_type = random.choices(
                ["spelling", "punctuation", "case"], weights=[0.6, 0.25, 0.15]
            )[0]

            if error_type == "spelling" and len(word) > 1:
                corrupted_words[idx] = self._corrupt_word(word)
            elif error_type == "punctuation":
                corrupted_words[idx] = self._corrupt_punctuation(word)
            elif error_type == "case":
                corrupted_words[idx] = self._corrupt_case(word)

        corrupted_text = " ".join(corrupted_words)
        return corrupted_text

    def _corrupt_word(self, word: str) -> str:
        op = random.choice(["insert", "delete", "substitute", "swap"])
        if op == "insert":
            pos = random.randint(0, len(word))
            char = random.choice(self.letters)
            return word[:pos] + char + word[pos:]
        elif op == "delete" and len(word) > 1:
            pos = random.randint(0, len(word) - 1)
            return word[:pos] + word[pos + 1 :]
        elif op == "substitute":
            pos = random.randint(0, len(word) - 1)
            char = random.choice(self.letters)
            return word[:pos] + char + word[pos + 1 :]
        elif op == "swap" and len(word) > 1:
            pos = random.randint(0, len(word) - 2)
            lst = list(word)
            lst[pos], lst[pos + 1] = lst[pos + 1], lst[pos]
            return "".join(lst)
        return word

    def _corrupt_punctuation(self, word: str) -> str:
        op = random.choice(["remove", "duplicate", "replace"])
        punct = random.choice(self.punctuations)
        if op == "remove":
            return re.sub(rf"[{''.join(self.punctuations)}]", "", word)
        elif op == "duplicate":
            return word + punct
        elif op == "replace":
            return re.sub(rf"[{''.join(self.punctuations)}]", punct, word)
        return word

    def _corrupt_case(self, word: str) -> str:
        if word.islower():
            return word.capitalize()
        elif word.isupper():
            return word.lower()
        else:
            return "".join(
                c.upper() if random.random() < 0.5 else c.lower() for c in word
            )

    def _corrupt_with_llm(
        self,
        row: pd.Series,
    ) -> str:
        passage: str = row["target_text"]
        error_types: tp.List[str] = self._sample_error_types()
        num_of_errors: int = self._sample_num_errors(passage=passage)
        prompt = (
            self.prompt_template.replace("<<PASSAGE>>", row["target_text"])
            .replace("<<NUM_ERRORS>>", str(num_of_errors))
            .replace("<<ERROR_TYPES>>", json.dumps(error_types))
        )

        response = "".join(
            query_llm(
                model=self.model_name,
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=500,
            )
        )

        return response.strip()

    def _sample_error_types(self):
        r = random.random()
        if r < 0.7:
            return [
                random.choices(
                    ["spelling", "punctuation", "case"], weights=[65, 25, 10]
                )[0]
            ]
        elif r < 0.9:
            return random.sample(["spelling", "punctuation", "case"], 2)
        else:
            return ["spelling", "punctuation", "case"]

    def _sample_num_errors(self, passage: str) -> int:
        length: int = len(passage)

        if length < 150:
            choices = [1, 2, 3, 4]
            weights = [0.3, 0.4, 0.25, 0.05]
        else:
            choices = [1, 2, 3, 4, 5, 6, 7, 8]
            weights = [0.02, 0.03, 0.1, 0.2, 0.25, 0.2, 0.1, 0.1]

        return random.choices(choices, weights=weights, k=1)[0]
