import json
import random
import typing as tp
from pathlib import Path

import pandas as pd

from spellchecker.data.processors.llm_processor import LLMProcessor
from spellchecker.data.processors.utils import query_llm


class Corruptor(LLMProcessor):
    """Text corruption processor."""

    def __init__(self, prompt_path: Path = Path("../prompts/corruptor.txt"), **kwargs):
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_template = f.read()

        super().__init__(
            prompt_template=prompt_template,
            output_column="target_text",
            temperature=0.7,
            **kwargs
        )

    def process_row(
        self,
        row: pd.Series,
        num_errors: int = 3,
        error_types: tp.Optional[tp.List[str]] = None,
    ) -> str:

        if error_types is None:
            error_types = ["spelling", "punctuation", "case"]

        prompt = self.prompt_template.format(
            passage=row["target_text"],
            num_errors=num_errors,
            error_types=json.dumps(error_types),
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

    def sample_num_errors(
        self, passage: str, max_errors_short: int = 3, max_errors_long: int = 6
    ) -> int:
        """
        Sample the number of SEC errors to insert based on passage length."""
        num_tokens = len(passage.split())

        # Decide max errors based on length
        max_errors = max_errors_short if num_tokens < 40 else max_errors_long

        # Probabilistic distribution for number of errors
        distribution: tp.List[float] = [0.05, 0.35, 0.30, 0.20, 0.10]

        if max_errors > 4:
            extra = max_errors - 4
            distribution += [0.05] * extra

        total = sum(distribution[: max_errors + 1])
        probs = [p / total for p in distribution[: max_errors + 1]]

        errors = random.choices(range(max_errors + 1), weights=probs, k=1)[0]
        return errors
