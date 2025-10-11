import re
from pathlib import Path

import pandas as pd

from spellchecker.data.processors.llm_processor import LLMProcessor
from spellchecker.data.processors.utils import query_llm


class Validator(LLMProcessor):
    """Text validation processor."""

    def __init__(self, prompt_path: Path = Path("../prompts/validator.txt"), **kwargs):
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_template = f.read()

        super().__init__(
            prompt_template=prompt_template,
            output_column="is_valid",
            temperature=0.0,
            max_tokens=10,
            **kwargs
        )

    def process_row(self, row: pd.Series) -> str:
        prompt = self.prompt_template.format(
            source=row["source_text"], target=row["target_text"]
        )

        response = "".join(
            query_llm(
                prompt,
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        )

        valid_pattern = re.compile(r"^\s*true\s*$", re.IGNORECASE)
        return bool(valid_pattern.match(response))
