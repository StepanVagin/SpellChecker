import re
from pathlib import Path
import pandas as pd
from difflib import SequenceMatcher
import Levenshtein


class HeuristicValidator:
    def __init__(
        self,
        max_edit_ratio: float = 0.7,
        min_word_similarity: float = 0.4,
        max_word_diff: int = 9,
        allow_case_normalization: bool = True,
        invalid_chars_pattern: str = r"[^\w\s.,!?;:'\"()\[\]{}@#$%&*/\\+=<>€£¥*-]",
    ):
        self.max_edit_ratio = max_edit_ratio
        self.min_word_similarity = min_word_similarity
        self.max_word_diff = max_word_diff
        self.allow_case_normalization = allow_case_normalization
        self.invalid_chars_pattern = re.compile(invalid_chars_pattern)

    def is_valid_pair(self, source: str, target: str) -> bool:
        if not isinstance(source, str) or not isinstance(target, str):
            return False

        source, target = source.strip(), target.strip()
        if not source or not target:
            return False

        if source == target:
            return False

        if self.invalid_chars_pattern.search(target):
            return False

        # edit distance ratio
        ratio = Levenshtein.distance(source, target) / max(len(source), len(target))
        if ratio > self.max_edit_ratio:
            return False

        # word-level similarity
        source_words = source.split()
        target_words = target.split()
        if abs(len(target_words) - len(source_words)) > self.max_word_diff:
            return False

        matcher = SequenceMatcher(None, source_words, target_words)
        if matcher.ratio() < self.min_word_similarity:
            return False

        # punctuation / spacing check
        if re.search(r"\s{2,}", target):
            return False
        if re.search(r"[.,!?]{2,}", target):
            return False

        # case normalization allowance
        if self.allow_case_normalization and source.lower() == target.lower():
            return True

        # numbers / symbols consistency
        source_digits = re.findall(r"\d", source)
        target_digits = re.findall(r"\d", target)
        if source_digits != target_digits:
            return False

        return True

    def process_dataframe(self, input_file: Path) -> pd.DataFrame:
        df = pd.read_csv(input_file)
        if not {"source_text", "target_text"}.issubset(df.columns):
            raise ValueError(
                "Input CSV must contain 'source_text' and 'target_text' columns"
            )

        df["is_valid"] = df.apply(
            lambda row: self.is_valid_pair(row["source_text"], row["target_text"]),
            axis=1,
        )
        return df
