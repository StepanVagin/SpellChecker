import json
import re
import typing as tp
from pathlib import Path

import pandas as pd


class GitHubTyposParser:
    """Parser for the GitHub Typo Corpus"""

    def __init__(self) -> None:
        self.records: tp.List[tp.Dict[str, tp.Union[str, float, bool]]] = []

    def parse_file(
        self, file_path: tp.Union[str, Path], filtration: tp.Optional[bool] = True
    ) -> pd.DataFrame:
        """Parse the GitHub Typo Corpus JSONL file"""
        file_path = Path(file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    commit = json.loads(line)
                    self._process_commit(commit)
                except json.JSONDecodeError:
                    continue

        df = pd.DataFrame(self.records)

        if filtration:
            df = df[
                (df["prob_typo"] > 0.9)
                & (df["language"] == "eng")
                & (df["file_path"].str.contains(".md"))
            ]
            df = self.apply_text_filters(df)

        return df

    def apply_text_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply text quality filters to the dataset."""
        URL_PATTERN = re.compile(r"https?://[^\s]+|www\.[^\s]+")
        EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
        CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```|`[^`]+`")
        EXCESSIVE_SPECIAL_CHARS = re.compile(r"[^\w\s]{5,}")

        # Remove URLs and emails
        df = df[~df["source_text"].str.contains(URL_PATTERN, regex=True, na=False)]
        df = df[~df["source_text"].str.contains(EMAIL_PATTERN, regex=True, na=False)]

        # Remove code blocks
        df = df[
            ~df["source_text"].str.contains(CODE_BLOCK_PATTERN, regex=True, na=False)
        ]

        # Remove excessive special characters
        df = df[
            ~df["source_text"].str.contains(
                EXCESSIVE_SPECIAL_CHARS, regex=True, na=False
            )
        ]

        # Min&max text length
        MIN_TEXT_LENGTH = 45
        MAX_TEXT_LENGTH = 450
        df = df[df["source_text"].str.len() >= MIN_TEXT_LENGTH]
        df = df[df["target_text"].str.len() >= MIN_TEXT_LENGTH]
        df = df[df["source_text"].str.len() <= MAX_TEXT_LENGTH]
        df = df[df["target_text"].str.len() <= MAX_TEXT_LENGTH]

        # Remove duplicates
        df = df.drop_duplicates(subset=["source_text", "target_text"])

        return df

    def _process_commit(self, commit: tp.Dict[str, tp.Any]) -> None:
        """Process a single commit from the corpus"""
        for edit in commit.get("edits", []):
            # Skip if source or target text is missing
            if not edit.get("src") or not edit.get("tgt"):
                continue

            self.records.append(
                {
                    "repo": commit["repo"],
                    "commit_hash": commit["commit"],
                    "commit_message": commit["message"],
                    "file_path": edit["src"]["path"],
                    "language": edit["src"]["lang"],
                    "source_text": edit["src"]["text"],
                    "target_text": edit["tgt"]["text"],
                    "prob_typo": edit.get("prob_typo"),
                    "is_typo": edit.get("is_typo", False),
                }
            )
