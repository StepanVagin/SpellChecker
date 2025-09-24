import json
import typing as tp
from pathlib import Path

import pandas as pd


class GitHubTyposParser:
    """Parser for the GitHub Typo Corpus"""

    def __init__(self) -> None:
        self.records: tp.List[tp.Dict[str, tp.Union[str, float, bool]]] = []

    def parse_file(self, file_path: tp.Union[str, Path]) -> pd.DataFrame:
        """Parse the GitHub Typo Corpus JSONL file"""
        file_path = Path(file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    commit = json.loads(line)
                    self._process_commit(commit)
                except json.JSONDecodeError:
                    continue

        return pd.DataFrame(self.records)

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
                    "language": edit["src"]["lang"],
                    "source_text": edit["src"]["text"],
                    "target_text": edit["tgt"]["text"],
                    "prob_typo": edit.get("prob_typo"),
                    "is_typo": edit.get("is_typo", False),
                }
            )
