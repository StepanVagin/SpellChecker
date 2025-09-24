import re
import typing as tp
from pathlib import Path

import pandas as pd


class BirkbeckSpellingParser:
    """Parser for the Birkbeck Spelling Error Corpus"""

    def __init__(self) -> None:
        self.records: tp.List[tp.Dict[str, tp.Union[str, int]]] = []

    def parse_file(self, file_path: tp.Union[str, Path]) -> pd.DataFrame:
        """Parse the Birkbeck spelling corpus file"""
        file_path = Path(file_path)
        with open(file_path, "r", encoding="latin1") as f:
            current_section: tp.Optional[str] = None

            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Track section headers
                if line.startswith("$ Type"):
                    current_section = line.split(":")[1].strip()
                    continue

                # Skip other metadata lines
                if line.startswith("$"):
                    continue

                # Process error entries
                self._process_entry(line, current_section, file_path.name)

        return pd.DataFrame(self.records)

    def _process_entry(
        self, line: str, error_type: tp.Optional[str], source_file: str
    ) -> None:
        """Process a single corpus entry line"""
        parts = re.split(r"\s{2,}", line)
        if len(parts) < 2:
            return

        error_word = parts[0]
        correction = parts[1].strip("()")
        context = parts[2] if len(parts) > 2 else ""

        # Build full sentences
        source = (
            context.replace("*", error_word)
            if "*" in context
            else f"{error_word} {context}".strip()
        )
        target = (
            context.replace("*", correction)
            if "*" in context
            else f"{correction} {context}".strip()
        )

        self.records.append(
            {
                "source_text": source,
                "target_text": target,
                "error_word": error_word,
                "correction": correction,
                "error_type": error_type or "unspecified",
                "source_file": source_file,
            }
        )
