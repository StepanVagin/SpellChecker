import re
import typing as tp

import jsonlines
import pandas as pd
from datasets import load_dataset
from nltk.tokenize import sent_tokenize
from tqdm import tqdm


class PileCorpusParser:
    """Parser and sampler for Pile corpus with passage splitting and filtering."""

    ALLOWED_SOURCES: tp.Set[str] = {
        "Pile-CC",
        "BookCorpus2",
        "PhilPapers",
        "PubMed Central",
        "Wikipedia (en)",
    }

    def __init__(self, target_sample: int = 60000):
        self.records: tp.List[tp.Dict[str, tp.Any]] = []
        self.target_sample = target_sample

    @staticmethod
    def split_into_passages(
        text: str,
        min_words: int = 20,
        max_words: int = 100,
        step: int = 10,
        max_digit_ratio: float = 0.15,
        remove_links: bool = True,
    ) -> tp.List[str]:
        if not text or not text.strip():
            return []

        text = re.sub(r"\n+", " ", text)
        sentences = sent_tokenize(text)
        passages = []

        for start in range(len(sentences)):
            current_passage = []
            current_len = 0
            for i in range(start, len(sentences)):
                words = sentences[i].split()
                current_passage.append(sentences[i])
                current_len += len(words)

                if current_len >= min_words:
                    if current_len > max_words:
                        break

                    passage_text = " ".join(current_passage).strip()

                    if remove_links and re.search(r"https?://|www\.|<", passage_text):
                        break
                    elif (
                        sum(c.isdigit() for c in passage_text)
                        / max(len(passage_text), 1)
                        > max_digit_ratio
                    ):
                        break
                    else:
                        passages.append(passage_text)
                        break

        return passages

    def _process_text(self, text: str) -> None:
        """Split text into passages and add to records"""
        passages = self.split_into_passages(text)
        for passage in passages:
            self.records.append({"passage": passage})

    def apply_text_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply quality filters to passages"""
        URL_PATTERN = re.compile(r"https?://[^\s]+|www\.[^\s]+")
        EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
        HTML_PATTERN = re.compile(r"<[^>]+>")
        EXCESSIVE_SPECIAL_CHARS = re.compile(r"[^\w\s]{5,}")
        MAX_LENGTH = 500

        df = df[~df["passage"].str.contains(URL_PATTERN, regex=True, na=False)]
        df = df[~df["passage"].str.contains(EMAIL_PATTERN, regex=True, na=False)]
        df = df[~df["passage"].str.contains(HTML_PATTERN, regex=True, na=False)]
        df = df[
            ~df["passage"].str.contains(EXCESSIVE_SPECIAL_CHARS, regex=True, na=False)
        ]
        df = df[df["passage"].apply(len) < MAX_LENGTH]

        # Length filters
        MIN_WORDS = 6
        MAX_WORDS = 250
        df["num_words"] = df["passage"].str.split().apply(len)
        df = df[df["num_words"] >= MIN_WORDS]
        df = df[df["num_words"] <= MAX_WORDS]

        # Remove duplicates
        df = df.drop_duplicates(subset=["passage"])

        return df

    def download_and_parse(self, split: str = "train") -> pd.DataFrame:
        """Download Pile-uncopyrighted, filter sources, and parse passages"""
        ds = load_dataset("monology/pile-uncopyrighted", split=split, streaming=True)
        pbar = tqdm(total=self.target_sample, desc="Sampling passages")

        for example in ds:
            meta = example.get("meta", {})
            source = meta.get("pile_set_name")
            text = example.get("text", "")

            if source not in self.ALLOWED_SOURCES or not text.strip():
                continue

            self._process_text(text)
            pbar.n = min(len(self.records), self.target_sample)
            pbar.refresh()

            if len(self.records) >= self.target_sample:
                break

        pbar.close()
        df = pd.DataFrame(self.records)
        df = self.apply_text_filters(df)

        if len(df) > self.target_sample:
            df = df.sample(self.target_sample, random_state=42).reset_index(drop=True)

        return df

    def save_jsonl(
        self, df: pd.DataFrame, path: str = "unsupervised_sampled_pile.jsonl"
    ):
        with jsonlines.open(path, mode="w") as writer:
            writer.write_all(df.to_dict(orient="records"))
        print(f"Saved {len(df)} passages to {path}")

    def save_csv(self, df: pd.DataFrame, path: str = "unsupervised_sampled_pile.csv"):
        df_to_save = df.rename(columns={"passage": "target_text"})
        df_to_save.to_csv(path, index=False, encoding="utf-8")
        print(f"Saved {len(df_to_save)} passages to {path}")
