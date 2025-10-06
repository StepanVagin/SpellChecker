"""
Parsers for unsupervised text corpora used for n-gram language model training.

Supports:
- Wikipedia Dumps
- CC-News (Common Crawl News)
- BookCorpus
"""

import json
import re
import typing as tp
from pathlib import Path

import pandas as pd


class UniversalTextCleaner:
    """Clean and normalize text for n-gram language model training"""

    def __init__(
        self,
        min_length: int = 10,
        max_length: int = 1000,
        remove_urls: bool = True,
        remove_emails: bool = True,
        normalize_whitespace: bool = True,
    ):
        """
        Initialize text cleaner.

        Args:
            min_length: Minimum text length in characters
            max_length: Maximum text length in characters
            remove_urls: Whether to remove URLs
            remove_emails: Whether to remove email addresses
            normalize_whitespace: Whether to normalize whitespace
        """
        self.min_length = min_length
        self.max_length = max_length
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.normalize_whitespace = normalize_whitespace

    def clean(self, text: str) -> tp.Optional[str]:
        """
        Clean a single text string.

        Args:
            text: Input text

        Returns:
            Cleaned text or None if text should be filtered out
        """
        if not text or not isinstance(text, str):
            return None

        # Remove URLs
        if self.remove_urls:
            text = re.sub(
                r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
                "",
                text,
            )

        # Remove emails
        if self.remove_emails:
            text = re.sub(r"\S+@\S+", "", text)

        # Normalize whitespace
        if self.normalize_whitespace:
            text = re.sub(r"\s+", " ", text)
            text = text.strip()

        # Filter by length
        if len(text) < self.min_length or len(text) > self.max_length:
            return None

        return text


class WikipediaParser:
    """
    Parser for Wikipedia dumps.

    Expects preprocessed Wikipedia text files (one article per line) or
    extracted text from wikiextractor output.
    """

    def __init__(self, cleaner: tp.Optional[UniversalTextCleaner] = None):
        """
        Initialize parser.

        Args:
            cleaner: Text cleaner instance. If None, uses default settings.
        """
        self.cleaner = cleaner or UniversalTextCleaner()

    def parse_wiki_extractor_output(
        self, input_dir: tp.Union[str, Path]
    ) -> tp.Iterator[str]:
        """
        Parse WikiExtractor output directory.

        WikiExtractor creates nested directories with files like wiki_00, wiki_01, etc.
        Each file contains multiple articles in a specific format.

        Args:
            input_dir: Path to WikiExtractor output directory

        Yields:
            Cleaned text passages
        """
        input_path = Path(input_dir)

        # Check if directory has any wiki files
        wiki_files = list(input_path.rglob("wiki_*"))
        if not wiki_files:
            raise FileNotFoundError(f"No Wikipedia files found in {input_path}")

        # Recursively find all wiki_* files
        for wiki_file in sorted(wiki_files):
            if not wiki_file.is_file():
                continue

            print(f"Processing {wiki_file}...")

            with open(wiki_file, "r", encoding="utf-8") as f:
                current_article = []

                for line in f:
                    line = line.strip()

                    # Skip document markers
                    if line.startswith("<doc") or line.startswith("</doc>"):
                        if current_article:
                            text = " ".join(current_article)
                            cleaned = self.cleaner.clean(text)
                            if cleaned:
                                yield cleaned
                            current_article = []
                        continue

                    if line:
                        current_article.append(line)

                # Don't forget the last article
                if current_article:
                    text = " ".join(current_article)
                    cleaned = self.cleaner.clean(text)
                    if cleaned:
                        yield cleaned

    def parse_plain_text(
        self, input_file: tp.Union[str, Path]
    ) -> tp.Iterator[str]:
        """
        Parse plain text file (one document/paragraph per line).

        Args:
            input_file: Path to plain text file

        Yields:
            Cleaned text passages
        """
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                cleaned = self.cleaner.clean(line.strip())
                if cleaned:
                    yield cleaned

    def save_to_file(
        self, input_path: tp.Union[str, Path], output_file: tp.Union[str, Path]
    ) -> int:
        """
        Parse Wikipedia data and save to output file.

        Args:
            input_path: Path to input directory or file
            output_file: Path to output text file

        Returns:
            Number of text passages saved
        """
        input_path = Path(input_path)
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        count = 0
        with open(output_path, "w", encoding="utf-8") as out_f:
            if input_path.is_dir():
                # Process WikiExtractor output
                for text in self.parse_wiki_extractor_output(input_path):
                    out_f.write(text + "\n")
                    count += 1
            else:
                # Process plain text file
                for text in self.parse_plain_text(input_path):
                    out_f.write(text + "\n")
                    count += 1

        print(f"Saved {count} passages to {output_path}")
        return count


class CCNewsParser:
    """
    Parser for CC-News (Common Crawl News) dataset.

    CC-News is distributed as WARC files or JSONL files with news articles.
    """

    def __init__(self, cleaner: tp.Optional[UniversalTextCleaner] = None):
        """
        Initialize parser.

        Args:
            cleaner: Text cleaner instance. If None, uses default settings.
        """
        self.cleaner = cleaner or UniversalTextCleaner()

    def parse_jsonl(self, input_file: tp.Union[str, Path]) -> tp.Iterator[str]:
        """
        Parse CC-News JSONL file.

        Expected format: Each line is a JSON object with fields like:
        - text or maintext: The article text
        - title: Article title

        Args:
            input_file: Path to JSONL file

        Yields:
            Cleaned article texts
        """
        with open(input_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    article = json.loads(line.strip())

                    # Try different field names for text content
                    text = article.get("text") or article.get("maintext") or ""
                    title = article.get("title", "")

                    # Combine title and text
                    full_text = f"{title}. {text}" if title else text

                    cleaned = self.cleaner.clean(full_text)
                    if cleaned:
                        yield cleaned

                except json.JSONDecodeError:
                    print(f"Warning: Could not parse JSON at line {line_num}")
                    continue
                except Exception as e:
                    print(f"Warning: Error processing line {line_num}: {e}")
                    continue

    def save_to_file(
        self, input_file: tp.Union[str, Path], output_file: tp.Union[str, Path]
    ) -> int:
        """
        Parse CC-News data and save to output file.

        Args:
            input_file: Path to input JSONL file
            output_file: Path to output text file

        Returns:
            Number of articles saved
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        count = 0
        with open(output_path, "w", encoding="utf-8") as out_f:
            for text in self.parse_jsonl(input_file):
                out_f.write(text + "\n")
                count += 1

        print(f"Saved {count} articles to {output_path}")
        return count


class BookCorpusParser:
    """
    Parser for BookCorpus dataset.

    BookCorpus typically comes as plain text files, one book per file,
    or as a dataset with sentences/paragraphs.
    """

    def __init__(self, cleaner: tp.Optional[UniversalTextCleaner] = None):
        """
        Initialize parser.

        Args:
            cleaner: Text cleaner instance. If None, uses default settings.
        """
        self.cleaner = cleaner or UniversalTextCleaner(
            min_length=20, max_length=2000  # Books have longer passages
        )

    def parse_text_file(self, input_file: tp.Union[str, Path]) -> tp.Iterator[str]:
        """
        Parse a single book text file.

        Splits into paragraphs for manageable chunks.

        Args:
            input_file: Path to book text file

        Yields:
            Cleaned paragraphs
        """
        with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # Split by double newlines (paragraphs)
        paragraphs = re.split(r"\n\s*\n", content)

        for paragraph in paragraphs:
            cleaned = self.cleaner.clean(paragraph.strip())
            if cleaned:
                yield cleaned

    def parse_directory(self, input_dir: tp.Union[str, Path]) -> tp.Iterator[str]:
        """
        Parse all text files in a directory.

        Args:
            input_dir: Path to directory containing book files

        Yields:
            Cleaned paragraphs from all books
        """
        input_path = Path(input_dir)

        # Find all .txt files
        for txt_file in sorted(input_path.rglob("*.txt")):
            print(f"Processing {txt_file.name}...")
            try:
                for paragraph in self.parse_text_file(txt_file):
                    yield paragraph
            except Exception as e:
                print(f"Warning: Error processing {txt_file}: {e}")
                continue

    def save_to_file(
        self, input_path: tp.Union[str, Path], output_file: tp.Union[str, Path]
    ) -> int:
        """
        Parse BookCorpus data and save to output file.

        Args:
            input_path: Path to input directory or file
            output_file: Path to output text file

        Returns:
            Number of paragraphs saved
        """
        input_path = Path(input_path)
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        count = 0
        with open(output_path, "w", encoding="utf-8") as out_f:
            if input_path.is_dir():
                for text in self.parse_directory(input_path):
                    out_f.write(text + "\n")
                    count += 1
            else:
                for text in self.parse_text_file(input_path):
                    out_f.write(text + "\n")
                    count += 1

        print(f"Saved {count} passages to {output_path}")
        return count


def process_unsupervised_corpus(
    corpus_type: str,
    input_path: tp.Union[str, Path],
    output_file: tp.Union[str, Path],
    cleaner: tp.Optional[UniversalTextCleaner] = None,
) -> int:
    """
    Universal function to process any unsupervised corpus.

    Args:
        corpus_type: Type of corpus ('wikipedia', 'ccnews', or 'bookcorpus')
        input_path: Path to input data
        output_file: Path to output text file
        cleaner: Optional text cleaner instance

    Returns:
        Number of passages processed

    Raises:
        ValueError: If corpus_type is not recognized
    """
    corpus_type = corpus_type.lower()

    if corpus_type == "wikipedia":
        parser = WikipediaParser(cleaner)
        return parser.save_to_file(input_path, output_file)
    elif corpus_type == "ccnews":
        parser = CCNewsParser(cleaner)
        return parser.save_to_file(input_path, output_file)
    elif corpus_type == "bookcorpus":
        parser = BookCorpusParser(cleaner)
        return parser.save_to_file(input_path, output_file)
    else:
        raise ValueError(
            f"Unknown corpus type: {corpus_type}. "
            f"Must be one of: wikipedia, ccnews, bookcorpus"
        )


