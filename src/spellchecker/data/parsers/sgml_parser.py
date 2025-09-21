import pandas as pd
import re
import typing as tp
from pathlib import Path


class SGMLParser:
    """Universal parser for SGML format"""

    def parse_file(self, file_path: str) -> tp.List[tp.Dict[str, tp.Any]]:
        """Parse single SGML file"""
        with open(file_path, "r", encoding="utf-8") as f:
            content: str = f.read()

        annotations: tp.List[tp.Dict[str, tp.Any]] = []

        for doc_match in re.finditer(
            r'<DOC nid="(\d+)">(.*?)</DOC>', content, re.DOTALL
        ):
            doc_id: str = doc_match.group(1)
            doc_content: str = doc_match.group(2)

            # Extract paragraphs
            paragraphs: tp.List[str] = self._extract_paragraphs(doc_content)
            if not paragraphs:
                continue

            # Extract annotations
            doc_annotations: tp.List[str] = self._extract_annotations(
                doc_content, doc_id, paragraphs
            )
            annotations.extend(doc_annotations)

        return annotations

    def _extract_paragraphs(self, doc_content: str) -> tp.List[str]:
        """Extract paragraphs from document"""
        text_match: str = re.search(r"<TEXT>(.*?)</TEXT>", doc_content, re.DOTALL)
        if not text_match:
            return []

        paragraphs: tp.List[str] = []
        for p_match in re.finditer(r"<P>(.*?)</P>", text_match.group(1), re.DOTALL):
            paragraphs.append(p_match.group(1).strip())

        return paragraphs

    def _extract_annotations(
        self, doc_content: str, doc_id: str, paragraphs: tp.List[str]
    ) -> tp.List[tp.Dict[str, tp.Any]]:
        """Extract error annotations from document"""
        annotations: tp.List[tp.Dict[str, tp.Any]] = []

        for ann_match in re.finditer(
            r'<ANNOTATION teacher_id="(\d+)">(.*?)</ANNOTATION>', doc_content, re.DOTALL
        ):
            teacher_id: str = ann_match.group(1)

            for mistake in re.finditer(
                r'<MISTAKE start_par="(\d+)" start_off="(\d+)" end_par="(\d+)" end_off="(\d+)">(.*?)</MISTAKE>',
                ann_match.group(2),
                re.DOTALL,
            ):

                start_par, start_off, end_par, end_off = map(int, mistake.groups()[:4])
                mistake_content: str = mistake.group(5)

                # Extract error type and correction
                error_type_match = re.search(r"<TYPE>(.*?)</TYPE>", mistake_content)
                correction_match = re.search(
                    r"<CORRECTION>(.*?)</CORRECTION>", mistake_content
                )

                if not error_type_match:
                    continue

                error_type: str = error_type_match.group(1)
                correction: str = correction_match.group(1) if correction_match else ""

                # Extract original text
                original_text: str = self._extract_original_text(
                    paragraphs, start_par, start_off, end_par, end_off
                )

                annotations.append(
                    {
                        "doc_id": doc_id,
                        "teacher_id": teacher_id,
                        "error_type": error_type,
                        "original_text": original_text,
                        "correction": correction,
                        "start_par": start_par,
                        "start_off": start_off,
                        "end_par": end_par,
                        "end_off": end_off,
                    }
                )

        return annotations

    def _extract_original_text(
        self,
        paragraphs: tp.List[str],
        start_par: int,
        start_off: int,
        end_par: int,
        end_off: int,
    ) -> str:
        """Safely extract original text from paragraphs"""
        try:
            if start_par >= len(paragraphs) or end_par >= len(paragraphs):
                return "INVALID_PAR"

            if start_par != end_par:
                return "MULTI_PAR"

            para: str = paragraphs[start_par]

            if start_off < 0 or end_off > len(para) or start_off > end_off:
                return f"INVALID_POS({start_off}:{end_off})"

            original: str = para[start_off:end_off]
            return original if original else "INSERTION"

        except Exception as e:
            return f"ERROR:{e}"

    def parse_directory(self, directory_path: str) -> pd.DataFrame:
        """Parse all SGML files in directory"""
        all_annotations: tp.List[tp.Dict[str, tp.Any]] = []

        for sgml_file in Path(directory_path).rglob("*.sgml"):
            print(f"Processing {sgml_file.name}...")

            file_annotations: tp.Dict[str, tp.Any] = self.parse_file(sgml_file)

            # Add source info
            for ann in file_annotations:
                ann["source_file"] = sgml_file.name
                ann["source_dir"] = sgml_file.parent.name

            all_annotations.extend(file_annotations)
            print(f"  -> {len(file_annotations)} annotations")

        return pd.DataFrame(all_annotations)

    def extract_sentence_pairs(self, directory_path: str) -> pd.DataFrame:
        """Extract sentence-level source-target pairs with error corrections"""
        sentence_pairs: tp.List[tp.Dict[str, tp.Any]] = []

        for sgml_file in Path(directory_path).rglob("*.sgml"):
            print(f"Processing {sgml_file.name} for sentence pairs...")

            with open(sgml_file, "r", encoding="utf-8") as f:
                content: str = f.read()

            for doc_match in re.finditer(
                r'<DOC nid="(\d+)">(.*?)</DOC>', content, re.DOTALL
            ):
                doc_id: str = doc_match.group(1)
                doc_content: str = doc_match.group(2)

                paragraphs: tp.List[str] = self._extract_paragraphs(doc_content)
                if not paragraphs:
                    continue

                doc_errors: tp.List[tp.Dict[str, tp.Any]] = self._extract_annotations(
                    doc_content, doc_id, paragraphs
                )

                # Process each paragraph for sentence extraction
                for par_idx, paragraph in enumerate(paragraphs):
                    sentences: tp.List[str] = re.split(r"(?<=[.!?])\s+", paragraph)
                    sent_start: int = 0

                    for sent_text in sentences:
                        if not sent_text.strip():
                            continue

                        sent_pos: int = paragraph.find(sent_text, sent_start)
                        if sent_pos == -1:
                            sent_start += len(sent_text) + 1
                            continue

                        sent_end: int = sent_pos + len(sent_text)

                        # Find errors in this sentence
                        sent_errors: tp.List[tp.Dict[str, tp.Any]] = [
                            error
                            for error in doc_errors
                            if (
                                error["start_par"] == par_idx
                                and sent_pos <= error["start_off"] < sent_end
                            )
                        ]

                        # Create sentence pair if there are errors
                        if sent_errors:
                            corrected_sent: str = sent_text
                            for error in sorted(
                                sent_errors, key=lambda x: x["start_off"], reverse=True
                            ):
                                rel_start: int = error["start_off"] - sent_pos
                                rel_end: int = error["end_off"] - sent_pos

                                if 0 <= rel_start <= len(
                                    corrected_sent
                                ) and 0 <= rel_end <= len(corrected_sent):
                                    corrected_sent = (
                                        corrected_sent[:rel_start]
                                        + error["correction"]
                                        + corrected_sent[rel_end:]
                                    )

                            sentence_pairs.append(
                                {
                                    "doc_id": doc_id,
                                    "paragraph_id": par_idx,
                                    "source_sentence": sent_text.strip(),
                                    "target_sentence": corrected_sent.strip(),
                                    "error_types": [
                                        e["error_type"] for e in sent_errors
                                    ],
                                    "num_errors": len(sent_errors),
                                    "source_file": sgml_file.name,
                                }
                            )

                        sent_start = sent_end

        print(f"  -> {len(sentence_pairs)} sentence pairs extracted")
        return pd.DataFrame(sentence_pairs)


def load_sgml_annotations(sgml_path: str) -> pd.DataFrame:
    """Load annotations from SGML corpus"""
    return SGMLParser().parse_directory(sgml_path)


def extract_sgml_sentences(sgml_path: str) -> pd.DataFrame:
    """Extract sentence pairs from SGML corpus"""
    return SGMLParser().extract_sentence_pairs(sgml_path)
