"""Data parsers for spell checking datasets"""

from .birkbeck_parser import BirkbeckSpellingParser
from .sgml_parser import (SGMLParser, extract_sgml_sentences,
                          load_sgml_annotations)
from .unsupervised_parser import (BookCorpusParser, CCNewsParser,
                                  UniversalTextCleaner, WikipediaParser)

__all__ = [
    "BirkbeckSpellingParser",
    "SGMLParser",
    "load_sgml_annotations",
    "extract_sgml_sentences",
    "WikipediaParser",
    "CCNewsParser",
    "BookCorpusParser",
    "UniversalTextCleaner",
]


