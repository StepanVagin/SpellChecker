# SpellChecker

A comprehensive spell checking system with support for both supervised and unsupervised learning approaches.

## Features

- **Supervised Datasets**: NUCLE, CoNLL-2014, Birkbeck Spelling Corpus, GitHub Typo Corpus
- **Unsupervised Datasets**: Wikipedia, CC-News, BookCorpus for n-gram language model training
- **Data Parsers**: Robust parsers for multiple data formats (SGML, JSONL, plain text)
- **Pipeline Scripts**: Automated download and processing workflows

## Quick Start

### Install Dependencies

```bash
pip install pandas

# For unsupervised data
pip install wikiextractor datasets
```

### Download Supervised Data

For training supervised spell checkers:

```bash
bash scripts/data_preparation/download_raw_data.sh
```

### Download Unsupervised Data

For n-gram language model training:

```bash
bash scripts/data_preparation/download_unsupervised_data.sh
```

See [Quick Start Guide](docs/quick_start_unsupervised.md) for detailed instructions.

## Data Pipeline

### Supervised Datasets

The project includes parsers for:
- **NUCLE Corpus**: Non-native English learner errors
- **CoNLL-2014**: Grammar error correction test data
- **Birkbeck Spelling Corpus (1986)**: Classic spelling error corpus
- **GitHub Typo Corpus**: Real-world typos from code commits

### Unsupervised Datasets

For n-gram language model training:
- **Wikipedia Dumps**: ~20GB of encyclopedia articles
- **CC-News**: News articles from Common Crawl
- **BookCorpus**: 74M sentences from books

See [Unsupervised Data Pipeline](docs/unsupervised_data_pipeline.md) for complete documentation.

## Project Structure

```
SpellChecker/
├── data/                           # Data directory (gitignored)
│   ├── raw/                       # Raw downloaded data
│   │   ├── supervised/            # Supervised datasets
│   │   └── unsupervised/          # Unsupervised corpora
│   └── processed/                 # Processed data ready for training
│       ├── supervised/
│       └── unsupervised/
├── src/
│   └── spellchecker/
│       └── data/
│           └── parsers/           # Data parsers
│               ├── __init__.py
│               ├── birkbeck_parser.py
│               ├── sgml_parser.py
│               └── unsupervised_parser.py
├── scripts/
│   └── data_preparation/          # Data download and processing scripts
│       ├── download_raw_data.sh           # Download supervised data
│       ├── download_unsupervised_data.sh  # Download unsupervised data
│       ├── process_unsupervised_data.py   # Process unsupervised data
│       ├── example_ngram_usage.py         # N-gram training example
│       └── README.md
├── docs/                          # Documentation
│   ├── design_document.md
│   ├── unsupervised_data_pipeline.md
│   └── quick_start_unsupervised.md
└── notebooks/                     # Jupyter notebooks for exploration
```

## Usage Examples

### Process Unsupervised Data

```bash
# Process all corpora
python scripts/data_preparation/process_unsupervised_data.py --all

# Process individual corpus
python scripts/data_preparation/process_unsupervised_data.py \
    --corpus wikipedia \
    --input data/raw/unsupervised/wikipedia/extracted \
    --output data/processed/unsupervised/wikipedia.txt
```

### Train N-Gram Model

```bash
# Run example
python scripts/data_preparation/example_ngram_usage.py

# Or use KenLM for production
pip install kenlm
lmplz -o 5 < data/processed/unsupervised/wikipedia.txt > model.arpa
build_binary model.arpa model.klm
```

### Use Parsers in Python

```python
from spellchecker.data.parsers import WikipediaParser, UniversalTextCleaner

# Create parser with custom cleaner
cleaner = UniversalTextCleaner(min_length=20, max_length=500)
parser = WikipediaParser(cleaner)

# Process data
parser.save_to_file(
    input_path="data/raw/unsupervised/wikipedia/extracted",
    output_file="data/processed/unsupervised/wikipedia.txt"
)
```

## Documentation

- **[Quick Start Guide](docs/quick_start_unsupervised.md)**: Get started quickly with unsupervised data
- **[Unsupervised Data Pipeline](docs/unsupervised_data_pipeline.md)**: Complete pipeline documentation
- **[Design Document](docs/design_document.md)**: System design and architecture
- **[Data Preparation README](scripts/data_preparation/README.md)**: Detailed data preparation instructions

## Development

### Adding New Parsers

1. Create parser class in `src/spellchecker/data/parsers/`
2. Implement standard interface (see existing parsers)
3. Add to `__init__.py`
4. Update download scripts as needed

### Testing

```bash
# Test parsers on small samples
python -c "from spellchecker.data.parsers import WikipediaParser; parser = WikipediaParser(); print('Parser loaded successfully')"
```

## License

See individual dataset licenses:
- Wikipedia: CC BY-SA
- CC-News: Varies by source
- BookCorpus: Check Hugging Face terms

## Contributing

Contributions welcome! Please ensure:
- Code follows existing style
- New parsers follow standard interface
- Documentation is updated
- Tests are included

## References

- [Wikipedia Dumps](https://dumps.wikimedia.org/)
- [Common Crawl](https://commoncrawl.org/)
- [Hugging Face Datasets](https://huggingface.co/datasets)
- [KenLM](https://github.com/kpu/kenlm)
- [Birkbeck Spelling Corpus](https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/0643)
