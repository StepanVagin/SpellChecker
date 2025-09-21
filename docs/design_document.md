# Spelling Error Correction (SEC)

## Team "SpellChecker" Information

- **Stepan Vagin**: s.vagin@innopolis.university
- **Egor Chernobrovkin**: e.chernobrovkin@innopolis.university
- **Lana Ermolaeva**: l.ermolaeva@innopolis.university
- **Link to the GitHub repository**: https://github.com/StepanVagin/SpellChecker

## Problem Definition

Spelling Error Correction is the task of automatically detecting and correcting orthographic errors in text while preserving the intended words and minimizing unnecessary changes.

## Formal Definition

Given a source sentence S with potential spelling errors, generate a corrected sentence T that:

1. Contains only correctly spelled words;

2. Preserves the intended meaning of S;

3. Applies minimal necessary character-level edits.

## Project goal
The final goal of the project is to develop and compare two spelling error correction systems - a baseline N-gram model and an advanced T5 transformer - that automatically detect and correct orthographic errors while preserving intended meaning. The project aims to advance understanding of how different computational approaches handle spelling correction challenges. The products will benefit language learners, business, and professionals by automated text correction systems. The team targets >85% correction accuracy and will provide evidence about the effectiveness of statistical versus deep learning methodologies through comprehensive evaluation. Ultimately, this project expands knowledge of automated text correction while creating practical solutions that improve communication quality across multiple real-world applications.

## Product Business and Social Value
SEC systems provide significant value across multiple domains by improving text readability and professionalism. SEC focuses on character-level errors within individual words, including typos, misspellings, and common orthographic mistakes (e.g., "recieve" → "receive", "teh" → "the").

Spell checkers save time and money by automatically fixing errors before they reach customers. They protect brand reputation by ensuring professional communication, reduce manual proofreading costs, and improve data quality for analytics.

E-commerce platforms suffer from users' orthographical errors, with 32% of search queries containing spelling mistakes. Given that search functionality directly affects revenue and customer experience, spelling correction systems provide measurable business value through improved product discovery and reduced customer frustration. [*]

Controlled studies in educational settings reveal significant learning benefits from spelling correction technology. Research with English as a Second Language (ESL) students demonstrated that spell-checker tools improved both error detection and correction learning, with effects persisting in delayed posttests. The study found that students using spell-checker aids showed superior performance compared to control groups, with learning transferring to different contexts and maintaining durability over time [**]

## Target Users
- **Language Learning and Accessibility**: Non-native English speakers represent a significant user base, with spelling correction systems providing educational feedback and confidence-building support;

- **Educational Sector**: Students and educators benefit from spelling correction tools for academic writing;

- **E-commerce and Digital Platforms**: Online retailers utilize spelling correction to improve search functionality and user experience;

- **Professional Applications**: Business professionals rely on spelling correction for maintaining professional credibility and ensuring polished communication.

## State-Of-The-Art solutions
**Contemporary smartphone autocorrect systems** build upon T9's foundation while incorporating advanced language modeling. These systems combine dictionary-based lookup with statistical language models that consider:

- Word frequency patterns from large text corpora;

- Contextual probability based on surrounding words;

- User-specific typing patterns and vocabulary preferences;

- Real-time error pattern recognition for common typos.

The basic algorithm remains similar to traditional spell-checkers: compare input against dictionary, suggest alternatives for non-matches, and predict intended words before completion. However, modern implementations utilize machine learning to improve suggestion accuracy and adapt to evolving language usage patterns.

**BART (Bidirectional and Auto-Regressive Transformers)** within [SAGE](https://github.com/ai-forever/sage) uses a denoising autoencoder structure. The encoder, similar to BERT, processes the entire corrupted sentence simultaneously using multi-head attention to understand context. The GPT-like decoder then generates corrected text word-by-word, attending to both encoder output and previous decoder states. With ~140 million parameters, BART ([link 1](https://www.geeksforgeeks.org/artificial-intelligence/bart-model-for-text-auto-completion-in-nlp/), [link 2](https://www.projectpro.io/article/transformers-bart-model-explained/553)) combines the bidirectional context understanding of BERT (110M parameters) with the autoregressive generation capabilities of GPT-1 (117M parameters), making it particularly effective for spelling error correction tasks.

**SAGE (Spell checking via Augmentation and Generative distribution Emulation)** combines transformer models with a sophisticated data generation approach. Creates realistic spelling errors using statistical analysis of human error patterns and heuristic corruption methods. [SAGE](https://github.com/ai-forever/sage) utilizes models like BART, T5, and mT5 for sequence-to-sequence correction. Pre-training on synthetic data followed by fine-tuning on real datasets.

## Specialized Correction Systems
Research has produced specialized correctors for specific demographics and contexts:

**KidSpell**: A phonetic-based correction model specifically designed for children's spelling errors, demonstrating superior accuracy compared to general-purpose correctors when handling developmental spelling patterns. [Research](https://www.researchgate.net/publication/341722766_KidSpell_A_Child-Oriented_Rule-Based_Phonetic_Spellchecker)

**Clinical Text Correctors**: Specialized systems for medical documentation that incorporate domain-specific vocabularies and error patterns common in healthcare settings. [Example](https://pmc.ncbi.nlm.nih.gov/articles/PMC11044887/)

**Multilingual Systems**: Advanced correctors handling code-switching and cross-linguistic interference patterns, particularly valuable for ESL learners and multilingual environments. [Example](https://www.microsoft.com/en-us/research/blog/speller100-zero-shot-spelling-correction-at-scale-for-100-plus-languages/)

## Datasets

### Training Datasets

Here are the datasets we will use for training our models:

**Unsupervised datasets**

- [Wikipedia Dumps](https://dumps.wikimedia.org/) (regularly updated text dumps of Wikipedia articles);

- [CC-News](https://huggingface.co/datasets/vblagoje/cc_news) (news articles from Common Crawl);

- [BookCorpus](https://huggingface.co/datasets/bookcorpus/bookcorpus) (collection of novels from unpublished authors);

**Supervised datasets**

- [Lang8 Corpus](https://github.com/google-research-datasets/clang8) (sentence pairs extracted from the Lang8 language learning platform);

- [NUCLE](https://cogcomp.seas.upenn.edu/page/resource_view/57) (sentences with annotations of 27 error types, professionally annotated by English instructors);

- [FCE](https://www.kaggle.com/datasets/studentramya/fce-v21) (exam essays with detailed error annotations);

**Evaluation Datasets**

For evaluation, we will manually collect a balanced dataset from all mentioned sources to cover three primary error categories:

- **Punctuation errors** - missing, incorrect, or redundant punctuation marks (commas, periods, apostrophes);

- **Spelling errors** - typos, misspellings, and character-level mistakes within words;

- **Case errors** - incorrect capitalization in proper nouns, sentence beginnings, and acronyms.

The evaluation set will maintain equal representation of each error type to ensure unbiased model performance assessment across all correction categories.

## Baseline Solution

For the baseline solution we will start from N-grams language models that learn statistical distribution of text with no spelling errors. During inference state, we will predict the probability of the next n-gram given previous context, if the probability is less than a threshold, then a correction should be applied.

We will train N-gram models (with different values of N) on a corpus of correctly spelled texts. For correction, we will generate candidate replacements using edit distance algorithms and select the most probable variant based on the learned N-gram distributions.

**For error detection, we will:**

1. Calculate the probability *P(w_i | w_{i-n+1}...w_{i-1})* for each word in the input text;

2. Flag words with probability below an empirically determined threshold T as potential errors.

**For error correction, we will:**

1. Generate candidate replacements for each flagged word by:

    - Considering words with similar edit distance (Levenshtein distance ≤ 2);

    - Including words frequently appearing in similar contexts from the training corpus;

2. Select the most probable word w'i that maximizes *P(w'i | w{i-n+1}...w{i-1})* from the candidates;

3. Replace the flagged word with this most probable candidate.

## Advanced Solution
For production-grade SEC system, we will implement a T5-based approach that treats spelling correction as a text-to-text generation task. This method leverages the power of pre-trained transformer models to understand context and generate corrections.

**Architecture Overview**

We will use the datasets described above to fine-tune T5 for the sequence-to-sequence spelling correction task. The model will learn to map corrupted text inputs to their corrected outputs, leveraging T5's pre-trained knowledge and contextual understanding to handle various error types effectively.

**Key Advantages**

Unlike traditional SEC methods (n-gram language models, rule-based systems, etc.), T5 can leverage bidirectional context to make informed corrections. It handles out-of-vocabulary words gracefully and can adapt to domain-specific terminology through fine-tuning. The model also learns implicit language patterns, enabling it to correct errors that simple edit-distance algorithms would miss.

## Acceptance Criteria
We are planning to achieve:

- 20-25% improvement in F1-score over our baseline solution;

- \>85% correction accuracy on our evaluation set;

- Robust performance on out-of-vocabulary words and domain-specific terminology;

- Demonstration of deep learning advantages for real-world spelling correction tasks.

## Metrics
For evaluating our GEC system, we will use the following metrics:

- **Exact Match (EM)**: Measures the percentage of sentences that are completely corrected to match the reference. This is a strict metric that requires perfect correction, providing a clear but demanding assessment of system performance;

- **Precision**: Measures the proportion of correctly identified and corrected errors out of all errors that the system attempted to correct. High precision indicates that the system makes few incorrect corrections;

- **Recall**: Measures the proportion of actual errors that were correctly identified and corrected out of all actual errors in the text. High recall indicates that the system is able to detect and correct most of the existing errors;

- **F1 Score**: The harmonic mean of precision and recall, providing a balanced measure of the system's performance. It gives an overall view of how well the system identifies and corrects errors, considering both the accuracy of corrections and the ability to find all errors.

## Work distribution
**Stepan Vagin**:

- Unsupervised data pipeline;

- N-gram model training;

- N-gram inference implementation;

- Out-of-vocabulary testing.

**Egor Chernobrovkin**:

- Supervised dataset preparation;

- T5 data preprocessing;

- T5 fine-tuning;

- T5 inference pipeline.

**Lana Ermolaeva**:

- Evaluation dataset curation;

- Metrics framework development;

- Metrics measurement;

- Comparative analysis of all models.

## Literature
- [Typos Correction Training Against Misspellings from Text-to-Text Transformer](https://aclanthology.org/2024.lrec-main.1470.pdf)

- [Unsupervised Context-Sensitive Spelling Correction of English and Dutch Clinical Free-Text with Word and Character N-Gram Embeddings](https://arxiv.org/abs/1710.07045)

- [A Methodology for Generative Spelling Correction via Statistics-Based Spell Corruption](https://aclanthology.org/2024.findings-eacl.10.pdf)

- [*] [Learning-to-Spell: Weak Supervision based Query Correction in E-Commerce Search with Small Strong Labels](https://dl.acm.org/doi/10.1145/3511808.3557113)

- [**] [Effects of spell checkers on English as a second language students’ incidental spelling learning](https://pure.eur.nl/files/47915967/REPUB_98410.pdf)