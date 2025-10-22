"""
Evaluation script for N-gram spelling checker - SPELLING ERRORS ONLY
Filters CoNLL-14 to only evaluate Mec (Mechanics) errors which include spelling mistakes
Ignores grammar errors completely
"""

import os
import sys
import pandas as pd
import json
import ast
from typing import List, Tuple, Dict, Set
import re

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from spellchecker.models.ngram_model import NGramModel, SpellingChecker


def load_dataset_with_error_types(filepath: str) -> Tuple[List[str], List[str], List[List[str]]]:
    """Load dataset and return source, target sentences, and error types"""
    print(f"Loading dataset from {filepath}...")
    df = pd.read_csv(filepath)
    
    source_sentences = df['source_sentence'].tolist()
    target_sentences = df['target_sentence'].tolist()
    
    # Parse error_types from string representation (if available)
    error_types = []
    if 'error_types' in df.columns:
        for et in df['error_types']:
            try:
                error_types.append(ast.literal_eval(et))
            except:
                error_types.append([])
    else:
        # If no error_types column, assume all are spelling errors (like Birkbeck)
        error_types = [['Spelling'] for _ in range(len(source_sentences))]
    
    print(f"Loaded {len(source_sentences)} sentence pairs")
    return source_sentences, target_sentences, error_types


def filter_spelling_only_sentences(source_sentences: List[str], 
                                   target_sentences: List[str],
                                   error_types: List[List[str]]) -> Tuple[List[str], List[str]]:
    """
    Filter dataset to only include sentences with Mec (Mechanics) errors
    which typically include spelling, punctuation, and capitalization errors
    """
    filtered_sources = []
    filtered_targets = []
    
    spelling_error_types = {'Mec', 'Spelling'}  # Mechanics errors - includes spelling
    
    for source, target, errors in zip(source_sentences, target_sentences, error_types):
        # Check if any error in this sentence is a spelling-related error
        has_spelling_error = any(error in spelling_error_types for error in errors)
        
        if has_spelling_error:
            filtered_sources.append(source)
            filtered_targets.append(target)
    
    print(f"Filtered to {len(filtered_sources)} sentences with spelling/mechanics errors")
    print(f"Removed {len(source_sentences) - len(filtered_sources)} sentences with only grammar errors")
    
    return filtered_sources, filtered_targets


def identify_spelling_changes(source: str, target: str) -> List[Tuple[str, str]]:
    """
    Identify word-level changes that are likely spelling errors (not grammar)
    Returns list of (source_word, target_word) tuples
    """
    import difflib
    
    # Tokenize
    source_words = source.lower().split()
    target_words = target.lower().split()
    
    # Find differences
    matcher = difflib.SequenceMatcher(None, source_words, target_words)
    
    spelling_changes = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace' and (i2 - i1) == (j2 - j1) == 1:
            # Single word replacement - check if it's a spelling correction
            src_word = source_words[i1]
            tgt_word = target_words[j1]
            
            # Remove punctuation for comparison
            src_clean = re.sub(r'[^\w]', '', src_word)
            tgt_clean = re.sub(r'[^\w]', '', tgt_word)
            
            if src_clean and tgt_clean:
                # Calculate similarity - high similarity suggests spelling error
                similarity = difflib.SequenceMatcher(None, src_clean, tgt_clean).ratio()
                
                # If words are similar (>50% match), likely a spelling correction
                # or if source word is clearly misspelled (not in common patterns)
                if similarity > 0.5:
                    spelling_changes.append((src_clean, tgt_clean))
    
    return spelling_changes


def preprocess_sentence(sentence: str) -> str:
    """Preprocess sentence for comparison"""
    sentence = sentence.lower().strip()
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence


def tokenize_sentence(sentence: str) -> List[str]:
    """Tokenize sentence into words, extracting only alphanumeric content"""
    tokens = sentence.strip().split()
    words = []
    for token in tokens:
        match = re.search(r'[a-zA-Z0-9]+', token)
        if match:
            words.append(match.group(0).lower())
    return words


def calculate_spelling_metrics(source_sentences: List[str], 
                               predicted_sentences: List[str], 
                               target_sentences: List[str]) -> dict:
    """
    Calculate evaluation metrics focusing ONLY on spelling corrections
    Uses word-level similarity to identify spelling errors vs grammar errors
    """
    import difflib
    
    exact_matches = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    total_sentences = len(source_sentences)
    total_spelling_errors = 0
    
    for source, predicted, target in zip(source_sentences, predicted_sentences, target_sentences):
        # Preprocess
        source_proc = preprocess_sentence(source)
        predicted_proc = preprocess_sentence(predicted)
        target_proc = preprocess_sentence(target)
        
        # Check exact match
        if predicted_proc == target_proc:
            exact_matches += 1
        
        # Tokenize
        source_tokens = tokenize_sentence(source_proc)
        predicted_tokens = tokenize_sentence(predicted_proc)
        target_tokens = tokenize_sentence(target_proc)
        
        # Find spelling errors in source by comparing with target
        source_target_matcher = difflib.SequenceMatcher(None, source_tokens, target_tokens)
        
        spelling_errors = {}  # Maps source word to target correction
        
        for tag, i1, i2, j1, j2 in source_target_matcher.get_opcodes():
            if tag == 'replace':
                # Only count as spelling error if it's a single word change with high similarity
                if (i2 - i1) == 1 and (j2 - j1) == 1:
                    src_word = source_tokens[i1] if i1 < len(source_tokens) else None
                    tgt_word = target_tokens[j1] if j1 < len(target_tokens) else None
                    
                    if src_word and tgt_word:
                        # Calculate similarity
                        similarity = difflib.SequenceMatcher(None, src_word, tgt_word).ratio()
                        
                        # High similarity = likely spelling error
                        # Low similarity = likely wrong word choice (grammar)
                        if similarity > 0.4:  # Threshold for spelling similarity
                            spelling_errors[src_word] = tgt_word
                            total_spelling_errors += 1
        
        # Find corrections made by spell checker
        source_pred_matcher = difflib.SequenceMatcher(None, source_tokens, predicted_tokens)
        
        corrections_made = {}
        
        for tag, i1, i2, j1, j2 in source_pred_matcher.get_opcodes():
            if tag == 'replace':
                if (i2 - i1) == 1 and (j2 - j1) == 1:
                    src_word = source_tokens[i1] if i1 < len(source_tokens) else None
                    pred_word = predicted_tokens[j1] if j1 < len(predicted_tokens) else None
                    
                    if src_word and pred_word:
                        corrections_made[src_word] = pred_word
        
        # Evaluate spelling corrections
        for src_word, tgt_word in spelling_errors.items():
            if src_word in corrections_made:
                pred_word = corrections_made[src_word]
                if pred_word == tgt_word:
                    # Correct correction
                    true_positives += 1
                else:
                    # Incorrect correction
                    false_positives += 1
                    false_negatives += 1
            else:
                # Missed error
                false_negatives += 1
        
        # Count unnecessary corrections (only if they look like spelling changes)
        for src_word, pred_word in corrections_made.items():
            if src_word not in spelling_errors:
                # Check if this looks like a spelling correction attempt
                similarity = difflib.SequenceMatcher(None, src_word, pred_word).ratio()
                if similarity > 0.4:  # Looks like a spelling change
                    false_positives += 1
    
    # Calculate metrics
    exact_match = (exact_matches / total_sentences * 100) if total_sentences > 0 else 0
    precision = (true_positives / (true_positives + false_positives) * 100) if (true_positives + false_positives) > 0 else 0
    recall = (true_positives / (true_positives + false_negatives) * 100) if (true_positives + false_negatives) > 0 else 0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    
    return {
        'exact_match': exact_match,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'exact_matches': exact_matches,
        'total_sentences': total_sentences,
        'total_spelling_errors': total_spelling_errors
    }


def load_ngram_models(model_dir: str, limit_vocab: int = 10000) -> Tuple[List[NGramModel], int]:
    """Load trained N-gram models"""
    models = []
    original_vocab_size = 0
    
    for n in [1, 2, 3]:
        model_path = os.path.join(model_dir, f'{n}gram_model.json')
        if os.path.exists(model_path):
            print(f"Loading {n}-gram model from {model_path}...")
            model = NGramModel(n=n)
            model.load_model(model_path)
            
            if n == 1:
                original_vocab_size = len(model.vocabulary)
                
                if limit_vocab and len(model.vocabulary) > limit_vocab:
                    print(f"Limiting vocabulary from {len(model.vocabulary)} to {limit_vocab} most common words...")
                    word_counts = [(word, count) for (word,), count in model.ngram_counts.items()]
                    word_counts.sort(key=lambda x: x[1], reverse=True)
                    top_words = set([word for word, _ in word_counts[:limit_vocab]])
                    model.vocabulary = top_words
            
            models.append(model)
        else:
            print(f"Warning: {n}-gram model not found at {model_path}")
    
    return models, original_vocab_size


def correct_sentences(spelling_checker: SpellingChecker, 
                     sentences: List[str],
                     max_samples: int = None) -> List[str]:
    """Correct a list of sentences"""
    corrected = []
    total = len(sentences) if max_samples is None else min(max_samples, len(sentences))
    
    print(f"Correcting sentences...")
    for i, sentence in enumerate(sentences[:total]):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{total} sentences ({(i+1)/total*100:.1f}%)")
        
        corrected_text, _ = spelling_checker.correct_text(sentence)
        corrected.append(corrected_text)
    
    print(f"  Completed: {total}/{total} sentences (100.0%)")
    return corrected


def evaluate_spelling_dataset(dataset_name: str, 
                              source_sentences: List[str],
                              target_sentences: List[str],
                              error_types: List[List[str]],
                              spelling_checker: SpellingChecker,
                              max_samples: int = None):
    """Evaluate spelling checker on dataset - SPELLING ERRORS ONLY"""
    
    print(f"\n{'='*70}")
    print(f"Evaluating on {dataset_name} - SPELLING ERRORS ONLY")
    print(f"{'='*70}")
    
    # Filter to only sentences with spelling/mechanics errors
    filtered_sources, filtered_targets = filter_spelling_only_sentences(
        source_sentences, target_sentences, error_types
    )
    
    if len(filtered_sources) == 0:
        print("No spelling errors found in this dataset!")
        return None
    
    # Limit samples if specified
    if max_samples is not None:
        filtered_sources = filtered_sources[:max_samples]
        filtered_targets = filtered_targets[:max_samples]
    
    # Correct the sentences
    print(f"\nCorrecting {len(filtered_sources)} sentences...")
    predicted_sentences = correct_sentences(spelling_checker, filtered_sources)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_spelling_metrics(filtered_sources, predicted_sentences, filtered_targets)
    
    # Print results
    print(f"\n{dataset_name} Results (SPELLING ONLY):")
    print(f"{'-'*70}")
    print(f"Exact Match (EM):          {metrics['exact_match']:.2f}%")
    print(f"Precision:                 {metrics['precision']:.2f}%")
    print(f"Recall:                    {metrics['recall']:.2f}%")
    print(f"F1 Score:                  {metrics['f1_score']:.2f}%")
    print(f"\nDetailed Statistics:")
    print(f"Total sentences:           {metrics['total_sentences']}")
    print(f"Total spelling errors:     {metrics['total_spelling_errors']}")
    print(f"Exact matches:             {metrics['exact_matches']}")
    print(f"True positives:            {metrics['true_positives']}")
    print(f"False positives:           {metrics['false_positives']}")
    print(f"False negatives:           {metrics['false_negatives']}")
    
    # Show examples
    print(f"\nSample corrections (first 5):")
    print(f"{'-'*70}")
    for i in range(min(5, len(filtered_sources))):
        print(f"\nExample {i+1}:")
        print(f"Source:    {filtered_sources[i]}")
        print(f"Predicted: {predicted_sentences[i]}")
        print(f"Target:    {filtered_targets[i]}")
    
    return metrics


def main():
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data', 'processed')
    model_dir = os.path.join(base_dir, 'models', 'ngram')
    
    # Dataset paths
    conll_path = os.path.join(data_dir, 'conll_nucle_sft.csv')
    birkbeck_path = os.path.join(data_dir, 'birkbeck_spelling_corpus.csv')
    
    # Check if datasets exist
    if not os.path.exists(conll_path):
        print(f"Error: Dataset not found at {conll_path}")
        return
    
    # Load N-gram models
    print("Loading N-gram models...")
    ngram_models, original_vocab_size = load_ngram_models(model_dir, limit_vocab=20000)
    
    if not ngram_models:
        print("Error: No N-gram models found!")
        return
    
    print(f"Loaded {len(ngram_models)} N-gram models")
    print(f"Original vocabulary size: {original_vocab_size} words")
    
    # Create spelling checker
    spelling_checker = SpellingChecker(
        ngram_models=ngram_models,
        probability_threshold=0.0001
    )
    
    print(f"Active vocabulary size: {len(spelling_checker.vocabulary)} words")
    
    # Load CoNLL-14 dataset with error types
    print("\n" + "="*70)
    print("LOADING CoNLL-14 DATASET")
    print("="*70)
    conll_source, conll_target, conll_error_types = load_dataset_with_error_types(conll_path)
    
    # Analyze error types
    from collections import Counter
    all_error_types = []
    for errors in conll_error_types:
        all_error_types.extend(errors)
    
    error_counter = Counter(all_error_types)
    print(f"\nError type distribution:")
    print(f"  Mec (Mechanics - spelling/punct): {error_counter.get('Mec', 0)}")
    print(f"  Grammar errors: {sum(count for etype, count in error_counter.items() if etype != 'Mec')}")
    print(f"  Total errors: {sum(error_counter.values())}")
    
    # Evaluate on CoNLL-14 (spelling errors only)
    conll_metrics = evaluate_spelling_dataset(
        "CoNLL-14 (NUCLE)",
        conll_source,
        conll_target,
        conll_error_types,
        spelling_checker,
        max_samples=None  # Use all spelling error sentences
    )
    
    # Also evaluate on Birkbeck (which is all spelling errors)
    if os.path.exists(birkbeck_path):
        print("\n" + "="*70)
        print("LOADING BIRKBECK DATASET")
        print("="*70)
        birkbeck_source, birkbeck_target, birkbeck_error_types = load_dataset_with_error_types(birkbeck_path)
        
        birkbeck_metrics = evaluate_spelling_dataset(
            "Birkbeck Spelling Corpus",
            birkbeck_source,
            birkbeck_target,
            birkbeck_error_types,
            spelling_checker,
            max_samples=100
        )
    else:
        birkbeck_metrics = None
        print("\nBirkbeck dataset not found, skipping...")
    
    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY - SPELLING ERRORS ONLY")
    print(f"{'='*70}")
    print(f"\n{'Dataset':<30} {'EM':<10} {'Precision':<12} {'Recall':<10} {'F1':<10}")
    print(f"{'-'*70}")
    
    if conll_metrics:
        print(f"{'CoNLL-14 (Spelling Only)':<30} "
              f"{conll_metrics['exact_match']:>8.2f}% "
              f"{conll_metrics['precision']:>10.2f}% "
              f"{conll_metrics['recall']:>8.2f}% "
              f"{conll_metrics['f1_score']:>8.2f}%")
    
    if birkbeck_metrics:
        print(f"{'Birkbeck Spelling Corpus':<30} "
              f"{birkbeck_metrics['exact_match']:>8.2f}% "
              f"{birkbeck_metrics['precision']:>10.2f}% "
              f"{birkbeck_metrics['recall']:>8.2f}% "
              f"{birkbeck_metrics['f1_score']:>8.2f}%")
    
    # Save results
    results = {
        'conll_spelling_only': conll_metrics,
        'birkbeck': birkbeck_metrics
    }
    
    results_path = os.path.join(base_dir, 'evaluation_results_spelling_only.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()

