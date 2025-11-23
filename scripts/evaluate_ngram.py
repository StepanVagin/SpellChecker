#!/usr/bin/env python3
"""
Evaluation script for N-gram spelling checker
Calculates Exact Match (EM), Precision, Recall, and F1 Score
"""

import os
import sys
import pandas as pd
import json
import ast
from typing import List, Tuple
import re
from pathlib import Path

# Add the src directory to the path (go up one level from scripts/)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spellchecker.models.ngram_model import NGramModel, SpellingChecker


def load_dataset(filepath: str) -> Tuple[List[str], List[str]]:
    """Load dataset and return source and target sentences"""
    print(f"Loading dataset from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Handle different column name formats
    if 'source_text' in df.columns and 'target_text' in df.columns:
        source_col = df['source_text']
        target_col = df['target_text']
    elif 'source_sentence' in df.columns and 'target_sentence' in df.columns:
        source_col = df['source_sentence']
        target_col = df['target_sentence']
    else:
        raise ValueError(f"Dataset must have 'source_text'/'target_text' or 'source_sentence'/'target_sentence' columns. Found: {df.columns.tolist()}")
    
    # Convert to string and handle NaN values
    source_sentences = []
    target_sentences = []
    
    for src, tgt in zip(source_col, target_col):
        # Convert to string, handle NaN/None values
        src_str = str(src) if pd.notna(src) else ""
        tgt_str = str(tgt) if pd.notna(tgt) else ""
        
        # Skip rows where both are empty
        if src_str.strip() or tgt_str.strip():
            source_sentences.append(src_str)
            target_sentences.append(tgt_str)
    
    print(f"Loaded {len(source_sentences)} sentence pairs")
    return source_sentences, target_sentences


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


def preprocess_sentence(sentence: str) -> str:
    """Preprocess sentence for comparison"""
    # Handle None, NaN, or non-string values
    if sentence is None:
        return ""
    
    # Check for NaN (pandas float NaN)
    try:
        import math
        if isinstance(sentence, float) and math.isnan(sentence):
            return ""
    except:
        pass
    
    # Convert to string if not already
    if not isinstance(sentence, str):
        sentence = str(sentence)
    
    # Skip if it's the string "nan"
    if sentence.lower() == "nan":
        return ""
    
    # Convert to lowercase
    sentence = sentence.lower().strip()
    # Remove extra whitespace
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence


def tokenize_sentence(sentence: str) -> List[str]:
    """Tokenize sentence into words, extracting only alphanumeric content"""
    import re
    # Split by whitespace
    tokens = sentence.strip().split()
    # Extract only the alphanumeric part of each token
    words = []
    for token in tokens:
        match = re.search(r'[a-zA-Z0-9]+', token)
        if match:
            words.append(match.group(0).lower())
    return words


def calculate_metrics(source_sentences: List[str], 
                     predicted_sentences: List[str], 
                     target_sentences: List[str]) -> dict:
    """
    Calculate evaluation metrics:
    - Exact Match (EM): Percentage of sentences that are completely corrected
    - Precision: Proportion of correctly corrected errors out of all corrections
    - Recall: Proportion of actual errors that were correctly corrected
    - F1 Score: Harmonic mean of precision and recall
    
    Uses word-level alignment that handles different sentence lengths
    """
    import difflib
    
    exact_matches = 0
    true_positives = 0  # Correctly corrected errors
    false_positives = 0  # Incorrect corrections (changed but still wrong)
    false_negatives = 0  # Missed errors (should have changed but didn't)
    
    total_sentences = len(source_sentences)
    
    for source, predicted, target in zip(source_sentences, predicted_sentences, target_sentences):
        # Preprocess for comparison
        source_proc = preprocess_sentence(source)
        predicted_proc = preprocess_sentence(predicted)
        target_proc = preprocess_sentence(target)
        
        # Check exact match
        if predicted_proc == target_proc:
            exact_matches += 1
        
        # Tokenize for word-level comparison
        source_tokens = tokenize_sentence(source_proc)
        predicted_tokens = tokenize_sentence(predicted_proc)
        target_tokens = tokenize_sentence(target_proc)
        
        # Use sequence matcher for intelligent alignment
        # This handles insertions/deletions better
        matcher = difflib.SequenceMatcher(None, source_tokens, target_tokens)
        
        # Track which positions have errors
        errors_at_source_pos = {}  # Maps source position to target word
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace':
                # Word(s) changed
                for i in range(i1, i2):
                    if i < len(source_tokens) and j1 < len(target_tokens):
                        errors_at_source_pos[source_tokens[i]] = target_tokens[j1] if j1 < len(target_tokens) else None
                        j1 += 1
            elif tag == 'delete':
                # Word(s) deleted in target (grammar fix, not spelling)
                # Not counted as spelling error
                pass
            elif tag == 'insert':
                # Word(s) inserted in target (grammar fix, not spelling)
                # Not counted as spelling error
                pass
        
        # Now compare predicted with source and target
        pred_matcher = difflib.SequenceMatcher(None, source_tokens, predicted_tokens)
        
        corrections_made = {}  # Maps source word to predicted word
        for tag, i1, i2, j1, j2 in pred_matcher.get_opcodes():
            if tag == 'replace':
                for i in range(i1, i2):
                    if i < len(source_tokens) and j1 < len(predicted_tokens):
                        corrections_made[source_tokens[i]] = predicted_tokens[j1]
                        j1 += 1
        
        # Evaluate corrections
        for src_word, tgt_word in errors_at_source_pos.items():
            if src_word in corrections_made:
                pred_word = corrections_made[src_word]
                if pred_word == tgt_word:
                    # Correct correction
                    true_positives += 1
                else:
                    # Incorrect correction attempt
                    false_positives += 1
                    false_negatives += 1
            else:
                # Error not corrected
                false_negatives += 1
        
        # Count unnecessary corrections (false positives)
        for src_word, pred_word in corrections_made.items():
            if src_word not in errors_at_source_pos:
                # Made a change where there was no error
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
        'total_sentences': total_sentences
    }


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


def load_ngram_models(model_dir: str, limit_vocab: int = None) -> Tuple[List[NGramModel], int]:
    """Load trained N-gram models and optionally limit vocabulary for faster evaluation
    
    Args:
        model_dir: Directory containing model files
        limit_vocab: Maximum vocabulary size (None for no limit)
    """
    models = []
    original_vocab_size = 0
    
    # Load 1-gram, 2-gram, and 3-gram models
    for n in [1, 2, 3]:
        model_path = os.path.join(model_dir, f'{n}gram_model.json')
        if os.path.exists(model_path):
            print(f"Loading {n}-gram model from {model_path}...")
            model = NGramModel(n=n)
            model.load_model(model_path)
            
            if n == 1:
                original_vocab_size = len(model.vocabulary)
                
                # Limit vocabulary to most frequent words for faster evaluation (if requested)
                if limit_vocab is not None and len(model.vocabulary) > limit_vocab:
                    print(f"Limiting vocabulary from {len(model.vocabulary)} to {limit_vocab} most common words...")
                    # Get most frequent words based on unigram counts
                    word_counts = [(word, count) for (word,), count in model.ngram_counts.items()]
                    word_counts.sort(key=lambda x: x[1], reverse=True)
                    top_words = set([word for word, _ in word_counts[:limit_vocab]])
                    model.vocabulary = top_words
                else:
                    print(f"Using full vocabulary: {len(model.vocabulary)} words")
            
            models.append(model)
        else:
            print(f"Warning: {n}-gram model not found at {model_path}")
    
    if len(models) == 0:
        print("Error: No n-gram models found!")
    
    return models, original_vocab_size


def correct_sentences(spelling_checker: SpellingChecker, 
                     sentences: List[str],
                     max_samples: int = None) -> List[str]:
    """Correct a list of sentences using the spelling checker"""
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


def save_evaluation_csv(source_sentences: List[str],
                        target_sentences: List[str],
                        predicted_sentences: List[str],
                        output_path: str):
    """
    Save evaluation results to CSV with columns:
    - original: source sentence with errors
    - target: correct target sentence
    - response: predicted/corrected sentence
    - correct: True/False whether prediction matches target
    """
    import csv
    
    # Prepare data
    data = []
    for source, target, predicted in zip(source_sentences, target_sentences, predicted_sentences):
        # Normalize for comparison (lowercase and strip whitespace)
        target_normalized = preprocess_sentence(target)
        predicted_normalized = preprocess_sentence(predicted)
        
        # Check if correct
        is_correct = target_normalized == predicted_normalized
        
        data.append({
            'original': source,
            'target': target,
            'response': predicted,
            'correct': is_correct
        })
    
    # Write to CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['original', 'target', 'response', 'correct']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(data)
    
    print(f"Saved {len(data)} evaluation results to {output_path}")


def evaluate_dataset(dataset_name: str, 
                     source_sentences: List[str],
                     target_sentences: List[str],
                     spelling_checker: SpellingChecker,
                     max_samples: int = None,
                     save_csv: bool = False,
                     csv_path: str = None):
    """Evaluate spelling checker on a dataset"""
    
    print(f"\n{'='*70}")
    print(f"Evaluating on {dataset_name}")
    print(f"{'='*70}")
    
    # Limit samples if specified
    if max_samples is not None:
        source_sentences = source_sentences[:max_samples]
        target_sentences = target_sentences[:max_samples]
    
    # Correct the sentences
    print(f"\nCorrecting {len(source_sentences)} sentences...")
    predicted_sentences = correct_sentences(spelling_checker, source_sentences)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(source_sentences, predicted_sentences, target_sentences)
    
    # Print results
    print(f"\n{dataset_name} Results:")
    print(f"{'-'*70}")
    print(f"Exact Match (EM):      {metrics['exact_match']:.2f}%")
    print(f"Precision:             {metrics['precision']:.2f}%")
    print(f"Recall:                {metrics['recall']:.2f}%")
    print(f"F1 Score:              {metrics['f1_score']:.2f}%")
    print(f"\nDetailed Statistics:")
    print(f"Total sentences:       {metrics['total_sentences']}")
    print(f"Exact matches:         {metrics['exact_matches']}")
    print(f"True positives:        {metrics['true_positives']}")
    print(f"False positives:       {metrics['false_positives']}")
    print(f"False negatives:       {metrics['false_negatives']}")
    
    # Show some examples
    print(f"\nSample corrections (first 5):")
    print(f"{'-'*70}")
    for i in range(min(5, len(source_sentences))):
        print(f"\nExample {i+1}:")
        print(f"Source:    {source_sentences[i][:100]}...")
        print(f"Predicted: {predicted_sentences[i][:100]}...")
        print(f"Target:    {target_sentences[i][:100]}...")
    
    # Save to CSV if requested
    if save_csv and csv_path:
        save_evaluation_csv(
            source_sentences, 
            target_sentences, 
            predicted_sentences, 
            csv_path
        )
        print(f"\nDetailed results saved to {csv_path}")
    
    return metrics


def evaluate_spelling_dataset(dataset_name: str, 
                              source_sentences: List[str],
                              target_sentences: List[str],
                              error_types: List[List[str]],
                              spelling_checker: SpellingChecker,
                              max_samples: int = None,
                              save_csv: bool = False,
                              csv_path: str = None):
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
    
    # Save to CSV if requested
    if save_csv and csv_path:
        save_evaluation_csv(
            filtered_sources, 
            filtered_targets, 
            predicted_sentences, 
            csv_path
        )
        print(f"\nDetailed results saved to {csv_path}")
    
    return metrics


def main():
    # Get project root (go up one level from scripts/)
    project_root = Path(__file__).parent.parent
    base_dir = str(project_root)
    
    # Paths relative to project root
    model_dir = os.path.join(base_dir, 'models', 'ngram')
    test_path = os.path.join(base_dir, 'test.csv')
    
    # Check if test dataset exists
    if not os.path.exists(test_path):
        print(f"Error: Test dataset not found at {test_path}")
        return
    
    # Load N-gram models with full vocabulary (no limit)
    print("Loading N-gram models...")
    ngram_models, original_vocab_size = load_ngram_models(model_dir, limit_vocab=None)
    
    if not ngram_models:
        print("Error: No N-gram models found!")
        return
    
    print(f"Loaded {len(ngram_models)} N-gram models")
    print(f"Original vocabulary size: {original_vocab_size} words")
    
    # Create spelling checker
    print("Using spelling checker...")
    spelling_checker = SpellingChecker(
        ngram_models=ngram_models,
        probability_threshold=0.0001 
    )
    
    print(f"Active vocabulary size: {len(spelling_checker.vocabulary)} words")
    print(f"\nEvaluating on test.csv (full dataset, no sample limit)...\n")
    
    # Load test dataset
    test_source, test_target = load_dataset(test_path)
    
    # Evaluate on test dataset - full evaluation (no sample limit)
    test_csv_path = os.path.join(base_dir, 'test_evaluation_results.csv')
    test_metrics = evaluate_dataset(
        "Test Dataset",
        test_source,
        test_target,
        spelling_checker,
        max_samples=None,  # No limit - evaluate on full dataset
        save_csv=True,
        csv_path=test_csv_path
    )
    
    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Dataset':<35} {'EM':<10} {'Precision':<12} {'Recall':<10} {'F1':<10}")
    print(f"{'-'*75}")
    print(f"{'Test Dataset':<35} "
          f"{test_metrics['exact_match']:>8.2f}% "
          f"{test_metrics['precision']:>10.2f}% "
          f"{test_metrics['recall']:>8.2f}% "
          f"{test_metrics['f1_score']:>8.2f}%")
    
    # Save results to file
    results = {
        'test': test_metrics
    }
    
    results_path = os.path.join(base_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    print(f"Detailed results saved to {test_csv_path}")


if __name__ == '__main__':
    main()

