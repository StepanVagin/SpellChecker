"""
Detailed error analysis for N-gram spelling checker
Shows specific examples of errors, misses, and corrections
"""

import os
import sys
import pandas as pd
import re
from typing import List, Tuple
import difflib

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from spellchecker.models.ngram_model import NGramModel, SpellingChecker


def highlight_differences(text1: str, text2: str, label1: str = "Text1", label2: str = "Text2"):
    """Highlight differences between two texts"""
    words1 = text1.split()
    words2 = text2.split()
    
    matcher = difflib.SequenceMatcher(None, words1, words2)
    
    result1 = []
    result2 = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            result1.extend(words1[i1:i2])
            result2.extend(words2[j1:j2])
        elif tag == 'replace':
            result1.extend([f"[{w}]" for w in words1[i1:i2]])
            result2.extend([f"[{w}]" for w in words2[j1:j2]])
        elif tag == 'delete':
            result1.extend([f"[{w}]" for w in words1[i1:i2]])
        elif tag == 'insert':
            result2.extend([f"[{w}]" for w in words2[j1:j2]])
    
    print(f"  {label1:12} {' '.join(result1)}")
    print(f"  {label2:12} {' '.join(result2)}")


def extract_word_changes(source: str, predicted: str, target: str):
    """Extract word-level changes"""
    import re
    
    # Extract alphanumeric words
    def extract_words(text):
        return [w.lower() for w in re.findall(r'\b[a-z]+\b', text.lower())]
    
    src_words = extract_words(source)
    pred_words = extract_words(predicted)
    tgt_words = extract_words(target)
    
    changes = {
        'errors_in_source': [],  # Words that need correction
        'corrections_made': [],  # Corrections attempted
        'correct_fixes': [],     # Successful corrections
        'incorrect_fixes': [],   # Wrong corrections
        'missed_errors': []      # Errors not fixed
    }
    
    # Find errors in source
    matcher_src_tgt = difflib.SequenceMatcher(None, src_words, tgt_words)
    for tag, i1, i2, j1, j2 in matcher_src_tgt.get_opcodes():
        if tag == 'replace' and (i2 - i1) == 1 and (j2 - j1) == 1:
            src_word = src_words[i1]
            tgt_word = tgt_words[j1]
            if src_word != tgt_word:
                changes['errors_in_source'].append((src_word, tgt_word))
    
    # Find corrections made
    matcher_src_pred = difflib.SequenceMatcher(None, src_words, pred_words)
    for tag, i1, i2, j1, j2 in matcher_src_pred.get_opcodes():
        if tag == 'replace' and (i2 - i1) == 1 and (j2 - j1) == 1:
            src_word = src_words[i1]
            pred_word = pred_words[j1]
            if src_word != pred_word:
                changes['corrections_made'].append((src_word, pred_word))
    
    # Classify corrections
    for src_word, pred_word in changes['corrections_made']:
        # Check if this was an actual error
        was_error = False
        correct_target = None
        for err_src, err_tgt in changes['errors_in_source']:
            if err_src == src_word:
                was_error = True
                correct_target = err_tgt
                break
        
        if was_error:
            if pred_word == correct_target:
                changes['correct_fixes'].append((src_word, pred_word, correct_target))
            else:
                changes['incorrect_fixes'].append((src_word, pred_word, correct_target))
        else:
            # Changed a word that wasn't an error
            changes['incorrect_fixes'].append((src_word, pred_word, 'NO_ERROR'))
    
    # Find missed errors
    for src_word, tgt_word in changes['errors_in_source']:
        was_corrected = False
        for corr_src, corr_pred in changes['corrections_made']:
            if corr_src == src_word:
                was_corrected = True
                break
        if not was_corrected:
            changes['missed_errors'].append((src_word, tgt_word))
    
    return changes


def main():
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data', 'processed')
    model_dir = os.path.join(base_dir, 'models', 'ngram')
    
    print("="*80)
    print("N-GRAM SPELLING CHECKER - DETAILED ERROR ANALYSIS")
    print("="*80)
    
    # Load models
    print("\nLoading models...")
    models = []
    for n in [1, 2, 3]:
        model = NGramModel(n=n)
        model.load_model(os.path.join(model_dir, f'{n}gram_model.json'))
        models.append(model)
    
    # Limit vocabulary for speed
    print("Limiting vocabulary to 20,000 most common words...")
    word_counts = [(word, count) for (word,), count in models[0].ngram_counts.items()]
    word_counts.sort(key=lambda x: x[1], reverse=True)
    top_words = set([word for word, _ in word_counts[:20000]])
    for model in models:
        model.vocabulary = top_words
    
    checker = SpellingChecker(models, probability_threshold=0.0001)
    print(f"Vocabulary size: {len(checker.vocabulary)} words\n")
    
    # Load datasets
    print("Loading datasets...")
    conll_df = pd.read_csv(os.path.join(data_dir, 'conll_nucle_sft.csv'))
    birkbeck_df = pd.read_csv(os.path.join(data_dir, 'birkbeck_spelling_corpus.csv'))
    
    # Analyze Birkbeck (pure spelling errors)
    print("\n" + "="*80)
    print("BIRKBECK SPELLING CORPUS - ERROR EXAMPLES")
    print("="*80)
    
    correct_fixes = []
    incorrect_fixes = []
    missed_errors = []
    
    # Analyze first 30 examples
    for idx in range(min(30, len(birkbeck_df))):
        source = str(birkbeck_df.iloc[idx]['source_sentence'])
        target = str(birkbeck_df.iloc[idx]['target_sentence'])
        
        predicted, _ = checker.correct_text(source)
        
        changes = extract_word_changes(source, predicted, target)
        
        if changes['correct_fixes']:
            correct_fixes.append((idx, source, predicted, target, changes))
        if changes['incorrect_fixes']:
            incorrect_fixes.append((idx, source, predicted, target, changes))
        if changes['missed_errors']:
            missed_errors.append((idx, source, predicted, target, changes))
    
    # Show examples
    print(f"\n✅ CORRECT FIXES (First 10 examples)")
    print("-"*80)
    for i, (idx, source, predicted, target, changes) in enumerate(correct_fixes[:10], 1):
        print(f"\nExample {i} (Sentence #{idx}):")
        for src, pred, tgt in changes['correct_fixes']:
            print(f"  ✓ Fixed: '{src}' → '{pred}' (target: '{tgt}')")
        highlight_differences(source[:100], predicted[:100], "Source", "Corrected")
        print()
    
    print(f"\n❌ INCORRECT FIXES (First 10 examples)")
    print("-"*80)
    for i, (idx, source, predicted, target, changes) in enumerate(incorrect_fixes[:10], 1):
        print(f"\nExample {i} (Sentence #{idx}):")
        for src, pred, tgt in changes['incorrect_fixes']:
            if tgt == 'NO_ERROR':
                print(f"  ✗ Wrong: Changed '{src}' → '{pred}' (but '{src}' was correct!)")
            else:
                print(f"  ✗ Wrong: Changed '{src}' → '{pred}' (should be '{tgt}')")
        highlight_differences(source[:100], predicted[:100], "Source", "Corrected")
        print(f"  {'Target':12} {target[:100]}")
        print()
    
    print(f"\n⚠️  MISSED ERRORS (First 20 examples)")
    print("-"*80)
    for i, (idx, source, predicted, target, changes) in enumerate(missed_errors[:20], 1):
        print(f"\nExample {i} (Sentence #{idx}):")
        for src, tgt in changes['missed_errors'][:3]:  # Show first 3 missed errors per sentence
            print(f"  ⚠️  Missed: '{src}' should be '{tgt}'")
        print(f"  Source:   {source[:100]}")
        print(f"  Target:   {target[:100]}")
        print()
    
    # Statistics
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    print(f"\nBirkbeck Corpus (30 sentences analyzed):")
    print(f"  Sentences with correct fixes:   {len(correct_fixes)}")
    print(f"  Sentences with incorrect fixes: {len(incorrect_fixes)}")
    print(f"  Sentences with missed errors:   {len(missed_errors)}")
    
    # Total counts
    total_correct = sum(len(c[4]['correct_fixes']) for c in correct_fixes)
    total_incorrect = sum(len(c[4]['incorrect_fixes']) for c in incorrect_fixes)
    total_missed = sum(len(c[4]['missed_errors']) for c in missed_errors)
    
    print(f"\nWord-level statistics:")
    print(f"  Correct corrections:   {total_correct}")
    print(f"  Incorrect corrections: {total_incorrect}")
    print(f"  Missed errors:         {total_missed}")
    
    if total_correct + total_incorrect > 0:
        precision = total_correct / (total_correct + total_incorrect) * 100
        print(f"\nPrecision: {precision:.2f}%")
    
    if total_correct + total_missed > 0:
        recall = total_correct / (total_correct + total_missed) * 100
        print(f"Recall:    {recall:.2f}%")
    
    # Analyze CoNLL
    print("\n" + "="*80)
    print("CoNLL-14 - ERROR EXAMPLES")
    print("="*80)
    
    conll_missed = []
    
    for idx in range(min(20, len(conll_df))):
        source = str(conll_df.iloc[idx]['source_sentence'])
        target = str(conll_df.iloc[idx]['target_sentence'])
        
        predicted, _ = checker.correct_text(source)
        
        changes = extract_word_changes(source, predicted, target)
        
        if changes['missed_errors']:
            conll_missed.append((idx, source, predicted, target, changes))
    
    print(f"\n⚠️  MISSED ERRORS IN CoNLL-14 (First 15 examples)")
    print("-"*80)
    print("Note: Most errors in CoNLL-14 are GRAMMAR errors, not spelling errors.\n")
    
    for i, (idx, source, predicted, target, changes) in enumerate(conll_missed[:15], 1):
        print(f"\nExample {i} (Sentence #{idx}):")
        for src, tgt in changes['missed_errors'][:2]:  # Show first 2 missed per sentence
            # Check if it's likely a spelling error vs grammar error
            if len(src) > 2 and len(tgt) > 2:
                # Calculate edit distance
                from difflib import SequenceMatcher
                similarity = SequenceMatcher(None, src, tgt).ratio()
                if similarity > 0.5:  # Likely spelling
                    print(f"  ⚠️  SPELLING: '{src}' should be '{tgt}'")
                else:  # Likely grammar
                    print(f"  ℹ️  GRAMMAR: '{src}' should be '{tgt}' (word choice/grammar)")
        print(f"  Source: {source[:90]}...")
        print()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()

