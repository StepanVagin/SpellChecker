"""
N-gram language model implementation for spelling correction.
"""

import re
import math
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set
import json
import os
from dataclasses import dataclass

try:
    from nltk.corpus import words as nltk_words
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


@dataclass
class CorrectionResult:
    """Result of a spelling correction attempt"""
    original_word: str
    corrected_word: str
    probability: float
    confidence: float
    edit_distance: int


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for the spelling checker"""
    exact_match: float
    precision: float
    recall: float
    f1_score: float
    total_errors: int
    corrected_errors: int
    false_positives: int


class NGramModel:
    """N-gram language model for spelling correction"""
    
    def __init__(self, n: int = 3):
        self.n = n
        self.ngram_counts = defaultdict(int)
        self.vocabulary = set()
        self.total_ngrams = 0
        self.smoothing_factor = 0.1  # Laplace smoothing
        
    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess text: lowercase, remove punctuation, split into words"""
        # Convert to lowercase and remove extra whitespace
        text = text.lower().strip()
        # Keep only letters, numbers, and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        # Split into words and remove empty strings
        words = [word for word in text.split() if word]
        return words
    
    def train(self, corpus: List[str]):
        """Train the N-gram model on a corpus of clean text"""
        print(f"Training {self.n}-gram model on {len(corpus)} texts...")
        
        for text in corpus:
            words = self.preprocess_text(text)
            if len(words) < self.n:
                continue
                
            # Add words to vocabulary
            self.vocabulary.update(words)
            
            # Generate N-grams
            for i in range(len(words) - self.n + 1):
                ngram = tuple(words[i:i + self.n])
                self.ngram_counts[ngram] += 1
                self.total_ngrams += 1
        
        print(f"Vocabulary size: {len(self.vocabulary)}")
        print(f"Total {self.n}-grams: {self.total_ngrams}")
    
    def get_probability(self, ngram: Tuple[str, ...]) -> float:
        """Calculate probability of an N-gram with Laplace smoothing"""
        if len(ngram) != self.n:
            return 0.0
            
        count = self.ngram_counts[ngram]
        # Laplace smoothing: (count + smoothing_factor) / (total + smoothing_factor * vocabulary_size)
        probability = (count + self.smoothing_factor) / (self.total_ngrams + self.smoothing_factor * len(self.vocabulary))
        return probability
    
    def get_log_probability(self, ngram: Tuple[str, ...]) -> float:
        """Calculate log probability of an N-gram"""
        prob = self.get_probability(ngram)
        return math.log(prob) if prob > 0 else float('-inf')
    
    def get_word_probability(self, word: str, context: List[str] = None) -> float:
        """Get probability of a word given context"""
        if context is None:
            context = []
        
        # For unigram model, just return word probability
        if self.n == 1:
            return self.get_probability((word,))
        
        # For higher-order models, use context
        if len(context) >= self.n - 1:
            ngram = tuple(context[-(self.n-1):] + [word])
        else:
            # Pad with special tokens if context is too short
            padded_context = ['<START>'] * (self.n - 1 - len(context)) + context
            ngram = tuple(padded_context + [word])
        
        return self.get_probability(ngram)
    
    def save_model(self, filepath: str):
        """Save the trained model to a file"""
        # Convert tuple keys to string keys for JSON serialization
        ngram_counts_serializable = {
            '|'.join(ngram): count 
            for ngram, count in self.ngram_counts.items()
        }
        
        model_data = {
            'n': self.n,
            'ngram_counts': ngram_counts_serializable,
            'vocabulary': list(self.vocabulary),
            'total_ngrams': self.total_ngrams,
            'smoothing_factor': self.smoothing_factor
        }
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model from a file"""
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        self.n = model_data['n']
        
        # Convert string keys back to tuple keys
        ngram_counts_dict = {}
        for key, count in model_data['ngram_counts'].items():
            ngram = tuple(key.split('|'))
            ngram_counts_dict[ngram] = count
        
        self.ngram_counts = defaultdict(int, ngram_counts_dict)
        self.vocabulary = set(model_data['vocabulary'])
        self.total_ngrams = model_data['total_ngrams']
        self.smoothing_factor = model_data['smoothing_factor']
        print(f"Model loaded from {filepath}")


class EditDistance:
    """Edit distance algorithms for candidate generation"""
    
    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings"""
        if len(s1) < len(s2):
            return EditDistance.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    @staticmethod
    def generate_edits(word: str) -> Set[str]:
        """Generate all possible single-edit variations of a word"""
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)
    
    @staticmethod
    def generate_candidates(word: str, vocabulary: Set[str], max_distance: int = 2) -> List[Tuple[str, int]]:
        """Generate candidate corrections for a word using fast edits-based approach"""
        candidates = []
        
        # Distance 1: direct edits
        edits1 = EditDistance.generate_edits(word)
        for candidate in edits1:
            if candidate in vocabulary:
                candidates.append((candidate, 1))
        
        # Distance 2: edits of edits (if needed)
        if max_distance >= 2 and len(candidates) < 10:
            edits2 = set()
            for edit1 in edits1:
                edits2.update(EditDistance.generate_edits(edit1))
            for candidate in edits2:
                if candidate in vocabulary and candidate not in [c[0] for c in candidates]:
                    candidates.append((candidate, 2))
        
        # Sort by edit distance (distance 1 first, then distance 2)
        candidates.sort(key=lambda x: x[1])
        return candidates[:50]  # Limit to top 50 candidates for efficiency


class SpellingChecker:
    """Main spelling checker using N-gram models"""
    
    def __init__(self, ngram_models: List[NGramModel], probability_threshold: float = 0.001):
        self.ngram_models = ngram_models
        self.probability_threshold = probability_threshold
        self.vocabulary = set()
        
        # Combine vocabulary from all models
        for model in ngram_models:
            self.vocabulary.update(model.vocabulary)
    
    def check_word(self, word: str, context: List[str] = None) -> CorrectionResult:
        """Check if a word needs correction and suggest the best correction"""
        word_lower = word.lower()
        
        # If word is in vocabulary, it's likely correct - keep it
        if word_lower in self.vocabulary:
            prob = self.ngram_models[0].get_word_probability(word_lower, context)
            return CorrectionResult(
                original_word=word,
                corrected_word=word,
                probability=prob,
                confidence=1.0,
                edit_distance=0
            )
        
        # Word not in vocabulary - generate candidates
        candidates = EditDistance.generate_candidates(word_lower, self.vocabulary, max_distance=2)
        
        if not candidates:
            # No candidates found - keep original
            return CorrectionResult(
                original_word=word,
                corrected_word=word,
                probability=0.0,
                confidence=0.0,
                edit_distance=0
            )
        
        # Score candidates using N-gram models
        best_candidate = None
        best_score = float('-inf')
        
        for candidate_word, edit_dist in candidates:
            # Calculate probability using all available models (average)
            probs = []
            for model in self.ngram_models:
                model_prob = model.get_word_probability(candidate_word, context)
                if model_prob > 0:
                    probs.append(model_prob)
            
            if not probs:
                continue
                
            prob = max(probs)  # Use best probability from any model
            
            # Prefer edit distance 1 over 2
            # Give strong preference to closer matches
            edit_penalty = edit_dist * 0.01 if edit_dist == 1 else edit_dist * 0.05
            score = prob - edit_penalty
            
            if score > best_score:
                best_score = score
                best_candidate = (candidate_word, edit_dist, prob)
        
        # Only accept correction if it's significantly better and above threshold
        if best_candidate and best_candidate[2] > self.probability_threshold:
            # Additional check: prefer edit distance 1 corrections
            if best_candidate[1] == 1 or best_candidate[2] > self.probability_threshold * 2:
                return CorrectionResult(
                    original_word=word,
                    corrected_word=best_candidate[0],
                    probability=best_candidate[2],
                    confidence=min(1.0, best_candidate[2] * 10),
                    edit_distance=best_candidate[1]
                )
        
        # Keep original word if no confident candidate found
        return CorrectionResult(
            original_word=word,
            corrected_word=word,
            probability=0.0,
            confidence=0.0,
            edit_distance=0
        )
    
    def correct_text(self, text: str) -> Tuple[str, List[CorrectionResult]]:
        """Correct spelling errors in a text, preserving punctuation"""
        # Split into tokens, preserving spaces
        tokens = text.split()
        corrected_tokens = []
        corrections = []
        
        for i, token in enumerate(tokens):
            # Separate word from surrounding punctuation
            # Match leading punctuation, word, trailing punctuation
            import re
            match = re.match(r'^([^a-zA-Z0-9]*)([a-zA-Z0-9]+)([^a-zA-Z0-9]*)$', token)
            
            if match:
                prefix, word, suffix = match.groups()
                
                # Get context (previous actual words, not punctuation)
                context_words = []
                for j in range(max(0, i-3), i):
                    prev_match = re.match(r'^[^a-zA-Z0-9]*([a-zA-Z0-9]+)[^a-zA-Z0-9]*$', tokens[j])
                    if prev_match:
                        context_words.append(prev_match.group(1).lower())
                
                # Check and correct the word
                result = self.check_word(word, context_words)
                
                # Preserve original capitalization pattern if word wasn't corrected
                if result.corrected_word != result.original_word:
                    # Apply original capitalization pattern to correction
                    if word.isupper():
                        corrected_word = result.corrected_word.upper()
                    elif word[0].isupper():
                        corrected_word = result.corrected_word.capitalize()
                    else:
                        corrected_word = result.corrected_word
                else:
                    corrected_word = word  # Keep original if no correction
                
                # Reconstruct token with punctuation
                corrected_token = prefix + corrected_word + suffix
                corrected_tokens.append(corrected_token)
                corrections.append(result)
            else:
                # Token is pure punctuation or doesn't match pattern - keep as is
                corrected_tokens.append(token)
                # Create a no-change result
                corrections.append(CorrectionResult(
                    original_word=token,
                    corrected_word=token,
                    probability=1.0,
                    confidence=1.0,
                    edit_distance=0
                ))
        
        corrected_text = ' '.join(corrected_tokens)
        return corrected_text, corrections


class Evaluator:
    """Evaluation metrics for the spelling checker"""
    
    @staticmethod
    def evaluate(reference_texts: List[str], predicted_texts: List[str], 
                original_texts: List[str]) -> EvaluationMetrics:
        """Evaluate spelling checker performance"""
        
        exact_matches = 0
        total_corrections = 0
        correct_corrections = 0
        total_errors = 0
        
        for ref, pred, orig in zip(reference_texts, predicted_texts, original_texts):
            # Exact match
            if ref.strip().lower() == pred.strip().lower():
                exact_matches += 1
            
            # Count errors and corrections
            ref_words = ref.lower().split()
            pred_words = pred.lower().split()
            orig_words = orig.lower().split()
            
            # Count actual errors (words that differ between original and reference)
            for orig_word, ref_word in zip(orig_words, ref_words):
                if orig_word != ref_word:
                    total_errors += 1
            
            # Count corrections made (words that differ between original and prediction)
            for orig_word, pred_word in zip(orig_words, pred_words):
                if orig_word != pred_word:
                    total_corrections += 1
                    # Check if correction is correct
                    if pred_word == ref_word:
                        correct_corrections += 1
        
        # Calculate metrics
        exact_match = exact_matches / len(reference_texts) if reference_texts else 0
        precision = correct_corrections / total_corrections if total_corrections > 0 else 0
        recall = correct_corrections / total_errors if total_errors > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return EvaluationMetrics(
            exact_match=exact_match,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            total_errors=total_errors,
            corrected_errors=correct_corrections,
            false_positives=total_corrections - correct_corrections
        )


def load_training_corpus(filepath: str) -> List[str]:
    """Load training corpus from a text file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        texts = f.readlines()
    return [text.strip() for text in texts if text.strip()]


def load_english_dictionary() -> List[str]:
    """Load a comprehensive English dictionary"""
    print("Loading English dictionary...")
    
    # Try to use NLTK words corpus
    if NLTK_AVAILABLE:
        try:
            # Download words corpus if not already downloaded
            try:
                word_list = nltk_words.words()
            except LookupError:
                print("Downloading NLTK words corpus...")
                nltk.download('words', quiet=True)
                word_list = nltk_words.words()
            
            # Create sentences from words for training
            # Group words into chunks to create context
            chunk_size = 10
            sentences = []
            for i in range(0, len(word_list), chunk_size):
                chunk = word_list[i:i+chunk_size]
                sentences.append(' '.join(chunk))
            
            print(f"Loaded {len(word_list)} words from NLTK corpus")
            return sentences
        except Exception as e:
            print(f"Error: Could not load NLTK words: {e}")
            raise RuntimeError("NLTK words corpus is required but not available. Please install nltk and download the words corpus.")

