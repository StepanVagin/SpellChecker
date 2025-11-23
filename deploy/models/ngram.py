import typing as tp
import os
import sys
from pathlib import Path

# Add the src directory to the path to import the n-gram model
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from spellchecker.models.ngram_model import NGramModel, SpellingChecker


class NGramSpellChecker:
    def __init__(
        self,
        model_dir: str,
        device: str = "cpu",  # Not used for n-gram, kept for API compatibility
        batch_size: int = 1,  # Not used for n-gram, kept for API compatibility
        max_length: int = 128,  # Not used for n-gram, kept for API compatibility
        probability_threshold: float = 0.000001,
    ):
        """
        Initialize the N-gram spelling checker.
        
        Args:
            model_dir: Directory containing the n-gram model JSON files (1gram_model.json, 2gram_model.json, 3gram_model.json)
            device: Not used (kept for API compatibility)
            batch_size: Not used (kept for API compatibility)
            max_length: Not used (kept for API compatibility)
            probability_threshold: Minimum probability threshold for accepting corrections
        """
        print(f"[INFO] Loading n-gram models from {model_dir}")
        
        ngram_models = []
        for n in [1, 2, 3]:
            model_path = os.path.join(model_dir, f'{n}gram_model.json')
            if os.path.exists(model_path):
                print(f"[INFO] Loading {n}-gram model from {model_path}...")
                model = NGramModel(n=n)
                model.load_model(model_path)
                ngram_models.append(model)
            else:
                print(f"[WARNING] {model_path} not found, skipping {n}-gram model")
        
        if not ngram_models:
            raise ValueError(f"No n-gram models found in {model_dir}. Expected files: 1gram_model.json, 2gram_model.json, 3gram_model.json")
        
        self.checker = SpellingChecker(ngram_models, probability_threshold=probability_threshold)
        self.probability_threshold = probability_threshold
        
        print(f"[INFO] Successfully loaded {len(ngram_models)} n-gram models")
        print(f"[INFO] Vocabulary size: {len(self.checker.vocabulary)} words")

    def predict(self, texts: tp.List[str]) -> tp.List[str]:
        """
        Predict corrected text for a list of input texts.
        
        Args:
            texts: List of input texts to correct
            
        Returns:
            List of corrected texts
        """
        corrected_texts = []
        for text in texts:
            corrected_text, _ = self.checker.correct_text(text)
            corrected_texts.append(corrected_text)
        return corrected_texts

