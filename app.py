"""
Simple web interface for testing N-gram spelling checker.

This Flask application provides a user-friendly interface to test the
n-gram spelling correction system trained on unsupervised data.

Run with: python app.py
"""

from flask import Flask, render_template, request, jsonify
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from spellchecker.models.ngram_model import (
    NGramModel,
    SpellingChecker,
)

app = Flask(__name__)

checker = None
model_info = {}


def load_models(model_dir: str = "models/ngram"):
    """Load pre-trained n-gram models"""
    global checker, model_info
    
    print(f"Loading models from {model_dir}...")
    
    ngram_models = []
    for n in [1, 2, 3]:
        model_path = os.path.join(model_dir, f'{n}gram_model.json')
        if os.path.exists(model_path):
            print(f"Loading {n}-gram model...")
            model = NGramModel(n=n)
            model.load_model(model_path)
            ngram_models.append(model)
        else:
            print(f"Warning: {model_path} not found!")
    
    if not ngram_models:
        print("Error: No models loaded!")
        return False
    
    checker = SpellingChecker(ngram_models, probability_threshold=0.000001)
    
    model_info = {
        'num_models': len(ngram_models),
        'vocabulary_size': len(checker.vocabulary),
        'models_loaded': [f"{m.n}-gram" for m in ngram_models]
    }
    
    print(f"Successfully loaded {len(ngram_models)} models")
    print(f"Vocabulary size: {len(checker.vocabulary)} words")
    
    return True


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', model_info=model_info)


@app.route('/check', methods=['POST'])
def check_spelling():
    """Check spelling for submitted text"""
    if not checker:
        return jsonify({
            'error': 'Models not loaded. Please train models first.'
        }), 500
    
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({
            'error': 'No text provided'
        }), 400
    
    corrected_text, corrections = checker.correct_text(text)
    
    corrections_list = []
    for corr in corrections:
        if corr.original_word != corr.corrected_word:
            corrections_list.append({
                'original': corr.original_word,
                'corrected': corr.corrected_word,
                'confidence': round(corr.confidence, 3),
                'probability': round(corr.probability, 6),
                'edit_distance': corr.edit_distance
            })
    
    return jsonify({
        'original': text,
        'corrected': corrected_text,
        'corrections': corrections_list,
        'num_corrections': len(corrections_list)
    })


@app.route('/api/model_info')
def get_model_info():
    """Get information about loaded models"""
    return jsonify(model_info)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='N-gram Spelling Checker Web Interface')
    parser.add_argument('--models', type=str, default='models/ngram',
                       help='Directory containing trained models')
    parser.add_argument('--port', type=int, default=5001,
                       help='Port to run the server on')
    parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode')
    
    args = parser.parse_args()
    
    print("="*60)
    print("N-gram Spelling Checker - Web Interface")
    print("="*60)
    
    if load_models(args.models):
        print("\n" + "="*60)
        print("Starting web server...")
        print("="*60)
        print(f"\nOpen your browser and go to: http://localhost:{args.port}")
        print("\nPress Ctrl+C to stop the server\n")
        
        app.run(debug=args.debug, port=args.port, host='0.0.0.0')
    else:
        print("\n" + "="*60)
        print("Error: Could not load models!")
        print("="*60)
        print("\nPlease train models first using:")
        print("  python scripts/train_ngram_model.py --data data/processed/unsupervised/*.txt --output models/ngram")
        print()
        sys.exit(1)

