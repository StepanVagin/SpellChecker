"""
Unified FastAPI application for multi-model spelling checker.

This application provides both a web interface and API endpoints for
spelling correction using N-gram and T5 models.

Run with: python app.py
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from models.t5 import T5Model
from models.ngram import NGramSpellChecker
from schema import (
    PredictRequest,
    PredictResponse,
    CheckRequest,
    CheckResponse,
    CorrectionDetail,
)

app = FastAPI(title="SpellChecker API", description="Multi-model spelling checker with web interface")

# Setup templates - adjust path for deploy directory
templates_dir = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# Model registry
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "t5-base-gec": {
        "class": T5Model,
        "path": "../models/gec-t5_small",
        "device": "cpu",
        "batch_size": 1,
        "max_length": 128,
        "type": "t5",
    },
    "ngram": {
        "class": NGramSpellChecker,
        "path": "../models/ngram",
        "probability_threshold": 0.000001,
        "type": "ngram",
    },
}

LOADED_MODELS: Dict[str, Any] = {}
MODEL_INFO: Dict[str, Dict[str, Any]] = {}


def load_ngram_model(model_name: str = "ngram") -> bool:
    """Load N-gram models"""
    global MODEL_INFO
    
    if model_name not in MODEL_REGISTRY:
        MODEL_INFO["ngram"] = {
            "name": "N-gram",
            "available": False,
            "error": "Model not in registry"
        }
        return False
    
    cfg = MODEL_REGISTRY[model_name]
    # Handle relative paths from deploy directory
    if cfg["path"].startswith("../"):
        model_path = Path(__file__).parent.parent / cfg["path"][3:]
    else:
        model_path = Path(cfg["path"])
    
    if not model_path.exists():
        MODEL_INFO["ngram"] = {
            "name": "N-gram",
            "available": False,
            "error": f"Model directory not found: {model_path}"
        }
        return False
    
    try:
        checker = cfg["class"](
            model_dir=str(model_path),
            probability_threshold=cfg.get("probability_threshold", 0.000001)
        )
        LOADED_MODELS[model_name] = checker
        
        MODEL_INFO["ngram"] = {
            "name": "N-gram",
            "num_models": checker.models_loaded,
            "vocabulary_size": checker.vocabulary_size,
            "models_loaded": checker.models_loaded,
            "available": True
        }
        
        print(f"[INFO] Successfully loaded N-gram model")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to load N-gram model: {e}")
        MODEL_INFO["ngram"] = {
            "name": "N-gram",
            "available": False,
            "error": str(e)
        }
        return False


def load_t5_model(model_name: str = "t5-base-gec") -> bool:
    """Load T5 model"""
    global MODEL_INFO
    
    if model_name not in MODEL_REGISTRY:
        MODEL_INFO["t5"] = {
            "name": "T5",
            "available": False,
            "error": "Model not in registry"
        }
        return False
    
    cfg = MODEL_REGISTRY[model_name]
    # Handle relative paths from deploy directory
    if cfg["path"].startswith("../"):
        model_path = Path(__file__).parent.parent / cfg["path"][3:]
    else:
        model_path = Path(cfg["path"])
    
    if not model_path.exists():
        MODEL_INFO["t5"] = {
            "name": "T5",
            "available": False,
            "error": f"Model directory not found: {model_path}"
        }
        return False
    
    try:
        model = cfg["class"](
            model_dir=str(model_path),
            device=cfg["device"],
            batch_size=cfg["batch_size"],
            max_length=cfg["max_length"],
        )
        LOADED_MODELS[model_name] = model
        
        MODEL_INFO["t5"] = {
            "name": "T5",
            "model_type": "Transformer",
            "available": True
        }
        
        print(f"[INFO] Successfully loaded T5 model")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to load T5 model: {e}")
        MODEL_INFO["t5"] = {
            "name": "T5",
            "available": False,
            "error": str(e)
        }
        return False


def get_models_to_load():
    """Determine which models to load based on environment variable."""
    ngram_only = os.getenv("NGRAM_ONLY", "false").lower() in ("true", "1", "yes")
    if ngram_only:
        return ["ngram"]
    return list(MODEL_REGISTRY.keys())


@app.on_event("startup")
async def preload_models():
    """Preload models on startup"""
    print("="*60)
    print("Multi-Model Spelling Checker - FastAPI")
    print("="*60)
    
    models_to_load = get_models_to_load()
    
    # Check for skip-t5 flag (handled via environment variable or command line)
    # This is handled by get_models_to_load() via NGRAM_ONLY env var
    # Command line skip-t5 is handled in __main__ before startup
    
    if len(models_to_load) < len(MODEL_REGISTRY):
        print(f"[INFO] Loading only: {', '.join(models_to_load)}")
    
    # Load N-gram models
    ngram_loaded = False
    if "ngram" in models_to_load:
        ngram_loaded = load_ngram_model()
    
    # Load T5 model (optional, will fail gracefully if not available)
    t5_loaded = False
    if "t5-base-gec" in models_to_load:
        t5_loaded = load_t5_model()
    
    print("\n" + "="*60)
    print("Model Status:")
    print(f"  N-gram: {'✓' if ngram_loaded else '✗'}")
    print(f"  T5: {'✓' if t5_loaded else '✗'}")
    print("="*60)
    
    if not ngram_loaded and not t5_loaded:
        print("\n[WARNING] No models loaded! The API will return errors.")


def get_model(model_name: str):
    """Get a loaded model by name, lazy loading if not already loaded"""
    if model_name not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found in registry. Available models: {list(MODEL_REGISTRY.keys())}"
        )
    
    # Lazy load if not already loaded
    if model_name not in LOADED_MODELS:
        cfg = MODEL_REGISTRY[model_name]
        # Handle relative paths from deploy directory
        if cfg["path"].startswith("../"):
            model_path = Path(__file__).parent.parent / cfg["path"][3:]
        else:
            model_path = Path(cfg["path"])
        
        if not model_path.exists():
            raise HTTPException(
                status_code=500,
                detail=f"Model directory not found: {model_path}"
            )
        
        try:
            if cfg.get("type") == "ngram":
                checker = cfg["class"](
                    model_dir=str(model_path),
                    probability_threshold=cfg.get("probability_threshold", 0.000001)
                )
                LOADED_MODELS[model_name] = checker
                MODEL_INFO["ngram"] = {
                    "name": "N-gram",
                    "num_models": checker.models_loaded,
                    "vocabulary_size": checker.vocabulary_size,
                    "models_loaded": checker.models_loaded,
                    "available": True
                }
            else:  # T5 model
                model = cfg["class"](
                    model_dir=str(model_path),
                    device=cfg["device"],
                    batch_size=cfg["batch_size"],
                    max_length=cfg["max_length"],
                )
                LOADED_MODELS[model_name] = model
                MODEL_INFO["t5"] = {
                    "name": "T5",
                    "model_type": "Transformer",
                    "available": True
                }
            print(f"[INFO] Lazy loaded model '{model_name}'")
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load model '{model_name}': {str(e)}"
            )
    
    return LOADED_MODELS[model_name]


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the web interface"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "model_info": MODEL_INFO
    })


@app.get("/api/")
async def api_root():
    """API root endpoint"""
    return {
        "message": "API is running",
        "available_models": list(LOADED_MODELS.keys()),
        "all_registered_models": list(MODEL_REGISTRY.keys()),
    }


@app.get("/api/model_info")
async def get_model_info():
    """Get information about loaded models"""
    return MODEL_INFO


@app.post("/check", response_model=CheckResponse)
async def check_spelling(req: CheckRequest):
    """Check spelling for submitted text (web interface endpoint)"""
    if not req.text:
        raise HTTPException(status_code=400, detail="No text provided")
    
    # Route to appropriate model
    if req.model == "ngram":
        if "ngram" not in LOADED_MODELS:
            raise HTTPException(
                status_code=500,
                detail="N-gram models not loaded. Please train models first."
            )
        
        checker = LOADED_MODELS["ngram"]
        corrected_text, corrections_list = checker.predict_with_details(req.text)
        
        corrections = [
            CorrectionDetail(**corr) for corr in corrections_list
        ]
        
        return CheckResponse(
            original=req.text,
            corrected=corrected_text,
            corrections=corrections,
            num_corrections=len(corrections),
            model_used="N-gram"
        )
    
    elif req.model == "t5":
        if "t5-base-gec" not in LOADED_MODELS:
            raise HTTPException(
                status_code=500,
                detail="T5 model not loaded. Please check model availability."
            )
        
        model = LOADED_MODELS["t5-base-gec"]
        predictions = model.predict([req.text])
        corrected_text = predictions[0] if predictions else req.text
        
        return CheckResponse(
            original=req.text,
            corrected=corrected_text,
            corrections=[],
            num_corrections=0 if req.text == corrected_text else 1,
            model_used="T5",
            note="T5 model provides sentence-level corrections without word-level details"
        )
    
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model type: {req.model}. Use 'ngram' or 't5'"
        )


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """Batch prediction endpoint (API endpoint)"""
    if not req.texts:
        raise HTTPException(status_code=400, detail="No texts provided")
    
    model = get_model(req.model_name)
    predictions = model.predict(req.texts)
    
    return PredictResponse(
        model_name=req.model_name,
        predictions=predictions
    )


if __name__ == "__main__":
    import uvicorn
    
    parser = argparse.ArgumentParser(description="Multi-Model Spelling Checker Web Interface")
    parser.add_argument(
        "--ngram-models",
        "--models",
        type=str,
        default="../models/ngram",
        help="Directory containing trained N-gram models"
    )
    parser.add_argument(
        "--t5-model",
        type=str,
        default="../models/gec-t5_small",
        help="Directory containing trained T5 model"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5001,
        help="Port to run the server on"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to"
    )
    parser.add_argument(
        "--skip-t5",
        action="store_true",
        help="Skip loading T5 model"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    args = parser.parse_args()
    
    # Handle skip-t5 flag
    if args.skip_t5:
        os.environ["NGRAM_ONLY"] = "true"
    
    # Update model paths in registry
    MODEL_REGISTRY["ngram"]["path"] = args.ngram_models
    MODEL_REGISTRY["t5-base-gec"]["path"] = args.t5_model
    
    print(f"\nStarting server on http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop\n")
    
    uvicorn.run(
        "app:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )
