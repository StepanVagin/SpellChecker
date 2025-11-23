import os
from fastapi import FastAPI, HTTPException
from models.t5 import T5Model
from models.ngram import NGramSpellChecker
from schema import PredictRequest, PredictResponse

app = FastAPI(title="SpellChecker API")

MODEL_REGISTRY = {
    "t5-base-gec": {
        "class": T5Model,
        "path": "../models/gec-t5_small",
        "device": "cpu",
        "batch_size": 1,
        "max_length": 128,
    },
    "ngram": {
        "class": NGramSpellChecker,
        "path": "../models/ngram",
        "device": "cpu",
        "batch_size": 1,
        "max_length": 128,
        "probability_threshold": 0.000001,
    },
}

LOADED_MODELS = {}


def get_models_to_load():
    """Determine which models to load based on environment variable."""
    ngram_only = os.getenv("NGRAM_ONLY", "false").lower() in ("true", "1", "yes")
    if ngram_only:
        return ["ngram"]
    return list(MODEL_REGISTRY.keys())


@app.on_event("startup")
async def preload_models():
    models_to_load = get_models_to_load()
    
    if len(models_to_load) < len(MODEL_REGISTRY):
        print(f"[INFO] Loading only: {', '.join(models_to_load)}")
    
    for name in models_to_load:
        if name not in MODEL_REGISTRY:
            print(f"[WARNING] Model '{name}' not found in registry, skipping")
            continue
            
        cfg = MODEL_REGISTRY[name]
        init_kwargs = {
            "model_dir": cfg["path"],
            "device": cfg["device"],
            "batch_size": cfg["batch_size"],
            "max_length": cfg["max_length"],
        }
        # Add probability_threshold for n-gram model if present
        if "probability_threshold" in cfg:
            init_kwargs["probability_threshold"] = cfg["probability_threshold"]
        LOADED_MODELS[name] = cfg["class"](**init_kwargs)
        print(f"[INFO] Preloaded model '{name}'")


def get_model(name: str):
    if name not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Model '{name}' not found")
    if name not in LOADED_MODELS:
        cfg = MODEL_REGISTRY[name]
        init_kwargs = {
            "model_dir": cfg["path"],
            "device": cfg["device"],
            "batch_size": cfg["batch_size"],
            "max_length": cfg["max_length"],
        }
        # Add probability_threshold for n-gram model if present
        if "probability_threshold" in cfg:
            init_kwargs["probability_threshold"] = cfg["probability_threshold"]
        LOADED_MODELS[name] = cfg["class"](**init_kwargs)
    return LOADED_MODELS[name]


@app.get("/")
async def root():
    return {
        "message": "API is running",
        "available_models": list(LOADED_MODELS.keys()),
        "all_registered_models": list(MODEL_REGISTRY.keys()),
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if not req.texts:
        raise HTTPException(status_code=400, detail="No texts provided")
    model = get_model(req.model_name)
    preds = model.predict(req.texts)
    return PredictResponse(model_name=req.model_name, predictions=preds)
