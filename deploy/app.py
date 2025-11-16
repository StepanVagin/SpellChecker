from fastapi import FastAPI, HTTPException
from models.t5 import T5Model
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
}

LOADED_MODELS = {}


@app.on_event("startup")
async def preload_models():
    for name, cfg in MODEL_REGISTRY.items():
        LOADED_MODELS[name] = cfg["class"](
            model_dir=cfg["path"],
            device=cfg["device"],
            batch_size=cfg["batch_size"],
            max_length=cfg["max_length"],
        )
        print(f"[INFO] Preloaded model '{name}'")


def get_model(name: str) -> T5Model:
    if name not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Model '{name}' not found")
    if name not in LOADED_MODELS:
        cfg = MODEL_REGISTRY[name]
        LOADED_MODELS[name] = cfg["class"](
            model_dir=cfg["path"],
            device=cfg["device"],
            batch_size=cfg["batch_size"],
            max_length=cfg["max_length"],
        )
    return LOADED_MODELS[name]


@app.get("/")
async def root():
    return {
        "message": "API is running",
        "available_models": list(MODEL_REGISTRY.keys()),
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if not req.texts:
        raise HTTPException(status_code=400, detail="No texts provided")
    model = get_model(req.model_name)
    preds = model.predict(req.texts)
    return PredictResponse(model_name=req.model_name, predictions=preds)
