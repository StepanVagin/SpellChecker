import typing as tp
from pydantic import BaseModel


class PredictRequest(BaseModel):
    model_name: str
    texts: tp.List[str]


class PredictResponse(BaseModel):
    model_name: str
    predictions: tp.List[str]
