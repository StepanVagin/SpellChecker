import typing as tp
from pydantic import BaseModel
from typing import Optional, List


class PredictRequest(BaseModel):
    model_name: str
    texts: tp.List[str]


class PredictResponse(BaseModel):
    model_name: str
    predictions: tp.List[str]


class CorrectionDetail(BaseModel):
    original: str
    corrected: str
    confidence: float
    edit_distance: int


class CheckRequest(BaseModel):
    text: str
    model: str  # "ngram" or "t5"


class CheckResponse(BaseModel):
    original: str
    corrected: str
    corrections: List[CorrectionDetail]
    num_corrections: int
    model_used: str
    note: Optional[str] = None
