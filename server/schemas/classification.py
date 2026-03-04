from pydantic import BaseModel
from typing import List

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: str
    raw_output: List[List[float]]
    image_url: str
    rebate: float
    created_at: str


class HistoryItem(BaseModel):
    id: int
    category: str
    date: str
    rebate: float
    image_url: str
