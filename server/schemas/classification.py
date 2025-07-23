from pydantic import BaseModel
from typing import List

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    raw_output: List[List[float]]
    image_url: str
    rebate: float
