from fastapi import APIRouter, Depends, File, Form, UploadFile, Request
from sqlalchemy.orm import Session
from db.deps import get_db
from models import ClassificationLog
from services.classification.image_upload import save_uploaded_image
from services.classification.inference import (
    preprocess_image,
    run_inference_model1,
    run_inference_model2,
)
import random, os

router = APIRouter()
DEV_BASE_URL = "http://localhost:8080"

def to_dict(row: ClassificationLog) -> dict:
    return {
        "predicted_class": row.predicted_class,
        "confidence": f"{row.confidence*100:.2f}%",
        "raw_output": row.raw_output,
        "image_url": row.image_url,
        "rebate": row.rebate
    }

@router.post("/predict/model1")
async def predict_model1(
    request: Request,
    image: UploadFile = File(...),
    weight: float = Form(...),
    db: Session = Depends(get_db)
):
    weight = weight * 0.054
    img_bytes = await image.read()
    tensor = preprocess_image(img_bytes)
    res = run_inference_model1(tensor, weight_grams=weight)

    filename = os.path.basename(save_uploaded_image(img_bytes, image.filename))
    url = f"{DEV_BASE_URL}/static/{filename}"
    rebate = 0.1

    row = ClassificationLog(
        predicted_class=res["predicted_class"],
        confidence=res["confidence"],
        raw_output=res["raw_output"],
        image_url=url,
        rebate=rebate,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return to_dict(row)

@router.post("/predict/model2")
async def predict_model2(
    request: Request,
    image: UploadFile = File(...),
    weight: float = Form(...),
    db: Session = Depends(get_db)
):
    weight = weight * 0.054
    img_bytes = await image.read()
    tensor = preprocess_image(img_bytes)
    res = run_inference_model2(tensor, weight_grams=weight)

    filename = os.path.basename(save_uploaded_image(img_bytes, image.filename))
    url = f"{DEV_BASE_URL}/static/{filename}"
    rebate = 0.1

    row = ClassificationLog(
        predicted_class=res["predicted_class"],
        confidence=res["confidence"],
        raw_output=res["raw_output"],
        image_url=url,
        rebate=rebate
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return to_dict(row)

@router.get("/latest")
def latest(db: Session = Depends(get_db)):
    rows = (
        db.query(ClassificationLog)
          .order_by(ClassificationLog.created_at.desc())
          .limit(3)
          .all()
    )
    return [to_dict(r) for r in rows]

@router.get("/history")
def history(db: Session = Depends(get_db)):
    rows = (
        db.query(ClassificationLog)
          .order_by(ClassificationLog.created_at.desc())
          .all()
    )
    return [
        {
            "id": r.id,
            "category": r.predicted_class,
            "date": r.created_at.strftime("%Y-%m-%d"),
            "rebate": round(r.rebate, 2),
            "image_url": r.image_url,
        }
        for r in rows
    ]
