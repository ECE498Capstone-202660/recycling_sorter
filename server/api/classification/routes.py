import os
import json
import base64
import datetime as dt
from fastapi import APIRouter, Depends, HTTPException, File, Form, UploadFile, Header
from sqlalchemy.orm import Session
from db.deps import get_db
from models import ClassificationLog, User, KioskLock
from schemas.classification import PredictionResponse, HistoryItem
from services.user.get_user import get_current_user
from services.classification.image_upload import save_uploaded_image
from services.classification.inference import preprocess_image, run_inference_model
from openai import OpenAI

router = APIRouter()
MACHINE_TOKEN = os.getenv("MACHINE_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CATEGORIES = ["Glass", "Metal", "Paper", "Plastic", "Trash"]

_openai_client: OpenAI | None = None

def to_dict(row: ClassificationLog) -> dict:
    return {
        "predicted_class": row.predicted_class,
        "confidence": f"{row.confidence*100:.2f}%",
        "raw_output": row.raw_output,
        "image_url": row.image_url,
        "rebate": row.rebate,
        "created_at": row.created_at.isoformat()
    }


def _require_machine_token(x_machine_token: str | None) -> None:
    if not MACHINE_TOKEN or not x_machine_token or x_machine_token != MACHINE_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid machine token")


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        if not OPENAI_API_KEY:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


def _normalize_label(label: str) -> str:
    for cat in CATEGORIES:
        if label.strip().lower() == cat.lower():
            return cat
    return label.strip().title()


def _openai_classify(image_bytes: bytes, content_type: str) -> tuple[str, float]:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{content_type};base64,{b64}"

    client = _get_openai_client()
    prompt = (
        "Classify the object into exactly one of these five categories:\n"
        "- Glass: glass bottles\n"
        "- Metal: metal bottles/cans\n"
        "- Paper: paper\n"
        "- Plastic: plastic bottles\n"
        "- Trash: any other object that does not belong to the previous four\n"
        "Return ONLY JSON with keys: label, confidence. Use exactly one of these labels: Glass, Metal, Paper, Plastic, Trash.\n"
        "confidence must be a number from 0 to 1."
    )

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                ],
            }
        ],
        response_format={"type": "json_object"},
    )

    raw_text = (resp.choices[0].message.content or "").strip()
    if not raw_text:
        raise ValueError("Empty OpenAI response")
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON from OpenAI: {e}. Raw: {repr(raw_text[:300])}") from e
    label = _normalize_label(payload.get("label", ""))
    if label not in CATEGORIES:
        raise ValueError(f"Invalid label: {label}")
    confidence = float(payload.get("confidence", 0.0))
    return label, confidence

def _get_lock(db: Session) -> KioskLock:
    lock = db.query(KioskLock).first()
    if lock:
        return lock
    lock = KioskLock(is_locked=False)
    db.add(lock)
    db.commit()
    db.refresh(lock)
    return lock


@router.post("/predict")
async def predict(
    image: UploadFile = File(...),
    weight: float = Form(...),
    x_machine_token: str | None = Header(default=None, alias="X-Machine-Token"),
    db: Session = Depends(get_db)
):
    _require_machine_token(x_machine_token)

    lock = _get_lock(db)
    now = dt.datetime.utcnow()
    if lock.is_locked and lock.expires_at and lock.expires_at <= now:
        lock.is_locked = False
        lock.locked_by_user_id = None
        lock.expires_at = None
        db.commit()

    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    img_bytes = await image.read()
    if len(img_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file.")

    weight = weight * 0.054
    tensor = preprocess_image(img_bytes)
    res = run_inference_model(tensor, weight_grams=weight)
    print("CNN result: predicted_class=%s confidence=%s" % (res["predicted_class"], res["confidence"]))

    try:
        gpt_label, gpt_conf = _openai_classify(img_bytes, image.content_type)
        print("OpenAI result: label=%s confidence=%s" % (gpt_label, gpt_conf))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI output error: {exc}")

    cnn_label = _normalize_label(res["predicted_class"])
    use_gpt = cnn_label != gpt_label
    final_label = gpt_label if use_gpt else cnn_label
    final_conf = gpt_conf if use_gpt else float(res["confidence"])
    raw_output = [[final_conf]] if use_gpt else res["raw_output"]

    rebate_by_category = {
        "Glass": 0.10,
        "Metal": 0.10,
        "Paper": 0.10,
        "Plastic": 0.10,
        "Trash": 0.00,
    }
    rebate = rebate_by_category.get(final_label, 0.0)

    if lock.is_locked and lock.locked_by_user_id:
        url = save_uploaded_image(img_bytes, image.filename)
        row = ClassificationLog(
            user_id=lock.locked_by_user_id,
            predicted_class=final_label,
            confidence=final_conf,
            raw_output=raw_output,
            image_url=url,
            rebate=rebate,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return to_dict(row)

    return {
        "predicted_class": final_label,
        "confidence": f"{final_conf*100:.2f}%",
        "raw_output": raw_output,
        "image_url": None,
        "rebate": rebate,
        "created_at": dt.datetime.utcnow().isoformat(),
    }


@router.get("/latest", response_model=list[PredictionResponse])
def latest(
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    user = db.query(User).filter_by(username=current_user).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    rows = (
        db.query(ClassificationLog)
          .filter(ClassificationLog.user_id == user.id)
          .order_by(ClassificationLog.created_at.desc())
          .limit(3)
          .all()
    )
    return [to_dict(r) for r in rows]

@router.get("/history", response_model=list[HistoryItem])
def history(
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    user = db.query(User).filter_by(username=current_user).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    rows = (
        db.query(ClassificationLog)
          .filter(ClassificationLog.user_id == user.id)
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
