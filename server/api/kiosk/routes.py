import datetime as dt
import os
from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException, Header
from sqlalchemy.orm import Session
from db.deps import get_db
from models import SortingJob, ClassificationLog, User
from services.user.get_user import get_current_user
from services.classification.image_upload import save_uploaded_image
from services.classification.inference import preprocess_image, run_inference_model

router = APIRouter()
MACHINE_TOKEN = os.getenv("MACHINE_TOKEN")
JOB_TIMEOUT_SECONDS = 180


def _require_machine_token(x_machine_token: str | None) -> None:
    if not MACHINE_TOKEN or not x_machine_token or x_machine_token != MACHINE_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid machine token")


@router.post("/machine/detected")
def machine_detected(
    x_machine_token: str | None = Header(default=None, alias="X-Machine-Token"),
    db: Session = Depends(get_db)
):
    _require_machine_token(x_machine_token)

    existing = (
        db.query(SortingJob)
        .filter(
            SortingJob.status.in_(["awaiting_user", "queued", "processing"]),
        )
        .first()
    )
    if existing:
        raise HTTPException(status_code=409, detail="Job already active")

    job = SortingJob(user_id=None, status="awaiting_user")
    db.add(job)
    db.commit()
    db.refresh(job)
    return {"job_id": job.id, "status": job.status}


@router.get("/kiosk/status")
def kiosk_status(
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    user = db.query(User).filter_by(username=current_user).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

    now = dt.datetime.utcnow()
    pending = (
        db.query(SortingJob)
        .filter(SortingJob.status == "awaiting_user", SortingJob.user_id.is_(None))
        .order_by(SortingJob.created_at.asc())
        .first()
    )
    if pending:
        if pending.created_at and (now - pending.created_at).total_seconds() > JOB_TIMEOUT_SECONDS:
            pending.status = "expired"
            db.commit()
        else:
            return {"job_id": pending.id, "status": pending.status}

    user_job = (
        db.query(SortingJob)
        .filter(SortingJob.user_id == user.id)
        .order_by(SortingJob.created_at.desc())
        .first()
    )
    if not user_job or user_job.status == "acknowledged":
        active = (
            db.query(SortingJob)
            .filter(SortingJob.status.in_(["queued", "processing", "done"]))
            .first()
        )
        if active:
            return {"status": "busy"}
        return {"status": "idle"}
    return {"job_id": user_job.id, "status": user_job.status}


@router.post("/kiosk/accept")
def kiosk_accept(
    job_id: int = Form(...),
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    user = db.query(User).filter_by(username=current_user).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

    job = db.query(SortingJob).filter_by(id=job_id).with_for_update().first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    now = dt.datetime.utcnow()
    if job.status != "awaiting_user" or job.user_id is not None:
        raise HTTPException(status_code=409, detail="Job not awaiting user")
    if job.created_at and (now - job.created_at).total_seconds() > JOB_TIMEOUT_SECONDS:
        job.status = "expired"
        db.commit()
        raise HTTPException(status_code=410, detail="Job expired")

    job.user_id = user.id
    job.status = "queued"
    db.commit()
    return {"job_id": job.id, "status": job.status}


@router.post("/kiosk/ack")
def kiosk_ack(
    job_id: int = Form(...),
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    user = db.query(User).filter_by(username=current_user).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

    job = db.query(SortingJob).filter_by(id=job_id, user_id=user.id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "done":
        raise HTTPException(status_code=409, detail="Job not completed")

    job.status = "acknowledged"
    db.commit()
    return {"job_id": job.id, "status": job.status}


@router.get("/machine/next")
def machine_next(
    x_machine_token: str | None = Header(default=None, alias="X-Machine-Token"),
    db: Session = Depends(get_db)
):
    _require_machine_token(x_machine_token)
    job = (
        db.query(SortingJob)
        .filter(SortingJob.status == "queued")
        .order_by(SortingJob.created_at.asc())
        .first()
    )
    if not job:
        return {"job_id": None}
    job.status = "processing"
    job.started_at = dt.datetime.utcnow()
    db.commit()
    return {"job_id": job.id, "user_id": job.user_id, "status": job.status}


@router.post("/machine/result")
async def machine_result(
    job_id: int = Form(...),
    image: UploadFile = File(...),
    weight: float = Form(...),
    x_machine_token: str | None = Header(default=None, alias="X-Machine-Token"),
    db: Session = Depends(get_db)
):
    _require_machine_token(x_machine_token)

    job = db.query(SortingJob).filter_by(id=job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "processing":
        raise HTTPException(status_code=409, detail="Job not in processing state")
    if job.user_id is None:
        raise HTTPException(status_code=409, detail="Job not assigned to user")

    weight = weight * 0.054
    img_bytes = await image.read()
    tensor = preprocess_image(img_bytes)
    res = run_inference_model(tensor, weight_grams=weight)

    url = save_uploaded_image(img_bytes, image.filename)
    rebate = 0.1

    row = ClassificationLog(
        user_id=job.user_id,
        predicted_class=res["predicted_class"],
        confidence=res["confidence"],
        raw_output=res["raw_output"],
        image_url=url,
        rebate=rebate,
    )
    db.add(row)

    job.status = "done"
    job.completed_at = dt.datetime.utcnow()
    db.commit()
    db.refresh(row)
    return {
        "predicted_class": row.predicted_class,
        "confidence": f"{row.confidence*100:.2f}%",
        "raw_output": row.raw_output,
        "image_url": row.image_url,
        "rebate": row.rebate,
        "job_id": job.id,
    }
