import os
import time
import uuid
from fastapi import HTTPException

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
except Exception:  # pragma: no cover - optional dependency for local dev
    boto3 = None
    BotoCoreError = ClientError = Exception

def _s3_config():
    bucket = os.getenv("S3_BUCKET_NAME")
    region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    if not (bucket and region and access_key and secret_key):
        return None
    return {
        "bucket": bucket,
        "region": region,
        "access_key": access_key,
        "secret_key": secret_key,
    }

def save_uploaded_image(image: bytes, filename: str) -> str:
    if not (filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg')):
        raise HTTPException(status_code=400, detail="Only .jpg/.jpeg files are allowed")
    s3_cfg = _s3_config()
    if not s3_cfg:
        raise HTTPException(status_code=500, detail="S3 is not configured")
    if boto3 is None:
        raise HTTPException(status_code=500, detail="boto3 is not installed")
    key = f"uploads/{int(time.time())}-{uuid.uuid4().hex}.jpg"
    try:
        client = boto3.client(
            "s3",
            region_name=s3_cfg["region"],
            aws_access_key_id=s3_cfg["access_key"],
            aws_secret_access_key=s3_cfg["secret_key"],
        )
        client.put_object(
            Bucket=s3_cfg["bucket"],
            Key=key,
            Body=image,
            ContentType="image/jpeg",
        )
    except (BotoCoreError, ClientError) as exc:
        raise HTTPException(status_code=500, detail="S3 upload failed") from exc

    return f"https://{s3_cfg['bucket']}.s3.{s3_cfg['region']}.amazonaws.com/{key}"
