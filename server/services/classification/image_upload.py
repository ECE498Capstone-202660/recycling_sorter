import os
import time
from fastapi import HTTPException

def save_uploaded_image(image: bytes, filename: str) -> str:
    if not (filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg')):
        raise HTTPException(status_code=400, detail="Only .jpg/.jpeg files are allowed")
    data_dir = os.path.join(os.path.dirname(__file__), '../..', 'static')
    os.makedirs(data_dir, exist_ok=True)
    save_name = f'upload-{int(time.time() * 1e9)}.jpg'
    filepath = os.path.join(data_dir, save_name)
    with open(filepath, 'wb') as f:
        f.write(image)
    return filepath 