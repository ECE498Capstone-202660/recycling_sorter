import base64
import random
from fastapi import APIRouter, Form, UploadFile, File, Request, WebSocket, WebSocketDisconnect
from services.model.image_upload import save_uploaded_image
from services.model.inference import preprocess_image, run_inference
# from fastapi import Depends, HTTPException
# from fastapi.security import OAuth2PasswordBearer
# from services.auth.core import SECRET_KEY, ALGORITHM, user_exists
# from jose import jwt, JWTError
import io
from typing import Optional, List
import os
import json

router = APIRouter()
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")
# def get_current_user(token: str = Depends(oauth2_scheme)):
#     try:
#         payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
#         username: str = payload.get("sub")
#         if username is None or not user_exists(username):
#             raise HTTPException(status_code=401, detail="Invalid authentication credentials")
#         return username
#     except JWTError:
#         raise HTTPException(status_code=401, detail="Invalid authentication credentials")

# In-memory store for the latest result (legacy, keep for reference)
LATEST_RESULT: Optional[dict] = None
# Store the latest 3 results for WebSocket broadcast
LATEST_RESULTS: List[dict] = []
active_connections: List[WebSocket] = []

# Set your dev base URL here. Change to your local IP if needed.
DEV_BASE_URL = "http://localhost:8080"  # Or e.g. "http://192.168.x.x:8080"

async def broadcast_latest_results():
    data = json.dumps(LATEST_RESULTS)
    for ws in active_connections:
        try:
            await ws.send_text(data)
        except Exception:
            pass  # Ignore failed sends

@router.websocket("/ws/latest-results")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        # Send the current latest results on connect
        await websocket.send_text(json.dumps(LATEST_RESULTS))
        while True:
            await websocket.receive_text()  # Keep alive
    except WebSocketDisconnect:
        active_connections.remove(websocket)

@router.post("/upload-image")
# Authentication is bypassed for now. To re-enable, uncomment the lines above and add user: str = Depends(get_current_user)
def upload_image(image: UploadFile = File(...)):
    file_bytes = image.file.read()
    filepath = save_uploaded_image(file_bytes, image.filename)
    return {"message": "Image uploaded and saved successfully", "filepath": filepath}

@router.post("/predict")
async def predict(request: Request, image: UploadFile = File(...), weight: float = Form(...)):
    global LATEST_RESULT, LATEST_RESULTS
    image_bytes = await image.read()
    input_tensor = preprocess_image(image_bytes)
    result = run_inference(input_tensor, weight_grams=weight)
    # Save image and get filename
    filepath = save_uploaded_image(image_bytes, image.filename)
    filename = os.path.basename(filepath)
    # Use DEV_BASE_URL for dev; switch to request.base_url for prod
    image_url = f"{DEV_BASE_URL}/static/{filename}"
    rebate = round(random.uniform(0.1, 1.0), 2)
    new_result = {
        "predicted_class": result["predicted_class"],
        "confidence": f"{result['confidence']*100:.2f}%",
        "raw_output": result["raw_output"],
        "image_url": image_url,
        "rebate": rebate
    }
    LATEST_RESULT = new_result  # Keep legacy single result for /latest-result
    LATEST_RESULTS.insert(0, new_result)
    if len(LATEST_RESULTS) > 3:
        LATEST_RESULTS = LATEST_RESULTS[:3]
    await broadcast_latest_results()
    return new_result

@router.get("/latest-result")
def latest_result():
    if LATEST_RESULT is None:
        return {"detail": "No result available yet."}
    return LATEST_RESULT 