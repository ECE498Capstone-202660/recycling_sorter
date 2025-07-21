from fastapi import APIRouter, Form, UploadFile, File
from services.model.image_upload import save_uploaded_image
from services.model.inference import preprocess_image, run_inference
# from fastapi import Depends, HTTPException
# from fastapi.security import OAuth2PasswordBearer
# from services.auth.core import SECRET_KEY, ALGORITHM, user_exists
# from jose import jwt, JWTError
import io

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

@router.post("/upload-image")
# Authentication is bypassed for now. To re-enable, uncomment the lines above and add user: str = Depends(get_current_user)
def upload_image(image: UploadFile = File(...)):
    file_bytes = image.file.read()
    filepath = save_uploaded_image(file_bytes, image.filename)
    return {"message": "Image uploaded and saved successfully", "filepath": filepath}

@router.post("/predict")
async def predict(image: UploadFile = File(...), weight: float = Form(...)):
    image_bytes = await image.read()
    input_tensor = preprocess_image(image_bytes)
    result = run_inference(input_tensor, weight_grams=weight)
    return {
        "predicted_class": result["predicted_class"],
        "confidence": f"{result['confidence']*100:.2f}%",
        "raw_output": result["raw_output"]
    } 