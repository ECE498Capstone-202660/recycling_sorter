from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from api.auth.routes import router as auth_router
from api.model.routes import router as model_router
from api.rebate.routes import router as rebate_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8081",
        "http://192.168.0.185:8081"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/auth")
app.include_router(model_router, prefix="/model")
app.include_router(rebate_router, prefix="/rebate")

# Serve the data directory as static files at /static
app.mount("/static", StaticFiles(directory="data"), name="static") 