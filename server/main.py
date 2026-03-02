from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from db import Base, engine
from api.auth.routes import router as auth_router
from api.classification.routes import router as classification_router
from api.rebate.routes import router as rebate_router
from api.user.routes import router as user_router

app = FastAPI()

Base.metadata.create_all(bind=engine)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8081",
        "http://192.168.0.185:8081",
        "https://recycling-sorter.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/auth")
app.include_router(classification_router, prefix="/classification")
app.include_router(rebate_router, prefix="/rebate")
app.include_router(user_router, prefix="/user")

app.mount("/static", StaticFiles(directory="static"), name="static")
