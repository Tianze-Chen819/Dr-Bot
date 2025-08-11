# web/app.py
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from package import Model

# Optional: choose a base model (smaller = faster on Mac)
os.environ.setdefault("DRBOT_BASE", "Qwen/Qwen2.5-1.5B-Instruct")

app = FastAPI(title="Dr-Bot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# Load once at startup (first run may take a minute to download weights)
bot = Model()

class Q(BaseModel):
    question: str

@app.post("/api/predict")
def predict(q: Q):
    answer = bot.predict(q.question)
    return {"answer": answer}

# Serve the frontend
app.mount("/", StaticFiles(directory="web/static", html=True), name="static")
