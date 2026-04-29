"""
main.py  –  FastAPI Backend for Heart Disease Prediction
=========================================================

Endpoints
---------
POST /predict          → run LDA prediction + return plot as base64 PNG
GET  /health           → liveness check

Run
---
uvicorn main:app --reload --port 8000
"""

import os
import io
import base64
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Heart Disease Prediction API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

def _load(filename):
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        raise RuntimeError(f"Missing model file: {filename}")
    return joblib.load(path)

_model = _lda_viz = _pca_viz = None
_viz_coords = _viz_labels = None

def get_models():
    global _model, _lda_viz, _pca_viz, _viz_coords, _viz_labels

    if _model is None:
        _model = _load("lda_model.pkl")
        _lda_viz = _load("lda_viz.pkl")
        _pca_viz = _load("pca_viz.pkl")

        _viz_coords = np.load(os.path.join(MODEL_DIR, "viz_coords.npy"))
        _viz_labels = np.load(os.path.join(MODEL_DIR, "viz_labels.npy"))

    return _model, _lda_viz, _pca_viz, _viz_coords, _viz_labels

# ─────────────────────────────────────────────────────────────
# Request / Response Models (FIXED FOR PYDANTIC V2)
# ─────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    age: float = Field(..., ge=20, le=100)
    sex: int = Field(..., ge=0, le=1)
    cp: int = Field(..., ge=0, le=3)
    trestbps: float = Field(..., ge=80, le=250)
    chol: float = Field(..., ge=100, le=600)
    fbs: int = Field(..., ge=0, le=1)
    restecg: int = Field(..., ge=0, le=2)
    thalach: float = Field(..., ge=60, le=220)
    exang: int = Field(..., ge=0, le=1)

class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    plot_base64: str


# ─────────────────────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────────────────────
_FEATURE_DEFAULTS = {
    "age": 55,
    "sex": 1,
    "cp": 0,
    "trestbps": 130,
    "chol": 246,
    "fbs": 0,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
}
_FEATURE_ORDER = [
    "age","sex","cp","trestbps","chol",
    "fbs","restecg","thalach","exang"
]


def build_feature_vector(req: PredictRequest) -> np.ndarray:
    vals = {
        "age": req.age,
        "sex": req.sex,
        "cp": req.cp,
        "trestbps": req.trestbps,
        "chol": req.chol,
        "fbs": req.fbs,
        "restecg": req.restecg,
        "thalach": req.thalach,
        "exang": req.exang,
    }

    return np.array([vals[f] for f in _FEATURE_ORDER], dtype=float).reshape(1, -1)


# ─────────────────────────────────────────────────────────────
# Plot Generator
# ─────────────────────────────────────────────────────────────
def make_plot(coords, labels, user_coord):
    fig, ax = plt.subplots(figsize=(6, 4.5), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")

    colors = {0: "#22c55e", 1: "#ef4444"}
    names = {0: "No Disease", 1: "Heart Disease"}

    for cls in [0, 1]:
        mask = labels == cls
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=colors[cls], label=names[cls],
            alpha=0.6, s=20
        )

    # User point
    ax.scatter(
        user_coord[0, 0], user_coord[0, 1],
        c="#facc15", s=200,
        edgecolors="white", linewidths=1.5,
        marker="*", label="You"
    )

    ax.set_title("LDA Class Separation", color="white")
    ax.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    return base64.b64encode(buf.read()).decode()


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Heart Disease API running 🚀"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        model, lda_viz, pca_viz, coords, labels = get_models()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Feature vector
    X = build_feature_vector(req)

    # Prediction (pipeline handles scaling)
    pred = int(model.predict(X)[0])
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
    else:
        score = model.decision_function(X)
        proba = [1 - score, score]

    confidence = float(round(proba[pred], 4))
    label = "Heart Disease" if pred == 1 else "No Heart Disease"

    # Visualization
    scaler = model.named_steps["scaler"]
    X_scaled = scaler.transform(X)

    ld1 = lda_viz.transform(X_scaled)
    pc2 = pca_viz.transform(X_scaled)
    user_coord = np.hstack([ld1, pc2])

    plot_b64 = make_plot(coords, labels, user_coord)

    return PredictResponse(
        prediction=label,
        confidence=confidence,
        plot_base64=plot_b64
    )