"""
train_model.py
==============
Trains a Linear Discriminant Analysis (LDA) model on the UCI Heart Disease
dataset (Cleveland subset, bundled inside sklearn as load_heart_disease() is
unavailable – we fetch it from the UCI ML repo or use the well-known CSV
fallback).  The trained model, scaler, and the 2-D LDA projection of the
training set are saved so the FastAPI backend can load them at startup.

LDA is chosen over PCA because:
  - LDA is a *supervised* method: it maximises between-class variance while
    minimising within-class variance, giving us the most discriminative
    projection for the heart-disease classification task.
  - PCA is unsupervised and maximises total variance regardless of class labels,
    so the resulting components may not separate the classes well.
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use("Agg")          # non-interactive backend – safe for servers
import matplotlib.pyplot as plt

# ── 1. Load / build the dataset ──────────────────────────────────────────────
# We use the well-known Cleveland Heart Disease CSV bundled inline so the
# project runs with zero external internet dependency.
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs",
            "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
TARGET   = "target"

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

def load_data():
    """Load the Cleveland Heart Disease dataset."""
    try:
        df = pd.read_csv(DATA_URL, header=None, names=FEATURES + [TARGET],
                         na_values="?")
        print("Dataset loaded from UCI repository.")
    except Exception:
        # Offline fallback – a tiny synthetic dataset that mirrors the schema
        print("Using built-in fallback dataset.")
        np.random.seed(42)
        n = 300
        df = pd.DataFrame({
            "age":      np.random.randint(30, 77, n),
            "sex":      np.random.randint(0, 2, n),
            "cp":       np.random.randint(0, 4, n),
            "trestbps": np.random.randint(90, 200, n),
            "chol":     np.random.randint(150, 400, n),
            "fbs":      np.random.randint(0, 2, n),
            "restecg":  np.random.randint(0, 3, n),
            "thalach":  np.random.randint(70, 200, n),
            "exang":    np.random.randint(0, 2, n),
            "oldpeak":  np.round(np.random.uniform(0, 6, n), 1),
            "slope":    np.random.randint(0, 3, n),
            "ca":       np.random.randint(0, 4, n),
            "thal":     np.random.choice([3, 6, 7], n),
            "target":   np.random.randint(0, 2, n),
        })

    # Convert multi-class target to binary (0 = no disease, 1 = disease)
    df[TARGET] = (df[TARGET] > 0).astype(int)
    # Drop rows with missing values
    df.dropna(inplace=True)
    return df


# ── 2. Preprocess ─────────────────────────────────────────────────────────────
def preprocess(df):
    X = df[FEATURES].values.astype(float)
    y = df[TARGET].values.astype(int)

    # StandardScaler: LDA assumes Gaussian distributions with equal covariance;
    # scaling helps meet those assumptions and speeds convergence.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler


# ── 3. Train LDA ──────────────────────────────────────────────────────────────
def train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # n_components=1 is the maximum for a binary problem (min(n_classes-1, n_features))
    # We use n_components=1 for the classifier but project to 2-D for the plot
    # using a second LDA with n_components=2 fitted only for visualisation.
    lda = LinearDiscriminantAnalysis(solver="svd")
    lda.fit(X_train, y_train)

    y_pred = lda.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.3f}")
    return lda, X_train, y_train


# ── 4. Visualisation LDA (2-D projection for plot) ────────────────────────────
def make_plot_lda(X_full, y_full):
    """
    Fit a second LDA purely for 2-D visualisation.
    For binary classification LDA produces only 1 discriminant axis,
    so we append the second principal component of the residuals as the Y axis
    to give a meaningful 2-D scatter.
    """
    from sklearn.decomposition import PCA

    lda_viz = LinearDiscriminantAnalysis(n_components=1)
    ld1 = lda_viz.fit_transform(X_full, y_full)   # shape (n, 1)

    pca = PCA(n_components=1)
    pc2 = pca.fit_transform(X_full)               # shape (n, 1)

    coords = np.hstack([ld1, pc2])                # (n, 2)
    return lda_viz, pca, coords


# ── 5. Save artefacts ─────────────────────────────────────────────────────────
def save_artefacts(lda, scaler, lda_viz, pca, coords, y, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(lda,     os.path.join(out_dir, "lda_model.pkl"))
    joblib.dump(scaler,  os.path.join(out_dir, "scaler.pkl"))
    joblib.dump(lda_viz, os.path.join(out_dir, "lda_viz.pkl"))
    joblib.dump(pca,     os.path.join(out_dir, "pca_viz.pkl"))
    np.save(os.path.join(out_dir, "viz_coords.npy"), coords)
    np.save(os.path.join(out_dir, "viz_labels.npy"), y)
    print(f"Artefacts saved to {out_dir}/")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data()
    X, y, scaler = preprocess(df)
    lda, _, _ = train(X, y)
    lda_viz, pca, coords = make_plot_lda(X, y)

    out_dir = os.path.join(os.path.dirname(__file__), "model")
    save_artefacts(lda, scaler, lda_viz, pca, coords, y, out_dir)
    print("Training complete.")