"""
train_model.py (UPDATED)
=======================
- Trains LDA model
- Compares with Logistic Regression & Random Forest
- Saves metrics + graphs in /metrics folder
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
FEATURES = ["age","sex","cp","trestbps","chol","fbs",
            "restecg","thalach","exang","oldpeak","slope","ca","thal"]
TARGET = "target"

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

MODEL_DIR = "model"
METRICS_DIR = "metrics"


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
def load_data():
    df = pd.read_csv(DATA_URL, header=None, names=FEATURES + [TARGET], na_values="?")
    df[TARGET] = (df[TARGET] > 0).astype(int)
    df.dropna(inplace=True)
    return df


# ─────────────────────────────────────────────
# PREPROCESS
# ─────────────────────────────────────────────
def preprocess(df):
    X = df[FEATURES].values.astype(float)
    y = df[TARGET].values.astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler


# ─────────────────────────────────────────────
# TRAIN MODELS
# ─────────────────────────────────────────────
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    models = {
        "LDA": LinearDiscriminantAnalysis(),
        "Logistic": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100)
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results[name] = {
            "model": model,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "y_pred": y_pred,
            "y_test": y_test,
            "proba": model.predict_proba(X_test)[:, 1]
        }

    return results, X_train, y_train


# ─────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────
def save_confusion_matrix(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(f"{name} Confusion Matrix")
    plt.savefig(f"{METRICS_DIR}/{name}_confusion.png")
    plt.close()


def save_roc(y_true, y_score, name):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
    plt.plot([0,1],[0,1])
    plt.legend()
    plt.title(f"{name} ROC Curve")
    plt.savefig(f"{METRICS_DIR}/{name}_roc.png")
    plt.close()


def save_metrics_bar(results):
    names = list(results.keys())
    acc = [results[n]["accuracy"] for n in names]

    plt.figure()
    plt.bar(names, acc)
    plt.title("Model Accuracy Comparison")
    plt.savefig(f"{METRICS_DIR}/accuracy_comparison.png")
    plt.close()


# ─────────────────────────────────────────────
# VISUALIZATION (LDA)
# ─────────────────────────────────────────────
def make_viz(X, y):
    lda_viz = LinearDiscriminantAnalysis(n_components=1)
    ld1 = lda_viz.fit_transform(X, y)

    pca = PCA(n_components=1)
    pc2 = pca.fit_transform(X)

    coords = np.hstack([ld1, pc2])
    return lda_viz, pca, coords


# ─────────────────────────────────────────────
# SAVE EVERYTHING
# ─────────────────────────────────────────────
def save_all(results, scaler, lda_viz, pca, coords, y):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    # Save main model (LDA)
    joblib.dump(results["LDA"]["model"], f"{MODEL_DIR}/lda_model.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    joblib.dump(lda_viz, f"{MODEL_DIR}/lda_viz.pkl")
    joblib.dump(pca, f"{MODEL_DIR}/pca_viz.pkl")

    np.save(f"{MODEL_DIR}/viz_coords.npy", coords)
    np.save(f"{MODEL_DIR}/viz_labels.npy", y)

    # Save metrics
    for name, res in results.items():
        save_confusion_matrix(res["y_test"], res["y_pred"], name)
        save_roc(res["y_test"], res["proba"], name)

        with open(f"{METRICS_DIR}/{name}_report.txt", "w") as f:
            f.write(classification_report(res["y_test"], res["y_pred"]))

    save_metrics_bar(results)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data()
    X, y, scaler = preprocess(df)

    results, X_train, y_train = train_models(X, y)

    lda_viz, pca, coords = make_viz(X, y)

    save_all(results, scaler, lda_viz, pca, coords, y)

    print("✅ Training + Metrics saved successfully")