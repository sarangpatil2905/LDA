"""
Heart Disease Classification: LDA vs Logistic Regression vs Random Forest
==========================================================================
Improved pipeline aligned with Isnanto et al., ICENIS 2023
  - Uses MinMaxScaler (paper §2.2) instead of StandardScaler
  - Uses 75/25 train/test split (paper §3) instead of 80/20
  - Exposes 9-feature subset matching paper's Table 1 columns a–i
  - Adds 5-class experiment to match both paper conditions (§4)
  - Fixes precision formula in report (TP/(TP+FP), not (TP+TN)/(TP+FP))
  - Fixes solver/shrinkage=None edge case for eigen solver
  - Separates CV accuracy from test metrics in bar charts
  - Retains all original plots + adds 5-class confusion matrix
"""
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, os, joblib
warnings.filterwarnings("ignore")

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler          # FIX: paper uses Min-Max
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    average_precision_score, classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
from sklearn.metrics import precision_recall_curve

# ─── Palette & Style ─────────────────────────────────────────────────────────
PALETTE = {
    "bg":      "#0D1117", "card":    "#161B22", "border":  "#30363D",
    "accent1": "#58A6FF", "accent2": "#F78166", "accent3": "#3FB950",
    "accent4": "#D2A8FF", "accent5": "#FFA657",
    "text":    "#E6EDF3", "subtext": "#8B949E",
}
MODEL_COLORS = {"LDA": PALETTE["accent1"], "LR": PALETTE["accent2"], "RF": PALETTE["accent3"]}

plt.rcParams.update({
    "figure.facecolor": PALETTE["bg"],  "axes.facecolor":  PALETTE["card"],
    "axes.edgecolor":   PALETTE["border"], "axes.labelcolor": PALETTE["text"],
    "axes.titlecolor":  PALETTE["text"],   "xtick.color":     PALETTE["subtext"],
    "ytick.color":      PALETTE["subtext"],"text.color":      PALETTE["text"],
    "grid.color":       PALETTE["border"], "grid.alpha":      0.5,
    "legend.facecolor": PALETTE["card"],   "legend.edgecolor":PALETTE["border"],
    "font.family": "DejaVu Sans", "font.size": 10,
})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE_DIR, "models")
os.makedirs(OUT, exist_ok=True)

# All 13 Cleveland features
ALL_FEATURES = ["age","sex","cp","trestbps","chol","fbs",
                "restecg","thalach","exang","oldpeak","slope","ca","thal"]

# FIX: Paper Table 1 uses 9 features (columns a–i): age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang
# The paper does NOT use oldpeak, slope, ca, thal (no j–m columns in Table 1)
PAPER_FEATURES = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang"]

FEAT_LABELS = {
    "age":"Age","sex":"Sex","cp":"Chest Pain Type","trestbps":"Resting BP",
    "chol":"Cholesterol","fbs":"Fasting BS","restecg":"Rest ECG",
    "thalach":"Max Heart Rate","exang":"Exercise Angina","oldpeak":"ST Depression",
    "slope":"ST Slope","ca":"Major Vessels","thal":"Thalassemia"
}

# Paper reported baseline (2-class, 75/25 split, 9 features, MinMax scaler)
PAPER_BASELINE = {"acc": 0.8122, "prec": 0.82, "rec": 0.81, "f1": 0.81}


# ─── 1. Load Data ─────────────────────────────────────────────────────────────
def load_data():
    try:
        df = pd.read_csv(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
            header=None, names=ALL_FEATURES+["target"], na_values="?"
        )
        print("  Loaded from UCI.")
    except Exception:
        print("  Using synthetic fallback dataset.")
        np.random.seed(42)
        n = 303
        age      = np.random.normal(54, 9, n).clip(29, 77).astype(int)
        sex      = np.random.binomial(1, 0.68, n)
        cp       = np.random.choice([0,1,2,3], n, p=[0.47,0.17,0.28,0.08])
        trestbps = np.random.normal(131, 17, n).clip(90, 200).astype(int)
        chol     = np.random.normal(246, 51, n).clip(126, 410).astype(int)
        fbs      = np.random.binomial(1, 0.15, n)
        restecg  = np.random.choice([0,1,2], n, p=[0.51,0.48,0.01])
        thalach  = np.random.normal(149, 22, n).clip(70, 202).astype(int)
        exang    = np.random.binomial(1, 0.33, n)
        oldpeak  = np.random.exponential(1.05, n).clip(0, 6.2).round(1)
        slope    = np.random.choice([0,1,2], n, p=[0.07,0.46,0.47])
        ca       = np.random.choice([0,1,2,3], n, p=[0.59,0.22,0.12,0.07])
        thal     = np.random.choice([3,6,7], n, p=[0.55,0.06,0.39])
        risk = (age>55)*0.4 + sex*0.3 + (cp>1)*0.5 + (thalach<140)*0.3 \
             + exang*0.4 + (oldpeak>1)*0.3 + (ca>0)*0.5 + (thal==7)*0.3
        prob = 1/(1+np.exp(-(risk-1.2)))
        target = np.random.binomial(1, prob, n)
        df = pd.DataFrame({
            "age":age,"sex":sex,"cp":cp,"trestbps":trestbps,"chol":chol,
            "fbs":fbs,"restecg":restecg,"thalach":thalach,"exang":exang,
            "oldpeak":oldpeak,"slope":slope,"ca":ca,"thal":thal,"target":target
        })

    # Binary target: 0=no disease, 1=disease (paper condition 1)
    df["target_binary"] = (df["target"] > 0).astype(int)
    # 5-class target: 0–4 (paper condition 2); cap values >4 to 4
    df["target_5class"] = df["target"].clip(0, 4).astype(int)
    df.dropna(inplace=True)
    print(f"  Dataset: {len(df)} samples | Disease rate: {df.target_binary.mean():.1%}")
    return df


# ─── 2. Model Pipelines + Hyperparameter Grids ───────────────────────────────
def get_models_and_grids():
    """
    FIX: Use MinMaxScaler throughout to match paper §2.2.
    FIX: Replace shrinkage=None with shrinkage=0.0 for eigen solver to avoid
         undefined behaviour; SVD (shrinkage=None) silently activates for eigen
         when shrinkage is unset.
    """
    def make_pipeline(clf):
        # MinMaxScaler: paper §2.2 uses x_sc = (x - x_min)/(x_max - x_min)
        return Pipeline([("scaler", MinMaxScaler()), ("clf", clf)])

    models = {
        "LDA": make_pipeline(LinearDiscriminantAnalysis()),
        "LR":  make_pipeline(LogisticRegression(max_iter=2000, random_state=42)),
        "RF":  make_pipeline(RandomForestClassifier(random_state=42)),
    }

    grids = {
        # FIX: shrinkage=None replaced with 0.0 for solvers that need explicit float
        "LDA": [
            {
                "clf__solver": ["lsqr"],
                "clf__shrinkage": ["auto", 0.0, 0.1, 0.2, 0.3, 0.5],
                "clf__tol": [1e-4, 1e-3],
            },
            {
                "clf__solver": ["eigen"],
                "clf__shrinkage": ["auto", 0.0, 0.1, 0.2, 0.3],
                "clf__tol": [1e-4, 1e-3],
            }
        ],
        "LR": {
            "clf__C":       [0.01, 0.1, 0.5, 1, 5, 10],
            "clf__solver":  ["lbfgs", "liblinear"],
            "clf__penalty": ["l2"],
        },
        "RF": {
            "clf__n_estimators":    [100, 200, 300],
            "clf__max_depth":       [None, 5, 10, 15],
            "clf__min_samples_split": [2, 5],
        },
    }
    return models, grids


# ─── 3. Train & Evaluate (binary) ────────────────────────────────────────────
def train_models(X_train, y_train, X_test, y_test):
    models, grids = get_models_and_grids()
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    results = {}

    for name, pipe in models.items():
        print(f"\n  [{name}] Grid search (MinMaxScaler, 75/25 split)...")
        gs = GridSearchCV(pipe, grids[name], cv=cv,
                          scoring="accuracy", n_jobs=-1, verbose=0)
        gs.fit(X_train, y_train)
        best = gs.best_estimator_

        cv_scores = cross_val_score(best, X_train, y_train, cv=cv, scoring="accuracy")

        y_pred  = best.predict(X_test)
        if hasattr(best, "predict_proba"):
            y_proba = best.predict_proba(X_test)[:, 1]
        else:
            y_proba = best.decision_function(X_test)

        # FIX: correct precision = TP/(TP+FP), not (TP+TN)/(TP+FP) (paper typo in eq.6)
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)  # TP/(TP+FP)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        cm   = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc     = auc(fpr, tpr)
        pr_p, pr_r, _ = precision_recall_curve(y_test, y_proba)
        avg_pr      = average_precision_score(y_test, y_proba)

        results[name] = {
            "model": best, "best_params": gs.best_params_,
            "acc": acc, "prec": prec, "rec": rec, "f1": f1,
            "cm": cm, "fpr": fpr, "tpr": tpr, "roc_auc": roc_auc,
            "pr_p": pr_p, "pr_r": pr_r, "avg_pr": avg_pr,
            "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std(),
            "y_pred": y_pred, "y_proba": y_proba,
        }
        delta = acc - PAPER_BASELINE["acc"]
        print(f"    Best params : {gs.best_params_}")
        print(f"    Accuracy    : {acc:.4f} ({delta:+.4f} vs paper)")
        print(f"    ROC AUC     : {roc_auc:.4f}")
        print(f"    CV Accuracy : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return results


# ─── 4. 5-Class Experiment (paper condition 2) ───────────────────────────────
def run_5class_lda(X_train5, y_train5, X_test5, y_test5):
    """
    Replicates the paper's 5-class experiment.
    LDA only; no GridSearch needed (paper reports plain LDA with no tuning detail).
    Uses MinMaxScaler + default SVD solver for 5-class (shrinkage only valid for binary).
    """
    pipe = Pipeline([
        ("scaler", MinMaxScaler()),
        ("lda", LinearDiscriminantAnalysis()),
    ])
    pipe.fit(X_train5, y_train5)
    y_pred5 = pipe.predict(X_test5)
    acc5 = accuracy_score(y_test5, y_pred5)
    cm5  = confusion_matrix(y_test5, y_pred5)
    report5 = classification_report(y_test5, y_pred5,
                  target_names=["No Disease","Stage 1","Stage 2","Stage 3","Stage 4"],
                  zero_division=0)
    print(f"\n  [LDA 5-class] Accuracy: {acc5:.4f} (paper: 59.38%)")
    print(f"  Classification report:\n{report5}")
    return pipe, acc5, cm5, report5


# ─── PLOTS ────────────────────────────────────────────────────────────────────

def plot_confusion_matrices(results, y_test):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=PALETTE["bg"])
    fig.suptitle("Confusion Matrices — LDA vs Logistic Regression vs Random Forest\n"
                 "(MinMaxScaler | 75/25 split | 9 features — aligned with Isnanto et al. 2023)",
                 fontsize=14, fontweight="bold", color=PALETTE["text"], y=1.03)

    for ax, (name, r) in zip(axes, results.items()):
        col = MODEL_COLORS[name]
        cm  = r["cm"]
        sns.heatmap(cm, annot=True, fmt="d", ax=ax,
                    cmap=sns.light_palette(col, as_cmap=True),
                    linewidths=2, linecolor=PALETTE["border"],
                    annot_kws={"size": 22, "weight": "bold"},
                    xticklabels=["Pred: No", "Pred: Yes"],
                    yticklabels=["Act: No",  "Act: Yes"],
                    cbar=False)
        ax.set_title(
            f"{name}\nAcc: {r['acc']:.3f}  |  AUC: {r['roc_auc']:.3f}",
            fontsize=13, fontweight="bold", color=col, pad=10
        )
        tn, fp, fn, tp = cm.ravel()
        for val, label, pos in [
            (tp,"TP",(1.15,1.5)), (tn,"TN",(0.15,0.5)),
            (fp,"FP",(0.15,1.5)), (fn,"FN",(1.15,0.5))
        ]:
            ax.text(pos[0], pos[1], label, ha="center", va="center",
                    fontsize=9, color=PALETTE["subtext"], alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"{OUT}/01_confusion_matrices.png", dpi=150,
                bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print("  ✓ 01_confusion_matrices.png")


def plot_5class_confusion(cm5, acc5):
    """New plot for paper's condition 2 (5-class LDA)."""
    fig, ax = plt.subplots(figsize=(9, 7), facecolor=PALETTE["bg"])
    labels = ["No Disease","Stage 1","Stage 2","Stage 3","Stage 4"]
    sns.heatmap(cm5, annot=True, fmt="d", ax=ax,
                cmap=sns.light_palette(PALETTE["accent4"], as_cmap=True),
                linewidths=1.5, linecolor=PALETTE["border"],
                annot_kws={"size": 14, "weight": "bold"},
                xticklabels=labels, yticklabels=labels, cbar=False)
    ax.set_title(
        f"LDA — 5-Class Confusion Matrix\n"
        f"Acc: {acc5:.3f}  (paper: 59.38%)",
        fontsize=13, fontweight="bold", color=PALETTE["accent4"], pad=10
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{OUT}/07_5class_confusion.png", dpi=150,
                bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print("  ✓ 07_5class_confusion.png")


def plot_5class_accuracy(cm5):
    acc_per_class = cm5.diagonal() / cm5.sum(axis=1)

    labels = ["No Disease","Stage1","Stage2","Stage3","Stage4"]

    plt.figure(figsize=(8,5))
    plt.bar(labels, acc_per_class)
    plt.ylim(0,1)
    plt.title("Per-Class Accuracy (5-Class LDA)")
    plt.ylabel("Accuracy")

    for i, v in enumerate(acc_per_class):
        plt.text(i, v+0.02, f"{v:.2f}", ha="center")

    plt.savefig(f"{OUT}/10_5class_class_accuracy.png", dpi=150)
    plt.close()
    print("  ✓ 10_5class_class_accuracy.png")

def plot_5class_lda_projection(X_scaled, y_5):
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_lda = lda.fit_transform(X_scaled, y_5)

    plt.figure(figsize=(10,7))

    for cls in np.unique(y_5):
        mask = y_5 == cls
        plt.scatter(X_lda[mask,0], X_lda[mask,1], label=f"Class {cls}", alpha=0.6)

    plt.xlabel("LD1")
    plt.ylabel("LD2")
    plt.title("LDA Projection (5-Class)")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.savefig(f"{OUT}/11_5class_lda_projection.png", dpi=150)
    plt.close()
    print("  ✓ 11_5class_lda_projection.png")

def plot_roc_pr_curves(results):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=PALETTE["bg"])
    fig.suptitle("ROC & Precision-Recall Curves",
                 fontsize=15, fontweight="bold", color=PALETTE["text"])

    for name, r in results.items():
        col = MODEL_COLORS[name]
        axes[0].plot(r["fpr"], r["tpr"], color=col, lw=2.5,
                     label=f"{name} (AUC={r['roc_auc']:.3f})", alpha=0.9)
        axes[1].plot(r["pr_r"], r["pr_p"], color=col, lw=2.5,
                     label=f"{name} (AP={r['avg_pr']:.3f})", alpha=0.9)

    axes[0].plot([0,1],[0,1], color=PALETTE["subtext"], lw=1.5,
                 linestyle="--", alpha=0.6, label="Random")
    axes[0].set_xlabel("False Positive Rate", fontsize=12)
    axes[0].set_ylabel("True Positive Rate", fontsize=12)
    axes[0].set_title("ROC Curve", fontsize=14, fontweight="bold")
    axes[0].legend(loc="lower right", fontsize=11)
    axes[0].grid(alpha=0.35)
    axes[0].set_xlim([-0.02, 1.02]); axes[0].set_ylim([-0.02, 1.05])

    axes[1].set_xlabel("Recall", fontsize=12)
    axes[1].set_ylabel("Precision", fontsize=12)
    axes[1].set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    axes[1].legend(loc="upper right", fontsize=11)
    axes[1].grid(alpha=0.35)
    axes[1].set_xlim([-0.02, 1.02]); axes[1].set_ylim([-0.02, 1.05])

    plt.tight_layout()
    plt.savefig(f"{OUT}/02_roc_pr_curves.png", dpi=150,
                bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print("  ✓ 02_roc_pr_curves.png")


def plot_metrics_comparison(results):
    """
    FIX: Separate test metrics from CV accuracy — they measure different things.
    Test metrics (left group) vs CV accuracy (right panel, with error bars).
    """
    test_metrics = ["acc", "prec", "rec", "f1", "roc_auc"]
    test_labels  = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC AUC"]
    model_names  = list(results.keys())
    colors       = [MODEL_COLORS[m] for m in model_names]

    fig = plt.figure(figsize=(20, 10), facecolor=PALETTE["bg"])
    fig.suptitle("Model Performance — Test Metrics vs CV Accuracy\n"
                 "(MinMaxScaler | 75/25 split | 9-feature paper subset)",
                 fontsize=15, fontweight="bold", color=PALETTE["text"], y=1.01)

    # 5 test metric subplots
    for idx, (metric, label) in enumerate(zip(test_metrics, test_labels)):
        ax = fig.add_subplot(2, 3, idx+1)
        vals = [results[m][metric] for m in model_names]
        best_idx = int(np.argmax(vals))
        bars = ax.bar(model_names, vals, color=colors,
                      edgecolor=PALETTE["border"], linewidth=1.5, alpha=0.85, width=0.5)
        bars[best_idx].set_edgecolor("white")
        bars[best_idx].set_linewidth(3)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=12, fontweight="bold", color=PALETTE["text"])

        if metric in PAPER_BASELINE:
            ax.axhline(PAPER_BASELINE[metric], color=PALETTE["accent5"],
                       linestyle="--", lw=1.8, alpha=0.85,
                       label=f"Paper LDA ({PAPER_BASELINE[metric]:.2f})")
            ax.legend(fontsize=9)

        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.set_ylim(0.5, 1.08)
        ax.grid(axis="y", alpha=0.4)
        ax.tick_params(axis="x", labelsize=12)

    # CV accuracy subplot (with error bars) — clearly labelled as training estimate
    ax_cv = fig.add_subplot(2, 3, 6)
    cv_means = [results[m]["cv_mean"] for m in model_names]
    cv_stds  = [results[m]["cv_std"]  for m in model_names]
    bars = ax_cv.bar(model_names, cv_means, color=colors,
                     edgecolor=PALETTE["border"], linewidth=1.5, alpha=0.85, width=0.5)
    ax_cv.errorbar(model_names, cv_means, yerr=cv_stds,
                   fmt="none", color="white", capsize=8, elinewidth=2, capthick=2)
    for bar, val in zip(bars, cv_means):
        ax_cv.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                   f"{val:.3f}", ha="center", va="bottom",
                   fontsize=12, fontweight="bold", color=PALETTE["text"])
    ax_cv.set_title("CV Accuracy (10-fold)\n[training estimate ± std]",
                    fontsize=11, fontweight="bold")
    ax_cv.set_ylim(0.5, 1.08)
    ax_cv.grid(axis="y", alpha=0.4)
    ax_cv.tick_params(axis="x", labelsize=12)
    ax_cv.annotate("Training data only", xy=(0.5, 0.02), xycoords="axes fraction",
                   ha="center", fontsize=9, color=PALETTE["subtext"], style="italic")

    plt.tight_layout()
    plt.savefig(f"{OUT}/03_metrics_comparison.png", dpi=150,
                bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print("  ✓ 03_metrics_comparison.png")

def plot_5class_roc(pipe, X_test, y_test):
    classes = np.unique(y_test)
    y_bin = label_binarize(y_test, classes=classes)

    y_score = pipe.predict_proba(X_test)

    plt.figure(figsize=(10, 7))
    
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"Class {cls} (AUC={roc_auc:.2f})")

    plt.plot([0,1],[0,1],'k--', lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("5-Class ROC Curve (One-vs-Rest)")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.savefig(f"{OUT}/08_5class_roc.png", dpi=150)
    plt.close()
    print("  ✓ 08_5class_roc.png")

def plot_lda_projection(X_scaled, y, results):
    lda_viz = LinearDiscriminantAnalysis(n_components=1)
    ld1 = lda_viz.fit_transform(X_scaled, y)
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(X_scaled)
    coords = np.hstack([ld1, pc1])

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=PALETTE["bg"])
    fig.suptitle("LDA Projection — Discriminant Space Visualization",
                 fontsize=15, fontweight="bold", color=PALETTE["text"])

    for cls, col, lbl, marker in [
        (0, PALETTE["accent3"], "No Disease", "o"),
        (1, PALETTE["accent2"], "Disease",    "^"),
    ]:
        mask = y == cls
        axes[0].scatter(coords[mask,0], coords[mask,1], c=col, alpha=0.55,
                        s=50, label=lbl, marker=marker,
                        edgecolors=PALETTE["border"], linewidths=0.5)

    for cls, col in [(0, PALETTE["accent3"]), (1, PALETTE["accent2"])]:
        mask = y == cls
        xy = coords[mask].T
        if xy.shape[1] > 2:
            try:
                kde = gaussian_kde(xy)
                xg = np.linspace(coords[:,0].min()-0.5, coords[:,0].max()+0.5, 60)
                yg = np.linspace(coords[:,1].min()-0.5, coords[:,1].max()+0.5, 60)
                XX, YY = np.meshgrid(xg, yg)
                Z = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
                axes[0].contour(XX, YY, Z, levels=4, colors=[col],
                                alpha=0.5, linewidths=1.2)
            except Exception:
                pass

    axes[0].set_xlabel("LD1 (Linear Discriminant)", fontsize=11)
    axes[0].set_ylabel("PC2 (Residual Variance)", fontsize=11)
    axes[0].set_title("LDA Scatter Plot", fontsize=13, fontweight="bold")
    axes[0].legend(fontsize=11); axes[0].grid(alpha=0.35)

    x_range = np.linspace(ld1.min(), ld1.max(), 300)
    for cls, col, lbl in [(0, PALETTE["accent3"],"No Disease"),
                           (1, PALETTE["accent2"],"Disease")]:
        mask = y == cls
        data = ld1[mask, 0]
        axes[1].hist(data, bins=25, alpha=0.4, color=col, label=lbl,
                     edgecolor=PALETTE["border"], density=True)
        kde = gaussian_kde(data)
        axes[1].plot(x_range, kde(x_range), color=col, lw=2.5)

    axes[1].set_xlabel("LD1 Score", fontsize=11)
    axes[1].set_ylabel("Density", fontsize=11)
    axes[1].set_title("LD1 Score Distribution by Class", fontsize=13, fontweight="bold")
    axes[1].legend(fontsize=11); axes[1].grid(alpha=0.35)

    plt.tight_layout()
    plt.savefig(f"{OUT}/04_lda_projection.png", dpi=150,
                bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print("  ✓ 04_lda_projection.png")

def plot_5class_pr(pipe, X_test, y_test):
    classes = np.unique(y_test)
    y_bin = label_binarize(y_test, classes=classes)
    y_score = pipe.predict_proba(X_test)

    plt.figure(figsize=(10, 7))

    for i, cls in enumerate(classes):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_score[:, i])
        plt.plot(rec, prec, lw=2, label=f"Class {cls}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("5-Class Precision-Recall Curve")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.savefig(f"{OUT}/09_5class_pr.png", dpi=150)
    plt.close()
    print("  ✓ 09_5class_pr.png")

def plot_lda_hyperparameter_search(X_train, y_train):
    """Visualize LDA shrinkage parameter effect on CV accuracy (MinMaxScaled)."""
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    shrinkage_vals = np.linspace(0.0, 1.0, 40)
    lsqr_scores, eigen_scores = [], []

    scaler = MinMaxScaler()          # FIX: use MinMaxScaler here too
    Xs = scaler.fit_transform(X_train)

    for s in shrinkage_vals:
        for solver, score_list in [("lsqr", lsqr_scores), ("eigen", eigen_scores)]:
            lda = LinearDiscriminantAnalysis(solver=solver, shrinkage=float(s))
            sc = cross_val_score(lda, Xs, y_train, cv=cv, scoring="accuracy")
            score_list.append(sc.mean())

    auto_scores = {}
    for solver in ["lsqr", "eigen"]:
        lda = LinearDiscriminantAnalysis(solver=solver, shrinkage="auto")
        sc = cross_val_score(lda, Xs, y_train, cv=cv, scoring="accuracy")
        auto_scores[solver] = sc.mean()

    fig, ax = plt.subplots(figsize=(14, 6), facecolor=PALETTE["bg"])
    ax.plot(shrinkage_vals, lsqr_scores, color=PALETTE["accent1"], lw=2.5,
            label="solver=lsqr", alpha=0.9)
    ax.plot(shrinkage_vals, eigen_scores, color=PALETTE["accent4"], lw=2.5,
            label="solver=eigen", alpha=0.9, linestyle="--")
    ax.axhline(auto_scores["lsqr"], color=PALETTE["accent1"], lw=1.5,
               linestyle=":", alpha=0.7,
               label=f"lsqr + auto shrinkage ({auto_scores['lsqr']:.3f})")
    ax.axhline(auto_scores["eigen"], color=PALETTE["accent4"], lw=1.5,
               linestyle=":", alpha=0.7,
               label=f"eigen + auto shrinkage ({auto_scores['eigen']:.3f})")
    ax.axhline(PAPER_BASELINE["acc"], color=PALETTE["accent2"], lw=1.8,
               linestyle="-.", alpha=0.8,
               label=f"Paper LDA baseline ({PAPER_BASELINE['acc']:.3f})")

    best_idx_lsqr = int(np.argmax(lsqr_scores))
    ax.scatter(shrinkage_vals[best_idx_lsqr], lsqr_scores[best_idx_lsqr],
               color="white", s=120, zorder=5,
               label=f"Best lsqr: λ={shrinkage_vals[best_idx_lsqr]:.2f}, "
                     f"acc={lsqr_scores[best_idx_lsqr]:.3f}")

    ax.set_xlabel("Shrinkage Parameter (λ)", fontsize=12)
    ax.set_ylabel("10-Fold CV Accuracy", fontsize=12)
    ax.set_title("LDA Hyperparameter Tuning — Shrinkage Regularization\n"
                 "(MinMaxScaler | 75% training data)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(alpha=0.4)
    ax.set_ylim(0.7, 1.0)

    plt.tight_layout()
    plt.savefig(f"{OUT}/05_lda_shrinkage_tuning.png", dpi=150,
                bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print("  ✓ 05_lda_shrinkage_tuning.png")


def plot_summary_heatmap(results):
    metrics = ["acc", "prec", "rec", "f1", "roc_auc", "avg_pr", "cv_mean"]
    labels  = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC AUC",
               "Avg Precision", "CV Accuracy"]
    model_names = list(results.keys())
    data = np.array([[results[m][met] for met in metrics] for m in model_names])

    fig, ax = plt.subplots(figsize=(13, 5), facecolor=PALETTE["bg"])
    fig.suptitle("Model × Metric Summary Heatmap",
                 fontsize=15, fontweight="bold", color=PALETTE["text"])

    im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0.5, vmax=1.0)
    plt.colorbar(im, ax=ax, label="Score", shrink=0.9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=11)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=13, fontweight="bold")

    for i in range(len(model_names)):
        for j in range(len(metrics)):
            val = data[i, j]
            color = "black" if val > 0.75 else "white"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color)

    for j in range(len(metrics)):
        best_i = int(np.argmax(data[:, j]))
        ax.add_patch(plt.Rectangle((j-0.5, best_i-0.5), 1, 1,
                                    fill=False, edgecolor=PALETTE["accent1"],
                                    linewidth=2.5))
    plt.tight_layout()
    plt.savefig(f"{OUT}/06_summary_heatmap.png", dpi=150,
                bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print("  ✓ 06_summary_heatmap.png")


def save_report(results, X_test, y_test, df, acc5, report5):
    lines = []
    lines.append("=" * 68)
    lines.append("HEART DISEASE CLASSIFICATION — LDA vs LR vs RF")
    lines.append("Alignment: Isnanto et al., ICENIS 2023 (E3S Conf 448:02053)")
    lines.append("=" * 68)
    lines.append(f"\nDataset   : {len(df)} samples | Features: 9 (paper subset) | Binary + 5-class")
    lines.append(f"Scaler    : MinMaxScaler (paper §2.2 — NOT StandardScaler)")
    lines.append(f"Split     : 75% train / 25% test (paper §3 — NOT 80/20)")
    lines.append(f"CV        : Stratified 10-fold on training data only")
    lines.append(f"Precision : TP/(TP+FP) — NOTE: paper eq.6 has a typo writing (TP+TN)/(TP+FP)\n")
    lines.append("─" * 68)
    lines.append("PAPER BASELINE (LDA, 9 features, MinMax, 75/25 split):")
    lines.append("  Accuracy: 81.22%  |  Precision: 0.82  |  Recall: 0.81  |  F1: 0.81")
    lines.append("─" * 68 + "\n")

    hdr = f"{'Model':6} | {'Acc':7} | {'Prec':7} | {'Rec':7} | {'F1':7} | {'AUC':7} | {'CV Acc':14} | {'ΔAcc':7}"
    lines.append(hdr)
    lines.append("─" * len(hdr))
    for name, r in results.items():
        delta = r["acc"] - PAPER_BASELINE["acc"]
        lines.append(
            f"{name:6} | {r['acc']:.4f} | {r['prec']:.4f} | {r['rec']:.4f} | "
            f"{r['f1']:.4f} | {r['roc_auc']:.4f} | "
            f"{r['cv_mean']:.4f}±{r['cv_std']:.4f} | {delta:+.4f}"
        )

    lines.append("\n" + "─" * 68)
    lines.append("5-CLASS LDA EXPERIMENT (paper condition 2):")
    lines.append(f"  Accuracy : {acc5:.4f}  (paper: 59.38%)")
    lines.append(f"\n{report5}")

    lines.append("─" * 68)
    best_name = max(results, key=lambda m: results[m]["acc"])
    best = results[best_name]
    lines.append(f"\nBEST MODEL BY ACCURACY: {best_name}")
    lines.append(f"  Accuracy   : {best['acc']:.4f} ({(best['acc']-PAPER_BASELINE['acc'])*100:+.2f}pp vs paper)")
    lines.append(f"  ROC AUC    : {best['roc_auc']:.4f}")
    lines.append(f"  Best Params: {best['best_params']}")

    lines.append("\n" + "─" * 68)
    lines.append("BEST PARAMS PER MODEL:\n")
    for name, r in results.items():
        lines.append(f"  {name}: {r['best_params']}")

    lines.append("\n" + "─" * 68)
    lines.append("DETAILED CLASSIFICATION REPORTS (binary)\n")
    for name, r in results.items():
        lines.append(f"\n── {name} ────────────────────────")
        lines.append(classification_report(y_test, r["y_pred"],
                                            target_names=["No Disease","Disease"]))

    path = f"{OUT}/metrics_report.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("  ✓ metrics_report.txt")
    return path


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*65)
    print("  HEART DISEASE: LDA vs LR vs RF — PAPER-ALIGNED PIPELINE")
    print("  Isnanto et al., ICENIS 2023")
    print("="*65)

    print("\n[1/7] Loading data...")
    df = load_data()

    # FIX: Use paper's 9-feature subset (columns a–i of Table 1)
    X_paper = df[PAPER_FEATURES].values.astype(float)
    y_bin   = df["target_binary"].values.astype(int)
    y_5     = df["target_5class"].values.astype(int)

    # FIX: 75/25 split (paper §3), not 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X_paper, y_bin, test_size=0.25, random_state=42, stratify=y_bin
    )
    counts = Counter(y_5)
    print("5-class distribution:", counts)

    stratify_5 = y_5 if min(counts.values()) >= 2 else None

    X_train5, X_test5, y_train5, y_test5 = train_test_split(
        X_paper, y_5, test_size=0.25, random_state=42, stratify=stratify_5
    )

    print("\n[2/7] LDA shrinkage sweep (MinMaxScaler)...")
    plot_lda_hyperparameter_search(X_train, y_train)

    print("\n[3/7] Training & tuning 3 models (MinMaxScaler, GridSearchCV 10-fold)...")
    results = train_models(X_train, y_train, X_test, y_test)

    print("\n[4/7] 5-class LDA experiment (paper condition 2)...")
    pipe5, acc5, cm5, report5 = run_5class_lda(X_train5, y_train5, X_test5, y_test5)

    print("\n[5/7] Generating plots...")
    plot_confusion_matrices(results, y_test)
    plot_5class_confusion(cm5, acc5)
    plot_roc_pr_curves(results)
    plot_metrics_comparison(results)
    plot_5class_roc(pipe5, X_test5, y_test5)
    plot_5class_pr(pipe5, X_test5, y_test5)
    plot_5class_accuracy(cm5)

    scaler_viz = MinMaxScaler()
    scaler_viz.fit(X_train)
    X_scaled = scaler_viz.transform(X_paper)
    plot_lda_projection(X_scaled, y_bin, results)
    plot_summary_heatmap(results)

    print("\n[6/7] Saving text report...")
    save_report(results, X_test, y_test, df, acc5, report5)

    print("\n[7/7] Saving best model...")
    best_name = max(results, key=lambda m: results[m]["acc"])
    joblib.dump(results[best_name]["model"], f"{OUT}/best_model_{best_name}.pkl")

    # Save LDA artifacts for API use
    lda_pipeline = results["LDA"]["model"]
    joblib.dump(lda_pipeline, f"{OUT}/lda_model.pkl")
    lda_viz_model = LinearDiscriminantAnalysis(n_components=1)
    lda_viz_model.fit(X_scaled, y_bin)
    pca_viz = PCA(n_components=1)
    pca_viz.fit(X_scaled)
    joblib.dump(lda_viz_model, f"{OUT}/lda_viz.pkl")
    joblib.dump(pca_viz, f"{OUT}/pca_viz.pkl")
    np.save(f"{OUT}/viz_coords.npy", np.hstack([
        lda_viz_model.transform(X_scaled),
        pca_viz.transform(X_scaled)
    ]))
    np.save(f"{OUT}/viz_labels.npy", y_bin)
    np.save(f"{OUT}/paper_features.npy", np.array(PAPER_FEATURES))

    print("\n" + "="*65)
    print(f"  DONE. Outputs → {OUT}/")
    print(f"\n  SUMMARY vs Paper (Isnanto et al. 2023):")
    print(f"  {'Model':<6} | {'Acc':>7} | {'ΔAcc':>7} | {'AUC':>7}")
    print(f"  {'Paper LDA':<6} | {PAPER_BASELINE['acc']:>7.4f} | {'baseline':>7} | {'n/a':>7}")
    for name, r in results.items():
        delta = r['acc'] - PAPER_BASELINE['acc']
        print(f"  {name:<6}     | {r['acc']:>7.4f} | {delta:>+7.4f} | {r['roc_auc']:>7.4f}")
    print(f"\n  5-class LDA: {acc5:.4f} (paper: 59.38%)")
    print("="*65)