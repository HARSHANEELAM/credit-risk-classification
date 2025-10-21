import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.inspection import permutation_importance

TARGET = "risk_label"
DATA_PATH = "data/credit_risk_sample.csv"
MODEL_PATH = "src/model_pipeline.joblib"
IMAGES_DIR = "images"

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def plot_roc_curve(y_true_bin, y_proba, save_path: str):
    # y_true_bin must be 0/1 (we set 1 = "bad")
    fpr, tpr, _ = roc_curve(y_true_bin, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC curve")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()

def plot_permutation_importance(model, X, y_true_bin, save_path: str, top_n: int = 15):
    # Get feature names after preprocessing if available
    try:
        prep = model.named_steps["prep"]
        ct = prep.named_steps["preprocess"]
        feature_names = ct.get_feature_names_out()
    except Exception:
        feature_names = np.array([f"f_{i}" for i in range(X.shape[1])])

    r = permutation_importance(
        model, X, y_true_bin, scoring="roc_auc", n_repeats=5, random_state=42, n_jobs=-1
    )

    importances = r.importances_mean
    idx = np.argsort(importances)[-top_n:]
    top_feats = feature_names[idx]
    top_vals = importances[idx]

    order = np.argsort(top_vals)
    top_feats = top_feats[order]
    top_vals = top_vals[order]

    plt.figure(figsize=(8, 6))
    plt.barh(range(len(top_vals)), top_vals)
    plt.yticks(range(len(top_vals)), top_feats)
    plt.xlabel("Permutation Importance (AUC drop)")
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()

def main():
    ensure_dir(IMAGES_DIR)

    # Load data
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET])
    y_str = df[TARGET]                    # "bad" / "good"
    y_bin = (y_str == "bad").astype(int)  # 1 = bad (positive class), 0 = good

    # Load trained model
    model = joblib.load(MODEL_PATH)

    # Predictions
    proba = model.predict_proba(X)[:, 1]  # probability of "bad"
    preds = model.predict(X)

    # Text metrics
    print("ROC-AUC:", roc_auc_score(y_bin, proba))
    print("\nClassification Report:\n", classification_report(y_str, preds))
    print("\nConfusion Matrix:\n", confusion_matrix(y_str, preds, labels=["bad", "good"]))

    # Charts
    plot_roc_curve(y_bin, proba, os.path.join(IMAGES_DIR, "roc_curve.png"))
    plot_permutation_importance(model, X, y_bin, os.path.join(IMAGES_DIR, "feature_importance.png"))

    print("\nSaved charts to images/roc_curve.png and images/feature_importance.png")

if __name__ == "__main__":
    main()
