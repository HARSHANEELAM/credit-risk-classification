import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

TARGET = "risk_label"
DATA_PATH = "data/credit_risk_sample.csv"
MODEL_PATH = "src/model_pipeline.joblib"

def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    model = joblib.load(MODEL_PATH)
    proba = model.predict_proba(X)[:, 1]
    preds = model.predict(X)

    print("ROC-AUC:", roc_auc_score(y, proba))
    print("\nClassification Report:\n", classification_report(y, preds))
    print("\nConfusion Matrix:\n", confusion_matrix(y, preds))

if __name__ == "__main__":
    main()
