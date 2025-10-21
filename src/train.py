import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score
from preprocessing import build_preprocess_pipeline, split_X_y

TARGET = "risk_label"
DATA_PATH = "data/credit_risk_sample.csv"
MODEL_PATH = "src/model_pipeline.joblib"

def main():
    df = pd.read_csv(DATA_PATH)
    preprocess = build_preprocess_pipeline(df, TARGET)
    X, y = split_X_y(df, TARGET)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    candidates = {
        "log_reg": (LogisticRegression(max_iter=200), {"clf__C":[0.1,1,10]}),
        "rf":      (RandomForestClassifier(random_state=42), {"clf__n_estimators":[100,300], "clf__max_depth":[None,10,20]}),
        "svm":     (SVC(kernel="rbf", probability=True), {"clf__C":[0.5,1,2], "clf__gamma":["scale","auto"]}),
    }

    best_auc, best_name, best_pipe = -1.0, None, None

    for name, (clf, grid) in candidates.items():
        pipe = Pipeline(steps=[("prep", preprocess.pipeline), ("clf", clf)])
        search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=3, n_jobs=-1)
        search.fit(X_train, y_train)
        preds = search.predict(X_val)
        proba = search.predict_proba(X_val)[:,1]
        auc = roc_auc_score(y_val, proba)
        acc = accuracy_score(y_val, preds)
        print(f"{name} -> AUC: {auc:.3f} | ACC: {acc:.3f} | best_params: {search.best_params_}")
        if auc > best_auc:
            best_auc, best_name, best_pipe = auc, name, search.best_estimator_

    print(f"BEST: {best_name} with AUC={best_auc:.3f}")
    joblib.dump(best_pipe, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

if __name__ == "__main__":
    main()
