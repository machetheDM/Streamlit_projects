"""
model.py  —  Learner 4 module
-------------------------------
RandomForest model that predicts whether a province-subject combination
will fall BELOW the national STEM pass-rate threshold in the next year.
Binary classification: 0 = On Track  |  1 = At Risk

Contributor: SKYLearn-Innovation Learner 4
Mentor: Dingaan Mahlatse Machethe
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "rf_model.pkl")
ENCODERS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "label_encoders.pkl")

AT_RISK_THRESHOLD = 60.0
RANDOM_STATE = 42


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Engineer features for training.
    Returns (X, y, label_encoders).
    """
    data = df.copy()
    data["at_risk"] = (data["pass_rate"] < AT_RISK_THRESHOLD).astype(int)

    encoders = {}
    for col in ["province_code", "subject", "subject_type"]:
        le = LabelEncoder()
        data[col + "_enc"] = le.fit_transform(data[col])
        encoders[col] = le

    feature_cols = [
        "year", "province_code_enc", "subject_enc", "subject_type_enc",
        "registered", "absentee_rate",
    ]
    X = data[feature_cols]
    y = data["at_risk"]
    return X, y, encoders


def train(df: pd.DataFrame) -> dict:
    """
    Train a RandomForest classifier and return metrics + model.
    Saves model and encoders to disk.
    """
    X, y, encoders = build_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(float((y_pred == y_test).mean()), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "report": classification_report(y_test, y_pred, target_names=["On Track", "At Risk"]),
    }

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    with open(ENCODERS_PATH, "wb") as f:
        pickle.dump(encoders, f)

    return {"model": clf, "encoders": encoders, "metrics": metrics}


def load_model() -> tuple:
    """Load the saved model and encoders from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Run train() first.")
    with open(MODEL_PATH, "rb") as f:
        clf = pickle.load(f)
    with open(ENCODERS_PATH, "rb") as f:
        encoders = pickle.load(f)
    return clf, encoders


def predict_next_year(
    clf,
    encoders: dict,
    province_code: str,
    subject: str,
    subject_type: str,
    registered: int,
    absentee_rate: float,
    next_year: int,
) -> dict:
    """
    Predict at-risk status for a single province-subject in a given year.
    """
    row = {
        "year": next_year,
        "province_code_enc": encoders["province_code"].transform([province_code])[0],
        "subject_enc": encoders["subject"].transform([subject])[0],
        "subject_type_enc": encoders["subject_type"].transform([subject_type])[0],
        "registered": registered,
        "absentee_rate": absentee_rate,
    }
    X = pd.DataFrame([row])
    prediction = int(clf.predict(X)[0])
    probability = float(clf.predict_proba(X)[0][1])
    return {
        "at_risk": prediction == 1,
        "risk_probability": round(probability * 100, 1),
        "label": "At Risk" if prediction == 1 else "On Track",
    }


def feature_importance(clf, feature_names: list[str]) -> pd.DataFrame:
    """Return a sorted DataFrame of feature importances."""
    return (
        pd.DataFrame({"feature": feature_names, "importance": clf.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    from preprocess import load_clean
    df = load_clean()
    result = train(df)
    print("Accuracy:", result["metrics"]["accuracy"])
    print("ROC-AUC :", result["metrics"]["roc_auc"])
    print(result["metrics"]["report"])
