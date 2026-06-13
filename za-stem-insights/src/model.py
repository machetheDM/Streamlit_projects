"""
model.py  —  Supervised ML (MASTERY tier)
--------------------------------------------
Supervised ML to predict whether a province-subject combination will fall
BELOW the national STEM pass-rate threshold in the next year.
Binary classification: 0 = On Track  |  1 = At Risk

Compares 3 algorithms (Logistic Regression, Random Forest, XGBoost) with
stratified K-fold cross-validation, ROC-AUC, and SHAP explainability.

Base model (RandomForest): SKYLearn-Innovation Learner 4 (high school)
Multi-model comparison, K-fold CV, SHAP explainability: Dingaan Mahlatse
Machethe (SKYLearn-Innovation head — MSc Data Science, UEL)
"""

import os
import pickle
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve,
    confusion_matrix, precision_score, recall_score, f1_score, accuracy_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

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


# ─────────────────────────────────────────────────────────────
# MASTERY tier — multi-model comparison, cross-validation, SHAP
# ─────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "year", "province_code_enc", "subject_enc", "subject_type_enc",
    "registered", "absentee_rate",
]


def get_model_zoo() -> dict:
    """
    Return the 3 candidate classifiers.
    Logistic Regression is wrapped in a pipeline with StandardScaler since
    it is scale-sensitive (tree models are not).
    """
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }


def compare_models(df: pd.DataFrame, n_splits: int = 5) -> dict:
    """
    Train and compare all models with stratified K-fold cross-validation.
    Returns a dict with a comparison DataFrame, fitted models, test split,
    and ROC curve data for each model.
    """
    X, y, encoders = build_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rows, fitted, roc_data = [], {}, {}

    for name, model in get_model_zoo().items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="roc_auc", n_jobs=-1)

        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = {"fpr": fpr, "tpr": tpr, "auc": roc_auc_score(y_test, y_prob)}
        fitted[name] = model

        rows.append({
            "Model": name,
            "CV ROC-AUC (mean)": round(cv_scores.mean(), 4),
            "CV ROC-AUC (std)": round(cv_scores.std(), 4),
            "Test Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Test Precision": round(precision_score(y_test, y_pred), 4),
            "Test Recall": round(recall_score(y_test, y_pred), 4),
            "Test F1": round(f1_score(y_test, y_pred), 4),
            "Test ROC-AUC": round(roc_auc_score(y_test, y_prob), 4),
            "Train Time (s)": round(train_time, 3),
        })

    comparison = pd.DataFrame(rows).sort_values("Test ROC-AUC", ascending=False).reset_index(drop=True)
    best_name = comparison.iloc[0]["Model"]

    return {
        "comparison": comparison,
        "models": fitted,
        "roc_data": roc_data,
        "best_model_name": best_name,
        "best_model": fitted[best_name],
        "encoders": encoders,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "feature_names": FEATURE_COLS,
    }


def confusion(model, X_test: pd.DataFrame, y_test: pd.Series) -> np.ndarray:
    """Return the 2x2 confusion matrix for a fitted model."""
    return confusion_matrix(y_test, model.predict(X_test))


def shap_importance(model, X_sample: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean absolute SHAP values per feature for a tree-based model.
    Returns a DataFrame sorted by importance. Falls back gracefully if the
    model type is unsupported by the fast TreeExplainer.
    """
    import shap

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        mean_abs = np.abs(shap_values).mean(axis=0)
    except Exception:
        explainer = shap.Explainer(model.predict, X_sample)
        shap_values = explainer(X_sample).values
        mean_abs = np.abs(shap_values).mean(axis=0)

    return (
        pd.DataFrame({"feature": X_sample.columns, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    from preprocess import load_clean
    df = load_clean()
    result = train(df)
    print("Accuracy:", result["metrics"]["accuracy"])
    print("ROC-AUC :", result["metrics"]["roc_auc"])
    print(result["metrics"]["report"])
