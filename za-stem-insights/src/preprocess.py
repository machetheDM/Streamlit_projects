"""
preprocess.py  —  Learner 1 module
------------------------------------
Loads the raw matric results CSV, validates columns, handles missing
values, and returns a clean DataFrame ready for analysis.

Contributor: SKYLearn-Innovation Learner 1
Mentor: Dingaan Mahlatse Machethe
"""

import os
import pandas as pd

RAW_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "raw", "matric_results_2014_2023.csv"
)
PROCESSED_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed", "matric_clean.csv"
)

REQUIRED_COLUMNS = [
    "year", "province_code", "province", "subject", "subject_type",
    "registered", "wrote", "passed", "distinctions",
    "pass_rate", "avg_score", "absentee_rate", "distinction_rate",
]


def ensure_raw_data(path: str = RAW_PATH) -> None:
    """
    Guarantee the raw CSV exists. If missing (e.g. fresh clone on Streamlit
    Cloud where data is gitignored), generate it automatically.
    """
    if os.path.exists(path):
        return
    import importlib.util
    gen_path = os.path.join(os.path.dirname(__file__), "..", "data", "generate_sample_data.py")
    spec = importlib.util.spec_from_file_location("generate_sample_data", gen_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    """Load the raw CSV and return a DataFrame, generating it if absent."""
    ensure_raw_data(path)
    df = pd.read_csv(path)
    return df


def validate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Raise if any required column is missing."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop nulls, enforce dtypes, clip impossible values.
    """
    df = df.dropna(subset=["pass_rate", "province", "subject"]).copy()
    df["year"] = df["year"].astype(int)
    df["pass_rate"] = df["pass_rate"].clip(0, 100)
    df["avg_score"] = df["avg_score"].clip(0, 100)
    df["absentee_rate"] = df["absentee_rate"].clip(0, 100)
    df["distinction_rate"] = df["distinction_rate"].clip(0, 100)
    df["subject_type"] = df["subject_type"].str.strip()
    df["province"] = df["province"].str.strip()
    return df.reset_index(drop=True)


def save_processed(df: pd.DataFrame, path: str = PROCESSED_PATH) -> None:
    """Save the clean DataFrame to the processed folder."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved clean data → {path}  ({len(df):,} rows)")


def run_pipeline(raw_path: str = RAW_PATH) -> pd.DataFrame:
    """Full preprocessing pipeline: load → validate → clean → save."""
    df = load_raw(raw_path)
    df = validate_columns(df)
    df = clean(df)
    save_processed(df)
    return df


def load_clean(path: str = PROCESSED_PATH) -> pd.DataFrame:
    """
    Quick loader used by the dashboard and other modules.
    If the processed file is missing, runs the pipeline automatically.
    """
    if not os.path.exists(path):
        return run_pipeline()
    return pd.read_csv(path)


if __name__ == "__main__":
    df = run_pipeline()
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Years : {sorted(df['year'].unique())}")
    print(f"Provinces: {df['province'].nunique()}")
