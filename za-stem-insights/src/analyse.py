"""
analyse.py  —  Learner 3 module
---------------------------------
Statistical analysis: national trends, provincial comparisons,
STEM vs non-STEM gap, subject performance, at-risk flagging.

Contributor: SKYLearn-Innovation Learner 3
Mentor: Dingaan Mahlatse Machethe
"""

import pandas as pd
import numpy as np


def national_trend(df: pd.DataFrame, subject_type: str | None = None) -> pd.DataFrame:
    """Average pass rate per year, optionally filtered by subject_type."""
    data = df.copy()
    if subject_type:
        data = data[data["subject_type"] == subject_type]
    trend = (
        data.groupby("year")
        .agg(
            avg_pass_rate=("pass_rate", "mean"),
            total_registered=("registered", "sum"),
            total_passed=("passed", "sum"),
        )
        .reset_index()
    )
    trend["national_pass_rate"] = (
        trend["total_passed"] / trend["total_registered"] * 100
    ).round(2)
    return trend


def province_summary(df: pd.DataFrame, year: int | None = None) -> pd.DataFrame:
    """Per-province average pass rate, optional year filter."""
    data = df if year is None else df[df["year"] == year]
    return (
        data.groupby(["province_code", "province"])
        .agg(
            avg_pass_rate=("pass_rate", "mean"),
            total_registered=("registered", "sum"),
            total_passed=("passed", "sum"),
        )
        .assign(
            national_pass_rate=lambda x: (x["total_passed"] / x["total_registered"] * 100).round(2),
            avg_pass_rate=lambda x: x["avg_pass_rate"].round(2),
        )
        .reset_index()
        .sort_values("avg_pass_rate", ascending=False)
    )


def stem_vs_nonstem(df: pd.DataFrame) -> pd.DataFrame:
    """Year-by-year STEM vs non-STEM average pass rate comparison."""
    return (
        df.groupby(["year", "subject_type"])
        .agg(avg_pass_rate=("pass_rate", "mean"))
        .round(2)
        .reset_index()
    )


def subject_ranking(df: pd.DataFrame, year: int | None = None) -> pd.DataFrame:
    """Rank all subjects by average pass rate for a given year."""
    data = df if year is None else df[df["year"] == year]
    return (
        data.groupby(["subject", "subject_type"])
        .agg(avg_pass_rate=("pass_rate", "mean"))
        .round(2)
        .reset_index()
        .sort_values("avg_pass_rate", ascending=False)
    )


def flag_at_risk(df: pd.DataFrame, threshold: float = 60.0) -> pd.DataFrame:
    """
    Flag province-subject combinations where pass rate < threshold.
    Returns rows sorted by pass_rate ascending (worst first).
    """
    latest_year = df["year"].max()
    latest = df[df["year"] == latest_year].copy()
    at_risk = latest[latest["pass_rate"] < threshold].copy()
    return at_risk.sort_values("pass_rate")[
        ["province", "subject", "subject_type", "pass_rate", "registered", "passed"]
    ].reset_index(drop=True)


def year_on_year_change(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate year-on-year pass rate change per province-subject pair."""
    pivot = df.pivot_table(
        index=["province", "subject"], columns="year", values="pass_rate"
    )
    years = sorted(df["year"].unique())
    changes = []
    for i in range(1, len(years)):
        prev, curr = years[i - 1], years[i]
        if prev in pivot.columns and curr in pivot.columns:
            delta = (pivot[curr] - pivot[prev]).reset_index()
            delta["year"] = curr
            delta["change"] = delta[curr] if curr in delta.columns else delta[0]
            delta = delta[["province", "subject", "year"]].copy()
            delta["change"] = (pivot[curr] - pivot[prev]).values
            changes.append(delta)
    return pd.concat(changes, ignore_index=True) if changes else pd.DataFrame()
