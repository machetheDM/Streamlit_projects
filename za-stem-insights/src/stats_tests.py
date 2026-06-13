"""
stats_tests.py  —  Statistical Inference (TUTOR tier)
--------------------------------------------------------
Formal hypothesis testing on the matric dataset:
  1. Independent t-test / Mann-Whitney U — STEM vs Non-STEM pass rates
  2. One-way ANOVA — pass rate differences across provinces
  3. Pearson & Spearman correlation — absenteeism vs pass rate
  4. Linear regression trend test — is the national trend significant?

Each test reports the statistic, p-value, and a plain-English conclusion
at alpha = 0.05.

Contributor: SKYLearn-Innovation Tutor (university undergraduate)
Reviewed by: Dingaan Mahlatse Machethe (SKYLearn-Innovation head)
"""

import pandas as pd
import numpy as np
from scipy import stats

ALPHA = 0.05


def _conclude(p: float, h1: str, h0: str) -> str:
    return h1 if p < ALPHA else h0


def stem_vs_nonstem_test(df: pd.DataFrame) -> dict:
    """
    Test whether STEM and Non-STEM pass rates differ significantly.
    Runs Shapiro normality check, then t-test or Mann-Whitney accordingly.
    """
    stem = df[df["subject_type"] == "STEM"]["pass_rate"]
    nonstem = df[df["subject_type"] == "Non-STEM"]["pass_rate"]

    normal = (
        stats.shapiro(stem.sample(min(500, len(stem)), random_state=42))[1] > ALPHA
        and stats.shapiro(nonstem.sample(min(500, len(nonstem)), random_state=42))[1] > ALPHA
    )

    if normal:
        stat, p = stats.ttest_ind(stem, nonstem, equal_var=False)
        test_name = "Welch's t-test"
    else:
        stat, p = stats.mannwhitneyu(stem, nonstem, alternative="two-sided")
        test_name = "Mann-Whitney U"

    return {
        "test": test_name,
        "statistic": round(float(stat), 4),
        "p_value": float(p),
        "stem_mean": round(float(stem.mean()), 2),
        "nonstem_mean": round(float(nonstem.mean()), 2),
        "conclusion": _conclude(
            p,
            "STEM and Non-STEM pass rates differ significantly (reject H0).",
            "No significant difference between STEM and Non-STEM pass rates.",
        ),
    }


def province_anova(df: pd.DataFrame) -> dict:
    """
    One-way ANOVA: do mean pass rates differ across the 9 provinces?
    """
    groups = [g["pass_rate"].values for _, g in df.groupby("province")]
    stat, p = stats.f_oneway(*groups)
    return {
        "test": "One-way ANOVA",
        "statistic": round(float(stat), 4),
        "p_value": float(p),
        "conclusion": _conclude(
            p,
            "At least one province's mean pass rate differs significantly (reject H0).",
            "No significant difference in pass rates across provinces.",
        ),
    }


def absentee_correlation(df: pd.DataFrame) -> dict:
    """
    Pearson and Spearman correlation between absentee rate and pass rate.
    """
    pearson_r, pearson_p = stats.pearsonr(df["absentee_rate"], df["pass_rate"])
    spearman_r, spearman_p = stats.spearmanr(df["absentee_rate"], df["pass_rate"])
    return {
        "test": "Pearson & Spearman correlation",
        "pearson_r": round(float(pearson_r), 4),
        "pearson_p": float(pearson_p),
        "spearman_r": round(float(spearman_r), 4),
        "spearman_p": float(spearman_p),
        "conclusion": _conclude(
            pearson_p,
            f"Significant correlation (r={pearson_r:.3f}) between absenteeism and pass rate.",
            "No significant correlation between absenteeism and pass rate.",
        ),
    }


def trend_significance(df: pd.DataFrame) -> dict:
    """
    Linear regression of national pass rate on year to test whether the
    long-term trend slope is significantly different from zero.
    """
    yearly = df.groupby("year")["pass_rate"].mean().reset_index()
    result = stats.linregress(yearly["year"], yearly["pass_rate"])
    direction = "increasing" if result.slope > 0 else "decreasing"
    return {
        "test": "Linear regression (trend)",
        "slope_per_year": round(float(result.slope), 4),
        "r_squared": round(float(result.rvalue ** 2), 4),
        "p_value": float(result.pvalue),
        "conclusion": _conclude(
            result.pvalue,
            f"Significant {direction} trend: {result.slope:.3f} pp/year (reject H0).",
            "No statistically significant national trend over time.",
        ),
    }


def run_all_tests(df: pd.DataFrame) -> dict:
    """Run the full battery of tests and return a results dict."""
    return {
        "STEM vs Non-STEM": stem_vs_nonstem_test(df),
        "Province ANOVA": province_anova(df),
        "Absenteeism Correlation": absentee_correlation(df),
        "National Trend": trend_significance(df),
    }


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from preprocess import load_clean

    df = load_clean()
    for name, res in run_all_tests(df).items():
        print(f"\n=== {name} ===")
        for k, v in res.items():
            print(f"  {k}: {v}")
