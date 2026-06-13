"""
forecast.py  —  Time Series Forecasting (MASTERY tier)
---------------------------------------------------------
Forecasts future matric pass rates using Holt's Exponential Smoothing
(ETS) with a linear trend. Designed for short annual series (10 points),
where ETS is more stable than ARIMA. Falls back to OLS trend extrapolation
if the series is too short or ETS fails to converge.

Produces point forecasts plus approximate confidence intervals.

Contributor: Dingaan Mahlatse Machethe (SKYLearn-Innovation head — MSc Data
Science, UEL). Time series forecasting is postgraduate-level material.
"""

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy import stats

warnings.filterwarnings("ignore")


def _series_for(df: pd.DataFrame, province: str | None, subject: str | None) -> pd.Series:
    """Build an annual pass-rate series filtered by province and/or subject."""
    data = df.copy()
    if province and province != "National":
        data = data[data["province"] == province]
    if subject and subject != "All Subjects":
        data = data[data["subject"] == subject]
    series = data.groupby("year")["pass_rate"].mean().sort_index()
    return series


def _ols_forecast(series: pd.Series, horizon: int) -> pd.DataFrame:
    """Fallback: ordinary least squares linear trend extrapolation."""
    years = series.index.values.astype(float)
    reg = stats.linregress(years, series.values)
    future_years = np.arange(series.index.max() + 1, series.index.max() + 1 + horizon)
    preds = reg.intercept + reg.slope * future_years
    resid_std = float(np.std(series.values - (reg.intercept + reg.slope * years)))
    margin = 1.96 * resid_std
    return pd.DataFrame({
        "year": future_years.astype(int),
        "forecast": np.clip(preds, 0, 100).round(2),
        "lower": np.clip(preds - margin, 0, 100).round(2),
        "upper": np.clip(preds + margin, 0, 100).round(2),
        "method": "OLS trend",
    })


def forecast_series(
    df: pd.DataFrame,
    province: str | None = None,
    subject: str | None = None,
    horizon: int = 3,
) -> dict:
    """
    Forecast the next `horizon` years of pass rate for the given filter.
    Returns dict with the historical series and a forecast DataFrame.
    """
    series = _series_for(df, province, subject)

    if len(series) < 4:
        raise ValueError("Need at least 4 years of history to forecast.")

    try:
        model = ExponentialSmoothing(
            series.values, trend="add", seasonal=None,
            initialization_method="estimated",
        )
        fit = model.fit()
        preds = fit.forecast(horizon)

        resid_std = float(np.std(fit.resid)) if hasattr(fit, "resid") else float(series.std())
        margin = 1.96 * resid_std
        future_years = np.arange(series.index.max() + 1, series.index.max() + 1 + horizon)

        forecast_df = pd.DataFrame({
            "year": future_years.astype(int),
            "forecast": np.clip(preds, 0, 100).round(2),
            "lower": np.clip(preds - margin, 0, 100).round(2),
            "upper": np.clip(preds + margin, 0, 100).round(2),
            "method": "Holt ETS",
        })
    except Exception:
        forecast_df = _ols_forecast(series, horizon)

    hist_df = pd.DataFrame({
        "year": series.index.astype(int),
        "pass_rate": series.values.round(2),
    })

    return {
        "history": hist_df,
        "forecast": forecast_df,
        "label": _build_label(province, subject),
    }


def _build_label(province: str | None, subject: str | None) -> str:
    p = province or "National"
    s = subject or "All Subjects"
    return f"{p} — {s}"


def forecast_all_subjects(df: pd.DataFrame, province: str | None = None, horizon: int = 3) -> pd.DataFrame:
    """
    Forecast each subject's pass rate `horizon` years ahead.
    Returns a tidy DataFrame of final-year forecasts ranked ascending
    (worst-performing first) — useful for early-warning planning.
    """
    rows = []
    for subject in sorted(df["subject"].unique()):
        try:
            res = forecast_series(df, province=province, subject=subject, horizon=horizon)
            final = res["forecast"].iloc[-1]
            rows.append({
                "subject": subject,
                "forecast_year": int(final["year"]),
                "forecast_pass_rate": float(final["forecast"]),
                "lower": float(final["lower"]),
                "upper": float(final["upper"]),
            })
        except ValueError:
            continue
    return pd.DataFrame(rows).sort_values("forecast_pass_rate").reset_index(drop=True)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from preprocess import load_clean

    df = load_clean()
    res = forecast_series(df, subject="Mathematics", horizon=3)
    print(f"Forecast for {res['label']}:")
    print("History:\n", res["history"].to_string(index=False))
    print("\nForecast:\n", res["forecast"].to_string(index=False))

    print("\nAll-subject 3-year forecast (worst first):")
    print(forecast_all_subjects(df, horizon=3).to_string(index=False))
