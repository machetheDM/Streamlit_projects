"""
generate_sample_data.py
-----------------------
Builds a province x subject x year dataset of South African National Senior
Certificate (Matric) results, ANCHORED ON REAL PUBLISHED FIGURES from the
Department of Basic Education (DBE).

What is real (calibration targets):
  * National NSC pass rate per year, 2014-2023 (REAL_NATIONAL_PASS_RATE).
    Source: DBE NSC Examination Reports / widely reported national results.
  * 2023 pass rate for each of the 9 provinces (REAL_PROVINCE_PASS_RATE_2023).
    Source: matric.co.za / DBE 2023 NSC results.

What is modelled:
  The DBE does not publish a single clean machine-readable
  per-subject-per-province-per-year CSV. So province-year OVERALL pass rates
  are calibrated to match the official figures above, and subject-level
  granularity is layered on using subject offsets that reflect the
  well-documented STEM difficulty gap (Mathematics is consistently the
  hardest subject). Enrolment figures approximate real provincial scale.

The result keeps every province-year average pass rate within ~2pp of the
official published value, so the trends, rankings and STEM gap shown in the
dashboard are realistic rather than arbitrary.

Run this once before notebooks or the dashboard:
    python data/generate_sample_data.py
"""

import pandas as pd
import numpy as np
import os

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

# --- REAL DATA: official national NSC pass rate (%) per year ---------------
# Source: DBE NSC Examination Reports (2014-2023).
REAL_NATIONAL_PASS_RATE = {
    2014: 75.8, 2015: 70.7, 2016: 72.5, 2017: 75.1, 2018: 78.2,
    2019: 81.3, 2020: 76.2, 2021: 76.4, 2022: 80.1, 2023: 82.9,
}

# --- REAL DATA: 2023 provincial NSC pass rate (%) --------------------------
# Source: matric.co.za / DBE 2023 NSC results.
REAL_PROVINCE_PASS_RATE_2023 = {
    "EC": 81.4, "FS": 89.0, "GP": 85.4, "KZN": 86.4, "LP": 79.5,
    "MP": 77.0, "NC": 75.8, "NW": 81.6, "WC": 81.5,
}

PROVINCES = {
    "EC": "Eastern Cape",
    "FS": "Free State",
    "GP": "Gauteng",
    "KZN": "KwaZulu-Natal",
    "LP": "Limpopo",
    "MP": "Mpumalanga",
    "NC": "Northern Cape",
    "NW": "North West",
    "WC": "Western Cape",
}

SUBJECTS = {
    "Mathematics":          {"type": "STEM", "base_pass": 52},
    "Physical Sciences":    {"type": "STEM", "base_pass": 58},
    "Life Sciences":        {"type": "STEM", "base_pass": 67},
    "Technical Mathematics":{"type": "STEM", "base_pass": 56},
    "Technical Sciences":   {"type": "STEM", "base_pass": 54},
    "Mathematical Literacy":{"type": "Non-STEM", "base_pass": 78},
    "English Home Lang":    {"type": "Non-STEM", "base_pass": 81},
    "Accounting":           {"type": "Non-STEM", "base_pass": 70},
    "Business Studies":     {"type": "Non-STEM", "base_pass": 73},
    "Geography":            {"type": "Non-STEM", "base_pass": 75},
}

# Each province's offset (pp) from the national pass rate, derived from the
# REAL 2023 provincial figures relative to the real 2023 national rate (82.9).
# Assumed roughly stable across years (a standard, defensible simplification).
_NATIONAL_2023 = REAL_NATIONAL_PASS_RATE[2023]
PROVINCE_OFFSET = {
    p: round(rate - _NATIONAL_2023, 2)
    for p, rate in REAL_PROVINCE_PASS_RATE_2023.items()
}

# Subject pass-rate offset (pp) relative to the province-year OVERALL rate.
# Calibrated to the well-documented reality that STEM subjects (especially
# Mathematics) pass at far lower rates than humanities/commerce subjects.
# The offsets are centred so their mean is ~0; this keeps each province-year
# AVERAGE subject pass rate aligned with the real published overall figure
# while preserving the realistic spread between subjects.
SUBJECT_OFFSET = {
    "Mathematics":           -15,
    "Technical Sciences":    -13,
    "Technical Mathematics": -11,
    "Physical Sciences":      -7,
    "Life Sciences":           1,
    "Accounting":              2,
    "Geography":               6,
    "Business Studies":        9,
    "Mathematical Literacy":  11,
    "English Home Lang":      14,
}

PROVINCE_ENROLLMENT = {
    "EC": 80_000, "FS": 38_000, "GP": 145_000,
    "KZN": 130_000, "LP": 75_000, "MP": 55_000,
    "NC": 14_000, "NW": 45_000, "WC": 72_000,
}

YEARS = list(range(2014, 2024))


def _generate_rows() -> list[dict]:
    rows = []
    for year in YEARS:
        for pcode, pname in PROVINCES.items():
            base_enrollment = PROVINCE_ENROLLMENT[pcode]
            # Province-year OVERALL pass rate, anchored to real published data.
            province_overall = (
                REAL_NATIONAL_PASS_RATE[year]
                + PROVINCE_OFFSET[pcode]
                + rng.normal(0, 0.8)
            )
            for subject, info in SUBJECTS.items():
                registered = int(
                    base_enrollment * rng.uniform(0.18, 0.32)
                    * (1 + rng.normal(0, 0.04))
                )
                wrote = int(registered * rng.uniform(0.94, 0.99))
                raw_pass_rate = (
                    province_overall
                    + SUBJECT_OFFSET[subject]
                    + rng.normal(0, 2.0)
                )
                pass_rate = float(np.clip(raw_pass_rate, 30, 99))
                passed = int(wrote * pass_rate / 100)
                avg_score = float(np.clip(pass_rate * 0.72 + rng.normal(0, 3), 25, 85))
                distinctions = int(passed * rng.uniform(0.04, 0.18))

                rows.append({
                    "year":           year,
                    "province_code":  pcode,
                    "province":       pname,
                    "subject":        subject,
                    "subject_type":   info["type"],
                    "registered":     registered,
                    "wrote":          wrote,
                    "passed":         passed,
                    "distinctions":   distinctions,
                    "pass_rate":      round(pass_rate, 2),
                    "avg_score":      round(avg_score, 2),
                    "absentee_rate":  round((registered - wrote) / registered * 100, 2),
                    "distinction_rate": round(distinctions / passed * 100 if passed > 0 else 0, 2),
                })
    return rows


def main() -> None:
    rows = _generate_rows()
    df = pd.DataFrame(rows)

    out_raw = os.path.join(os.path.dirname(__file__), "raw", "matric_results_2014_2023.csv")
    df.to_csv(out_raw, index=False)
    print(f"Raw data saved  → {out_raw}  ({len(df):,} rows)")

    out_proc = os.path.join(os.path.dirname(__file__), "processed", "matric_clean.csv")
    df.to_csv(out_proc, index=False)
    print(f"Processed data  → {out_proc}")


if __name__ == "__main__":
    main()
