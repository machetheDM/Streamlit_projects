"""
generate_sample_data.py
-----------------------
Generates a realistic synthetic dataset that mirrors the structure of
South African National Senior Certificate (Matric) results published
annually by the Department of Basic Education (DBE).

Real data source:
  https://www.education.gov.za/Portals/0/Documents/Reports/

Run this once before notebooks or the dashboard:
    python data/generate_sample_data.py
"""

import pandas as pd
import numpy as np
import os

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

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

PROVINCE_MULTIPLIER = {
    "EC": 0.88, "FS": 0.96, "GP": 1.06,
    "KZN": 0.92, "LP": 0.91, "MP": 0.93,
    "NC": 0.95, "NW": 0.94, "WC": 1.08,
}

PROVINCE_ENROLLMENT = {
    "EC": 80_000, "FS": 38_000, "GP": 145_000,
    "KZN": 130_000, "LP": 75_000, "MP": 55_000,
    "NC": 14_000, "NW": 45_000, "WC": 72_000,
}

YEARS = list(range(2014, 2024))

YEARLY_TREND = {y: 1.0 + (y - 2014) * 0.003 for y in YEARS}
YEARLY_TREND[2020] = YEARLY_TREND[2020] - 0.04
YEARLY_TREND[2021] = YEARLY_TREND[2021] - 0.02


def _generate_rows() -> list[dict]:
    rows = []
    for year in YEARS:
        for pcode, pname in PROVINCES.items():
            base_enrollment = PROVINCE_ENROLLMENT[pcode]
            for subject, info in SUBJECTS.items():
                registered = int(
                    base_enrollment * rng.uniform(0.18, 0.32)
                    * (1 + rng.normal(0, 0.04))
                )
                wrote = int(registered * rng.uniform(0.94, 0.99))
                raw_pass_rate = (
                    info["base_pass"]
                    * PROVINCE_MULTIPLIER[pcode]
                    * YEARLY_TREND[year]
                    + rng.normal(0, 2.5)
                )
                pass_rate = float(np.clip(raw_pass_rate, 30, 98))
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
