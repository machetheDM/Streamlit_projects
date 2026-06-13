"""
db.py  —  SQL data layer (SQLite)
------------------------------------
Loads the cleaned matric dataset into an on-disk SQLite database and exposes
a thin query API. This demonstrates SQL as a first-class skill: the dashboard's
SQL Explorer page and several analytics run real SQL against this database
rather than in-memory Pandas only.

Uses the Python standard-library `sqlite3` driver (no extra dependency).

Contributor: Dingaan Mahlatse Machethe (SKYLearn-Innovation head)
"""

import os
import sqlite3
import pandas as pd

from preprocess import load_clean

DB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed", "matric.db"
)
TABLE = "matric"

# Read-only queries are allowed; everything else is rejected in run_query().
_FORBIDDEN = (
    "insert", "update", "delete", "drop", "alter", "create",
    "replace", "attach", "detach", "pragma", "vacuum",
)

# Curated example queries surfaced in the SQL Explorer page.
EXAMPLE_QUERIES = {
    "STEM vs Non-STEM pass rate by year": """
SELECT year,
       subject_type,
       ROUND(AVG(pass_rate), 2) AS avg_pass_rate
FROM matric
GROUP BY year, subject_type
ORDER BY year, subject_type;
""".strip(),
    "Top 10 worst province-subject combinations (2023)": """
SELECT province,
       subject,
       ROUND(AVG(pass_rate), 2) AS pass_rate
FROM matric
WHERE year = 2023
GROUP BY province, subject
ORDER BY pass_rate ASC
LIMIT 10;
""".strip(),
    "Provincial ranking with learner volume (2023)": """
SELECT province,
       ROUND(AVG(pass_rate), 2)       AS avg_pass_rate,
       SUM(wrote)                      AS learners_wrote,
       SUM(passed)                     AS learners_passed,
       SUM(distinctions)               AS distinctions
FROM matric
WHERE year = 2023
GROUP BY province
ORDER BY avg_pass_rate DESC;
""".strip(),
    "Mathematics pass rate trend by province": """
SELECT province,
       year,
       ROUND(pass_rate, 2) AS pass_rate
FROM matric
WHERE subject = 'Mathematics'
ORDER BY province, year;
""".strip(),
    "Subjects below the 60% benchmark (national, 2023)": """
SELECT subject,
       subject_type,
       ROUND(AVG(pass_rate), 2) AS national_pass_rate
FROM matric
WHERE year = 2023
GROUP BY subject, subject_type
HAVING national_pass_rate < 60
ORDER BY national_pass_rate ASC;
""".strip(),
}


def build_database(force: bool = False) -> str:
    """
    Create/refresh the SQLite database from the cleaned dataset.
    Returns the database path. Idempotent unless force=True.
    """
    if force and os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    df = load_clean()
    conn = sqlite3.connect(DB_PATH)
    try:
        df.to_sql(TABLE, conn, if_exists="replace", index=False)
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_year ON {TABLE}(year)")
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_prov ON {TABLE}(province)")
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_subj ON {TABLE}(subject)")
        conn.commit()
    finally:
        conn.close()
    return DB_PATH


def get_connection() -> sqlite3.Connection:
    """Open a connection, building the database on first use."""
    if not os.path.exists(DB_PATH):
        build_database()
    return sqlite3.connect(DB_PATH)


def run_query(sql: str) -> pd.DataFrame:
    """
    Execute a read-only SQL query and return a DataFrame.
    Rejects any statement that attempts to modify the database.
    """
    stripped = sql.strip().rstrip(";")
    lowered = stripped.lower()

    if not lowered.startswith(("select", "with")):
        raise ValueError("Only SELECT/WITH (read-only) queries are allowed.")
    if ";" in stripped:
        raise ValueError("Multiple statements are not allowed.")
    for word in _FORBIDDEN:
        if f" {word} " in f" {lowered} ":
            raise ValueError(f"'{word.upper()}' statements are not permitted.")

    conn = get_connection()
    try:
        return pd.read_sql_query(stripped, conn)
    finally:
        conn.close()


def table_schema() -> pd.DataFrame:
    """Return the column schema of the matric table for display."""
    conn = get_connection()
    try:
        info = pd.read_sql_query(f"PRAGMA table_info({TABLE});", conn)
        return info[["name", "type"]].rename(columns={"name": "column", "type": "type"})
    finally:
        conn.close()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    path = build_database(force=True)
    print(f"Database built → {path}")
    print("\nSchema:")
    print(table_schema().to_string(index=False))
    print("\nExample query result:")
    print(run_query(EXAMPLE_QUERIES["Provincial ranking with learner volume (2023)"]).to_string(index=False))
