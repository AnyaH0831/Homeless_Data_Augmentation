"""
clean_lanark_bnl.py

Cleans the raw "Anonymized_BNL_Data_-_Lanark_County_cleaned.xlsx" export
(sheet: "Community By-Name List") into a tidy, analysis-ready CSV.

The raw sheet has:
  - 2 merged "section title" rows at the top (rows 1-2)
  - the real column headers on row 3 (some with embedded newlines)
  - a "data type hint" row (row 4, values like "Free text", "Date", "DD")
  - actual client records starting around row 5
  - many trailing blank / formula-template rows with no real data
  - several columns that are live Excel formulas (status flags, derived
    dates, etc.) rather than static values

This script:
  1. Loads the workbook with data_only=True so formula cells resolve to
     their last-calculated cached values instead of formula strings.
  2. Locates the header row dynamically (by searching for the
     "Unique identifier" cell) instead of hard-coding row/column numbers,
     so it's more robust to minor layout changes.
  3. Drops rows with no client identifier (i.e. the empty template rows).
  4. Selects and renames the columns relevant for downstream analysis /
     merging, and recodes a handful of them into clean flags.
  5. Writes the result to lanark_bnl_cleaned.csv.

Usage:
    python clean_lanark_bnl.py \
        --input "Anonymized_BNL_Data_-_Lanark_County_cleaned.xlsx" \
        --output "lanark_bnl_cleaned.csv"
"""

import argparse
import re
import sys
from datetime import datetime, date

import openpyxl
import pandas as pd

SHEET_NAME = "Community By-Name List"

# Maps a normalized (lowercased, whitespace-collapsed) substring found in
# the raw header text -> the clean column name we want in the output.
# Using substring matching (not exact match) because several raw headers
# contain embedded newlines / trailing spaces / section-number prefixes
# (e.g. "3.1\nChronically homeless").
HEADER_MAP = {
    "unique identifier": "client_id",
    "last contact date": "last_contact_date",
    "assessment type": "assessment_type",
    "location": "location",
    "last name": "last_name_raw",
    "first name": "first_name_raw",
    "number of months experiencing homelessness in past\nyear": "months_homeless_past_year",
    "number of months experiencing homelessness in past\n3 years": "months_homeless_past_3yr",
    "current sleeping arrangements": "sleeping_arrangement_raw",
    "housed": "housed_date",
    "veteran status confirmed": "veteran_status_confirmed",
    "veteran status": "veteran_status_raw",
    "gender identity": "gender_raw",
    "date of birth": "date_of_birth",
    "age calculator": "age_raw",
    "head of household": "head_of_household",
    "number of children": "num_children",
    "indigenous identity": "indigenous_identity_raw",
    "referral agency": "referral_agency",
    "added to by-name list date": "added_to_bnl_date",
    "asylum seeker": "asylum_seeker_raw",
    "3.1\nchronically homeless": "chronic_flag_raw",
    "3.1\nyouth": "youth_flag_raw",
    "3.1\nindigenous": "indigenous_flag_raw_criteria",
    "income source": "income_type_raw",
    "do you have an income source": "has_income_source_raw",
    "health challenges: mental health issue": "mental_health_raw",
    "health challenges: substance use issue": "substance_use_raw",
    "institutional involvement": "institutional_involvement_raw",
}

# The columns from the SASM <-> Lanark mapping table that actually have a
# usable Lanark counterpart (i.e. the ones used as the common schema in
# merge_datasets.py). Columns from the mapping table with "No match" or
# only a very weak match (race, education, lgbtq, foster_care_history,
# shelter_type, immigrant, incarceration_history, housing_loss_income/health)
# are intentionally excluded here since they don't have a real Lanark
# source column. "year" is derived from last_contact_date's year (see
# `clean()`) since Lanark has no single dedicated "year" field.
FINAL_COLUMNS = [
    "row_id",
    "year",
    "age",
    "years_homeless",
    "gender",
    "has_dependents",
    "mental_health",
    "substance_use",
    "outdoor_sleeping",
    "chronic_homeless",
    "youth",
    "indigenous_flag",
    "no_income",
    "income_type",
]


def normalize_header(text):
    if text is None:
        return ""
    text = str(text).lower()
    text = text.replace("\r", "\n")
    text = re.sub(r"\s+", " ", text.replace("\n", "\n")).strip()
    # keep newlines out of the comparison but preserve them in HEADER_MAP
    # keys by comparing against a newline-normalized version too
    return text


def build_header_index(ws, max_scan_rows=10):
    """
    Scan the first `max_scan_rows` rows to find the header row (the row
    containing a cell that mentions "unique identifier"), then build a
    {clean_column_name: column_index} map using HEADER_MAP substring
    matches.
    """
    header_row_idx = None
    header_cells = None

    for row_idx in range(1, max_scan_rows + 1):
        row_values = [
            ws.cell(row=row_idx, column=c).value
            for c in range(1, ws.max_column + 1)
        ]
        joined = " ".join(str(v) for v in row_values if v is not None).lower()
        if "unique identifier" in joined:
            header_row_idx = row_idx
            header_cells = row_values
            break

    if header_row_idx is None:
        raise RuntimeError(
            "Could not locate the header row (no cell containing "
            "'unique identifier' found in the first "
            f"{max_scan_rows} rows). Inspect the sheet layout and update "
            "clean_lanark_bnl.py accordingly."
        )

    col_index = {}
    for col_num, raw_value in enumerate(header_cells, start=1):
        if raw_value is None:
            continue
        raw_lower = str(raw_value).lower().replace("\r", "\n")
        for needle, clean_name in HEADER_MAP.items():
            if needle in raw_lower and clean_name not in col_index:
                col_index[clean_name] = col_num

    missing = set(HEADER_MAP.values()) - set(col_index.keys())
    if missing:
        print(
            f"[warn] Could not find a matching header for: {sorted(missing)}. "
            "These columns will be empty in the output. Check the raw "
            "headers if this is unexpected.",
            file=sys.stderr,
        )

    return header_row_idx, col_index


def load_raw_rows(path):
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    ws = wb[SHEET_NAME]
    header_row_idx, col_index = build_header_index(ws)

    # NOTE: in this anonymized export, the "Unique identifier/HIFIS client
    # ID" column is blank on every row (only the ID itself was stripped
    # during anonymization -- other fields weren't). So we can't use it to
    # detect which rows are real records. Instead, a row counts as "real"
    # if any of a few reliably-populated fields are non-blank, and we
    # assign our own synthetic row_id for merging purposes.
    validity_cols = [
        c for c in ("last_contact_date", "location", "last_name_raw", "gender_raw")
        if c in col_index
    ]

    records = []
    next_row_id = 1
    for row in ws.iter_rows(min_row=header_row_idx + 2):  # skip data-type-hint row too
        is_real_row = any(
            row[col_index[c] - 1].value not in (None, "", " ")
            for c in validity_cols
        )
        if not is_real_row:
            continue

        record = {}
        for clean_name, col_num in col_index.items():
            record[clean_name] = row[col_num - 1].value
        record["row_id"] = next_row_id
        next_row_id += 1
        records.append(record)

    wb.close()
    return pd.DataFrame.from_records(records)


def to_lower_str(v):
    """Safely lowercase/strip any cell value, treating every flavor of
    missing (None, NaN, pandas NA) as an empty string. Using this instead
    of `.astype(str).str.lower()` sidesteps a pandas gotcha where, on the
    nullable "string" dtype, .astype(str) can leave missing values as a
    raw float NaN rather than converting them to the text "nan"."""
    if pd.isna(v):
        return ""
    return str(v).strip().lower()


def yes_no_to_flag(series):
    """Map common BNL yes/unknown/blank text values to 0/1, leaving
    unparseable values as NA."""
    def _map(v):
        if v is None:
            return 0
        s = str(v).strip().lower()
        if s in ("yes", "y", "confirmed", "true", "1", "\u00fc", "chronic", "youth"):
            return 1
        if s in ("no", "n", "false", "0", "unknown", "", "---", "n/a"):
            return 0
        return pd.NA
    return series.apply(_map)


def safe_col(df, name):
    """Always return a proper Series aligned to df's index, even if the
    column wasn't found in the source workbook (df.get() would otherwise
    return None, which breaks any chained .str/.apply/pd.to_numeric call)."""
    if name in df.columns:
        return df[name]
    return pd.Series(pd.NA, index=df.index, dtype=object)


def compute_age(row):
    dob = row.get("date_of_birth")
    if isinstance(dob, (datetime, date)):
        today = datetime.today().date()
        dob_date = dob.date() if isinstance(dob, datetime) else dob
        return today.year - dob_date.year - (
            (today.month, today.day) < (dob_date.month, dob_date.day)
        )
    return pd.NA


def clean(df):
    df = df.copy()

    # --- Age ---
    df["age"] = df.apply(compute_age, axis=1)

    # --- Gender ---
    df["gender"] = (
        safe_col(df, "gender_raw").apply(to_lower_str)
        .replace({"": pd.NA})
    )

    # --- Years homeless (derived from months-in-past-year) ---
    df["years_homeless"] = pd.to_numeric(
        safe_col(df, "months_homeless_past_year"), errors="coerce"
    ) / 12.0

    # --- Chronic homelessness flag (checkbox column: 'ü' = checked,
    # blank = unchecked) ---
    df["chronic_homeless"] = (
        safe_col(df, "chronic_flag_raw").apply(to_lower_str)
        .apply(lambda s: 1 if s == "ü" else 0)
    )

    # --- Youth flag (checkbox column: 'ü' = checked, blank = unchecked) ---
    df["youth"] = (
        safe_col(df, "youth_flag_raw").apply(to_lower_str)
        .apply(lambda s: 1 if s == "ü" else 0)
    )

    # --- Indigenous flag ---
    df["indigenous_flag"] = (
        safe_col(df, "indigenous_identity_raw").apply(to_lower_str)
        .apply(lambda s: 0 if s in ("non-indigenous", "") else 1)
    )

    # --- Outdoor / unsheltered sleeping flag ---
    df["outdoor_sleeping"] = (
        safe_col(df, "sleeping_arrangement_raw").apply(to_lower_str)
        .apply(lambda s: 1 if "unshelter" in s else 0)
    )

    # --- Has dependents (children being housed, or family household) ---
    num_children = pd.to_numeric(safe_col(df, "num_children"), errors="coerce").fillna(0)
    is_family_head = safe_col(df, "head_of_household").apply(to_lower_str) == "family"
    df["has_dependents"] = ((num_children > 0) | is_family_head).astype(int)

    # --- Mental health / substance use (extracted-from-notes flags) ---
    df["mental_health"] = yes_no_to_flag(safe_col(df, "mental_health_raw"))
    df["substance_use"] = yes_no_to_flag(safe_col(df, "substance_use_raw"))

    # --- No income flag (inverse of "has an income source") ---
    has_income = yes_no_to_flag(safe_col(df, "has_income_source_raw"))
    df["no_income"] = has_income.apply(lambda v: pd.NA if pd.isna(v) else int(not v))

    # --- Income type ---
    df["income_type"] = safe_col(df, "income_type_raw")

    # Drop rows that turned out to be fully blank after all this (defensive).
    # (client_id is unusable here since it's blank in the source file --
    # see load_raw_rows -- so we rely on row_id, which is always set for
    # every row that passed the real-row check.)
    df = df.dropna(subset=["row_id"]).reset_index(drop=True)

    # --- year: extracted from last_contact_date's year (no single
    # "year" field exists in the Lanark data, so this is used as the
    # closest available proxy) ---
    df["year"] = pd.to_datetime(
        safe_col(df, "last_contact_date"), errors="coerce"
    ).dt.year

    # Keep only the columns that are actually in the mapping table.
    df = df[FINAL_COLUMNS]

    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="Anonymized_BNL_Data_-_Lanark_County_cleaned.xlsx",
        help="Path to the raw BNL Excel file.",
    )
    parser.add_argument(
        "--output",
        default="lanark_bnl_cleaned.csv",
        help="Path to write the cleaned CSV to.",
    )
    args = parser.parse_args()

    print(f"[info] Loading raw workbook: {args.input}")
    raw_df = load_raw_rows(args.input)
    print(f"[info] Loaded {len(raw_df)} raw client records.")

    cleaned_df = clean(raw_df)
    print(f"[info] Cleaned to {len(cleaned_df)} records, {cleaned_df.shape[1]} columns.")

    cleaned_df.to_csv(args.output, index=False)
    print(f"[info] Wrote cleaned data to: {args.output}")


if __name__ == "__main__":
    main()