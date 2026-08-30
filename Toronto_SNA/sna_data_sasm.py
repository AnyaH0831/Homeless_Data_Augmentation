"""
sna_pipeline_sasm.py
────────────────────
NEW pipeline using SASM (Small Area Synthetic Microdata) optimization.

HOW THIS DIFFERS FROM sna_pipeline.py (OLD PIPELINE)
─────────────────────────────────────────────────────
OLD:  generate_individuals()     — Gaussian copula probabilistic sampling
      • Samples each person's attributes from marginal distributions
      • Injects correlations via Cholesky decomposition
      • Result is APPROXIMATELY consistent with SNA aggregates

NEW:  generate_individuals_sasm() — Integer optimization (this file)
      • Finds combination counts by solving: minimize ||WX' - Y||²
      • Result is PROVABLY as close to SNA aggregates as the solver can get
      • Verifiable quality metric: d_p = ||WX' - Y||² (printed per year)

Everything else (data loading, interpolation, shelter flow calibration,
forecasting model) is IDENTICAL to the old pipeline so results are
directly comparable.

Usage:
    python sna_pipeline_sasm.py --local --local-flow source_data/toronto-shelter-system-flow.csv
    # Default: observed totals (typically ~100k+ rows across all years)
    
    python sna_pipeline_sasm.py --local --local-flow source_data/toronto-shelter-system-flow.csv --use-observed-totals
    # Generate individuals matching actual observed totals (~6400-8000 per year)
    
    python sna_pipeline_sasm.py --local --use-observed-totals --skip-model
    # Generate synthetic data only without training the forecasting model
    
    # NOTE: Fixed sample-size mode has been removed.
    # SASM now always uses observed/calibrated totals.
"""

import argparse
import io
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# Import the SASM generator only.
from sasm_generator import generate_individuals_sasm

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Identical to old pipeline — same data sources

CKAN_BASE = "https://ckan0.cf.opendata.inter.prod-toronto.ca"

PACKAGE_IDS = {
    2013: "2013-street-needs-assessment-results",
    2018: "2018-street-needs-assessment-results",
    2021: "2021-street-needs-assessment-results",
}

LOCAL_FILES = {
    2013: "source_data/2013-street-needs-assessment-results.xlsx",
    2018: "source_data/2018-street-needs-assessment-results.xlsx",
    2021: "source_data/2021-street-needs-assessment-results.xlsx",
}

# ── ROW MAPPING ───────────────────────────────────────────────────────────────
# Identical to old pipeline

ROW_MAP = {
    "TotalSurveys":                          "total_surveyed",
    "2_AgeAverage":                          "age_avg",
    "2_AgeCount":                            "age_count",
    "4_YearHomelessAverage":                 "years_homeless_avg",
    "26_GenderIdentityCount":                "gender_count",
    "26_GenderIdentityMale":                 "n_male",
    "26_GenderIdentityFemale":               "n_female",
    "26_GenderIdentityTransMale":            "n_trans_male",
    "26_GenderIdentityTransFemale":          "n_trans_female",
    "26_GenderIdentityTwoSpirit":            "n_two_spirit",
    "26_GenderIdentityNonBinary":            "n_nonbinary",
    "18_IndigenousCount":                    "indigenous_count",
    "18_FirstNations":                       "n_first_nations",
    "18_Metis":                              "n_metis",
    "18_Inuit":                              "n_inuit",
    "20_RaceEthnicityCount":                 "race_count",
    "20_RaceEthnicityBlackCanadianAmerican": "n_black_cdn",
    "20_RaceEthnicityBlackAfrican":          "n_black_african",
    "20_RaceEthnicityBlackAfroCaribbean":    "n_black_caribbean",
    "20_RaceEthnicityWhite":                 "n_white",
    "23_HealthChallengesCount":              "health_count",
    "23_MentalHealthIssueYes":               "n_mental_health",
    "23_SubstanceUseIssueYes":               "n_substance_use",
    "23_PhysicalLimitationYes":              "n_physical_health",
    "9_OverNightLocationCount":              "overnight_count",
    "9_OvernightLocationOutdoorsYes":        "n_outdoor",
    "9_EmergencyShelterYes":                 "n_emergency_shelter",
    "11_ImmigrantStatusCount":               "immigrant_count",
    "11_Immigrant":                          "n_immigrant",
    "11_Refugee":                            "n_refugee",
    "11_RefugeeClaimant":                    "n_refugee_claimant",
    "28_LGBTQS2Count":                       "lgbtq_count",
    "28_Yes":                                "n_lgbtq",
    "22_FosterCount":                        "foster_count",
    "22_Yes":                                "n_foster",
    "6_HousingLossCount":                    "housing_loss_count",
    "6_HousingLossNotEnoughIncome":          "n_housing_loss_income",
    "6_HousingLossMentalHealth":             "n_housing_loss_mental",
    "6_HousingLossSubstanceUse":             "n_housing_loss_substance",
    "29_IncomeCount":                        "income_count",
    "29_IncomeNoIncome":                     "n_no_income",
    "32_ServiceUseCount":                    "service_count",
    "32_PrisonOrJailYes":                    "n_incarceration",
}

RATIO_MAP = {
    ("n_male",                  "gender_count"):       "pct_male",
    ("n_female",                "gender_count"):       "pct_female",
    ("n_trans_male",            "gender_count"):       "pct_trans_male",
    ("n_trans_female",          "gender_count"):       "pct_trans_female",
    ("n_two_spirit",            "gender_count"):       "pct_two_spirit",
    ("n_nonbinary",             "gender_count"):       "pct_nonbinary",
    ("n_first_nations",         "indigenous_count"):   "pct_first_nations",
    ("n_metis",                 "indigenous_count"):   "pct_metis",
    ("n_inuit",                 "indigenous_count"):   "pct_inuit",
    ("n_white",                 "race_count"):         "pct_white",
    ("n_mental_health",         "health_count"):       "pct_mental_health",
    ("n_substance_use",         "health_count"):       "pct_substance_use",
    ("n_physical_health",       "health_count"):       "pct_physical_health",
    ("n_outdoor",               "overnight_count"):    "pct_outdoor_sleeping",
    ("n_emergency_shelter",     "overnight_count"):    "pct_emergency_shelter",
    ("n_immigrant",             "immigrant_count"):    "pct_immigrant",
    ("n_refugee",               "immigrant_count"):    "pct_refugee",
    ("n_refugee_claimant",      "immigrant_count"):    "pct_refugee_claimant",
    ("n_lgbtq",                 "lgbtq_count"):        "pct_lgbtq",
    ("n_foster",                "foster_count"):       "pct_foster_care_history",
    ("n_housing_loss_income",   "housing_loss_count"): "pct_housing_loss_income",
    ("n_housing_loss_mental",   "housing_loss_count"): "pct_housing_loss_health",
    ("n_housing_loss_substance","housing_loss_count"): "pct_housing_loss_substance",
    ("n_no_income",             "income_count"):       "pct_no_income",
    ("n_incarceration",         "service_count"):      "pct_incarceration_history",
}

PROPORTION_COLS = list(RATIO_MAP.values())


# ── DATA LOADING ──────────────────────────────────────────────────────────────
# Identical to old pipeline — shared helper functions

def _normalize_text(value) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())

def _pick_value_column(columns) -> str:
    normalized = {_normalize_text(col): col for col in columns}
    for candidate in ("totalaverage", "total"):
        if candidate in normalized:
            return normalized[candidate]
    for col in reversed(list(columns)):
        if str(col).strip() and not str(col).startswith("Unnamed"):
            return col
    return columns[-1]

def _normalize_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).map(_normalize_text)

def _find_value(df, *, row_terms=(), question_terms=(), response_terms=(),
                meta_value_terms=(), sum_matches=False) -> float:
    mask = pd.Series(True, index=df.index)
    if row_terms:
        row_norm = _normalize_series(df["row_name"])
        for t in row_terms:
            mask &= row_norm.str.contains(_normalize_text(t), na=False)
    if question_terms and "question" in df.columns:
        q_norm = _normalize_series(df["question"])
        for t in question_terms:
            mask &= q_norm.str.contains(_normalize_text(t), na=False)
    if response_terms and "response" in df.columns:
        r_norm = _normalize_series(df["response"])
        for t in response_terms:
            mask &= r_norm.str.contains(_normalize_text(t), na=False)
    if meta_value_terms and "meta_value" in df.columns:
        m_norm = _normalize_series(df["meta_value"])
        for t in meta_value_terms:
            mask &= m_norm.str.contains(_normalize_text(t), na=False)
    matches = df.loc[mask, "value"].fillna(0.0)
    if matches.empty:
        return 0.0
    return float(matches.sum() if sum_matches else matches.iloc[0])


def compute_derived(agg: dict) -> dict:
    """Compute composite fields — identical to old pipeline."""
    rc = max(agg.get("race_count", 1), 1)
    # Combine three Black subgroups into one pct_black
    agg["pct_black"] = float(np.clip(
        (agg.get("n_black_cdn", 0) + agg.get("n_black_african", 0) + agg.get("n_black_caribbean", 0)) / rc,
        0.0, 1.0
    ))
    # Also store the raw total for SASM's Y extraction
    agg["n_black_total"] = (
        agg.get("n_black_cdn", 0) + agg.get("n_black_african", 0) + agg.get("n_black_caribbean", 0)
    )

    ic = max(agg.get("indigenous_count", 1), 1)
    agg["pct_indigenous"] = float(np.clip(
        (agg.get("n_first_nations", 0) + agg.get("n_metis", 0) + agg.get("n_inuit", 0)) / ic,
        0.0, 1.0
    ))
    # Also store the raw total for SASM's Y extraction
    agg["n_indigenous_total"] = (
        agg.get("n_first_nations", 0) + agg.get("n_metis", 0) + agg.get("n_inuit", 0)
    )

    avg = max(agg.get("years_homeless_avg", 3.0), 0.1)
    agg["pct_chronic"] = float(np.clip(1 - np.exp(-avg / 3.5), 0.1, 0.9))

    agg["pct_other_race"] = float(np.clip(
        1 - agg.get("pct_black", 0) - agg.get("pct_white", 0) - agg.get("pct_indigenous", 0),
        0.01, 0.99
    ))
    agg["pct_trans_nonbinary"] = float(np.clip(
        agg.get("pct_trans_male", 0) + agg.get("pct_trans_female", 0) +
        agg.get("pct_two_spirit", 0) + agg.get("pct_nonbinary", 0),
        0.01, 0.99
    ))
    agg.setdefault("age_std", 14.0)
    return agg


def apply_realistic_bounds(agg: dict, year: int) -> dict:
    """
    Apply realistic bounds based on literature and domain knowledge.
    
    For years 2013-2017 (interpolated, no SNA data), use external prevalence
    rates from published homeless health studies instead of interpolation.
    
    Later years (2018+) use clipping to constrain to literature ranges.
    """
    # External prevalence rates from published studies (CDC, CCSA, HUD, Burt 2005, Fazel & Geddes 2018)
    # These are used for interpolated years (2013-2017) where SNA data is missing
    EXTERNAL_RATES = {
        2013: {'mental_health': 0.32, 'substance_use': 0.24, 'outdoor_sleeping': 0.14, 
               'foster_care': 0.10, 'incarceration': 0.08},
        2014: {'mental_health': 0.33, 'substance_use': 0.24, 'outdoor_sleeping': 0.15,
               'foster_care': 0.10, 'incarceration': 0.08},
        2015: {'mental_health': 0.34, 'substance_use': 0.25, 'outdoor_sleeping': 0.16,
               'foster_care': 0.11, 'incarceration': 0.09},
        2016: {'mental_health': 0.36, 'substance_use': 0.26, 'outdoor_sleeping': 0.17,
               'foster_care': 0.11, 'incarceration': 0.09},
        2017: {'mental_health': 0.38, 'substance_use': 0.27, 'outdoor_sleeping': 0.18,
               'foster_care': 0.12, 'incarceration': 0.09},
    }
    
    # For early years, use external prevalence rates instead of interpolation
    if year in EXTERNAL_RATES:
        rates = EXTERNAL_RATES[year]
        agg["pct_mental_health"] = float(rates['mental_health'])
        agg["pct_substance_use"] = float(rates['substance_use'])
        agg["pct_outdoor_sleeping"] = float(rates['outdoor_sleeping'])
        agg["pct_foster_care_history"] = float(rates['foster_care'])
        agg["pct_incarceration_history"] = float(rates['incarceration'])
    else:
        # For 2018+, use literature-based bounds (clipping)
        # Mental health: literature range 15-60%
        agg["pct_mental_health"] = float(np.clip(agg.get("pct_mental_health", 0.35), 0.15, 0.60))
        
        # Substance use: literature range 15-45%
        agg["pct_substance_use"] = float(np.clip(agg.get("pct_substance_use", 0.25), 0.15, 0.45))
        
        # Outdoor sleeping: literature range 10-35%
        agg["pct_outdoor_sleeping"] = float(np.clip(agg.get("pct_outdoor_sleeping", 0.18), 0.10, 0.35))
        
        # Foster care history: literature range 5-30%
        agg["pct_foster_care_history"] = float(np.clip(agg.get("pct_foster_care_history", 0.12), 0.05, 0.30))
        
        # Incarceration history: literature range 5-25%
        agg["pct_incarceration_history"] = float(np.clip(agg.get("pct_incarceration_history", 0.08), 0.05, 0.25))
    
    return agg


def fetch_xlsx_from_api(year: int) -> bytes:
    resp = requests.get(
        f"{CKAN_BASE}/api/3/action/package_show",
        params={"id": PACKAGE_IDS[year]}, timeout=30,
    )
    resp.raise_for_status()
    resources = resp.json()["result"]["resources"]
    for res in resources:
        fmt = (res.get("format") or "").lower()
        url = res.get("url", "")
        if "xlsx" in fmt or url.endswith(".xlsx"):
            print(f"  [{year}] Downloading: {url}")
            dl = requests.get(url, timeout=60)
            dl.raise_for_status()
            return dl.content
    raise ValueError(f"No xlsx resource found for SNA {year}.")


def load_sna_xlsx(source) -> pd.DataFrame:
    """Parse Export + Key-Rows sheets into a merged DataFrame."""
    buf = io.BytesIO(source) if isinstance(source, bytes) else source
    export = pd.read_excel(buf, sheet_name="Export")
    export.columns = [str(c).strip() for c in export.columns]
    row_col   = export.columns[0]
    value_col = _pick_value_column(export.columns[1:])
    export    = export[[row_col, value_col]].rename(columns={row_col: "row_name", value_col: "value"})

    key_rows = pd.read_excel(buf, sheet_name="Key-Rows")
    key_rows.columns = [str(c).strip() for c in key_rows.columns]
    key_rows = key_rows.iloc[:, :5].copy()
    key_rows.columns = ["row_name", "question", "response", "meta_value", "notes"]

    df = key_rows.merge(export, on="row_name", how="left")
    df["row_name"] = df["row_name"].astype(str).str.strip()
    df["value"]    = pd.to_numeric(df["value"], errors="coerce")
    return df


def extract_aggregates(df: pd.DataFrame) -> dict:
    """Extract all aggregate statistics from a loaded SNA sheet."""
    lookup = df.assign(_rn=_normalize_series(df["row_name"])).set_index("_rn")["value"].to_dict()

    def from_rows(*labels):
        for label in labels:
            v = lookup.get(_normalize_text(label))
            if v is not None and not pd.isna(v):
                return float(v)
        return 0.0

    agg = {key: from_rows(sna_row) for sna_row, key in ROW_MAP.items()}

    # Semantic fallbacks for schema differences across years
    agg["total_surveyed"] = max(agg.get("total_surveyed", 0.0),
        _find_value(df, question_terms=("total surveys completed",), meta_value_terms=("count",)))
    
    # Years homeless — handle 2013 (1_YEARSHOMELESS), 2018 (4_TIMEHOMELESSAVERAGE in days), 2021 (4_YearHomelessAverage)
    yh_2013 = from_rows("1_YEARSHOMELESS", "1_YEARSHOMELESSAVERAGE")
    yh_2018 = from_rows("4_TIMEHOMELESSAVERAGE") / 365.0  # Convert days to years
    yh_fallback = _find_value(df, question_terms=("how long you have been homeless",), meta_value_terms=("average",))
    agg["years_homeless_avg"] = max(agg.get("years_homeless_avg", 0.0), yh_2013, yh_2018, yh_fallback)
    
    # Age — handle 2013 (3_AGE), 2018 (2_AGEAVERAGE), 2021 (2_AgeAverage)
    age_2013 = from_rows("3_AGE")
    age_2018 = from_rows("2_AGEAVERAGE")
    age_fallback = _find_value(df, question_terms=("how old are you",), meta_value_terms=("average",))
    agg["age_avg"] = max(agg.get("age_avg", 0.0), age_2013, age_2018, age_fallback)
    
    # Gender — handle 2013 (4_MALE/FEMALE), 2018 (15_MALE/FEMALE)
    agg["n_male"] = max(agg.get("n_male", 0.0),
        from_rows("4_MALE", "15_MALE"),
        _find_value(df, question_terms=("gender",), response_terms=("male",), meta_value_terms=("count",)))
    agg["n_female"] = max(agg.get("n_female", 0.0),
        from_rows("4_FEMALE", "15_FEMALE"),
        _find_value(df, question_terms=("gender",), response_terms=("female",), meta_value_terms=("count",)))
    
    agg["n_mental_health"] = max(agg.get("n_mental_health", 0.0),
        _find_value(df, question_terms=("mental health",), response_terms=("yes",), meta_value_terms=("count",)))
    agg["n_substance_use"] = max(agg.get("n_substance_use", 0.0),
        _find_value(df, question_terms=("substance use",), response_terms=("yes",), meta_value_terms=("count",)))
    agg["n_outdoor"] = max(agg.get("n_outdoor", 0.0),
        _find_value(df, question_terms=("overnight location",), response_terms=("outdoor",), meta_value_terms=("count",)))
    agg["n_white"] = max(agg.get("n_white", 0.0),
        _find_value(df, question_terms=("race", "ethnicity"), response_terms=("white",), meta_value_terms=("count",)))
    agg["n_lgbtq"] = max(agg.get("n_lgbtq", 0.0),
        _find_value(df, question_terms=("lgbtq",), response_terms=("yes",), meta_value_terms=("count",)))
    agg["n_foster"] = max(agg.get("n_foster", 0.0),
        _find_value(df, question_terms=("foster",), response_terms=("yes",), meta_value_terms=("count",)))
    agg["n_incarceration"] = max(agg.get("n_incarceration", 0.0),
        _find_value(df, question_terms=("prison", "jail"), response_terms=("yes",), meta_value_terms=("count",)))
    agg["n_no_income"] = max(agg.get("n_no_income", 0.0),
        _find_value(df, question_terms=("income source",), response_terms=("no income",), meta_value_terms=("count",)))

    for (num_key, den_key), pct_key in RATIO_MAP.items():
        num = agg.get(num_key, 0.0)
        den = max(agg.get(den_key, 1.0), 1.0)
        agg[pct_key] = float(np.clip(num / den, 0.0, 1.0))

    return compute_derived(agg)


def load_all_years(use_local: bool = False) -> dict:
    results = {}
    for year in PACKAGE_IDS:
        print(f"Loading SNA {year}...")
        raw = None
        if not use_local:
            try:
                raw = fetch_xlsx_from_api(year)
                print(f"  [{year}] Loaded from API.")
            except Exception as e:
                print(f"  [{year}] API failed ({e}), trying local...")
        if raw is None:
            lp = Path(LOCAL_FILES[year])
            if not lp.exists():
                raise FileNotFoundError(f"Local file not found: {lp}")
            raw = lp.read_bytes()
            print(f"  [{year}] Loaded from local file.")
        df_sheet = load_sna_xlsx(raw)
        results[year] = extract_aggregates(df_sheet)
        results[year]["year"] = year
    return results


# ── INTERPOLATION / EXTRAPOLATION ─────────────────────────────────────────────
# Identical to old pipeline

def interpolate_aggregates(observed: dict, all_years: list) -> pd.DataFrame:
    obs_df = pd.DataFrame(observed).T.astype(float)
    obs_df.index = obs_df.index.astype(int)
    obs_df.index.name = "year"

    interp = obs_df.reindex(pd.Index(all_years, name="year"))
    interp = interp.interpolate(method="index", limit_direction="both")

    obs_years = sorted(observed.keys())
    first_yr, last_yr = obs_years[0], obs_years[-1]
    span = last_yr - first_yr

    for col in interp.columns:
        if col == "year":
            continue
        v_first = observed[first_yr].get(col)
        v_last  = observed[last_yr].get(col)
        if v_first is None or v_last is None:
            continue
        slope = (v_last - v_first) / span if span else 0
        for yr in all_years:
            if yr > last_yr:
                val = v_last + slope * (yr - last_yr)
                if col in PROPORTION_COLS or col.startswith("pct_"):
                    val = float(np.clip(val, 0.01, 0.99))
                interp.loc[yr, col] = val

    interp["total_surveyed"] = interp["total_surveyed"].round().astype(int)
    return interp


# ── REGION-YEAR FEATURES ──────────────────────────────────────────────────────
# Same structure as old pipeline; column names match because SASM output
# uses the same column names (mental_health, substance_use, etc.)




# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SASM pipeline for synthetic data generation")
    parser.add_argument("--local", action="store_true",
                        help="Load SNA xlsx locally instead of API")
    args = parser.parse_args()

    all_years = list(range(2013, 2027))

    # ── Step 1: Load SNA aggregates ───────────────────────────────────────────
    print("=" * 65)
    print("STEP 1: Loading SNA aggregate data")
    print("=" * 65)
    observed = load_all_years(use_local=args.local)
    agg_df = interpolate_aggregates(observed, all_years)

    # ── Step 2: Apply realistic bounds to SNA-derived proportions ────────────
    print("\n" + "=" * 65)
    print("STEP 2: Applying realistic bounds to SNA-derived proportions")
    print("=" * 65)
    for year in agg_df.index:
        agg_df.loc[year] = apply_realistic_bounds(agg_df.loc[year].to_dict(), year)

    print(agg_df[["total_surveyed", "pct_mental_health", "pct_outdoor_sleeping", "pct_chronic"]].round(3))

    # ── Step 3: SASM individual generation ───────────────────────────────────
    # This is the KEY difference from the old pipeline.
    # Instead of copula sampling, we solve an integer optimization problem
    # to find combination counts X' that minimize ||WX' - Y||²
    observed_years = sorted(observed.keys())
    print("\n" + "=" * 65)
    print("STEP 3: SASM optimization-based individual generation")
    print("  (minimize ||WX' - Y||² per year)")
    print(f"  Using observed SNA years only: {observed_years}")
    print("=" * 65)
    df_individuals = generate_individuals_sasm(
        agg_df,
        use_observed_totals=True,
        years=observed_years,
    )
    df_individuals.to_csv("synthetic data/sasm_synthetic_individuals.csv", index=False)
    print(f"\nSaved: synthetic data/sasm_synthetic_individuals.csv  ({len(df_individuals):,} rows)")
    print("\n" + "=" * 65)
    print("Done! Synthetic data generated using SASM approach.")
    print("=" * 65)


if __name__ == "__main__":
    main()