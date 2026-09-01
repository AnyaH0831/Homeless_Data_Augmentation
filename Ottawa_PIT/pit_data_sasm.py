"""
Ottawa PIT synthetic microdata generator using the SASM optimizer.

This follows the same structure as the Toronto SNA pipeline:
1. Read Ottawa PIT aggregate survey rows.
2. Convert them to a one-row-per-year aggregate table.
3. Pass the table to the SASM generator's `generate_individuals_sasm()`.
4. Save synthetic individual-level records.

The chronic flag is computed using the rule:
    chronic_homeless = 1 if (days_homeless > 180) or (episodes > 3)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from sasm_generator import generate_individuals_sasm

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "source_data" / "Point_in_Time_Count_EN.csv"
OUT_PATH = ROOT / "synthetic_data" / "sasm_synthetic_individuals.csv"


def parse_period(value: object) -> int:
    if pd.isna(value):
        return -1
    match = re.search(r"(\d{4})$", str(value).strip())
    return int(match.group(1)) if match else -1


def normalize_text(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower()).strip()


def get_share(df: pd.DataFrame, *, sector: str, year: int, question_contains: str, response_value: str) -> float:
    subset = df[
        (df["Sector"] == sector)
        & (df["year"] == year)
        & df["Question"].fillna("").astype(str).str.contains(question_contains, case=False, na=False)
    ].copy()
    if subset.empty:
        return 0.0

    subset = subset[
        subset["Response"].fillna("").astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
        .str.contains(normalize_text(response_value), case=False, na=False)
    ]
    if subset.empty:
        return 0.0

    return float(subset["Percent"].iloc[0]) / 100.0


def get_total_for_year(df: pd.DataFrame, *, sector: str, year: int) -> int:
    subset = df[(df["Sector"] == sector) & (df["year"] == year)].copy()
    if subset.empty:
        return 0
    denom = subset["Denominator"].dropna().astype(float)
    if denom.empty:
        return 0
    return int(denom.max())


def weighted_age_midpoint(df: pd.DataFrame, *, sector: str, year: int) -> float:
    age_map = {
        "13 to 18 years old": 15.5,
        "18 to 24 years old": 21,
        "25 to 49 years old": 37,
        "50 to 64 years old": 57,
        "Over 65 years old": 70,
    }
    total = 0.0
    weighted = 0.0
    for response, midpoint in age_map.items():
        share = get_share(
            df,
            sector=sector,
            year=year,
            question_contains="how old are you",
            response_value=response,
        )
        if share > 0:
            weighted += share * midpoint
            total += share
    if total <= 0:
        return 35.0
    return float(weighted / total)


def estimate_years_homeless(df: pd.DataFrame, *, sector: str, year: int, current_age_avg: float) -> float:
    """
    Estimate average years homeless from:
    1. Age when first experienced homelessness (midpoint)
    2. Current average age
    3. Time in Ottawa (if available)
    
    This approximates years_homeless_avg using available Ottawa data.
    """
    first_homeless_age_map = {
        "Under 12 years old": 8,
        "13 to 17 years old": 15,
        "18 to 24 years old": 21,
        "25 to 49 years old": 37,
        "50 to 64 years old": 57,
        "Over 65 years old": 70,
    }
    
    total_weight = 0.0
    weighted_first_age = 0.0
    for response, midpoint in first_homeless_age_map.items():
        share = get_share(
            df,
            sector=sector,
            year=year,
            question_contains="how old were you the first time you experienced homelessness",
            response_value=response,
        )
        if share > 0:
            weighted_first_age += share * midpoint
            total_weight += share
    
    if total_weight <= 0:
        return 4.0  # Default fallback
    
    avg_first_homeless_age = weighted_first_age / total_weight
    years_since_first = max(current_age_avg - avg_first_homeless_age, 0.5)
    
    # Adjust downward: not all time since first homelessness was continuous
    # Use the 180-day threshold as a proxy: if >56.8% are >180 days in past year,
    # assume they've been homeless longer on average
    pt_over_180 = get_share(
        df,
        sector=sector,
        year=year,
        question_contains="how much time have you experienced homelessness over the past year",
        response_value="More than 180 days",
    )
    
    # Estimate: if >50% are >180 days/year, they've likely been homeless 3+ years on average
    if pt_over_180 > 0.5:
        return float(min(years_since_first, 8.0))  # Cap at 8 years as reasonable max
    else:
        return float(min(years_since_first * 0.5, 3.0))  # Shorter tenure for less chronic


def build_year_aggregate(df: pd.DataFrame, *, sector: str, year: int) -> dict:
    total = get_total_for_year(df, sector=sector, year=year)
    if total <= 0:
        return {}

    pt_180 = get_share(
        df,
        sector=sector,
        year=year,
        question_contains="how much time have you experienced homelessness over the past year",
        response_value="More than 180 days",
    )
    pt_episodes = get_share(
        df,
        sector=sector,
        year=year,
        question_contains="how many different times have you experienced homelessness in the past year",
        response_value="3 or more episodes of homelessness",
    )
    
    # If episodes question is missing (2021, 2024), fall back to 180-day rule only
    if pt_episodes == 0.0:
        pct_chronic = pt_180
        has_episodes = False
    else:
        pct_chronic = 1.0 - (1.0 - pt_180) * (1.0 - pt_episodes)
        has_episodes = True

    # Extract education levels
    pct_less_than_hs = get_share(
        df,
        sector=sector,
        year=year,
        question_contains="what is the highest level of education",
        response_value="Less than high school",
    )
    pct_hs_graduate = get_share(
        df,
        sector=sector,
        year=year,
        question_contains="what is the highest level of education",
        response_value="High school graduate",
    )
    pct_some_post_sec = get_share(
        df,
        sector=sector,
        year=year,
        question_contains="what is the highest level of education",
        response_value="Some post-secondary",
    )
    pct_post_sec_higher = get_share(
        df,
        sector=sector,
        year=year,
        question_contains="what is the highest level of education",
        response_value="Post-secondary or higher",
    )
    # Normalize education to sum to 1.0
    edu_total = pct_less_than_hs + pct_hs_graduate + pct_some_post_sec + pct_post_sec_higher
    if edu_total > 0:
        pct_less_than_hs /= edu_total
        pct_hs_graduate /= edu_total
        pct_some_post_sec /= edu_total
        pct_post_sec_higher /= edu_total
    else:
        # Default distribution if missing
        pct_less_than_hs = 0.35
        pct_hs_graduate = 0.27
        pct_some_post_sec = 0.12
        pct_post_sec_higher = 0.20

    # Extract dependents
    pct_has_dependents = get_share(
        df,
        sector=sector,
        year=year,
        question_contains="dependents staying",
        response_value="Accompanied by dependent",
    )

    # Extract income sources (consolidated into categories)
    pct_disability = get_share(
        df,
        sector=sector,
        year=year,
        question_contains="what are your sources of income",
        response_value="Disability benefit",
    )
    pct_employment = (
        get_share(df, sector=sector, year=year, question_contains="what are your sources of income", response_value="Full-time employment")
        + get_share(df, sector=sector, year=year, question_contains="what are your sources of income", response_value="Part-time employment")
        + get_share(df, sector=sector, year=year, question_contains="what are your sources of income", response_value="Casual employment")
    )
    pct_welfare = get_share(
        df,
        sector=sector,
        year=year,
        question_contains="what are your sources of income",
        response_value="Ontario Works",
    )
    pct_informal = get_share(
        df,
        sector=sector,
        year=year,
        question_contains="what are your sources of income",
        response_value="Informal income",
    )
    pct_no_income = get_share(
        df,
        sector=sector,
        year=year,
        question_contains="what are your sources of income",
        response_value="No income",
    )
    # For "other" income, capture remaining types
    pct_other_income = (
        get_share(df, sector=sector, year=year, question_contains="what are your sources of income", response_value="Employment insurance")
        + get_share(df, sector=sector, year=year, question_contains="what are your sources of income", response_value="Child and family tax")
        + get_share(df, sector=sector, year=year, question_contains="what are your sources of income", response_value="GST/HST")
        + get_share(df, sector=sector, year=year, question_contains="what are your sources of income", response_value="Seniors")
        + get_share(df, sector=sector, year=year, question_contains="what are your sources of income", response_value="Money from family")
        + get_share(df, sector=sector, year=year, question_contains="what are your sources of income", response_value="Other")
    )

    row = {
        "total_surveyed": int(total),
        "pct_male": get_share(df, sector=sector, year=year, question_contains="what gender do you identify with", response_value="Men"),
        "pct_female": get_share(df, sector=sector, year=year, question_contains="what gender do you identify with", response_value="Women"),
        "pct_trans_nonbinary": get_share(df, sector=sector, year=year, question_contains="what gender do you identify with", response_value="Non-binary and other identities"),
        "pct_white": get_share(df, sector=sector, year=year, question_contains="do you identify with any racial identities", response_value="Non-racialized"),
        "pct_indigenous": 0.0,
        "pct_black": 0.0,
        "pct_other_race": max(0.0, 1.0 - get_share(df, sector=sector, year=year, question_contains="do you identify with any racial identities", response_value="Non-racialized")),
        "pct_less_than_hs": float(pct_less_than_hs),
        "pct_hs_graduate": float(pct_hs_graduate),
        "pct_some_post_sec": float(pct_some_post_sec),
        "pct_post_sec_higher": float(pct_post_sec_higher),
        "pct_has_dependents": float(min(max(pct_has_dependents, 0.0), 1.0)),
        "pct_mental_health": get_share(df, sector=sector, year=year, question_contains="health challenges at this time", response_value="Mental health issue"),
        "pct_substance_use": get_share(df, sector=sector, year=year, question_contains="health challenges at this time", response_value="Substance use issue"),
        "pct_outdoor_sleeping": get_share(df, sector=sector, year=year, question_contains="where are you staying tonight", response_value="Unsheltered")
        + get_share(df, sector=sector, year=year, question_contains="where are you staying tonight", response_value="Makeshift Shelter, Tent or Shack"),
        "pct_chronic": float(min(max(pct_chronic, 0.0), 1.0)),
        "age_avg": weighted_age_midpoint(df, sector=sector, year=year),
        "age_std": 14.0,
        "years_homeless_avg": estimate_years_homeless(df, sector=sector, year=year, current_age_avg=weighted_age_midpoint(df, sector=sector, year=year)),
        "pct_lgbtq": get_share(df, sector=sector, year=year, question_contains="how do you describe your sexual orientation", response_value="Gay, lesbian and other"),
        "pct_disability_income": float(pct_disability),
        "pct_employment_income": float(pct_employment),
        "pct_welfare_income": float(pct_welfare),
        "pct_informal_income": float(pct_informal),
        "pct_no_income": float(pct_no_income),
        "pct_other_income": float(pct_other_income),
    }
    return row


def load_pit_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df.columns = [str(c).strip() for c in df.columns]
    df["year"] = df["Period"].map(parse_period)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Ottawa PIT synthetic individuals with SASM")
    parser.add_argument("--sector", default="All", help="Sector filter to build, defaults to All")
    args = parser.parse_args()

    df = load_pit_data()
    years = sorted(df["year"].dropna().astype(int).unique().tolist())

    # Check which years have the episodes question for diagnostics
    has_episodes_by_year = {}
    for year in years:
        subset = df[
            (df["Sector"] == args.sector) 
            & (df["year"] == year)
            & df["Question"].fillna("").astype(str).str.contains("how many different times have you experienced", case=False, na=False)
        ]
        has_episodes_by_year[year] = len(subset) > 0

    rows = []
    for year in years:
        agg = build_year_aggregate(df, sector=args.sector, year=year)
        if agg:
            rows.append({
                "year": year, 
                "has_episodes_data": has_episodes_by_year.get(year, False),
                **agg
            })

    if not rows:
        raise RuntimeError(f"No PIT rows found for sector '{args.sector}' in {DATA_PATH}")

    agg_df = pd.DataFrame(rows).set_index("year").sort_index()
    
    # Print diagnostic info
    print("\n=== Chronic Calculation Method ===")
    for year in agg_df.index:
        has_ep = agg_df.loc[year, "has_episodes_data"]
        method = "(180 days) OR (episodes)" if has_ep else "(180 days only - episodes Q missing)"
        print(f"  {year}: {method}")
    print()
    
    agg_df = agg_df.drop(columns=["has_episodes_data"]).astype(float)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    generated = generate_individuals_sasm(
        agg_df,
        use_observed_totals=True,
        years=list(agg_df.index),
    )

    generated.to_csv(OUT_PATH, index=False)
    print(f"Saved synthetic PIT microdata to: {OUT_PATH}")
    print(generated.head())


if __name__ == "__main__":
    main()
