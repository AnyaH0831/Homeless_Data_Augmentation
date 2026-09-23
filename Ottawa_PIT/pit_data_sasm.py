"""
Ottawa PIT synthetic microdata generator using the SASM optimizer.

This follows the same structure as the Toronto SNA pipeline:
1. Read Ottawa PIT aggregate survey rows.
2. Convert them to a one-row-per-year aggregate table.
3. Pass the table to the SASM generator's `generate_individuals_sasm()`.
4. Save synthetic individual-level records.

The chronic flag is computed using the rule:
    chronic_homeless = 1 if (days_homeless > 180) or (episodes > 3)

CHANGES FROM PRIOR VERSION
────────────────────────────
1. ALLOWED_YEARS whitelist added. Ottawa has run exactly three PiT counts:
   2018, 2021, and 2024. Rather than trusting whatever distinct years happen
   to show up after regex-parsing the `Period` column (which could include a
   combined/rolled-up row or a mis-parsed value), we now explicitly filter to
   only these three years. Anything else found in the file is reported and
   dropped rather than silently included.
2. get_share() now does an EXACT match on the normalized response string
   instead of substring containment. Under the old `.str.contains(...)`
   check, querying for "Men" would also match "Women" (since "women"
   contains "men" as a substring after normalization), and because only
   `.iloc[0]` was returned, this could silently return the wrong bucket's
   percentage depending on row order. Every response lookup now requires an
   exact match after normalization.
3. VERIFIED against the actual uploaded CSV (Point_in_Time_Count_EN.csv):
   - The racial-identity question only ever has two responses in this data:
     "Non-racialized" and "Racialized". There is NO Black/Indigenous
     breakdown anywhere in the file, in any year or sector. An earlier draft
     of this script tried to query for "Black" and "Indigenous" responses —
     that would have silently returned 0.0 forever (those labels don't
     exist), which is worse than an honest limitation because it looks like
     real extraction. pct_black and pct_indigenous are therefore explicitly
     fixed at 0.0 with this data source; pct_other_race absorbs the entire
     "Racialized" share undifferentiated. If a Black/Indigenous breakdown is
     needed, it isn't available from this file and would require different
     source data.
   - get_share() now does an EXACT match on the normalized response string
     instead of substring containment, fixing a real collision where a query
     for "Men" would also match "Women" (since "women" contains "men" as a
     substring). All response_value strings below have been updated to the
     FULL literal label text as it actually appears in the CSV — several
     income-source labels are longer than what a truncated query would catch
     (e.g. the real response is "Informal income (eg. bottle returns)", not
     "Informal income"), so exact matching required fixing those strings too.
   - Outdoor-sleeping terminology changed between survey years: 2018 uses
     "Unsheltered" + "Makeshift Shelter, Tent or Shack"; 2021/2024 replaced
     the second label with "Encampment". The old query only ever summed the
     2018-era labels, silently undercounting unsheltered/outdoor homelessness
     in 2021 and 2024. Now sums all three (mutually exclusive per year).
   - Employment income format changed too: 2018 uses one generic
     "Employment" response; 2021/2024 split it into "Full-time employment" /
     "Part-time employment" / "Casual employment". Now sums all four labels
     (only the relevant ones are ever populated in a given year).
   - The "dependents staying" question and the "No income" response do not
     exist at all in the 2018 survey (confirmed across every sector) — these
     will genuinely be 0.0 for 2018. That reflects what was actually asked
     that year, not a bug, and isn't fixable from this file alone.
   - Age, education, and first-homeless-age category labels match this
     script's expectations for sector="All" (the default) across all three
     years. The "Family" and "Youth" sectors use different, coarser category
     labels for age and education that will NOT match if you run with
     --sector Family or --sector Youth — those runs will silently fall back
     to default distributions for those fields. Not fixed here since it's
     out of scope for the default "All" sector this script targets.
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

# Ottawa has conducted exactly three Point-in-Time counts: 2018, 2021, 2024.
# Only these years are processed — no interpolated/extrapolated or otherwise
# spurious "years" parsed from the Period column are allowed through.
ALLOWED_YEARS = {2018, 2021, 2024}


def parse_period(value: object) -> int:
    if pd.isna(value):
        return -1
    match = re.search(r"(\d{4})$", str(value).strip())
    return int(match.group(1)) if match else -1


def normalize_text(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower()).strip()


def get_share(df: pd.DataFrame, *, sector: str, year: int, question_contains: str, response_value: str) -> float:
    """
    Return the Percent (as a 0-1 share) of respondents in `sector`/`year`
    whose Question contains `question_contains` and whose Response EXACTLY
    matches `response_value` after normalization (lowercase, alnum-only).

    Uses exact match rather than substring containment: under containment,
    a query for "Men" would incorrectly also match "Women" (since "women"
    contains "men" as a substring), silently returning the wrong bucket.
    """
    subset = df[
        (df["Sector"] == sector)
        & (df["year"] == year)
        & df["Question"].fillna("").astype(str).str.contains(question_contains, case=False, na=False)
    ].copy()
    if subset.empty:
        return 0.0

    target = normalize_text(response_value)
    normalized_responses = subset["Response"].fillna("").astype(str).map(normalize_text)
    subset = subset[normalized_responses == target]
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
    # Use the 180-day threshold as a proxy: if >50% are >180 days in past year,
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
        # Default distribution if missing (e.g. 2018 has no education
        # question at all). Education is a hard partition in the SASM
        # attribute space, so these four values MUST sum to exactly 1.0 —
        # the previous defaults (0.35+0.27+0.12+0.20 = 0.94) summed to only
        # 94%, creating the same kind of mathematically-forced inconsistency
        # that the gender fix above addresses, and it was the single largest
        # source of 2018's poor fit (every education category showed an
        # identical +21-person residual). Same relative proportions,
        # renormalized to sum to 1.0.
        pct_less_than_hs = 0.35 / 0.94
        pct_hs_graduate = 0.27 / 0.94
        pct_some_post_sec = 0.12 / 0.94
        pct_post_sec_higher = 0.20 / 0.94

    # Extract dependents. NOTE: this question does not exist at all in the
    # 2018 survey (confirmed across every sector) — pct_has_dependents will
    # genuinely be 0.0 for 2018, reflecting a question that wasn't asked that
    # year, not a bug.
    pct_has_dependents = get_share(
        df,
        sector=sector,
        year=year,
        question_contains="dependents staying",
        response_value="Accompanied by dependent(s)",
    )

    # Extract income sources (consolidated into categories).
    # Response labels verified against the real CSV. All queries use the full
    # literal label text since get_share() now requires an exact match.
    pct_disability = get_share(
        df,
        sector=sector,
        year=year,
        question_contains="what are your sources of income",
        response_value="Disability benefit",
    )
    # 2018 reports a single generic "Employment" response; 2021/2024 split it
    # into Full-time/Part-time/Casual employment. Summing all four is safe
    # since only the relevant subset is ever populated in a given year.
    pct_employment = (
        get_share(df, sector=sector, year=year, question_contains="what are your sources of income", response_value="Full-time employment")
        + get_share(df, sector=sector, year=year, question_contains="what are your sources of income", response_value="Part-time employment")
        + get_share(df, sector=sector, year=year, question_contains="what are your sources of income", response_value="Casual employment")
        + get_share(df, sector=sector, year=year, question_contains="what are your sources of income", response_value="Employment")
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
        response_value="Informal income (eg. bottle returns)",
    )
    # "No income" does not exist as a response category in the 2018 survey —
    # pct_no_income will genuinely be 0.0 for 2018, not a bug.
    pct_no_income = get_share(
        df,
        sector=sector,
        year=year,
        question_contains="what are your sources of income",
        response_value="No income",
    )
    # Remaining income types, using the full literal response text for each.
    pct_other_income = (
        get_share(df, sector=sector, year=year, question_contains="what are your sources of income", response_value="Employment insurance")
        + get_share(df, sector=sector, year=year, question_contains="what are your sources of income", response_value="Child and family tax benefits")
        + get_share(df, sector=sector, year=year, question_contains="what are your sources of income", response_value="GST/HST refund")
        + get_share(df, sector=sector, year=year, question_contains="what are your sources of income", response_value="Seniors' benefits")
        + get_share(df, sector=sector, year=year, question_contains="what are your sources of income", response_value="Money from family/friends")
        + get_share(df, sector=sector, year=year, question_contains="what are your sources of income", response_value="Student loans")
        + get_share(df, sector=sector, year=year, question_contains="what are your sources of income", response_value="Other")
    )

    # Gender. These are a hard partition in the SASM attribute space — every
    # synthetic individual is assigned exactly one of male/female/trans_
    # nonbinary — but the raw survey shares never sum to exactly 1.0 (missing
    # "prefer not to say"/non-response is dropped from these three buckets).
    # Confirmed against this CSV: gender shares sum to 98.7% (2018), 98.5%
    # (2021), and just 93.2% (2024). Left unnormalized, this creates a
    # mathematically infeasible target for the optimizer — it's told the
    # three gender categories must sum to e.g. 93.2% of the total AND that
    # everyone must fall into one of those three categories AND that the
    # grand total is 100%, which cannot all be true at once. That forced
    # compromise directly produced the elevated quality-metric errors,
    # especially the ~7% mean error seen for 2024. Renormalizing to sum to
    # 1.0 (same treatment education already gets) removes this inconsistency.
    pct_male_raw = get_share(df, sector=sector, year=year, question_contains="what gender do you identify with", response_value="Men")
    pct_female_raw = get_share(df, sector=sector, year=year, question_contains="what gender do you identify with", response_value="Women")
    pct_trans_raw = get_share(df, sector=sector, year=year, question_contains="what gender do you identify with", response_value="Non-binary and other identities")
    gender_total = pct_male_raw + pct_female_raw + pct_trans_raw
    if gender_total > 0:
        pct_male = pct_male_raw / gender_total
        pct_female = pct_female_raw / gender_total
        pct_trans_nonbinary = pct_trans_raw / gender_total
    else:
        pct_male, pct_female, pct_trans_nonbinary = 0.65, 0.28, 0.04  # fallback defaults if data missing

    # Race/ethnicity. VERIFIED: this survey's racial-identity question only
    # ever has two responses, in every year and sector: "Non-racialized" and
    # "Racialized". There is no Black/Indigenous breakdown available in this
    # data source at all. pct_black and pct_indigenous are therefore fixed at
    # 0.0 (not extracted, since there is nothing to extract), and the entire
    # "Racialized" share is folded into pct_other_race undifferentiated. If
    # a Black/Indigenous breakdown is needed, this file cannot provide it.
    pct_non_racialized = get_share(
        df, sector=sector, year=year,
        question_contains="do you identify with any racial identities", response_value="Non-racialized",
    )
    pct_black = 0.0
    pct_indigenous = 0.0
    pct_other_race = max(0.0, 1.0 - pct_non_racialized)

    row = {
        "total_surveyed": int(total),
        "pct_male": pct_male,
        "pct_female": pct_female,
        "pct_trans_nonbinary": pct_trans_nonbinary,
        "pct_white": pct_non_racialized,
        "pct_indigenous": float(pct_indigenous),
        "pct_black": float(pct_black),
        "pct_other_race": float(pct_other_race),
        "pct_less_than_hs": float(pct_less_than_hs),
        "pct_hs_graduate": float(pct_hs_graduate),
        "pct_some_post_sec": float(pct_some_post_sec),
        "pct_post_sec_higher": float(pct_post_sec_higher),
        "pct_has_dependents": float(min(max(pct_has_dependents, 0.0), 1.0)),
        "pct_mental_health": get_share(df, sector=sector, year=year, question_contains="health challenges at this time", response_value="Mental health issue"),
        # 2018 used the label "Addiction" for this question; 2021/2024
        # renamed it "Substance use issue". Querying only the newer label
        # silently returned 0.0 for 2018, which in turn collapsed the MH-SU
        # joint constraint to ~0 and badly distorted 2018's fit. Sum both
        # labels — only the relevant one is ever populated in a given year.
        "pct_substance_use": get_share(df, sector=sector, year=year, question_contains="health challenges at this time", response_value="Substance use issue")
        + get_share(df, sector=sector, year=year, question_contains="health challenges at this time", response_value="Addiction"),
        # "Makeshift Shelter, Tent or Shack" (2018 terminology) was replaced
        # by "Encampment" in 2021/2024 — sum all three since only the
        # relevant labels for a given year are ever populated. The old
        # 2-label sum silently undercounted outdoor/unsheltered homelessness
        # in 2021 and 2024.
        "pct_outdoor_sleeping": get_share(df, sector=sector, year=year, question_contains="where are you staying tonight", response_value="Unsheltered")
        + get_share(df, sector=sector, year=year, question_contains="where are you staying tonight", response_value="Makeshift Shelter, Tent or Shack")
        + get_share(df, sector=sector, year=year, question_contains="where are you staying tonight", response_value="Encampment"),
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
    parsed_years = sorted(df["year"].dropna().astype(int).unique().tolist())

    # Restrict strictly to the three real PiT survey years (2018, 2021, 2024).
    # Report anything else found and dropped, for transparency.
    dropped_years = [y for y in parsed_years if y not in ALLOWED_YEARS]
    years = [y for y in parsed_years if y in ALLOWED_YEARS]
    if dropped_years:
        print(f"Dropping non-survey years found in Period column: {dropped_years}")
    missing_years = sorted(ALLOWED_YEARS - set(years))
    if missing_years:
        print(f"Warning: expected PiT years not found in data: {missing_years}")

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
        raise RuntimeError(f"No PIT rows found for sector '{args.sector}' in {DATA_PATH} for years {sorted(ALLOWED_YEARS)}")

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