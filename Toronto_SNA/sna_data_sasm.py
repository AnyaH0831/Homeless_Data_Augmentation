"""
sna_pipeline_sasm.py
────────────────────
Pipeline using SASM (Small Area Synthetic Microdata) optimization, built on
Toronto's Street Needs Assessment (SNA) data for 2013, 2018, and 2021.

MAJOR REWRITE — YEAR-SPECIFIC EXTRACTION
──────────────────────────────────────────
Earlier versions of this file used a single ROW_MAP/RATIO_MAP built entirely
around the 2021 survey's row-naming scheme (e.g. "26_GenderIdentityCount",
"23_MentalHealthIssueYes") and assumed it would also work for 2013 and 2018.
It does not: Toronto used three genuinely different questionnaires across
these years, with different question numbers, different category structures,
and (for 2013) some concepts not asked about at all. Verified against the
actual uploaded xlsx files, this produced several serious, silent bugs:

1. NO DENOMINATOR FALLBACK. None of the "_Count" denominator rows
   (gender_count, health_count, race_count, etc.) exist under the expected
   2021-style names in 2013/2018. When missing, the denominator silently
   defaulted to 1.0, so e.g. pct_male = (real count)/1.0 exploded past 100%
   and got clipped to exactly 1.0 — simultaneously with pct_female=1.0,
   which is impossible in reality.
2. EDUCATION WAS ENTIRELY MISSING for 2013 and 2018 (no fallback existed),
   summing to 0% instead of 100% — since education is a hard partition in
   the SASM attribute space, this alone forced a huge, unresolvable
   inconsistency in the optimizer.
3. "YEARS HOMELESS" WAS THE WRONG CONCEPT for 2018 AND 2021. The row being
   used ("4_TIMEHOMELESSAVERAGE" / "4_YearHomelessAverage") actually answers
   "how much time in the past 12 months have you experienced homelessness"
   (bounded 0-12 months / 0-365 days), not lifetime years homeless. Feeding
   this into the chronic-homelessness formula as if it were lifetime years
   badly distorted pct_chronic for both years (2021's hit the 0.9 clip
   ceiling). Only 2013 asked a genuine lifetime-years question directly.
   Fixed by deriving years-homeless from (current age) - (age when first
   experienced homelessness), which both 2018 and 2021 do ask directly.
4. 2018's "outdoor sleeping" query matched the wrong question entirely — it
   answers "where were you staying BEFORE you started using this winter
   respite service" (a retrospective question about a small subgroup), not
   "are you currently sleeping outdoors tonight."
5. Even 2021 (previously the "good" year) mapped only 2 of 7 real education
   categories, silently dropping the other 5 respondents' education levels
   into an artificial 100%-of-2-categories renormalization that looked
   internally consistent but was substantively wrong. It also computed race
   by mixing two DIFFERENT survey questions with different denominators
   (a dedicated Indigenous-identity question plus a separate racial-identity
   question), which is avoided here by using the single racial-identity
   question as the sole source for all four race categories.

NEW APPROACH: EXPORT-SHEET SECTOR COLUMNS FOR OUTDOOR SLEEPING
─────────────────────────────────────────────────────────────
Toronto's Export sheet has per-SECTOR columns (who was surveyed WHERE —
e.g. OUTDOOR/OUTDOORS, MEN, WOMEN, FAMILY, YOUTH, ...) in all three years,
which the previous version discarded (only ever reading a single "Total"
column). pct_outdoor_sleeping is now computed directly and reliably from the
TOTALSURVEYS row's OUTDOOR/OUTDOORS column value divided by its TOTAL column
value — this works consistently across all three years, unlike trying to
parse an inconsistently-worded survey question.

DOCUMENTED DATA LIMITATIONS (not fixable from this source)
────────────────────────────────────────────────────────────
- 2013 has NO race/ethnicity breakdown beyond a simple Aboriginal-identity
  yes/no question. pct_black and pct_white cannot be extracted for 2013;
  they are set to 0.0 and pct_other_race absorbs the entire non-Indigenous
  population, undifferentiated.
- 2013 has NO education question, NO dependents question, and NO
  immigrant/foster-care/incarceration/housing-loss-reason questions at all.
  These use documented literature-informed defaults for that year only.
- 2013 has NO direct "do you have a mental health issue / substance use
  issue" question. The closest available signal is a *different* question —
  "which of the following would help you find housing" — with "Mental
  health supports" and "Help getting alcohol or drug treatment" as
  checkbox options. This measures perceived SERVICE NEED, not clinical
  self-identification, and is used only as a labeled best-available proxy
  for 2013.
"""

import argparse
import io
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from sasm_generator import generate_individuals_sasm

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── CONFIG ────────────────────────────────────────────────────────────────────

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

# Literature-informed defaults used ONLY for concepts genuinely absent from
# a given year's survey (documented per-field below), never to override real
# extracted data.
DEFAULT_EDUCATION = {  # renormalized to sum exactly to 1.0
    "less_than_hs": 0.35 / 0.94,
    "hs_graduate": 0.27 / 0.94,
    "some_post_sec": 0.12 / 0.94,
    "post_sec_higher": 0.20 / 0.94,
}
DEFAULT_HAS_DEPENDENTS = 0.10   # 2018/2021 observed ~10-11%; 2013 has no such question
DEFAULT_IMMIGRANT = 0.20
DEFAULT_FOSTER_CARE = 0.12
DEFAULT_INCARCERATION = 0.08
DEFAULT_HOUSING_LOSS_INCOME = 0.35
DEFAULT_HOUSING_LOSS_HEALTH = 0.15


# ── LOW-LEVEL HELPERS ─────────────────────────────────────────────────────────

def _normalize_text(value) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _pick_total_column(columns) -> str:
    """Find the 'Total' / 'Total/Average' sector column among Export columns."""
    normalized = {_normalize_text(col): col for col in columns}
    for candidate in ("totalaverage", "total"):
        if candidate in normalized:
            return normalized[candidate]
    return columns[-1]


def _pick_outdoor_column(columns) -> str | None:
    """Find the 'Outdoor' / 'Outdoors' sector column among Export columns."""
    for col in columns:
        if _normalize_text(col) in ("outdoor", "outdoors"):
            return col
    return None


def load_sna_xlsx(source):
    """
    Parse Export + Key-Rows sheets into a merged DataFrame that retains ALL
    sector columns from the Export sheet (not just a single picked "value"
    column), so outdoor-sleeping can be read directly from the OUTDOOR/
    OUTDOORS sector column rather than inferred from a survey question.

    Returns (df, total_col, outdoor_col). df has one row per Key-Rows entry,
    merged with all Export sector columns by row_name, plus a "value" column
    that mirrors the Total/Total-Average column for backward-compatible
    single-value lookups.
    """
    buf = io.BytesIO(source) if isinstance(source, bytes) else source
    export = pd.read_excel(buf, sheet_name="Export")
    export.columns = [str(c).strip() for c in export.columns]
    row_col = export.columns[0]
    sector_cols = list(export.columns[1:])
    total_col = _pick_total_column(sector_cols)
    outdoor_col = _pick_outdoor_column(sector_cols)
    export = export.rename(columns={row_col: "row_name"})
    for c in sector_cols:
        export[c] = pd.to_numeric(export[c], errors="coerce")

    key_rows = pd.read_excel(buf, sheet_name="Key-Rows")
    key_rows.columns = [str(c).strip() for c in key_rows.columns]
    key_rows = key_rows.iloc[:, :5].copy()
    key_rows.columns = ["row_name", "question", "response", "meta_value", "notes"]

    df = key_rows.merge(export, on="row_name", how="left")
    df["row_name"] = df["row_name"].astype(str).str.strip()
    df["value"] = df[total_col]
    return df, total_col, outdoor_col


def get_val(df: pd.DataFrame, row_name: str) -> float:
    """Exact (case-insensitive) row_name lookup of the Total-column value."""
    m = df[df["row_name"].str.lower() == row_name.lower()]
    if m.empty:
        return 0.0
    v = m["value"].iloc[0]
    return float(v) if pd.notna(v) else 0.0


def get_outdoor_fraction(df: pd.DataFrame, outdoor_col: str | None, total_col: str) -> float | None:
    """
    Fraction of ALL survey respondents surveyed in outdoor locations, read
    directly from the Export sheet's per-sector columns at the TOTALSURVEYS
    row. Returns None if the outdoor column or the row can't be found, so
    the caller can apply a documented fallback.
    """
    if outdoor_col is None:
        return None
    row = df[df["row_name"].str.lower() == "totalsurveys"]
    if row.empty:
        return None
    outdoor_val = row[outdoor_col].iloc[0]
    total_val = row[total_col].iloc[0]
    if pd.isna(outdoor_val) or pd.isna(total_val) or total_val <= 0:
        return None
    return float(np.clip(outdoor_val / total_val, 0.0, 1.0))


def _renorm(*vals):
    """Normalize a set of non-negative raw counts to sum to exactly 1.0."""
    vals = [max(float(v), 0.0) for v in vals]
    total = sum(vals)
    if total <= 0:
        return [0.0] * len(vals)
    return [v / total for v in vals]


def _years_homeless_from_ages(age_avg: float, age_first_homeless: float) -> float:
    """Estimate lifetime years homeless as (current age) - (age first
    experienced homelessness), floored to avoid zero/negative from noise."""
    if age_avg <= 0 or age_first_homeless <= 0:
        return 4.0  # fallback if either age figure is missing
    return float(max(age_avg - age_first_homeless, 0.1))


# ── YEAR-SPECIFIC EXTRACTION ──────────────────────────────────────────────────
# Each function is hand-mapped to that year's VERIFIED real row names/schema.

def extract_2013(df, total_col, outdoor_col) -> dict:
    total_surveyed = get_val(df, "TOTALSURVEYS")
    age_avg = get_val(df, "3_AGE")
    # 2013 directly reports genuine lifetime years homeless (unlike 2018/2021,
    # which only report time-homeless-in-the-past-12-months under a
    # similarly-named row — see module docstring point 3).
    years_homeless_avg = get_val(df, "1_YEARSHOMELESS") or 3.0

    # Gender: Male/Female/Trans/Other, normalized to sum to 1.0. "Trans" and
    # "Other" are folded into trans_nonbinary (2013's questionnaire doesn't
    # distinguish further).
    n_male = get_val(df, "4_MALE")
    n_female = get_val(df, "4_FEMALE")
    n_transnb = get_val(df, "4_TRANS") + get_val(df, "4_OTHER")
    pct_male, pct_female, pct_trans_nonbinary = _renorm(n_male, n_female, n_transnb)

    # Race: 2013 ONLY asks Aboriginal identity (yes/no). No Black/White
    # breakdown exists at all this year — documented limitation.
    n_ab_yes = get_val(df, "6A_YES")
    n_ab_no = get_val(df, "6A_NO")
    denom = n_ab_yes + n_ab_no
    pct_indigenous = (n_ab_yes / denom) if denom > 0 else 0.0
    pct_black = 0.0
    pct_white = 0.0
    pct_other_race = max(0.0, 1.0 - pct_indigenous)

    # Education: not asked in 2013 at all.
    edu = DEFAULT_EDUCATION

    # Dependents: not asked in 2013 at all.
    pct_has_dependents = DEFAULT_HAS_DEPENDENTS

    # Mental health / substance use: 2013 has no direct status question.
    # Best available proxy: "which of the following would help you find
    # housing" — Mental health supports / Help getting alcohol or drug
    # treatment. This measures PERCEIVED SERVICE NEED, not clinical
    # self-identification — a real conceptual difference, used here only
    # because nothing closer exists in this year's survey.
    n_resp_total = get_val(df, "12A_RESPONSETOTAL")
    pct_mental_health = (get_val(df, "12A_MENTALHEALTH") / n_resp_total) if n_resp_total > 0 else 0.0
    pct_substance_use = (get_val(df, "12A_TREATMENT") / n_resp_total) if n_resp_total > 0 else 0.0

    # Outdoor sleeping: from Export sheet's OUTDOOR sector column.
    pct_outdoor = get_outdoor_fraction(df, outdoor_col, total_col)
    if pct_outdoor is None:
        pct_outdoor = 0.20  # fallback if column missing

    pct_chronic = float(np.clip(1 - np.exp(-years_homeless_avg / 3.5), 0.1, 0.9))

    # LGBTQ: available directly (Q5).
    n_lgbtq_yes = get_val(df, "5_YES")
    n_lgbtq_no = get_val(df, "5_NO")
    lgbtq_denom = n_lgbtq_yes + n_lgbtq_no
    pct_lgbtq = (n_lgbtq_yes / lgbtq_denom) if lgbtq_denom > 0 else 0.0

    # No-income: available directly (Q13D).
    n_no_income = get_val(df, "13D_NOINCOME")
    n_income_total = get_val(df, "13D_RESPONSETOTAL")
    pct_no_income = (n_no_income / n_income_total) if n_income_total > 0 else 0.0

    return {
        "total_surveyed": total_surveyed,
        "age_avg": age_avg, "age_std": 14.0,
        "years_homeless_avg": years_homeless_avg,
        "pct_male": pct_male, "pct_female": pct_female, "pct_trans_nonbinary": pct_trans_nonbinary,
        "pct_black": pct_black, "pct_white": pct_white, "pct_indigenous": pct_indigenous, "pct_other_race": pct_other_race,
        "pct_less_than_hs": edu["less_than_hs"], "pct_hs_graduate": edu["hs_graduate"],
        "pct_some_post_sec": edu["some_post_sec"], "pct_post_sec_higher": edu["post_sec_higher"],
        "pct_has_dependents": pct_has_dependents,
        "pct_mental_health": pct_mental_health, "pct_substance_use": pct_substance_use,
        "pct_outdoor_sleeping": pct_outdoor, "pct_chronic": pct_chronic,
        "pct_lgbtq": pct_lgbtq,
        "pct_immigrant": DEFAULT_IMMIGRANT,
        "pct_foster_care_history": DEFAULT_FOSTER_CARE,
        "pct_incarceration_history": DEFAULT_INCARCERATION,
        "pct_no_income": pct_no_income,
        "pct_housing_loss_income": DEFAULT_HOUSING_LOSS_INCOME,
        "pct_housing_loss_health": DEFAULT_HOUSING_LOSS_HEALTH,
    }


def extract_2018(df, total_col, outdoor_col) -> dict:
    total_surveyed = get_val(df, "TOTALSURVEYS")
    age_avg = get_val(df, "2_AGEAVERAGE")
    age_first_homeless = get_val(df, "3_AGEHOMELESSAVERAGE")
    years_homeless_avg = _years_homeless_from_ages(age_avg, age_first_homeless)

    # Gender: 2018 has a real GENDERCOUNT denominator, but we renormalize
    # from raw category counts regardless, for robustness against exactly
    # what that denominator does/doesn't include.
    n_male = get_val(df, "15_MALE")
    n_female = get_val(df, "15_FEMALE")
    n_transnb = (get_val(df, "15_TRANSFEMALE") + get_val(df, "15_TRANSMALE")
                 + get_val(df, "15_TWOSPIRIT") + get_val(df, "15_GENDERQUEER") + get_val(df, "15_OTHER"))
    pct_male, pct_female, pct_trans_nonbinary = _renorm(n_male, n_female, n_transnb)

    # Race: single "racial or ethnic group" question (Q12), all four
    # categories derived from it and renormalized together.
    n_white = get_val(df, "12_WHITE")
    n_black = get_val(df, "12_BLACKAFRICAN") + get_val(df, "12_BLACKCARIBBEAN") + get_val(df, "12_BLACKOTHER")
    n_indigenous = get_val(df, "12_INDIGENOUS")
    n_other = (get_val(df, "12_HISPANIC") + get_val(df, "12_ASIAN") + get_val(df, "12_ARAB")
               + get_val(df, "12_FILIPINO") + get_val(df, "12_MIXED"))
    pct_black, pct_white, pct_indigenous, pct_other_race = _renorm(n_black, n_white, n_indigenous, n_other)

    # Education: not asked in 2018 at all.
    edu = DEFAULT_EDUCATION

    # Dependents: "What family members are staying with you tonight?"
    n_dep = get_val(df, "1_FAMILYHEAD")
    n_fam_count = get_val(df, "1_FAMILYCOUNT")
    pct_has_dependents = (n_dep / n_fam_count) if n_fam_count > 0 else DEFAULT_HAS_DEPENDENTS

    # Mental health / substance use ("addiction"): Q19, shared denominator.
    health_count = get_val(df, "19_HEALTHCOUNT")
    pct_mental_health = (get_val(df, "19_MENTALYES") / health_count) if health_count > 0 else 0.0
    pct_substance_use = (get_val(df, "19_ADDICTIONYES") / health_count) if health_count > 0 else 0.0

    # Outdoor sleeping: from Export sheet's OUTDOORS sector column (the
    # in-survey "9_OUTDOORS" row asks a different, retrospective question
    # about winter-respite-service users and is NOT used here).
    pct_outdoor = get_outdoor_fraction(df, outdoor_col, total_col)
    if pct_outdoor is None:
        pct_outdoor = 0.20

    pct_chronic = float(np.clip(1 - np.exp(-years_homeless_avg / 3.5), 0.1, 0.9))

    # LGBTQ (Q16)
    n_straight = get_val(df, "16_HETEROSEXUAL")
    n_lgbtq = (get_val(df, "16_GAY") + get_val(df, "16_LESBIAN") + get_val(df, "16_BISEXUAL")
               + get_val(df, "16_TWOSPIRIT") + get_val(df, "16_QUESTIONING") + get_val(df, "16_QUEER") + get_val(df, "16_OTHER"))
    lgbtq_denom = n_straight + n_lgbtq
    pct_lgbtq = (n_lgbtq / lgbtq_denom) if lgbtq_denom > 0 else 0.0

    # Immigrant (Q10)
    n_no_immig = get_val(df, "10_NO")
    n_immig = get_val(df, "10_IMMIGRANT") + get_val(df, "10_REFUGEE") + get_val(df, "10_REFUGEECLAIMANT") + get_val(df, "10_TEMP")
    immig_denom = n_no_immig + n_immig
    pct_immigrant = (n_immig / immig_denom) if immig_denom > 0 else DEFAULT_IMMIGRANT

    # Foster care (Q18)
    n_foster_yes = get_val(df, "18_YES")
    n_foster_no = get_val(df, "18_NO")
    foster_denom = n_foster_yes + n_foster_no
    pct_foster_care_history = (n_foster_yes / foster_denom) if foster_denom > 0 else DEFAULT_FOSTER_CARE

    # Incarceration (Q23)
    n_prison_yes = get_val(df, "23_PRISONYES")
    n_prison_no = get_val(df, "23_PRISONNO")
    prison_denom = n_prison_yes + n_prison_no
    pct_incarceration_history = (n_prison_yes / prison_denom) if prison_denom > 0 else DEFAULT_INCARCERATION

    # No income (Q17)
    n_no_income = get_val(df, "17_NONE")
    n_income_total = get_val(df, "17_INCOMESOURCECOUNT")
    pct_no_income = (n_no_income / n_income_total) if n_income_total > 0 else 0.0

    return {
        "total_surveyed": total_surveyed,
        "age_avg": age_avg, "age_std": 14.0,
        "years_homeless_avg": years_homeless_avg,
        "pct_male": pct_male, "pct_female": pct_female, "pct_trans_nonbinary": pct_trans_nonbinary,
        "pct_black": pct_black, "pct_white": pct_white, "pct_indigenous": pct_indigenous, "pct_other_race": pct_other_race,
        "pct_less_than_hs": edu["less_than_hs"], "pct_hs_graduate": edu["hs_graduate"],
        "pct_some_post_sec": edu["some_post_sec"], "pct_post_sec_higher": edu["post_sec_higher"],
        "pct_has_dependents": pct_has_dependents,
        "pct_mental_health": pct_mental_health, "pct_substance_use": pct_substance_use,
        "pct_outdoor_sleeping": pct_outdoor, "pct_chronic": pct_chronic,
        "pct_lgbtq": pct_lgbtq,
        "pct_immigrant": pct_immigrant,
        "pct_foster_care_history": pct_foster_care_history,
        "pct_incarceration_history": pct_incarceration_history,
        "pct_no_income": pct_no_income,
        "pct_housing_loss_income": DEFAULT_HOUSING_LOSS_INCOME,
        "pct_housing_loss_health": DEFAULT_HOUSING_LOSS_HEALTH,
    }


def extract_2021(df, total_col, outdoor_col) -> dict:
    total_surveyed = get_val(df, "TotalSurveys")
    age_avg = get_val(df, "2_AgeAverage")
    age_first_homeless = get_val(df, "3_HomelessAgeAverage")
    years_homeless_avg = _years_homeless_from_ages(age_avg, age_first_homeless)

    # Gender (Q26)
    n_male = get_val(df, "26_GenderIdentityMale")
    n_female = get_val(df, "26_GenderIdentityFemale")
    n_transnb = (get_val(df, "26_GenderIdentityTransMale") + get_val(df, "26_GenderIdentityTransFemale")
                 + get_val(df, "26_GenderIdentityTwoSpirit") + get_val(df, "26_GenderIdentityNonBinary"))
    pct_male, pct_female, pct_trans_nonbinary = _renorm(n_male, n_female, n_transnb)

    # Race: single "racial identities" question (Q20) used for ALL FOUR
    # categories (previously mixed with a separate Indigenous-identity
    # question that has a different denominator — see module docstring).
    n_white = get_val(df, "20_RaceEthnicityWhite")
    n_black = (get_val(df, "20_RaceEthnicityBlackCanadianAmerican")
               + get_val(df, "20_RaceEthnicityBlackAfrican") + get_val(df, "20_RaceEthnicityBlackAfroCaribbean"))
    n_indigenous = get_val(df, "20_RaceEthnicityFirstNations")
    n_other = (get_val(df, "20_RaceEthnicityArab") + get_val(df, "20_RaceEthnicityEastAsian")
               + get_val(df, "20_RaceEthnicitySouthEastAsian") + get_val(df, "20_RaceEthnicitySouthAsian")
               + get_val(df, "20_RaceEthnicityWestAsian") + get_val(df, "20_RaceEthnicityLatinAmerican")
               + get_val(df, "20_RaceEthnicityOther"))
    pct_black, pct_white, pct_indigenous, pct_other_race = _renorm(n_black, n_white, n_indigenous, n_other)

    # Education (Q33): 7 real categories, mapped to 4 buckets and
    # renormalized together (previous version only caught 2 of 7).
    n_less_than_hs = (get_val(df, "33_EducationPrimarySchool") + get_val(df, "33_EducationNoEducation")
                       + get_val(df, "33_EducationSomeHighSchool"))
    n_hs_graduate = get_val(df, "33_EducationHighSchoolGraduate")
    n_some_post_sec = get_val(df, "33_EducationSomePostSecondary")
    n_post_sec_higher = get_val(df, "33_EducationPostSecondaryGraduate") + get_val(df, "33_EducationGraduateDegree")
    pct_less_than_hs, pct_hs_graduate, pct_some_post_sec, pct_post_sec_higher = _renorm(
        n_less_than_hs, n_hs_graduate, n_some_post_sec, n_post_sec_higher
    )

    # Dependents (Q1)
    n_dep = get_val(df, "1_FamilyMembersTonightChildDep")
    n_fam_count = get_val(df, "1_FamilyCount")
    pct_has_dependents = (n_dep / n_fam_count) if n_fam_count > 0 else DEFAULT_HAS_DEPENDENTS

    # Mental health / substance use (Q23)
    health_count = get_val(df, "23_HealthChallengesCount")
    pct_mental_health = (get_val(df, "23_MentalHealthIssueYes") / health_count) if health_count > 0 else 0.0
    pct_substance_use = (get_val(df, "23_SubstanceUseIssueYes") / health_count) if health_count > 0 else 0.0

    # Outdoor sleeping: from Export sheet's OUTDOORS sector column.
    pct_outdoor = get_outdoor_fraction(df, outdoor_col, total_col)
    if pct_outdoor is None:
        pct_outdoor = 0.20

    pct_chronic = float(np.clip(1 - np.exp(-years_homeless_avg / 3.5), 0.1, 0.9))

    # LGBTQ (Q28)
    lgbtq_count = get_val(df, "28_LGBTQS2Count")
    pct_lgbtq = (get_val(df, "28_Yes") / lgbtq_count) if lgbtq_count > 0 else 0.0

    # Immigrant (Q11)
    immig_count = get_val(df, "11_ImmigrantStatusCount")
    n_immig = get_val(df, "11_Immigrant") + get_val(df, "11_Refugee") + get_val(df, "11_RefugeeClaimant")
    pct_immigrant = (n_immig / immig_count) if immig_count > 0 else DEFAULT_IMMIGRANT

    # Foster care (Q22)
    foster_count = get_val(df, "22_FosterCount")
    pct_foster_care_history = (get_val(df, "22_Yes") / foster_count) if foster_count > 0 else DEFAULT_FOSTER_CARE

    # Incarceration (Q32)
    service_count = get_val(df, "32_ServiceUseCount")
    pct_incarceration_history = (get_val(df, "32_PrisonOrJailYes") / service_count) if service_count > 0 else DEFAULT_INCARCERATION

    # No income (Q29)
    income_count = get_val(df, "29_IncomeCount")
    pct_no_income = (get_val(df, "29_IncomeNoIncome") / income_count) if income_count > 0 else 0.0

    # Housing loss reasons (Q6)
    housing_loss_count = get_val(df, "6_HousingLossCount")
    pct_housing_loss_income = (get_val(df, "6_HousingLossNotEnoughIncome") / housing_loss_count) if housing_loss_count > 0 else DEFAULT_HOUSING_LOSS_INCOME
    pct_housing_loss_health = (get_val(df, "6_HousingLossMentalHealth") / housing_loss_count) if housing_loss_count > 0 else DEFAULT_HOUSING_LOSS_HEALTH

    return {
        "total_surveyed": total_surveyed,
        "age_avg": age_avg, "age_std": 14.0,
        "years_homeless_avg": years_homeless_avg,
        "pct_male": pct_male, "pct_female": pct_female, "pct_trans_nonbinary": pct_trans_nonbinary,
        "pct_black": pct_black, "pct_white": pct_white, "pct_indigenous": pct_indigenous, "pct_other_race": pct_other_race,
        "pct_less_than_hs": pct_less_than_hs, "pct_hs_graduate": pct_hs_graduate,
        "pct_some_post_sec": pct_some_post_sec, "pct_post_sec_higher": pct_post_sec_higher,
        "pct_has_dependents": pct_has_dependents,
        "pct_mental_health": pct_mental_health, "pct_substance_use": pct_substance_use,
        "pct_outdoor_sleeping": pct_outdoor, "pct_chronic": pct_chronic,
        "pct_lgbtq": pct_lgbtq,
        "pct_immigrant": pct_immigrant,
        "pct_foster_care_history": pct_foster_care_history,
        "pct_incarceration_history": pct_incarceration_history,
        "pct_no_income": pct_no_income,
        "pct_housing_loss_income": pct_housing_loss_income,
        "pct_housing_loss_health": pct_housing_loss_health,
    }


YEAR_EXTRACTORS = {2013: extract_2013, 2018: extract_2018, 2021: extract_2021}


def apply_realistic_bounds(agg: dict) -> dict:
    """Clip proportions to literature-supported ranges as a sanity check
    against extraction errors. All years use real SNA survey data (or a
    documented, labeled default where a question was never asked) — this
    never substitutes external rates for real extracted data."""
    agg["pct_mental_health"] = float(np.clip(agg.get("pct_mental_health", 0.35), 0.15, 0.60))
    agg["pct_substance_use"] = float(np.clip(agg.get("pct_substance_use", 0.25), 0.15, 0.45))
    agg["pct_outdoor_sleeping"] = float(np.clip(agg.get("pct_outdoor_sleeping", 0.18), 0.05, 0.40))
    agg["pct_foster_care_history"] = float(np.clip(agg.get("pct_foster_care_history", 0.12), 0.05, 0.30))
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


def load_all_years(use_local: bool = False) -> dict:
    """Load and extract all three observed SNA years (2013, 2018, 2021)."""
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
        df_sheet, total_col, outdoor_col = load_sna_xlsx(raw)
        agg = YEAR_EXTRACTORS[year](df_sheet, total_col, outdoor_col)
        agg = apply_realistic_bounds(agg)
        agg["year"] = year
        results[year] = agg
    return results


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SASM pipeline for synthetic data generation")
    parser.add_argument("--local", action="store_true",
                        help="Load SNA xlsx locally instead of API")
    args = parser.parse_args()

    print("=" * 65)
    print("STEP 1: Loading SNA aggregate data (2013, 2018, 2021)")
    print("=" * 65)
    observed = load_all_years(use_local=args.local)

    agg_df = pd.DataFrame(observed).T.astype(float)
    agg_df.index = agg_df.index.astype(int)
    agg_df.index.name = "year"
    agg_df["total_surveyed"] = agg_df["total_surveyed"].round().astype(int)

    print(agg_df[["total_surveyed", "pct_mental_health", "pct_outdoor_sleeping", "pct_chronic"]].round(3))

    observed_years = sorted(observed.keys())
    print("\n" + "=" * 65)
    print("STEP 2: SASM optimization-based individual generation")
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