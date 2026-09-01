"""
sasm_generator.py  (v3 — correlation fix)
──────────────────────────────────────────
SASM optimization-based synthetic microdata generator.

Based on: Lin & Xiao (2023). "Generating Small Areal Synthetic Microdata
from Public Aggregated Data Using an Optimization Method."
The Professional Geographer, 75(6), 905-915.

CHANGELOG FROM v2
──────────────────
v1 Bug 1 fixed: Y built from proportions × target_total (consistent scale)
v1 Bug 2 fixed: Extra binary features use flat marginals, no per-individual boost
v1 Bug 3 fixed: indigenous_flag derived from race assignment, no double-sample

v3 NEW FIX — MH/SU joint constraint:
  The optimizer only sees marginal constraints for mental_health and
  substance_use separately. Since the W matrix has no row for their
  joint distribution, the solver distributes them near-independently
  (correlation ≈ 0). In reality they co-occur at ~0.55 correlation.

  Fix: add a synthetic joint constraint row to W and Y:
    "n_mh_and_su" — people with BOTH mental_health=1 AND substance_use=1.
  The target count is estimated from the two marginals plus the known
  correlation using the tetrachoric approximation:
    P(A=1, B=1) ≈ P(A) * P(B) + r * sqrt(P(A)*P(A_0)*P(B)*P(B_0))
  where r is the target correlation (0.45, midpoint of literature range).
  This adds one row to W and one value to Y, which biases the optimizer
  toward the correct joint distribution at negligible computational cost.
"""

import itertools
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from scipy.linalg import cholesky
from scipy.stats import norm as sp_norm

warnings.filterwarnings("ignore")
np.random.seed(42)

# Target correlation between mental_health and substance_use
# Literature range: 0.35–0.55; we use 0.45 as midpoint
MH_SU_TARGET_CORR = 0.45


# ── STEP 1: ATTRIBUTE SPACE ───────────────────────────────────────────────────

OPTIMIZED_ATTRS = [
    ("gender",         ["male", "female", "trans_nonbinary"]),
    ("race",           ["black", "white", "indigenous", "other"]),
    ("education",      ["less_than_hs", "hs_graduate", "some_post_sec", "post_sec_higher"]),
    ("has_dependents", [1, 0]),
    ("mental_health",  [1, 0]),
    ("substance_use",  [1, 0]),
    ("outdoor_sleep",  [1, 0]),
    ("chronic",        [1, 0]),
]

ALL_COMBINATIONS = [
    dict(zip([a[0] for a in OPTIMIZED_ATTRS], vals))
    for vals in itertools.product(*[a[1] for a in OPTIMIZED_ATTRS])
]
M = len(ALL_COMBINATIONS)  # 192


# ── STEP 2: W MATRIX WITH JOINT CONSTRAINT ───────────────────────────────────

def build_W_and_constraint_names():
    """
    Build the aggregation matrix W.
    Includes a joint MH+SU constraint row to enforce co-occurrence correlation.
    """
    constraints = [
        ("n_male",             lambda c: c["gender"] == "male"),
        ("n_female",           lambda c: c["gender"] == "female"),
        ("n_trans_nonbinary",  lambda c: c["gender"] == "trans_nonbinary"),
        ("n_black",            lambda c: c["race"] == "black"),
        ("n_white",            lambda c: c["race"] == "white"),
        ("n_indigenous",       lambda c: c["race"] == "indigenous"),
        ("n_other_race",       lambda c: c["race"] == "other"),
        ("n_less_than_hs",     lambda c: c["education"] == "less_than_hs"),
        ("n_hs_graduate",      lambda c: c["education"] == "hs_graduate"),
        ("n_some_post_sec",    lambda c: c["education"] == "some_post_sec"),
        ("n_post_sec_higher",  lambda c: c["education"] == "post_sec_higher"),
        ("n_has_dependents",   lambda c: c["has_dependents"] == 1),
        ("n_mental_health",    lambda c: c["mental_health"] == 1),
        ("n_substance_use",    lambda c: c["substance_use"] == 1),
        # Joint constraint: both MH=1 AND SU=1
        # This row tells the optimizer how many people should have BOTH
        ("n_mh_and_su",        lambda c: c["mental_health"] == 1 and c["substance_use"] == 1),
        ("n_outdoor",          lambda c: c["outdoor_sleep"] == 1),
        ("n_chronic",          lambda c: c["chronic"] == 1),
        ("total_surveyed",     lambda c: True),
    ]
    W = np.zeros((len(constraints), M), dtype=float)
    for i, (_, fn) in enumerate(constraints):
        for k, combo in enumerate(ALL_COMBINATIONS):
            W[i, k] = 1.0 if fn(combo) else 0.0
    return W, [c[0] for c in constraints]

W_MATRIX, CONSTRAINT_NAMES = build_W_and_constraint_names()
Q = len(CONSTRAINT_NAMES)


# ── STEP 3: Y VECTOR ─────────────────────────────────────────────────────────

def estimate_joint_mh_su(p_mh: float, p_su: float, r: float = MH_SU_TARGET_CORR) -> float:
    """
    Estimate P(MH=1 AND SU=1) from marginals and target correlation.

    Uses the tetrachoric approximation for binary variables:
        P(A=1, B=1) = P(A)*P(B) + r * sqrt(P(A)*(1-P(A)) * P(B)*(1-P(B)))

    This is not exact but is a well-established approximation that produces
    the right direction of co-occurrence for our purposes.
    """
    return float(np.clip(
        p_mh * p_su + r * np.sqrt(p_mh * (1 - p_mh) * p_su * (1 - p_su)),
        0.0, min(p_mh, p_su)  # joint can't exceed either marginal
    ))


def extract_Y(row: pd.Series, target_total: int) -> np.ndarray:
    """
    Build target aggregate count vector Y at target_total scale.
    All values from proportions × target_total for consistent scaling.
    Includes estimated joint MH+SU count using tetrachoric approximation.
    """
    n    = float(target_total)
    p_mh = float(row.get("pct_mental_health",    0.40))
    p_su = float(row.get("pct_substance_use",     0.28))

    y_dict = {
        "n_male":             n * float(row.get("pct_male",              0.65)),
        "n_female":           n * float(row.get("pct_female",            0.28)),
        "n_trans_nonbinary":  n * float(row.get("pct_trans_nonbinary",   0.04)),
        "n_black":            n * float(row.get("pct_black",             0.30)),
        "n_white":            n * float(row.get("pct_white",             0.30)),
        "n_indigenous":       n * float(row.get("pct_indigenous",        0.18)),
        "n_other_race":       n * float(row.get("pct_other_race",        0.22)),
        "n_less_than_hs":     n * float(row.get("pct_less_than_hs",     0.35)),
        "n_hs_graduate":      n * float(row.get("pct_hs_graduate",       0.27)),
        "n_some_post_sec":    n * float(row.get("pct_some_post_sec",    0.12)),
        "n_post_sec_higher":  n * float(row.get("pct_post_sec_higher",  0.20)),
        "n_has_dependents":   n * float(row.get("pct_has_dependents",   0.11)),
        "n_mental_health":    n * p_mh,
        "n_substance_use":    n * p_su,
        # Joint constraint: estimated from marginals + literature correlation
        "n_mh_and_su":        n * estimate_joint_mh_su(p_mh, p_su),
        "n_outdoor":          n * float(row.get("pct_outdoor_sleeping",  0.20)),
        "n_chronic":          n * float(row.get("pct_chronic",           0.32)),
        "total_surveyed":     n,
    }
    return np.array([y_dict[name] for name in CONSTRAINT_NAMES], dtype=float)


# ── STEP 4: SOLVE OPTIMIZATION ───────────────────────────────────────────────

def solve_optimization(y: np.ndarray, total: int) -> np.ndarray:
    """Solve minimize ||Wx - y||² s.t. x >= 0, then round to integers."""
    result = lsq_linear(
        W_MATRIX, y,
        bounds=(0, np.inf),
        method="bvls",
        tol=1e-10,
        max_iter=50000,
    )
    x_cont = result.x

    # Largest-remainder integer rounding to preserve total exactly
    x_floor   = np.floor(x_cont).astype(int)
    remainder = total - x_floor.sum()
    if remainder > 0:
        top_k = np.argsort(x_cont - x_floor)[::-1][:int(remainder)]
        x_floor[top_k] += 1
    elif remainder < 0:
        nz = np.where(x_floor > 0)[0]
        rm = np.argsort(x_cont[nz])[:abs(int(remainder))]
        x_floor[nz[rm]] -= 1

    return np.clip(x_floor, 0, None)


# ── STEP 5: EXPAND TO INDIVIDUALS ────────────────────────────────────────────

def expand_combinations_to_individuals(x, year, row):
    """Expand integer combination counts into one-row-per-person DataFrame."""
    age_avg = float(row.get("age_avg",           40.0))
    age_std = float(row.get("age_std",            14.0))
    yh_avg  = float(row.get("years_homeless_avg",  4.0))

    records = []
    for k, combo in enumerate(ALL_COMBINATIONS):
        n_k = int(x[k])
        if n_k == 0:
            continue

        shift = 3.0 if combo["chronic"] == 1 else 0.0
        ages  = np.clip(np.random.normal(age_avg + shift, age_std, n_k), 15, 85)

        if combo["chronic"] == 1:
            yrs = np.clip(np.random.pareto(1.5, n_k) * yh_avg + 1.0, 1.0, 30.0)
        else:
            yrs = np.clip(np.random.exponential(max(yh_avg * 0.4, 0.3), n_k), 0.05, 8.0)

        shelters = (
            ["outdoor"] * n_k if combo["outdoor_sleep"] == 1
            else np.random.choice(
                ["emergency_shelter", "respite", "other"],
                size=n_k, p=[0.82, 0.11, 0.07]
            ).tolist()
        )

        for i in range(n_k):
            records.append({
                "year":             year,
                "age":              round(float(ages[i]), 1),
                "years_homeless":   round(float(yrs[i]),  2),
                "gender":           combo["gender"],
                "race":             combo["race"],
                "education":        combo["education"],
                "has_dependents":   combo["has_dependents"],
                "mental_health":    combo["mental_health"],
                "substance_use":    combo["substance_use"],
                "outdoor_sleeping": combo["outdoor_sleep"],
                "chronic_homeless": combo["chronic"],
                "shelter_type":     shelters[i],
                "youth":            int(ages[i] < 25),
                "indigenous_flag":  int(combo["race"] == "indigenous"),
            })

    return pd.DataFrame(records)


# ── STEP 6: ADD UNCONSTRAINED BINARY FEATURES ─────────────────────────────────

EXTRA_BINARY_FEATURES = [
    "lgbtq", "immigrant", "foster_care_history",
    "incarceration_history", "no_income",
    "housing_loss_income", "housing_loss_health",
]

EXTRA_CORR = np.array([
#  lgb   imm   fos   inc   noi   hli   hlh
  [1.00, 0.08, 0.30, 0.10, 0.12, 0.05, 0.08],
  [0.08, 1.00, 0.05, 0.04, 0.10, 0.15, 0.05],
  [0.30, 0.05, 1.00, 0.30, 0.20, 0.08, 0.12],
  [0.10, 0.04, 0.30, 1.00, 0.22, 0.10, 0.12],
  [0.12, 0.10, 0.20, 0.22, 1.00, 0.15, 0.10],
  [0.05, 0.15, 0.08, 0.10, 0.15, 1.00, 0.15],
  [0.08, 0.05, 0.12, 0.12, 0.10, 0.15, 1.00],
])
EXTRA_CHOL = cholesky(EXTRA_CORR, lower=True)


def add_extra_binary_features(df: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    """Append unconstrained binary features via Gaussian copula, flat marginals."""
    n = len(df)
    probs = {
        "lgbtq":                 float(row.get("pct_lgbtq",                0.12)),
        "immigrant":             float(row.get("pct_immigrant",             0.22)),
        "foster_care_history":   float(row.get("pct_foster_care_history",   0.16)),
        "incarceration_history": float(row.get("pct_incarceration_history", 0.21)),
        "no_income":             float(row.get("pct_no_income",             0.15)),
        "housing_loss_income":   float(row.get("pct_housing_loss_income",   0.40)),
        "housing_loss_health":   float(row.get("pct_housing_loss_health",   0.17)),
    }
    z_corr = np.random.randn(n, len(EXTRA_BINARY_FEATURES)) @ EXTRA_CHOL.T
    u      = sp_norm.cdf(z_corr)
    for i, feat in enumerate(EXTRA_BINARY_FEATURES):
        df[feat] = (u[:, i] < probs[feat]).astype(int)
    
    # Add income_type as categorical unconstrained feature
    # Consolidated from survey: primary income source distribution
    # If not extracted from survey, use reasonable defaults
    try:
        disability_val = float(row.get("pct_disability_income", 0.30))
        employment_val = float(row.get("pct_employment_income", 0.08))
        welfare_val = float(row.get("pct_welfare_income", 0.40))
        informal_val = float(row.get("pct_informal_income", 0.05))
        other_val = float(row.get("pct_other_income", 0.17))
        
        income_dist = np.array([disability_val, employment_val, welfare_val, informal_val, other_val], dtype=float)
        # Ensure no NaN values
        if np.any(~np.isfinite(income_dist)):
            income_dist = np.array([0.3, 0.08, 0.4, 0.05, 0.17], dtype=float)
        income_dist = income_dist / np.sum(income_dist)  # Normalize to sum to 1
        
        income_sources = np.random.choice(
            ["disability", "employment", "welfare", "informal", "other"],
            size=n,
            p=income_dist
        )
    except Exception:
        # Fallback to equal distribution
        income_sources = np.random.choice(
            ["disability", "employment", "welfare", "informal", "other"],
            size=n
        )
    
    df["income_type"] = income_sources
    
    return df


# ── STEP 7: QUALITY METRIC ────────────────────────────────────────────────────

def compute_sasm_quality(df_year: pd.DataFrame, y: np.ndarray) -> dict:
    """Compute d_p = ||WX' - Y||² and related metrics."""
    x_prime = np.zeros(M, dtype=float)
    for k, combo in enumerate(ALL_COMBINATIONS):
        mask = (
            (df_year["gender"]           == combo["gender"])        &
            (df_year["race"]             == combo["race"])          &
            (df_year["education"]        == combo["education"])     &
            (df_year["has_dependents"]   == combo["has_dependents"]) &
            (df_year["mental_health"]    == combo["mental_health"]) &
            (df_year["substance_use"]    == combo["substance_use"]) &
            (df_year["outdoor_sleeping"] == combo["outdoor_sleep"]) &
            (df_year["chronic_homeless"] == combo["chronic"])
        )
        x_prime[k] = mask.sum()

    wx_prime = W_MATRIX @ x_prime
    diff     = wx_prime - y
    d_p      = float(np.sum(diff ** 2))
    rmse     = float(np.sqrt(np.mean(diff ** 2)))
    max_e    = float(np.max(np.abs(diff)))
    valid    = y > 1.0
    pct_e    = float(np.mean(np.abs(diff[valid]) / y[valid]) * 100) if valid.any() else 0.0

    return {"d_p": d_p, "rmse": rmse, "max_error": max_e, "mean_pct_error": pct_e}


# ── MAIN ENTRY POINT ──────────────────────────────────────────────────────────

def generate_individuals_sasm(
    agg_df: pd.DataFrame,
    sample_size: int = 2000,
    use_observed_totals: bool = False,
    years: list | None = None,
) -> pd.DataFrame:
    """
    Generate synthetic individuals for all years in agg_df via SASM optimization.
    If `years` is provided, only those years are processed.

    Args:
        agg_df: DataFrame with aggregate statistics (one row per year)
        sample_size: Fixed sample size per year (default: 2000). Ignored if use_observed_totals=True.
        use_observed_totals: If True, use agg_df['total_surveyed'] as per-year sample sizes.
                            If False (default), use fixed sample_size for all years.
        years: Optional list/array of target years to process. If omitted, all years are used.
    """
    all_records = []
    quality_log = []

    if years is None:
        target_years = sorted(pd.Index(agg_df.index).unique())
    else:
        target_years = sorted(pd.Index(list(years)).unique())

    print(f"\n{'Year':>6}  {'Total':>8}  {'d_p':>8}  {'RMSE':>7}  {'MaxErr':>7}  {'MeanPct%':>9}  {'MH-SU corr':>11}")
    print("─" * 70)

    for year in target_years:
        if year not in agg_df.index:
            continue
        row = agg_df.loc[year]
        if use_observed_totals:
            total = int(row.get('total_surveyed', sample_size))
        else:
            total = sample_size

        y       = extract_Y(row, total)
        x       = solve_optimization(y, total)
        df_year = expand_combinations_to_individuals(x, year, row)
        df_year = add_extra_binary_features(df_year, row)

        quality = compute_sasm_quality(df_year, y)
        quality_log.append({"year": year, **quality})

        mh_su_corr = df_year[["mental_health", "substance_use"]].corr().iloc[0, 1]

        print(
            f"{year:>6}  {total:>8}  {quality['d_p']:>8.2f}  "
            f"{quality['rmse']:>7.3f}  {quality['max_error']:>7.1f}  "
            f"{quality['mean_pct_error']:>9.2f}%  {mh_su_corr:>11.3f}"
        )
        all_records.append(df_year)

    print("─" * 70)
    print("  RMSE < 2.0, MeanPct% < 2%, MH-SU corr 0.35-0.55 = excellent")

    if not all_records:
        return pd.DataFrame()

    df_all = pd.concat(all_records, ignore_index=True)
    pd.DataFrame(quality_log).to_csv("sasm_quality_log.csv", index=False)
    print("Saved sasm_quality_log.csv")
    return df_all
