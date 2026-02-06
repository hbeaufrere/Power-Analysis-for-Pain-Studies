"""
Power analysis engine for repeated-measures analgesiometric studies
analysed with linear mixed models.

Provides both analytical (closed-form) and simulation-based power
calculations for crossover and parallel-group designs.

Analytical approach
-------------------
For balanced designs the treatment-effect test in a linear mixed model
reduces to a paired (crossover) or two-sample (parallel) t-test on
subject-level means.  Degrees of freedom follow the Satterthwaite /
Kenward-Roger logic for simple random-intercept models.

Simulation approach
-------------------
Full data are generated under H1, a mixed model is fit with
``statsmodels.MixedLM``, and the treatment p-value is recorded.
Power = proportion of significant results across replications.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional


# ── effect-profile helpers ─────────────────────────────────────────────────

def generate_effect_profile(
    timepoints: list[float],
    peak_effect: float,
    time_to_peak_h: float,
    duration_h: float,
) -> np.ndarray:
    """Generate a pharmacodynamic effect profile over *timepoints*.

    Uses a modified Bateman-function shape:
        f(t) = peak × (t / t_peak)^a × exp(a × (1 - t / t_peak))
    where *a* is chosen so that f(duration_h) ≈ 0.05 × peak.
    """
    t = np.asarray(timepoints, dtype=float)
    if time_to_peak_h <= 0:
        time_to_peak_h = 0.01
    # shape parameter so that effect is ~5 % of peak at *duration_h*
    if duration_h <= time_to_peak_h:
        a = 2.0
    else:
        # solve a*(1 - dur/tpeak) + a*ln(dur/tpeak) = ln(0.05)
        ratio = duration_h / time_to_peak_h
        target = np.log(0.05)
        # a * (1 - ratio + ln(ratio)) = target  →  a = target / (1 - ratio + ln(ratio))
        denom = 1.0 - ratio + np.log(ratio)
        if denom >= 0:
            a = 2.0
        else:
            a = max(target / denom, 0.5)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(t > 0, t / time_to_peak_h, 0.0)
        profile = peak_effect * np.where(
            t > 0,
            np.power(ratio, a) * np.exp(a * (1.0 - ratio)),
            0.0,
        )
    # zero out anything beyond 1.5× duration (negligible tail)
    profile[t > 1.5 * duration_h] = 0.0
    return profile


# ── analytical power functions ─────────────────────────────────────────────

def _nct_power(effect: float, se: float, df: float, alpha: float) -> float:
    """Power from a two-sided t-test via the non-central t distribution."""
    if se <= 0 or df < 1:
        return np.nan
    ncp = effect / se
    # For very large non-centrality parameters the nct CDF becomes
    # numerically unstable — power is effectively 1.0.
    if abs(ncp) > 37:
        return 1.0
    crit = stats.t.ppf(1.0 - alpha / 2.0, df)
    # P(|T| > crit) where T ~ nct(df, ncp)
    upper_cdf = stats.nct.cdf(crit, df, ncp)
    lower_cdf = stats.nct.cdf(-crit, df, ncp)
    # scipy's nct can return NaN for the far tail when ncp is moderate-
    # to-large; in that case the tail contribution is negligible.
    if np.isnan(lower_cdf):
        lower_cdf = 0.0
    if np.isnan(upper_cdf):
        upper_cdf = 0.0
    power = 1.0 - upper_cdf + lower_cdf
    return float(np.clip(power, 0, 1))


def analytical_power_crossover(
    n_subjects: int,
    n_timepoints: int,
    mean_effect: float,
    within_subject_sd: float,
    alpha: float = 0.05,
    test: str = "overall",
    effect_profile: Optional[np.ndarray] = None,
) -> float:
    """Analytical power for a **crossover** repeated-measures design.

    Parameters
    ----------
    n_subjects : int
        Number of subjects (each receives every treatment).
    n_timepoints : int
        Number of post-baseline measurement occasions.
    mean_effect : float
        Expected mean treatment effect (averaged across time-points for
        the *overall* test, or the peak value for the *peak* test).
    within_subject_sd : float
        Residual SD (σ_ε) from the linear mixed model.
    alpha : float
        Significance level.
    test : str
        ``"overall"`` – test of the overall (average) treatment effect.
        ``"peak"``    – test at the single time-point of maximum effect.
        ``"interaction"`` – treatment × time interaction F-test.

    Returns
    -------
    float   Power ∈ [0, 1].
    """
    n = n_subjects
    J = max(n_timepoints, 1)
    sigma = within_subject_sd

    if test == "overall":
        # Paired t-test on subject means averaged over J time-points
        se = np.sqrt(2.0 * sigma ** 2 / (n * J))
        df = n - 1
        return _nct_power(mean_effect, se, df, alpha)

    elif test == "peak":
        # Paired t-test at one time-point
        se = np.sqrt(2.0 * sigma ** 2 / n)
        df = n - 1
        return _nct_power(mean_effect, se, df, alpha)

    elif test == "interaction":
        if effect_profile is None:
            return np.nan
        # F-test for treatment × time interaction
        delta = np.asarray(effect_profile)
        delta_bar = delta.mean()
        ssq = np.sum((delta - delta_bar) ** 2)
        lam = n * ssq / (2.0 * sigma ** 2)  # non-centrality parameter
        df1 = J - 1
        df2 = (n - 1) * (J - 1)
        if df1 < 1 or df2 < 1:
            return np.nan
        if lam > 500:
            return 1.0
        crit = stats.f.ppf(1.0 - alpha, df1, df2)
        power = 1.0 - stats.ncf.cdf(crit, df1, df2, lam)
        return float(np.clip(power, 0, 1))

    return np.nan


def analytical_power_parallel(
    n_per_group: int,
    n_timepoints: int,
    mean_effect: float,
    between_subject_sd: float,
    within_subject_sd: float,
    alpha: float = 0.05,
    test: str = "overall",
    effect_profile: Optional[np.ndarray] = None,
) -> float:
    """Analytical power for a **parallel-group** repeated-measures design.

    Parameters
    ----------
    n_per_group : int
        Subjects per treatment arm.
    n_timepoints : int
        Number of post-baseline measurement occasions.
    mean_effect : float
        Expected treatment-vs-control difference.
    between_subject_sd : float
        Between-subject SD (σ_b).
    within_subject_sd : float
        Residual SD (σ_ε).
    alpha, test, effect_profile
        See :func:`analytical_power_crossover`.
    """
    n = n_per_group
    J = max(n_timepoints, 1)
    sigma_b = between_subject_sd
    sigma_e = within_subject_sd

    if test == "overall":
        # Two-sample t-test on subject-level means
        var_subj_mean = sigma_b ** 2 + sigma_e ** 2 / J
        se = np.sqrt(2.0 * var_subj_mean / n)
        df = 2 * (n - 1)
        return _nct_power(mean_effect, se, df, alpha)

    elif test == "peak":
        # Comparison at a single time-point
        var_single = sigma_b ** 2 + sigma_e ** 2
        se = np.sqrt(2.0 * var_single / n)
        df = 2 * (n - 1)
        return _nct_power(mean_effect, se, df, alpha)

    elif test == "interaction":
        if effect_profile is None:
            return np.nan
        delta = np.asarray(effect_profile)
        delta_bar = delta.mean()
        ssq = np.sum((delta - delta_bar) ** 2)
        lam = n * ssq / (2.0 * sigma_e ** 2)
        df1 = J - 1
        df2 = 2 * (n - 1) * (J - 1)
        if df1 < 1 or df2 < 1:
            return np.nan
        if lam > 500:
            return 1.0
        crit = stats.f.ppf(1.0 - alpha, df1, df2)
        power = 1.0 - stats.ncf.cdf(crit, df1, df2, lam)
        return float(np.clip(power, 0, 1))

    return np.nan


# ── sample-size search ─────────────────────────────────────────────────────

def find_sample_size(
    target_power: float,
    design: str,
    n_timepoints: int,
    mean_effect: float,
    within_subject_sd: float,
    between_subject_sd: float = 0.0,
    alpha: float = 0.05,
    test: str = "overall",
    effect_profile: Optional[np.ndarray] = None,
    min_n: int = 3,
    max_n: int = 200,
) -> dict:
    """Binary search for the minimum sample size achieving *target_power*.

    Returns a dict with ``n``, ``power``, and ``power_curve`` (list of
    (n, power) tuples for plotting).
    """
    power_fn = (
        analytical_power_crossover if design == "Crossover"
        else analytical_power_parallel
    )

    def _power(n):
        kw = dict(
            n_timepoints=n_timepoints,
            mean_effect=mean_effect,
            within_subject_sd=within_subject_sd,
            alpha=alpha,
            test=test,
            effect_profile=effect_profile,
        )
        if design == "Crossover":
            kw["n_subjects"] = n
        else:
            kw["n_per_group"] = n
            kw["between_subject_sd"] = between_subject_sd
        return power_fn(**kw)

    # build full power curve for plotting
    ns = sorted(set(
        list(range(min_n, min(max_n + 1, 51)))
        + list(range(50, min(max_n + 1, 101), 5))
        + list(range(100, max_n + 1, 10))
    ))
    curve = [(n, _power(n)) for n in ns]

    # find minimum n achieving target power
    result_n = None
    for n, pwr in curve:
        if pwr >= target_power:
            result_n = n
            break

    # refine: if jump was large, check intermediates
    if result_n is not None and result_n > min_n:
        prev_n = max(min_n, result_n - 10)
        for n in range(prev_n, result_n):
            if _power(n) >= target_power:
                result_n = n
                break

    result_power = _power(result_n) if result_n else _power(max_n)
    if result_n is None:
        result_n = max_n  # cap

    return {
        "n": result_n,
        "power": result_power,
        "target_achieved": result_power >= target_power,
        "power_curve": curve,
    }


# ── sensitivity helpers ────────────────────────────────────────────────────

def sensitivity_table(
    design: str,
    n_timepoints: int,
    within_subject_sd: float,
    between_subject_sd: float,
    alpha: float,
    test: str,
    effect_sizes: list[float],
    sample_sizes: list[int],
    effect_profile_generator=None,
) -> pd.DataFrame:
    """Compute power for every (effect-size, sample-size) combination.

    Returns a DataFrame with effect sizes as rows and sample sizes as columns.
    """
    power_fn = (
        analytical_power_crossover if design == "Crossover"
        else analytical_power_parallel
    )
    data = {}
    for n in sample_sizes:
        col = []
        for es in effect_sizes:
            ep = None
            if effect_profile_generator is not None:
                ep = effect_profile_generator(es)
            kw = dict(
                n_timepoints=n_timepoints,
                mean_effect=es,
                within_subject_sd=within_subject_sd,
                alpha=alpha,
                test=test,
                effect_profile=ep,
            )
            if design == "Crossover":
                kw["n_subjects"] = n
            else:
                kw["n_per_group"] = n
                kw["between_subject_sd"] = between_subject_sd
            col.append(round(power_fn(**kw), 3))
        data[f"n = {n}"] = col
    df = pd.DataFrame(data, index=[f"{es}" for es in effect_sizes])
    df.index.name = "Effect size"
    return df


# ── simulation-based power (optional validation) ──────────────────────────

def simulate_power(
    n_subjects: int,
    design: str,
    n_timepoints: int,
    treatment_effect_profile: np.ndarray,
    between_subject_sd: float,
    within_subject_sd: float,
    n_simulations: int = 500,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Simulation-based power using statsmodels MixedLM.

    Returns dict with ``power``, ``ci_lower``, ``ci_upper``, ``n_converged``.
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        return {"power": np.nan, "ci_lower": np.nan, "ci_upper": np.nan,
                "n_converged": 0, "error": "statsmodels not installed"}

    rng = np.random.default_rng(seed)
    n = n_subjects
    J = n_timepoints
    profile = np.asarray(treatment_effect_profile)

    significant = 0
    converged = 0

    for _ in range(n_simulations):
        rows = []
        if design == "Crossover":
            # each subject gets treatment and control
            b = rng.normal(0, between_subject_sd, size=n)
            for subj in range(n):
                for trt in [0, 1]:
                    for j in range(J):
                        mu = 50.0 + b[subj] + trt * profile[j]
                        y = mu + rng.normal(0, within_subject_sd)
                        rows.append({
                            "subject": f"S{subj}",
                            "treatment": trt,
                            "time": j,
                            "y": y,
                        })
        else:
            # parallel: n per group
            for grp in [0, 1]:
                b = rng.normal(0, between_subject_sd, size=n)
                for subj in range(n):
                    for j in range(J):
                        mu = 50.0 + b[subj] + grp * profile[j]
                        y = mu + rng.normal(0, within_subject_sd)
                        rows.append({
                            "subject": f"G{grp}_S{subj}",
                            "treatment": grp,
                            "time": j,
                            "y": y,
                        })

        df = pd.DataFrame(rows)
        try:
            model = smf.mixedlm(
                "y ~ C(treatment) + C(time) + C(treatment):C(time)",
                df,
                groups=df["subject"],
            )
            result = model.fit(reml=True, method="powell", maxiter=200)
            converged += 1
            # test of treatment main effect
            for key in result.pvalues.index:
                if "treatment" in key.lower() and "time" not in key.lower():
                    if result.pvalues[key] < alpha:
                        significant += 1
                    break
        except Exception:
            pass

    power = significant / max(converged, 1)
    # Wilson confidence interval
    z = 1.96
    n_conv = max(converged, 1)
    centre = (significant + z ** 2 / 2) / (n_conv + z ** 2)
    half = z * np.sqrt(
        (significant * (n_conv - significant) / n_conv + z ** 2 / 4)
        / (n_conv + z ** 2)
    )
    return {
        "power": round(power, 4),
        "ci_lower": round(max(centre - half, 0), 4),
        "ci_upper": round(min(centre + half, 1), 4),
        "n_converged": converged,
        "n_simulations": n_simulations,
    }
