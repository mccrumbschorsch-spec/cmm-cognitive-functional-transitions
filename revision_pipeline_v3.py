#!/usr/bin/env python3
"""Locked reconstruction of the CMM cognitive-functional analysis.

The pipeline is intentionally strict. It refuses to guess variable mappings,
time units, disease coding, or composite construction. A cohort is analysed
only after its release, waves, exact time mapping, non-overlapping cognitive
and functional inputs, CMM coding, and optional death/censoring definitions
have been declared in a JSON configuration file.

Primary estimands
------------------
1. Continuous cognitive and functional change, fitted separately with GEE.
2. Any cognitive and any functional transition, fitted with logistic GEE.
3. Formal CMM-change-by-domain interaction in a stacked GEE.
4. Four-state decomposition as a secondary analysis.

Sensitivity structures implemented here include fixed-reference versus
wave-relative standardisation, raw change adjusted for exact interval length,
prevalence-matched thresholds, stricter thresholds, persistent impairment,
CMM4 versus complete CMM5, baseline domain position, participant clustering,
and a death absorbing state when validated mortality coding is supplied.  The
pipeline constructs a scheduled-wave response table from the audited wave
sequence, separates death before the next wave from non-death nonresponse, and
uses interval-specific stabilised inverse-probability-of-censoring weights as a
sensitivity analysis.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import build_design_matrices
from scipy import optimize, special, stats
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import brier_score_loss, roc_auc_score
from statsmodels.genmod.cov_struct import Independence
from statsmodels.genmod.families import Binomial, Gaussian


CORE_CMM4 = ("hypertension", "diabetes", "heart", "stroke")
CMM5 = CORE_CMM4 + ("cholesterol",)
STATE_ORDER = ("unimpaired", "cognitive_only", "functional_only", "joint", "death")
ADJUSTMENT_SETS = {
    "base": [],
    "mental_health": ["depression", "psychiatric"],
    "lifestyle": ["depression", "psychiatric", "smoking", "bmi", "physical_activity", "alcohol"],
    "socioeconomic": ["depression", "psychiatric", "smoking", "bmi", "physical_activity", "alcohol", "wealth", "marital"],
    "medication_burden": ["depression", "psychiatric", "smoking", "bmi", "physical_activity", "alcohol", "wealth", "marital", "polypharmacy"],
}

RESULT_SCHEMA = [
    "cohort", "analysis", "adjustment_set", "state_definition", "standardization", "persistent",
    "outcome", "exposure", "term", "estimate", "std_error", "p_value", "n", "clusters",
    "converged", "covariates_used", "estimand", "death_handling", "mortality_available",
]
FAILURE_SCHEMA = [
    "cohort", "analysis", "adjustment_set", "state_definition", "standardization", "persistent",
    "outcome", "exposure", "reason", "n", "clusters", "events", "non_events", "exposure_levels",
    "mortality_available",
]
POOLED_SCHEMA = [
    "analysis", "adjustment_set", "state_definition", "standardization", "persistent", "outcome",
    "exposure", "term", "estimand", "death_handling", "k", "pooled", "ci_low", "ci_high",
    "prediction_low", "prediction_high", "tau2", "i2", "cohorts", "covariates_by_cohort",
    "mortality_availability_by_cohort", "n_total", "clusters_total", "scale",
]
PERFORMANCE_METRIC_COLUMNS = [
    f"{metric}_{prefix}_{model}"
    for prefix in ["reviewer", "expanded"]
    for metric in ["auc", "brier", "calibration_intercept", "calibration_slope"]
    for model in ["base", "plus_cmm4"]
] + [
    f"delta_{metric}_{prefix}"
    for prefix in ["reviewer", "expanded"]
    for metric in ["auc", "brier"]
]
PERFORMANCE_SCHEMA = [
    "cohort", "outcome", "n", "participants", "events", "folds", "status", "reason", "estimand",
    "death_handling", "mortality_available", "bootstrap_iterations_completed",
] + PERFORMANCE_METRIC_COLUMNS + [
    suffix
    for metric in PERFORMANCE_METRIC_COLUMNS
    for suffix in [f"{metric}_ci_low", f"{metric}_ci_high"]
]
RISK_SCHEMA = [
    "cohort", "outcome", "n", "events", "status", "reason", "estimand", "death_handling",
    "mortality_available", "risk_cmm4_0", "risk_cmm4_1", "risk_cmm4_2", "risk_cmm4_ge3",
    "risk_difference_ge3_vs_0", "risk_difference_ci_low", "risk_difference_ci_high",
    "reciprocal_positive_risk_difference", "reciprocal_is_descriptive_not_nns",
]


class ConfigurationError(ValueError):
    pass


def ensure_schema(frame: pd.DataFrame, required_columns: list[str]) -> pd.DataFrame:
    """Return a CSV-safe frame with headers even when no rows are estimable."""
    out = frame.copy() if frame is not None else pd.DataFrame()
    for column in required_columns:
        if column not in out:
            out[column] = pd.Series(index=out.index, dtype="object")
    extras = [column for column in out.columns if column not in required_columns]
    return out[[*required_columns, *extras]]


def require(mapping: dict[str, Any], key: str, context: str) -> Any:
    value = mapping.get(key)
    if value in (None, "", "__REQUIRED__"):
        raise ConfigurationError(f"{context}: required field '{key}' is not configured")
    return value


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def canonical_code(value: Any) -> Any:
    """Canonicalise common CSV representations without guessing semantics."""
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    try:
        number = float(text)
        if np.isfinite(number):
            return str(int(number)) if number.is_integer() else format(number, ".15g")
    except ValueError:
        pass
    return text.casefold()


def recode_binary(series: pd.Series, spec: dict[str, Any], label: str) -> pd.Series:
    positive = {canonical_code(x) for x in require(spec, "positive_values", label)}
    negative = {canonical_code(x) for x in require(spec, "negative_values", label)}
    declared_missing = {canonical_code(x) for x in spec.get("missing_values", [])}
    if positive & negative or (positive | negative) & declared_missing:
        raise ConfigurationError(f"{label}: positive, negative and missing codes must not overlap")
    raw = series.map(canonical_code).astype("string")
    raw = raw.mask(raw.isin(declared_missing))
    unknown = raw.notna() & ~raw.isin(positive | negative)
    if unknown.any():
        examples = raw[unknown].value_counts().head(8).to_dict()
        raise ConfigurationError(f"{label}: undeclared non-missing codes {examples}")
    return pd.Series(np.where(raw.isin(positive), 1.0, np.where(raw.isin(negative), 0.0, np.nan)), index=series.index)


def recode_numeric(series: pd.Series, spec: dict[str, Any], label: str) -> pd.Series:
    """Strictly parse a numeric field after masking declared missing codes."""
    raw_codes = series.map(canonical_code).astype("string")
    declared_missing = {canonical_code(x) for x in spec.get("missing_values", [])}
    declared_mask = raw_codes.isin(declared_missing)
    values = pd.to_numeric(series.mask(declared_mask), errors="coerce")
    undeclared_non_numeric = series.notna() & ~declared_mask & values.isna()
    if undeclared_non_numeric.any():
        examples = series[undeclared_non_numeric].astype(str).value_counts().head(8).to_dict()
        raise ConfigurationError(f"{label}: undeclared non-numeric codes {examples}")
    if spec.get("min") is not None and (values.dropna() < float(spec["min"])).any():
        raise ConfigurationError(f"{label}: value below declared minimum")
    if spec.get("max") is not None and (values.dropna() > float(spec["max"])).any():
        raise ConfigurationError(f"{label}: value above declared maximum")
    return values


def reference_mask(frame: pd.DataFrame, reference_wave: Any) -> pd.Series:
    return frame["wave"].map(canonical_code) == canonical_code(reference_wave)


def z_from_reference(values: pd.Series, mask: pd.Series) -> tuple[pd.Series, float, float]:
    reference = numeric(values[mask]).dropna()
    mean = float(reference.mean())
    sd = float(reference.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        raise ConfigurationError("Reference-wave standard deviation is zero or unavailable")
    return (numeric(values) - mean) / sd, mean, sd


def wave_z(frame: pd.DataFrame, value: str) -> pd.Series:
    mean = frame.groupby("wave")[value].transform("mean")
    sd = frame.groupby("wave")[value].transform("std").replace(0, np.nan)
    return (frame[value] - mean) / sd


def cronbach_alpha(matrix: pd.DataFrame) -> float:
    complete = matrix.dropna()
    k = complete.shape[1]
    if k < 2 or len(complete) < 30:
        return np.nan
    item_variance = complete.var(ddof=1).sum()
    total_variance = complete.sum(axis=1).var(ddof=1)
    return float(k / (k - 1) * (1 - item_variance / total_variance)) if total_variance > 0 else np.nan


def one_factor_omega(matrix: pd.DataFrame) -> float:
    """One-factor coefficient omega from the complete-case correlation matrix.

    This is a transparent Pearson-correlation omega suitable as a common
    cross-cohort reliability audit.  For binary ADL/IADL inputs it is reported
    as a Pearson-item omega, with alpha/KR-20 retained as the fallback; it is
    not mislabeled as polychoric/ordinal omega.
    """
    complete = matrix.dropna()
    if complete.shape[1] < 2 or len(complete) < 100:
        return np.nan
    correlation = complete.corr().to_numpy(float)
    if not np.isfinite(correlation).all():
        return np.nan
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    leading = int(np.argmax(eigenvalues))
    if eigenvalues[leading] <= 1:
        return np.nan
    loadings = eigenvectors[:, leading] * math.sqrt(float(eigenvalues[leading]))
    if loadings.sum() < 0:
        loadings = -loadings
    uniqueness = np.clip(1 - np.square(loadings), 0, 1)
    numerator = float(np.square(loadings.sum()))
    denominator = numerator + float(uniqueness.sum())
    return numerator / denominator if denominator > 0 else np.nan


def build_time(frame: pd.DataFrame, spec: dict[str, Any], context: str) -> pd.Series:
    mode = require(spec, "mode", context)
    if mode == "date":
        column = require(spec, "column", context)
        dates = pd.to_datetime(frame[column], errors="coerce")
        return dates.dt.year + (dates.dt.dayofyear - 1) / 365.2425
    if mode == "year_month":
        year = numeric(frame[require(spec, "year_column", context)])
        month_column = spec.get("month_column")
        month = numeric(frame[month_column]) if month_column else pd.Series(6.5, index=frame.index)
        invalid_month = month.notna() & ~month.between(1, 12)
        if invalid_month.any():
            examples = month[invalid_month].value_counts().head(8).to_dict()
            raise ConfigurationError(f"{context}: month must be between 1 and 12; found {examples}")
        return year + (month - 0.5) / 12.0
    if mode == "wave_map":
        column = require(spec, "wave_column", context)
        mapping = {canonical_code(k): float(v) for k, v in require(spec, "wave_to_year", context).items()}
        result = frame[column].map(canonical_code).map(mapping)
        missing = frame.loc[result.isna() & frame[column].notna(), column].astype(str).unique()
        if len(missing):
            raise ConfigurationError(f"{context}: waves missing from wave_to_year: {missing[:10].tolist()}")
        return result
    raise ConfigurationError(f"{context}: unsupported time mode '{mode}'")


def component_matrix(frame: pd.DataFrame, spec: dict[str, Any], domain: str) -> tuple[pd.DataFrame, list[str]]:
    mode = require(spec, "mode", domain)
    if mode not in {"total", "components"}:
        raise ConfigurationError(f"{domain}: mode must be 'total' or 'components'")
    inputs = require(spec, "inputs", domain)
    if not isinstance(inputs, list) or not inputs:
        raise ConfigurationError(f"{domain}: inputs must be a non-empty list")
    columns = [require(x, "column", f"{domain} input") for x in inputs]
    if len(columns) != len(set(columns)):
        raise ConfigurationError(f"{domain}: duplicate input columns are not allowed")
    declared_components = set(spec.get("component_columns", []))
    if mode == "total" and declared_components.intersection(columns):
        raise ConfigurationError(f"{domain}: total and component columns cannot be combined")
    matrix = pd.DataFrame(index=frame.index)
    for item in inputs:
        column = item["column"]
        expected_role = "validated_total" if mode == "total" else "component"
        if require(item, "role", f"{domain}:{column}") != expected_role:
            raise ConfigurationError(f"{domain}:{column}: role must be '{expected_role}' for mode '{mode}'")
        raw = frame[column]
        missing_codes = {canonical_code(x) for x in item.get("missing_values", [])}
        declared_missing = raw.map(canonical_code).isin(missing_codes)
        value = numeric(raw.mask(declared_missing))
        undeclared_non_numeric = raw.notna() & ~declared_missing & value.isna()
        if undeclared_non_numeric.any():
            examples = raw[undeclared_non_numeric].astype(str).value_counts().head(8).to_dict()
            raise ConfigurationError(f"{domain}:{column}: undeclared non-numeric codes {examples}")
        valid_min = item.get("min")
        valid_max = item.get("max")
        if valid_min is not None and (value.dropna() < float(valid_min)).any():
            raise ConfigurationError(f"{domain}:{column}: values below declared minimum {valid_min}")
        if valid_max is not None and (value.dropna() > float(valid_max)).any():
            raise ConfigurationError(f"{domain}:{column}: values above declared maximum {valid_max}")
        if item.get("reverse", False):
            declared_min = require(item, "min", f"{domain}:{column}")
            declared_max = require(item, "max", f"{domain}:{column}")
            value = float(declared_max) + float(declared_min) - value
        matrix[column] = value
    if mode == "total" and len(columns) != 1:
        raise ConfigurationError(f"{domain}: total mode requires exactly one input")
    return matrix, columns


def build_composite(
    frame: pd.DataFrame,
    spec: dict[str, Any],
    reference_wave: Any,
    domain: str,
    higher_is_worse: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    matrix, columns = component_matrix(frame, spec, domain)
    ref = reference_mask(frame, reference_wave)
    minimum_fraction = float(spec.get("minimum_fraction", 1.0))
    required_count = max(1, math.ceil(len(columns) * minimum_fraction))
    standardised = pd.DataFrame(index=frame.index)
    item_parameters = {}
    for column in columns:
        z, mean, sd = z_from_reference(matrix[column], ref)
        standardised[column] = z
        item_parameters[column] = {"reference_mean": mean, "reference_sd": sd}
    if spec["mode"] == "total":
        raw = matrix.iloc[:, 0]
    else:
        raw = standardised.mean(axis=1, skipna=True)
        raw[standardised.notna().sum(axis=1) < required_count] = np.nan
    fixed_z, composite_mean, composite_sd = z_from_reference(raw, ref)
    if not higher_is_worse:
        # Cognition remains higher-is-better. Decline is sign-reversed later.
        pass
    source_metric_type = "validated_total_score" if spec["mode"] == "total" else "fixed_reference_item_z_mean"
    output = pd.DataFrame({f"{domain}_source_metric": raw, f"{domain}_z_fixed": fixed_z}, index=frame.index)
    temporary = frame.copy()
    temporary[f"{domain}_source_metric"] = raw
    output[f"{domain}_z_wave"] = wave_z(temporary, f"{domain}_source_metric")
    item_wave_audit = []
    reliability_by_wave = []
    for wave, indices in frame.groupby("wave").groups.items():
        wave_matrix = matrix.loc[indices]
        reliability_by_wave.append({
            "wave": str(wave),
            "n_rows": int(len(wave_matrix)),
            "complete_case_n": int(wave_matrix.dropna().shape[0]),
            "alpha_complete_case": cronbach_alpha(standardised.loc[indices]),
            "omega_one_factor_complete_case": one_factor_omega(standardised.loc[indices]),
        })
        for column in columns:
            values = wave_matrix[column]
            item_wave_audit.append({
                "wave": str(wave),
                "column": column,
                "n_observed": int(values.notna().sum()),
                "missing_percent": float(100 * values.isna().mean()),
                "minimum_observed": float(values.min()) if values.notna().any() else np.nan,
                "maximum_observed": float(values.max()) if values.notna().any() else np.nan,
            })
    audit = {
        "domain": domain,
        "mode": spec["mode"],
        "source_metric_type": source_metric_type,
        "inputs": columns,
        "implemented_reliability_metric": "complete_case_one_factor_Pearson_omega_and_alpha",
        "reliability_boundary": "binary-item omega is Pearson-based, not polychoric ordinal omega; alpha equals KR-20 for binary items",
        "minimum_fraction": minimum_fraction,
        "required_count": required_count,
        "reference_wave": reference_wave,
        "reference_n": int(ref.sum()),
        "alpha_reference_complete_case": cronbach_alpha(standardised.loc[ref]),
        "omega_reference_complete_case": one_factor_omega(standardised.loc[ref]),
        "composite_reference_mean": composite_mean,
        "composite_reference_sd": composite_sd,
        "item_parameters": item_parameters,
        "item_wave_audit": item_wave_audit,
        "reliability_by_wave": reliability_by_wave,
    }
    return output, audit


def carry_forward_ever(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        # Carry forward only a previously observed positive diagnosis. Missing
        # waves following an observed negative remain missing, so completeness
        # cannot be manufactured by forward-filling zeroes.
        def transform(series: pd.Series) -> pd.Series:
            prior_positive = series.eq(1).cummax()
            # An ever-diagnosed condition remains positive after its first
            # positive report, even if a later interview contains an observed
            # zero. Otherwise an apparent recovery would create -1/+1 CMM
            # changes that contradict the declared persistence semantics.
            return series.mask(prior_positive, 1.0)
        out[column] = out.groupby("id", sort=False)[column].transform(transform)
    return out


def build_cmm(frame: pd.DataFrame, specs: dict[str, Any], context: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    out = frame.copy()
    audit = []
    for disease in CMM5:
        disease_spec = specs.get(disease)
        if disease_spec is None:
            out[disease] = np.nan
            out[f"{disease}_reported"] = np.nan
            audit.append({"disease": disease, "available": False, "column": None})
            continue
        column = require(disease_spec, "column", f"{context}:{disease}")
        out[disease] = recode_binary(out[column], disease_spec, f"{context}:{disease}")
        out[f"{disease}_reported"] = out[disease]
        persistence = require(disease_spec, "persistence", f"{context}:{disease}")
        if persistence not in {"ever", "current"}:
            raise ConfigurationError(f"{context}:{disease}: persistence must be 'ever' or 'current'")
        audit.append({"disease": disease, "available": True, "column": column, "persistence": persistence})
    ever_columns = [x for x in CMM5 if specs.get(x, {}).get("persistence") == "ever"]
    out = carry_forward_ever(out, ever_columns)
    out["cmm4"] = out[list(CORE_CMM4)].sum(axis=1, min_count=len(CORE_CMM4))
    out["cmm5"] = out[list(CMM5)].sum(axis=1, min_count=len(CMM5))
    return out, audit


def optional_numeric(frame: pd.DataFrame, config: dict[str, Any], key: str) -> pd.Series:
    column = config.get(key)
    if not column:
        return pd.Series(np.nan, index=frame.index)
    if isinstance(column, dict):
        column = require(column, "column", f"covariate:{key}")
    return numeric(frame[column])


def optional_covariate(frame: pd.DataFrame, config: dict[str, Any], key: str) -> pd.Series:
    specification = config.get(key)
    if not specification:
        return pd.Series(np.nan, index=frame.index)
    if isinstance(specification, str):
        # Backward-compatible numeric covariate; the audited configuration
        # should prefer an explicit type.
        return recode_numeric(frame[specification], {}, f"covariate:{key}")
    column = require(specification, "column", f"covariate:{key}")
    kind = require(specification, "type", f"covariate:{key}")
    if kind == "binary":
        return recode_binary(frame[column], specification, f"covariate:{key}")
    if kind == "categorical":
        values = frame[column].map(canonical_code).astype("string")
        missing_values = {canonical_code(x) for x in specification.get("missing_values", [])}
        return values.mask(values.isin(missing_values))
    if kind == "numeric":
        return recode_numeric(frame[column], specification, f"covariate:{key}")
    raise ConfigurationError(f"covariate:{key}: unsupported type '{kind}'")


def harmonise_cohort(name: str, spec: dict[str, Any], input_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    release = require(spec, "release", name)
    reference_wave = require(spec, "reference_wave", name)
    source = input_root / require(spec, "file", name)
    frame = pd.read_csv(source, low_memory=False)
    source_rows = len(frame)
    id_column = require(spec, "id_column", name)
    wave_column = require(spec, "wave_column", name)
    required_columns = [id_column, wave_column]
    for column in required_columns:
        if column not in frame.columns:
            raise ConfigurationError(f"{name}: configured column '{column}' is absent")
    frame["id"] = frame[id_column].astype("string")
    frame["wave"] = frame[wave_column].map(canonical_code).astype("string")
    frame["cohort"] = name
    mortality = spec.get("mortality")
    death_wave = canonical_code(mortality.get("endpoint_wave_value")) if mortality else None
    included_waves = {canonical_code(x) for x in require(spec, "included_waves", name)}
    eligible_waves = included_waves | ({death_wave} if death_wave else set())
    frame = frame[frame["wave"].isin(eligible_waves)].copy()
    included_wave_rows = len(frame)
    if frame.empty:
        raise ConfigurationError(f"{name}: no rows remain after included_waves filtering")
    wave_sequence = [canonical_code(x) for x in require(spec, "wave_sequence", name)]
    if len(wave_sequence) != len(set(wave_sequence)):
        raise ConfigurationError(f"{name}: wave_sequence contains duplicates")
    if set(wave_sequence) != included_waves:
        raise ConfigurationError(f"{name}: included_waves and wave_sequence must contain the same living waves")
    wave_order = {wave: index for index, wave in enumerate(wave_sequence)}
    frame["scheduled_index"] = frame["wave"].map(wave_order)
    unexpected_unmapped = frame["scheduled_index"].isna() & frame["wave"].ne(death_wave)
    if unexpected_unmapped.any():
        raise ConfigurationError(f"{name}: a living wave is absent from wave_sequence")
    frame["time_years"] = build_time(frame, require(spec, "time", name), f"{name}:time")
    invalid_time_rows = int(frame["time_years"].isna().sum())
    missing_id_or_wave_rows = int(frame[["id", "wave"]].isna().any(axis=1).sum())
    frame = frame.dropna(subset=["id", "wave", "time_years"]).copy()
    frame = frame.sort_values(["id", "time_years"])
    if frame.duplicated(["id", "wave"], keep=False).any():
        examples = frame.loc[frame.duplicated(["id", "wave"], keep=False), ["id", "wave"]].head(10).to_dict("records")
        raise ConfigurationError(f"{name}: duplicate participant-wave records require audit: {examples}")
    if frame.duplicated(["id", "time_years"]).any():
        raise ConfigurationError(f"{name}: duplicate participant-time records remain")

    # Identify and validate dated death endpoints before constructing domain
    # scores. Otherwise carried placeholders on a terminal row can contaminate
    # reference and wave-relative standardisation among living observations.
    frame["mortality_available"] = bool(mortality)
    if mortality:
        encoding = require(mortality, "encoding", f"{name}:mortality")
        if encoding != "death_endpoint_row":
            raise ConfigurationError(
                f"{name}:mortality encoding must be 'death_endpoint_row'; "
                "an interview-status flag is not a validated competing-death endpoint"
            )
        column = require(mortality, "column", f"{name}:mortality")
        frame["death"] = recode_binary(frame[column], mortality, f"{name}:mortality")
        if not frame.loc[frame["death"].eq(1), "wave"].eq(death_wave).all():
            raise ConfigurationError(f"{name}: all positive mortality rows must use endpoint_wave_value={death_wave}")
        if frame.loc[frame["death"].ne(1), "wave"].eq(death_wave).any():
            raise ConfigurationError(f"{name}: endpoint wave rows must be coded as death")
        living_max_index = frame.loc[frame["death"].ne(1)].groupby("id")["scheduled_index"].max()
        frame.loc[frame["death"].eq(1), "scheduled_index"] = frame.loc[frame["death"].eq(1), "id"].map(living_max_index) + 1
        frame = frame[~(frame["death"].eq(1) & frame["scheduled_index"].isna())].copy()
        death_rows = frame[frame["death"] == 1].copy()
        repeated_deaths = death_rows.groupby("id").size()
        if (repeated_deaths > 1).any():
            examples = repeated_deaths[repeated_deaths > 1].head(8).to_dict()
            raise ConfigurationError(f"{name}: multiple death endpoint rows for participants {examples}")
        if not death_rows.empty:
            last_time = frame.groupby("id")["time_years"].transform("max")
            nonterminal_death = frame["death"].eq(1) & frame["time_years"].ne(last_time)
            if nonterminal_death.any():
                examples = frame.loc[nonterminal_death, ["id", "wave", "time_years"]].head(8).to_dict("records")
                raise ConfigurationError(f"{name}: death must be the final endpoint for each participant: {examples}")
            domain_input_columns = {
                item["column"]
                for domain in [require(spec, "cognition", name), require(spec, "function", name)]
                for item in require(domain, "inputs", f"{name}:domain")
            }
            cmm_specification = require(spec, "cmm", name)
            cmm_input_columns = {
                disease_specification["column"]
                for disease_specification in cmm_specification.values()
                if disease_specification and disease_specification.get("column")
            }
            endpoint_measurement_columns = domain_input_columns | cmm_input_columns
            frame.loc[frame["death"].eq(1), list(endpoint_measurement_columns)] = np.nan
    else:
        frame["death"] = np.nan

    frame, cmm_audit = build_cmm(frame, require(spec, "cmm", name), name)
    if mortality and frame["death"].eq(1).any():
        cmm_derived_columns = [
            *CMM5,
            *[f"{disease}_reported" for disease in CMM5],
            "cmm4",
            "cmm5",
        ]
        frame.loc[frame["death"].eq(1), cmm_derived_columns] = np.nan
    cognition, cog_audit = build_composite(frame, require(spec, "cognition", name), reference_wave, "cognition", False)
    function, func_audit = build_composite(frame, require(spec, "function", name), reference_wave, "function", True)
    overlap = set(cog_audit["inputs"]) & set(func_audit["inputs"])
    if overlap:
        raise ConfigurationError(f"{name}: cognition/function inputs overlap: {sorted(overlap)}")
    frame = pd.concat([frame, cognition, function], axis=1)
    frame["cognition_source_metric_type"] = cog_audit["source_metric_type"]
    frame["function_source_metric_type"] = func_audit["source_metric_type"]

    covariates = spec.get("covariates", {})
    for core_covariate in ["age", "sex", "education"]:
        if not isinstance(covariates.get(core_covariate), dict):
            raise ConfigurationError(f"{name}:covariates:{core_covariate} requires an explicit column/type mapping")
    frame["age"] = optional_covariate(frame, covariates, "age")
    frame["education"] = optional_covariate(frame, covariates, "education")
    frame["sex"] = optional_covariate(frame, covariates, "sex")
    for key in ["depression", "psychiatric", "smoking", "bmi", "physical_activity", "alcohol", "wealth", "marital", "polypharmacy"]:
        frame[key] = optional_covariate(frame, covariates, key)

    keep = [
        "cohort", "id", "wave", "scheduled_index", "time_years", "age", "sex", "education",
        *CMM5, *[f"{disease}_reported" for disease in CMM5], "cmm4", "cmm5", "cognition_source_metric", "cognition_z_fixed", "cognition_z_wave",
        "function_source_metric", "function_z_fixed", "function_z_wave", "cognition_source_metric_type", "function_source_metric_type", "depression", "psychiatric",
        "smoking", "bmi", "physical_activity", "alcohol", "wealth", "marital", "polypharmacy", "death", "mortality_available",
    ]
    audit = {
        "cohort": name,
        "release": release,
        "source": str(source),
        "source_rows": source_rows,
        "included_wave_rows_before_required_field_filter": included_wave_rows,
        "invalid_time_rows_excluded": invalid_time_rows,
        "missing_id_or_wave_rows_excluded": missing_id_or_wave_rows,
        "rows": len(frame),
        "participants": int(frame["id"].nunique()),
        "waves": sorted(frame["wave"].dropna().unique().tolist()),
        "wave_sequence": wave_sequence,
        "time_years_min": float(frame["time_years"].min()),
        "time_years_max": float(frame["time_years"].max()),
        "cmm": cmm_audit,
        "cognition": cog_audit,
        "function": func_audit,
        "mortality_available": bool(mortality),
        "death_endpoint_rows_excluded_from_domain_scoring": int(frame["death"].eq(1).sum()),
        "death_endpoint_rows_excluded_from_interview_cmm_scoring": int(frame["death"].eq(1).sum()),
    }
    return frame[keep], audit


def build_intervals(long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cohort, participant), group in long.groupby(["cohort", "id"], sort=False):
        g = group.sort_values("time_years").reset_index(drop=True)
        for index in range(len(g) - 1):
            start, end = g.iloc[index], g.iloc[index + 1]
            # Continuous and living-state analyses use scheduled adjacent
            # waves, not merely the next observed interview.  This prevents a
            # skipped wave from being mislabeled as one adjacent interval.
            if end["death"] != 1 and end["scheduled_index"] != start["scheduled_index"] + 1:
                continue
            dt = end["time_years"] - start["time_years"]
            if not np.isfinite(dt) or dt <= 0:
                continue
            row = {
                "cohort": cohort,
                "id": participant,
                "start_wave": start["wave"],
                "end_wave": end["wave"],
                "start_scheduled_index": start["scheduled_index"],
                "end_scheduled_index": end["scheduled_index"],
                "interval_years": dt,
                "age": start["age"],
                "sex": start["sex"],
                "education": start["education"],
                "baseline_cognition_z_fixed": start["cognition_z_fixed"],
                "baseline_function_z_fixed": start["function_z_fixed"],
                "baseline_cognition_z_wave": start["cognition_z_wave"],
                "baseline_function_z_wave": start["function_z_wave"],
                "baseline_cognition_source_metric": start["cognition_source_metric"],
                "baseline_function_source_metric": start["function_source_metric"],
                "cmm4_start": start["cmm4"],
                "cmm4_end": end["cmm4"],
                "cmm5_start": start["cmm5"],
                "cmm5_end": end["cmm5"],
                "cognition_change_fixed": end["cognition_z_fixed"] - start["cognition_z_fixed"],
                "function_change_fixed": end["function_z_fixed"] - start["function_z_fixed"],
                "cognition_change_wave": end["cognition_z_wave"] - start["cognition_z_wave"],
                "function_change_wave": end["function_z_wave"] - start["function_z_wave"],
                "cognition_change_source_metric": end["cognition_source_metric"] - start["cognition_source_metric"],
                "function_change_source_metric": end["function_source_metric"] - start["function_source_metric"],
                "death_next": end["death"] if pd.notna(end["death"]) else np.nan,
                "mortality_available": bool(start["mortality_available"]),
            }
            for cmm in ["cmm4", "cmm5"]:
                row[f"{cmm}_change_raw"] = row[f"{cmm}_end"] - row[f"{cmm}_start"] if pd.notna(row[f"{cmm}_start"]) and pd.notna(row[f"{cmm}_end"]) else np.nan
                row[f"{cmm}_change_annual"] = row[f"{cmm}_change_raw"] / dt if pd.notna(row[f"{cmm}_change_raw"]) else np.nan
            new_diseases = [
                disease for disease in CORE_CMM4
                if pd.notna(start[disease]) and pd.notna(end[disease]) and start[disease] == 0 and end[disease] == 1
            ]
            if any(pd.isna(start[disease]) or pd.isna(end[disease]) for disease in CORE_CMM4):
                row["cmm4_change_sustained_raw"] = np.nan
                row["sustained_confirmation_years"] = np.nan
            elif not new_diseases:
                row["cmm4_change_sustained_raw"] = row["cmm4_change_raw"]
                row["sustained_confirmation_years"] = dt
            elif index + 2 >= len(g):
                row["cmm4_change_sustained_raw"] = np.nan
                row["sustained_confirmation_years"] = np.nan
            else:
                confirmation = g.iloc[index + 2]
                if confirmation["death"] == 1 or confirmation["scheduled_index"] != start["scheduled_index"] + 2:
                    row["cmm4_change_sustained_raw"] = np.nan
                    row["sustained_confirmation_years"] = np.nan
                else:
                    confirmation_values = [confirmation[f"{disease}_reported"] for disease in new_diseases]
                    if any(pd.isna(value) for value in confirmation_values):
                        row["cmm4_change_sustained_raw"] = np.nan
                        row["sustained_confirmation_years"] = np.nan
                    else:
                        sustained_end = {
                            disease: (1.0 if disease not in new_diseases or confirmation[f"{disease}_reported"] == 1 else 0.0)
                            if end[disease] == 1 else end[disease]
                            for disease in CORE_CMM4
                        }
                        row["cmm4_change_sustained_raw"] = float(sum(sustained_end.values()) - start["cmm4"])
                        row["sustained_confirmation_years"] = float(confirmation["time_years"] - start["time_years"])
            for scale in ["fixed", "wave"]:
                row[f"cognitive_worsening_{scale}_raw"] = -row[f"cognition_change_{scale}"] if pd.notna(row[f"cognition_change_{scale}"]) else np.nan
                row[f"functional_worsening_{scale}_raw"] = row[f"function_change_{scale}"] if pd.notna(row[f"function_change_{scale}"]) else np.nan
                row[f"cognitive_worsening_{scale}_annual"] = -row[f"cognition_change_{scale}"] / dt if pd.notna(row[f"cognition_change_{scale}"]) else np.nan
                row[f"functional_worsening_{scale}_annual"] = row[f"function_change_{scale}"] / dt if pd.notna(row[f"function_change_{scale}"]) else np.nan
            for covariate in ["depression", "psychiatric", "smoking", "bmi", "physical_activity", "alcohol", "wealth", "marital", "polypharmacy"]:
                row[covariate] = start[covariate]
            rows.append(row)
    result = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    if not result.empty:
        result["interval_stratum"] = pd.cut(
            result["interval_years"],
            bins=[-np.inf, 1.5, 2.5, np.inf],
            labels=["le_1.5y", "gt_1.5_to_2.5y", "gt_2.5y"],
        ).astype("string")
        q1 = result.groupby("cohort")["interval_years"].transform(lambda x: x.quantile(0.25))
        q3 = result.groupby("cohort")["interval_years"].transform(lambda x: x.quantile(0.75))
        result["central_interval_range"] = result["interval_years"].between(q1, q3)
    return result


def build_scheduled_response_table(long: pd.DataFrame) -> pd.DataFrame:
    """Construct next-scheduled-wave response while separating competing death."""
    rows: list[dict[str, Any]] = []
    cohort_last_indices = long.loc[long["death"].ne(1)].groupby("cohort")["scheduled_index"].max().astype(int).to_dict()
    covariates = [
        "age", "sex", "education", "cmm4", "cognition_z_fixed", "function_z_fixed",
        "depression", "psychiatric", "smoking", "bmi", "physical_activity", "alcohol",
        "wealth", "marital",
    ]
    for (cohort, participant), group in long.groupby(["cohort", "id"], sort=False):
        living = group[group["death"].ne(1)].sort_values("scheduled_index")
        if living.empty:
            continue
        living_by_index = {int(row["scheduled_index"]): row for _, row in living.iterrows()}
        death_indices = {
            int(row["scheduled_index"])
            for _, row in group[group["death"].eq(1)].iterrows()
            if pd.notna(row["scheduled_index"])
        }
        cohort_last_index = cohort_last_indices[cohort]
        indices = sorted(living_by_index)
        for index in indices:
            if index >= cohort_last_index:
                continue
            start = living_by_index[index]
            expected = index + 1
            responded = int(expected in living_by_index)
            competing_death = int(expected in death_indices)
            row: dict[str, Any] = {
                "cohort": cohort,
                "id": participant,
                "start_wave": start["wave"],
                "start_scheduled_index": index,
                "responded_next_scheduled_wave": responded,
                "competing_death_before_next_wave": competing_death,
                "eligible_non_death_response_model": int(not competing_death),
                "prior_adjacent_response": int((index - 1) in living_by_index),
            }
            for covariate in covariates:
                row[covariate] = start[covariate]
            rows.append(row)
    return pd.DataFrame(rows)


def estimate_ipcw(response: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate interval-specific stabilised IPCW within each cohort."""
    weighted_parts: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for cohort, cohort_data in response.groupby("cohort", sort=False):
        data = cohort_data[cohort_data["eligible_non_death_response_model"].eq(1)].copy()
        outcome = "responded_next_scheduled_wave"
        predictors = ["age", "education", "cmm4", "cognition_z_fixed", "function_z_fixed", "prior_adjacent_response"]
        mental = "depression" if data["depression"].notna().mean() >= 0.5 else "psychiatric"
        if data[mental].notna().mean() >= 0.5:
            predictors.append(mental)
        # Explicit missing indicators plus median imputation keep the response
        # denominator representative instead of silently dropping nonresponders.
        for column in predictors:
            values = numeric(data[column])
            data[f"{column}_missing"] = values.isna().astype(int)
            median = float(values.median()) if values.notna().any() else 0.0
            data[column] = values.fillna(median)
        data["sex"] = data["sex"].astype("string").fillna("missing")
        numerator_formula = f"{outcome} ~ C(start_wave)"
        denominator_terms = ["C(start_wave)", "C(sex)", *predictors, *[f"{column}_missing" for column in predictors]]
        denominator_formula = f"{outcome} ~ " + " + ".join(denominator_terms)
        try:
            numerator = smf.glm(numerator_formula, data=data, family=Binomial()).fit(maxiter=200)
            denominator = smf.glm(denominator_formula, data=data, family=Binomial()).fit(maxiter=200)
            p_num = np.clip(np.asarray(numerator.predict(data), float), 0.01, 0.99)
            p_den = np.clip(np.asarray(denominator.predict(data), float), 0.01, 0.99)
            data["ipcw_untruncated"] = p_num / p_den
            observed_weights = data.loc[data[outcome].eq(1), "ipcw_untruncated"]
            lower = float(observed_weights.quantile(0.01))
            upper = float(observed_weights.quantile(0.99))
            data["ipcw"] = data["ipcw_untruncated"].clip(lower, upper)
            weighted_parts.append(data)
            audits.append({
                "cohort": cohort,
                "scheduled_risk_rows": int(len(cohort_data)),
                "competing_deaths_excluded_from_censoring_model": int(cohort_data["competing_death_before_next_wave"].sum()),
                "non_death_response_model_rows": int(len(data)),
                "responded_rows": int(data[outcome].sum()),
                "non_death_nonresponse_rows": int((1 - data[outcome]).sum()),
                "response_percent": float(100 * data[outcome].mean()),
                "weight_p01": lower,
                "weight_median": float(data.loc[data[outcome].eq(1), "ipcw"].median()),
                "weight_p99": upper,
                "weight_max_after_truncation": float(data.loc[data[outcome].eq(1), "ipcw"].max()),
                "status": "estimated",
                "reason": "",
            })
        except Exception as exc:
            audits.append({
                "cohort": cohort,
                "scheduled_risk_rows": int(len(cohort_data)),
                "competing_deaths_excluded_from_censoring_model": int(cohort_data["competing_death_before_next_wave"].sum()),
                "non_death_response_model_rows": int(len(data)),
                "responded_rows": int(data[outcome].sum()),
                "non_death_nonresponse_rows": int((1 - data[outcome]).sum()),
                "status": "not_estimable",
                "reason": str(exc),
            })
    weighted = pd.concat(weighted_parts, ignore_index=True) if weighted_parts else pd.DataFrame()
    return weighted, pd.DataFrame(audits)


def ipcw_sensitivity_models(
    intervals: pd.DataFrame,
    transitions: pd.DataFrame,
    weighted_response: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    results: list[pd.DataFrame] = []
    failures: list[dict[str, Any]] = []
    if weighted_response.empty:
        return pd.DataFrame(), pd.DataFrame()
    weights = weighted_response.loc[
        weighted_response["responded_next_scheduled_wave"].eq(1),
        ["cohort", "id", "start_wave", "ipcw"],
    ].drop_duplicates(["cohort", "id", "start_wave"])
    interval_data = intervals.merge(weights, on=["cohort", "id", "start_wave"], how="left", validate="many_to_one")
    transition_data = transitions.merge(weights, on=["cohort", "id", "start_wave"], how="left", validate="many_to_one")
    for cohort, data in interval_data.groupby("cohort"):
        for outcome, baseline in [
            ("cognitive_worsening_fixed_raw", "baseline_cognition_z_fixed"),
            ("functional_worsening_fixed_raw", "baseline_function_z_fixed"),
        ]:
            needed = [outcome, baseline, "cmm4_change_raw", "cmm4_start", "id", "age", "sex", "education", "interval_years", "ipcw"]
            d = data.dropna(subset=needed).copy()
            selected, missing = resolve_adjustment_covariates(d, ADJUSTMENT_SETS["mental_health"])
            if selected is None:
                failures.append(failure_record(cohort=cohort, analysis="continuous_gee_ipcw", adjustment_set="mental_health", outcome=outcome, exposure="cmm4_change_raw", reason=f"required covariates unavailable: {','.join(missing)}", data=d))
                continue
            d = d.dropna(subset=selected)
            if len(d) < 500 or d["cmm4_change_raw"].nunique() < 2:
                failures.append(failure_record(cohort=cohort, analysis="continuous_gee_ipcw", adjustment_set="mental_health", outcome=outcome, exposure="cmm4_change_raw", reason="insufficient weighted observations or exposure variation", data=d))
                continue
            formula = f"{outcome} ~ cmm4_change_raw + cmm4_start + {baseline} + age + C(sex) + education + interval_years" + (" + " + " + ".join(selected) if selected else "")
            try:
                _, rows = gee_fit(formula, d, outcome, Gaussian(), weight_column="ipcw")
                rows = rows[rows["term"].eq("cmm4_change_raw")].copy()
                rows["cohort"] = cohort; rows["analysis"] = "continuous_gee_ipcw"; rows["adjustment_set"] = "mental_health"
                rows["exposure"] = "cmm4_change_raw"; rows["state_definition"] = "continuous"; rows["standardization"] = "fixed"; rows["persistent"] = False
                rows["covariates_used"] = ",".join(selected); rows["estimand"] = "ipcw_observed_interval_change"; rows["death_handling"] = "competing_death_excluded_from_censoring_model"
                results.append(rows)
            except Exception as exc:
                failures.append(failure_record(cohort=cohort, analysis="continuous_gee_ipcw", adjustment_set="mental_health", outcome=outcome, exposure="cmm4_change_raw", reason=str(exc), data=d))
    origin = transition_data[(transition_data["state_from"].eq("unimpaired")) & transition_data["state_to"].ne("death")]
    for cohort, data in origin.groupby("cohort"):
        for outcome in ["any_cognitive", "any_functional"]:
            needed = [outcome, "cmm4_start", "id", "age", "sex", "education", "interval_years", "baseline_cognition", "baseline_function", "ipcw"]
            d = data.dropna(subset=needed).copy()
            selected, missing = resolve_adjustment_covariates(d, ADJUSTMENT_SETS["mental_health"])
            if selected is None:
                failures.append(failure_record(cohort=cohort, analysis="transition_gee_ipcw", adjustment_set="mental_health", outcome=outcome, exposure="cmm4_start", reason=f"required covariates unavailable: {','.join(missing)}", data=d)); continue
            d = d.dropna(subset=selected)
            events = int(d[outcome].sum()); non_events = int(len(d) - events)
            if len(d) < 500 or min(events, non_events) < 20 or d["cmm4_start"].nunique() < 2:
                failures.append(failure_record(cohort=cohort, analysis="transition_gee_ipcw", adjustment_set="mental_health", outcome=outcome, exposure="cmm4_start", reason="insufficient weighted observations, events, or exposure variation", data=d)); continue
            formula = f"{outcome} ~ cmm4_start + baseline_cognition + baseline_function + age + C(sex) + education + interval_years" + (" + " + " + ".join(selected) if selected else "")
            try:
                _, rows = gee_fit(
                    formula, d, outcome, Binomial(), weight_column="ipcw",
                    covariance_type="robust",
                    required_terms=["cmm4_start"],
                )
                rows = rows[rows["term"].eq("cmm4_start")].copy()
                rows["cohort"] = cohort; rows["analysis"] = "transition_gee_ipcw"; rows["adjustment_set"] = "mental_health"
                rows["exposure"] = "cmm4_start"; rows["state_definition"] = str(data["state_definition"].iloc[0]); rows["standardization"] = str(data["standardization"].iloc[0]); rows["persistent"] = False
                rows["covariates_used"] = ",".join(selected); rows["estimand"] = "ipcw_survivor_conditional_domain_transition"; rows["death_handling"] = "competing_death_excluded_from_censoring_model"
                results.append(rows)
            except Exception as exc:
                failures.append(failure_record(cohort=cohort, analysis="transition_gee_ipcw", adjustment_set="mental_health", outcome=outcome, exposure="cmm4_start", reason=str(exc), data=d))
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame(), pd.DataFrame(failures)


def source_metric_wave_fixed_effect_models(
    intervals: pd.DataFrame,
    *,
    adjustment_names: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cohort-specific source-metric sensitivity models with wave fixed effects.

    In total-score mode the metric is the declared source total. In component
    mode it is the mean of fixed-reference item z scores, not a raw score.
    Units are therefore retained at cohort level and are not meta-analysed.
    """
    results, failures = [], []
    data = intervals.copy()
    data["cognitive_worsening_source_metric"] = -data["cognition_change_source_metric"]
    data["functional_worsening_source_metric"] = data["function_change_source_metric"]
    for cohort, cohort_data in data.groupby("cohort"):
        for outcome, baseline in [
            ("cognitive_worsening_source_metric", "baseline_cognition_source_metric"),
            ("functional_worsening_source_metric", "baseline_function_source_metric"),
        ]:
            for adjustment_set, covariates in adjustment_subset(adjustment_names).items():
                selected_covariates, missing_covariates = resolve_adjustment_covariates(cohort_data, covariates)
                if selected_covariates is None:
                    failures.append(failure_record(
                        cohort=cohort,
                        analysis="continuous_source_metric_wave_fe",
                        adjustment_set=adjustment_set,
                        outcome=outcome,
                        exposure="cmm4_change_raw",
                        reason=f"required covariates unavailable: {','.join(missing_covariates)}",
                        data=cohort_data,
                    ))
                    continue
                needed = [
                    outcome, baseline, "cmm4_change_raw", "cmm4_start", "id", "age", "sex",
                    "education", "interval_years", "start_wave", *selected_covariates,
                ]
                d = cohort_data.dropna(subset=needed).copy()
                if len(d) < 500 or d["cmm4_change_raw"].nunique() < 2:
                    failures.append(failure_record(
                        cohort=cohort,
                        analysis="continuous_source_metric_wave_fe",
                        adjustment_set=adjustment_set,
                        outcome=outcome,
                        exposure="cmm4_change_raw",
                        reason="insufficient observations or exposure variation",
                        data=d,
                    ))
                    continue
                rhs = " + ".join([
                    "cmm4_change_raw", "cmm4_start", baseline, "age", "C(sex)", "education",
                    "interval_years", "C(start_wave)", *selected_covariates,
                ])
                try:
                    _, rows = gee_fit(f"{outcome} ~ {rhs}", d, outcome, Gaussian())
                    rows["cohort"] = cohort
                    rows["analysis"] = "continuous_source_metric_wave_fe"
                    rows["adjustment_set"] = adjustment_set
                    rows["exposure"] = "cmm4_change_raw"
                    rows["state_definition"] = "continuous"
                    rows["standardization"] = "source_total_or_fixed_reference_item_z_mean"
                    rows["persistent"] = False
                    rows["covariates_used"] = ",".join(selected_covariates)
                    rows["estimand"] = "observed_interval_change"
                    rows["death_handling"] = "not_applicable_to_observed_interval_change"
                    results.append(rows[rows["term"] == "cmm4_change_raw"])
                except Exception as exc:
                    failures.append(failure_record(
                        cohort=cohort,
                        analysis="continuous_source_metric_wave_fe",
                        adjustment_set=adjustment_set,
                        outcome=outcome,
                        exposure="cmm4_change_raw",
                        reason=str(exc),
                        data=d,
                    ))
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame(), pd.DataFrame(failures)


def interval_length_sensitivity_models(intervals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fixed-reference CMM4 models within prespecified interval subsets."""
    results, failures = [], []
    subsets: list[tuple[str, pd.DataFrame]] = [("central_cohort_iqr", intervals[intervals["central_interval_range"]].copy())]
    subsets.extend((str(label), group.copy()) for label, group in intervals.groupby("interval_stratum", dropna=False))
    for interval_group, subset in subsets:
        for cohort, data in subset.groupby("cohort"):
            for outcome, baseline in [
                ("cognitive_worsening_fixed_raw", "baseline_cognition_z_fixed"),
                ("functional_worsening_fixed_raw", "baseline_function_z_fixed"),
            ]:
                needed = [outcome, baseline, "cmm4_change_raw", "cmm4_start", "id", "age", "sex", "education", "interval_years"]
                d = data.dropna(subset=needed).copy()
                if len(d) < 500 or d["cmm4_change_raw"].nunique() < 2:
                    failures.append(failure_record(
                        cohort=cohort,
                        analysis=f"continuous_interval_{interval_group}",
                        adjustment_set="base",
                        outcome=outcome,
                        exposure="cmm4_change_raw",
                        reason="insufficient observations or exposure variation",
                        data=d,
                    ))
                    continue
                rhs = " + ".join(["cmm4_change_raw", "cmm4_start", baseline, "age", "C(sex)", "education", "interval_years"])
                try:
                    _, rows = gee_fit(f"{outcome} ~ {rhs}", d, outcome, Gaussian())
                    rows["cohort"] = cohort
                    rows["analysis"] = f"continuous_interval_{interval_group}"
                    rows["adjustment_set"] = "base"
                    rows["exposure"] = "cmm4_change_raw"
                    rows["state_definition"] = "continuous"
                    rows["standardization"] = "fixed"
                    rows["persistent"] = False
                    rows["estimand"] = "observed_interval_change_within_interval_subset"
                    rows["death_handling"] = "not_applicable_to_observed_interval_change"
                    results.append(rows[rows["term"] == "cmm4_change_raw"])
                except Exception as exc:
                    failures.append(failure_record(
                        cohort=cohort,
                        analysis=f"continuous_interval_{interval_group}",
                        adjustment_set="base",
                        outcome=outcome,
                        exposure="cmm4_change_raw",
                        reason=str(exc),
                        data=d,
                    ))
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame(), pd.DataFrame(failures)


def sustained_diagnosis_sensitivity_models(intervals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Require a newly reported CMM4 condition to be reported again next wave."""
    results, failures = [], []
    for cohort, data in intervals.groupby("cohort"):
        for standardization in ["fixed", "wave"]:
            for outcome, baseline in [
                (f"cognitive_worsening_{standardization}_raw", f"baseline_cognition_z_{standardization}"),
                (f"functional_worsening_{standardization}_raw", f"baseline_function_z_{standardization}"),
            ]:
                needed = [outcome, baseline, "cmm4_change_sustained_raw", "cmm4_start", "id", "age", "sex", "education", "interval_years", "sustained_confirmation_years"]
                d = data.dropna(subset=needed).copy()
                if len(d) < 500 or d["cmm4_change_sustained_raw"].nunique() < 2:
                    failure = failure_record(
                        cohort=cohort,
                        analysis="continuous_sustained_diagnosis",
                        adjustment_set="base",
                        outcome=outcome,
                        exposure="cmm4_change_sustained_raw",
                        reason="insufficient observations or sustained-exposure variation",
                        data=d,
                    )
                    failure["standardization"] = standardization
                    failures.append(failure)
                    continue
                rhs = " + ".join(["cmm4_change_sustained_raw", "cmm4_start", baseline, "age", "C(sex)", "education", "interval_years", "sustained_confirmation_years"])
                try:
                    _, rows = gee_fit(f"{outcome} ~ {rhs}", d, outcome, Gaussian())
                    rows["cohort"] = cohort
                    rows["analysis"] = "continuous_sustained_diagnosis"
                    rows["adjustment_set"] = "base"
                    rows["exposure"] = "cmm4_change_sustained_raw"
                    rows["state_definition"] = "continuous"
                    rows["standardization"] = standardization
                    rows["persistent"] = False
                    rows["estimand"] = "observed_interval_change_with_sustained_new_diagnosis"
                    rows["death_handling"] = "requires_observed_confirmation_and_is_survivor_selected"
                    results.append(rows[rows["term"] == "cmm4_change_sustained_raw"])
                except Exception as exc:
                    failure = failure_record(
                        cohort=cohort,
                        analysis="continuous_sustained_diagnosis",
                        adjustment_set="base",
                        outcome=outcome,
                        exposure="cmm4_change_sustained_raw",
                        reason=str(exc),
                        data=d,
                    )
                    failure["standardization"] = standardization
                    failures.append(failure)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame(), pd.DataFrame(failures)


def closest_observed_tail_cut(values: pd.Series, target: float, lower_tail: bool) -> tuple[float, float]:
    """Choose an observed cut-point without splitting tied scores.

    The chosen score minimises the absolute target-realisation difference.
    If two observed scores are equally close, the one producing the smaller
    tail occupancy is selected (the prespecified stricter tie-break).
    """
    observed = numeric(values).dropna()
    if observed.empty:
        return np.nan, np.nan
    candidates = np.sort(observed.unique())
    occupancy = np.array([
        observed.le(cut).mean() if lower_tail else observed.ge(cut).mean()
        for cut in candidates
    ])
    distance = np.abs(occupancy - target)
    best_distance = float(distance.min())
    eligible = np.flatnonzero(np.isclose(distance, best_distance, rtol=0, atol=1e-12))
    strictest_occupancy = float(occupancy[eligible].min())
    strictest = eligible[np.isclose(occupancy[eligible], strictest_occupancy, rtol=0, atol=1e-12)]
    # A remaining tie can occur only with degenerate floating representations;
    # choose the more extreme score deterministically.
    index = int(strictest[0] if lower_tail else strictest[-1])
    return float(candidates[index]), float(occupancy[index])


def apply_states(
    long: pd.DataFrame,
    definition: str,
    prevalence: float | None = None,
    standardization: str = "fixed",
) -> pd.DataFrame:
    out = long.copy()
    if standardization not in {"fixed", "wave"}:
        raise ValueError(standardization)
    cognition_column = f"cognition_z_{standardization}"
    function_column = f"function_z_{standardization}"
    out["target_occupancy"] = np.nan
    if definition in {"primary", "legacy"}:
        cog_cut, func_cut = -0.43, 0.43
        out["cog_cut"] = cog_cut
        out["func_cut"] = func_cut
        out["cognitive_impairment"] = out[cognition_column] <= cog_cut
        out["functional_impairment"] = out[function_column] >= func_cut
    elif definition == "strict":
        cog_cut, func_cut = -0.67, 0.67
        out["cog_cut"] = cog_cut
        out["func_cut"] = func_cut
        out["cognitive_impairment"] = out[cognition_column] <= cog_cut
        out["functional_impairment"] = out[function_column] >= func_cut
    elif definition == "matched_prevalence":
        if prevalence is None:
            raise ValueError("prevalence is required")
        # Build both domain cut-points on the same living, jointly observed
        # cohort-wave denominator.  Discrete function scores remain tied;
        # identical values are never split to force an exact percentage.
        living = out["death"].ne(1) & out[[cognition_column, function_column]].notna().all(axis=1)
        out["cog_cut"] = np.nan
        out["func_cut"] = np.nan
        out.loc[living, "target_occupancy"] = prevalence
        for _, indices in out.loc[living].groupby(["cohort", "wave"], sort=False).groups.items():
            cog_cut, _ = closest_observed_tail_cut(out.loc[indices, cognition_column], prevalence, lower_tail=True)
            func_cut, _ = closest_observed_tail_cut(out.loc[indices, function_column], prevalence, lower_tail=False)
            out.loc[indices, "cog_cut"] = cog_cut
            out.loc[indices, "func_cut"] = func_cut
        out["cognitive_impairment"] = out[cognition_column] <= out["cog_cut"]
        out["functional_impairment"] = out[function_column] >= out["func_cut"]
    else:
        raise ValueError(definition)
    missing = out[[cognition_column, function_column]].isna().any(axis=1)
    out["state"] = np.select(
        [out["cognitive_impairment"] & out["functional_impairment"], out["cognitive_impairment"], out["functional_impairment"]],
        ["joint", "cognitive_only", "functional_only"],
        default="unimpaired",
    )
    out.loc[missing, "state"] = np.nan
    out.loc[out["death"] == 1, "state"] = "death"
    out["state_definition"] = ("legacy" if definition == "primary" else definition) if prevalence is None else f"matched_{prevalence:.2f}"
    out["standardization"] = standardization
    return out


def build_transitions(stated: pd.DataFrame, persistent: bool = False) -> pd.DataFrame:
    rows = []
    for (cohort, participant), group in stated.groupby(["cohort", "id"], sort=False):
        g = group.sort_values("time_years").reset_index(drop=True)
        for index in range(len(g) - 1):
            start, end = g.iloc[index], g.iloc[index + 1]
            if pd.isna(start["state"]) or pd.isna(end["state"]) or start["state"] == "death":
                continue
            if end["state"] != "death" and end["scheduled_index"] != start["scheduled_index"] + 1:
                continue
            if persistent and (index + 2 >= len(g) or pd.isna(g.iloc[index + 2]["state"])):
                continue
            to_state = end["state"]
            confirmation_state = g.iloc[index + 2]["state"] if index + 2 < len(g) else np.nan
            if persistent and (
                confirmation_state == "death"
                or g.iloc[index + 2]["scheduled_index"] != start["scheduled_index"] + 2
            ):
                # Persistence is not observable after a competing death.
                continue
            row = {
                "cohort": cohort,
                "id": participant,
                "state_definition": start["state_definition"],
                "standardization": start["standardization"],
                "persistent": persistent,
                "state_from": start["state"],
                "state_to": to_state,
                "confirmation_state": confirmation_state,
                "start_wave": start["wave"],
                "end_wave": end["wave"],
                "confirmation_wave": g.iloc[index + 2]["wave"] if persistent else np.nan,
                "start_scheduled_index": start["scheduled_index"],
                "end_scheduled_index": end["scheduled_index"],
                "interval_years": end["time_years"] - start["time_years"],
                "age": start["age"],
                "sex": start["sex"],
                "education": start["education"],
                "cmm4_start": start["cmm4"],
                "cmm5_start": start["cmm5"],
                "baseline_cognition": start[f"cognition_z_{start['standardization']}"],
                "baseline_function": start[f"function_z_{start['standardization']}"],
                "confirmation_years": (g.iloc[index + 2]["time_years"] - start["time_years"]) if persistent else np.nan,
                "mortality_available": bool(start["mortality_available"]),
            }
            for covariate in ["depression", "psychiatric", "smoking", "bmi", "physical_activity", "alcohol", "wealth", "marital", "polypharmacy"]:
                row[covariate] = start[covariate]
            rows.append(row)
    transitions = pd.DataFrame(rows)
    if transitions.empty:
        return transitions
    cognitive_at_end = transitions["state_to"].isin(["cognitive_only", "joint"])
    functional_at_end = transitions["state_to"].isin(["functional_only", "joint"])
    if persistent:
        cognitive_confirmed = transitions["confirmation_state"].isin(["cognitive_only", "joint"])
        functional_confirmed = transitions["confirmation_state"].isin(["functional_only", "joint"])
        transitions["any_cognitive"] = (cognitive_at_end & cognitive_confirmed).astype(int)
        transitions["any_functional"] = (functional_at_end & functional_confirmed).astype(int)
        transitions["cognitive_only"] = ((transitions["state_to"] == "cognitive_only") & (transitions["confirmation_state"] == "cognitive_only")).astype(int)
        transitions["functional_only"] = ((transitions["state_to"] == "functional_only") & (transitions["confirmation_state"] == "functional_only")).astype(int)
        transitions["joint"] = ((transitions["state_to"] == "joint") & (transitions["confirmation_state"] == "joint")).astype(int)
    else:
        transitions["any_cognitive"] = cognitive_at_end.astype(int)
        transitions["any_functional"] = functional_at_end.astype(int)
        transitions["cognitive_only"] = (transitions["state_to"] == "cognitive_only").astype(int)
        transitions["functional_only"] = (transitions["state_to"] == "functional_only").astype(int)
        transitions["joint"] = (transitions["state_to"] == "joint").astype(int)
    transitions["death"] = (transitions["state_to"] == "death").astype(int)
    return transitions


def gee_fit(
    formula: str,
    data: pd.DataFrame,
    outcome: str,
    family: Any,
    weight_column: str | None = None,
    covariance_type: str = "robust",
    required_terms: Iterable[str] | None = None,
) -> tuple[Any, pd.DataFrame]:
    model = smf.gee(
        formula,
        groups="id",
        data=data,
        family=family,
        # Independence is the prespecified working correlation; inference uses
        # the participant-clustered robust sandwich covariance returned by GEE.
        cov_struct=Independence(),
        weights=data[weight_column] if weight_column else None,
    )
    result = model.fit(maxiter=200, cov_type=covariance_type)
    if not bool(result.converged):
        raise RuntimeError("GEE did not converge")
    estimates = np.asarray(result.params, float)
    standard_errors = np.asarray(result.bse, float)
    if not np.isfinite(estimates).all():
        raise RuntimeError("GEE returned a non-finite coefficient")
    if required_terms is None:
        required_positions = np.arange(len(result.params))
    else:
        missing_terms = [term for term in required_terms if term not in result.params.index]
        if missing_terms:
            raise RuntimeError(f"GEE omitted required terms: {missing_terms}")
        required_positions = np.array([result.params.index.get_loc(term) for term in required_terms], dtype=int)
    required_errors = standard_errors[required_positions]
    if not np.isfinite(required_errors).all() or (required_errors <= 0).any():
        raise RuntimeError("GEE returned a non-finite or non-positive standard error for a target term")
    rows = pd.DataFrame({"term": result.params.index, "estimate": result.params.values, "std_error": result.bse.values, "p_value": result.pvalues.values})
    rows["outcome"] = outcome
    rows["n"] = int(result.nobs)
    rows["clusters"] = int(data["id"].nunique())
    rows["converged"] = bool(result.converged)
    return result, rows


def failure_record(
    *,
    cohort: str,
    analysis: str,
    adjustment_set: str,
    outcome: str,
    exposure: str,
    reason: str,
    data: pd.DataFrame | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "cohort": cohort,
        "analysis": analysis,
        "adjustment_set": adjustment_set,
        "outcome": outcome,
        "exposure": exposure,
        "reason": reason,
        "n": 0 if data is None else len(data),
        "clusters": 0 if data is None or "id" not in data else int(data["id"].nunique()),
        "events": np.nan,
        "non_events": np.nan,
        "exposure_levels": np.nan,
    }
    if data is not None:
        if outcome in data and set(data[outcome].dropna().unique()).issubset({0, 1}):
            row["events"] = int(data[outcome].sum())
            row["non_events"] = int(len(data) - data[outcome].sum())
        if exposure in data:
            row["exposure_levels"] = int(data[exposure].nunique(dropna=True))
    return row


def resolve_adjustment_covariates(data: pd.DataFrame, requested: list[str]) -> tuple[list[str] | None, list[str]]:
    """Allow one harmonised mental-health construct while requiring all others.

    Depression and psychiatric measures are treated as alternative or
    complementary mental-health constructs because not every cohort contains
    both. Later lifestyle, socioeconomic and medication-burden covariates are
    required when their adjustment set is requested. The exact terms retained
    are written to the model result.
    """
    mental_health = [name for name in ["depression", "psychiatric"] if name in requested]
    other = [name for name in requested if name not in mental_health]
    available_mental = [name for name in mental_health if name in data and data[name].notna().mean() >= 0.5]
    unavailable_other = [name for name in other if name not in data or data[name].notna().mean() < 0.5]
    if (mental_health and not available_mental) or unavailable_other:
        missing = (["depression_or_psychiatric"] if mental_health and not available_mental else []) + unavailable_other
        return None, missing
    return [*available_mental, *other], []


def cohort_mortality_available(data: pd.DataFrame) -> bool:
    if "mortality_available" not in data:
        return False
    values = data["mortality_available"].dropna().astype(bool).unique()
    if len(values) > 1:
        raise ConfigurationError("mortality availability must be constant within cohort")
    return bool(values[0]) if len(values) else False


def survivor_death_handling(data: pd.DataFrame) -> str:
    """Label the common observed-living, survivor-conditional estimand.

    Mortality availability is retained in its own output field. It must not
    split otherwise comparable survivor-conditional estimates into separate
    meta-analysis groups merely because one cohort lacks a validated endpoint.
    """
    return "observed_living_endpoint_survivor_conditional_model"


def adjustment_subset(names: Iterable[str] | None) -> dict[str, list[str]]:
    selected = list(ADJUSTMENT_SETS) if names is None else list(names)
    unknown = [name for name in selected if name not in ADJUSTMENT_SETS]
    if unknown:
        raise ConfigurationError(f"unknown adjustment sets: {unknown}")
    return {name: ADJUSTMENT_SETS[name] for name in selected}


def cohort_primary_models(
    intervals: pd.DataFrame,
    transitions: pd.DataFrame,
    *,
    continuous_adjustments: Iterable[str] | None = None,
    continuous_standardizations: Iterable[str] = ("fixed", "wave"),
    continuous_time_scales: Iterable[str] = ("raw", "annual"),
    transition_adjustments: Iterable[str] | None = None,
    cmm_names: Iterable[str] = ("cmm4", "cmm5"),
    transition_outcomes: Iterable[str] = ("any_cognitive", "any_functional", "cognitive_only", "functional_only", "joint", "death"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    estimates, failures = [], []
    for cohort, data in intervals.groupby("cohort"):
        for cmm in cmm_names:
            for standardization in continuous_standardizations:
                for time_scale in continuous_time_scales:
                    exposure = f"{cmm}_change_{time_scale}"
                    outcomes = [f"cognitive_worsening_{standardization}_{time_scale}", f"functional_worsening_{standardization}_{time_scale}"]
                    for outcome in outcomes:
                        analysis_name = f"continuous_gee_{standardization}_{'unannualised' if time_scale == 'raw' else 'annualised'}"
                        baseline_domain = "baseline_cognition_z_" + standardization if outcome.startswith("cognitive") else "baseline_function_z_" + standardization
                        needed = [outcome, exposure, f"{cmm}_start", baseline_domain, "id", "age", "sex", "education", "interval_years"]
                        for adjustment_set, covariates in adjustment_subset(continuous_adjustments).items():
                            d = data.dropna(subset=needed).copy()
                            selected_covariates, missing_covariates = resolve_adjustment_covariates(d, covariates)
                            if selected_covariates is None:
                                failures.append(failure_record(cohort=cohort, analysis=analysis_name, adjustment_set=adjustment_set, outcome=outcome, exposure=exposure, reason=f"required covariates unavailable: {','.join(missing_covariates)}", data=d))
                                continue
                            d = d.dropna(subset=selected_covariates)
                            if len(d) < 500 or d[exposure].nunique() < 2:
                                failures.append(failure_record(cohort=cohort, analysis=analysis_name, adjustment_set=adjustment_set, outcome=outcome, exposure=exposure, reason="insufficient observations or exposure variation", data=d))
                                continue
                            rhs = " + ".join([exposure, f"{cmm}_start", baseline_domain, "age", "C(sex)", "education", "interval_years", *selected_covariates])
                            try:
                                _, rows = gee_fit(f"{outcome} ~ {rhs}", d, outcome, Gaussian())
                                rows["cohort"] = cohort
                                rows["analysis"] = analysis_name
                                rows["adjustment_set"] = adjustment_set
                                rows["exposure"] = exposure
                                rows["state_definition"] = "continuous"
                                rows["standardization"] = standardization
                                rows["persistent"] = False
                                rows["covariates_used"] = ",".join(selected_covariates)
                                rows["estimand"] = "observed_interval_change"
                                rows["death_handling"] = "not_applicable_to_observed_interval_change"
                                estimates.append(rows[rows["term"] == exposure])
                            except Exception as exc:
                                failures.append(failure_record(cohort=cohort, analysis=analysis_name, adjustment_set=adjustment_set, outcome=outcome, exposure=exposure, reason=str(exc), data=d))

    origin = transitions[transitions["state_from"] == "unimpaired"].copy()
    state_definition = origin["state_definition"].iloc[0] if not origin.empty and origin["state_definition"].nunique() == 1 else "mixed"
    standardization = origin["standardization"].iloc[0] if not origin.empty and origin["standardization"].nunique() == 1 else "mixed"
    persistent = bool(origin["persistent"].iloc[0]) if not origin.empty and origin["persistent"].nunique() == 1 else False
    for cohort, data in origin.groupby("cohort"):
        mortality_available = cohort_mortality_available(data)
        for cmm in cmm_names:
            exposure = f"{cmm}_start"
            time_column = "confirmation_years" if persistent else "interval_years"
            for outcome in transition_outcomes:
                if persistent and outcome == "death":
                    continue
                needed = [outcome, exposure, "id", "age", "sex", "education", time_column, "baseline_cognition", "baseline_function"]
                for adjustment_set, covariates in adjustment_subset(transition_adjustments).items():
                    if outcome == "death" and not mortality_available:
                        failures.append(failure_record(cohort=cohort, analysis="transition_gee", adjustment_set=adjustment_set, outcome=outcome, exposure=exposure, reason="validated mortality endpoint unavailable", data=data))
                        continue
                    d = data.dropna(subset=needed).copy()
                    if outcome != "death":
                        # Domain-specific binary estimates are survivor-conditional;
                        # death is handled jointly in the multinomial model below.
                        d = d[d["state_to"] != "death"].copy()
                    selected_covariates, missing_covariates = resolve_adjustment_covariates(d, covariates)
                    if selected_covariates is None:
                        failures.append(failure_record(cohort=cohort, analysis="transition_gee", adjustment_set=adjustment_set, outcome=outcome, exposure=exposure, reason=f"required covariates unavailable: {','.join(missing_covariates)}", data=d))
                        continue
                    d = d.dropna(subset=selected_covariates)
                    events = int(d[outcome].sum())
                    non_events = int(len(d) - events)
                    if len(d) < 500 or min(events, non_events) < 20 or d[exposure].nunique() < 2:
                        failures.append(failure_record(cohort=cohort, analysis="transition_gee", adjustment_set=adjustment_set, outcome=outcome, exposure=exposure, reason="insufficient observations, events, or exposure variation", data=d))
                        continue
                    rhs = " + ".join([exposure, "age", "C(sex)", "education", time_column, "baseline_cognition", "baseline_function", *selected_covariates])
                    try:
                        _, rows = gee_fit(
                            f"{outcome} ~ {rhs}", d, outcome, Binomial(),
                            covariance_type="bias_reduced",
                            required_terms=[exposure],
                        )
                        rows["cohort"] = cohort
                        rows["analysis"] = "transition_gee"
                        rows["adjustment_set"] = adjustment_set
                        rows["exposure"] = exposure
                        rows["state_definition"] = state_definition
                        rows["standardization"] = standardization
                        rows["persistent"] = persistent
                        rows["covariates_used"] = ",".join(selected_covariates)
                        rows["estimand"] = "death_transition" if outcome == "death" else "survivor_conditional_domain_transition"
                        rows["death_handling"] = "validated_binary_death_endpoint" if outcome == "death" else survivor_death_handling(data)
                        rows["mortality_available"] = mortality_available
                        estimates.append(rows[rows["term"] == exposure])
                    except Exception as exc:
                        failures.append(failure_record(cohort=cohort, analysis="transition_gee", adjustment_set=adjustment_set, outcome=outcome, exposure=exposure, reason=str(exc), data=d))
    failure_frame = pd.DataFrame(failures)
    if not failure_frame.empty:
        fixed_mask = failure_frame["analysis"].astype(str).str.startswith("continuous_gee_fixed")
        wave_mask = failure_frame["analysis"].astype(str).str.startswith("continuous_gee_wave")
        transition_mask = failure_frame["analysis"].eq("transition_gee")
        failure_frame.loc[fixed_mask, ["state_definition", "standardization", "persistent"]] = ["continuous", "fixed", False]
        failure_frame.loc[wave_mask, ["state_definition", "standardization", "persistent"]] = ["continuous", "wave", False]
        failure_frame.loc[transition_mask, ["state_definition", "standardization", "persistent"]] = [state_definition, standardization, persistent]
    return pd.concat(estimates, ignore_index=True) if estimates else pd.DataFrame(), failure_frame


def stacked_domain_model(
    intervals: pd.DataFrame,
    *,
    adjustment_names: Iterable[str] | None = None,
    standardizations: Iterable[str] = ("fixed", "wave"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    results, failures = [], []
    for cohort, data in intervals.groupby("cohort"):
        # Use raw changes for the formal domain contrast and adjust for exact
        # interval length. This avoids placing the same interval denominator
        # in both the exposure and outcome; annualised estimates are retained
        # as a separately labelled sensitivity analysis above.
        for standardization in standardizations:
            cognitive = f"cognitive_worsening_{standardization}_raw"
            functional = f"functional_worsening_{standardization}_raw"
            baseline_cognition = f"baseline_cognition_z_{standardization}"
            baseline_function = f"baseline_function_z_{standardization}"
            base_columns = [
                "id", "start_wave", "age", "sex", "education", "interval_years", "cmm4_start",
                "cmm4_change_raw", baseline_cognition, baseline_function,
                cognitive, functional, *[x for x in ADJUSTMENT_SETS["medication_burden"]],
            ]
            base = data[base_columns].copy()
            for adjustment_set, covariates in adjustment_subset(adjustment_names).items():
                selected_covariates, missing_covariates = resolve_adjustment_covariates(base, covariates)
                if selected_covariates is None:
                    failure = failure_record(cohort=cohort, analysis="continuous_domain_interaction", adjustment_set=adjustment_set, outcome="stacked_worsening", exposure="cmm4_change_raw", reason=f"required covariates unavailable: {','.join(missing_covariates)}", data=base)
                    failure["standardization"] = standardization
                    failures.append(failure)
                    continue
                paired_needed = [cognitive, functional, "cmm4_change_raw", "cmm4_start", "start_wave", "age", "sex", "education", "interval_years", baseline_cognition, baseline_function, *selected_covariates]
                paired = base.dropna(subset=paired_needed).copy()
                d = pd.concat(
                    [
                        paired.assign(domain="cognitive", worsening=paired[cognitive]),
                        paired.assign(domain="functional", worsening=paired[functional]),
                    ],
                    ignore_index=True,
                )
                if len(d) < 1000 or d["cmm4_change_raw"].nunique() < 2:
                    failure = failure_record(cohort=cohort, analysis="continuous_domain_interaction", adjustment_set=adjustment_set, outcome="stacked_worsening", exposure="cmm4_change_raw", reason="insufficient paired observations or exposure variation", data=d)
                    failure["standardization"] = standardization
                    failures.append(failure)
                    continue
                domain_specific_terms = ["cmm4_change_raw", "cmm4_start", baseline_cognition, baseline_function, "age", "C(sex)", "education", "interval_years", *selected_covariates]
                rhs = f"C(domain, Treatment(reference='cognitive')) * ({' + '.join(domain_specific_terms)}) + C(start_wave)"
                try:
                    fit, rows = gee_fit(f"worsening ~ {rhs}", d, "stacked_worsening", Gaussian())
                    main_term = "cmm4_change_raw"
                    interaction_terms = [term for term in fit.params.index if "cmm4_change_raw" in term and ":" in term]
                    if len(interaction_terms) != 1:
                        raise RuntimeError(f"expected one CMM-by-domain interaction term; found {interaction_terms}")
                    interaction_term = interaction_terms[0]
                    covariance = fit.cov_params()
                    cognitive_estimate = float(fit.params[main_term])
                    difference_estimate = float(fit.params[interaction_term])
                    functional_estimate = cognitive_estimate + difference_estimate
                    functional_variance = float(
                        covariance.loc[main_term, main_term]
                        + covariance.loc[interaction_term, interaction_term]
                        + 2 * covariance.loc[main_term, interaction_term]
                    )
                    if not np.isfinite(functional_variance) or functional_variance <= 0:
                        raise RuntimeError("derived functional slope has invalid variance")
                    derived = pd.DataFrame([
                        {"term": "cmm4_change_raw[cognitive]", "estimate": cognitive_estimate, "std_error": float(fit.bse[main_term]), "p_value": float(fit.pvalues[main_term])},
                        {"term": "cmm4_change_raw[functional]", "estimate": functional_estimate, "std_error": math.sqrt(functional_variance), "p_value": float(2 * stats.norm.sf(abs(functional_estimate / math.sqrt(functional_variance))))},
                        {"term": "cmm4_change_raw[functional-minus-cognitive]", "estimate": difference_estimate, "std_error": float(fit.bse[interaction_term]), "p_value": float(fit.pvalues[interaction_term])},
                    ])
                    derived["outcome"] = "stacked_worsening"
                    derived["n"] = int(len(paired))
                    derived["clusters"] = int(paired["id"].nunique())
                    derived["converged"] = bool(fit.converged)
                    rows = derived
                    rows["cohort"] = cohort
                    rows["analysis"] = "continuous_domain_interaction"
                    rows["adjustment_set"] = adjustment_set
                    rows["exposure"] = "cmm4_change_raw"
                    rows["state_definition"] = "continuous"
                    rows["standardization"] = standardization
                    rows["persistent"] = False
                    rows["covariates_used"] = ",".join(selected_covariates)
                    rows["estimand"] = "paired_observed_interval_domain_contrast"
                    rows["death_handling"] = "not_applicable_to_observed_interval_change"
                    rows["paired_intervals"] = int(len(paired))
                    results.append(rows)
                except Exception as exc:
                    failure = failure_record(cohort=cohort, analysis="continuous_domain_interaction", adjustment_set=adjustment_set, outcome="stacked_worsening", exposure="cmm4_change_raw", reason=str(exc), data=d)
                    failure["standardization"] = standardization
                    failures.append(failure)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame(), pd.DataFrame(failures)


def stacked_binary_domain_model(
    transitions: pd.DataFrame,
    *,
    adjustment_names: Iterable[str] | None = None,
    cmm_names: Iterable[str] = ("cmm4", "cmm5"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Formally contrast CMM associations for any-cognitive vs any-functional."""
    results, failures = [], []
    origin = transitions[(transitions["state_from"] == "unimpaired") & (transitions["state_to"] != "death")].copy()
    for cohort, data in origin.groupby("cohort"):
        mortality_available = cohort_mortality_available(data)
        for cmm in cmm_names:
            exposure = f"{cmm}_start"
            persistent = bool(data["persistent"].iloc[0])
            time_column = "confirmation_years" if persistent else "interval_years"
            base_columns = ["id", "age", "sex", "education", time_column, exposure, "baseline_cognition", "baseline_function", "any_cognitive", "any_functional", *ADJUSTMENT_SETS["medication_burden"]]
            base = data[base_columns].copy()
            for adjustment_set, covariates in adjustment_subset(adjustment_names).items():
                selected_covariates, missing_covariates = resolve_adjustment_covariates(base, covariates)
                if selected_covariates is None:
                    failures.append(failure_record(cohort=cohort, analysis="binary_domain_interaction", adjustment_set=adjustment_set, outcome="any_domain", exposure=exposure, reason=f"required covariates unavailable: {','.join(missing_covariates)}", data=base))
                    continue
                paired_needed = ["any_cognitive", "any_functional", exposure, "id", "age", "sex", "education", time_column, "baseline_cognition", "baseline_function", *selected_covariates]
                paired = base.dropna(subset=paired_needed).copy()
                domain_counts = {
                    domain: (int(paired[outcome].sum()), int(len(paired) - paired[outcome].sum()))
                    for domain, outcome in [("cognitive", "any_cognitive"), ("functional", "any_functional")]
                }
                d = pd.concat(
                    [
                        paired.assign(domain="cognitive", impaired=paired["any_cognitive"]),
                        paired.assign(domain="functional", impaired=paired["any_functional"]),
                    ],
                    ignore_index=True,
                )
                if len(d) < 1000 or any(min(events, non_events) < 20 for events, non_events in domain_counts.values()) or d[exposure].nunique() < 2:
                    failures.append(failure_record(cohort=cohort, analysis="binary_domain_interaction", adjustment_set=adjustment_set, outcome="any_domain", exposure=exposure, reason="insufficient observations, events, or exposure variation", data=d))
                    continue
                domain_specific_terms = [exposure, "baseline_cognition", "baseline_function", "age", "C(sex)", "education", time_column, *selected_covariates]
                rhs = f"C(domain, Treatment(reference='cognitive')) * ({' + '.join(domain_specific_terms)})"
                try:
                    _, rows = gee_fit(
                        f"impaired ~ {rhs}", d, "any_domain", Binomial(),
                        covariance_type="bias_reduced",
                        required_terms=[
                            exposure,
                            f"C(domain, Treatment(reference='cognitive'))[T.functional]:{exposure}",
                        ],
                    )
                    rows["cohort"] = cohort
                    rows["analysis"] = "binary_domain_interaction"
                    rows["adjustment_set"] = adjustment_set
                    rows["exposure"] = exposure
                    rows["state_definition"] = data["state_definition"].iloc[0]
                    rows["standardization"] = data["standardization"].iloc[0]
                    rows["persistent"] = persistent
                    rows["covariates_used"] = ",".join(selected_covariates)
                    rows["estimand"] = "survivor_conditional_paired_domain_contrast"
                    rows["death_handling"] = survivor_death_handling(data)
                    rows["mortality_available"] = mortality_available
                    results.append(rows[rows["term"].str.contains(exposure)])
                except Exception as exc:
                    failures.append(failure_record(cohort=cohort, analysis="binary_domain_interaction", adjustment_set=adjustment_set, outcome="any_domain", exposure=exposure, reason=str(exc), data=d))
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame(), pd.DataFrame(failures)


def multinomial_transition_model(
    transitions: pd.DataFrame,
    *,
    adjustment_names: Iterable[str] | None = None,
    cmm_names: Iterable[str] = ("cmm4", "cmm5"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Secondary joint four/five-state model with participant-clustered covariance."""
    results, failures = [], []
    origin = transitions[(transitions["state_from"] == "unimpaired") & (~transitions["persistent"])].copy()
    code_map = {state: index for index, state in enumerate(STATE_ORDER)}
    for cohort, data in origin.groupby("cohort"):
        mortality_available = cohort_mortality_available(data)
        for cmm in cmm_names:
            exposure = f"{cmm}_start"
            for adjustment_set, covariates in adjustment_subset(adjustment_names).items():
                selected_covariates, missing_covariates = resolve_adjustment_covariates(data, covariates)
                if selected_covariates is None:
                    failures.append(failure_record(cohort=cohort, analysis="transition_multinomial_cluster", adjustment_set=adjustment_set, outcome="state_to", exposure=exposure, reason=f"required covariates unavailable: {','.join(missing_covariates)}", data=data))
                    continue
                needed = ["state_to", exposure, "id", "age", "sex", "education", "interval_years", "baseline_cognition", "baseline_function", *selected_covariates]
                d = data.dropna(subset=needed).copy()
                d["state_code"] = d["state_to"].map(code_map)
                counts = d["state_code"].value_counts()
                required_states = STATE_ORDER if mortality_available else STATE_ORDER[:-1]
                sparse_required_states = {
                    state: int(counts.get(code_map[state], 0))
                    for state in required_states
                    if counts.get(code_map[state], 0) < 20
                }
                if sparse_required_states:
                    model_name = "five-state" if mortality_available else "four-state"
                    failures.append(failure_record(
                        cohort=cohort,
                        analysis="transition_multinomial_cluster",
                        adjustment_set=adjustment_set,
                        outcome="state_to",
                        exposure=exposure,
                        reason=(
                            f"required {model_name} destination categories had fewer than "
                            f"20 transitions: {sparse_required_states}"
                        ),
                        data=d,
                    ))
                    continue
                if len(d) < 500 or d[exposure].nunique() < 2:
                    failures.append(failure_record(cohort=cohort, analysis="transition_multinomial_cluster", adjustment_set=adjustment_set, outcome="state_to", exposure=exposure, reason="insufficient observations or exposure variation", data=d))
                    continue
                rhs = " + ".join([exposure, "baseline_cognition", "baseline_function", "age", "C(sex)", "education", "interval_years", *selected_covariates])
                try:
                    model = smf.mnlogit(f"state_code ~ {rhs}", data=d)
                    fitted = model.fit(method="newton", maxiter=200, disp=False, cov_type="cluster", cov_kwds={"groups": d["id"]})
                    if not bool(fitted.mle_retvals.get("converged", False)):
                        raise RuntimeError("multinomial model did not converge")
                    if (
                        not np.isfinite(np.asarray(fitted.params, float)).all()
                        or not np.isfinite(np.asarray(fitted.bse, float)).all()
                        or (np.asarray(fitted.bse, float) <= 0).any()
                    ):
                        raise RuntimeError("multinomial model returned a non-finite estimate or a non-positive standard error")
                    for column in fitted.params.columns:
                        internal_category = int(column) + 1
                        original_code = int(float(fitted.model._ynames_map[internal_category]))
                        category = STATE_ORDER[original_code]
                        results.append(pd.DataFrame({
                            "term": [exposure],
                            "estimate": [float(fitted.params.loc[exposure, column])],
                            "std_error": [float(fitted.bse.loc[exposure, column])],
                            "p_value": [float(fitted.pvalues.loc[exposure, column])],
                            "outcome": [category],
                            "n": [len(d)],
                            "clusters": [int(d["id"].nunique())],
                            "converged": [True],
                            "cohort": [cohort],
                            "analysis": ["transition_multinomial_cluster"],
                            "adjustment_set": [adjustment_set],
                            "exposure": [exposure],
                            "state_definition": [data["state_definition"].iloc[0]],
                            "standardization": [data["standardization"].iloc[0]],
                            "persistent": [False],
                            "covariates_used": [",".join(selected_covariates)],
                            "estimand": ["joint_competing_state_transition"],
                            "death_handling": [
                                "validated_five_state_competing_transition"
                                if mortality_available
                                else "four_state_transition_mortality_endpoint_unavailable"
                            ],
                            "mortality_available": [mortality_available],
                        }))
                except Exception as exc:
                    failures.append(failure_record(cohort=cohort, analysis="transition_multinomial_cluster", adjustment_set=adjustment_set, outcome="state_to", exposure=exposure, reason=str(exc), data=d))
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame(), pd.DataFrame(failures)


def reml_hk(estimates: np.ndarray, standard_errors: np.ndarray) -> dict[str, float]:
    y = np.asarray(estimates, float)
    v = np.asarray(standard_errors, float) ** 2
    keep = np.isfinite(y) & np.isfinite(v) & (v > 0)
    y, v = y[keep], v[keep]
    k = len(y)
    if k < 2:
        return {"k": k, "pooled": np.nan, "ci_low": np.nan, "ci_high": np.nan, "prediction_low": np.nan, "prediction_high": np.nan, "tau2": np.nan, "i2": np.nan}

    def objective(tau2: float) -> float:
        w = 1 / (v + tau2)
        mean = np.sum(w * y) / np.sum(w)
        return 0.5 * (np.sum(np.log(v + tau2)) + np.log(np.sum(w)) + np.sum(w * (y - mean) ** 2))

    upper = max(float(np.var(y, ddof=1) * 10), 1.0)
    optimised = optimize.minimize_scalar(objective, bounds=(0, upper), method="bounded")
    tau2 = float(optimised.x)
    # Bounded scalar optimisation does not return the exact boundary. Treat a
    # numerically negligible improvement over tau2=0 as the REML boundary.
    if tau2 <= 1e-5 or objective(0.0) - objective(tau2) <= 1e-8:
        tau2 = 0.0
    w = 1 / (v + tau2)
    pooled = float(np.sum(w * y) / np.sum(w))
    conventional_var = 1 / np.sum(w)
    q_residual = float(np.sum(w * (y - pooled) ** 2))
    # Modified Knapp-Hartung avoids spuriously narrower intervals when the
    # residual scale estimate is below one.
    hk_var = conventional_var * max(q_residual / (k - 1), 1.0)
    critical = stats.t.ppf(0.975, df=k - 1)
    ci_low, ci_high = pooled - critical * math.sqrt(hk_var), pooled + critical * math.sqrt(hk_var)
    if k >= 3:
        prediction_se = math.sqrt(tau2 + hk_var)
        prediction_critical = stats.t.ppf(0.975, df=k - 2)
        prediction_low = pooled - prediction_critical * prediction_se
        prediction_high = pooled + prediction_critical * prediction_se
    else:
        prediction_low, prediction_high = np.nan, np.nan
    fixed_weights = 1 / v
    fixed_mean = float(np.sum(fixed_weights * y) / np.sum(fixed_weights))
    q_fixed = float(np.sum(fixed_weights * (y - fixed_mean) ** 2))
    i2 = 100 * max((q_fixed - (k - 1)) / q_fixed, 0.0) if q_fixed > 0 else 0.0
    return {"k": k, "pooled": pooled, "ci_low": ci_low, "ci_high": ci_high, "prediction_low": prediction_low, "prediction_high": prediction_high, "tau2": tau2, "i2": i2}


def pool_models(cohort_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if cohort_results.empty:
        return pd.DataFrame()
    cohort_results = cohort_results.copy()
    if "converged" in cohort_results:
        cohort_results = cohort_results[cohort_results["converged"].fillna(False)].copy()
    if "standardization" not in cohort_results:
        cohort_results["standardization"] = "not_applicable"
    for column in ["estimand", "death_handling"]:
        if column not in cohort_results:
            cohort_results[column] = "not_reported"
        cohort_results[column] = cohort_results[column].fillna("not_reported")
    # Source metric units differ between cohorts and are intentionally not pooled.
    cohort_results = cohort_results[cohort_results["analysis"] != "continuous_source_metric_wave_fe"].copy()
    group_columns = ["analysis", "adjustment_set", "state_definition", "standardization", "persistent", "outcome", "exposure", "term", "estimand", "death_handling"]
    for keys, group in cohort_results.dropna(subset=["exposure"]).groupby(group_columns, dropna=False):
        group = group[
            np.isfinite(pd.to_numeric(group["estimate"], errors="coerce"))
            & np.isfinite(pd.to_numeric(group["std_error"], errors="coerce"))
            & (pd.to_numeric(group["std_error"], errors="coerce") > 0)
        ].copy()
        if group.empty or group["cohort"].astype(str).nunique() < 2:
            continue
        result = reml_hk(group["estimate"].to_numpy(), group["std_error"].to_numpy())
        row = dict(zip(group_columns, keys))
        row.update(result)
        row["cohorts"] = ",".join(group["cohort"].astype(str))
        if "covariates_used" in group:
            row["covariates_by_cohort"] = ";".join(
                f"{cohort}:{covariates}"
                for cohort, covariates in zip(group["cohort"].astype(str), group["covariates_used"].fillna(""))
            )
        row["n_total"] = int(group["n"].sum())
        row["clusters_total"] = int(group["clusters"].sum())
        if "mortality_available" in group:
            row["mortality_availability_by_cohort"] = ";".join(
                f"{cohort}:{bool(available)}"
                for cohort, available in zip(group["cohort"].astype(str), group["mortality_available"].fillna(False))
            )
        if row["analysis"] == "binary_domain_interaction" and ":" in str(row["term"]):
            row["scale"] = "ratio_of_odds_ratios"
        elif row["analysis"] in {"transition_gee", "transition_gee_ipcw", "binary_domain_interaction"}:
            row["scale"] = "odds_ratio"
        elif row["analysis"] == "transition_multinomial_cluster":
            row["scale"] = "relative_risk_ratio"
        else:
            row["scale"] = "beta"
        if row["scale"] in {"odds_ratio", "ratio_of_odds_ratios", "relative_risk_ratio"}:
            for column in ["pooled", "ci_low", "ci_high", "prediction_low", "prediction_high"]:
                row[column] = math.exp(row[column])
        rows.append(row)
    return pd.DataFrame(rows)


def pool_held_constant_cohorts(cohort_results: pd.DataFrame) -> pd.DataFrame:
    if cohort_results.empty:
        return pd.DataFrame()
    comparison_keys = ["analysis", "state_definition", "standardization", "persistent", "outcome", "exposure", "term", "estimand", "death_handling"]
    filtered_parts = []
    for _, group in cohort_results.groupby(comparison_keys):
        cohort_sets = [set(subset["cohort"].astype(str)) for _, subset in group.groupby("adjustment_set")]
        if len(cohort_sets) < 2:
            continue
        common = set.intersection(*cohort_sets)
        if len(common) < 2:
            continue
        kept = group[group["cohort"].astype(str).isin(common)].copy()
        kept["held_constant_cohort_set"] = ",".join(sorted(common))
        filtered_parts.append(kept)
    if not filtered_parts:
        return pd.DataFrame()
    filtered = pd.concat(filtered_parts, ignore_index=True)
    pooled = pool_models(filtered)
    if pooled.empty:
        return pooled
    cohort_map = filtered.drop_duplicates(comparison_keys)[comparison_keys + ["held_constant_cohort_set"]]
    return pooled.merge(cohort_map, on=comparison_keys, how="left")


def prediction_metrics(
    outcome: np.ndarray,
    probability: np.ndarray,
    *,
    include_calibration: bool = True,
) -> dict[str, float]:
    clipped = np.clip(np.asarray(probability, float), 1e-6, 1 - 1e-6)
    observed = np.asarray(outcome, float)
    if np.unique(observed).size < 2:
        raise RuntimeError("calibration requires events and non-events")
    logit_probability = np.log(clipped / (1 - clipped))
    if np.unique(clipped).size < 5 or float(np.std(logit_probability)) <= 1e-8:
        raise RuntimeError("calibration is not estimable from near-constant predictions")
    metrics = {
        "auc": float(roc_auc_score(observed, clipped)),
        "brier": float(brier_score_loss(observed, clipped)),
    }
    if not include_calibration:
        return metrics

    # Fit the two calibration parameters directly. Repeated statsmodels GLM
    # fits inside a participant bootstrap retained large work arrays and could
    # exhaust memory in the largest cohorts. This bounded two-parameter
    # optimisation is numerically equivalent for the point calibration model.
    def logistic_loss(parameters: np.ndarray, slope_fixed: bool = False) -> float:
        if slope_fixed:
            linear = float(parameters[0]) + logit_probability
        else:
            linear = float(parameters[0]) + float(parameters[1]) * logit_probability
        return float(np.sum(np.logaddexp(0.0, linear) - observed * linear))

    def logistic_gradient(parameters: np.ndarray, slope_fixed: bool = False) -> np.ndarray:
        if slope_fixed:
            linear = float(parameters[0]) + logit_probability
            residual = special.expit(linear) - observed
            return np.array([float(residual.sum())])
        linear = float(parameters[0]) + float(parameters[1]) * logit_probability
        residual = special.expit(linear) - observed
        return np.array([
            float(residual.sum()),
            float(np.dot(residual, logit_probability)),
        ])

    calibration = optimize.minimize(
        logistic_loss,
        x0=np.array([0.0, 1.0]),
        jac=logistic_gradient,
        method="L-BFGS-B",
        options={"maxiter": 200, "gtol": 1e-8},
    )
    calibration_in_large = optimize.minimize(
        lambda value: logistic_loss(np.asarray(value), slope_fixed=True),
        x0=np.array([0.0]),
        jac=lambda value: logistic_gradient(np.asarray(value), slope_fixed=True),
        method="L-BFGS-B",
        options={"maxiter": 200, "gtol": 1e-8},
    )
    calibration_values = np.r_[calibration.x, calibration_in_large.x]
    if (
        not calibration.success
        or not calibration_in_large.success
        or not np.isfinite(calibration_values).all()
    ):
        raise RuntimeError(
            "calibration optimisation failed: "
            f"slope_model={calibration.message}; intercept_model={calibration_in_large.message}"
        )
    metrics.update({
        "calibration_intercept": float(calibration_in_large.x[0]),
        "calibration_slope": float(calibration.x[1]),
    })
    return metrics


def discrimination(transitions: pd.DataFrame, bootstrap_iterations: int = 200) -> pd.DataFrame:
    """Participant-grouped cross-validated prediction performance."""
    rows = []
    origin = transitions[(transitions["state_from"] == "unimpaired") & (transitions["state_to"] != "death")].copy()
    for cohort, data in origin.groupby("cohort"):
        mortality_available = cohort_mortality_available(data)
        death_handling = survivor_death_handling(data)
        for outcome in ["any_cognitive", "any_functional"]:
            d = data.dropna(subset=[outcome, "cmm4_start", "age", "sex", "education", "baseline_cognition", "baseline_function", "interval_years", "id"]).copy()
            if len(d) < 500 or d[outcome].sum() < 20 or d[outcome].sum() > len(d) - 20:
                rows.append({
                    "cohort": cohort, "outcome": outcome, "n": len(d),
                    "participants": int(d["id"].nunique()),
                    "events": int(d[outcome].sum()) if len(d) else 0,
                    "status": "not_estimable", "reason": "insufficient observations, events or non-events",
                    "estimand": "survivor_conditional_prediction",
                    "death_handling": death_handling,
                    "mortality_available": mortality_available,
                })
                continue
            d["sex"] = pd.Categorical(d["sex"].astype("string"))
            event_groups = int(d.loc[d[outcome] == 1, "id"].nunique())
            nonevent_groups = int(d.loc[d[outcome] == 0, "id"].nunique())
            n_splits = min(5, int(d["id"].nunique()), event_groups, nonevent_groups)
            if n_splits < 3:
                rows.append({
                    "cohort": cohort,
                    "outcome": outcome,
                    "n": len(d),
                    "participants": int(d["id"].nunique()),
                    "events": int(d[outcome].sum()),
                    "status": "not_estimable",
                    "reason": "fewer than three participant groups with events or non-events",
                    "estimand": "survivor_conditional_prediction",
                    "death_handling": death_handling,
                    "mortality_available": mortality_available,
                })
                continue
            formulas = {
                "reviewer_base": f"{outcome} ~ age + C(sex) + education + baseline_function + interval_years",
                "reviewer_base_plus_cmm4": f"{outcome} ~ age + C(sex) + education + baseline_function + interval_years + cmm4_start",
                "expanded_base": f"{outcome} ~ age + C(sex) + education + baseline_cognition + baseline_function + interval_years",
                "expanded_base_plus_cmm4": f"{outcome} ~ age + C(sex) + education + baseline_cognition + baseline_function + interval_years + cmm4_start",
            }
            predictions = {label: np.full(len(d), np.nan) for label in formulas}
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=20260820)
            folds = list(splitter.split(d, d[outcome], groups=d["id"]))
            if any(
                min(int(d.iloc[index][outcome].sum()), int(len(index) - d.iloc[index][outcome].sum())) < 5
                for train_test in folds
                for index in train_test
            ):
                rows.append({
                    "cohort": cohort,
                    "outcome": outcome,
                    "n": len(d),
                    "participants": int(d["id"].nunique()),
                    "events": int(d[outcome].sum()),
                    "status": "not_estimable",
                    "reason": "a grouped cross-validation fold contained fewer than five events or non-events",
                    "estimand": "survivor_conditional_prediction",
                    "death_handling": death_handling,
                    "mortality_available": mortality_available,
                })
                continue
            try:
                for train_index, test_index in folds:
                    train, test = d.iloc[train_index], d.iloc[test_index]
                    for label, formula in formulas.items():
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always")
                            model = smf.glm(formula, data=train, family=sm.families.Binomial()).fit()
                        if caught:
                            raise RuntimeError("; ".join(sorted({str(item.message) for item in caught})[:3]))
                        if not np.isfinite(np.asarray(model.params, float)).all():
                            raise RuntimeError("cross-validation model returned a non-finite coefficient")
                        predictions[label][test_index] = np.asarray(model.predict(test), float)
            except Exception as exc:
                rows.append({
                    "cohort": cohort, "outcome": outcome, "n": len(d),
                    "participants": int(d["id"].nunique()), "events": int(d[outcome].sum()),
                    "status": "not_estimable", "reason": f"grouped cross-validation failed: {exc}",
                    "estimand": "survivor_conditional_prediction",
                    "death_handling": death_handling,
                    "mortality_available": mortality_available,
                })
                continue
            if any(np.isnan(probability).any() for probability in predictions.values()):
                rows.append({
                    "cohort": cohort, "outcome": outcome, "n": len(d),
                    "participants": int(d["id"].nunique()), "events": int(d[outcome].sum()),
                    "status": "not_estimable", "reason": "out-of-fold predictions were incomplete",
                    "estimand": "survivor_conditional_prediction",
                    "death_handling": death_handling,
                    "mortality_available": mortality_available,
                })
                continue
            try:
                metrics = {label: prediction_metrics(d[outcome].to_numpy(), probability) for label, probability in predictions.items()}
            except Exception as exc:
                rows.append({
                    "cohort": cohort, "outcome": outcome, "n": len(d),
                    "participants": int(d["id"].nunique()), "events": int(d[outcome].sum()),
                    "status": "not_estimable", "reason": f"out-of-fold calibration failed stability checks: {exc}",
                    "estimand": "survivor_conditional_prediction",
                    "death_handling": death_handling,
                    "mortality_available": mortality_available,
                })
                continue
            point: dict[str, float] = {}
            for prefix, base_label, plus_label in [
                ("reviewer", "reviewer_base", "reviewer_base_plus_cmm4"),
                ("expanded", "expanded_base", "expanded_base_plus_cmm4"),
            ]:
                point.update({
                    f"auc_{prefix}_base": metrics[base_label]["auc"],
                    f"auc_{prefix}_plus_cmm4": metrics[plus_label]["auc"],
                    f"delta_auc_{prefix}": metrics[plus_label]["auc"] - metrics[base_label]["auc"],
                    f"brier_{prefix}_base": metrics[base_label]["brier"],
                    f"brier_{prefix}_plus_cmm4": metrics[plus_label]["brier"],
                    f"delta_brier_{prefix}": metrics[plus_label]["brier"] - metrics[base_label]["brier"],
                    f"calibration_intercept_{prefix}_base": metrics[base_label]["calibration_intercept"],
                    f"calibration_slope_{prefix}_base": metrics[base_label]["calibration_slope"],
                    f"calibration_intercept_{prefix}_plus_cmm4": metrics[plus_label]["calibration_intercept"],
                    f"calibration_slope_{prefix}_plus_cmm4": metrics[plus_label]["calibration_slope"],
                })
            # Bootstrap AUC/Brier and their incremental contrasts. Calibration
            # is reported from grouped out-of-fold predictions, but is not
            # refitted hundreds of times solely to create unstable resampling
            # intervals.
            bootstrap_metric_names = [
                key for key in point
                if key.startswith("auc_") or key.startswith("brier_")
                or key.startswith("delta_auc_") or key.startswith("delta_brier_")
            ]
            bootstrap_values: dict[str, list[float]] = {key: [] for key in bootstrap_metric_names}
            if bootstrap_iterations > 0:
                rng = np.random.default_rng(20260820)
                # groupby.indices builds the participant-to-row map in O(n).
                # The prior equality-scan implementation was O(n * groups)
                # and could allocate tens of gigabytes in large cohorts.
                group_rows = {
                    group_id: np.asarray(indices, dtype=int)
                    for group_id, indices in d.reset_index(drop=True).groupby("id", sort=False).indices.items()
                }
                group_ids = np.asarray(list(group_rows), dtype=object)
                observed = d[outcome].to_numpy()
                for _ in range(bootstrap_iterations):
                    sampled_groups = rng.choice(group_ids, size=len(group_ids), replace=True)
                    sample_index = np.concatenate([group_rows[group_id] for group_id in sampled_groups])
                    y_sample = observed[sample_index]
                    if np.unique(y_sample).size < 2:
                        continue
                    try:
                        sampled_metrics = {
                            label: prediction_metrics(
                                y_sample, probability[sample_index], include_calibration=False
                            )
                            for label, probability in predictions.items()
                        }
                    except Exception:
                        continue
                    sampled_point: dict[str, float] = {}
                    for prefix, base_label, plus_label in [
                        ("reviewer", "reviewer_base", "reviewer_base_plus_cmm4"),
                        ("expanded", "expanded_base", "expanded_base_plus_cmm4"),
                    ]:
                        sampled_point.update({
                            f"auc_{prefix}_base": sampled_metrics[base_label]["auc"],
                            f"auc_{prefix}_plus_cmm4": sampled_metrics[plus_label]["auc"],
                            f"delta_auc_{prefix}": sampled_metrics[plus_label]["auc"] - sampled_metrics[base_label]["auc"],
                            f"brier_{prefix}_base": sampled_metrics[base_label]["brier"],
                            f"brier_{prefix}_plus_cmm4": sampled_metrics[plus_label]["brier"],
                            f"delta_brier_{prefix}": sampled_metrics[plus_label]["brier"] - sampled_metrics[base_label]["brier"],
                        })
                    for key, value in sampled_point.items():
                        if key in bootstrap_values:
                            bootstrap_values[key].append(value)
                    del sampled_metrics, sampled_point, sample_index, sampled_groups, y_sample
                    if (_ + 1) % 25 == 0:
                        gc.collect()
            row = {
                "cohort": cohort,
                "outcome": outcome,
                "n": len(d),
                "participants": int(d["id"].nunique()),
                "events": int(d[outcome].sum()),
                "folds": n_splits,
                "status": "estimated",
                "reason": "",
                "estimand": "survivor_conditional_prediction",
                "death_handling": death_handling,
                "mortality_available": mortality_available,
                "bootstrap_iterations_completed": min((len(values) for values in bootstrap_values.values()), default=0),
                **point,
            }
            for key, values in bootstrap_values.items():
                row[f"{key}_ci_low"] = float(np.quantile(values, 0.025)) if values else np.nan
                row[f"{key}_ci_high"] = float(np.quantile(values, 0.975)) if values else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def absolute_risk_contrasts(transitions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    origin = transitions[(transitions["state_from"] == "unimpaired") & (transitions["state_to"] != "death")].copy()
    for cohort, data in origin.groupby("cohort"):
        mortality_available = cohort_mortality_available(data)
        death_handling = survivor_death_handling(data)
        for outcome in ["any_cognitive", "any_functional"]:
            needed = [outcome, "cmm4_start", "age", "sex", "education", "baseline_cognition", "baseline_function", "interval_years", "id"]
            d = data.dropna(subset=needed).copy()
            d["cmm4_group"] = pd.Categorical(
                np.select(
                    [d["cmm4_start"] == 0, d["cmm4_start"] == 1, d["cmm4_start"] == 2, d["cmm4_start"] >= 3],
                    ["0", "1", "2", "3+"],
                    default=None,
                ),
                categories=["0", "1", "2", "3+"],
            )
            d = d.dropna(subset=["cmm4_group"])
            group_outcomes = pd.crosstab(d["cmm4_group"], d[outcome]).reindex(index=["0", "1", "2", "3+"], columns=[0, 1], fill_value=0)
            if len(d) < 500 or (group_outcomes < 5).any().any():
                rows.append({
                    "cohort": cohort, "outcome": outcome, "n": len(d), "events": int(d[outcome].sum()),
                    "status": "not_estimable", "reason": "a CMM4 category contained fewer than five events or non-events",
                    "estimand": "survivor_conditional_standardised_risk",
                    "death_handling": death_handling,
                    "mortality_available": mortality_available,
                })
                continue
            formula = f"{outcome} ~ age + C(sex) + education + baseline_cognition + baseline_function + interval_years + C(cmm4_group, Treatment(reference='0'))"
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    model = smf.glm(formula, data=d, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": d["id"]})
                if not np.isfinite(np.asarray(model.params, float)).all():
                    raise RuntimeError("non-finite coefficient")
                burden_terms = [name for name in model.params.index if "C(cmm4_group" in name]
                burden_errors = pd.to_numeric(model.bse.reindex(burden_terms), errors="coerce").to_numpy()
                if len(burden_terms) != 3 or not np.isfinite(burden_errors).all() or (burden_errors <= 0).any():
                    raise RuntimeError("non-finite or non-positive CMM4 contrast standard error")
            except Exception as exc:
                rows.append({
                    "cohort": cohort, "outcome": outcome, "n": len(d), "events": int(d[outcome].sum()),
                    "status": "not_estimable", "reason": f"risk model failed stability checks: {exc}",
                    "estimand": "survivor_conditional_standardised_risk",
                    "death_handling": death_handling,
                    "mortality_available": mortality_available,
                })
                continue
            probabilities, gradients = {}, {}
            design_info = model.model.data.design_info
            for burden in ["0", "1", "2", "3+"]:
                counterfactual = d.copy()
                counterfactual["cmm4_group"] = pd.Categorical([burden] * len(counterfactual), categories=["0", "1", "2", "3+"])
                probabilities[burden] = float(model.predict(counterfactual).mean())
                design = np.asarray(build_design_matrices([design_info], counterfactual, return_type="dataframe")[0], float)
                predicted = np.asarray(model.predict(counterfactual), float)
                gradients[burden] = np.mean((predicted * (1 - predicted))[:, None] * design, axis=0)
            difference = probabilities["3+"] - probabilities["0"]
            difference_gradient = gradients["3+"] - gradients["0"]
            difference_variance = float(difference_gradient @ np.asarray(model.cov_params()) @ difference_gradient)
            difference_se = float(np.sqrt(difference_variance)) if np.isfinite(difference_variance) and difference_variance > 0 else np.nan
            ci_reason = "" if np.isfinite(difference_se) else "point estimate available; delta-method risk-difference variance was not positive and finite"
            rows.append({
                "cohort": cohort,
                "outcome": outcome,
                "n": len(d),
                "events": int(d[outcome].sum()),
                "status": "estimated" if np.isfinite(difference_se) else "estimated_point_only",
                "reason": ci_reason,
                "estimand": "survivor_conditional_standardised_risk",
                "death_handling": death_handling,
                "mortality_available": mortality_available,
                "risk_cmm4_0": probabilities["0"],
                "risk_cmm4_1": probabilities["1"],
                "risk_cmm4_2": probabilities["2"],
                "risk_cmm4_ge3": probabilities["3+"],
                "risk_difference_ge3_vs_0": difference,
                "risk_difference_ci_low": difference - 1.96 * difference_se,
                "risk_difference_ci_high": difference + 1.96 * difference_se,
                "reciprocal_positive_risk_difference": (1 / difference) if difference > 0 else np.nan,
                "reciprocal_is_descriptive_not_nns": True,
            })
    return pd.DataFrame(rows)


def interval_audit(intervals: pd.DataFrame) -> pd.DataFrame:
    return intervals.groupby("cohort")["interval_years"].agg(
        intervals="size", median="median", q1=lambda x: x.quantile(0.25), q3=lambda x: x.quantile(0.75), minimum="min", maximum="max"
    ).reset_index()


def state_prevalence(states: pd.DataFrame) -> pd.DataFrame:
    usable = states.dropna(subset=["state"]).copy()
    parts = []
    for population, subset in [
        ("five_state_including_death", usable),
        ("living_domain_states", usable[usable["state"] != "death"]),
    ]:
        counts = (
            subset.groupby(["cohort", "mortality_available", "standardization", "state_definition", "state"], dropna=False)
            .size()
            .rename("n")
            .reset_index()
        )
        totals = counts.groupby(["cohort", "mortality_available", "standardization", "state_definition"])["n"].transform("sum")
        counts["percent"] = 100 * counts["n"] / totals
        if population == "five_state_including_death":
            counts["population"] = np.where(
                counts["mortality_available"],
                "five_state_including_validated_death",
                "four_state_mortality_endpoint_unavailable",
            )
        else:
            counts["population"] = np.where(
                counts["mortality_available"],
                "living_domain_states_with_validated_death_endpoint",
                "living_domain_states_mortality_endpoint_unavailable",
            )
        parts.append(counts)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def threshold_audit(states: pd.DataFrame) -> pd.DataFrame:
    # Death endpoints may intentionally lack cognition/function and must not
    # enter the living-domain threshold denominator.
    usable = states[states["state"].notna() & (states["state"] != "death")].copy()
    rows = []
    keys = ["cohort", "mortality_available", "wave", "standardization", "state_definition"]
    for group_keys, group in usable.groupby(keys, dropna=False):
        cog = numeric(group[f"cognition_z_{group_keys[3]}"])
        func = numeric(group[f"function_z_{group_keys[3]}"])
        cog_cut = float(group["cog_cut"].dropna().iloc[0])
        func_cut = float(group["func_cut"].dropna().iloc[0])
        target = numeric(group["target_occupancy"]).dropna()
        target_percent = float(100 * target.iloc[0]) if not target.empty else np.nan
        cognitive_realised = float(100 * cog.le(cog_cut).mean())
        functional_realised = float(100 * func.ge(func_cut).mean())
        rows.append({
            **dict(zip(keys, group_keys)),
            "common_living_observed_n": int(len(group)),
            "target_percent": target_percent,
            "cognitive_cut": cog_cut,
            "cognitive_percent_below_cut": float(100 * cog.lt(cog_cut).mean()),
            "cognitive_percent_equal_cut": float(100 * cog.eq(cog_cut).mean()),
            "cognitive_percent_at_or_below_cut": cognitive_realised,
            "cognitive_empirical_percentile_lower": float(100 * cog.lt(cog_cut).mean()),
            "cognitive_empirical_percentile_upper": cognitive_realised,
            "cognitive_target_minus_realised_pp": target_percent - cognitive_realised if np.isfinite(target_percent) else np.nan,
            "functional_cut": func_cut,
            "functional_percent_above_cut": float(100 * func.gt(func_cut).mean()),
            "functional_percent_equal_cut": float(100 * func.eq(func_cut).mean()),
            "functional_percent_at_or_above_cut": functional_realised,
            "functional_empirical_percentile_lower": float(100 * func.lt(func_cut).mean()),
            "functional_empirical_percentile_upper": float(100 * func.le(func_cut).mean()),
            "functional_target_minus_realised_pp": target_percent - functional_realised if np.isfinite(target_percent) else np.nan,
        })
    return pd.DataFrame(rows)


def state_reversion(transitions: pd.DataFrame) -> pd.DataFrame:
    observed = transitions[~transitions["persistent"]].copy()
    observed["returned_to_unimpaired"] = (observed["state_to"] == "unimpaired").astype(int)
    return (
        observed.groupby(["cohort", "mortality_available", "standardization", "state_definition", "state_from"], dropna=False)
        .agg(
            origin_intervals=("state_to", "size"),
            returned_to_unimpaired_n=("returned_to_unimpaired", "sum"),
            reversion_percent=("returned_to_unimpaired", lambda x: 100 * x.mean()),
        )
        .reset_index()
    )


def persistent_eligibility_audit(states: pd.DataFrame) -> pd.DataFrame:
    """Audit scheduled-wave confirmation rather than next-observed selection."""
    counts: dict[tuple[Any, ...], dict[str, int]] = {}
    keys = ["cohort", "mortality_available", "standardization", "state_definition"]
    for (cohort, mortality_available, standardization, state_definition, participant), group in states.groupby(
        [*keys, "id"], dropna=False, sort=False
    ):
        living = group[group["state"].ne("death")].sort_values("scheduled_index")
        by_index = {int(row["scheduled_index"]): row for _, row in living.iterrows() if pd.notna(row["scheduled_index"])}
        death_indices = {
            int(row["scheduled_index"])
            for _, row in group[group["state"].eq("death")].iterrows()
            if pd.notna(row["scheduled_index"])
        }
        for start_index, start in by_index.items():
            if pd.isna(start["state"]):
                continue
            key = (cohort, bool(mortality_available), standardization, state_definition)
            bucket = counts.setdefault(key, {
                "candidate_start_waves": 0,
                "missing_scheduled_end": 0,
                "missing_end_domain_state": 0,
                "death_before_or_at_end": 0,
                "missing_scheduled_confirmation": 0,
                "missing_confirmation_domain_state": 0,
                "death_before_or_at_confirmation": 0,
                "eligible_confirmed_intervals": 0,
            })
            bucket["candidate_start_waves"] += 1
            end = by_index.get(start_index + 1)
            if end is None:
                if (start_index + 1) in death_indices:
                    bucket["death_before_or_at_end"] += 1
                else:
                    bucket["missing_scheduled_end"] += 1
                continue
            if pd.isna(end["state"]):
                bucket["missing_end_domain_state"] += 1
                continue
            confirmation = by_index.get(start_index + 2)
            if confirmation is None:
                if (start_index + 2) in death_indices:
                    bucket["death_before_or_at_confirmation"] += 1
                else:
                    bucket["missing_scheduled_confirmation"] += 1
                continue
            if pd.isna(confirmation["state"]):
                bucket["missing_confirmation_domain_state"] += 1
                continue
            bucket["eligible_confirmed_intervals"] += 1
    return pd.DataFrame([{**dict(zip(keys, key)), **value} for key, value in counts.items()])


def transition_event_summary(transitions: pd.DataFrame) -> pd.DataFrame:
    origin = transitions[transitions["state_from"] == "unimpaired"].copy()
    rows = []
    for keys, group in origin.groupby(["cohort", "mortality_available", "standardization", "state_definition", "persistent"], dropna=False):
        for outcome in ["any_cognitive", "any_functional", "cognitive_only", "functional_only", "joint", "death"]:
            mortality_available = bool(keys[1])
            persistent = bool(keys[4])
            if (persistent or not mortality_available) and outcome == "death":
                continue
            rows.append({
                "cohort": keys[0],
                "mortality_available": mortality_available,
                "standardization": keys[2],
                "state_definition": keys[3],
                "persistent": persistent,
                "outcome": outcome,
                "intervals": len(group),
                "events": int(group[outcome].sum()),
                "event_percent": 100 * float(group[outcome].mean()),
            })
    return pd.DataFrame(rows)


def distribution_audit(long: pd.DataFrame) -> pd.DataFrame:
    """Describe raw and standardised domain distributions by cohort and wave."""
    rows = []
    for (cohort, wave), group in long.groupby(["cohort", "wave"], dropna=False):
        mortality_available = cohort_mortality_available(group)
        death_endpoint_rows = int(group["death"].eq(1).sum())
        group = group[group["death"] != 1].copy()
        for domain in ["cognition", "function"]:
            for scale in ["source_metric", "z_fixed", "z_wave"]:
                column = f"{domain}_{scale}"
                values = numeric(group[column])
                observed = values.dropna()
                observed_minimum = float(observed.min()) if not observed.empty else np.nan
                observed_maximum = float(observed.max()) if not observed.empty else np.nan
                metric_type = (
                    str(group[f"{domain}_source_metric_type"].dropna().iloc[0])
                    if scale == "source_metric" and group[f"{domain}_source_metric_type"].notna().any()
                    else scale
                )
                rows.append({
                    "cohort": cohort,
                    "mortality_available": mortality_available,
                    "wave": wave,
                    "domain": domain,
                    "scale": scale,
                    "metric_type": metric_type,
                    "rows": len(group),
                    "death_endpoint_rows_excluded": death_endpoint_rows,
                    "observed_n": int(observed.size),
                    "missing_percent": float(100 * values.isna().mean()),
                    "mean": float(observed.mean()) if not observed.empty else np.nan,
                    "sd": float(observed.std(ddof=1)) if observed.size > 1 else np.nan,
                    "skewness": float(stats.skew(observed, bias=False)) if observed.size > 2 else np.nan,
                    "minimum": observed_minimum,
                    "p05": float(observed.quantile(0.05)) if not observed.empty else np.nan,
                    "p25": float(observed.quantile(0.25)) if not observed.empty else np.nan,
                    "median": float(observed.median()) if not observed.empty else np.nan,
                    "p75": float(observed.quantile(0.75)) if not observed.empty else np.nan,
                    "p95": float(observed.quantile(0.95)) if not observed.empty else np.nan,
                    "maximum": observed_maximum,
                    "zero_percent": float(100 * observed.eq(0).mean()) if not observed.empty else np.nan,
                    "observed_floor_percent": float(100 * observed.eq(observed_minimum).mean()) if not observed.empty else np.nan,
                    "observed_ceiling_percent": float(100 * observed.eq(observed_maximum).mean()) if not observed.empty else np.nan,
                })
    return pd.DataFrame(rows)


def origin_state_occupancy(transitions: pd.DataFrame) -> pd.DataFrame:
    """Interval-origin occupancy, distinct from participant-wave prevalence."""
    counts = (
        transitions.groupby(["cohort", "mortality_available", "standardization", "state_definition", "persistent", "state_from"], dropna=False)
        .size()
        .rename("origin_intervals")
        .reset_index()
    )
    if counts.empty:
        return counts
    denominator = counts.groupby(["cohort", "mortality_available", "standardization", "state_definition", "persistent"])["origin_intervals"].transform("sum")
    counts["origin_percent"] = 100 * counts["origin_intervals"] / denominator
    return counts


def registered_analysis_grid(cohorts: Iterable[str]) -> pd.DataFrame:
    """Exact prespecified cohort-by-model cells actually executed by ``main``.

    Earlier versions expanded the Cartesian product of every threshold,
    exposure and adjustment even though most combinations were never intended
    to run. That made thousands of legitimate omissions look unfinished.
    """
    rows: list[dict[str, Any]] = []

    def add(cohort: str, analysis: str, adjustment_set: str, state_definition: str, standardization: str, persistent: bool, outcome: str, exposure: str) -> None:
        rows.append({
            "cohort": cohort,
            "analysis": analysis,
            "adjustment_set": adjustment_set,
            "state_definition": state_definition,
            "standardization": standardization,
            "persistent": persistent,
            "outcome": outcome,
            "exposure": exposure,
            "registered_model_cell": True,
        })

    for cohort in map(str, cohorts):
        # Primary continuous CMM4 fixed-reference raw-change sequence.
        for adjustment_set in ADJUSTMENT_SETS:
            for domain in ["cognitive", "functional"]:
                add(cohort, "continuous_gee_fixed_unannualised", adjustment_set, "continuous", "fixed", False,
                    f"{domain}_worsening_fixed_raw", "cmm4_change_raw")
        # Standardisation, annualisation and complete CMM5 sensitivities.
        for domain in ["cognitive", "functional"]:
            add(cohort, "continuous_gee_wave_unannualised", "mental_health", "continuous", "wave", False,
                f"{domain}_worsening_wave_raw", "cmm4_change_raw")
            add(cohort, "continuous_gee_fixed_annualised", "mental_health", "continuous", "fixed", False,
                f"{domain}_worsening_fixed_annual", "cmm4_change_annual")
            add(cohort, "continuous_gee_fixed_unannualised", "mental_health", "continuous", "fixed", False,
                f"{domain}_worsening_fixed_raw", "cmm5_change_raw")
            add(cohort, "continuous_gee_ipcw", "mental_health", "continuous", "fixed", False,
                f"{domain}_worsening_fixed_raw", "cmm4_change_raw")

        for standardization in ["fixed", "wave"]:
            add(cohort, "continuous_domain_interaction", "mental_health", "continuous", standardization, False,
                "stacked_worsening", "cmm4_change_raw")
        for outcome in ["cognitive_worsening_source_metric", "functional_worsening_source_metric"]:
            add(cohort, "continuous_source_metric_wave_fe", "mental_health", "continuous",
                "source_total_or_fixed_reference_item_z_mean", False, outcome, "cmm4_change_raw")
        for interval_group in ["central_cohort_iqr", "le_1.5y", "gt_1.5_to_2.5y", "gt_2.5y"]:
            for outcome in ["cognitive_worsening_fixed_raw", "functional_worsening_fixed_raw"]:
                add(cohort, f"continuous_interval_{interval_group}", "base", "continuous", "fixed", False, outcome, "cmm4_change_raw")
        for standardization in ["fixed", "wave"]:
            for outcome in [f"cognitive_worsening_{standardization}_raw", f"functional_worsening_{standardization}_raw"]:
                add(cohort, "continuous_sustained_diagnosis", "base", "continuous", standardization, False, outcome, "cmm4_change_sustained_raw")

        # Binary survivor-conditional domain outcomes.
        for state_definition in ["legacy", "strict", "matched_0.15", "matched_0.25"]:
            outcomes = ["any_cognitive", "any_functional"]
            if state_definition == "legacy":
                outcomes += ["cognitive_only", "functional_only", "joint", "death"]
            for outcome in outcomes:
                add(cohort, "transition_gee", "mental_health", state_definition, "fixed", False, outcome, "cmm4_start")
        for outcome in ["any_cognitive", "any_functional"]:
            for adjustment_set in ADJUSTMENT_SETS:
                add(cohort, "transition_gee", adjustment_set, "matched_0.20", "fixed", False, outcome, "cmm4_start")
            add(cohort, "transition_gee", "mental_health", "matched_0.20", "fixed", False, outcome, "cmm5_start")
            add(cohort, "transition_gee", "mental_health", "matched_0.20", "fixed", True, outcome, "cmm4_start")
            add(cohort, "transition_gee", "mental_health", "legacy", "wave", False, outcome, "cmm4_start")
            add(cohort, "transition_gee", "mental_health", "matched_0.20", "wave", False, outcome, "cmm4_start")
            add(cohort, "transition_gee_ipcw", "mental_health", "matched_0.20", "fixed", False, outcome, "cmm4_start")
        for outcome in ["cognitive_only", "functional_only", "joint", "death"]:
            add(cohort, "transition_gee", "mental_health", "matched_0.20", "fixed", False, outcome, "cmm4_start")
        for state_definition in ["legacy", "matched_0.20"]:
            add(cohort, "binary_domain_interaction", "mental_health", state_definition, "fixed", False, "any_domain", "cmm4_start")
        add(cohort, "transition_multinomial_cluster", "mental_health", "matched_0.20", "fixed", False, "state_to", "cmm4_start")
    return pd.DataFrame(rows).drop_duplicates()


def analysis_completion_matrix(results: pd.DataFrame, failures: pd.DataFrame, registry: pd.DataFrame | None = None) -> pd.DataFrame:
    keys = ["cohort", "analysis", "adjustment_set", "state_definition", "standardization", "persistent", "outcome", "exposure"]
    parts = []
    if not results.empty:
        result_cells = results.copy()
        # Multinomial rows are emitted one per non-reference destination state,
        # but the registered model cell is the joint state_to fit.
        result_cells.loc[result_cells["analysis"].eq("transition_multinomial_cluster"), "outcome"] = "state_to"
        success = result_cells.groupby(keys, dropna=False).size().rename("successful_target_terms").reset_index()
        success["failed_or_skipped_models"] = 0
        parts.append(success)
    if not failures.empty:
        failed = failures.groupby(keys, dropna=False).size().rename("failed_or_skipped_models").reset_index()
        failed["successful_target_terms"] = 0
        parts.append(failed)
    observed = (
        pd.concat(parts, ignore_index=True)
        .groupby(keys, dropna=False)[["successful_target_terms", "failed_or_skipped_models"]]
        .sum()
        .reset_index()
        if parts
        else pd.DataFrame(columns=[*keys, "successful_target_terms", "failed_or_skipped_models"])
    )
    if registry is None:
        output = observed
        output["registered_model_cell"] = False
    else:
        expected = registry[keys + ["registered_model_cell"]].drop_duplicates()
        output = expected.merge(observed, on=keys, how="outer")
        output["registered_model_cell"] = output["registered_model_cell"].fillna(False).astype(bool)
    for column in ["successful_target_terms", "failed_or_skipped_models"]:
        output[column] = pd.to_numeric(output[column], errors="coerce").fillna(0).astype(int)
    output["completion_status"] = np.select(
        [output["successful_target_terms"] > 0, output["failed_or_skipped_models"] > 0],
        ["estimated", "failed_or_skipped"],
        default="not_attempted_or_zero_eligible_intervals",
    )
    return output.sort_values(keys).reset_index(drop=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configured_thresholds(config: dict[str, Any]) -> list[tuple[str, float | None]]:
    definitions = require(require(config, "analysis_options", "config"), "threshold_definitions", "analysis_options")
    parsed: list[tuple[str, float | None]] = []
    for item in definitions:
        name = require(item, "name", "threshold definition")
        if name in {"legacy", "strict"}:
            expected = (-0.43, 0.43) if name == "legacy" else (-0.67, 0.67)
            actual = (float(require(item, "cognitive_cut", name)), float(require(item, "functional_cut", name)))
            if actual != expected:
                raise ConfigurationError(f"{name}: configured fixed cut-points do not match the registered values {expected}")
            parsed.append((name, None))
        elif name.startswith("matched_"):
            target = float(require(item, "target_occupancy", name))
            if not 0 < target < 0.5 or name != f"matched_{target:.2f}":
                raise ConfigurationError(f"{name}: invalid or inconsistent target occupancy")
            parsed.append(("matched_prevalence", target))
        else:
            raise ConfigurationError(f"Unsupported registered threshold definition: {name}")
    if ["legacy", "strict", "matched_0.15", "matched_0.20", "matched_0.25"] != [
        (definition if prevalence is None else f"matched_{prevalence:.2f}") for definition, prevalence in parsed
    ]:
        raise ConfigurationError("threshold definitions must preserve the registered order and complete set")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if str(require(config, "schema_version", "config")) != "3.1":
        raise ConfigurationError("filled rerun configuration must use schema_version 3.1")
    args.output.mkdir(parents=True, exist_ok=True)

    long_parts, audits = [], []
    for cohort, specification in require(config, "cohorts", "config").items():
        cohort_long, audit = harmonise_cohort(cohort, specification, args.input_root)
        long_parts.append(cohort_long)
        audits.append(audit)
    long = pd.concat(long_parts, ignore_index=True)
    print(f"[v3] harmonised rows={len(long):,} cohorts={long['cohort'].nunique()}", flush=True)
    intervals = build_intervals(long)
    print(f"[v3] constructed scheduled-wave/death intervals={len(intervals):,}", flush=True)

    threshold_audit_parts: list[pd.DataFrame] = []
    state_prevalence_parts: list[pd.DataFrame] = []
    persistent_audit_parts: list[pd.DataFrame] = []
    state_reversion_parts: list[pd.DataFrame] = []
    transition_summary_parts: list[pd.DataFrame] = []
    origin_occupancy_parts: list[pd.DataFrame] = []
    result_parts: list[pd.DataFrame] = []
    failure_parts: list[pd.DataFrame] = []
    primary_transitions: pd.DataFrame | None = None
    empty_intervals = intervals.iloc[0:0].copy()
    standardizations = require(config["analysis_options"], "standardizations", "analysis_options")
    if standardizations != ["fixed", "wave"]:
        raise ConfigurationError("registered standardizations must be ['fixed', 'wave']")
    for standardization in standardizations:
        for definition, prevalence in configured_thresholds(config):
            stated = apply_states(long, definition, prevalence, standardization=standardization)
            threshold_audit_parts.append(threshold_audit(stated))
            state_prevalence_parts.append(state_prevalence(stated))
            persistent_audit_parts.append(persistent_eligibility_audit(stated))
            state_name = stated["state_definition"].iloc[0]
            primary_binary_definition = require(config["analysis_options"], "primary_binary_definition", "analysis_options")
            primary_continuous_standardization = require(config["analysis_options"], "primary_continuous_standardization", "analysis_options")
            for persistent in [False, True]:
                transition_subset = build_transitions(stated, persistent=persistent)
                if transition_subset.empty:
                    continue
                transition_summary_parts.append(transition_event_summary(transition_subset))
                origin_occupancy_parts.append(origin_state_occupancy(transition_subset))
                if not persistent:
                    state_reversion_parts.append(state_reversion(transition_subset))
                primary_binary_cell = standardization == primary_continuous_standardization and state_name == primary_binary_definition and not persistent
                legacy_cell = standardization == "fixed" and state_name == "legacy" and not persistent
                model_calls: list[dict[str, Any]] = []
                if primary_binary_cell:
                    primary_transitions = transition_subset.copy()
                    # Primary CMM4, inclusive domains and sequential adjustment.
                    model_calls.append(dict(
                        intervals=intervals, cmm_names=["cmm4"],
                        continuous_adjustments=list(ADJUSTMENT_SETS), continuous_standardizations=["fixed"], continuous_time_scales=["raw"],
                        transition_adjustments=list(ADJUSTMENT_SETS), transition_outcomes=["any_cognitive", "any_functional"],
                    ))
                    # Standardisation and annualisation sensitivities use the
                    # prespecified mental-health adjustment only.
                    model_calls.append(dict(
                        intervals=intervals, cmm_names=["cmm4"],
                        continuous_adjustments=["mental_health"], continuous_standardizations=["wave"], continuous_time_scales=["raw"],
                        transition_adjustments=[], transition_outcomes=[],
                    ))
                    model_calls.append(dict(
                        intervals=intervals, cmm_names=["cmm4"],
                        continuous_adjustments=["mental_health"], continuous_standardizations=["fixed"], continuous_time_scales=["annual"],
                        transition_adjustments=[], transition_outcomes=[],
                    ))
                    # True complete CMM5 is a sensitivity, not a second full
                    # sequential-adjustment grid.
                    model_calls.append(dict(
                        intervals=intervals, cmm_names=["cmm5"],
                        continuous_adjustments=["mental_health"], continuous_standardizations=["fixed"], continuous_time_scales=["raw"],
                        transition_adjustments=["mental_health"], transition_outcomes=["any_cognitive", "any_functional"],
                    ))
                    # Four/five-state decomposition is secondary.
                    model_calls.append(dict(
                        intervals=empty_intervals, cmm_names=["cmm4"],
                        continuous_adjustments=[], continuous_standardizations=["fixed"], continuous_time_scales=["raw"],
                        transition_adjustments=["mental_health"], transition_outcomes=["cognitive_only", "functional_only", "joint", "death"],
                    ))
                elif legacy_cell:
                    model_calls.append(dict(
                        intervals=empty_intervals, cmm_names=["cmm4"],
                        continuous_adjustments=[], continuous_standardizations=["fixed"], continuous_time_scales=["raw"],
                        transition_adjustments=["mental_health"], transition_outcomes=["any_cognitive", "any_functional", "cognitive_only", "functional_only", "joint", "death"],
                    ))
                else:
                    registered_sensitivity = (
                        (standardization == "fixed" and not persistent and state_name in {"strict", "matched_0.15", "matched_0.25"})
                        or (standardization == "fixed" and persistent and state_name == "matched_0.20")
                        or (standardization == "wave" and not persistent and state_name in {"legacy", "matched_0.20"})
                    )
                    if registered_sensitivity:
                        model_calls.append(dict(
                            intervals=empty_intervals, cmm_names=["cmm4"],
                            continuous_adjustments=[], continuous_standardizations=["fixed"], continuous_time_scales=["raw"],
                            transition_adjustments=["mental_health"], transition_outcomes=["any_cognitive", "any_functional"],
                        ))
                for call in model_calls:
                    model_intervals = call.pop("intervals")
                    results, model_failures = cohort_primary_models(model_intervals, transition_subset, **call)
                    if not results.empty:
                        result_parts.append(results)
                    if not model_failures.empty:
                        for column, value in [("state_definition", state_name), ("standardization", standardization), ("persistent", persistent)]:
                            if column not in model_failures:
                                model_failures[column] = value
                            else:
                                model_failures.loc[model_failures[column].isna(), column] = value
                        failure_parts.append(model_failures)
                if primary_binary_cell or legacy_cell:
                    binary_results, binary_failures = stacked_binary_domain_model(
                        transition_subset,
                        adjustment_names=["mental_health"],
                        cmm_names=["cmm4"],
                    )
                    if not binary_results.empty:
                        result_parts.append(binary_results)
                    if not binary_failures.empty:
                        binary_failures["state_definition"] = state_name
                        binary_failures["standardization"] = standardization
                        binary_failures["persistent"] = persistent
                        failure_parts.append(binary_failures)
                if primary_binary_cell:
                    multinomial_results, multinomial_failures = multinomial_transition_model(
                        transition_subset,
                        adjustment_names=["mental_health"],
                        cmm_names=["cmm4"],
                    )
                    if not multinomial_results.empty:
                        result_parts.append(multinomial_results)
                    if not multinomial_failures.empty:
                        multinomial_failures["state_definition"] = state_name
                        multinomial_failures["standardization"] = standardization
                        multinomial_failures["persistent"] = False
                        failure_parts.append(multinomial_failures)
            del stated
            gc.collect()
            print(f"[v3] completed states/models standardization={standardization} definition={state_name}", flush=True)
    if primary_transitions is None:
        raise RuntimeError("registered primary transition set was not constructed")
    threshold_audit_output = pd.concat(threshold_audit_parts, ignore_index=True)
    state_prevalence_output = pd.concat(state_prevalence_parts, ignore_index=True)
    persistent_audit_output = pd.concat(persistent_audit_parts, ignore_index=True)
    state_reversion_output = pd.concat(state_reversion_parts, ignore_index=True)
    transition_summary = pd.concat(transition_summary_parts, ignore_index=True)
    origin_occupancy_output = pd.concat(origin_occupancy_parts, ignore_index=True)
    interaction_results, interaction_failures = stacked_domain_model(
        intervals,
        adjustment_names=["mental_health"],
        standardizations=["fixed", "wave"],
    )
    if not interaction_results.empty:
        result_parts.append(interaction_results)
    if not interaction_failures.empty:
        interaction_failures["state_definition"] = "continuous"
        interaction_failures["persistent"] = False
        failure_parts.append(interaction_failures)
    source_metric_results, source_metric_failures = source_metric_wave_fixed_effect_models(
        intervals,
        adjustment_names=["mental_health"],
    )
    if not source_metric_results.empty:
        result_parts.append(source_metric_results)
    if not source_metric_failures.empty:
        source_metric_failures["state_definition"] = "continuous"
        source_metric_failures["standardization"] = "source_total_or_fixed_reference_item_z_mean"
        source_metric_failures["persistent"] = False
        failure_parts.append(source_metric_failures)
    interval_results, interval_failures = interval_length_sensitivity_models(intervals)
    if not interval_results.empty:
        result_parts.append(interval_results)
    if not interval_failures.empty:
        interval_failures["state_definition"] = "continuous"
        interval_failures["standardization"] = "fixed"
        interval_failures["persistent"] = False
        failure_parts.append(interval_failures)
    sustained_results, sustained_failures = sustained_diagnosis_sensitivity_models(intervals)
    if not sustained_results.empty:
        result_parts.append(sustained_results)
    if not sustained_failures.empty:
        sustained_failures["state_definition"] = "continuous"
        sustained_failures["persistent"] = False
        failure_parts.append(sustained_failures)
    response_table = build_scheduled_response_table(long)
    weighted_response, ipcw_audit = estimate_ipcw(response_table)
    ipcw_results, ipcw_failures = ipcw_sensitivity_models(intervals, primary_transitions, weighted_response)
    if not ipcw_results.empty:
        result_parts.append(ipcw_results)
    if not ipcw_failures.empty:
        failure_parts.append(ipcw_failures)
    all_results = pd.concat(result_parts, ignore_index=True) if result_parts else pd.DataFrame()
    all_failures = pd.concat(failure_parts, ignore_index=True) if failure_parts else pd.DataFrame()
    mortality_map = long.groupby("cohort")["mortality_available"].first().astype(bool).to_dict()
    for frame in [all_results, all_failures]:
        if not frame.empty:
            frame["mortality_available"] = frame["cohort"].astype(str).map(mortality_map)
    all_results = ensure_schema(all_results, RESULT_SCHEMA)
    all_failures = ensure_schema(all_failures, FAILURE_SCHEMA)
    pooled = ensure_schema(pool_models(all_results), POOLED_SCHEMA)
    pooled_held_constant = ensure_schema(pool_held_constant_cohorts(all_results), [*POOLED_SCHEMA, "held_constant_cohort_set"])
    # Write a recoverable aggregate checkpoint before prediction bootstrapping.
    all_results.to_csv(args.output / "cohort_model_results_v3.csv", index=False)
    pooled.to_csv(args.output / "pooled_model_results_reml_hk_v3.csv", index=False)
    pooled_held_constant.to_csv(args.output / "pooled_model_results_held_constant_v3.csv", index=False)
    all_failures.to_csv(args.output / "model_failure_matrix_v3.csv", index=False)
    threshold_audit_output.to_csv(args.output / "threshold_realisation_audit_v3.csv", index=False)
    state_prevalence_output.to_csv(args.output / "state_prevalence_v3.csv", index=False)
    state_reversion_output.to_csv(args.output / "state_reversion_v3.csv", index=False)
    persistent_audit_output.to_csv(args.output / "persistent_eligibility_audit_v3.csv", index=False)
    transition_summary.to_csv(args.output / "transition_event_summary_v3.csv", index=False)
    origin_occupancy_output.to_csv(args.output / "interval_origin_state_occupancy_v3.csv", index=False)
    distribution_audit(long).to_csv(args.output / "domain_distribution_audit_v3.csv", index=False)
    interval_audit(intervals).to_csv(args.output / "interval_length_audit_v3.csv", index=False)
    ipcw_audit.to_csv(args.output / "ipcw_response_weight_audit_v3.csv", index=False)
    print("[v3] core aggregate checkpoint written; starting prediction performance", flush=True)
    bootstrap_iterations = int(config.get("analysis_options", {}).get("performance_bootstrap_iterations", 200))
    performance = ensure_schema(discrimination(primary_transitions, bootstrap_iterations=bootstrap_iterations), PERFORMANCE_SCHEMA)
    risk_contrasts = ensure_schema(absolute_risk_contrasts(primary_transitions), RISK_SCHEMA)
    transition_summary = ensure_schema(
        transition_summary,
        ["cohort", "mortality_available", "standardization", "state_definition", "persistent", "outcome", "intervals", "events", "event_percent"],
    )
    completion = analysis_completion_matrix(all_results, all_failures, registered_analysis_grid(config["cohorts"].keys()))

    if bool(config["analysis_options"].get("write_participant_level_outputs", False)):
        long.to_parquet(args.output / "harmonised_long_v3.parquet", index=False)
        intervals.to_parquet(args.output / "adjacent_intervals_v3.parquet", index=False)
        primary_transitions.to_parquet(args.output / "primary_transitions_v3.parquet", index=False)
    threshold_audit_output.to_csv(args.output / "threshold_realisation_audit_v3.csv", index=False)
    state_prevalence_output.to_csv(args.output / "state_prevalence_v3.csv", index=False)
    state_reversion_output.to_csv(args.output / "state_reversion_v3.csv", index=False)
    persistent_audit_output.to_csv(args.output / "persistent_eligibility_audit_v3.csv", index=False)
    transition_summary.to_csv(args.output / "transition_event_summary_v3.csv", index=False)
    origin_occupancy_output.to_csv(args.output / "interval_origin_state_occupancy_v3.csv", index=False)
    distribution_audit(long).to_csv(args.output / "domain_distribution_audit_v3.csv", index=False)
    all_results.to_csv(args.output / "cohort_model_results_v3.csv", index=False)
    pooled.to_csv(args.output / "pooled_model_results_reml_hk_v3.csv", index=False)
    pooled_held_constant.to_csv(args.output / "pooled_model_results_held_constant_v3.csv", index=False)
    all_failures.to_csv(args.output / "model_failure_matrix_v3.csv", index=False)
    completion.to_csv(args.output / "analysis_completion_matrix_v3.csv", index=False)
    performance.to_csv(args.output / "discrimination_v3.csv", index=False)
    risk_contrasts.to_csv(args.output / "absolute_risk_contrasts_v3.csv", index=False)
    interval_audit(intervals).to_csv(args.output / "interval_length_audit_v3.csv", index=False)
    ipcw_audit.to_csv(args.output / "ipcw_response_weight_audit_v3.csv", index=False)
    with (args.output / "harmonisation_audit_v3.json").open("w", encoding="utf-8") as handle:
        json.dump(audits, handle, indent=2, ensure_ascii=False)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_sha256": sha256_file(Path(__file__)),
        "config_sha256": sha256_file(args.config),
        "config": str(args.config.resolve()),
        "participant_level_outputs_written": bool(config["analysis_options"].get("write_participant_level_outputs", False)),
        "inputs": [
            {
                "cohort": cohort,
                "path": str((args.input_root / specification["file"]).resolve()),
                "sha256": sha256_file(args.input_root / specification["file"]),
            }
            for cohort, specification in config["cohorts"].items()
        ],
    }
    (args.output / "analysis_manifest_v3.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(pooled.to_string(index=False))


if __name__ == "__main__":
    main()
