from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


PIPELINE = Path(__file__).resolve().with_name("revision_pipeline_v3.py")
SPEC = importlib.util.spec_from_file_location("revision_pipeline_v3", PIPELINE)
assert SPEC is not None and SPEC.loader is not None
v3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v3)


def test_survivor_estimand_is_not_split_by_mortality_availability() -> None:
    available = pd.DataFrame({"mortality_available": [True, True]})
    unavailable = pd.DataFrame({"mortality_available": [False, False]})
    assert v3.survivor_death_handling(available) == v3.survivor_death_handling(unavailable)


def test_prediction_metrics_are_finite_without_statsmodels_calibration_loop() -> None:
    rng = np.random.default_rng(20260821)
    observed = rng.binomial(1, 0.25, 3000)
    probability = np.clip(0.08 + 0.45 * rng.random(3000), 1e-6, 1 - 1e-6)
    metrics = v3.prediction_metrics(observed, probability)
    assert set(metrics) == {"auc", "brier", "calibration_intercept", "calibration_slope"}
    assert np.isfinite(list(metrics.values())).all()


def test_registered_grid_contains_only_executed_plan() -> None:
    grid = v3.registered_analysis_grid(["a"])
    assert len(grid) < 100
    assert not (
        (grid["analysis"] == "binary_domain_interaction")
        & (grid["adjustment_set"] == "base")
    ).any()
    assert (
        (grid["analysis"] == "transition_gee")
        & (grid["state_definition"] == "matched_0.20")
        & (grid["standardization"] == "fixed")
    ).any()


def test_ipcw_transition_is_pooled_on_odds_ratio_scale() -> None:
    rows = []
    for cohort, estimate in [("a", 0.20), ("b", 0.30)]:
        rows.append({
            "cohort": cohort,
            "analysis": "transition_gee_ipcw",
            "adjustment_set": "mental_health",
            "state_definition": "matched_0.20",
            "standardization": "fixed",
            "persistent": False,
            "outcome": "any_functional",
            "exposure": "cmm4_start",
            "term": "cmm4_start",
            "estimate": estimate,
            "std_error": 0.05,
            "n": 1000,
            "clusters": 500,
            "converged": True,
            "estimand": "ipcw_survivor_conditional_domain_transition",
            "death_handling": "competing_death_excluded_from_censoring_model",
            "mortality_available": True,
            "covariates_used": "depression",
        })
    pooled = v3.pool_models(pd.DataFrame(rows))
    assert pooled.iloc[0]["scale"] == "odds_ratio"
    assert pooled.iloc[0]["pooled"] > 1


def test_multinomial_completion_maps_destinations_to_registered_state_fit() -> None:
    results = pd.DataFrame([
        {
            "cohort": "a", "analysis": "transition_multinomial_cluster",
            "adjustment_set": "mental_health", "state_definition": "matched_0.20",
            "standardization": "fixed", "persistent": False,
            "outcome": "functional_only", "exposure": "cmm4_start",
        }
    ])
    registry = pd.DataFrame([
        {
            "cohort": "a", "analysis": "transition_multinomial_cluster",
            "adjustment_set": "mental_health", "state_definition": "matched_0.20",
            "standardization": "fixed", "persistent": False,
            "outcome": "state_to", "exposure": "cmm4_start",
            "registered_model_cell": True,
        }
    ])
    completion = v3.analysis_completion_matrix(results, pd.DataFrame(), registry)
    assert len(completion) == 1
    assert completion.iloc[0]["completion_status"] == "estimated"
