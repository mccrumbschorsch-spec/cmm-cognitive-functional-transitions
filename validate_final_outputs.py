#!/usr/bin/env python3
"""Compact, fail-fast validation of the publication aggregate outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    root = args.results.resolve()

    pooled = pd.read_csv(root / "pooled_model_results_reml_hk_v3.csv")
    cohorts = pd.read_csv(root / "cohort_model_results_v3.csv")
    thresholds = pd.read_csv(root / "threshold_realisation_audit_v3.csv")
    performance = pd.read_csv(root / "discrimination_v3.csv")
    completion = pd.read_csv(root / "analysis_completion_matrix_v3.csv")
    failures = pd.read_csv(root / "model_failure_matrix_v3.csv")
    intervals = pd.read_csv(root / "interval_length_audit_v3.csv")

    require(len(intervals) == 6 and set(intervals.cohort) == {"charls", "elsa", "hrs", "klosa", "mhas", "share"},
            "interval audit must contain all six cohorts")
    require((intervals["median"] > 0).all(), "interval medians must be positive")

    primary = pooled[
        (pooled.analysis == "transition_gee")
        & (pooled.standardization == "fixed")
        & (pooled.state_definition == "matched_0.20")
        & (pooled.adjustment_set == "mental_health")
        & (pooled.persistent == False)
        & (pooled.exposure == "cmm4_start")
    ]
    require(set(primary.outcome) >= {"any_cognitive", "any_functional", "cognitive_only", "functional_only", "joint", "death"},
            "primary pooled transition outcomes are incomplete")
    require((primary.loc[primary.outcome != "death", "k"] == 6).all(), "living primary outcomes must pool six cohorts")
    require((primary.loc[primary.outcome != "death", "n_total"] == 266127).all(), "primary interval count changed")

    continuous = pooled[
        (pooled.analysis == "continuous_domain_interaction")
        & (pooled.standardization == "fixed")
        & (pooled.adjustment_set == "mental_health")
        & (pooled.exposure == "cmm4_change_raw")
    ]
    require(set(continuous.term) == {"cmm4_change_raw[cognitive]", "cmm4_change_raw[functional]", "cmm4_change_raw[functional-minus-cognitive]"},
            "formal paired-domain continuous results are incomplete")
    require((continuous.k == 6).all() and (continuous.n_total == 361562).all(),
            "formal paired-domain continuous denominators changed")

    numeric = pooled[["pooled", "ci_low", "ci_high"]].to_numpy(float)
    require(np.isfinite(numeric).all(), "pooled estimates contain non-finite values")
    require((pooled.ci_low <= pooled.pooled).all() and (pooled.pooled <= pooled.ci_high).all(),
            "pooled point estimate lies outside its CI")

    require(set(thresholds.state_definition) >= {"legacy", "strict", "matched_0.15", "matched_0.20", "matched_0.25"},
            "threshold audit is incomplete")
    require(set(performance.status) == {"estimated"} and len(performance) == 12,
            "all 12 grouped-CV performance rows must be estimated")
    require((performance.bootstrap_iterations_completed == 200).all(),
            "performance bootstrap count changed")

    require((completion.registered_model_cell == True).all(), "completion matrix contains unregistered cells")
    require(not completion.completion_status.astype(str).str.contains("not_attempted", case=False, na=False).any(),
            "registered cells were not attempted")
    require(len(completion) == 438, "registered model-cell count changed")
    require(len(completion[completion.completion_status == "estimated"]) == 376, "estimated model-cell count changed")
    require(len(failures) == 62, "failure/skip record count changed")
    require(not cohorts.duplicated([c for c in ["cohort", "analysis", "adjustment_set", "outcome", "exposure", "term", "standardization", "state_definition", "persistent"] if c in cohorts]).any(),
            "cohort model results contain duplicate registered keys")

    print("FINAL OUTPUT VALIDATION PASSED")
    print(f"pooled_rows={len(pooled)} cohort_rows={len(cohorts)} completion=376/438 failures={len(failures)}")
    print("primary_transition_intervals=266127 paired_continuous_intervals=361562 performance_rows=12")


if __name__ == "__main__":
    main()
