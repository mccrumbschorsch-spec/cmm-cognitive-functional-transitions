#!/usr/bin/env python3
"""Finish the reviewer rerun from the recoverable aggregate checkpoint.

This stage intentionally avoids repeating the hour-long continuous/state
audit. It reconstructs the registered binary transition cells with the
bias-reduced participant-clustered GEE covariance, replaces the corresponding
checkpoint rows, re-pools all cohort estimates, and computes grouped
out-of-fold prediction and absolute-risk summaries. No participant-level data
are written.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path

import pandas as pd


def load_pipeline(path: Path):
    spec = importlib.util.spec_from_file_location("revision_pipeline_v3", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def append_model_call(v3, transitions, *, adjustments, outcomes, cmm_names=("cmm4",)):
    empty = transitions.iloc[0:0].copy()
    return v3.cohort_primary_models(
        empty,
        transitions,
        continuous_adjustments=[],
        continuous_standardizations=["fixed"],
        continuous_time_scales=["raw"],
        transition_adjustments=adjustments,
        transition_outcomes=outcomes,
        cmm_names=cmm_names,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--prediction-only", action="store_true")
    parser.add_argument("--ipcw-only", action="store_true")
    parser.add_argument("--risk-only", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    v3 = load_pipeline(args.pipeline)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)

    long_parts = []
    harmonisation_audits = []
    for cohort, specification in config["cohorts"].items():
        cohort_long, audit = v3.harmonise_cohort(cohort, specification, args.input_root)
        long_parts.append(cohort_long)
        harmonisation_audits.append(audit)
    long = pd.concat(long_parts, ignore_index=True)
    print(f"[finalize] harmonised rows={len(long):,}", flush=True)

    if args.audit_only:
        (args.output / "harmonisation_audit_v3.json").write_text(
            json.dumps(harmonisation_audits, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        all_results = pd.read_csv(args.output / "cohort_model_results_v3.csv")
        all_failures = pd.read_csv(args.output / "model_failure_matrix_v3.csv")
        completion = v3.analysis_completion_matrix(
            all_results, all_failures, v3.registered_analysis_grid(config["cohorts"].keys())
        )
        completion.to_csv(args.output / "analysis_completion_matrix_v3.csv", index=False)
        manifest = {
            "created_utc": v3.datetime.now(v3.timezone.utc).isoformat(),
            "pipeline_sha256": v3.sha256_file(args.pipeline),
            "post_checkpoint_finalizer_sha256": v3.sha256_file(Path(__file__)),
            "config_sha256": v3.sha256_file(args.config),
            "config": str(args.config.resolve()),
            "participant_level_outputs_written": False,
            "inputs": [
                {
                    "cohort": cohort,
                    "path": str((args.input_root / specification["file"]).resolve()),
                    "sha256": v3.sha256_file(args.input_root / specification["file"]),
                }
                for cohort, specification in config["cohorts"].items()
            ],
        }
        (args.output / "analysis_manifest_v3.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print("[finalize] harmonisation audit, completion matrix and final manifest written", flush=True)
        return

    intervals = v3.build_intervals(long)
    mortality_map = long.groupby("cohort")["mortality_available"].first().astype(bool).to_dict()
    print(f"[finalize] constructed scheduled-wave/death intervals={len(intervals):,}", flush=True)

    if args.risk_only:
        stated = v3.apply_states(long, "matched_prevalence", 0.20, standardization="fixed")
        primary_transitions = v3.build_transitions(stated, persistent=False)
        risk = v3.ensure_schema(v3.absolute_risk_contrasts(primary_transitions), v3.RISK_SCHEMA)
        risk.to_csv(args.output / "absolute_risk_contrasts_v3.csv", index=False)
        all_results = pd.read_csv(args.output / "cohort_model_results_v3.csv")
        all_failures = pd.read_csv(args.output / "model_failure_matrix_v3.csv")
        completion = v3.analysis_completion_matrix(
            all_results, all_failures, v3.registered_analysis_grid(config["cohorts"].keys())
        )
        completion.to_csv(args.output / "analysis_completion_matrix_v3.csv", index=False)
        print("[finalize] risk-only output written", flush=True)
        return

    if args.ipcw_only:
        stated = v3.apply_states(long, "matched_prevalence", 0.20, standardization="fixed")
        primary_transitions = v3.build_transitions(stated, persistent=False)
        response_table = v3.build_scheduled_response_table(long)
        weighted_response, ipcw_audit = v3.estimate_ipcw(response_table)
        replacement_results, replacement_failures = v3.ipcw_sensitivity_models(
            intervals, primary_transitions, weighted_response
        )
        mortality_map = long.groupby("cohort")["mortality_available"].first().astype(bool).to_dict()
        for frame in [replacement_results, replacement_failures]:
            if not frame.empty:
                frame["mortality_available"] = frame["cohort"].astype(str).map(mortality_map)
        old_results = pd.read_csv(args.output / "cohort_model_results_v3.csv")
        old_failures = pd.read_csv(args.output / "model_failure_matrix_v3.csv")
        ipcw_analyses = {"continuous_gee_ipcw", "transition_gee_ipcw"}
        all_results = v3.ensure_schema(
            pd.concat([old_results[~old_results["analysis"].isin(ipcw_analyses)], replacement_results], ignore_index=True),
            v3.RESULT_SCHEMA,
        )
        all_failures = v3.ensure_schema(
            pd.concat([old_failures[~old_failures["analysis"].isin(ipcw_analyses)], replacement_failures], ignore_index=True),
            v3.FAILURE_SCHEMA,
        )
        pooled = v3.ensure_schema(v3.pool_models(all_results), v3.POOLED_SCHEMA)
        pooled_held = v3.ensure_schema(
            v3.pool_held_constant_cohorts(all_results),
            [*v3.POOLED_SCHEMA, "held_constant_cohort_set"],
        )
        all_results.to_csv(args.output / "cohort_model_results_v3.csv", index=False)
        all_failures.to_csv(args.output / "model_failure_matrix_v3.csv", index=False)
        pooled.to_csv(args.output / "pooled_model_results_reml_hk_v3.csv", index=False)
        pooled_held.to_csv(args.output / "pooled_model_results_held_constant_v3.csv", index=False)
        ipcw_audit.to_csv(args.output / "ipcw_response_weight_audit_v3.csv", index=False)
        print("[finalize] IPCW outputs replaced", flush=True)
        return

    if args.prediction_only:
        stated = v3.apply_states(long, "matched_prevalence", 0.20, standardization="fixed")
        primary_transitions = v3.build_transitions(stated, persistent=False)
        bootstrap_iterations = int(config.get("analysis_options", {}).get("performance_bootstrap_iterations", 200))
        performance = v3.ensure_schema(
            v3.discrimination(primary_transitions, bootstrap_iterations=bootstrap_iterations),
            v3.PERFORMANCE_SCHEMA,
        )
        risk = v3.ensure_schema(v3.absolute_risk_contrasts(primary_transitions), v3.RISK_SCHEMA)
        performance.to_csv(args.output / "discrimination_v3.csv", index=False)
        risk.to_csv(args.output / "absolute_risk_contrasts_v3.csv", index=False)
        all_results = pd.read_csv(args.output / "cohort_model_results_v3.csv")
        all_failures = pd.read_csv(args.output / "model_failure_matrix_v3.csv")
        completion = v3.analysis_completion_matrix(
            all_results, all_failures, v3.registered_analysis_grid(config["cohorts"].keys())
        )
        completion.to_csv(args.output / "analysis_completion_matrix_v3.csv", index=False)
        manifest = {
            "created_utc": v3.datetime.now(v3.timezone.utc).isoformat(),
            "pipeline_sha256": v3.sha256_file(args.pipeline),
            "post_checkpoint_finalizer_sha256": v3.sha256_file(Path(__file__)),
            "config_sha256": v3.sha256_file(args.config),
            "config": str(args.config.resolve()),
            "participant_level_outputs_written": False,
            "inputs": [
                {
                    "cohort": cohort,
                    "path": str((args.input_root / specification["file"]).resolve()),
                    "sha256": v3.sha256_file(args.input_root / specification["file"]),
                }
                for cohort, specification in config["cohorts"].items()
            ],
        }
        (args.output / "analysis_manifest_v3.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print("[finalize] prediction-only outputs written", flush=True)
        return

    result_parts: list[pd.DataFrame] = []
    failure_parts: list[pd.DataFrame] = []
    primary_transitions = None
    definitions = v3.configured_thresholds(config)
    for standardization in ["fixed", "wave"]:
        for definition, prevalence in definitions:
            stated = v3.apply_states(long, definition, prevalence, standardization=standardization)
            state_name = str(stated["state_definition"].iloc[0])
            # Only the registered wave-standardised cells are rebuilt, and
            # persistent models are registered only for the fixed 20% rule.
            if standardization == "wave" and state_name not in {"legacy", "matched_0.20"}:
                del stated
                gc.collect()
                continue
            persistence_values = (
                [False, True]
                if standardization == "fixed" and state_name == "matched_0.20"
                else [False]
            )
            for persistent in persistence_values:
                transitions = v3.build_transitions(stated, persistent=persistent)
                if transitions.empty:
                    continue
                calls = []
                binary_interaction = False
                if standardization == "fixed" and state_name == "matched_0.20" and not persistent:
                    primary_transitions = transitions.copy()
                    calls.extend([
                        (list(v3.ADJUSTMENT_SETS), ["any_cognitive", "any_functional"], ("cmm4",)),
                        (["mental_health"], ["any_cognitive", "any_functional"], ("cmm5",)),
                        (["mental_health"], ["cognitive_only", "functional_only", "joint", "death"], ("cmm4",)),
                    ])
                    binary_interaction = True
                elif standardization == "fixed" and state_name == "legacy" and not persistent:
                    calls.append((["mental_health"], ["any_cognitive", "any_functional", "cognitive_only", "functional_only", "joint", "death"], ("cmm4",)))
                    binary_interaction = True
                elif (
                    (standardization == "fixed" and not persistent and state_name in {"strict", "matched_0.15", "matched_0.25"})
                    or (standardization == "fixed" and persistent and state_name == "matched_0.20")
                    or (standardization == "wave" and not persistent and state_name in {"legacy", "matched_0.20"})
                ):
                    calls.append((["mental_health"], ["any_cognitive", "any_functional"], ("cmm4",)))
                for adjustments, outcomes, cmm_names in calls:
                    results, failures = append_model_call(
                        v3, transitions, adjustments=adjustments, outcomes=outcomes, cmm_names=cmm_names
                    )
                    if not results.empty:
                        result_parts.append(results)
                    if not failures.empty:
                        failure_parts.append(failures)
                if binary_interaction:
                    results, failures = v3.stacked_binary_domain_model(
                        transitions, adjustment_names=["mental_health"], cmm_names=["cmm4"]
                    )
                    if not results.empty:
                        result_parts.append(results)
                    if not failures.empty:
                        failures["state_definition"] = state_name
                        failures["standardization"] = standardization
                        failures["persistent"] = persistent
                        failure_parts.append(failures)
            del stated
            gc.collect()
            print(f"[finalize] binary cells complete {standardization}/{state_name}", flush=True)

    if primary_transitions is None:
        raise RuntimeError("primary matched_0.20 transitions were not constructed")
    response_table = v3.build_scheduled_response_table(long)
    weighted_response, ipcw_audit = v3.estimate_ipcw(response_table)
    ipcw_results, ipcw_failures = v3.ipcw_sensitivity_models(intervals, primary_transitions, weighted_response)
    if not ipcw_results.empty:
        result_parts.append(ipcw_results[ipcw_results["analysis"].eq("transition_gee_ipcw")].copy())
    if not ipcw_failures.empty:
        failure_parts.append(ipcw_failures[ipcw_failures["analysis"].eq("transition_gee_ipcw")].copy())

    replacement_results = pd.concat(result_parts, ignore_index=True) if result_parts else pd.DataFrame()
    replacement_failures = pd.concat(failure_parts, ignore_index=True) if failure_parts else pd.DataFrame()
    for frame in [replacement_results, replacement_failures]:
        if not frame.empty:
            frame["mortality_available"] = frame["cohort"].astype(str).map(mortality_map)

    old_results = pd.read_csv(args.output / "cohort_model_results_v3.csv")
    old_failures = pd.read_csv(args.output / "model_failure_matrix_v3.csv")
    replace_analyses = {"transition_gee", "transition_gee_ipcw", "binary_domain_interaction"}
    kept_results = old_results[~old_results["analysis"].isin(replace_analyses)].copy()
    kept_failures = old_failures[~old_failures["analysis"].isin(replace_analyses)].copy()
    all_results = v3.ensure_schema(pd.concat([kept_results, replacement_results], ignore_index=True), v3.RESULT_SCHEMA)
    all_failures = v3.ensure_schema(pd.concat([kept_failures, replacement_failures], ignore_index=True), v3.FAILURE_SCHEMA)
    pooled = v3.ensure_schema(v3.pool_models(all_results), v3.POOLED_SCHEMA)
    pooled_held = v3.ensure_schema(
        v3.pool_held_constant_cohorts(all_results),
        [*v3.POOLED_SCHEMA, "held_constant_cohort_set"],
    )
    all_results.to_csv(args.output / "cohort_model_results_v3.csv", index=False)
    all_failures.to_csv(args.output / "model_failure_matrix_v3.csv", index=False)
    pooled.to_csv(args.output / "pooled_model_results_reml_hk_v3.csv", index=False)
    pooled_held.to_csv(args.output / "pooled_model_results_held_constant_v3.csv", index=False)
    ipcw_audit.to_csv(args.output / "ipcw_response_weight_audit_v3.csv", index=False)
    print("[finalize] bias-reduced binary GEE replacement and pooling written", flush=True)

    bootstrap_iterations = int(config.get("analysis_options", {}).get("performance_bootstrap_iterations", 200))
    performance = v3.ensure_schema(
        v3.discrimination(primary_transitions, bootstrap_iterations=bootstrap_iterations),
        v3.PERFORMANCE_SCHEMA,
    )
    risk = v3.ensure_schema(v3.absolute_risk_contrasts(primary_transitions), v3.RISK_SCHEMA)
    performance.to_csv(args.output / "discrimination_v3.csv", index=False)
    risk.to_csv(args.output / "absolute_risk_contrasts_v3.csv", index=False)
    print("[finalize] prediction and absolute-risk outputs written", flush=True)

    completion = v3.analysis_completion_matrix(
        all_results, all_failures, v3.registered_analysis_grid(config["cohorts"].keys())
    )
    completion.to_csv(args.output / "analysis_completion_matrix_v3.csv", index=False)
    manifest = {
        "created_utc": v3.datetime.now(v3.timezone.utc).isoformat(),
        "pipeline_sha256": v3.sha256_file(args.pipeline),
        "post_checkpoint_finalizer_sha256": v3.sha256_file(Path(__file__)),
        "config_sha256": v3.sha256_file(args.config),
        "config": str(args.config.resolve()),
        "participant_level_outputs_written": False,
        "inputs": [
            {
                "cohort": cohort,
                "path": str((args.input_root / specification["file"]).resolve()),
                "sha256": v3.sha256_file(args.input_root / specification["file"]),
            }
            for cohort, specification in config["cohorts"].items()
        ],
    }
    (args.output / "analysis_manifest_v3.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
