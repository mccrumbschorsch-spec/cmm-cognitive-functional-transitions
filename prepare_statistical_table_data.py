#!/usr/bin/env python3
"""Prepare typed JSON tables from the publication aggregate outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def clean(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def rows_from_df(df: pd.DataFrame, columns: list[str] | None = None) -> list[list]:
    if columns is None:
        columns = list(df.columns)
    return [[clean(v) for v in row] for row in df[columns].itertuples(index=False, name=None)]


def table(title: str, note: str, columns: list[str], rows: list[list], source: str) -> dict:
    return {"title": title, "note": note, "columns": columns, "rows": rows, "source": source}


def sanitize(value):
    if isinstance(value, dict):
        return {k: sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    return clean(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare aggregate statistical tables as typed JSON.")
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parent,
                        help="Directory containing the main aggregate CSV/JSON files")
    parser.add_argument("--additional", type=Path, default=Path(__file__).resolve().parent,
                        help="Directory containing the additional sensitivity CSV files")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().with_name("statistical_tables.json"))
    args = parser.parse_args()
    results = args.results.resolve()
    additional = args.additional.resolve()

    pooled = pd.read_csv(results / "pooled_model_results_reml_hk_v3.csv")
    cohort = pd.read_csv(results / "cohort_model_results_v3.csv")
    thresholds = pd.read_csv(results / "threshold_realisation_audit_v3.csv")
    distributions = pd.read_csv(results / "domain_distribution_audit_v3.csv")
    prevalence = pd.read_csv(results / "state_prevalence_v3.csv")
    occupancy = pd.read_csv(results / "interval_origin_state_occupancy_v3.csv")
    reversion = pd.read_csv(results / "state_reversion_v3.csv")
    persistent = pd.read_csv(results / "persistent_eligibility_audit_v3.csv")
    intervals = pd.read_csv(results / "interval_length_audit_v3.csv")
    ipcw = pd.read_csv(results / "ipcw_response_weight_audit_v3.csv")
    events = pd.read_csv(results / "transition_event_summary_v3.csv")
    performance = pd.read_csv(results / "discrimination_v3.csv")
    risks = pd.read_csv(results / "absolute_risk_contrasts_v3.csv")
    held = pd.read_csv(results / "pooled_model_results_held_constant_v3.csv")
    failures = pd.read_csv(results / "model_failure_matrix_v3.csv")
    completion = pd.read_csv(results / "analysis_completion_matrix_v3.csv")
    harmon = json.loads((results / "harmonisation_audit_v3.json").read_text())
    klosa_rel = pd.read_csv(results / "klosa_item_reliability_v3.csv")
    additional_pool = pd.read_csv(additional / "additional_sensitivity_pooled_results.csv")
    reference_locked = pd.read_csv(additional / "reference_locked_threshold_audit.csv")
    reference_locked = reference_locked[reference_locked["common_living_observed_n"].gt(0)].copy()
    reliability_two = pd.read_csv(additional / "two_component_reliability.csv")
    reconciliation = pd.read_csv(additional / "sample_reconciliation.csv")
    extreme_intervals = pd.read_csv(additional / "extreme_interval_audit.csv")

    sheets: list[dict] = []
    sheets.append(table(
        "Supplementary Appendix 2 — statistical tables",
        "All values are aggregate outputs from the analysis locked on 21 August 2026. Distribution-defined states are not clinical diagnoses. Participant-level data are not included.",
        ["Item", "Value"],
        [
            ["Analysis lock", "2026-08-21"],
            ["Primary continuous", "Fixed-reference paired-domain participant-clustered GEE"],
            ["Primary binary", "20% cohort-wave target occupancy; any cognitive/any functional"],
            ["Primary CMM", "Row-complete CMM4; both endpoints required for change models and baseline CMM4 required for transition models"],
            ["Pooling", "REML with Hartung-Knapp CI; prediction interval for k>=3"],
            ["Participants appearing in >=1 included wave", 236088],
            ["Participants with >=2 included waves", 172518],
            ["Constructed intervals: consecutive scheduled-wave pairs or terminal-death intervals", 523570],
            ["Registered model cells", 438],
            ["Estimated model cells", 376],
            ["Failed/skipped cells with reasons", 62],
            ["Ethics guidance", "NHC Article 32, 国卫科教发〔2023〕4号"],
            ["Official ethics URL", "https://www.nhc.gov.cn/qjjys/c100016/202302/6b6e447b3edc4338856c9a652a85f44b.shtml"],
            ["Repository", "Data-free public package ready; verified public URL and persistent identifier pending deposit"],
        ],
        "analysis_manifest_v3.json; validate_final_outputs.py",
    ))

    s1_rows = [
        ["CHARLS", "Harmonized CHARLS D + 2020", "1-5", "1", 25873, 70067, "tr20; orient; ser7; draw", "6 ADL + 5 IADL", "difficulty", "Yes", "Yes"],
        ["ELSA", "Harmonized ELSA G.3", "1-9", "1", 19802, 67471, "tr20; orient", "6 ADL + 7 IADL", "difficulty", "Yes", "No—incomplete after 2012"],
        ["HRS", "Harmonized HRS D + RAND HRS 2020 v2", "5-15", "5", 36529, 180628, "validated cog27 total", "6 ADL + 5 IADL", "difficulty", "Yes", "Yes"],
        ["KLoSA", "Harmonized KLoSA E.2; DOI 10.34729/815E-WN35", "3-7", "3", 9216, 29954, "reconstructed 15-item K-MMSE", "7 ADL + 10 IADL", "help dependence", "No", "Yes"],
        ["MHAS", "Harmonized MHAS C.2", "1-5", "1", 26837, 53922, "memory domain; visual scanning", "6 ADL + 4 IADL", "difficulty", "No", "Yes"],
        ["SHARE", "Harmonized SHARE F.2; Release 9.0.0", "4-8", "4", 117831, 121528, "memory; ser7; verbal fluency", "6 ADL + 6 IADL", "difficulty", "Yes", "Yes"],
    ]
    sheets.append(table("S1. Cohort provenance and measurement map", "Non-overlapping cognition/function inputs and exact releases used in the rerun.",
                        ["Cohort", "Release", "Waves", "Reference wave", "Participants", "Constructed intervals", "Cognition", "Function", "Function construct", "CMM5", "Mortality endpoint"], s1_rows,
                        "revision_config_filled_20260821.json; harmonisation_audit_v3.json"))

    cont = pooled[(pooled.analysis == "continuous_domain_interaction") & (pooled.adjustment_set == "mental_health") & (pooled.standardization == "fixed")]
    cont_rows = []
    labels = {"cmm4_change_raw[cognitive]": "Cognitive worsening", "cmm4_change_raw[functional]": "Functional worsening", "cmm4_change_raw[functional-minus-cognitive]": "Functional minus cognitive"}
    for r in cont.itertuples():
        cont_rows.append([labels.get(r.term, r.term), r.k, r.n_total, r.clusters_total, r.pooled, r.ci_low, r.ci_high, r.prediction_low, r.prediction_high, r.i2, r.cohorts])
    sheets.append(table("S2. Primary paired continuous domain analysis", "Unannualised fixed-reference worsening; exact interval adjustment; participant-clustered GEE; beta per additional CMM4 condition accumulated.",
                        ["Outcome", "k", "Intervals", "Participant clusters", "Beta", "CI low", "CI high", "Prediction low", "Prediction high", "I2 (%)", "Cohorts"], cont_rows,
                        "pooled_model_results_reml_hk_v3.csv"))

    trans = pooled[(pooled.analysis == "transition_gee") & (pooled.adjustment_set == "mental_health") & (pooled.standardization == "fixed") & (pooled.state_definition == "matched_0.20") & (pooled.exposure == "cmm4_start") & (pooled.persistent == False)]
    bint = pooled[(pooled.analysis == "binary_domain_interaction") & (pooled.adjustment_set == "mental_health") & (pooled.standardization == "fixed") & (pooled.state_definition == "matched_0.20") & pooled.term.astype(str).str.contains("functional")]
    trans_rows = [["pooled primary result", r.outcome, r.scale, r.k, r.n_total, r.clusters_total, None, None, r.pooled, r.ci_low, r.ci_high, None, r.prediction_low, r.prediction_high, r.i2, r.cohorts] for r in trans.itertuples()]
    for r in bint.itertuples():
        trans_rows.append(["pooled primary result", "functional-to-cognitive domain contrast", r.scale, r.k, int(r.n_total/2), int(r.clusters_total), None, None, r.pooled, r.ci_low, r.ci_high, None, r.prediction_low, r.prediction_high, r.i2, r.cohorts])
    isolated = cohort[
        cohort.analysis.eq("transition_gee")
        & cohort.adjustment_set.eq("mental_health")
        & cohort.state_definition.eq("matched_0.20")
        & cohort.standardization.eq("fixed")
        & cohort.outcome.eq("cognitive_only")
        & cohort.exposure.eq("cmm4_start")
        & cohort.persistent.eq(False)
    ].copy()
    isolated_events = events[
        events.state_definition.eq("matched_0.20")
        & events.standardization.eq("fixed")
        & events.outcome.eq("cognitive_only")
        & events.persistent.eq(False)
    ][["cohort", "intervals", "events"]].drop_duplicates("cohort")
    isolated = isolated.merge(isolated_events, on="cohort", how="left", suffixes=("", "_event_count"))
    for r in isolated.sort_values("cohort").itertuples():
        estimate = float(np.exp(r.estimate))
        ci_low = float(np.exp(r.estimate - 1.96 * r.std_error))
        ci_high = float(np.exp(r.estimate + 1.96 * r.std_error))
        trans_rows.append([
            "cohort-specific isolated cognitive-only", r.cohort.upper(), "odds_ratio", None,
            r.n, r.clusters, r.events, r.intervals, estimate, ci_low, ci_high, r.p_value,
            None, None, None, r.cohort,
        ])
    sheets.append(table("S3. Primary 20% target-occupancy transitions", "Origin restricted to distribution-defined unimpaired; baseline CMM4; adjusted for both baseline domains and participant clustering. Cohort-specific isolated cognitive-only events are counted before model-specific complete-case restriction, while model intervals and clusters are the fitted GEE sample.",
                        ["Record type", "Outcome/cohort", "Scale", "k", "Model intervals", "Participant clusters", "Events before model restriction", "Event-count intervals", "Estimate", "CI low", "CI high", "P value", "Prediction low", "Prediction high", "I2 (%)", "Cohorts"], trans_rows,
                        "pooled_model_results_reml_hk_v3.csv; cohort_model_results_v3.csv; transition_event_summary_v3.csv"))

    threshold_block = thresholds.copy(); threshold_block.insert(0, "record_type", "threshold_realisation")
    distribution_block = distributions.copy(); distribution_block.insert(0, "record_type", "domain_distribution")
    s4_columns = []
    for frame in [threshold_block, distribution_block]:
        for column in frame.columns:
            if column not in s4_columns:
                s4_columns.append(column)
    s4 = pd.concat([threshold_block.reindex(columns=s4_columns), distribution_block.reindex(columns=s4_columns)], ignore_index=True)
    sheets.append(table("S4. Distribution and threshold realisation audit", "Threshold rows report cut-points, empirical percentile bounds, tie mass and realised occupancy for historical, strict and 15%/20%/25% definitions. Distribution rows report skewness, observed floor and ceiling occupancy, and zero occupancy. Tied observed scores were never split.",
                        s4_columns, rows_from_df(s4, s4_columns), "threshold_realisation_audit_v3.csv; domain_distribution_audit_v3.csv"))

    blocks = []
    for kind, df in [("state_prevalence", prevalence), ("interval_origin_occupancy", occupancy), ("one_interval_reversion", reversion), ("persistent_eligibility", persistent)]:
        x = df.copy(); x.insert(0, "record_type", kind); blocks.append(x)
    all_cols = []
    for x in blocks:
        for c in x.columns:
            if c not in all_cols: all_cols.append(c)
    state = pd.concat([x.reindex(columns=all_cols) for x in blocks], ignore_index=True)
    sheets.append(table("S5. State structure, reversion and persistence", "Record_type identifies the denominator and audit. Persistent endpoints require the next scheduled living wave.",
                        all_cols, rows_from_df(state, all_cols), "state_prevalence_v3.csv; interval_origin_state_occupancy_v3.csv; state_reversion_v3.csv; persistent_eligibility_audit_v3.csv"))

    two_lookup = {(r.cohort, r.domain): r for r in reliability_two.itertuples()}
    rel_rows = []
    for entry in harmon:
        for domain in ["cognition", "function"]:
            d = entry[domain]
            key = (entry["cohort"], "function_domains" if domain == "function" else "cognition_components")
            if key in two_lookup:
                r = two_lookup[key]
                rel_rows.append(["two-component/domain reference composite", entry["cohort"], domain, r.reference_wave, r.n_complete, None, None, r.pearson_r, r.spearman_brown, d.get("mode"), "Pearson r and Spearman-Brown; omega not reported"])
            else:
                rel_rows.append(["reference composite", entry["cohort"], domain, d.get("reference_wave"), d.get("reference_n"), d.get("alpha_reference_complete_case"), d.get("omega_reference_complete_case"), None, None, d.get("mode"), d.get("implemented_reliability_metric")])
    for r in klosa_rel.itertuples():
        rel_rows.append(["KLoSA item-level wave", "klosa", "cognition", r.wave, r.complete_case_n, r.alpha_complete_case, r.omega_one_factor_complete_case, None, None, "15 scored K-MMSE items", r.metric_boundary])
    sheets.append(table("S6. Composite reliability", "Alpha/omega are retained only for >=3 scored components. Two-component constructs use Pearson r and Spearman-Brown. HRS cognition is a validated total and internal consistency is not re-estimated.",
                        ["Record type", "Cohort", "Domain", "Wave", "Complete N", "Alpha", "Omega", "Pearson r", "Spearman-Brown", "Mode", "Metric boundary"], rel_rows,
                        "harmonisation_audit_v3.json; klosa_item_reliability_v3.csv; two_component_reliability.csv"))

    interval_block = intervals.copy(); interval_block.insert(0, "record_type", "distribution_summary")
    extreme_block = extreme_intervals.copy(); extreme_block.insert(0, "record_type", "extreme_interval_audit")
    s7_columns = []
    for frame in [interval_block, extreme_block]:
        for column in frame.columns:
            if column not in s7_columns:
                s7_columns.append(column)
    s7 = pd.concat([interval_block.reindex(columns=s7_columns), extreme_block.reindex(columns=s7_columns)], ignore_index=True)
    sheets.append(table("S7. Exact interval-length distributions and extreme-interval audit", "Duration in calendar years from verified interview year/month; scheduled wave adjacency required. Counts below 0.5 year and above 5 years are shown separately, including whether the endpoint was death.",
                        s7_columns, rows_from_df(s7, s7_columns), "interval_length_audit_v3.csv; extreme_interval_audit.csv"))

    ipcw_pool = pooled[pooled.analysis.isin(["transition_gee_ipcw", "continuous_gee_ipcw"])].copy()
    cols = ["Record type", "Cohort/outcome", "Scheduled risk rows", "Deaths excluded", "Response-model rows", "Responded", "Non-response", "Response (%)", "Weight p01", "Weight median", "Weight p99", "k", "Model intervals", "Estimate", "CI low", "CI high", "I2 (%)"]
    ipcw_rows = []
    for r in ipcw.itertuples():
        ipcw_rows.append(["weight audit", r.cohort, r.scheduled_risk_rows, r.competing_deaths_excluded_from_censoring_model, r.non_death_response_model_rows, r.responded_rows, r.non_death_nonresponse_rows, r.response_percent, r.weight_p01, r.weight_median, r.weight_p99, None, None, None, None, None, None])
    for r in ipcw_pool.itertuples():
        ipcw_rows.append(["pooled IPCW result", r.outcome, None, None, None, None, None, None, None, None, None, r.k, r.n_total, r.pooled, r.ci_low, r.ci_high, r.i2])
    sheets.append(table("S8. Non-death IPCW", "Deaths were excluded from the censoring model; stabilised weights were truncated at cohort-specific p01/p99.",
                        cols, ipcw_rows, "ipcw_response_weight_audit_v3.csv; pooled_model_results_reml_hk_v3.csv"))

    death_pool = pooled[((pooled.analysis == "transition_gee") & (pooled.outcome == "death")) | (pooled.analysis == "transition_multinomial_cluster")].copy()
    death_events = events[events.outcome.astype(str).str.lower().eq("death")].copy()
    cols = ["Record type", "Cohort(s)", "Outcome", "State definition", "Analysis", "k", "Intervals", "Events", "Event (%)", "Estimate", "CI low", "CI high", "I2 (%)", "Scale"]
    death_rows = []
    for r in death_pool.itertuples():
        death_rows.append(["pooled result", r.cohorts, r.outcome, r.state_definition, r.analysis, r.k, r.n_total, None, None, r.pooled, r.ci_low, r.ci_high, r.i2, r.scale])
    for r in death_events.itertuples():
        death_rows.append(["cohort death events", r.cohort, r.outcome, r.state_definition, "event count", None, r.intervals, r.events, r.event_percent, None, None, None, None, None])
    sheets.append(table("S9. Mortality and multinomial transitions", "ELSA mortality unavailable. Five-state multinomial pooling includes only converged full-category models.",
                        cols, death_rows, "transition_event_summary_v3.csv; pooled_model_results_reml_hk_v3.csv"))

    sheets.append(table("S10. Held-constant-cohort sequential adjustment", "Cohort composition is fixed within each comparison so attenuation is not caused by changing contributor sets.",
                        list(held.columns), rows_from_df(held), "pooled_model_results_held_constant_v3.csv"))

    perf_cols = [c for c in performance.columns if c not in [c for c in performance.columns if c.startswith("calibration_intercept") and c.endswith(("_ci_low", "_ci_high"))] and c not in [c for c in performance.columns if c.startswith("calibration_slope") and c.endswith(("_ci_low", "_ci_high"))]]
    sheets.append(table("S11. Participant-grouped prediction performance", "Five-fold StratifiedGroupKFold. AUC/Brier metrics and differences use 200 participant bootstrap samples; calibration values are point estimates.",
                        perf_cols, rows_from_df(performance, perf_cols), "discrimination_v3.csv"))

    sheets.append(table("S12. Standardised absolute risk contrasts", "CMM4>=3 versus CMM4=0 among observed living endpoints. Reciprocal risk difference is descriptive and is not a number-needed-to-screen.",
                        list(risks.columns), rows_from_df(risks), "absolute_risk_contrasts_v3.csv"))

    sens = pooled[
        pooled.state_definition.isin(["legacy", "strict", "matched_0.15", "matched_0.25"])
        | (pooled.standardization == "wave")
        | (pooled.exposure == "cmm5_start")
        | pooled.analysis.isin(["continuous_sustained_diagnosis", "continuous_gee_fixed_annualised", "continuous_interval_central_cohort_iqr", "continuous_interval_le_1.5y", "continuous_interval_gt_1.5_to_2.5y", "continuous_interval_gt_2.5y"])
        | (pooled.persistent == True)
    ].copy()
    sheets.append(table("S13. Sensitivity analyses", "Alternative thresholds, wave standardisation, CMM5, sustained diagnosis, persistence and interval specifications. These analyses were defined in the reconstruction plan before the locked rerun, not before the original submission.",
                        list(sens.columns), rows_from_df(sens), "pooled_model_results_reml_hk_v3.csv"))

    sheets.append(table("S14. Model failures and unavailable cells", "Every failed, skipped, unavailable or non-convergent registered model. These rows never enter pooling.",
                        list(failures.columns), rows_from_df(failures), "model_failure_matrix_v3.csv"))

    sheets.append(table("S15. Registered model completion", "Exact configuration-by-cohort registry. No unregistered result and no silently omitted registered cell.",
                        list(completion.columns), rows_from_df(completion), "analysis_completion_matrix_v3.csv"))

    sheets.append(table("S16. Additional analyses", "Fixed reference-wave thresholds, condition-restricted counts, CMM3 excluding stroke, temporal-boundary, cohort exclusions, KLoSA ADL-only analyses and principal models excluding intervals outside 0.5-5 years. Restricted disease counts differ in range from CMM4.",
                        list(additional_pool.columns), rows_from_df(additional_pool), "additional_sensitivity_pooled_results.csv"))

    sheets.append(table("S17. Fixed reference-wave threshold audit", "The reference-wave 20% cut-points were locked and applied unchanged thereafter; later-wave occupancy was not forced to 20%.",
                        list(reference_locked.columns), rows_from_df(reference_locked), "reference_locked_threshold_audit.csv"))

    rec = reconciliation.copy()
    rec.insert(0, "record_type", "sample_reconciliation")
    rel = reliability_two.copy()
    rel.insert(0, "record_type", "two_component_reliability")
    combined_cols = []
    for frame in [rec, rel]:
        for col in frame.columns:
            if col not in combined_cols:
                combined_cols.append(col)
    combined = pd.concat([rec.reindex(columns=combined_cols), rel.reindex(columns=combined_cols)], ignore_index=True)
    sheets.append(table("S18. Sample reconciliation and two-component reliability", "Original and reconstructed participant denominators are separated from measurement reliability. The reconstructed >=2-wave total is 172,518 versus 167,009 originally reported.",
                        combined_cols, rows_from_df(combined, combined_cols), "sample_reconciliation.csv; two_component_reliability.csv"))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(sanitize({"sheets": sheets}), ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"wrote {args.output} with {len(sheets)} sheets")


if __name__ == "__main__":
    main()
