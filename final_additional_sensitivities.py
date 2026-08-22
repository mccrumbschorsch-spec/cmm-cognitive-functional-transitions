from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import revision_pipeline_v3 as v3


def load_reconstructed(config_path: Path, input_root: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    parts = [v3.harmonise_cohort(name, spec, input_root)[0] for name, spec in config["cohorts"].items()]
    long = pd.concat(parts, ignore_index=True)
    long["cmm3_no_stroke"] = long[["hypertension", "diabetes", "heart"]].sum(axis=1, min_count=3)
    long["diabetes_stroke"] = long[["diabetes", "stroke"]].sum(axis=1, min_count=2)
    intervals = v3.build_intervals(long)
    for exposure in ["cmm3_no_stroke", "diabetes_stroke"]:
        start = long[["cohort", "id", "wave", exposure]].rename(columns={"wave": "start_wave", exposure: f"{exposure}_start"})
        end = long[["cohort", "id", "wave", exposure]].rename(columns={"wave": "end_wave", exposure: f"{exposure}_end"})
        intervals = intervals.merge(start, on=["cohort", "id", "start_wave"], how="left")
        intervals = intervals.merge(end, on=["cohort", "id", "end_wave"], how="left")
        intervals[f"{exposure}_change_raw"] = intervals[f"{exposure}_end"] - intervals[f"{exposure}_start"]
    return config, long, intervals


def reference_locked_states(long: pd.DataFrame, config: dict, target: float = 0.20) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = long.copy()
    out["cog_cut"] = np.nan
    out["func_cut"] = np.nan
    cuts = []
    for cohort, data in out.groupby("cohort", sort=False):
        reference = str(config["cohorts"][cohort]["reference_wave"])
        ref = data[
            data["wave"].astype(str).eq(reference)
            & data["death"].ne(1)
            & data[["cognition_z_fixed", "function_z_fixed"]].notna().all(axis=1)
        ]
        cog_cut, _ = v3.closest_observed_tail_cut(ref["cognition_z_fixed"], target, True)
        func_cut, _ = v3.closest_observed_tail_cut(ref["function_z_fixed"], target, False)
        out.loc[out["cohort"].eq(cohort), ["cog_cut", "func_cut"]] = [cog_cut, func_cut]
        for wave, wave_data in data.groupby("wave", sort=False):
            common = wave_data[
                wave_data["death"].ne(1)
                & wave_data[["cognition_z_fixed", "function_z_fixed"]].notna().all(axis=1)
            ]
            cuts.append({
                "cohort": cohort, "wave": wave, "reference_wave": reference,
                "cognitive_cut": cog_cut, "functional_cut": func_cut,
                "common_living_observed_n": len(common),
                "cognitive_occupancy": float(common["cognition_z_fixed"].le(cog_cut).mean()) if len(common) else np.nan,
                "functional_occupancy": float(common["function_z_fixed"].ge(func_cut).mean()) if len(common) else np.nan,
            })
    out["cognitive_impairment"] = out["cognition_z_fixed"].le(out["cog_cut"])
    out["functional_impairment"] = out["function_z_fixed"].ge(out["func_cut"])
    missing = out[["cognition_z_fixed", "function_z_fixed"]].isna().any(axis=1)
    out["state"] = np.select(
        [out["cognitive_impairment"] & out["functional_impairment"], out["cognitive_impairment"], out["functional_impairment"]],
        ["joint", "cognitive_only", "functional_only"], default="unimpaired",
    )
    out.loc[missing, "state"] = np.nan
    out.loc[out["death"].eq(1), "state"] = "death"
    out["state_definition"] = "reference_locked_0.20"
    out["standardization"] = "fixed"
    out["target_occupancy"] = target
    return out, pd.DataFrame(cuts)


def relabel(frame: pd.DataFrame, analysis: str, exposure: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    out["analysis"] = analysis
    out["exposure"] = exposure
    out["term"] = out["term"].astype(str).str.replace("cmm4_change_raw", exposure, regex=False).str.replace("cmm4_start", exposure, regex=False)
    return out


def fit_condition_exposure(intervals: pd.DataFrame, primary_transitions: pd.DataFrame, stem: str) -> pd.DataFrame:
    change, start = f"{stem}_change_raw", f"{stem}_start"
    continuous = intervals.copy()
    continuous["cmm4_change_raw"] = continuous[change]
    continuous["cmm4_start"] = continuous[start]
    separate, _ = v3.cohort_primary_models(
        continuous, primary_transitions.iloc[0:0], continuous_adjustments=["mental_health"],
        continuous_standardizations=["fixed"], continuous_time_scales=["raw"],
        transition_adjustments=[], cmm_names=["cmm4"], transition_outcomes=[],
    )
    interaction, _ = v3.stacked_domain_model(continuous, adjustment_names=["mental_health"], standardizations=["fixed"])
    transitions = primary_transitions.copy()
    mapping = intervals[["cohort", "id", "start_wave", start]].drop_duplicates(["cohort", "id", "start_wave"])
    transitions = transitions.drop(columns=[start], errors="ignore").merge(mapping, on=["cohort", "id", "start_wave"], how="left")
    transitions["cmm4_start"] = transitions[start]
    binary, _ = v3.cohort_primary_models(
        intervals.iloc[0:0], transitions, continuous_adjustments=[], transition_adjustments=["mental_health"],
        cmm_names=["cmm4"], transition_outcomes=["any_cognitive", "any_functional"],
    )
    binary_interaction, _ = v3.stacked_binary_domain_model(transitions, adjustment_names=["mental_health"], cmm_names=["cmm4"])
    return pd.concat([
        relabel(separate, f"{stem}_continuous_gee", change),
        relabel(interaction, f"{stem}_continuous_domain_interaction", change),
        relabel(binary, f"{stem}_transition_gee", start),
        relabel(binary_interaction, f"{stem}_binary_domain_interaction", start),
    ], ignore_index=True)


def fit_reference_locked(transitions: pd.DataFrame) -> pd.DataFrame:
    binary, _ = v3.cohort_primary_models(
        transitions.iloc[0:0], transitions, continuous_adjustments=[], transition_adjustments=["mental_health"],
        cmm_names=["cmm4"], transition_outcomes=["any_cognitive", "any_functional", "cognitive_only", "functional_only", "joint"],
    )
    interaction, _ = v3.stacked_binary_domain_model(transitions, adjustment_names=["mental_health"], cmm_names=["cmm4"])
    return pd.concat([binary, interaction], ignore_index=True)


def interval_extreme_audit(intervals: pd.DataFrame) -> pd.DataFrame:
    """Describe implausibly short/long scheduled intervals before model exclusions."""
    rows = []
    for cohort, data in intervals.groupby("cohort", sort=False):
        years = pd.to_numeric(data["interval_years"], errors="coerce")
        short = years.lt(0.5)
        long = years.gt(5.0)
        extreme = short | long
        rows.append({
            "cohort": cohort,
            "constructed_intervals": int(years.notna().sum()),
            "intervals_lt_0_5_year": int(short.sum()),
            "percent_lt_0_5_year": float(100 * short.mean()),
            "intervals_gt_5_years": int(long.sum()),
            "percent_gt_5_years": float(100 * long.mean()),
            "intervals_within_0_5_to_5_years": int((~extreme & years.notna()).sum()),
            "minimum_years": float(years.min()),
            "maximum_years": float(years.max()),
            "extreme_intervals_ending_in_death": int((extreme & data["death_next"].eq(1)).sum()),
            "extreme_intervals_with_living_endpoint": int((extreme & data["death_next"].ne(1)).sum()),
            "audit_interpretation": "calendar timing/proxy fields require source review; death endpoints are counted separately",
        })
    return pd.DataFrame(rows)


def fit_extreme_interval_restriction(
    intervals: pd.DataFrame,
    primary_transitions: pd.DataFrame,
    lower: float = 0.5,
    upper: float = 5.0,
) -> pd.DataFrame:
    """Repeat the principal models after excluding intervals outside 0.5-5 years."""
    continuous = intervals[intervals["interval_years"].between(lower, upper, inclusive="both")].copy()
    transitions = primary_transitions[
        primary_transitions["interval_years"].between(lower, upper, inclusive="both")
    ].copy()
    separate, _ = v3.cohort_primary_models(
        continuous,
        transitions.iloc[0:0],
        continuous_adjustments=["mental_health"],
        continuous_standardizations=["fixed"],
        continuous_time_scales=["raw"],
        transition_adjustments=[],
        cmm_names=["cmm4"],
        transition_outcomes=[],
    )
    continuous_interaction, _ = v3.stacked_domain_model(
        continuous, adjustment_names=["mental_health"], standardizations=["fixed"]
    )
    binary, _ = v3.cohort_primary_models(
        continuous.iloc[0:0],
        transitions,
        continuous_adjustments=[],
        transition_adjustments=["mental_health"],
        cmm_names=["cmm4"],
        transition_outcomes=["any_cognitive", "any_functional", "death"],
    )
    binary_interaction, _ = v3.stacked_binary_domain_model(
        transitions, adjustment_names=["mental_health"], cmm_names=["cmm4"]
    )
    frames = []
    for frame in [separate, continuous_interaction, binary, binary_interaction]:
        if frame.empty:
            continue
        x = frame.copy()
        x["analysis"] = "exclude_interval_outside_0_5_to_5y_" + x["analysis"].astype(str)
        frames.append(x)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def temporal_boundary(long: pd.DataFrame) -> pd.DataFrame:
    person_rows = []
    for (cohort, pid), group in long[long["death"].ne(1)].groupby(["cohort", "id"], sort=False):
        g = group.sort_values("time_years").drop_duplicates("time_years").reset_index(drop=True)
        g = g.dropna(subset=["cmm4", "cognition_z_fixed", "function_z_fixed", "age", "sex", "education"])
        if len(g) < 4:
            continue
        split = len(g) // 2
        pre_first, pre_last = g.iloc[0], g.iloc[split - 1]
        post_first, post_last = g.iloc[split], g.iloc[-1]
        pre_years = pre_last["time_years"] - pre_first["time_years"]
        post_years = post_last["time_years"] - post_first["time_years"]
        if pre_years <= 0 or post_years <= 0:
            continue
        cognitive = -(post_last["cognition_z_fixed"] - post_first["cognition_z_fixed"]) / post_years
        functional = (post_last["function_z_fixed"] - post_first["function_z_fixed"]) / post_years
        person_rows.append({
            "cohort": cohort, "id": pid, "pre_cmm4_slope": (pre_last["cmm4"] - pre_first["cmm4"]) / pre_years,
            "cognitive_delayed_slope": cognitive, "functional_delayed_slope": functional,
            "mean_delayed_slope": np.mean([cognitive, functional]), "baseline_cmm4": pre_first["cmm4"],
            "baseline_cognition_post": post_first["cognition_z_fixed"], "baseline_function_post": post_first["function_z_fixed"],
            "age": pre_first["age"], "sex": pre_first["sex"], "education": pre_first["education"],
            "pre_years": pre_years, "post_years": post_years,
        })
    people = pd.DataFrame(person_rows)
    results = []
    for outcome in ["cognitive_delayed_slope", "functional_delayed_slope", "mean_delayed_slope"]:
        for cohort, data in people.groupby("cohort"):
            d = data.dropna().copy()
            if len(d) < 500 or d["pre_cmm4_slope"].nunique() < 2:
                continue
            model = smf.ols(
                f"{outcome} ~ pre_cmm4_slope + baseline_cmm4 + baseline_cognition_post + baseline_function_post + age + C(sex) + education + pre_years + post_years",
                data=d,
            ).fit(cov_type="HC3")
            results.append({
                "cohort": cohort, "analysis": "temporal_boundary_landmark", "adjustment_set": "base",
                "state_definition": "continuous", "standardization": "fixed", "persistent": False,
                "outcome": outcome, "exposure": "pre_cmm4_slope", "term": "pre_cmm4_slope",
                "estimate": model.params["pre_cmm4_slope"], "std_error": model.bse["pre_cmm4_slope"],
                "p_value": model.pvalues["pre_cmm4_slope"], "n": int(model.nobs), "clusters": int(model.nobs),
                "converged": True, "covariates_used": "baseline_cmm4,baseline_cognition_post,baseline_function_post,age,sex,education,pre_years,post_years",
                "estimand": "person_level_early_cmm_accumulation_to_later_domain_slope",
                "death_handling": "living_participants_with_four_or_more_observed_waves", "mortality_available": False,
            })
    return pd.DataFrame(results)


def exclusion_repool(main_results: pd.DataFrame, excluded: str, label: str) -> pd.DataFrame:
    selected = main_results[
        ((main_results["analysis"].eq("continuous_domain_interaction")) & main_results["standardization"].eq("fixed") & main_results["adjustment_set"].eq("mental_health") & main_results["exposure"].eq("cmm4_change_raw"))
        | ((main_results["analysis"].isin(["transition_gee", "binary_domain_interaction"])) & main_results["state_definition"].eq("matched_0.20") & main_results["standardization"].eq("fixed") & main_results["adjustment_set"].eq("mental_health") & main_results["exposure"].eq("cmm4_start") & main_results["persistent"].eq(False) & main_results["outcome"].isin(["any_cognitive", "any_functional", "any_domain"]))
    ].copy()
    selected = selected[selected["cohort"].ne(excluded)]
    selected["analysis"] = label + "_" + selected["analysis"].astype(str)
    return selected


def klosa_adl_only_sensitivity(
    long: pd.DataFrame, input_root: Path, config: dict, main_results: pd.DataFrame
) -> pd.DataFrame:
    alternate = long[long["cohort"].eq("klosa")].copy()
    source = pd.read_csv(input_root / config["cohorts"]["klosa"]["file"], low_memory=False)
    source["id"] = source[config["cohorts"]["klosa"]["id_column"]].astype("string")
    source["wave"] = source[config["cohorts"]["klosa"]["wave_column"]].astype(str).str.replace(r"\.0$", "", regex=True)
    source["adl_only"] = pd.to_numeric(source["function_adl_fraction"], errors="coerce")
    reference = source[source["wave"].eq(str(config["cohorts"]["klosa"]["reference_wave"]))]["adl_only"].dropna()
    mean, sd = float(reference.mean()), float(reference.std(ddof=0))
    source["function_z_adl_only"] = (source["adl_only"] - mean) / sd
    alternate = alternate.drop(columns=["function_z_adl_only"], errors="ignore").merge(
        source[["id", "wave", "function_z_adl_only"]].drop_duplicates(["id", "wave"]),
        on=["id", "wave"], how="left",
    )
    living = alternate["death"].ne(1)
    alternate.loc[living, "function_z_fixed"] = alternate.loc[living, "function_z_adl_only"]
    intervals = v3.build_intervals(alternate)
    stated = v3.apply_states(alternate, "matched_prevalence", 0.20, "fixed")
    transitions = v3.build_transitions(stated, persistent=False)
    separate, _ = v3.cohort_primary_models(
        intervals, transitions, continuous_adjustments=["mental_health"],
        continuous_standardizations=["fixed"], continuous_time_scales=["raw"],
        transition_adjustments=["mental_health"], cmm_names=["cmm4"],
        transition_outcomes=["any_cognitive", "any_functional"],
    )
    continuous_interaction, _ = v3.stacked_domain_model(intervals, adjustment_names=["mental_health"], standardizations=["fixed"])
    binary_interaction, _ = v3.stacked_binary_domain_model(transitions, adjustment_names=["mental_health"], cmm_names=["cmm4"])
    alternate_rows = pd.concat([separate, continuous_interaction, binary_interaction], ignore_index=True)
    alternate_rows = alternate_rows[
        alternate_rows["analysis"].isin(["continuous_domain_interaction", "transition_gee", "binary_domain_interaction"])
        & alternate_rows["outcome"].isin(["stacked_worsening", "any_cognitive", "any_functional", "any_domain"])
    ]
    common = exclusion_repool(main_results, "klosa", "temporary")
    common["analysis"] = common["analysis"].str.replace("temporary_", "", regex=False)
    combined = pd.concat([common, alternate_rows], ignore_index=True)
    combined["analysis"] = "klosa_adl_only_" + combined["analysis"].astype(str)
    return combined


def two_component_reliability(config: dict, input_root: Path) -> pd.DataFrame:
    rows = []
    for cohort, spec in config["cohorts"].items():
        data = pd.read_csv(input_root / spec["file"], low_memory=False)
        data["wave_key"] = data[spec["wave_column"]].astype(str).str.replace(r"\.0$", "", regex=True)
        ref = data[data["wave_key"].eq(str(spec["reference_wave"]))]
        pairs = [("function_domains", "function_adl_fraction", "function_iadl_fraction")]
        cognition = spec["cognition"]
        if cognition["mode"] == "components" and len(cognition["inputs"]) == 2:
            pairs.append(("cognition_components", cognition["inputs"][0]["column"], cognition["inputs"][1]["column"]))
        for domain, first, second in pairs:
            complete = ref[[first, second]].apply(pd.to_numeric, errors="coerce").dropna()
            r = float(complete.corr().iloc[0, 1]) if len(complete) > 2 else np.nan
            sb = 2 * r / (1 + r) if np.isfinite(r) and r > -1 else np.nan
            rows.append({"cohort": cohort, "domain": domain, "reference_wave": spec["reference_wave"], "n_complete": len(complete), "pearson_r": r, "spearman_brown": sb, "omega_reported": False})
    return pd.DataFrame(rows)


def sample_reconciliation(config: dict, long: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
    original = {"charls": 21126, "elsa": 15171, "hrs": 31284, "klosa": 8181, "mhas": 10633, "share": 80614}
    reasons = {
        "charls": "source reconstruction and complete included-wave roster",
        "elsa": "minor roster reconstruction difference",
        "hrs": "source reconstruction and complete included-wave roster",
        "klosa": "K-MMSE reconstruction and included waves 3-7",
        "mhas": "restored refreshed samples and corrected scheduled-wave denominator",
        "share": "new headline counts all participants in cognition-compatible waves 4-8; original >=2-wave count used a wider wave set",
    }
    rows = []
    for cohort, data in long.groupby("cohort"):
        living = data[data["death"].ne(1)]
        wave_count = living.groupby("id")["wave"].nunique()
        cohort_intervals = intervals[intervals["cohort"].eq(cohort)]
        rows.append({
            "cohort": cohort, "original_reported_participants_ge2_waves": original[cohort],
            "reconstructed_all_unique_included_waves": int(living["id"].nunique()),
            "reconstructed_participants_ge2_waves": int(wave_count.ge(2).sum()),
            "reconstructed_participants_ge3_waves": int(wave_count.ge(3).sum()),
            "participants_with_constructed_interval": int(cohort_intervals["id"].nunique()),
            "constructed_intervals": len(cohort_intervals),
            "difference_ge2_vs_original": int(wave_count.ge(2).sum() - original[cohort]), "main_reason": reasons[cohort],
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--main-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config, long, intervals = load_reconstructed(args.config, args.input_root)
    primary_stated = v3.apply_states(long, "matched_prevalence", 0.20, "fixed")
    primary_transitions = v3.build_transitions(primary_stated, persistent=False)
    locked_stated, locked_audit = reference_locked_states(long, config)
    locked_transitions = v3.build_transitions(locked_stated, persistent=False)
    results = [fit_reference_locked(locked_transitions)]
    results += [fit_condition_exposure(intervals, primary_transitions, "diabetes_stroke")]
    results += [fit_condition_exposure(intervals, primary_transitions, "cmm3_no_stroke")]
    results += [temporal_boundary(long)]
    results += [fit_extreme_interval_restriction(intervals, primary_transitions)]
    main_results = pd.read_csv(args.main_results, low_memory=False)
    results += [exclusion_repool(main_results, "elsa", "exclude_elsa")]
    results += [exclusion_repool(main_results, "klosa", "exclude_klosa")]
    results += [klosa_adl_only_sensitivity(long, args.input_root, config, main_results)]
    cohort = pd.concat([x for x in results if not x.empty], ignore_index=True)
    pooled = v3.pool_models(cohort)
    # pool_models exponentiates only its registered exact binary analysis
    # names. These additional analyses carry explicit prefixes, so convert
    # their log-odds estimates here and label the interaction correctly.
    additional_binary = pooled["scale"].eq("beta") & pooled["analysis"].str.contains("transition_gee|binary_domain_interaction", regex=True, na=False)
    interaction = additional_binary & pooled["term"].astype(str).str.contains(":")
    for column in ["pooled", "ci_low", "ci_high", "prediction_low", "prediction_high"]:
        pooled.loc[additional_binary, column] = np.exp(pd.to_numeric(pooled.loc[additional_binary, column], errors="coerce"))
    pooled.loc[additional_binary, "scale"] = "odds_ratio"
    pooled.loc[interaction, "scale"] = "ratio_of_odds_ratios"
    cohort.to_csv(args.output / "additional_sensitivity_cohort_results.csv", index=False)
    pooled.to_csv(args.output / "additional_sensitivity_pooled_results.csv", index=False)
    locked_audit.to_csv(args.output / "reference_locked_threshold_audit.csv", index=False)
    two_component_reliability(config, args.input_root).to_csv(args.output / "two_component_reliability.csv", index=False)
    sample_reconciliation(config, long, intervals).to_csv(args.output / "sample_reconciliation.csv", index=False)
    interval_extreme_audit(intervals).to_csv(args.output / "extreme_interval_audit.csv", index=False)
    print(f"additional cohort rows={len(cohort)} pooled rows={len(pooled)}")


if __name__ == "__main__":
    main()
