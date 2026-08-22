#!/usr/bin/env python3
"""Create audited, analysis-ready cohort files from the local source archive.

The script is intentionally project-specific.  It replaces the archived
candidate-column selection with explicit, non-overlapping measurement maps,
repairs two verified upstream construction defects, and writes only to a
caller-supplied working directory.  Source files are read-only and no
participant-level output belongs in the public code repository.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


COHORTS = ("charls", "elsa", "hrs", "klosa", "mhas", "share")


def base_headers(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    renamed = {column: column.split(" (", 1)[0].strip() for column in frame.columns}
    if len(set(renamed.values())) != len(renamed):
        raise ValueError(f"Duplicate base headers after label removal: {path}")
    return frame.rename(columns=renamed)


def numeric(frame: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(frame[name], errors="coerce") if name in frame else pd.Series(np.nan, index=frame.index)


def item_fraction(frame: pd.DataFrame, columns: list[str], minimum: int) -> pd.Series:
    matrix = frame[columns].apply(pd.to_numeric, errors="coerce")
    invalid = ~matrix.isna() & ~matrix.isin([0, 1])
    if invalid.any().any():
        examples = {column: sorted(matrix.loc[invalid[column], column].unique())[:8] for column in columns if invalid[column].any()}
        raise ValueError(f"Non-binary ADL/IADL values: {examples}")
    score = matrix.mean(axis=1, skipna=True)
    return score.mask(matrix.notna().sum(axis=1) < minimum)


def bounded(series: pd.Series, minimum: float, maximum: float) -> tuple[pd.Series, int]:
    value = pd.to_numeric(series, errors="coerce")
    invalid = value.notna() & ~value.between(minimum, maximum)
    return value.mask(invalid), int(invalid.sum())


def patch_charls_stroke(frame: pd.DataFrame, source_root: Path) -> int:
    raw_path = source_root / "1. CHARLS  中国/CHARLS_中国/Raw_data/2018charls/Health_Status_and_Functioning.dta"
    raw = pd.read_stata(raw_path, columns=["ID", "da007_8_"], convert_categoricals=False)
    raw["_id_key"] = pd.to_numeric(raw["ID"], errors="coerce").astype("Int64").astype("string")
    lookup = raw.drop_duplicates("_id_key").set_index("_id_key")["da007_8_"]
    target = frame["wave"].eq(4) & frame["stroke"].isna()
    keys = pd.to_numeric(frame.loc[target, "ID"], errors="coerce").astype("Int64").astype("string")
    raw_codes = keys.map(lookup)
    replacement = raw_codes.map({1.0: 1.0, 2.0: 0.0})
    frame.loc[target, "stroke"] = replacement.to_numpy()
    return int(replacement.notna().sum())


def klosa_first_imputation(source_root: Path, wave: int) -> pd.DataFrame:
    raw_root = source_root / f"4. KLoSA 韩国/KLoSA_韩国/Raw_data/wave{wave}"
    paths = [raw_root / f"e_imp_w0{wave}.dta"]
    if wave == 5:
        paths.append(raw_root / "e_imp_w05_new.dta")
    parts = []
    extractor = Path(__file__).with_name("extract_klosa_cognition.R")
    with tempfile.TemporaryDirectory(prefix="klosa_cognition_") as temporary:
        for index, path in enumerate(paths):
            extracted = Path(temporary) / f"wave{wave}_{index}.csv"
            subprocess.run(
                ["Rscript", str(extractor), str(path), str(extracted), str(wave)],
                check=True,
                capture_output=True,
                text=True,
            )
            parts.append(pd.read_csv(extracted, low_memory=False))
    raw = pd.concat(parts, ignore_index=True)
    if raw.duplicated("pid").any():
        raise ValueError(f"KLoSA wave {wave}: duplicate pid after first-imputation concatenation")
    prefix = f"w0{wave}C"
    value = lambda number: pd.to_numeric(raw[f"{prefix}{number}"], errors="coerce")
    scored = pd.DataFrame({"pid": pd.to_numeric(raw["pid"], errors="coerce")})
    scored["k_weekday"] = value(402).map({5.0: 0.0, 1.0: 1.0})
    scored["k_date"] = value(401).where(value(401).between(1, 3), np.where(value(401).eq(5), 0.0, np.nan))
    scored["k_season"] = value(403).map({5.0: 0.0, 1.0: 1.0})
    scored["k_place"] = value(404).map({5.0: 0.0, 1.0: 1.0})
    scored["k_address"] = value(405).where(value(405).between(1, 4), np.where(value(405).eq(5), 0.0, np.nan))
    scored["k_immediate"] = value(406).where(value(406).between(1, 3), np.where(value(406).eq(5), 0.0, np.nan))
    serial = pd.concat([value(number) for number in range(407, 412)], axis=1)
    serial.columns = [f"serial_{number}" for number in range(1, 6)]
    serial = serial.replace({1.0: 1.0, 5.0: 0.0})
    serial = serial.where(serial.isin([0.0, 1.0]))
    scored["k_serial"] = serial.sum(axis=1, min_count=5)
    scored["k_delayed"] = value(412).where(value(412).between(1, 3), np.where(value(412).eq(5), 0.0, np.nan))
    for number, name in [(413, "k_object1"), (414, "k_object2"), (415, "k_repeat"), (418, "k_write"), (419, "k_draw")]:
        scored[name] = value(number).map({5.0: 0.0, 1.0: 1.0})
    scored["k_command"] = value(416).where(value(416).between(1, 3), np.where(value(416).eq(5), 0.0, np.nan))
    # The archived do-file created an intermediate 0/1/2 variable and then
    # scored only code 3 (read and obey) as correct.  We create the final item
    # directly, avoiding the unretained intermediate variable.
    scored["k_read_obey"] = value(417).map({1.0: 0.0, 3.0: 1.0, 5.0: 0.0})
    items = [column for column in scored if column.startswith("k_")]
    scored["klosa_mmse30"] = scored[items].sum(axis=1, min_count=len(items))
    scored["wave"] = wave
    return scored


def patch_klosa_cognition(frame: pd.DataFrame, source_root: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    rebuilt = pd.concat([klosa_first_imputation(source_root, wave) for wave in range(3, 8)], ignore_index=True)
    frame["pid"] = pd.to_numeric(frame["pid"], errors="coerce")
    frame["wave"] = pd.to_numeric(frame["wave"], errors="coerce")
    merged = frame.merge(rebuilt, on=["pid", "wave"], how="left", validate="one_to_one")
    audit = {
        f"wave_{wave}_complete_mmse30": int(merged.loc[merged["wave"].eq(wave), "klosa_mmse30"].notna().sum())
        for wave in range(3, 8)
    }
    return merged.copy(), audit


def construct_domains(cohort: str, frame: pd.DataFrame) -> dict[str, object]:
    maps = {
        "charls": {
            "cognition": ["tr20", "orient", "ser7", "draw"],
            "adl": ["dressa", "batha", "eata", "beda", "toilta", "urina"],
            "iadl": ["moneya", "medsa", "shopa", "mealsa", "housewka"],
            "adl_min": 6, "iadl_min": 5,
        },
        "elsa": {
            "cognition": ["tr20", "orient"],
            "adl": ["walkra", "dressa", "batha", "eata", "beda", "toilta"],
            "iadl": ["mapa", "phonea", "medsa", "shopa", "mealsa", "housewka", "moneya"],
            "adl_min": 5, "iadl_min": 6,
        },
        "hrs": {
            "cognition": ["cog27"],
            "adl": ["walkra", "dressa", "batha", "eata", "beda", "toilta"],
            "iadl": ["moneya", "phonea", "medsa", "mealsa", "shopa"],
            "adl_min": 6, "iadl_min": 5,
        },
        "klosa": {
            "cognition": ["klosa_mmse30"],
            "adl": ["dressb", "bathb", "eatb", "toiltb", "bedb_k", "brushb", "urinb"],
            "iadl": ["mealsb", "shopb", "medsb", "moneyb", "phoneb", "transb", "gooutb", "laundryb", "housewkb", "groomb"],
            "adl_min": 7, "iadl_min": 10,
        },
        "mhas": {
            "cognition": ["mhas_memory", "vscan"],
            "adl": ["walkra", "dressa", "batha", "eata", "beda", "toilta"],
            "iadl": ["moneya", "medsa", "shopa", "mealsa"],
            "adl_min": 6, "iadl_min": 4,
        },
        "share": {
            "cognition": ["share_memory", "ser7", "verbf"],
            "adl": ["walkra", "dressa", "batha", "eata", "beda", "toilta"],
            "iadl": ["phonea", "medsa", "moneya", "shopa", "mealsa", "mapa"],
            "adl_min": 6, "iadl_min": 6,
        },
    }
    if cohort == "mhas":
        frame["mhas_memory"] = frame[["imrc8", "dlrc8"]].apply(pd.to_numeric, errors="coerce").mean(axis=1).mask(frame[["imrc8", "dlrc8"]].notna().sum(axis=1) < 2)
    if cohort == "share":
        frame["share_memory"] = frame[["imrc", "dlrc"]].apply(pd.to_numeric, errors="coerce").mean(axis=1).mask(frame[["imrc", "dlrc"]].notna().sum(axis=1) < 2)
    specification = maps[cohort]
    frame["function_adl_fraction"] = item_fraction(frame, specification["adl"], int(specification["adl_min"]))
    frame["function_iadl_fraction"] = item_fraction(frame, specification["iadl"], int(specification["iadl_min"]))
    return specification


def append_death_endpoints(
    output: pd.DataFrame,
    source_frame: pd.DataFrame,
    source_id: str,
    *,
    enabled: bool,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if not enabled or "radyear" not in source_frame:
        return output, {"mortality_endpoint_available": 0, "death_endpoint_rows_added": 0}
    mortality = pd.DataFrame({
        "id": source_frame[source_id].astype("string"),
        "death_year": pd.to_numeric(source_frame["radyear"], errors="coerce"),
        "death_month": pd.to_numeric(source_frame["radmonth"], errors="coerce") if "radmonth" in source_frame else np.nan,
    })
    mortality["death_month"] = mortality["death_month"].where(mortality["death_month"].between(1, 12), 6.5)
    mortality = mortality.dropna(subset=["death_year"]).sort_values(["id", "death_year"]).drop_duplicates("id")
    mortality["death_time"] = mortality["death_year"] + (mortality["death_month"] - 0.5) / 12
    living = output.copy()
    living["interview_time"] = living["interview_year"] + (living["interview_month"] - 0.5) / 12
    last = living.groupby("id", as_index=False)["interview_time"].max().rename(columns={"interview_time": "last_interview_time"})
    deaths = mortality.merge(last, on="id", how="inner")
    administrative_end = float(living["interview_time"].max()) + 0.5
    deaths = deaths[
        deaths["death_time"].gt(deaths["last_interview_time"])
        & deaths["death_time"].le(administrative_end)
    ].copy()
    endpoint = pd.DataFrame(np.nan, index=range(len(deaths)), columns=output.columns)
    endpoint["id"] = deaths["id"].to_numpy()
    endpoint["wave"] = "death"
    endpoint["interview_year"] = deaths["death_year"].to_numpy()
    endpoint["interview_month"] = deaths["death_month"].to_numpy()
    endpoint["death_endpoint"] = 1.0
    combined = pd.concat([output, endpoint], ignore_index=True)
    return combined, {
        "mortality_endpoint_available": 1,
        "unique_participants_with_recorded_death_year": int(mortality["id"].nunique()),
        "death_endpoint_rows_added": int(len(endpoint)),
        "death_endpoint_rule": "recorded death after last interview and no later than administrative end plus 0.5 years",
    }


def prepare_cohort(cohort: str, source_root: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    source = source_root / "csv 版本 清洗后" / f"{cohort}.csv"
    frame = base_headers(source)
    audit: dict[str, object] = {"cohort": cohort, "source": str(source), "source_rows": int(len(frame))}
    if cohort == "charls":
        audit["charls_wave4_stroke_values_recovered"] = patch_charls_stroke(frame, source_root)
    if cohort == "klosa":
        frame, audit["klosa_cognition_rebuild"] = patch_klosa_cognition(frame, source_root)
    specification = construct_domains(cohort, frame)

    columns = {
        "charls": dict(id="ID", year="iwy", month="iwm", age="age", sex="ragender", education="raeducl", depression="cesd10", psychiatric="psyche", smoking="smoken", bmi="bmi", activity="mdactx_c", alcohol="drinkl", wealth="hatotfa", marital="marry", heart="hearte", cholesterol="dyslipe"),
        "elsa": dict(id="idauniqc", year="iwindy", month="iwindm", age="agey", sex="ragender", education="raeducl", depression="cesd", psychiatric="psyche", smoking="smoken", bmi="mbmi", activity="mdactx_e", alcohol="drink", wealth="hatotb", marital="mstath", heart="hearte", cholesterol="hchole"),
        "hrs": dict(id="hhidpn", year="iwendy", month="iwendm", age="ragey_e", sex="ragender", education="raeducl", depression="cesd", psychiatric="psyche", smoking="smoken", bmi="bmi", activity="mdactx", alcohol="drink", wealth=None, marital="mstath", heart="hearte", cholesterol="hchole"),
        "klosa": dict(id="pid", year="iwy", month="iwm", age="agey", sex="ragender", education="raeducl", depression="cesd_combined", psychiatric="psyche", smoking="smoken", bmi="bmi", activity="vigact", alcohol="drink", wealth="wealth", marital="mstath", heart="hearte", cholesterol=None),
        # A harmonised broad heart-disease variable is present only in waves
        # 4-5.  The all-wave CMM4 therefore uses the explicitly labelled,
        # doctor-diagnosed heart-attack indicator and reports this limitation.
        "mhas": dict(id="rahhidnp", year="iwy", month="iwm", age="agey", sex="ragender", education="raeducl", depression="cesd_m", psychiatric=None, smoking="smoken", bmi="bmi", activity="vigact", alcohol="drink", wealth=None, marital="mstath", heart="hrtatte", cholesterol=None),
        "share": dict(id="mergeid", year="iwy", month="iwm", age="agey", sex="ragender", education="raeducl", depression="eurod", psychiatric="psyche", smoking="smoken", bmi="bmi", activity="mdactx", alcohol="drinkxw", wealth="hhatotb", marital="mstath", heart="hearte", cholesterol="hchole"),
    }[cohort]
    if cohort == "klosa":
        frame["cesd_combined"] = numeric(frame, "cesd10a").combine_first(numeric(frame, "cesd10b"))

    output = pd.DataFrame(index=frame.index)
    output["id"] = frame[columns["id"]].astype("string")
    output["wave"] = pd.to_numeric(frame["wave"], errors="coerce").astype("Int64").astype("string")
    output["interview_year"] = numeric(frame, columns["year"])
    output["interview_month"] = numeric(frame, columns["month"])
    for target, source_name in [("hypertension", "hibpe"), ("diabetes", "diabe"), ("heart", columns["heart"]), ("stroke", "stroke")]:
        output[target] = numeric(frame, source_name)
    output["cholesterol"] = numeric(frame, columns["cholesterol"]) if columns["cholesterol"] else np.nan
    for column in specification["cognition"]:
        output[column] = numeric(frame, column)
    output["function_adl_fraction"] = frame["function_adl_fraction"]
    output["function_iadl_fraction"] = frame["function_iadl_fraction"]
    for target in ["age", "sex", "education", "depression", "psychiatric", "smoking", "bmi", "activity", "alcohol", "wealth", "marital"]:
        source_name = columns[target]
        output[target] = numeric(frame, source_name) if source_name else np.nan
    output["bmi"], invalid_bmi = bounded(output["bmi"], 10, 80)
    audit["bmi_outside_prespecified_10_80_set_missing"] = invalid_bmi
    output["death_endpoint"] = 0.0
    # ELSA mortality fields in the local file stop in 2012 while interviews
    # continue to 2019, so ELSA is explicitly excluded from the death-state
    # model rather than treated as complete linkage.
    output, death_audit = append_death_endpoints(
        output,
        frame,
        columns["id"],
        enabled=cohort in {"charls", "hrs", "klosa", "mhas", "share"},
    )
    audit["mortality"] = death_audit
    audit["output_rows"] = int(len(output))
    audit["domain_map"] = specification
    audit["complete_cmm4_rows"] = int(output[["hypertension", "diabetes", "heart", "stroke"]].notna().all(axis=1).sum())
    audit["complete_cognition_rows"] = int(output[list(specification["cognition"])].notna().all(axis=1).sum())
    audit["complete_function_domain_rows"] = int(output[["function_adl_fraction", "function_iadl_fraction"]].notna().all(axis=1).sum())
    return output, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    audits = []
    for cohort in COHORTS:
        frame, audit = prepare_cohort(cohort, args.source_root)
        frame.to_csv(args.output / f"{cohort}.csv", index=False)
        audits.append(audit)
    (args.output / "input_construction_audit.json").write_text(json.dumps(audits, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
