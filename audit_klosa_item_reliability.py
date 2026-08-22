#!/usr/bin/env python3
"""Audit KLoSA K-MMSE item reliability from official first imputations."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pandas as pd


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    prep = load_module("prepare_rebuild_inputs", here / "prepare_rebuild_inputs.py")
    v3 = load_module("revision_pipeline_v3", here / "revision_pipeline_v3.py")
    rows = []
    for wave in range(3, 8):
        data = prep.klosa_first_imputation(args.source_root, wave)
        item_columns = [column for column in data if column.startswith("k_")]
        complete = data[item_columns].dropna()
        rows.append({
            "wave": wave,
            "complete_case_n": len(complete),
            "scored_item_count": len(item_columns),
            "alpha_complete_case": v3.cronbach_alpha(complete),
            "omega_one_factor_complete_case": v3.one_factor_omega(complete),
            "metric_boundary": "Pearson one-factor omega; alpha reported in parallel",
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
