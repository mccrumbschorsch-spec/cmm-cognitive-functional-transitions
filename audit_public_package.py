#!/usr/bin/env python3
"""Fail-fast audit of the flat, data-free public analysis package."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


FORBIDDEN_SUFFIXES = {".dta", ".sav", ".sas7bdat", ".parquet", ".feather", ".docx", ".pdf", ".tif", ".tiff"}
FORBIDDEN_NAMES = {"build_docx_outputs.py", "manuscript_revised_working.md", "supplementary_methods_revised_working.md"}
IDENTIFIER_COLUMNS = {"id", "pid", "hhidpn", "mergeid", "idauniqc", "rahhidnp"}
REQUIRED = {
    "prepare_rebuild_inputs.py", "extract_klosa_cognition.R", "revision_pipeline_v3.py",
    "export_working_dta_to_csv.py", "stata_paths.do",
    "finalize_binary_and_prediction.py", "audit_klosa_item_reliability.py",
    "final_additional_sensitivities.py", "make_revised_figures.py",
    "prepare_statistical_table_data.py", "build_statistical_tables.py",
    "validate_final_outputs.py", "test_revision_pipeline_v3.py",
    "README.md", "DATA_ACCESS.md", "CITATION.cff",
    "revision_config_filled_20260821.json", "revision_config_template.json",
    "requirements.txt", "R_PACKAGES.txt", "analysis_completion_matrix_v3.csv",
    "model_failure_matrix_v3.csv", "analysis_manifest_public.json",
}


def main() -> None:
    root = Path(__file__).resolve().parent
    nested = [path for path in root.rglob("*") if path.is_dir() and path != root and path.name != "__pycache__"]
    if nested:
        raise AssertionError(f"Package is not flat: {nested}")
    files = {path.name: path for path in root.iterdir() if path.is_file()}
    missing = sorted(REQUIRED - files.keys())
    if missing:
        raise AssertionError(f"Missing required public files: {missing}")
    forbidden = sorted(name for name, path in files.items() if name in FORBIDDEN_NAMES or path.suffix.lower() in FORBIDDEN_SUFFIXES)
    if forbidden:
        raise AssertionError(f"Forbidden manuscript/private/binary source files: {forbidden}")

    do_files = sorted(root.glob("harmonise_*.do"))
    if len(do_files) != 42:
        raise AssertionError(f"Expected 42 cohort harmonisation do-files, found {len(do_files)}")
    for path in do_files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"\b[A-Za-z]:[\\/]", text) or any(token in text for token in ("/Users/", "/Volumes/")):
            raise AssertionError(f"Machine-specific path remains in {path.name}")

    for path in root.glob("*.csv"):
        columns = {column.lower() for column in pd.read_csv(path, nrows=0).columns}
        identifiers = sorted(columns & IDENTIFIER_COLUMNS)
        if identifiers:
            raise AssertionError(f"Potential participant identifiers in {path.name}: {identifiers}")

    manifest = json.loads((root / "analysis_manifest_public.json").read_text(encoding="utf-8"))
    if manifest.get("participant_level_outputs_written") is not False:
        raise AssertionError("Public manifest does not disable participant-level outputs")
    for item in manifest.get("inputs", []):
        if not str(item.get("path", "")).startswith("private_input/"):
            raise AssertionError("Manifest contains a machine-specific input path")

    print("PUBLIC PACKAGE AUDIT PASSED")
    print(f"flat_files={len(files)} participant_level_files=0 forbidden_submission_files=0")


if __name__ == "__main__":
    main()
