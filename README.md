# CMM and cognitive-functional transitions: public analysis package

This is the flat, data-free reproducibility package for the analysis locked on
21 August 2026 across CHARLS, ELSA, HRS, KLoSA, MHAS and SHARE.

All public files are intentionally kept in this single directory. The package
contains the complete processing and analysis chain needed after an authorised
user supplies the cohort source files. It does not contain manuscript,
response-letter or office-document generation code.

## Included

### Data construction and harmonisation

- `stata_paths.do` — the only file in which users set six authorised local
  cohort roots.
- `harmonise_<cohort>_wave*.do` and `harmonise_<cohort>_merge.do` — the 42
  cohort/wave Stata scripts that create the six private working DTA files from
  the authorised harmonised and wave-specific releases. Machine-specific `E:`
  paths were removed. The verified KLoSA and CHARLS repairs are applied in the
  subsequent Python reconstruction rather than concealed in these provenance
  scripts.
- `export_working_dta_to_csv.py` — the documented private DTA-to-CSV bridge that
  was missing from the archived workflow.
- `prepare_rebuild_inputs.py` — constructs audited private analysis inputs from
  the authorised cohort source tree, repairs the verified CHARLS stroke omission,
  reconstructs non-overlapping cognitive/functional measures and creates
  validated terminal-death endpoints.
- `extract_klosa_cognition.R` — extracts the first official KLoSA imputation and
  reconstructs the 15-item K-MMSE input used by the Python workflow.
- `revision_config_filled_20260821.json` — locked cohort, variable, wave,
  threshold and model configuration.
- `revision_config_template.json` — schema template for independent adaptation.

### Statistical analyses

- `revision_pipeline_v3.py` — harmonisation checks, exact interval construction,
  continuous and transition GEE, threshold analyses, persistent outcomes,
  mortality/multinomial models, IPCW, meta-analysis and completion tracking.
- `finalize_binary_and_prediction.py` — guarded binary refits, grouped
  cross-validation, absolute-risk estimation and refreshed completion records.
- `audit_klosa_item_reliability.py` — KLoSA item-level alpha/omega audit.
- `final_additional_sensitivities.py` — fixed reference-wave thresholds,
  disease-restricted CMM definitions, temporal-boundary models, cohort
  exclusions, KLoSA ADL-only and extreme-interval sensitivities.

### Tables, figures and validation

- `make_revised_figures.py` — regenerates Figures 1–3 and Supplementary Figure S1
  from aggregate CSV files only.
- `prepare_statistical_table_data.py` and `build_statistical_tables.py` — create
  the 19-sheet aggregate statistical workbook with public Python dependencies.
- `validate_final_outputs.py` — fail-fast checks on the locked statistical output.
- `test_revision_pipeline_v3.py` — regression tests for critical pipeline logic.
- `audit_public_package.py` — checks the flat-file and data-governance boundary.

### Locked non-disclosive outputs

The CSV/JSON files in this directory contain cohort-model or pooled estimates,
counts, audit summaries, model-completion records and failure reasons. They do
not contain participant identifiers or participant-level rows. The four PNG
files and `Supplementary_Appendix_2_Tables.xlsx` are the locked aggregate
displays generated from those outputs.

## Explicitly excluded

- original or harmonised participant-level cohort data;
- derived participant-level intervals, states, predictions or weights;
- credentials, access tokens and machine-specific source paths;
- manuscripts, supplementary prose, response letters and DOCX/PDF builders;
- QA screenshots, temporary files and duplicate repository drafts.

See `DATA_ACCESS.md` for the authorised-input boundary.

## Software

Python versions and packages are in `requirements.txt`. KLoSA extraction also
requires R and the packages in `R_PACKAGES.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Reproduction order

Run from this directory. Keep private inputs and newly generated outputs outside
the public repository directory.

```bash
# 0. Set the six authorised project roots in stata_paths.do. In Stata, run the
#    wave scripts required for each cohort and then its harmonise_*_merge.do.
#    The merge scripts write private Working_data/<cohort>.dta files.

# 1. Export the six private working DTA files. This output remains private.
python export_working_dta_to_csv.py \
  --source-root /path/to/authorised/seven-database/project/root \
  --output /path/to/authorised/seven-database/project/root/working_exports

# 2. Build private, analysis-ready inputs from the authorised working exports.
python prepare_rebuild_inputs.py \
  --source-root /path/to/authorised/seven-database/project/root \
  --output /secure/path/private_input

# 3. Run the locked main pipeline.
python revision_pipeline_v3.py \
  --config revision_config_filled_20260821.json \
  --input-root /secure/path/private_input \
  --output /work/path/outputs_v3

# 4. Complete guarded binary, IPCW, prediction and risk analyses.
python finalize_binary_and_prediction.py \
  --config revision_config_filled_20260821.json \
  --input-root /secure/path/private_input \
  --output /work/path/outputs_v3 \
  --pipeline revision_pipeline_v3.py

# 5. Audit KLoSA item reliability.
python audit_klosa_item_reliability.py \
  --source-root /path/to/authorised/seven-database/project/root \
  --output /work/path/outputs_v3/klosa_item_reliability_v3.csv

# 6. Run the additional sensitivity analyses.
python final_additional_sensitivities.py \
  --config revision_config_filled_20260821.json \
  --input-root /secure/path/private_input \
  --main-results /work/path/outputs_v3/cohort_model_results_v3.csv \
  --output /work/path/additional_outputs

# 7. Validate the final aggregate outputs.
python validate_final_outputs.py /work/path/outputs_v3

# 8. Regenerate the aggregate figures.
python make_revised_figures.py \
  --results /work/path/outputs_v3 \
  --additional /work/path/additional_outputs \
  --output /work/path/figures

# 9. Regenerate the aggregate table workbook.
python prepare_statistical_table_data.py \
  --results /work/path/outputs_v3 \
  --additional /work/path/additional_outputs \
  --output /work/path/statistical_tables.json
python build_statistical_tables.py \
  --input /work/path/statistical_tables.json \
  --output /work/path/Supplementary_Appendix_2_Tables.xlsx

# 10. Run code regression tests and audit this public package.
pytest -q -p no:cacheprovider test_revision_pipeline_v3.py
python audit_public_package.py
```

## Locked validation state

- 76 principal pooled result rows;
- 423 cohort-model result rows;
- 438 registered cohort-model cells: 376 estimated and 62 failed/skipped with
  explicit reasons;
- 59 additional pooled sensitivity rows;
- 361,562 intervals in the primary paired continuous analysis;
- 266,127 intervals in the primary 20% target-occupancy transition analysis.

## Public release

Before creating the GitHub/archival release, the authors must select a software
licence. After publication, add the verified repository URL and persistent
identifier to `CITATION.cff`, the manuscript and the response letter. Do not
invent or pre-fill a DOI.

## Upstream Stata boundary

The 42 Stata do-files are the project scripts that produced the private working
DTA files and are included for full provenance. They require licensed Stata and
the exact authorised releases named in the configuration and supplementary
methods. They were path-sanitised for public release; the historical source
archive was not rerun in the current Python-only validation environment. The
locked 21 August analysis was rerun from their verified working exports, with
the documented CHARLS stroke and KLoSA cognition repairs performed by
`prepare_rebuild_inputs.py`.
