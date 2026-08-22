# Reconstructed analysis workflow

The v3 analysis was locked and rerun on 21 August 2026. It rebuilds the six cohort inputs, repairs KLoSA cognition, uses exact interview dates, constructs complete CMM4/CMM5 exposures, fits participant-clustered models, and writes aggregate outputs only.

## Order of execution

1. `extract_klosa_cognition.R` reconstructs the KLoSA 15-item K-MMSE from the source wave files.
2. `prepare_rebuild_inputs.py` reads the cohort CSVs, standardises source names, repairs documented upstream omissions, reconstructs scheduled response/death records and writes private working inputs.
3. `revision_pipeline_v3.py` performs harmonisation, fixed/wave standardisation, continuous GEE, threshold audits, persistent outcomes, death/multinomial analyses, IPCW, meta-analysis and completion tracking.
4. `finalize_binary_and_prediction.py` refits guarded binary models, performs grouped cross-validation and absolute-risk estimation, and refreshes pooling/completion files.
5. `audit_klosa_item_reliability.py` calculates item-level KLoSA K-MMSE reliability.
6. `final_additional_sensitivities.py` fits fixed reference-wave thresholds, disease-restricted and CMM3-without-stroke exposures, temporal-boundary models, cohort exclusions, KLoSA ADL-only and extreme-interval sensitivities, and writes aggregate audits.
7. `validate_final_outputs.py` applies fail-fast publication checks.
8. `make_revised_figures.py` regenerates Figures 1-3 and Supplementary Figure S1 from the final aggregate CSVs.

## Locked choices

- Primary continuous model: fixed-reference, unannualised paired cognitive/functional change; exact interval adjustment; CMM4 change-by-domain interaction.
- Primary binary model: fixed-reference, cohort-wave 20% target-occupancy states; any cognitive and any functional outcomes.
- Threshold sensitivities: 15%, 25%, historical ±0.43, strict ±0.67 and persistent impairment.
- Absolute-position sensitivity: reference-wave 20% cut-points locked and applied unchanged thereafter.
- Ascertainment/measurement sensitivities: diabetes plus stroke, CMM3 excluding stroke, ELSA/KLoSA exclusions and KLoSA ADL-only scoring.
- CMM: complete four-component denominator at both endpoints; complete CMM5 only where all five components are observed.
- Repeated intervals: participant-clustered GEE.
- Pooling: REML with Hartung-Knapp confidence intervals and prediction intervals for k≥3.
- Mortality: terminal absorbing state in five validated cohorts; ELSA labelled unavailable.
- Attrition: non-death IPCW with deaths excluded from the censoring model.
- Reproducibility: no participant-level outputs.

## Private data boundary

The cohort files are not distributed. Users must obtain each cohort through its official access route and provide paths/configuration locally. The public package contains only scripts, configuration templates, aggregate results, figures and validation records.

## Validation

The final compact validation reports 76 pooled result rows, 423 cohort result rows and 438 registered model cells: 376 estimated and 62 failed/skipped with explicit reasons. No registered cell was silently omitted and no unregistered estimate was pooled.
