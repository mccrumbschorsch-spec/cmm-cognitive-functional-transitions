# Data access and governance boundary

Participant-level data are not distributed in this repository.

The analysis uses deidentified data from CHARLS, ELSA, HRS, KLoSA, MHAS and
SHARE. Users must obtain each study under its official public, registered or
controlled-access conditions. The harmonised releases and included waves used
for the locked analysis are recorded in
`revision_config_filled_20260821.json`.

The `harmonise_<cohort>_*.do` files document the raw/harmonised-release to
working-DTA stage. `export_working_dta_to_csv.py` then creates the six private
working exports. `prepare_rebuild_inputs.py --source-root` expects that
authorised local project tree, including those exports and the specific
source-wave files needed for the verified CHARLS and KLoSA repairs. Source
files are read only. Reconstructed participant-level CSVs are written solely to
caller-supplied private directories and must never be committed to the public
repository.

The public `analysis_manifest_public.json` reports repository-relative
placeholders and SHA-256 fingerprints of the locked private inputs. It does not
contain machine-specific paths or participant data.

The public CSV and JSON outputs contain cohort-level model estimates, pooled
estimates, counts, distribution summaries, reliability summaries, performance
metrics, model-completion records and failure reasons only.
