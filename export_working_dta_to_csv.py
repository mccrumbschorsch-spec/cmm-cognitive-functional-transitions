#!/usr/bin/env python3
"""Export the six private Stata working files to CSV for the Python pipeline.

The Stata harmonisation scripts create one working DTA per cohort. This bridge
performs the previously undocumented DTA-to-CSV step. Outputs are participant-
level and must remain outside the public repository.
"""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path

import pandas as pd


WORKING_FILES = {
    "charls": "1. CHARLS  中国/CHARLS_中国/Working_data/charls.dta",
    "elsa": "2. ELSA 英国/Working_data/elsa.dta",
    "hrs": "3. HRS  美国/HRS_美国/Working_data/hrs.dta",
    "klosa": "4. KLoSA 韩国/KLoSA_韩国/Working_data/klosa.dta",
    "mhas": "6. MHAS 墨西哥/MHAS_墨西哥/Working_data/mhas.dta",
    "share": "7.SHARE 欧洲/SHARE_欧洲/Working_data/share.dta",
}


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True,
                        help="Authorised seven-database project root")
    parser.add_argument("--output", type=Path, required=True,
                        help="Private output directory, normally SOURCE_ROOT/working_exports")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    audit = []
    for cohort, relative in WORKING_FILES.items():
        source = args.source_root / relative
        if not source.exists():
            raise FileNotFoundError(source)
        frame = pd.read_stata(source, convert_categoricals=False)
        target = args.output / f"{cohort}.csv"
        frame.to_csv(target, index=False)
        audit.append({
            "cohort": cohort,
            "source_relative_path": relative,
            "rows": int(len(frame)),
            "columns": int(len(frame.columns)),
            "source_sha256": file_sha256(source),
            "export_sha256": file_sha256(target),
        })
        print(f"{cohort}: {len(frame):,} rows x {len(frame.columns):,} columns -> {target}")

    (args.output / "working_export_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
