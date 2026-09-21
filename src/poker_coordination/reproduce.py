from __future__ import annotations

import csv
import json
from pathlib import Path

from .schema import EVIDENCE_FIELDS, SUBMISSION_FIELDS
from .validate import read_submission, sha256, validate_submission

EXPECTED_HASHES = {
    "base_v50": "b6196e8d05e445a1ab1f8abcbb77b9d8324a3dd7bd3678b243011b7efcf2952a",
    "patch_v51": "914a39f72217e7180795f7b4aa0765112cf44451cc49ff73657fe8528f8dd799",
    "patch_v53": "7aa09aa9b9fbc7da904f7f05a809aa7478d319736d2c1f0b14d99eed48c235ce",
    "v51": "c8b0f4c260f7766366ab2060bdc91849d24f0bd44d9dcf35c63c02fff582e370",
    "v53": "3991740457962e77d38a43859d00e2c31dbbbbd3a01d06946ff99f10d98fadd4",
}


def _read_patch(path: Path, fields: list[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        expected = ["pair_id", *fields]
        if reader.fieldnames != expected:
            raise ValueError(f"Unexpected patch schema in {path}: {reader.fieldnames}")
        rows = list(reader)
    if len({row["pair_id"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate pair_id in {path}")
    return rows


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUBMISSION_FIELDS, lineterminator="\r\n")
        writer.writeheader()
        writer.writerows(rows)


def reproduce(version: str, artifact_dir: str | Path, output: str | Path) -> dict[str, object]:
    artifact_dir = Path(artifact_dir)
    output = Path(output)
    base_path = artifact_dir / "submission_v50_soft_top5_consensus.csv"
    v51_path = artifact_dir / "v51_risk_patch.csv"
    v53_path = artifact_dir / "v53_evidence_patch.csv"
    if sha256(base_path) != EXPECTED_HASHES["base_v50"]:
        raise ValueError("Frozen v50 base hash does not match the audited manifest")
    if sha256(v51_path) != EXPECTED_HASHES["patch_v51"]:
        raise ValueError("Frozen v51 patch hash does not match the audited manifest")
    if sha256(v53_path) != EXPECTED_HASHES["patch_v53"]:
        raise ValueError("Frozen v53 patch hash does not match the audited manifest")

    rows = [dict(row) for row in read_submission(base_path)]
    by_pair = {row["pair_id"]: row for row in rows}
    if version == "v51":
        fields = ["risk_score"]
        patch = _read_patch(v51_path, fields)
    elif version == "v53":
        fields = EVIDENCE_FIELDS
        patch = _read_patch(v53_path, fields)
    else:
        raise ValueError(f"Unsupported selected version: {version}")
    for item in patch:
        pair_id = item["pair_id"]
        if pair_id not in by_pair:
            raise ValueError(f"Patch contains an unknown pair: {pair_id}")
        by_pair[pair_id].update({field: item[field] for field in fields})
    _write(output, rows)
    report = validate_submission(output, EXPECTED_HASHES[version])
    report.update(version=version, changed_rows=len(patch))
    return report


def reproduce_all(artifact_dir: str | Path, output_dir: str | Path) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = [
        reproduce("v53", artifact_dir, output_dir / "submission_v53.csv"),
        reproduce("v51", artifact_dir, output_dir / "submission_v51.csv"),
    ]
    payload = {"status": "passed", "reproductions": reports}
    (output_dir / "reproduction_report.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload

