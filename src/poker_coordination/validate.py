from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from .schema import (
    ALLOWED_BEHAVIORS,
    EVIDENCE_FIELDS,
    EXPECTED_EVIDENCE_REFERENCES,
    EXPECTED_ROWS,
    SUBMISSION_FIELDS,
)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_submission(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != SUBMISSION_FIELDS:
            raise ValueError(f"Unexpected schema in {path}: {reader.fieldnames}")
        rows = list(reader)
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS:,} rows in {path}; found {len(rows):,}")
    if len({row["pair_id"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate pair_id in {path}")
    return rows


def validate_submission(path: str | Path, expected_hash: str | None = None) -> dict[str, object]:
    path = Path(path)
    rows = read_submission(path)
    evidence_references = 0
    for row_number, row in enumerate(rows, start=2):
        risk = float(row["risk_score"])
        if not 0.0 <= risk <= 1.0:
            raise ValueError(f"Risk outside [0,1] on CSV row {row_number}")
        if row["predicted_behavior"] not in ALLOWED_BEHAVIORS:
            raise ValueError(f"Unknown behavior on CSV row {row_number}")
        evidence = [row[field] for field in EVIDENCE_FIELDS]
        if evidence == ["NO_EVIDENCE"] * 5:
            continue
        if len(set(evidence)) != 5 or any(not hand.startswith("H") for hand in evidence):
            raise ValueError(f"Invalid evidence list on CSV row {row_number}")
        evidence_references += 5
    if evidence_references != EXPECTED_EVIDENCE_REFERENCES:
        raise ValueError(
            f"Expected {EXPECTED_EVIDENCE_REFERENCES:,} evidence references; "
            f"found {evidence_references:,}"
        )
    digest = sha256(path)
    if expected_hash is not None and digest != expected_hash:
        raise ValueError(f"SHA-256 mismatch for {path}: {digest} != {expected_hash}")
    return {
        "path": str(path),
        "rows": len(rows),
        "unique_pairs": len(rows),
        "evidence_references": evidence_references,
        "sha256": digest,
    }

