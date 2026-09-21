# Data card

## Contents

The public artifact dataset contains only the compact inputs needed to rebuild the selected CSVs:

| File | Description |
|---|---|
| `submission_v50_soft_top5_consensus.csv` | 112,540-row frozen base submission. All v51/v53 construction starts here. |
| `v51_risk_patch.csv` | 1,486 pair-level risk replacements copied from the selected v51 risk component. |
| `v53_evidence_patch.csv` | 230 pair-level five-hand evidence replacements used by selected v53. |
| `artifact_manifest.json` | Hashes, row counts, submission IDs, and scope of the replay. |

## Schema

The submission schema is stable across all three CSVs:

| Column | Type | Meaning |
|---|---|---|
| `pair_id` | identifier | Anonymized pair key used to join rows. It is not a model feature. |
| `risk_score` | decimal in [0, 1] | Pair-level suspicion score. |
| `predicted_behavior` | categorical | `none`, `directed_transfer`, `soft_play`, `coordinated_isolation`, or `other_coordination`. |
| `evidence_hand_1` ... `evidence_hand_5` | identifier | Five distinct shared hand IDs, or five `NO_EVIDENCE` values. |

Patch schemas use `pair_id` plus only the cells they replace. A patch is applied after verifying the base hash and before validating the final output.

## Provenance

The source logs supplied by the competition are synthetic, anonymized poker activity. The bundle was derived from the author's selected competition submissions on September 20, 2026. It contains no private leaderboard data and no unreleased labels. The artifact is published solely to satisfy the competition's reproducibility requirement; competition terms govern reuse of derived identifiers and values.

The generator uses gameplay-derived values only. It does not inspect ID formatting, row order, file order, generator internals, or hidden data.

## Limitations

This is a derived submission artifact, not a general-purpose player database. The evidence lists are ranking outputs for review, not proof of collusion. The v51 and v53 patches are intentionally sparse so a reviewer can inspect exactly which cells changed.

