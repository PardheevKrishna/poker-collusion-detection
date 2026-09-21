from __future__ import annotations

SUBMISSION_FIELDS = [
    "pair_id",
    "risk_score",
    "predicted_behavior",
    "evidence_hand_1",
    "evidence_hand_2",
    "evidence_hand_3",
    "evidence_hand_4",
    "evidence_hand_5",
]

EVIDENCE_FIELDS = [f"evidence_hand_{rank}" for rank in range(1, 6)]

ALLOWED_BEHAVIORS = {
    "none",
    "directed_transfer",
    "soft_play",
    "coordinated_isolation",
    "other_coordination",
}

EXPECTED_ROWS = 112_540
EXPECTED_EVIDENCE_REFERENCES = 562_700

