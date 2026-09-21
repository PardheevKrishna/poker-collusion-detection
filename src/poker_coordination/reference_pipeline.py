"""Small, dependency-light reference implementation of the modeling stages.

The competition submission used richer cross-fitted LightGBM/CatBoost and
LambdaMART components. These functions keep the same data flow visible and
make a fresh, compact refit possible for reviewers with the competition data.
They deliberately exclude player identifiers from predictive columns.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    stage: str
    description: str


FEATURE_SPEC = (
    FeatureSpec("shared_hand_count", "pair", "Number of hands in which both players sat together"),
    FeatureSpec("fold_asymmetry", "pair", "Absolute difference in fold rates within shared hands"),
    FeatureSpec("raise_asymmetry", "pair", "Absolute difference in raise rates"),
    FeatureSpec("net_transfer_asymmetry", "pair", "Absolute normalized difference in net chips"),
    FeatureSpec("showdown_asymmetry", "pair", "Difference in showdown participation"),
    FeatureSpec("partner_facing_surprisal", "event", "Observed minus expected action under context"),
    FeatureSpec("street_pressure", "event", "Bet/raise pressure after the pair meets"),
    FeatureSpec("chronology_rank", "event", "Relative time within the pair's history"),
)


def symmetric_pair_features(event_rows: np.ndarray) -> np.ndarray:
    """Build orientation-invariant event features from numeric event rows.

    Input columns are ``[hands, fold_rate_a, fold_rate_b, raise_rate_a,
    raise_rate_b, transfer_a, transfer_b, showdown_a, showdown_b,
    partner_surprisal, pressure, chronology]``. Sorting the two player
    columns by a value-derived rule makes the result invariant to seat order.
    """
    x = np.asarray(event_rows, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] < 12:
        raise ValueError("event_rows must be a 2-D array with at least 12 columns")
    result = np.column_stack(
        [
            x[:, 0],
            np.abs(x[:, 1] - x[:, 2]),
            np.abs(x[:, 3] - x[:, 4]),
            np.abs(x[:, 5] - x[:, 6]),
            np.abs(x[:, 7] - x[:, 8]),
            x[:, 9:12],
        ]
    )
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def connected_pool_groups(player_pairs: Iterable[tuple[str, str]]) -> np.ndarray:
    """Return stable connected-component group IDs for player-pool splits."""
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a

    pairs = list(player_pairs)
    for left, right in pairs:
        union(str(left), str(right))
    roots = {root: index for index, root in enumerate(sorted({find(p) for pair in pairs for p in pair}))}
    return np.asarray([roots[find(left)] for left, _ in pairs], dtype=np.int32)


def stable_group_folds(groups: np.ndarray, n_splits: int = 4) -> np.ndarray:
    """Assign complete groups to folds using a deterministic round-robin."""
    groups = np.asarray(groups)
    unique = np.unique(groups)
    mapping = {group: index % n_splits for index, group in enumerate(unique)}
    return np.asarray([mapping[group] for group in groups], dtype=np.int8)


def fit_reference_models(X: np.ndarray, y_risk: np.ndarray, y_behavior: np.ndarray, folds: np.ndarray):
    """Fit compact risk and behavior models for a clean-room refit demo."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import LabelEncoder

    X = np.asarray(X, dtype=np.float32)
    risk = HistGradientBoostingClassifier(max_iter=160, max_leaf_nodes=15, learning_rate=0.06, random_state=2026)
    risk.fit(X, np.asarray(y_risk, dtype=np.int8))
    encoder = LabelEncoder().fit(np.asarray(y_behavior))
    behavior = HistGradientBoostingClassifier(max_iter=180, max_leaf_nodes=15, learning_rate=0.05, random_state=2027)
    behavior.fit(X, encoder.transform(np.asarray(y_behavior)))
    return {"risk": risk, "behavior": behavior, "behavior_encoder": encoder, "folds": folds}


def score_reference_models(models: dict, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    risk_score = models["risk"].predict_proba(np.asarray(X, dtype=np.float32))[:, 1]
    behavior_code = models["behavior"].predict(np.asarray(X, dtype=np.float32))
    behavior = models["behavior_encoder"].inverse_transform(behavior_code)
    return risk_score, behavior


def rank_top_five(hand_ids: np.ndarray, scores: np.ndarray) -> list[str]:
    """Stable top-five evidence post-processing with no duplicate hands."""
    hand_ids = np.asarray(hand_ids)
    scores = np.asarray(scores, dtype=np.float64)
    order = sorted(range(len(hand_ids)), key=lambda i: (-scores[i], str(hand_ids[i])))
    selected: list[str] = []
    for index in order:
        hand = str(hand_ids[index])
        if hand not in selected:
            selected.append(hand)
        if len(selected) == 5:
            break
    return selected + ["NO_EVIDENCE"] * (5 - len(selected))

