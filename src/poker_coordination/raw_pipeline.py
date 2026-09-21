"""Executable clean-room feature, training, inference, and post-processing path.

This module is intentionally compact enough to audit in a notebook. It uses
only observable gameplay and does not try to recreate the historical cache
tree. The exact selected-file replay remains in ``reproduce.py``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


EVENT_FEATURES = [
    "final_pot",
    "players_dealt",
    "players_at_showdown",
    "left_total_contribution",
    "right_total_contribution",
    "left_net_chips",
    "right_net_chips",
    "left_raises",
    "right_raises",
    "left_bets",
    "right_bets",
    "left_calls",
    "right_calls",
    "left_folds",
    "right_folds",
    "transfer_imbalance",
    "fold_asymmetry",
    "aggression_asymmetry",
]

PAIR_FEATURES = [
    "shared_hand_count",
    "final_pot_mean",
    "transfer_imbalance_mean",
    "transfer_imbalance_max",
    "fold_asymmetry_mean",
    "aggression_asymmetry_mean",
    "left_net_chips_mean",
    "right_net_chips_mean",
    "players_at_showdown_mean",
]


def _polars():
    try:
        import polars as pl
    except ImportError as exc:  # pragma: no cover - depends on reviewer image
        raise RuntimeError("The optional refit path needs polars>=1.0") from exc
    return pl


def connected_pool_groups(pairs: pd.DataFrame) -> np.ndarray:
    """Assign connected player pools to stable integer groups."""
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

    for left, right in pairs[["player_1", "player_2"]].itertuples(index=False, name=None):
        union(str(left), str(right))
    roots = {find(value) for value in parent}
    root_to_group = {root: index for index, root in enumerate(sorted(roots))}
    return np.asarray([root_to_group[find(str(left))] for left in pairs["player_1"]], dtype=np.int32)


def build_pair_hand_features(data_root: str | Path, pairs: pd.DataFrame, max_pairs: int | None = None,
                             seats_table=None, action_summary=None) -> pd.DataFrame:
    """Join both players to shared hands and aggregate their action context.

    The result has one row per ``pair_id, hand_id``. Joins are on gameplay
    keys only; player IDs are discarded before model fitting.
    """
    pl = _polars()
    pairs = pairs[["pair_id", "player_1", "player_2"]].drop_duplicates().copy()
    if max_pairs is not None:
        pairs = pairs.head(max_pairs)
    pair_keys = pl.from_pandas(pairs).lazy()
    root = Path(data_root)
    seat_cols = [
        "hand_id", "player_id", "total_contribution", "net_chips",
        "folded", "went_to_showdown", "won_share",
    ]
    seats = (seats_table.lazy() if seats_table is not None
             else pl.scan_parquet(str(root / "seats.parquet"))).select(seat_cols)
    left = (
        seats.join(pair_keys, left_on="player_id", right_on="player_1", how="inner")
        .select(["pair_id", "hand_id", *[c for c in seat_cols if c not in ("hand_id", "player_id")]])
        .rename({c: f"left_{c}" for c in seat_cols if c not in ("hand_id", "player_id")})
    )
    right = (
        seats.join(pair_keys, left_on="player_id", right_on="player_2", how="inner")
        .select(["pair_id", "hand_id", *[c for c in seat_cols if c not in ("hand_id", "player_id")]])
        .rename({c: f"right_{c}" for c in seat_cols if c not in ("hand_id", "player_id")})
    )
    shared = left.join(right, on=["pair_id", "hand_id"], how="inner")
    hands = pl.scan_parquet(str(root / "hands.parquet")).select(
        ["hand_id", "final_pot", "players_dealt", "players_at_showdown"]
    )
    shared = shared.join(hands, on="hand_id", how="left")
    hand_ids = shared.select("hand_id").unique().collect().get_column("hand_id").to_list()

    actions = (
        (action_summary.lazy() if action_summary is not None
         else pl.scan_parquet(str(root / "actions.parquet")))
        .filter(pl.col("hand_id").is_in(hand_ids))
        .select(["hand_id", "player_id", "raises", "bets", "calls", "folds", "action_count", "action_amount"])
    )
    action_names = ["raises", "bets", "calls", "folds", "action_count", "action_amount"]
    left_action = actions.rename({"player_id": "left_player_id", **{c: f"left_{c}" for c in action_names}})
    right_action = actions.rename({"player_id": "right_player_id", **{c: f"right_{c}" for c in action_names}})
    # Add player keys only for the join, then remove them before returning.
    shared = shared.with_columns([
        pl.lit(None, dtype=pl.String).alias("left_player_id"),
        pl.lit(None, dtype=pl.String).alias("right_player_id"),
    ])
    # Rebuild the player keys from the original pair table through the joins.
    key_left = seats.join(pair_keys, left_on="player_id", right_on="player_1", how="inner").select(["pair_id", "hand_id", "player_id"])
    key_right = seats.join(pair_keys, left_on="player_id", right_on="player_2", how="inner").select(["pair_id", "hand_id", "player_id"])
    shared = shared.drop(["left_player_id", "right_player_id"])
    shared = shared.join(key_left.rename({"player_id": "left_player_id"}), on=["pair_id", "hand_id"], how="left")
    shared = shared.join(key_right.rename({"player_id": "right_player_id"}), on=["pair_id", "hand_id"], how="left")
    shared = shared.join(left_action, on=["hand_id", "left_player_id"], how="left")
    shared = shared.join(right_action, on=["hand_id", "right_player_id"], how="left")
    numeric = [c for c in shared.collect_schema().names() if c.startswith(("left_", "right_"))]
    result = shared.with_columns([pl.col(c).fill_null(0) for c in numeric]).with_columns(
        [
            (pl.col("left_net_chips").abs() + pl.col("right_net_chips").abs()).alias("transfer_denominator"),
            ((pl.col("left_net_chips") - pl.col("right_net_chips")).abs() /
             (pl.col("left_net_chips").abs() + pl.col("right_net_chips").abs() + 1)).alias("transfer_imbalance"),
            (pl.col("left_folded").cast(pl.Int8) - pl.col("right_folded").cast(pl.Int8)).abs().alias("fold_asymmetry"),
            ((pl.col("left_raises") + pl.col("left_bets")) -
             (pl.col("right_raises") + pl.col("right_bets"))).abs().alias("aggression_asymmetry"),
        ]
    ).drop(["left_player_id", "right_player_id", "transfer_denominator"])
    result = result.select(["pair_id", "hand_id", *EVENT_FEATURES])
    return result.collect().to_pandas()


def aggregate_pair_features(event_features: pd.DataFrame) -> pd.DataFrame:
    """Aggregate pair-hand rows into order-invariant pair predictors."""
    if event_features.empty:
        return pd.DataFrame(columns=["pair_id", *PAIR_FEATURES])
    grouped = event_features.groupby("pair_id", sort=False)
    pair = grouped.agg(
        shared_hand_count=("hand_id", "nunique"),
        final_pot_mean=("final_pot", "mean"),
        transfer_imbalance_mean=("transfer_imbalance", "mean"),
        transfer_imbalance_max=("transfer_imbalance", "max"),
        fold_asymmetry_mean=("fold_asymmetry", "mean"),
        aggression_asymmetry_mean=("aggression_asymmetry", "mean"),
        left_net_chips_mean=("left_net_chips", "mean"),
        right_net_chips_mean=("right_net_chips", "mean"),
        players_at_showdown_mean=("players_at_showdown", "mean"),
    ).reset_index()
    return pair[["pair_id", *PAIR_FEATURES]]


def fit_models(train_pairs: pd.DataFrame, train_events: pd.DataFrame) -> dict:
    """Fit pair risk/behavior and hand evidence models."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import LabelEncoder

    pair_features = aggregate_pair_features(train_events)
    frame = train_pairs.merge(pair_features, on="pair_id", how="left").fillna(0)
    X = frame[PAIR_FEATURES].to_numpy(dtype=np.float32)
    risk = HistGradientBoostingClassifier(max_iter=160, max_leaf_nodes=15, learning_rate=.06, random_state=2026)
    risk.fit(X, frame["label"].to_numpy(dtype=np.int8))
    encoder = LabelEncoder().fit(frame["behavior_family"].astype(str))
    behavior = HistGradientBoostingClassifier(max_iter=180, max_leaf_nodes=15, learning_rate=.05, random_state=2027)
    behavior.fit(X, encoder.transform(frame["behavior_family"].astype(str)))

    event_X = train_events[EVENT_FEATURES].fillna(0).to_numpy(dtype=np.float32)
    event_y = train_events["is_evidence"].to_numpy(dtype=np.int8)
    event_model = HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=15, learning_rate=.08, random_state=2028)
    weights = np.where(event_y == 1, 8.0, 1.0)
    event_model.fit(event_X, event_y, sample_weight=weights)
    return {"pair_features": PAIR_FEATURES, "risk": risk, "behavior": behavior,
            "behavior_encoder": encoder, "event_features": EVENT_FEATURES,
            "event_model": event_model}


def infer_submission(models: dict, eval_pairs: pd.DataFrame, eval_events: pd.DataFrame) -> pd.DataFrame:
    """Score evaluation pairs, retrieve five hands, and write submission rows."""
    pair_features = aggregate_pair_features(eval_events)
    frame = eval_pairs[["pair_id"]].merge(pair_features, on="pair_id", how="left").fillna(0)
    X = frame[models["pair_features"]].to_numpy(dtype=np.float32)
    risk = models["risk"].predict_proba(X)[:, 1]
    behavior = models["behavior_encoder"].inverse_transform(models["behavior"].predict(X))
    event_X = eval_events[models["event_features"]].fillna(0).to_numpy(dtype=np.float32)
    evidence_score = models["event_model"].predict_proba(event_X)[:, 1]
    scored = eval_events[["pair_id", "hand_id"]].copy()
    scored["evidence_score"] = evidence_score
    evidence = {}
    for pair_id, group in scored.groupby("pair_id", sort=False):
        ordered = group.sort_values(["evidence_score", "hand_id"], ascending=[False, True])
        selected = list(dict.fromkeys(ordered.hand_id.astype(str)))[:5]
        # The competition schema permits either five valid, distinct hands or
        # an explicit all-NO_EVIDENCE row. Never mix the two states.
        evidence[pair_id] = selected if len(selected) == 5 else ["NO_EVIDENCE"] * 5
    result = pd.DataFrame({"pair_id": eval_pairs.pair_id.astype(str), "risk_score": risk,
                           "predicted_behavior": behavior})
    for rank in range(5):
        result[f"evidence_hand_{rank+1}"] = result.pair_id.map(lambda p, r=rank: evidence.get(p, ["NO_EVIDENCE"] * 5)[r])
    return result


def add_evidence_labels(events: pd.DataFrame, evidence: pd.DataFrame) -> pd.DataFrame:
    """Mark released evidence rows without treating omissions as negatives."""
    keys = evidence[["pair_id", "hand_id"]].drop_duplicates().assign(is_evidence=1)
    result = events.merge(keys, on=["pair_id", "hand_id"], how="left")
    result["is_evidence"] = result["is_evidence"].fillna(0).astype(np.int8)
    return result


def aggregate_action_table(data_root: str | Path):
    """Pre-aggregate observable actions once for bounded pair-batch joins."""
    pl = _polars()
    root = Path(data_root)
    return (
        pl.scan_parquet(str(root / "actions.parquet"))
        .select(["hand_id", "player_id", "action", "amount"])
        .with_columns([
            (pl.col("action") == "raise").cast(pl.Int16).alias("raises"),
            (pl.col("action") == "bet").cast(pl.Int16).alias("bets"),
            (pl.col("action") == "call").cast(pl.Int16).alias("calls"),
            (pl.col("action") == "fold").cast(pl.Int16).alias("folds"),
        ])
        .group_by(["hand_id", "player_id"])
        .agg([
            pl.sum("raises"), pl.sum("bets"), pl.sum("calls"), pl.sum("folds"),
            pl.len().alias("action_count"), pl.sum("amount").alias("action_amount"),
        ])
        .collect()
    )


def run_reference_refit(data_root: str | Path, max_train_pairs: int | None = None,
                        max_eval_pairs: int | None = None,
                        eval_batch_size: int = 2500) -> pd.DataFrame:
    """Run the complete compact refit from raw logs to a fresh CSV dataframe.

    Evaluation pairs are processed in bounded batches so the complete raw
    path does not need to materialize every shared hand in memory at once.
    """
    if eval_batch_size <= 0:
        raise ValueError("eval_batch_size must be positive")
    root = Path(data_root)
    pl = _polars()
    train = pd.read_csv(root / "development_labels.csv")
    evidence = pd.read_csv(root / "development_evidence.csv")
    evaluation = pd.read_csv(root / "evaluation_pairs.csv")
    seat_table = pl.read_parquet(root / "seats.parquet", columns=[
        "hand_id", "player_id", "total_contribution", "net_chips",
        "folded", "went_to_showdown", "won_share",
    ])
    action_summary = aggregate_action_table(root)
    train_events = build_pair_hand_features(root, train, max_train_pairs,
                                            seats_table=seat_table,
                                            action_summary=action_summary)
    train_events = add_evidence_labels(train_events, evidence)
    models = fit_models(train.head(max_train_pairs) if max_train_pairs else train, train_events)
    eval_subset = evaluation.head(max_eval_pairs) if max_eval_pairs else evaluation
    chunks = []
    for start in range(0, len(eval_subset), eval_batch_size):
        stop = min(start + eval_batch_size, len(eval_subset))
        pair_chunk = eval_subset.iloc[start:stop].copy()
        event_chunk = build_pair_hand_features(root, pair_chunk,
                                               seats_table=seat_table,
                                               action_summary=action_summary)
        chunks.append(infer_submission(models, pair_chunk, event_chunk))
    if not chunks:
        return pd.DataFrame(columns=["pair_id", "risk_score", "predicted_behavior",
                                     "evidence_hand_1", "evidence_hand_2", "evidence_hand_3",
                                     "evidence_hand_4", "evidence_hand_5"])
    return pd.concat(chunks, ignore_index=True)


def write_submission(submission: pd.DataFrame, output_path: str | Path) -> Path:
    """Validate columns and write a fresh inference dataframe as a CSV."""
    expected = [
        "pair_id", "risk_score", "predicted_behavior",
        "evidence_hand_1", "evidence_hand_2", "evidence_hand_3",
        "evidence_hand_4", "evidence_hand_5",
    ]
    if list(submission.columns) != expected:
        raise ValueError(f"unexpected submission columns: {list(submission.columns)}")
    if submission["pair_id"].duplicated().any():
        raise ValueError("pair_id values must be unique")
    if not submission["risk_score"].between(0.0, 1.0).all():
        raise ValueError("risk_score values must lie in [0, 1]")
    evidence_cols = expected[3:]
    if (submission[evidence_cols].nunique(axis=1) < 5).any():
        raise ValueError("each row must contain five distinct evidence values")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(destination, index=False)
    return destination
