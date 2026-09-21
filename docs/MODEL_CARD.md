# Model card

## Intended use

Rank player pairs and shared hands for human review of potentially coordinated poker activity in the competition's synthetic logs. The output is a triage aid and a reproducibility record.

## Pipeline

1. Parse hands, seats, and action logs while checking stable hand/action order.
2. Build symmetric pair-hand features: action context, partner-facing decisions, chip-transfer direction, showdown outcomes, card summaries, chronology, and player baselines.
3. Cross-fit a hand-event detector with complete player-pool holds. Public positive evidence is used as weak event supervision; unlisted hands from a positive pair are not assumed negative.
4. Aggregate cross-fitted event scores and behavior signatures to pair-level features.
5. Fit family-aware pair risk/behavior models and evidence rankers, then blend only within predeclared scopes.
6. Post-process stable five-hand lists, validate all references, and write the submission schema.

The selected files are generated exactly by the deterministic artifact replay. The repository also includes a compact clean-room refit implementation that exposes the same stage boundaries without pretending to reconstruct the historical 19 GB cache tree.

## Validation

Validation uses connected player-pool holds, shortened histories, chronological views, and stable evidence-reference checks. Reported validation is conditional on the released labels and capped evidence lists. It is not an estimate of private rank.

## Known limitations

The released fourth behavior is not fully represented by the development labels. Evidence labels are incomplete. Some final exploratory changes were retained for leaderboard selection despite uncertain hidden-set benefit. No risk score is calibrated as a probability, and no case review establishes player intent.

