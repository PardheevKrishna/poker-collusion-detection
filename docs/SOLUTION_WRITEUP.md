# Detect Suspicious Value Transfers in Poker — solution writeup

## Summary

Our solution ranks player pairs by a risk score, predicts a coordination behavior family, and returns five supporting hand IDs. The model uses only observable poker activity: player actions, timing/history, pot and stack context, board and hole-card information where available, and repeated interaction between the same two players. It does not use pair-ID formatting, row order, file order, or generator internals.

The two selected submissions were v53 (Kaggle submission 56391641) and v51 (submission 56354264). Their exact SHA-256 hashes are recorded in the public reproduction artifact. V53 is the primary entry: it retains v50 risk and behavior predictions and applies a fixed 25% precise soft-play MAP component to the existing soft-play evidence scope. V51 retains the earlier evidence and behavior predictions and applies the alternative anomaly-risk ranking to 1,486 rows as a hedge against an unseen behavior family.

## Data and feature design

We began with the supplied synthetic poker logs and the competition’s development labels. For each pair, we aggregate shared hands into order-invariant pair features: action counts and amounts by street, checks in heads-up and multiway situations, folds after partner actions, calls and raises facing the partner, pot and stack context, showdown and contribution summaries, and history features computed without using the held-out pool. Evidence models operate on hand-level representations and return a stable top-five list. Behavior and risk heads are kept separate so that evidence retrieval can improve without changing the other metric components.

The core training process uses nested pool-held validation and chronological stress views. The final soft-play refinement is a depth-three, 650-tree MAP ranker with a fixed 25% blend against the audited v50 score. It changes 230 evidence lists, including 91 top-five membership changes, while preserving every risk, behavior, and out-of-scope field. The selected v51 risk hedge changes 1,486 risk rows while preserving v50 evidence and behavior strings.

## Validation and selection

We evaluated full-history, two random shortened histories, and early and late chronological windows. Every candidate was checked for row count, schema, evidence references, stable ordering, unchanged out-of-scope rows, and reproducible hashes. Independent replay checks reconstructed the final v53 output and confirmed 112,540 rows and 562,700 evidence references. The v53 strict validation gate remained failed because the late mean was slightly negative; it was submitted only under a separately documented exploratory exception. We do not present that exception as a validation pass.

The final pair was selected for coverage of two different uncertainties: v53 changes evidence retrieval while retaining the established risk ordering, and v51 supplies a materially different risk ordering while retaining the established evidence. This is a portfolio judgment, not a claim that private performance can be inferred from public rank or that either entry guarantees first place.

## Reproduction

The companion [public Kaggle Notebook](https://www.kaggle.com/code/pardheev/from-poker-actions-to-suspicious-pair-evidence) runs in a clean environment using the [public artifact dataset](https://www.kaggle.com/datasets/pardheev/poker-selected-submission-reproduction) and Python’s standard library for the exact replay. The notebook also exposes the preprocessing, feature, validation, training, inference, and post-processing stages. The [public GitHub repository](https://github.com/pardheev/poker-collusion-detection) contains the same notebook source, a one-command replay script, a data card, model card, compliance checklist, and a GitHub Actions hash check. The exact replay verifies the frozen v50 base hash, applies the compact deterministic v53 evidence patch and v51 risk patch, writes both selected CSVs with the original CRLF formatting, validates the complete submission schema and evidence references, and asserts the exact hashes:

* v53 / 56391641: `3991740457962e77d38a43859d00e2c31dbbbbd3a01d06946ff99f10d98fadd4`
* v51 / 56354264: `c8b0f4c260f7766366ab2060bdc91849d24f0bd44d9dcf35c63c02fff582e370`

This is an exact artifact replay of the selected files, not a claim that the historical 19 GB intermediate training cache is silently regenerated. The notebook states this limitation explicitly and includes the method description, protocol, checksums, and five case reviews. The [case review appendix](https://www.kaggle.com/code/pardheev/from-poker-actions-to-suspicious-pair-evidence#Five-action-grounded-case-reviews) lists each pair ID, relevant submitted hand IDs, observable action pattern, and a plausible benign explanation.

## Case reviews and limitations

The five reviews are deliberately cautious. A repeated raise/call/fold pattern can be suspicious in context, but the same sequence can arise from position, hole-card strength, stack pressure, table dynamics, or ordinary differences in playing style. Evidence is therefore a review aid rather than proof of intent. All examples use anonymized evaluation pairs and submitted hand IDs; no private labels are asserted.

The competition data are synthetic, which makes controlled validation possible but does not make the hidden leaderboard predictable. Development labels were reused during model selection, history views are correlated, and the private leaderboard remains the authoritative evaluation. We report those limits directly so the result can be independently audited.
