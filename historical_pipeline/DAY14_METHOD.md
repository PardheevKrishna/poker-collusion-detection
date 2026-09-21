# September 19 research and submissions

The retained public best is **v50, 0.93876**, improving the opening **0.93827** by **0.00049**. The latest official check places Pardheev Krishna **first**, ahead of **0.93823** by **0.00053**. This is a measured improvement, not the requested drastic jump to 0.95. Private rank remains unknown. The competition closes September 20 at 22:00 UTC, September 21 at 03:30 India time.

## Official experiment ledger

| Entry | Change | Public score | Decision |
|---|---|---:|---|
| v47, 56353453 | Anomaly-risk alternative with all v39 evidence and behaviors fixed | 0.93827 | Tied; separate risk hedge |
| v48, 56353841 | Isolation evidence from chronological MAP ranker, fixed 25% blend | 0.93793 | Rejected |
| v49, 56353964 | Soft-play refinement of shared chronological model, fixed 25% blend | 0.93830 | Improved, then superseded by v50 |
| **v50, 56354034** | **MAP/Lambda consensus reorders only the existing five soft-play evidence hands** | **0.93876** | **Retained public best** |
| v51, 56354264 | V50 evidence with exactly the submitted v47 risk scores | 0.93876 | Tied best; selected risk alternative |

All five uploads have completed; no daily slots remain. Live official accounting is in [the ledger](submissions/day14_submission_ledger.json). No future public or private improvement is promised.

## Accepted evidence change

V50 retains every risk score and behavior label from v39. Among 660 eligible soft-play pairs, it reorders 193 evidence lists while preserving **every row's exact set of five evidence hands**. Other rows are unchanged. Each new model ranks only those five hands, and its ranking is mapped to the sorted existing scores. Final weights are 75% accepted score, 12.5% MAP and 12.5% chronological LambdaRank. This tests complementary ordering errors without introducing evidence members.

Models use gameplay/action features, nested hand/cue predictions, within-history summaries and a behavior indicator. Whole player pools are excluded from outer validation training. History features are recomputed within full, random shortened, early and late windows. The actual unknown fourth behavior is not represented by these soft-play evidence tests.

| Lambda model | Full MAP@5 gain | Random short A | Random short B | Early | Late | Full 95% pool-bootstrap interval |
|---|---:|---:|---:|---:|---:|---|
| Original | +0.001709 | +0.001637 | +0.003180 | +0.000284 | +0.009951 | [-0.002143, +0.005556] |
| 80% training pools, replica 1 | +0.001667 | +0.000355 | +0.002043 | +0.001080 | +0.006645 | [-0.002443, +0.005754] |
| 80% training pools, replica 2 | +0.001667 | +0.002438 | +0.005452 | +0.001883 | +0.001631 | [-0.002549, +0.005893] |

Each version has three positive full-history folds out of four. MAP and upstream models remain fixed during perturbations: this is **partial stability evidence**, not independent full-pipeline validation. Every interval includes zero. Reused public labels and multiple screens informed selection; these are exploratory comparisons, not corrected post-selection significance claims or winning probabilities. See [perturbation report](models/day14_consensus_perturbation/REPORT.md).

After v50's public gain, exactly one stronger weight was declared: 50% accepted score and 25% per new model. It lost to v50 in all five original-model views (-0.002666, -0.001541, -0.008331, -0.003683, -0.003318), and both partial perturbations failed. It was rejected without a submission. No weight grid was searched. See [stronger-weight report](models/day14_stronger_consensus/REPORT.md).

V50 source: submissions/submission_v50_soft_top5_consensus.csv. SHA-256: b6196e8d05e445a1ab1f8abcbb77b9d8324a3dd7bd3678b243011b7efcf2952a. Configuration and model provenance: [configuration](models/submission_v50_soft_top5_consensus_configuration.json), [manifest](models/day14_soft_top5_consensus/model_manifest.json).

## Validation corruption found and repaired

Independent semantic checks discovered an entirely zero-filled historical known-behavior outer-3 cue probability array and inner cue model. File-existence/archive checks had missed it. The affected isolation specialist trained without informative cue probabilities. Original affected isolation evidence measurements and new shared-model outer-3 comparisons are invalid; their historical reports remain marked provisional.

Frozen artifacts were preserved. The damaged inner model was exactly refitted with original exclusions, cue probabilities and isolation specialist rebuilt, and full/random/chronological baselines regenerated. Affected shared, MAP and family-refinement outer-3 models were refitted. Unaffected folds were copied with hash checks. Promoted new evidence results use corrected reports. The [repair manifest](models/day14_cue_repair/repair_manifest.json) verifies pool exclusions, finite normalized probabilities, durable writes, model reloads and unchanged source hashes.

Corrected full-history outer-3 isolation baseline MAP is 0.760000 versus invalid prior 0.750145. This concerns validation: production cue arrays/models passed checks, and submitted production files did not use the failed validation cache. Pair-risk and omitted-family results do not depend on it.

The scan checked all rows of 28 probability arrays and headers of 750 model files, finding only this array/model invalid under those checks. Header checks do not establish semantic correctness of every model. New prediction construction rejects non-finite/non-normalized probability inputs; uploads reject configurations explicitly marked provisional.

Corrected reports are under models/day14_repaired_shared/, models/day14_repaired_map/, models/day14_refinement/validation_cuefix.json and models/day14_repaired_top5/consensus_validation_rep_0.json. The old September 11 composite must not be cited as fully verified four-fold evidence performance.

## Research and rejected alternatives

The official discussion/page refresh found no new organizer clarification. Eleven public notebooks were inventoried and the five most recent sources inspected without execution. None supplied a verified method exceeding our score. Primary references and notebook-specific caveats are in [DAY14_RESEARCH.md](DAY14_RESEARCH.md). No external code was incorporated or private methods published.

The supplied gameplay is synthetic. Organizer documentation still requires observable behavior-specific evidence and does not establish uniform or earliest-five evidence sampling. Synthetic provenance does not remove withheld-mechanism, hidden-private-assignment, shortened-history or evidence-label uncertainty.

| Investigation | Outcome |
|---|---|
| Shared chronological LambdaRank / cross-entropy, pool perturbations and ensemble | No corrected five-view gate survivor as unrestricted evidence replacements |
| Conditional component reliability/agreement | No robust improvement; affected original isolation comparison remains provisional |
| Four history-free intrinsic ranker fits | Full-history losses across families; rejected; original isolation baseline comparison is provisional |
| Two temporal episode posterior rules | Failed across-view screens; episode activity alone does not identify planted evidence; affected original isolation comparison is provisional |
| Chronological MAP objective | Corrected isolation full gain +0.012138, random B -0.001543, interval includes zero; v48 lost publicly |
| Reduced 12.5% isolation MAP weight and top-five-only MAP | Still failed across-view screens |
| Shared model followed by family refinement | Soft full +0.009003, all four folds positive, interval [0.000601, 0.017356]; three shortened/early means negative; v49 public gain only 0.00003 |
| Single-model top-five reranking | No robust survivor; two-model soft consensus became v50 |
| Test-time averaging over within-view prefix/suffix windows | All families failed corrected robustness screen |
| Stronger v50 consensus weight | Lost in every unperturbed history view |

[Case reviews](DAY14_CASE_REVIEW.md) separate membership changes from ordering gains and context sensitivity from candidate removal. V49 full-history improvements were mostly reorderings; several shortened-history losses persisted with frozen full-history scores restricted to retained hands. These observations informed v50 but do not validate the unknown family.

## Secondary risk alternative and final selection

V51 combines **v50 evidence/behaviors** with **v47 risks** and scored **0.93876**, tying v50. It changes exactly 1,486 risk rows. Every nonrisk cell equals v50 and every risk string equals submitted v47. Its SHA-256 is c8b0f4c260f7766366ab2060bdc91849d24f0bd44d9dcf35c63c02fff582e370. The power-1.6 anomaly transform is a ranking rule, not probability calibration.

At 25x normal-pair validation weight, pair-AP gains are:

| Training scenario | Full | Short A | Short B |
|---|---:|---:|---:|
| All disclosed families | 0.000000 | -0.000140 | -0.000208 |
| Directed transfer omitted | +0.001009 | -0.006758 | -0.000147 |
| Soft play omitted | +0.209578 | +0.152062 | +0.148392 |
| Isolation omitted | -0.000014 | -0.000716 | -0.002112 |

This diversifies risk ranking while preserving v50 separately. Risk scores affect both PairAP (70% of the metric) and known-family BehaviorMAP (10%); evidence carries the remaining 20%. The alternative **fails** the across-setting gate and is not a demonstrated private improvement. Omitted soft play is a disclosed-family proxy, not the actual hidden mechanism. Earlier wording claiming the risk rule “secures private #1” was unsupported and is explicitly withdrawn here.

Kaggle permits two selected final entries. **V50 (56354034) and v51 (56354264) are now selected**, replacing old v39/v38. The UI confirmed the save and the official selected-submissions endpoint verified both IDs. [Selected entries snapshot](submissions/day14_selected_latest.json) records this state. Both entries score 0.93876 publicly; neither private score is known.

## Reproduction and artifact controls

Retained-artifact replay for v50: run .venv/Scripts/python.exe -X utf8 day14_infer_consensus.py --kind shared_soft_top5_consensus --family 2 --output submissions/submission_v50_soft_top5_consensus.csv.

This checks source models, reconstructs prior lists for all 660 scoped pairs, validates 112,540 rows and 562,700 evidence references, verifies membership preservation and enforces the immutable submitted hash. It uses retained foundation artifacts; a fresh complete retraining of every historical stage was not repeated. Risk-combination replay uses day14_prepare_final_hedge.py.

Submission management uses fresh official counts, SHA-256/validator agreement, duplicate-file/hash checks and atomic upload intent/receipt records. Root submission.csv is promoted only after a strictly higher completed public score; ties do not replace it. Final [integrity checks](models/day14_completion_integrity.json) passed for all five official scores/configurations/CSV hashes, root identity, exact v50 membership preservation, v51 component equality, corrected probability normalization, repair/source hashes and official selected IDs. No private labels, scores or competitor predictions are available. Neither 0.95 nor private #1 has been established.
