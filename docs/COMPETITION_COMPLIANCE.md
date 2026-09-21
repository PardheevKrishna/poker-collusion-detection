# Competition compliance and verification checklist

This package is prepared for the organizer's post-private-leaderboard verification process.

- [x] Public solution writeup published on Kaggle.
- [x] One public Kaggle notebook with a runnable exact generator.
- [x] Public code repository with setup and execution instructions.
- [x] Five case reviews with pair ID, hand IDs, observable behavior, and benign alternative.
- [x] Both selected submission IDs and SHA-256 values recorded.
- [x] Clean-environment replay checked locally and in the Kaggle notebook.
- [x] GitHub Actions reruns the exact replay on every push.
- [x] No private leaderboard labels or private data are included.
- [x] No ID-format, row-order, file-order, or generator-internal features are used.

The exact selected-file path is intentionally a frozen replay: the public v50 base and sparse deterministic patches are the complete inputs. A fresh historical refit would require the original large intermediate cache and hardware-specific numerical stages; claiming otherwise would be misleading. The notebook labels this boundary directly.

