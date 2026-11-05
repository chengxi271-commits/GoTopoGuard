# Implementation provenance

The release was assembled from the original PRICAI project only. It does not
include subsequent journal work, submission correspondence, publication forms,
private working notes, or the original Git history.

An earlier project snapshot at commit
`a72959587f553224ea8dc0b04823b4e913042dec` contained 33 recoverable source,
configuration, and test files. These belong to the earlier Stage 3–6 prototype.
The final paper's implementation instead resides in the later R-series scripts
retained in the working tree. The earlier `topoguard.py` was therefore not
substituted for the final S09 implementation.

The release restores the final R001–R017 utilities, R019R1 attack reproduction,
R020 cross-engine runners, and the final strict-control/statistical analyses.
Recorded source paths are listed in [recovery_sources.json](recovery_sources.json).

Packaging changes are limited to:

- exposing existing model helpers without importing unused live-game policies;
- extracting numerical R022 routines from the old manuscript-generation script;
- adding download, scoring, and reproduction entry points;
- replacing ELF-specific workstation paths with the caller's selected Python
  interpreter and inherited library environment;
- removing local path prefixes from released measurements and excluding
  non-paper exploratory game summaries;
- separating board-free metadata from raw third-party board/game collections.

The topology algorithm, training rules, labels, split functions, seeds, and
recorded numeric results were not rewritten to obtain better results.
