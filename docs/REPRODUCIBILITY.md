# Reproducing the PRICAI analyses

## Choose a starting point

| Goal | Command | Inputs |
|---|---|---|
| Inspect features | `python examples/minimal_features.py` | Included synthetic board |
| Check paper numbers | `python scripts/reproduce.py --suite summary` | Included compact records |
| Refit S09 | `python scripts/reproduce.py --suite guard-s09` | Downloadable measured-feature bundle |
| Refit detection baselines | `python scripts/reproduce.py --suite detection` | Same bundle |
| Refit 20-game early warning | `python scripts/reproduce.py --suite early-warning` | Same bundle |
| Refit strict/matched/mechanism controls | `python scripts/reproduce.py --suite controls` | Same bundle |
| Recompute paired test and bootstrap | `python scripts/reproduce.py --suite significance` | Same bundle |
| Recompute score/win-rate sensitivity | `python scripts/reproduce.py --suite stress-records` | Same bundle |
| Full 16-configuration guard sweep | `python scripts/reproduce.py --suite guard-sweep` | Same bundle |
| FPR-capped operating points | `python scripts/reproduce.py --suite operating-points` | Same bundle |
| Normal-aware calibration | `python scripts/reproduce.py --suite normal-aware` | Same bundle |

Run commands from the repository root. Install analysis dependencies with
`python -m pip install ".[reproduce]"`, then run `python scripts/fetch_artifacts.py`
once. Alternatively, download the evidence ZIP from the release page and pass
`--archive PATH` to the fetch script. Its default extraction destination is
`data/pricai2026/`; `reproduce.py --data-root PATH` supports another location.

Results are written to `reproduced/<suite>/`. `--output PATH` changes that parent
directory. Saved historical records are never used as the destination of these
wrappers. Numerical library and tree-model parallelism defaults to one thread;
set `GOTOPOGUARD_MODEL_N_JOBS` deliberately if you need more tree parallelism.

## Historical measurements and new fits

`artifacts/compact/` holds the original measured results, with local path
prefixes removed. `summary` recomputes fold means, paired counts, the exact
McNemar test, and the bootstrap interval, and checks 83 displayed quantities
against the camera-ready. It does not train models or regenerate engine queries.

The feature bundle supports fitting the original model implementations without
repeating expensive Go searches. On the release environment, S09 reproduces the
archived agreement/visit figures, the D5 generator reproduces all 1,008 records,
and early warning reproduces the 20-game/80-move result. Detection and strict-
control refits run successfully but some metrics differ from the archived run.
For example, B5 mean AUPRC is 0.9375 in the release test versus the paper's 0.9250.
The original complete dependency lock was not recovered; library/platform
differences are a possible explanation, not an established cause. Archived
values are preserved unchanged. We do not tune parameters to force a match.

The release analysis environment is pinned in `requirements-reproduce.txt`.
It is a tested environment, **not a recovered historical lockfile**.
See [validation](VALIDATION.md) for the checks actually performed.

## Models and evaluation protocols

The original study fits models within each evaluation protocol rather than
using one universal checkpoint. `train_models.py` exposes the original R020
full-data fitting routine for local inference and saves a joblib bundle plus
training metadata. These fresh inference models are not the held-out models
used for detection/early-warning scores. Load only trusted joblib files.

`guard-s09` validates the selected configuration; `guard-sweep` includes all 16
original choices. S09 is calibrated against a 4096-visit same-engine reference.
It is not an independently verified optimal policy or a live-game defense.

## Regenerating engine measurements

Data builders, query runners, upstream pins, and patches are included. See
[ENGINES.md](ENGINES.md) and [DATA_AND_MODELS.md](DATA_AND_MODELS.md). Fresh engine
execution requires upstream datasets, binaries, weights, and a suitable local
or remote environment. The release was checked using saved measurements; the
large GPU searches and 1,002-game attack run were not repeated for publication.

A later refit or engine run should be reported with its own environment and
results rather than replacing the historical measurements silently.
