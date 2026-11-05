# Release validation

Checked for v0.2.0 on Windows / Python 3.12.14, using the pinned release
dependencies. This records the checks performed, not an end-to-end GPU rerun.

- The restored source passed 111 unit tests.
- All 83 camera-ready numeric checks passed against the archived measurements.
- All 130 family-held-out detector evaluations ran from the released features.
- The strict-control, nearest-neighbor-matched, and mechanism-held-out analyses
  ran (39, 26, and 8 result rows respectively).
- S09 refitting reproduced agreement 0.8470279720, mean visits 874.9650350,
  and trigger rate 0.8059440560 over all 1,144 selected positions.
- Early-warning refitting reproduced warnings on all 20 held-out games and
  mean earliest sampled lead 80 at both recall thresholds.
- Paired significance was recomputed from the archived per-position choices.
- The D5 generator reproduced all 1,008 archived records exactly.
- Full-data inference-model fitting and scoring a synthetic 19x19 board ran.
- All five supplied engine patches passed application checks against their
  pinned upstream source (using `git apply --recount`).

Some current-environment detection and control refits differ from historical
results. For example:

| Mean metric | Archived paper run | Release-environment refit |
|---|---:|---:|
| B3 AUROC | 0.9577 | 0.9577 |
| B3 AUPRC | 0.9044 | 0.9045 |
| B5 AUROC | 0.9554 | 0.9556 |
| B5 AUPRC | 0.9250 | 0.9375 |
| Strict-control AUROC | 0.9587 | 0.9593 |
| Strict-control AUPRC | 0.9258 | 0.9331 |

The original complete dependency lock is unavailable, and the cause of these
differences has not been isolated. The code and archived results are retained
without tuning parameters or changing records to enforce agreement.

The full 16-configuration sweep, normal-aware sweep, FPR-capped refit suite,
large engine queries, and complete attack match were not rerun for this release.
Their original implementations and measured records are provided. Fresh engine
execution also requires upstream datasets, binaries, and weights.
