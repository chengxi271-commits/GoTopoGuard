# Paper-to-code map

| Paper component | Implementation | Recorded artifacts |
|---|---|---|
| Natural human/engine streams | `r006_r007_natural_split_build.py` | R006/R007 reports and board-free metadata |
| Adversarial replay windows | `r008_adversarial_replay_split.py` | R008 metadata/source URLs |
| One-stone perturbations | `r009_perturbation_split.py` | R009 metadata/report |
| Topology stress generators | `r010_topology_stress_generation.py` | R010 full synthetic positions |
| Board topology | `src/gotopobench/topology_features.py` and `src/gotopoguard/` | R011 features |
| Detection and search/symmetry inputs | R011 scripts | R011 feature tables and fold metrics |
| Early warning | `r012_lead_time_evaluation.py` | R012 records/report |
| Natural FPR and operating points | R013 scripts | R013 records/reports |
| Guarded-search calibration | R014/R014B/R014C scripts | Budget features, per-position choices, summaries |
| Normal-aware calibration | R015/R015B scripts | Natural-budget features, decisions, summaries |
| Budget/network stress | R016/R017 scripts | Query features and stability records |
| Published cyclic attack | R019R1 scripts | 1,002 per-game result records; original SGF files not bundled |
| Cross-engine comparisons | R020 scripts and upstream patches | Leela/ELF per-position results |
| Strict controls and significance | `analysis/r022_controls.py` | R022 enriched features, folds, and paired outcomes |

All R-series script paths above are under `experiments/scripts/`. The evidence
archive preserves the corresponding `experiments/results/` layout. Compact
records are also included in Git under `artifacts/compact/`.
