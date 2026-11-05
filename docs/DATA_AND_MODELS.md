# Data and models

## Released analysis inputs

The versioned evidence ZIP contains 83 files: measured feature tables, engine
query summaries, per-position evaluation records, source metadata, and all
1,008 generated D5 boards. `artifacts/evidence_manifest.json` lists each file,
its origin, row count, and whether source-board arrays were removed.

Natural human games, engine self-play, public adversarial games, and their
perturbations are represented by numerical measurements and board-free metadata
in this download. Their raw SGF collections and D1–D4 board arrays are not
redistributed. `metadata.jsonl` is sufficient for game/fold/lead-time lookup,
but is not a replacement for full positions when running new engine queries.

The software license covers the project's original implementation and synthetic
examples; it does not relicense upstream collections or model weights. The
numerical records are shared as the project's reported experimental outputs.

## Rebuild the board streams

The original builders are retained. Run `--help` for source and output options.
For example:

```bash
python experiments/scripts/r006_r007_natural_split_build.py --run-id R006 --split-id D1_NATURAL_HUMAN_JGDB --source jgdb --target-count 5000 --output-root experiments/results/R006_NATURAL_HUMAN_SPLIT --fail-on-incomplete
python experiments/scripts/r010_topology_stress_generation.py --per-family-count 144 --output-root experiments/results/R010_TOPOLOGY_STRESS_SPLIT --fail-on-issue
```

D2 uses the dated KataGo archive URLs recorded in its source manifest. D3 uses
20 games from the Go Attack example site and lead windows 80, 40, 20, 10, 5, 1,
and 0. D4 generates 7,000 perturbation pairs from the natural boards. Source
URLs and historic sampling details are preserved in the split reports and
`experiments/configs/pricai2026.json`. Upstream availability and collection
terms must be checked when reacquiring data; the historical website's bundled
JavaScript filename may change.

## Engine checkpoints

| Use | Recorded model identifier |
|---|---|
| Main KataGo b28 | `kata1-b28c512nbt-s12763923712-d5805955894.bin.gz` |
| KataGo b18 sensitivity | `kata1-b18c384nbt-s9131461376-d4087399203.bin.gz` |
| KataGo b40 sensitivity | `kata1-b40c256-s8326889472-d2025687566.bin.gz` |
| Published-attack victim | `kata1-b40c256-s11840935168-d2898845681.bin.gz` |
| Leela Zero | `0e9ea880fd3c4444695e8ff4b8a36310d2c03f7c858cadd37af3b76df1d1d15f.gz` |
| ELF OpenGo | `pretrained-go-19x19-v2.bin` |

Obtain weights through [KataGo](https://github.com/lightvector/KataGo),
[Go Attack's reproduction instructions](https://github.com/AlignmentResearch/go_attack/tree/a12008625445fb7a123d0d02cebbd11d6d416fd8/sabaki),
and the [ELF](https://github.com/AlignmentResearch/ELF) and
[Leela Zero](https://github.com/AlignmentResearch/leela-zero) upstream projects.
No engine, weight, CUDA library, or third-party game archive is included here.
