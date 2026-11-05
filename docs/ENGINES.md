# Engine setup and regeneration

Offline analysis does not require this setup. These instructions are for
regenerating the original engine measurements on a suitable machine.

## Versions

| Component | Recorded version |
|---|---|
| Main KataGo analysis engine | v1.16.5 |
| Go Attack | `a12008625445fb7a123d0d02cebbd11d6d416fd8` |
| Go Attack's KataGo-custom | `960cf2c6131b1145465a6539cb7581dd9430bc80` |
| Go Attack's KataGo-raw | `f2dc582f98a79fefeb11b2c37de7db0905318f4f` |
| AlignmentResearch ELF | `6b6c0bc3fd3b58c88e85836cbac923931e3e5118` |
| AlignmentResearch Leela Zero | `68386d6334641ba345a225cf3cf77ce39f646771` |

Clone Go Attack at the listed commit and initialize its submodules. Follow its
upstream build instructions rather than substituting the current stock KataGo
binary for the custom adversarial-search engine. Checkpoint identifiers are
listed in [DATA_AND_MODELS.md](DATA_AND_MODELS.md).

## Analysis queries

After rebuilding D1–D5, the R011 runner accepts explicit binary/config/model
paths; its shipped defaults preserve the original Windows development layout.
For another environment, pass your paths explicitly:

```bash
python experiments/scripts/r011_katago_search_features.py --katago-exe /path/to/katago --config /path/to/analysis.cfg --model /path/to/b28.bin.gz --visits 64,256 --output-root experiments/results/R011_KATAGO_SEARCH_FEATURES --fail-on-incomplete
```

R011 symmetry uses seven nonidentity transforms; R014 collects 1024/4096-visit
calibration queries; R016 compares 1024/8192 visits on D3/D4/D5; R017 evaluates
b18 and b40 at 1024 visits. Each script exposes its options through `--help`.
The original full remote runtime/configuration was not recovered as one frozen
environment, so a fresh run should record its own backend and configuration.

## Cross-engine patches

The R020 runner uses `gotopo-set_position` to load a board directly. Apply the
included patches to the exact source commits above, from the upstream checkout:

```bash
# ELF: apply in this order; /path/to/GoTopoGuard is your release checkout.
git apply --recount /path/to/GoTopoGuard/experiments/patches/elf_opengo_gcc11_tbb_compat.patch
git apply --recount /path/to/GoTopoGuard/experiments/patches/elf_opengo_gtp_disable_strict_model_version.patch
git apply --recount /path/to/GoTopoGuard/experiments/patches/elf_opengo_gotopo_set_position.patch

# Leela Zero, in its own checkout:
git apply --recount /path/to/GoTopoGuard/experiments/patches/leela_zero_gotopo_set_position.patch
```

The historical patches need `--recount` for their hand-written hunk lengths.
Their application was checked against the pinned source. This does not imply
that every modern compiler/CUDA combination was rebuilt and tested.

The small `katago_custom_opencl_gcc11.patch` applies to the pinned custom KataGo
tree using the same command. It is a build-compatibility patch, not a change to
the search algorithm.

For ELF, pass `--elf-python` pointing to the interpreter for your ELF install
and set its shared-library environment before starting R020. No workstation-
specific Conda paths are injected. The recorded comparisons used Leela Zero at
512 visits and ELF at 1,600 rollouts; compare moves, not cross-engine value scales.

## Published cyclic attack

`r019r1_main_adversary_evaluation.py` generates the custom-engine match config.
Use its `--help` and explicit binary/model paths; `--dry-run` prints the planned
command after prerequisite checks. The recorded setting used 600 adversary
visits, AMCTS-S, and the cp505/b40 victim variants. The saved run stopped after
1,002 completed games (984 adversary wins); this is the observed run, not a
claim that every fresh 1,000-per-matchup command terminates at 1,002 games.

The original raw SGF files for that run were not recovered in this checkout.
The release contains the 1,002 parsed game outcomes and the original parser,
configuration generator, and runner. No large engine experiment is launched by
the quickstart or by `scripts/reproduce.py`.
