# GoTopoGuard

### Topology-Aware Risk Monitoring for Rare Structural Failures in Neural Go Engines

**PRICAI 2026** · [Yixi Chen](https://orcid.org/0009-0004-7084-0395) · Sun Yat-sen University

[English](README.md) | [简体中文](README.zh-CN.md) | [繁體中文](README.zh-TW.md)

[Code & data](https://github.com/chengxi271-commits/GoTopoGuard/releases/tag/v0.2.0) ·
[Reproduction guide](docs/REPRODUCIBILITY.md) · [Feature reference](docs/FEATURES.md) ·
[Citation](#citation)

**What does a strong Go engine miss when a board becomes structurally unusual?**
GoTopoGuard examines this question through connected stone groups, liberties,
enclosed empty regions, and other board-graph features. These signals feed
TopoRisk, an external risk monitor, and a guarded-search calibration procedure
without retraining the underlying Go engine.

This repository contains the PRICAI implementation and recorded experiment
artifacts: topology extraction, benchmark construction, detection and early
warning, search-budget calibration, and budget/network/cross-engine analyses.
The analysis bundle lets you work with the original measured features without
first downloading engines or rerunning their searches.

## Getting started

Python 3.10+ is required. Python 3.12 is recommended for the tested analysis
environment supplied with this release.

```bash
git clone https://github.com/chengxi271-commits/GoTopoGuard.git
cd GoTopoGuard
git checkout v0.2.0
python -m pip install ".[reproduce]"
python examples/minimal_features.py
python scripts/reproduce.py --suite summary
```

The last command checks the paper's reported numbers against the included
records and writes JSON/CSV summaries to `reproduced/summary/`. It needs no GPU
or additional download. For the topology extractor alone, `pip install .` is
sufficient; the extractor has no third-party runtime dependencies.

### Use the topology features

```python
from gotopoguard import extract_topology_features, parse_board

board = parse_board([".....", ".BBB.", ".B.B.", ".BBB.", "....."])
features = extract_topology_features(board)
print(features["enclosed_empty_region_count"])  # 1
print(features["black_max_liberty_count"])      # 13
```

`B`, `W`, and `.` denote black stones, white stones, and empty intersections.
The API also exposes group membership, exact liberty sets, empty-region
boundaries, and articulation points. See the [feature reference](docs/FEATURES.md).

## Reproduce the analyses

Download the **44.8 MB** measured-feature and result bundle, then choose an analysis:

```bash
python scripts/fetch_artifacts.py
python scripts/reproduce.py --suite guard-s09
python scripts/reproduce.py --suite detection
python scripts/reproduce.py --suite early-warning
python scripts/reproduce.py --suite controls
```

These commands refit models or recompute statistics from the saved measurements;
they do not launch Go engines. Historical outputs remain separate from newly
generated results. Some refitted detection/control metrics differ in the release
environment; the original dependency lock was not recovered. The
[reproduction notes](docs/REPRODUCIBILITY.md) describe this distinction and the
remaining suites, including operating points, normal-aware calibration, and
paired significance tests.

For local board scoring, fit the original full-data diagnostic models:

```bash
python scripts/train_models.py
python scripts/score_board.py examples/board19.json
```

The example is a synthetic 19×19 input, not a benchmark case. These newly fitted
models are for inference; their training-set outputs are not held-out results.

## Results at a glance

Selected results reported in the PRICAI paper:

| Experiment | Result |
|---|---|
| B3 board + topology detection, family-held-out | AUROC **0.9577**, AUPRC **0.9044** |
| B5 with search and symmetry features | AUPRC **0.9250** |
| Early warning on 20 held-out adversarial games | All games warned; mean earliest sampled lead **80 moves** |
| S09 vs. uniform 1024 visits, on 1,144 stress positions | Agreement **0.8470 vs. 0.8427**; mean visits **875 vs. 1,024** |

Strict controls match or exceed the topology-enhanced detector in aggregate,
and the S09–uniform difference is not statistically resolved
(McNemar *p* = 0.4996). These results support diagnostic monitoring and
search-policy calibration, not a demonstrated live-game defense.

## Repository guide

| Location | Contents |
|---|---|
| `src/gotopoguard/` | Standalone feature API and command-line interface |
| `experiments/scripts/` | Original R-series data builders, model training/evaluation, and engine-query runners |
| `analysis/` | Strict controls, mechanism holdouts, and paired statistics |
| `experiments/configs/` | Recorded paper settings and pinned upstream versions |
| `artifacts/` | Compact historical results and the downloadable evidence manifest |
| `scripts/` | Download, reproduction, training, and scoring entry points |
| `tests/` | Board, replay, model, and evaluation tests |

To rebuild inputs or run the engines yourself, start with
[data and models](docs/DATA_AND_MODELS.md) and [engine setup](docs/ENGINES.md).
External binaries, weights, and third-party SGF collections are obtained from
their upstream sources rather than mirrored here.

## Citation

The paper is accepted at PRICAI 2026. This provisional citation will be updated
when the official proceedings metadata is verified:

```bibtex
@misc{chen2026gotopoguard,
  author = {Chen, Yixi},
  title = {{GoTopoGuard}: Topology-Aware Risk Monitoring for Rare Structural Failures in Neural Go Engines},
  year = {2026},
  note = {Accepted at PRICAI 2026},
  url = {https://github.com/chengxi271-commits/GoTopoGuard}
}
```

Please also identify the release or commit used. Software metadata is in
[`CITATION.cff`](CITATION.cff).

## License and contact

Original code is [Apache-2.0](LICENSE). Engine patches retain their upstream
licenses; see [third-party notices](docs/THIRD_PARTY_NOTICES.md). The paper and
external datasets/models have separate terms.

Questions and reproducible bug reports are welcome in
[Issues](https://github.com/chengxi271-commits/GoTopoGuard/issues).
For research correspondence: [chengxi271@gmail.com](mailto:chengxi271@gmail.com).
