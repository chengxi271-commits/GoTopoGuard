# GoTopoGuard

**Topology-Aware Risk Monitoring for Rare Structural Failures in Neural Go Engines**

PRICAI 2026 · Yixi Chen · [ORCID](https://orcid.org/0009-0004-7084-0395)

[Quickstart](#quickstart) · [Feature reference](docs/FEATURES.md) ·
[Reproducibility](docs/REPRODUCIBILITY.md) · [Citation](#citation) · [Apache-2.0](LICENSE)

Neural Go engines can be strong at ordinary play yet struggle with unusual
board structures. GoTopoGuard studies topology-aware diagnostic signals:
connected stone groups, their liberties, enclosed empty regions, and cut
points in the stone graph.

**v0.1.0 releases the standalone reference topology extractor.** It runs on
the Python standard library and needs no GPU, Go engine, or model download.
The trained TopoRisk scorer, guarded-search interface, benchmark builders,
and full paper experiments are **not included in this version**.

## Quickstart

Requires Python 3.10 or later. From a terminal:

```bash
git clone https://github.com/chengxi271-commits/GoTopoGuard.git
cd GoTopoGuard
git checkout v0.1.0
python -m pip install .
python examples/minimal_features.py
python -m unittest discover -s tests -v
```

The example describes an eight-stone black ring around one empty point:

```text
.....
.BBB.
.B.B.
.BBB.
.....
```

Expected features include one black group, 13 distinct liberties, one
enclosed empty region, and no black articulation points. The complete
expected output is in [`examples/expected_features.json`](examples/expected_features.json).

### Python API

```python
from gotopoguard import extract_topology_features, parse_board, stone_groups

board = parse_board([".....", ".BBB.", ".B.B.", ".BBB.", "....."])
features = extract_topology_features(board)
print(features["enclosed_empty_region_count"])  # 1
print(stone_groups(board, "B")[0]["liberty_count"])  # 13
```

`B`, `W`, and `.` denote black stones, white stones, and empty intersections.
Coordinates are zero-based `(column, row)`. Rectangular boards are accepted
for small examples; normal square Go boards work the same way. Parsing
checks the representation, not whether a position is reachable under Go rules.

### Command line

```bash
python -m gotopoguard examples/board.json
```

The input is a JSON array of board rows. The output contains the board and
its 14 aggregate features. Detailed groups, liberties, empty regions, and
articulation coordinates are available through the Python API.

## What is released?

| Component | v0.1.0 |
|---|---|
| Connected groups and exact distinct-liberty sets | Included |
| Empty regions, edge reachability, and boundary colors | Included |
| Stone-graph articulation points and 14 aggregate features | Included |
| API, command-line example, expected output, and unit tests | Included |
| Trained TopoRisk model and candidate/search/metamorphic features | Not yet released |
| Guarded-search calibration and evaluation scripts | Not yet released |
| GoTopoBench datasets and full paper reproduction | Not yet released |

The extractor is a component of the paper's framework, not a complete Go
engine or a failure classifier. An enclosed region is a geometric property,
not a claim about territory, life and death, or strategic risk. See the
[feature definitions](docs/FEATURES.md) for the precise semantics.

## Reproducing the paper

This release supports reproducing the extractor's output on supplied board
states. It does not reproduce the paper's AUROC/AUPRC, early-warning,
guarded-search, or cross-engine results. Passing the example is a software
check, not a replication of those experiments.

The [reproduction guide](docs/REPRODUCIBILITY.md) and
[artifact mapping](docs/RESULT_MAPPING.md) distinguish what is available
from the remaining experimental material. Engines, weights, and third-party
game archives are not bundled; see [data and models](docs/DATA_AND_MODELS.md).

In interpreting the paper, benchmark labels are provenance-defined rather
than causal ground truth; high-visit KataGo outputs are same-engine
references; cross-engine disagreement does not identify the correct move.
Guarded-search calibration is not a demonstration of live-game defense.

## Citation

The accompanying paper has been accepted at PRICAI 2026. Until its official
proceedings metadata is verified, the following citation deliberately
omits a DOI, volume, and page range:

```bibtex
@misc{chen2026gotopoguard,
  author = {Chen, Yixi},
  title = {{GoTopoGuard}: Topology-Aware Risk Monitoring for Rare Structural Failures in Neural Go Engines},
  year = {2026},
  note = {Accepted at PRICAI 2026},
  url = {https://github.com/chengxi271-commits/GoTopoGuard}
}
```

[`CITATION.cff`](CITATION.cff) identifies the software release. When reporting
use of this implementation, include the version or commit as well.

## License and contact

Original code, tests, synthetic examples, and repository documentation are
released under [Apache-2.0](LICENSE). The paper and third-party engines,
models, and datasets retain their separate terms and are not distributed here.

For reproducible bugs or technical questions, please
[open an issue](https://github.com/chengxi271-commits/GoTopoGuard/issues).
Include the board input, Python version, and observed versus expected output.
