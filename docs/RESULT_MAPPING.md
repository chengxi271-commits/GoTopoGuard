# Paper components and released artifacts

| Paper component | Available in v0.1.0 | What it supports |
|---|---|---|
| Board-graph topology | `src/gotopoguard/topology_features.py` | Deterministic group, liberty, region, and articulation features |
| Extractor examples | `examples/`, `tests/` | Software checks on synthetic board states |
| Natural/adversarial/stress benchmark construction | No | No dataset-level reproduction claim |
| TopoRisk detection and strict-control comparisons | No | No detection-table reproduction claim |
| Search and metamorphic features | No | No engine-query reproduction claim |
| Early warning and operating-point evaluation | No | No temporal-evaluation reproduction claim |
| Guarded-search calibration | No | No guard-policy reproduction claim |
| Budget, network, attack, and cross-engine analyses | No | No stress-analysis reproduction claim |

The example's expected JSON is a test fixture, not an experimental result.
The public API describes the released subset precisely in [FEATURES.md](FEATURES.md).
