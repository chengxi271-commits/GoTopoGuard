# Reproducibility

## Standalone extractor

Python 3.10+ is required. The runtime has no third-party dependencies.
Install from the tagged source tree with `python -m pip install .`, then run:

```bash
python -m unittest discover -s tests -v
python examples/minimal_features.py
python -m gotopoguard examples/board.json
```

Both example commands produce the JSON recorded in
`examples/expected_features.json`. Tests compare parsed JSON values rather
than relying on whitespace. They cover groups, deduplicated liberties,
enclosed and edge-touching regions, mixed-color boundaries, articulation
points, invalid row encodings, empty 19x19 boards, and color/reflection
transformations.

For an installed-package check, change to a directory outside this checkout
and run `python -c "import gotopoguard; print(gotopoguard.__version__)"`.
The source-tree tests also check development imports; they alone do not
establish that a built distribution contains all required files.

## Paper experiments

v0.1.0 does not include trained models, experiment configurations, benchmark
datasets, engine-query pipelines, or aggregate paper results. It therefore
does not provide commands to reproduce paper tables. In particular, a
successful extractor test is not evidence for detection accuracy, warning
lead time, or playing strength.

The [component mapping](RESULT_MAPPING.md) records this boundary. Later
experimental releases need documented input provenance, engine/network
versions, configurations, and a tested result-to-paper mapping. No release
date is promised for those components.
