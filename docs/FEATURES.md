# Feature reference

The extractor operates on a four-neighbor grid: diagonally touching stones
are not connected. Construct a board with `parse_board(rows)` before passing
it to the other functions. Rows must be nonempty and equal-length, with
cells drawn from `B`, `W`, and `.`. No side to move, move history, ko state,
captures, or rule configuration is represented.

## Detailed structures

- `stone_groups(board, color)`, for `color="B"` or `"W"`, returns connected
  same-color components. A group's liberties are the **distinct** adjacent
  empty intersections, not the number of stone-to-empty edges. Each record
  contains `color`, `stones`, `size`, `liberties`, and `liberty_count`.
- `empty_regions(board)` returns connected empty components. Each record
  contains `points`, `size`, `edge_reachable`, and `boundary_colors`.
  `edge_reachable` is true when the component contains any board-edge cell.
  Boundary colors are the colors of stones adjacent to the component.
- `stone_articulation_points(board, color)` returns stones whose removal
  splits their original same-color component into more than one component.
  This graph operation does not play a move or apply captures.
- All coordinates are zero-based `(column, row)` tuples, sorted
  lexicographically for deterministic output. JSON encodes tuples as arrays.

## Aggregate features

`extract_topology_features(board)` returns 14 integer-valued fields:

| Field | Definition |
|---|---|
| `board_width`, `board_height` | Number of columns and rows |
| `black_group_count`, `white_group_count` | Number of same-color connected components |
| `black_max_group_size`, `white_max_group_size` | Largest stone count among that color's groups; zero if absent |
| `black_max_liberty_count`, `white_max_liberty_count` | Largest distinct-liberty count among that color's groups; zero if absent |
| `empty_region_count` | Number of connected empty components |
| `enclosed_empty_region_count` | Empty components that do not reach the board edge |
| `black_bounded_empty_region_count`, `white_bounded_empty_region_count` | Enclosed components whose boundary-color set is exactly the indicated color |
| `black_articulation_count`, `white_articulation_count` | Number of articulation stones of each color |

A mixed-color enclosed region contributes to `enclosed_empty_region_count`
but not either single-color bounded count. Edge-touching empty regions are
not counted as enclosed, even if surrounded on their other sides by one
color. These definitions describe grid connectivity, not Go territory.

## Implementation scope

The reference implementation uses explicit component traversal. For each
candidate articulation stone it recomputes connectivity after removal;
this favors transparency over asymptotic speed. No latency guarantee is
attached to the release, and the paper's measured timing is not a benchmark
of every machine or board supplied by users.

This module exposes board-level features only. It does not implement shared-
liberty summaries, candidate-move effects, search statistics, symmetry
queries, a trained risk score, or guarded search.
