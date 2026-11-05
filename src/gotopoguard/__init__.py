# Copyright 2026 Yixi Chen
# SPDX-License-Identifier: Apache-2.0

"""GoTopoGuard reference utilities."""

from .topology_features import (
    empty_regions,
    extract_topology_features,
    parse_board,
    stone_articulation_points,
    stone_groups,
)

__all__ = [
    "empty_regions",
    "extract_topology_features",
    "parse_board",
    "stone_articulation_points",
    "stone_groups",
]


__version__ = "0.1.0"
