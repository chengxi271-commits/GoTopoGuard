"""Deterministic board-topology features for GoTopoBench."""

from __future__ import annotations


def parse_board(rows: list[str]) -> list[list[str]]:
    if not rows:
        raise ValueError("board must have at least one row")
    width = len(rows[0])
    if width == 0:
        raise ValueError("board rows must be nonempty")
    board = []
    for row in rows:
        if len(row) != width:
            raise ValueError("all board rows must have the same width")
        cells = list(row)
        invalid = [cell for cell in cells if cell not in {".", "B", "W"}]
        if invalid:
            raise ValueError(f"invalid board cell {invalid[0]!r}")
        board.append(cells)
    return board


def board_size(board: list[list[str]]) -> tuple[int, int]:
    return len(board[0]), len(board)


def neighbors(point: tuple[int, int], board: list[list[str]]):
    col, row = point
    width, height = board_size(board)
    if col > 0:
        yield col - 1, row
    if col + 1 < width:
        yield col + 1, row
    if row > 0:
        yield col, row - 1
    if row + 1 < height:
        yield col, row + 1


def _component(board: list[list[str]], start: tuple[int, int], target: str) -> set[tuple[int, int]]:
    seen = {start}
    stack = [start]
    while stack:
        point = stack.pop()
        for neighbor in neighbors(point, board):
            col, row = neighbor
            if neighbor not in seen and board[row][col] == target:
                seen.add(neighbor)
                stack.append(neighbor)
    return seen


def stone_groups(board: list[list[str]], color: str) -> list[dict]:
    seen = set()
    groups = []
    for row, cells in enumerate(board):
        for col, cell in enumerate(cells):
            point = (col, row)
            if cell != color or point in seen:
                continue
            stones = _component(board, point, color)
            seen.update(stones)
            liberties = set()
            for stone in stones:
                for neighbor in neighbors(stone, board):
                    ncol, nrow = neighbor
                    if board[nrow][ncol] == ".":
                        liberties.add(neighbor)
            groups.append(
                {
                    "color": color,
                    "stones": sorted(stones),
                    "size": len(stones),
                    "liberties": sorted(liberties),
                    "liberty_count": len(liberties),
                }
            )
    return sorted(groups, key=lambda group: group["stones"][0])


def empty_regions(board: list[list[str]]) -> list[dict]:
    seen = set()
    regions = []
    width, height = board_size(board)
    for row, cells in enumerate(board):
        for col, cell in enumerate(cells):
            point = (col, row)
            if cell != "." or point in seen:
                continue
            empties = _component(board, point, ".")
            seen.update(empties)
            boundary_colors = set()
            edge_reachable = False
            for empty in empties:
                ecol, erow = empty
                if ecol == 0 or erow == 0 or ecol == width - 1 or erow == height - 1:
                    edge_reachable = True
                for neighbor in neighbors(empty, board):
                    ncol, nrow = neighbor
                    stone = board[nrow][ncol]
                    if stone in {"B", "W"}:
                        boundary_colors.add(stone)
            regions.append(
                {
                    "points": sorted(empties),
                    "size": len(empties),
                    "edge_reachable": edge_reachable,
                    "boundary_colors": sorted(boundary_colors),
                }
            )
    return sorted(regions, key=lambda region: region["points"][0])


def _component_count_without(points: set[tuple[int, int]], removed: tuple[int, int], board: list[list[str]]) -> int:
    remaining = set(points)
    remaining.remove(removed)
    if not remaining:
        return 0
    components = 0
    while remaining:
        start = next(iter(remaining))
        components += 1
        stack = [start]
        remaining.remove(start)
        while stack:
            point = stack.pop()
            for neighbor in neighbors(point, board):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
    return components


def stone_articulation_points(board: list[list[str]], color: str) -> list[tuple[int, int]]:
    articulations = set()
    for group in stone_groups(board, color):
        stones = set(group["stones"])
        if len(stones) <= 2:
            continue
        for stone in stones:
            if _component_count_without(stones, stone, board) > 1:
                articulations.add(stone)
    return sorted(articulations)


def extract_topology_features(board: list[list[str]]) -> dict:
    black_groups = stone_groups(board, "B")
    white_groups = stone_groups(board, "W")
    regions = empty_regions(board)
    enclosed_regions = [region for region in regions if not region["edge_reachable"]]
    black_articulations = stone_articulation_points(board, "B")
    white_articulations = stone_articulation_points(board, "W")
    return {
        "board_width": board_size(board)[0],
        "board_height": board_size(board)[1],
        "black_group_count": len(black_groups),
        "white_group_count": len(white_groups),
        "black_max_group_size": max((group["size"] for group in black_groups), default=0),
        "white_max_group_size": max((group["size"] for group in white_groups), default=0),
        "black_max_liberty_count": max((group["liberty_count"] for group in black_groups), default=0),
        "white_max_liberty_count": max((group["liberty_count"] for group in white_groups), default=0),
        "empty_region_count": len(regions),
        "enclosed_empty_region_count": len(enclosed_regions),
        "black_bounded_empty_region_count": sum(1 for region in enclosed_regions if region["boundary_colors"] == ["B"]),
        "white_bounded_empty_region_count": sum(1 for region in enclosed_regions if region["boundary_colors"] == ["W"]),
        "black_articulation_count": len(black_articulations),
        "white_articulation_count": len(white_articulations),
    }
