"""Breadth-First Search — an uninformed search strategy.

BFS explores the campus graph level by level and therefore returns the path
with the fewest hops (not necessarily the cheapest one).
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

SearchResult = Tuple[Optional[List[str]], float, int]


def path_cost(graph: Dict[str, Dict[str, float]], path: List[str]) -> float:
    """Total edge weight of a path."""
    return sum(graph[path[i]][path[i + 1]] for i in range(len(path) - 1))


def bfs(graph: Dict[str, Dict[str, float]], start: str, goal: str) -> SearchResult:
    """Return (path, cost, nodes_expanded). Path is None when unreachable."""
    if start not in graph or goal not in graph:
        return None, 0.0, 0
    if start == goal:
        return [start], 0.0, 1

    frontier: deque[List[str]] = deque([[start]])
    visited = {start}
    expanded = 0

    while frontier:
        path = frontier.popleft()
        node = path[-1]
        expanded += 1

        for neighbour in graph.get(node, {}):
            if neighbour in visited:
                continue
            new_path = path + [neighbour]
            if neighbour == goal:
                return new_path, path_cost(graph, new_path), expanded
            visited.add(neighbour)
            frontier.append(new_path)

    return None, 0.0, expanded
