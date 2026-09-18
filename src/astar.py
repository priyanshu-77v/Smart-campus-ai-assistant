"""A* Search — an informed search strategy.

f(n) = g(n) + h(n) where g is the cost so far and h is the straight-line
distance heuristic supplied by the knowledge base.
"""

from __future__ import annotations

import heapq
from typing import Callable, Dict, List, Optional, Tuple

SearchResult = Tuple[Optional[List[str]], float, int]


def astar(
    graph: Dict[str, Dict[str, float]],
    start: str,
    goal: str,
    heuristic: Optional[Callable[[str, str], float]] = None,
) -> SearchResult:
    """Return (path, cost, nodes_expanded). Path is None when unreachable."""
    if start not in graph or goal not in graph:
        return None, 0.0, 0

    h = heuristic or (lambda a, b: 0.0)

    open_heap: List[Tuple[float, float, str]] = [(h(start, goal), 0.0, start)]
    came_from: Dict[str, str] = {}
    best_cost: Dict[str, float] = {start: 0.0}
    closed: set[str] = set()
    expanded = 0

    while open_heap:
        _f, g, node = heapq.heappop(open_heap)
        if node in closed:
            continue
        closed.add(node)
        expanded += 1

        if node == goal:
            path = [node]
            while path[-1] in came_from:
                path.append(came_from[path[-1]])
            path.reverse()
            return path, g, expanded

        for neighbour, weight in graph.get(node, {}).items():
            tentative = g + weight
            if tentative < best_cost.get(neighbour, float("inf")):
                best_cost[neighbour] = tentative
                came_from[neighbour] = node
                heapq.heappush(
                    open_heap, (tentative + h(neighbour, goal), tentative, neighbour)
                )

    return None, 0.0, expanded
