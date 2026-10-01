"""Group skeleton samples into full branches without straightening their paths."""

import networkx as nx
import numpy as np


def merge_degree_two_chains(graph):
    """Contract degree-two tree nodes, keeping root, forks and terminal nodes.

    The root follows RONALD's existing maximum coordinate-0 convention. Cyclic
    components are copied unchanged: a simple Graph cannot represent all parallel
    paths that contraction might create there. Input graph/arrays are untouched.
    """
    if graph.is_directed() or graph.is_multigraph():
        raise ValueError("Expected a simple undirected skeleton graph")
    result = nx.Graph()
    result.graph.update(
        {k: v for k, v in graph.graph.items() if k != "_ronald_edge_spatial_index"}
    )
    for component in nx.connected_components(graph):
        sub = graph.subgraph(component)
        if not nx.is_tree(sub):
            result.add_nodes_from((n, dict(d)) for n, d in sub.nodes(data=True))
            result.add_edges_from((u, v, dict(d)) for u, v, d in sub.edges(data=True))
            continue
        # Iterate in original graph order so tied maximum coordinates retain
        # exactly the same root as downstream get_node_order.
        root = max(sub.nodes, key=lambda n: sub.nodes[n]["o"][0])
        keep = {n for n in sub if sub.degree[n] != 2} | {root}
        result.add_nodes_from((n, dict(sub.nodes[n])) for n in sub if n in keep)
        visited = set()
        for start in sub:
            if start not in keep:
                continue
            for neighbor in sub.neighbors(start):
                if frozenset((start, neighbor)) in visited:
                    continue
                parts = []
                previous, current = start, neighbor
                source_edges = []
                while True:
                    edge = frozenset((previous, current))
                    visited.add(edge)
                    source_edges.append((previous, current))
                    first = np.asarray(sub.nodes[previous]["o"])
                    last = np.asarray(sub.nodes[current]["o"])
                    points = np.asarray(sub.edges[previous, current]["pts"])
                    if len(points):
                        forward = np.linalg.norm(points[0] - first) + np.linalg.norm(
                            points[-1] - last
                        )
                        reverse = np.linalg.norm(points[-1] - first) + np.linalg.norm(
                            points[0] - last
                        )
                        if reverse < forward:
                            points = points[::-1]
                    # Retain original sample points and include graph endpoints.
                    parts.append(np.concatenate((first[None], points, last[None])))
                    if current in keep:
                        break
                    next_node = next(n for n in sub.neighbors(current) if n != previous)
                    previous, current = current, next_node
                path = np.concatenate(parts)
                path = path[np.r_[True, np.any(path[1:] != path[:-1], axis=1)]]
                if len(path) == 1:
                    # Preserve zero-length edges for the caller's validity guards.
                    path = np.repeat(path, 2, axis=0)
                result.add_edge(
                    start,
                    current,
                    pts=path.copy(),
                    weight=float(
                        np.linalg.norm(
                            np.diff(path.astype(float), axis=0), axis=1
                        ).sum()
                    ),
                    source_edges=tuple(source_edges),
                )
        assert len(visited) == sub.number_of_edges(), (
            "Skeleton edge lost during chain merge"
        )
    # Keep original global insertion order as well, including tied roots in
    # disconnected graphs whose nodes were originally interleaved.
    ordered = nx.Graph()
    ordered.graph.update(result.graph)
    ordered.add_nodes_from((n, dict(result.nodes[n])) for n in graph if n in result)
    ordered.add_edges_from((u, v, dict(d)) for u, v, d in result.edges(data=True))
    return ordered
