import networkx as nx
import numpy as np
from ronald.modelling.graph_chains import merge_degree_two_chains


def graph(coords, edges):
    g = nx.Graph()
    for i, p in enumerate(coords):
        g.add_node(i, o=np.array(p), pts=np.array([p]))
    for i, (u, v) in enumerate(edges):
        path = np.array([coords[u], (np.array(coords[u]) + coords[v]) / 2, coords[v]])
        if i % 2:
            path = path[::-1]
        g.add_edge(u, v, pts=path)
    return g


def test_curved_chain_preserves_all_samples_and_orients_path():
    g = graph([(9, 0, 0), (8, 0, 1), (7, 1, 2), (6, 2, 1)], [(0, 1), (1, 2), (2, 3)])
    r = merge_degree_two_chains(g)
    assert set(r) == {0, 3} and r.number_of_edges() == 1
    path = r.edges[0, 3]["pts"]
    np.testing.assert_array_equal(path[[0, -1]], [g.nodes[0]["o"], g.nodes[3]["o"]])
    expected = {tuple(p) for _, _, d in g.edges(data=True) for p in d["pts"]}
    assert {tuple(p) for p in path} == expected
    assert len(r.edges[0, 3]["source_edges"]) == 3
    assert len(g) == 4


def test_bifurcation_and_degree_two_root_retained():
    g = graph(
        [(5, 0, 0), (10, 0, 0), (4, 0, 0), (3, 0, 0), (2, 1, 0), (1, -1, 0)],
        [(0, 1), (1, 2), (2, 3), (3, 4), (3, 5)],
    )
    r = merge_degree_two_chains(g)
    assert set(r) == {0, 1, 3, 4, 5}
    assert r.degree[1] == 2 and r.degree[3] == 3
    assert nx.is_tree(r)
    assert sum(len(d["source_edges"]) for _, _, d in r.edges(data=True)) == 5


def test_cycles_unchanged_and_disconnected_trees_preserved():
    g = graph(
        [(3, 0, 0), (2, 1, 0), (1, 0, 0), (6, 0, 0), (5, 1, 0), (4, 0, 0), (0, 0, 0)],
        [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5)],
    )
    r = merge_degree_two_chains(g)
    assert nx.number_connected_components(r) == 3
    assert set(r.edges) & {(0, 1), (0, 2), (1, 2)} == {(0, 1), (0, 2), (1, 2)}
    assert r.has_edge(3, 5) and 6 in r


def test_empty_and_coincident_nodes():
    assert len(merge_degree_two_chains(nx.Graph())) == 0
    g = graph([(1, 0, 0), (1, 0, 0), (0, 0, 0)], [(0, 1), (1, 2)])
    r = merge_degree_two_chains(g)
    assert len(r) == 2 and r.number_of_edges() == 1
    assert np.isfinite(r.edges[0, 2]["weight"])


def test_global_root_ties_preserve_original_node_order():
    g = graph([(0, 0, 0), (10, 1, 0), (10, 0, 0), (9, 1, 0)], [(0, 2), (1, 3)])
    root = max(g, key=lambda n: g.nodes[n]["o"][0])
    r = merge_degree_two_chains(g)
    assert max(r, key=lambda n: r.nodes[n]["o"][0]) == root
