import importlib
from types import SimpleNamespace
import networkx as nx
import numpy as np
import SimpleITK as sitk
from ronald.modelling.prepare_graph import prepare_graph, clean_airways_graph
from ronald.modelling.smooth_tree import get_node_order


def test_two_disconnected_tubes_survive_skeletonization():
    mask = np.zeros((35, 35, 35), np.uint8)
    mask[4:26, 5:8, 5:8] = 1
    mask[8:21, 24:27, 24:27] = 1
    image = sitk.GetImageFromArray(mask)
    default = prepare_graph(image)
    forest = prepare_graph(image, keep_all_components=True)
    assert nx.number_connected_components(default) == 1
    assert nx.number_connected_components(forest) == 2
    assert forest.number_of_edges() >= 2
    for component in nx.connected_components(forest):
        assert nx.is_tree(forest.subgraph(component))
    np.testing.assert_array_equal(sitk.GetArrayFromImage(image), mask)


def test_forest_traversal_roots_every_component_before_children():
    g = nx.Graph()
    # Deliberately interleave insertion order and reverse a smaller component.
    for n, z in [("a0", 0), ("b0", 2), ("a2", 10), ("b1", 7), ("a1", 5)]:
        g.add_node(n, o=np.array([z, 1, 1]))
    g.add_edges_from([("a0", "a1"), ("a1", "a2"), ("b0", "b1")])
    order = get_node_order(g)
    assert order == ["a2", "a1", "a0", "b1", "b0"]
    assert len(order) == len(set(order)) == len(g)


def test_clean_forest_retains_all_nodes_and_tree_edges():
    g = nx.Graph()
    for i in range(5):
        g.add_node(i, o=np.array([i, 0, 0]))
    g.add_edges_from([(0, 1), (1, 2), (2, 0), (3, 4)])
    forest = clean_airways_graph(g, keep_all_components=True)
    assert set(forest) == set(g)
    assert nx.is_forest(forest)
    assert nx.number_connected_components(forest) == 2
    assert g.number_of_edges() == 4


def test_kimimaro_retains_multiple_skeletons_and_disconnected_graph(monkeypatch):
    mod = importlib.import_module("ronald.modelling.kimimaro_graph")
    a = SimpleNamespace(
        vertices=np.array([[2, 2, 2], [3, 2, 2], [4, 2, 2]], float),
        edges=np.array([[0, 1], [1, 2]]),
    )
    b = SimpleNamespace(
        vertices=np.array([[8, 8, 8], [9, 8, 8]], float), edges=np.array([[0, 1]])
    )
    calls = []

    def fake(array, **kwargs):
        calls.append(kwargs)
        return {1: a, 2: b}

    monkeypatch.setattr(mod.kimimaro, "skeletonize", fake)
    arr = np.zeros((12, 12, 12), np.uint8)
    arr[2:5, 2, 2] = 1
    arr[8:10, 8, 8] = 1
    forest = mod.prepare_graph_kimimaro(arr, keep_all_components=True)
    assert nx.number_connected_components(forest) == 2
    assert forest.number_of_edges() == 3
    assert calls[-1]["dust_threshold"] == 0
    np.testing.assert_array_equal(
        mod._preprocess_for_kimimaro(arr, keep_all_components=True), arr
    )


def test_end_to_end_separate_tubes_are_connected_within_supported_walls():
    from scipy import ndimage as ndi
    from ronald.modelling.smooth_tree import model_tree

    z, y, x = np.indices((48, 28, 28))
    s = ((y - 14) ** 2 + (x - 14) ** 2 <= 4) & (
        ((z >= 5) & (z <= 20)) | ((z >= 25) & (z <= 40))
    )
    w = ((y - 14) ** 2 + (x - 14) ** 2 <= 9) & (z >= 5) & (z <= 40)
    image = lambda a: sitk.GetImageFromArray(a.astype(np.uint8))
    a = model_tree(image(w), image(s)) > 0
    assert np.all(a[s])
    assert not (a & ~w).any()
    assert ndi.label(a, structure=np.ones((3, 3, 3)))[1] == 1
