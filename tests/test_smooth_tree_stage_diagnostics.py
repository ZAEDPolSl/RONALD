import importlib
import networkx as nx
import numpy as np
import SimpleITK as sitk

smooth = importlib.import_module("ronald.modelling.smooth_tree")


def test_observer_separates_cylinder_join_recovery_and_closing_without_changing_result(
    monkeypatch,
):
    data = np.ones((7, 5, 5), dtype=np.uint8)
    image = sitk.GetImageFromArray(data)
    graph = nx.Graph()
    graph.add_node(0, o=np.array([5, 2, 2]))
    graph.add_node(1, o=np.array([1, 2, 2]))
    graph.add_edge(0, 1, pts=np.array([[5, 2, 2], [1, 2, 2]]), mask=1)
    constructor_flags = []

    class Analyser:
        fit_diagnostics = {}

        def __init__(self, **kwargs):
            constructor_flags.append(kwargs)

        def smooth_branch_points(self, *args, **kwargs):
            base = np.array([[1.0, 2.0, 2.0]])
            return np.array([[1, 2, 2]]), base, base, 1.0, [(base, base + 1)]

    class Observer:
        def __init__(self):
            self.cylinders = np.zeros_like(data, bool)
            self.stages = {}
            self.branches = []

        def record_cylinder_points(self, points):
            self.cylinders[tuple(points.T)] = True

        def record_branch(self, label, parent, child, analyser):
            self.branches.append((label, parent, child))

        def record_stage(self, name, arr):
            self.stages[name] = arr.copy()

    def joins(first, last, mask, *args, **kwargs):
        mask[2, 2, 2] = 1

    def recover(mask, *args, **kwargs):
        out = mask.copy()
        out[3, 2, 2] = 1
        return out

    def close(graph, labels, *args, **kwargs):
        out = labels.astype(np.uint8)
        out[4, 2, 2] = 1
        return out

    monkeypatch.setattr(smooth, "merge_degree_two_chains", lambda graph: graph)
    monkeypatch.setattr(smooth, "prepare_graph", lambda *a, **k: graph.copy())
    monkeypatch.setattr(
        smooth, "assign_branch", lambda mask, g, **kwargs: (mask.copy(), g)
    )
    monkeypatch.setattr(smooth, "BranchAnalyser", Analyser)
    monkeypatch.setattr(smooth, "fill_local_tube_join", joins)
    monkeypatch.setattr(smooth, "restore_connected_sato", recover)
    monkeypatch.setattr(smooth, "apply_smoothing_by_node_order", close)
    monkeypatch.setattr(smooth, "calculate_sato_mask", lambda *a, **k: (image, {}))
    monkeypatch.setattr(
        smooth,
        "connect_sato_components",
        lambda g, w, s, **k: (g, np.zeros_like(w), {}),
    )
    control = smooth.smooth_tree(image)
    observer = Observer()
    result = smooth.smooth_tree(image, diagnostics=observer)
    np.testing.assert_array_equal(
        sitk.GetArrayFromImage(result), sitk.GetArrayFromImage(control)
    )
    assert constructor_flags[0]["capture_geometry"] is False
    assert constructor_flags[1]["capture_geometry"] is True
    assert observer.cylinders.sum() == 1
    assert observer.stages["02_after_joins"].sum() == 2
    assert observer.stages["03_after_sato_recovery"].sum() == 3
    assert observer.stages["04_after_closing"].sum() == 4
    assert observer.branches == [(1, 0, 1)]
