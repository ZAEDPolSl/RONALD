"""Integration checks for segment-local PCA frames and invalid ellipse guards."""

import importlib

import networkx as nx
import numpy as np
import pytest

from ronald.modelling.principal_axes import PrincipalAxes3D
from ronald.modelling.local_cross_section import LocalCrossSectionSampler
from ronald.modelling.prepare_graph import assign_thickness

branch_module = importlib.import_module("ronald.modelling.model_branch")
recovery = importlib.import_module("ronald.modelling.ellipse_recovery")
smooth = importlib.import_module("ronald.modelling.smooth_tree")


def ellipse(center, major=2.0, minor=1.0):
    return (
        np.array(center, dtype=float),
        np.array([0.0, major, 0.0]),
        np.array([0.0, 0.0, minor]),
    )


def configured_analyser(points):
    sampler = LocalCrossSectionSampler(np.ones((20, 20, 20), bool))
    analyser = branch_module.BranchAnalyser(
        local_cross_section_sampler=sampler, sato_cross_section_sampler=sampler
    )
    analyser._sato_candidate_mask = np.zeros(len(points), bool)
    analyser._last_recovery_unresolved = False
    analyser.points = points.copy()
    base = PrincipalAxes3D().fit(points)
    analyser._base_principal_axes = base
    analyser._base_transformed_points = base.transform(points)
    analyser._restore_base_pca_frame()
    return analyser


def test_segment_axis_changes_preserve_world_bases_gaps_and_restore_between_options(
    monkeypatch,
):
    points = np.array(
        [(x, y, z) for x in (-3, 0, 3) for y in (-8, 0, 8) for z in (-1, 0, 1)],
        dtype=float,
    )
    branch = np.array(
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 1, 0], [2, 2, 0]], dtype=float
    )
    analyser = configured_analyser(points)
    base = analyser.principal_axes
    orders = []
    observed_world_bases = []

    def separate(endpoints):
        orders.append(analyser.principal_axes.axis_order_.copy())
        markers = [
            analyser.principal_axes.inverse_transform(
                analyser.ellipse_boundary(ellipse(point))
            )
            for point in endpoints
        ]
        observed_world_bases.append(markers)
        return [ellipse(endpoint) for endpoint in endpoints], [
            analyser.principal_axes.transform(b) for b in markers
        ]

    monkeypatch.setattr(analyser, "separate_branch", separate)

    def analyse(ellipses):
        analyser._last_capped_ellipses = ellipses
        return np.ones(len(points), dtype=bool)

    monkeypatch.setattr(analyser, "analyse_segment", analyse)
    selected, bases, gaps, _ = analyser.analyse_indices_option([0, 2, 4], branch)
    assert np.all(selected)
    assert orders[0][0] != orders[1][0]
    assert analyser.principal_axes is base
    assert analyser.transformed_points is analyser._base_transformed_points
    np.testing.assert_allclose(
        base.inverse_transform(bases[0]), observed_world_bases[0][0]
    )
    np.testing.assert_allclose(
        base.inverse_transform(bases[1]), observed_world_bases[1][1]
    )
    assert len(gaps) == 1
    np.testing.assert_allclose(gaps[0][0], observed_world_bases[0][1])
    np.testing.assert_allclose(gaps[0][1], observed_world_bases[1][0])
    # A new option must derive coordinates from the original fit, not from the
    # reordered frame of the preceding option's final segment.
    _, second_bases, _, _ = analyser.analyse_indices_option([0, 4], branch)
    assert analyser.principal_axes is base
    np.testing.assert_allclose(
        base.inverse_transform(second_bases[0]), observed_world_bases[-1][0]
    )
    np.testing.assert_allclose(
        base.inverse_transform(second_bases[1]), observed_world_bases[-1][1]
    )
    np.testing.assert_allclose(
        base.inverse_transform(analyser.transformed_points), points
    )


def test_all_unresolved_thicknesses_produce_finite_zero_closing_sizes():
    graph = nx.path_graph(3)
    nx.set_node_attributes(graph, 0.0, "thickness")
    result = assign_thickness(graph, [0, 1, 2])
    sizes = np.array([data["size"] for _, _, data in result.edges(data=True)])
    assert np.all(np.isfinite(sizes))
    np.testing.assert_array_equal(sizes, 0.0)


def test_zero_cylinder_fallback_preserves_sato_ellipses_without_caps(monkeypatch):
    points = np.array(
        [
            [0.0, 0.75, 0.0],
            [2.0, 0.75, 0.0],
            [2.0, 2.0, 0.0],
            [2.0, 0.0, 0.8],
            [1.0, 0.0, 0.0],
        ]
    )
    branch = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    analyser = configured_analyser(points)
    # Identity coordinates make the expected capped footprint explicit.
    analyser.principal_axes.components_ = np.eye(3)
    analyser.principal_axes.mean_ = np.zeros(3)
    analyser.transformed_points = points.copy()
    analyser._base_transformed_points = points.copy()
    analyser._last_recovery_unresolved = False
    upper = ellipse(branch[0], major=1.0, minor=0.5)
    lower = ellipse(branch[-1], major=3.0, minor=2.0)

    def separate(endpoints):
        return [upper, lower], [points[[0]], points[[1, 2, 3]]]

    monkeypatch.setattr(analyser, "separate_branch", separate)
    monkeypatch.setattr(
        branch_module,
        "is_point_in_cylinder",
        lambda points, *args: np.zeros(len(points), dtype=bool),
    )
    selected, bases, _, thickness = analyser.analyse_indices_option([0, 2], branch)
    # Join geometry uses the original Sato-supported ellipse; no proximal cap.
    assert max(abs(bases[1][:, 1])) <= 3.0 + 1e-8
    assert max(abs(bases[1][:, 2])) <= 2.0 + 1e-8
    assert thickness == pytest.approx(2.0)
    # The wider lower section retains its measured Sato footprint.
    np.testing.assert_array_equal(selected, [True, True, True, True, False])
    np.testing.assert_allclose(
        np.linalg.norm(analyser._last_capped_ellipses[1][1:], axis=1),
        [3.0, 2.0],
        atol=1e-9,
    )


@pytest.mark.parametrize("count", [2, 3, 4])
def test_short_branch_never_reverses_or_collapses_caps(monkeypatch, count):
    branch = np.column_stack((np.arange(count, dtype=float), np.zeros((count, 2))))
    points = np.array(
        [(x, y, z) for x in range(count) for y in (-1.0, 1.0) for z in (-1.0, 1.0)]
    )
    analyser = configured_analyser(points)
    observed = []

    def separate(endpoints):
        observed.append(analyser.principal_axes.inverse_transform(endpoints))
        analyser._last_recovery_unresolved = False
        return [ellipse(p) for p in endpoints], [np.array([p]) for p in endpoints]

    monkeypatch.setattr(analyser, "separate_branch", separate)
    _, bases, _, _ = analyser.analyse_indices_option([0, count - 1], branch)
    expected = branch[[0, -1]]
    np.testing.assert_allclose(observed[0], expected, atol=1e-10)
    first, last = [
        analyser.principal_axes.inverse_transform(base).mean(axis=0) for base in bases
    ]
    assert first[0] < last[0]


def test_unresolved_local_sections_do_not_generate_tubes(monkeypatch):
    points = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]])
    analyser = configured_analyser(points)

    def unresolved(endpoints):
        analyser._last_recovery_unresolved = True
        return [ellipse(p, 0, 0) for p in endpoints], []

    monkeypatch.setattr(analyser, "separate_branch", unresolved)

    def forbidden(*args, **kwargs):
        pytest.fail("Unresolved sections must not produce cylinders or fallback points")

    monkeypatch.setattr(analyser, "analyse_segment", forbidden)
    monkeypatch.setattr(analyser, "ellipse_point_mask", forbidden)
    selected, bases, gaps, thickness = analyser.analyse_indices_option([0, 2], points)
    assert not selected.any()
    assert all(base.shape == (0, 3) for base in bases)
    assert gaps == [] and thickness == 0
