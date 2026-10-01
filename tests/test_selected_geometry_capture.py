import numpy as np
from ronald.modelling.model_branch import BranchAnalyser
from ronald.modelling.principal_axes import PrincipalAxes3D
from ronald.modelling.local_cross_section import LocalCrossSectionSampler


def test_capture_does_not_change_fit_and_reports_world_vectors():
    local = np.array(
        [(x, y, z) for x in range(7) for y in (-1, 0, 1) for z in (-1, 0, 1)], float
    )
    rotation = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    points = local @ rotation + np.array([10.0, 20.0, 30.0])
    branch = np.column_stack((np.arange(7), np.zeros((7, 2)))) @ rotation + [10, 20, 30]
    mask = np.zeros((50, 50, 50), bool)
    points = points.astype(int)
    mask[tuple(points.T)] = True
    sampler = LocalCrossSectionSampler(mask)
    args = dict(local_cross_section_sampler=sampler, sato_cross_section_sampler=sampler)
    control = BranchAnalyser(**args)
    traced = BranchAnalyser(**args, capture_geometry=True)
    expected = control.smooth_branch_points(branch, points, pca_points=points)
    actual = traced.smooth_branch_points(branch, points, pca_points=points)
    for a, b in zip(actual[:3], expected[:3]):
        np.testing.assert_array_equal(a, b)
    assert actual[3] == expected[3]
    assert traced.selected_segments and not control.selected_segments
    first = traced.selected_segments[0]
    assert first["status"] == "valid"
    np.testing.assert_allclose(
        np.abs(first["pca_direction_zyx"]),
        np.abs(control.principal_axes.components_[0]),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        first["centers_zyx"],
        [expected[1].mean(axis=0), expected[2].mean(axis=0)],
        atol=1e-10,
    )
    for i in range(2):
        boundary = traced.ellipse_boundary(
            (
                first["centers_zyx"][i],
                first["major_vectors_zyx"][i],
                first["minor_vectors_zyx"][i],
            )
        )
        np.testing.assert_allclose(boundary, expected[i + 1], atol=1e-10)
    for axes in ("major_vectors_zyx", "minor_vectors_zyx"):
        np.testing.assert_allclose(
            first[axes] @ first["pca_direction_zyx"], 0, atol=1e-8
        )


def test_only_winning_option_geometry_is_kept(monkeypatch):
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 0.0, 0.0], [3.0, 1.0, 1.0]]
    )
    sampler = LocalCrossSectionSampler(np.ones((10, 10, 10), bool))
    analyser = BranchAnalyser(
        capture_geometry=True,
        local_cross_section_sampler=sampler,
        sato_cross_section_sampler=sampler,
    )

    def initialise(*args, **kwargs):
        analyser.points = points
        analyser.principal_axes = PrincipalAxes3D().fit(points)
        analyser.indices_options = [
            np.array([0, 3]),
            np.array([0, 1, 3]),
            np.array([0, 2, 3]),
        ]

    count = [0]

    def analyse(indices, branch):
        n = count[0]
        count[0] += 1
        analyser._last_actual_indices = indices.copy()
        analyser._option_segments = [dict(marker=n)]
        selected = np.arange(4) < [1, 3, 2][n]
        return selected, [points[:1], points[-1:]], [], 1.0

    monkeypatch.setattr(analyser, "initialise", initialise)
    monkeypatch.setattr(analyser, "analyse_indices_option", analyse)
    analyser.smooth_branch_points(points, points, pca_points=points)
    assert analyser.selected_option_index == 1
    assert analyser.selected_indices == [0, 1, 3]
    assert analyser.selected_segments == [dict(marker=1)]
