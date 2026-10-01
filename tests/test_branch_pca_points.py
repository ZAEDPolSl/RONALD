"""Sato controls PCA while all supported wall candidates remain eligible."""

import numpy as np
import pytest
from ronald.modelling.densify import densify_point_cloud
from ronald.modelling.local_cross_section import LocalCrossSectionSampler
from ronald.modelling.model_branch import BranchAnalyser
from ronald.modelling.principal_axes import PrincipalAxes3D


def clouds():
    walls = np.array(
        [
            (x, y, z)
            for x in (6, 10, 14)
            for y in (0, 5, 10, 15, 20)
            for z in (9, 10, 11)
        ]
    )
    sato = np.array([(x, 10, 10) for x in (6, 8, 10, 12, 14)])
    mask = np.zeros((25, 25, 25), bool)
    mask[tuple(sato.T)] = True
    sampler = LocalCrossSectionSampler(mask)
    return (
        walls,
        sato,
        BranchAnalyser(
            local_cross_section_sampler=sampler, sato_cross_section_sampler=sampler
        ),
    )


def test_densified_sato_controls_pca_and_preserves_wall_candidates():
    walls, support, analyser = clouds()
    expected = PrincipalAxes3D().fit(densify_point_cloud(support, factor=100))
    original_walls, original_support = walls.copy(), support.copy()
    analyser.initialise(support, points=walls, pca_points=support)
    np.testing.assert_array_equal(
        analyser.principal_axes.components_, expected.components_
    )
    np.testing.assert_array_equal(analyser.principal_axes.mean_, expected.mean_)
    np.testing.assert_allclose(
        abs(analyser.principal_axes.components_[0]), [1, 0, 0], atol=1e-12
    )
    np.testing.assert_array_equal(analyser.points, walls)
    np.testing.assert_allclose(
        analyser.principal_axes.inverse_transform(analyser.transformed_points),
        walls,
        atol=1e-12,
    )
    np.testing.assert_array_equal(walls, original_walls)
    np.testing.assert_array_equal(support, original_support)


def test_smoothing_can_keep_walls_outside_sato(monkeypatch):
    walls, support, analyser = clouds()

    def fake_analyse(indices, branch):
        selected = np.zeros(len(walls), bool)
        selected[0] = True
        return (
            selected,
            [analyser.transformed_points[[0]], analyser.transformed_points[[-1]]],
            [],
            2.0,
        )

    monkeypatch.setattr(analyser, "analyse_indices_option", fake_analyse)
    selected, first, second, thickness, gaps = analyser.smooth_branch_points(
        support, walls, pca_points=support
    )
    np.testing.assert_array_equal(selected, walls[[0]])
    np.testing.assert_allclose(first, walls[[0]], atol=1e-12)
    np.testing.assert_allclose(second, walls[[-1]])
    assert thickness == 2.0 and gaps == []


def test_two_distinct_sato_points_are_densified():
    walls, support, analyser = clouds()
    analyser.prepare_principal_axes(walls, pca_points=support[[0, -1]])
    np.testing.assert_allclose(
        abs(analyser.principal_axes.components_[0]), [1, 0, 0], atol=1e-12
    )


@pytest.mark.parametrize(
    "support",
    [
        np.empty((0, 3)),
        np.zeros((1, 3)),
        np.zeros((3, 3)),
        np.zeros((3, 2)),
        np.array([[0, 0, 0], [1, np.nan, 0]]),
        np.array([[0, 0, 0], [1, np.inf, 0]]),
    ],
)
def test_invalid_sato_support_raises_without_wall_pca_fallback(support):
    walls, _, analyser = clouds()
    with pytest.raises(ValueError, match="pca_points"):
        analyser.prepare_principal_axes(walls, pca_points=support)
