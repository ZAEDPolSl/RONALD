import numpy as np
import pytest

from ronald.modelling.principal_axes import PrincipalAxes3D


def short_wide_cloud():
    return np.array(
        [(x, y, z) for x in (-2, 0, 2) for y in (-8, 0, 8) for z in (-1, 0, 1)],
        dtype=float,
    ) + np.array([3.5, -2.0, 7.0])


def test_short_wide_cloud_uses_existing_longitudinal_axis():
    fitted = PrincipalAxes3D().fit(short_wide_cloud())
    np.testing.assert_allclose(abs(fitted.components_[0]), [0, 1, 0])
    reordered = fitted.reordered_for_direction([1, 0, 0])
    np.testing.assert_allclose(reordered.components_[0], [1, 0, 0])
    np.testing.assert_array_equal(reordered.axis_order_, [1, 0, 2])


def test_oblique_guide_does_not_replace_the_pca_axis():
    fitted = PrincipalAxes3D().fit(short_wide_cloud())
    guide = np.array([-1, 0.3, 0.2])
    reordered = fitted.reordered_for_direction(guide)
    np.testing.assert_allclose(reordered.components_[0], [-1, 0, 0])
    assert not np.allclose(reordered.components_[0], guide / np.linalg.norm(guide))
    np.testing.assert_array_equal(reordered.axis_signs_, [-1, 1, 1])


def test_coordinate_permutation_matches_transform_and_roundtrips():
    points = short_wide_cloud()
    # Avoid testing only an axis-aligned PCA frame.
    angle = 0.39
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    points = points @ rotation.T
    fitted = PrincipalAxes3D().fit(points)
    reordered = fitted.reordered_for_direction(-rotation[:, 0])
    np.testing.assert_allclose(
        reordered.components_ @ reordered.components_.T, np.eye(3), atol=1e-14
    )
    np.testing.assert_allclose(
        fitted.transform(points)[:, reordered.axis_order_] * reordered.axis_signs_,
        reordered.transform(points),
        atol=1e-14,
    )
    np.testing.assert_allclose(
        reordered.inverse_transform(reordered.transform(points)), points, atol=1e-14
    )
    np.testing.assert_array_equal(reordered.mean_, fitted.mean_)


def test_repeated_selections_leave_original_and_other_results_unchanged():
    fitted = PrincipalAxes3D().fit(short_wide_cloud())
    original_components = fitted.components_.copy()
    original_mean = fitted.mean_.copy()
    first = fitted.reordered_for_direction([-1, 0, 0])
    first_components = first.components_.copy()
    second = fitted.reordered_for_direction([0, 0, 1])
    np.testing.assert_array_equal(second.axis_order_, [2, 0, 1])
    np.testing.assert_array_equal(first.components_, first_components)
    np.testing.assert_array_equal(fitted.components_, original_components)
    np.testing.assert_array_equal(fitted.mean_, original_mean)
    assert not np.shares_memory(first.components_, fitted.components_)
    assert not np.shares_memory(first.mean_, fitted.mean_)
    assert not np.shares_memory(first.components_, second.components_)
    assert not hasattr(fitted, "axis_order_")


@pytest.mark.parametrize(
    "guide",
    [
        [0, 0, 0],
        [np.nan, 0, 1],
        [np.inf, 1, 0],
        [0, -np.inf, 1],
        [1, 0],
        [1, 0, 0, 0],
        [[1, 0, 0]],
        1,
    ],
)
def test_invalid_guides_raise(guide):
    fitted = PrincipalAxes3D().fit(short_wide_cloud())
    with pytest.raises(ValueError, match="direction"):
        fitted.reordered_for_direction(guide)


@pytest.mark.parametrize("scale", [1e-300, 1e300])
def test_finite_direction_scale_does_not_change_selection(scale):
    fitted = PrincipalAxes3D().fit(short_wide_cloud())
    reordered = fitted.reordered_for_direction(np.array([-1.0, 0.3, 0.2]) * scale)
    np.testing.assert_allclose(reordered.components_[0], [-1, 0, 0])
