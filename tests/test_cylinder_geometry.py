"""Regression cases for parallel-plane ellipse loft membership."""

import numpy as np
import pytest

from ronald.modelling.cone_construction import is_point_in_cylinder


def loft(
    points,
    c1=(0, 0, 0),
    c2=(10, 0, 0),
    a1=(0, 2, 0),
    b1=(0, 0, 1),
    a2=(0, 2, 0),
    b2=(0, 0, 1),
):
    return is_point_in_cylinder(np.asarray(points), c1, c2, a1, b1, a2, b2)


@pytest.mark.parametrize(
    "a2,b2",
    [
        ((0, -2, 0), (0, 0, 1)),
        ((0, 2, 0), (0, 0, -1)),
        ((0, -2, 0), (0, 0, -1)),
        ((0, 0, 1), (0, 2, 0)),
        ((0, 0, -1), (0, -2, 0)),
    ],
)
def test_equivalent_endpoint_axes_do_not_change_membership(a2, b2):
    points = np.random.default_rng(23).uniform((-1, -3, -2), (11, 3, 2), (500, 3))
    points = np.vstack((points, [5, 1.8, 0], [5, 0, 0.9]))
    np.testing.assert_array_equal(loft(points), loft(points, a2=a2, b2=b2))
    assert loft(points)[-2:].all()


def test_off_center_bases_keep_fitted_end_planes():
    # At either base, the point is inside its ellipse. A centerline-normal
    # height check incorrectly rejects the first and last of these.
    points = [
        [0, -1.8, 0],
        [0, 1.8, 0],
        [10, 5.8, 0],
        [5, 2, 0.9],
        [5, 2, 1.1],
        [-0.01, 0, 0],
        [10.01, 4, 0],
    ]
    np.testing.assert_array_equal(
        loft(points, c2=(10, 4, 0)), [True, True, True, True, False, False, False]
    )


def test_taper_interpolates_squared_radii():
    # Middle section radius sqrt((1**2 + 3**2)/2) = sqrt(5).
    points = [
        [0, 1, 0],
        [0, 1.01, 0],
        [5, 2.2, 0],
        [5, 2.3, 0],
        [10, 3, 0],
        [10, 3.01, 0],
    ]
    actual = loft(points, a1=(0, 1, 0), a2=(0, 3, 0))
    np.testing.assert_array_equal(actual, [True, False, True, False, True, False])


def test_reversing_endpoints_is_invariant():
    points = np.random.default_rng(52).uniform((-1, -4, -2), (11, 7, 2), (1000, 3))
    args = [(0, 0, 0), (10, 4, 0), (0, 2, 0), (0, 0, 1), (0, 3, 0), (0, 0, 0.5)]
    forward = is_point_in_cylinder(points, *args)
    reverse = is_point_in_cylinder(
        points, args[1], args[0], args[4], args[5], args[2], args[3]
    )
    assert forward.any()
    np.testing.assert_array_equal(forward, reverse)


@pytest.mark.parametrize(
    "a1,b1",
    [
        ((0, 0, 0), (0, 0, 1)),
        ((0, 1e-10, 0), (0, 0, 1)),
        ((0, 2, 0), (0, 1, 0)),
        ((0, float("nan"), 0), (0, 0, 1)),
    ],
)
def test_degenerate_base_is_empty(a1, b1):
    assert not loft([[0, 0, 0], [5, 0, 0], [10, 0, 0]], a1=a1, b1=b1).any()


def test_nonparallel_bases_are_explicitly_rejected():
    assert not loft([[5, 0, 0]], a2=(1, 2, 0)).any()


def test_same_plane_has_only_bounded_endpoint_disks():
    points = [
        [0, 0, 0],
        [0, 5, 0],
        [0, 2.5, 0],
        [0, 8, 0],
        [1, 0, 0],
        [-1, 5, 0],
        [0, 0, 1.01],
    ]
    np.testing.assert_array_equal(
        loft(points, c2=(0, 5, 0)), [True, True, False, False, False, False, False]
    )


def test_rotated_geometry_keeps_membership():
    rotation, _ = np.linalg.qr(np.random.default_rng(5).normal(size=(3, 3)))
    shift = np.array([40, -30, 12])
    points = np.random.default_rng(52).uniform((-1, -4, -2), (11, 7, 2), (1000, 3))
    args = np.array(
        [(0, 0, 0), (10, 4, 0), (0, 2, 0), (0, 0, 1), (0, 3, 0), (0, 0, 0.5)]
    )
    rotated = args @ rotation
    rotated[:2] += shift
    np.testing.assert_array_equal(
        is_point_in_cylinder(points, *args),
        is_point_in_cylinder(points @ rotation + shift, *rotated),
    )


def test_nonfinite_points_are_outside_and_empty_input_supported():
    assert not loft([[np.nan, 0, 0], [5, np.inf, 0]]).any()
    assert loft(np.empty((0, 3))).shape == (0,)
