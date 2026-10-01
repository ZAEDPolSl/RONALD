import json

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from ronald.modelling.sato_envelope import fit_sato_guided_ellipse


def points3(xy):
    return np.column_stack((np.zeros(len(xy)), xy))


def ring(a=1.0, b=1.0, rotation=0.0, count=128):
    t = np.linspace(0, 2 * np.pi, count, endpoint=False)
    rotate = np.array(
        [[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]]
    )
    return points3(np.column_stack((a * np.cos(t), b * np.sin(t))) @ rotate.T)


def shape(ellipse):
    axes = np.column_stack([v[1:] for v in ellipse[1:]])
    return axes @ axes.T


def assert_encloses(points, ellipse):
    delta = points[:, 1:] - ellipse[0][1:]
    quadratic = np.einsum("ij,jk,ik->i", delta, np.linalg.inv(shape(ellipse)), delta)
    assert np.max(quadratic) <= 1 + 1e-10


def assert_wall_containment(walls, ellipse):
    hull = ConvexHull(walls[:, 1:] - ellipse[0][1:])
    normal, distance = hull.equations[:, :2], -hull.equations[:, 2]
    support = np.sqrt(np.einsum("ij,jk,ik->i", normal, shape(ellipse), normal))
    assert np.all(support <= distance + 1e-8 * np.max(distance))
    assert np.max(support / distance) == pytest.approx(1.0, rel=1e-7)


@pytest.mark.parametrize(
    "a,b,rotation", [(2.0, 2.0, 0.0), (5.0, 1.0, 0.7), (100.0, 1.0, 1.1)]
)
def test_analytic_seed_and_maximum_proportional_expansion(a, b, rotation):
    sato = ring(a, b, rotation)
    walls = sato * 3
    ellipse, diagnostic = fit_sato_guided_ellipse(sato, walls, np.zeros(3))
    assert ellipse is not None
    assert diagnostic["seed_radii"] == pytest.approx([a, b], rel=1e-6)
    assert diagnostic["expansion_factor"] == pytest.approx(
        3 * np.cos(np.pi / 128), rel=1e-6
    )
    assert diagnostic["radii"][0] / diagnostic["radii"][1] == pytest.approx(
        a / b, rel=1e-6
    )
    assert diagnostic["status"] == "sato_guided"
    assert diagnostic["solver_converged"]
    assert not diagnostic["clipping_needed"]
    assert_encloses(sato, ellipse)
    assert_wall_containment(walls, ellipse)
    json.dumps(diagnostic, allow_nan=False)


def test_asymmetric_sato_keeps_exact_anchor_and_every_point():
    sato = np.concatenate([ring(), points3([[4.0, 0.2]])])
    center = np.array([7.0, 0.3, -0.2])
    ellipse, diagnostic = fit_sato_guided_ellipse(sato, None, center)
    assert np.array_equal(ellipse[0], center)
    assert diagnostic["no_expansion_reason"] == "walls_unavailable"
    assert diagnostic["seed_radii"][0] > 3.7
    assert_encloses(sato, ellipse)


def test_one_sided_wall_spur_does_not_inflate_opposite_side():
    walls = points3([[-2.0, -2.0], [-2.0, 2.0], [2.0, -2.0], [2.0, 2.0], [9.0, 0.0]])
    ellipse, diagnostic = fit_sato_guided_ellipse(ring(), walls, np.zeros(3))
    assert diagnostic["radii"] == pytest.approx([2.0, 2.0], rel=1e-6)
    assert_wall_containment(walls, ellipse)


def test_protruding_seed_kept_for_final_clipping_never_shrunk():
    sato = ring(3.0, 1.0)
    walls = points3([[-2.0, -2.0], [-2.0, 2.0], [2.0, -2.0], [2.0, 2.0]])
    ellipse, diagnostic = fit_sato_guided_ellipse(sato, walls, np.zeros(3))
    assert diagnostic["radii"] == pytest.approx([3.0, 1.0], rel=1e-6)
    assert diagnostic["expansion_factor"] == 1.0
    assert diagnostic["clipping_needed"]
    assert diagnostic["no_expansion_reason"] == "sato_seed_crosses_wall_hull"
    assert_encloses(sato, ellipse)


@pytest.mark.parametrize(
    "walls,reason",
    [
        (None, "walls_unavailable"),
        ([], "walls_unavailable"),
        ([[0.0, 0.0]], "invalid_walls_shape"),
        (np.full((4, 3), np.nan), "nonfinite_walls_input"),
        (np.zeros((4, 3)), "degenerate_walls_support"),
        (points3([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]), "degenerate_walls_support"),
        (
            points3([[0.0, -2.0], [0.0, 2.0], [2.0, -2.0], [2.0, 2.0]]),
            "center_not_strictly_inside_wall_hull",
        ),
        (
            points3([[1.0, -2.0], [1.0, 2.0], [2.0, -2.0], [2.0, 2.0]]),
            "center_not_strictly_inside_wall_hull",
        ),
    ],
)
def test_unusable_walls_retain_sato_seed(walls, reason):
    ellipse, diagnostic = fit_sato_guided_ellipse(ring(2.0, 1.0), walls, np.zeros(3))
    assert ellipse is not None
    assert diagnostic["no_expansion_reason"] == reason
    assert diagnostic["radii"] == pytest.approx([2.0, 1.0], rel=1e-6)
    assert diagnostic["expansion_factor"] == 1.0
    json.dumps(diagnostic, allow_nan=False)


@pytest.mark.parametrize(
    "sato",
    [
        None,
        [],
        [[1.0, 2.0]],
        np.zeros((2, 3)),
        np.zeros((10, 3)),
        np.full((4, 3), np.nan),
        points3([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
    ],
)
def test_missing_invalid_or_degenerate_sato_fails_without_wall_fallback(sato):
    ellipse, diagnostic = fit_sato_guided_ellipse(sato, ring(3.0, 2.0), np.zeros(3))
    assert ellipse is None
    assert diagnostic["status"] == "fallback"
    json.dumps(diagnostic, allow_nan=False)


@pytest.mark.parametrize("scale", [1e-8, 0.03, 1e5])
def test_rotation_translation_scale_and_no_mutation(scale):
    center = np.array([7.0, 10.0, -9.0]) * scale
    sato = ring(3.0, 1.0, 0.83) * scale + center
    walls = ring(6.0, 2.0, 0.83) * scale + center
    saved_sato, saved_walls, saved_center = sato.copy(), walls.copy(), center.copy()
    ellipse, diagnostic = fit_sato_guided_ellipse(sato, walls, center)
    assert ellipse is not None
    assert np.array_equal(ellipse[0], center)
    assert np.array_equal(sato, saved_sato)
    assert np.array_equal(walls, saved_walls)
    assert np.array_equal(center, saved_center)
    assert diagnostic["seed_radii"] == pytest.approx(
        np.array([3.0, 1.0]) * scale, rel=1e-6, abs=0.0
    )
    expected = shape((np.zeros(3), ring(3.0, 1.0, 0.83)[0], ring(3.0, 1.0, 0.83)[32]))
    assert shape(ellipse) == pytest.approx(
        expected * (scale * diagnostic["expansion_factor"]) ** 2, rel=1e-6, abs=0.0
    )
    assert_encloses(sato, ellipse)
    assert_wall_containment(walls, ellipse)


def test_attached_sato_sheet_is_enclosed_and_can_enlarge_the_seed():
    clean = ring()
    sheet = points3([[x, y] for x in np.linspace(0.5, 8.0, 30) for y in [-0.1, 0.1]])
    contaminated = np.concatenate([clean, sheet])
    clean_ellipse, clean_diagnostic = fit_sato_guided_ellipse(clean, None, np.zeros(3))
    ellipse, diagnostic = fit_sato_guided_ellipse(contaminated, None, np.zeros(3))
    assert diagnostic["seed_radii"][0] >= 8.0
    assert diagnostic["area"] > clean_diagnostic["area"] * 5
    assert_encloses(contaminated, ellipse)


def test_hull_reduction_retains_extreme_outlier_and_all_interior_samples():
    rng = np.random.default_rng(5)
    sato = np.concatenate(
        [
            ring(2.0, 1.0),
            points3(rng.uniform(-0.4, 0.4, (1000, 2))),
            points3([[9.0, 0.0]]),
        ]
    )
    ellipse, diagnostic = fit_sato_guided_ellipse(sato, None, np.zeros(3))
    assert ellipse is not None
    assert diagnostic["constraint_count"] < 130
    assert diagnostic["seed_radii"][0] >= 9.0
    assert diagnostic["sato_max_quadratic"] <= 1 + 1e-12
    assert_encloses(sato, ellipse)
