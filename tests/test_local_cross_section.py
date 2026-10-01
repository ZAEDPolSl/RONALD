import numpy as np
import pytest
from ronald.modelling.local_cross_section import LocalCrossSectionSampler


def test_round_tube_full_support_and_no_mutation():
    z, y, x = np.indices((9, 41, 41))
    mask = ((y - 20) ** 2 + (x - 20) ** 2 <= 100).astype(np.uint8)
    saved = mask.copy()
    sampler = LocalCrossSectionSampler(mask)
    points, diag = sampler.sample([4, 20, 20], np.eye(3))
    assert diag["status"] == "accepted" and diag["expansions"] > 0
    assert diag["patch_half_size"] == 16
    assert np.ptp(points[:, 1]) >= 19 and np.ptp(points[:, 2]) >= 19
    assert np.all(points[:, 0] == 4)
    np.testing.assert_array_equal(mask, saved)


def test_disconnected_neighbor_excluded():
    mask = np.zeros((9, 41, 41), np.uint8)
    mask[:, 18:23, 18:23] = 1
    mask[:, 18:23, 27:31] = 1
    points, diag = LocalCrossSectionSampler(mask).sample([4, 20, 20], np.eye(3))
    assert diag["status"] == "accepted"
    assert points[:, 2].max() < 23
    assert diag["component_points"] < diag["foreground_points"]


def test_oblique_plane_points_and_foreground_support():
    z, y, x = np.indices((31, 31, 31))
    mask = (z - 15) ** 2 + (y - 15) ** 2 + (x - 15) ** 2 <= 49
    t = 0.67
    frame = np.array(
        [[np.cos(t), np.sin(t), 0], [-np.sin(t), np.cos(t), 0], [0, 0, 1.0]]
    )
    center = np.array([15.0, 15.0, 15.0])
    points, diag = LocalCrossSectionSampler(mask).sample(center, frame)
    assert diag["status"] == "accepted"
    np.testing.assert_allclose((points - center) @ frame[0], 0, atol=1e-12)
    assert np.max(np.linalg.norm(points - center, axis=1)) < 8


def test_empty_center_does_not_snap_to_neighbor():
    mask = np.zeros((9, 31, 31), bool)
    mask[:, 15:20, 20:25] = True
    points, diag = LocalCrossSectionSampler(mask).sample([4, 15, 15], np.eye(3))
    assert points.shape == (0, 3)
    assert diag["reason"] == "center_not_in_slab_foreground"


def test_clipped_component_never_returns_artificial_boundary():
    mask = np.ones((9, 81, 81), bool)
    points, diag = LocalCrossSectionSampler(mask, max_half_size=16).sample(
        [4, 40, 40], np.eye(3)
    )
    assert len(points) == 0 and diag["reason"] == "component_exceeds_patch_limit"


def test_image_boundary_is_zero_outside_no_wrapping():
    mask = np.zeros((9, 15, 15), bool)
    mask[:, :4, :4] = True
    points, diag = LocalCrossSectionSampler(mask).sample([4, 0, 0], np.eye(3))
    assert diag["status"] == "accepted"
    assert points[:, 1:].min() >= -0.5 and points[:, 1:].max() <= 3.5


def test_one_voxel_has_dense_cell_support():
    mask = np.zeros((7, 7, 7), bool)
    mask[3, 3, 3] = True
    points, diag = LocalCrossSectionSampler(mask).sample([3, 3, 3], np.eye(3))
    assert diag["status"] == "accepted" and len(points) >= 4
    assert np.ptp(points[:, 1]) >= 0.5 and np.ptp(points[:, 2]) >= 0.5


def test_invalid_inputs():
    with pytest.raises(ValueError):
        LocalCrossSectionSampler(np.ones((3, 3)))
    sampler = LocalCrossSectionSampler(np.ones((3, 3, 3)))
    for center, frame in [([np.nan, 0, 0], np.eye(3)), ([1, 1, 1], np.zeros((3, 3)))]:
        with pytest.raises(ValueError):
            sampler.sample(center, frame)


def test_repeated_sampling_reads_changed_mask_and_new_geometry():
    mask = np.zeros((17, 25, 25), dtype=np.uint8)
    mask[:, 10:15, 10:15] = 1
    sampler = LocalCrossSectionSampler(mask, initial_half_size=4.0)
    before, _ = sampler.sample([8.0, 12.0, 12.0], np.eye(3))
    # Reusing coordinates must not reuse occupancy from the earlier mask.
    mask[:, 8:17, 8:17] = 1
    after, diagnostic = sampler.sample([8.0, 12.0, 12.0], np.eye(3))
    expected, expected_diagnostic = LocalCrossSectionSampler(
        mask, initial_half_size=4.0
    ).sample([8.0, 12.0, 12.0], np.eye(3))
    assert len(after) > len(before)
    np.testing.assert_array_equal(after, expected)
    assert diagnostic == expected_diagnostic
    angle = 0.43
    frame = np.array(
        [
            [np.cos(angle), np.sin(angle), 0.0],
            [-np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    center = [7.5, 11.5, 12.0]
    moved, moved_diagnostic = sampler.sample(center, frame)
    expected, expected_diagnostic = LocalCrossSectionSampler(
        mask, initial_half_size=4.0
    ).sample(center, frame)
    np.testing.assert_array_equal(moved, expected)
    assert moved_diagnostic == expected_diagnostic


def test_closed_voxel_boundary_keeps_centered_single_voxel_section():
    from ronald.modelling.sato_envelope import fit_sato_guided_ellipse

    mask = np.zeros((7, 7, 7), bool)
    mask[3, 3, 3] = True
    points, _ = LocalCrossSectionSampler(mask).sample([3, 3, 3], np.eye(3))
    ellipse, diag = fit_sato_guided_ellipse(
        points, points, np.array([3.0, 3.0, 3.0]), wall_sample_pitch=0.5
    )
    assert ellipse is not None
    np.testing.assert_allclose(diag["radii"], [np.sqrt(0.5), np.sqrt(0.5)], rtol=1e-5)
