import numpy as np
from scipy import ndimage

from ronald.modelling.local_tube_joins import fill_local_tube_join


def ellipse(center, normal=(1, 0, 0), radii=(3, 2), count=64):
    normal = np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal)
    reference = np.eye(3)[np.argmin(np.abs(normal))]
    u = np.cross(normal, reference)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    angles = np.arange(count) * (2 * np.pi / count)
    return (
        np.asarray(center)
        + radii[0] * np.cos(angles)[:, None] * u
        + radii[1] * np.sin(angles)[:, None] * v
    )


def test_short_collinear_gap_is_connected_without_radial_growth():
    mask = np.zeros((32, 32, 32), bool)
    walls = np.ones_like(mask)
    info = {}
    returned = fill_local_tube_join(
        ellipse((12, 16, 16)),
        ellipse((16, 16, 16)),
        mask,
        walls,
        cast_to_int=False,
        diagnostics=info,
    )
    assert returned is mask
    assert info["reason"] == "accepted"
    assert mask[12:17, 16, 16].all()
    assert ndimage.label(mask)[1] == 1
    assert not mask[:, 16, 20].any()
    assert not mask[:11].any() and not mask[18:].any()


def test_angled_join_follows_tubes_and_leaves_remote_wedge_empty():
    mask = np.zeros((40, 40, 40), bool)
    walls = np.ones_like(mask)
    info = {}
    fill_local_tube_join(
        ellipse((14, 20, 20), (1, 0, 0), (3, 3)),
        ellipse((20, 14, 20), (0, 1, 0), (3, 3)),
        mask,
        walls,
        junction=(20, 20, 20),
        cast_to_int=False,
        diagnostics=info,
    )
    assert info["reason"] == "accepted"
    assert mask[14:21, 20, 20].all()
    assert mask[20, 14:21, 20].all()
    assert not mask[15, 15, 20]
    assert ndimage.label(mask)[1] == 1


def test_nearby_unrelated_foreground_does_not_influence_closing():
    shape = (32, 32, 32)
    empty = np.zeros(shape, bool)
    original = empty.copy()
    original[13:17, 11, 14:19] = True
    original[0, 0, 0] = True
    with_neighbors = original.copy()
    walls = np.ones(shape, bool)
    args = (ellipse((12, 16, 16)), ellipse((16, 16, 16)))
    fill_local_tube_join(*args, empty, walls, cast_to_int=False)
    fill_local_tube_join(*args, with_neighbors, walls, cast_to_int=False)
    assert np.array_equal(with_neighbors, original | empty)


def test_new_voxels_are_wall_clipped_and_old_voxels_preserved():
    mask = np.zeros((32, 32, 32), bool)
    mask[0, 0, 0] = True
    walls = np.ones_like(mask)
    walls[:, :, :16] = False
    walls[0, 0, 0] = False
    before = mask.copy()
    fill_local_tube_join(
        ellipse((12, 16, 16)), ellipse((16, 16, 16)), mask, walls, cast_to_int=False
    )
    assert np.all(mask[before])
    assert not np.any((mask & ~before) & ~walls)


def test_parallel_separate_tubes_are_not_shifted_sideways_to_connect():
    mask = np.zeros((32, 32, 32), bool)
    info = {}
    fill_local_tube_join(
        ellipse((16, 8, 16), radii=(2, 2)),
        ellipse((16, 16, 16), radii=(2, 2)),
        mask,
        np.ones_like(mask),
        diagnostics=info,
    )
    assert info["reason"] == "junction_outside_tube"
    assert not mask.any()


def test_long_gap_is_rejected():
    mask = np.zeros((32, 32, 32), bool)
    info = {}
    fill_local_tube_join(
        ellipse((5, 16, 16), radii=(2, 2)),
        ellipse((25, 16, 16), radii=(2, 2)),
        mask,
        np.ones_like(mask),
        diagnostics=info,
    )
    assert info["reason"] == "extension_too_long"
    assert not mask.any()


def test_invalid_outline_is_skipped_without_changing_mask():
    for bad in [np.empty((0, 3)), np.zeros((64, 3)), np.full((64, 3), np.nan)]:
        mask = np.zeros((20, 20, 20), bool)
        mask[2, 3, 4] = True
        before = mask.copy()
        info = {}
        fill_local_tube_join(
            bad, ellipse((10, 10, 10)), mask, np.ones_like(mask), diagnostics=info
        )
        assert info["reason"] == "invalid_ellipse"
        assert np.array_equal(mask, before)


def test_rotated_border_join_does_not_wrap_around_image():
    mask = np.zeros((20, 20, 20), bool)
    info = {}
    normal = np.array([1, 1, 0]) / np.sqrt(2)
    fill_local_tube_join(
        ellipse((0, 0, 4), normal),
        ellipse((2, 2, 4), normal),
        mask,
        np.ones_like(mask),
        diagnostics=info,
    )
    assert info["reason"] == "accepted"
    assert mask.any()
    assert not mask[-1].any() and not mask[:, -1].any()


def test_casting_and_zero_closing_preserve_existing_interface():
    mask = np.zeros((20, 20, 20), bool)
    result = fill_local_tube_join(
        ellipse((8, 10, 10)),
        ellipse((10, 10, 10)),
        mask,
        np.ones_like(mask),
        closing_radius=0,
    )
    assert result.dtype.kind == "i"
    assert mask.any()
    assert np.array_equal(result.astype(bool), mask)
