import numpy as np
import pytest
from scipy import ndimage as ndi

from ronald.modelling.connected_tree import retain_main_connected_tree


def test_keeps_main_sato_overlap_not_biggest_final_component():
    mask = np.zeros((12, 12, 12), bool)
    mask[1:3, 1:3, 1:3] = True
    mask[6:11, 6:11, 6:11] = True
    sato = np.zeros_like(mask)
    sato[1:3, 1:3, 1:3] = True
    sato[8, 8, 8] = True
    kept, removed, report = retain_main_connected_tree(mask, sato)
    assert kept.sum() == 8 and removed.sum() == 125
    assert report["components_before"] == 2
    assert report["components_after"] == 1
    assert report["main_sato_overlap_voxels"] == 8


def test_actual_bridge_preserves_detached_branch_and_no_bridge_drops_it():
    mask = np.zeros((12, 8, 8), bool)
    mask[1:4, 3, 3] = True
    mask[8:10, 3, 3] = True
    sato = mask.copy()
    kept, removed, _ = retain_main_connected_tree(mask, sato)
    assert kept.sum() == 3 and removed.sum() == 2
    mask[4:8, 3, 3] = True
    kept, removed, report = retain_main_connected_tree(mask, sato)
    assert np.array_equal(kept, mask)
    assert not removed.any()
    assert report["components_after"] == 1


def test_fragmented_main_anchor_uses_greatest_overlap_not_arbitrary_fragment():
    sato = np.zeros((15, 7, 7), bool)
    sato[1:13, 3, 3] = True
    mask = sato.copy()
    mask[4:6, 3, 3] = False
    kept, removed, report = retain_main_connected_tree(mask, sato)
    assert kept[6:13, 3, 3].all()
    assert removed[1:4, 3, 3].all()
    assert report["main_sato_overlap_before"] == 10
    assert report["main_sato_overlap_voxels"] == 7
    assert report["main_sato_overlap_removed"] == 3


def test_26_connected_diagonal_bridge_counts():
    mask = np.eye(5, dtype=bool)[:, :, None] * np.ones((1, 1, 5), bool)
    sato = np.zeros_like(mask)
    sato[0, 0, 0] = True
    kept, removed, report = retain_main_connected_tree(mask, sato)
    assert np.array_equal(kept, mask)
    assert not removed.any()
    assert report["components_before"] == 1


def test_ties_are_deterministic_raster_first():
    sato = np.zeros((12, 5, 5), bool)
    sato[1:4, 2, 2] = True
    sato[7:10, 2, 2] = True
    kept, _, report = retain_main_connected_tree(sato, sato)
    assert report["main_sato_size_ties"] == 2
    assert kept[1:4, 2, 2].all() and not kept[7:].any()
    sato[4:7, 2, 2] = True
    mask = sato.copy()
    mask[4:7] = False
    kept, _, report = retain_main_connected_tree(mask, sato)
    assert report["final_overlap_ties"] == 2
    assert kept[1:4, 2, 2].all() and not kept[7:].any()


@pytest.mark.parametrize("case", ["both", "model", "sato", "disjoint"])
def test_empty_and_no_overlap_never_choose_arbitrary_component(case):
    mask = np.zeros((6, 6, 6), bool)
    sato = np.zeros_like(mask)
    if case in ("sato", "disjoint"):
        mask[1, 1, 1] = True
    if case in ("model", "disjoint"):
        sato[4, 4, 4] = True
    kept, removed, report = retain_main_connected_tree(mask, sato)
    assert not kept.any()
    assert np.array_equal(removed, mask)
    assert report["components_after"] == 0
    assert report["kept_label"] is None


def test_partition_and_inputs_preserved_including_volume_edges():
    rng = np.random.default_rng(4)
    mask = rng.random((15, 17, 13)) < 0.14
    sato = rng.random(mask.shape) < 0.1
    original_model, original_sato = mask.copy(), sato.copy()
    kept, removed, report = retain_main_connected_tree(mask, sato)
    assert np.array_equal(kept | removed, mask)
    assert not (kept & removed).any()
    assert np.array_equal(mask, original_model)
    assert np.array_equal(sato, original_sato)
    assert ndi.label(kept, np.ones((3, 3, 3)))[1] <= 1
    assert kept.dtype == bool and removed.dtype == bool
    assert report["kept_voxels"] + report["removed_voxels"] == mask.sum()


@pytest.mark.parametrize("shapes", [((3, 3), (3, 3)), ((3, 3, 3), (3, 3, 4))])
def test_invalid_shapes(shapes):
    with pytest.raises(ValueError, match="3D"):
        retain_main_connected_tree(np.zeros(shapes[0]), np.zeros(shapes[1]))
