"""Keep only the reconstructed component connected to the main Sato tree."""

import numpy as np
from scipy import ndimage as ndi


def retain_main_connected_tree(mask, sato_mask):
    """Return ``(kept, excluded, report)`` using actual 26-voxel connectivity.

    The largest original Sato component defines the main-tree anchor. Of the
    components in ``mask``, retain only the one with greatest positive overlap
    with that anchor. Final component volume never determines the choice.
    Equal Sato sizes and equal final overlap counts are resolved by the first
    component in z,y,x raster order, making ties deterministic. An absent or
    non-overlapping anchor yields an empty kept mask, rather than guessing.
    Inputs are never modified. Apply this after bridges and walls clipping.
    """
    model = np.asarray(mask, dtype=bool)
    sato = np.asarray(sato_mask, dtype=bool)
    if model.ndim != 3 or model.shape != sato.shape:
        raise ValueError("mask and sato_mask must be equally shaped 3D arrays")
    kept = np.zeros(model.shape, dtype=bool)
    excluded = model.copy()
    info = {
        "connectivity": 26,
        "tie_break": "lowest_label_in_zyx_raster_order",
        "input_voxels": int(model.sum()),
        "sato_voxels": int(sato.sum()),
        "sato_components": 0,
        "main_sato_label": None,
        "main_sato_voxels": 0,
        "main_sato_size_ties": 0,
        "components_before": 0,
        "components_after": 0,
        "kept_label": None,
        "kept_voxels": 0,
        "removed_voxels": int(model.sum()),
        "main_sato_overlap_voxels": 0,
        "main_sato_overlap_before": 0,
        "main_sato_overlap_removed": 0,
        "final_overlap_ties": 0,
        "roi_shape": [0, 0, 0],
    }
    if not info["input_voxels"] and not info["sato_voxels"]:
        info["reason"] = "empty_inputs"
        return kept, excluded, info

    # A common tight ROI avoids labelling the entire patient volume; use axis
    # projections rather than materializing every foreground coordinate.
    roi = []
    for axis in range(3):
        other = tuple(a for a in range(3) if a != axis)
        occupied = np.flatnonzero(model.any(axis=other) | sato.any(axis=other))
        roi.append(slice(int(occupied[0]), int(occupied[-1]) + 1))
    roi = tuple(roi)
    info["roi_shape"] = list(model[roi].shape)
    connectivity = np.ones((3, 3, 3), bool)
    labels, n_model = ndi.label(model[roi], structure=connectivity)
    info["components_before"] = int(n_model)
    if not info["sato_voxels"]:
        info["reason"] = "empty_sato_anchor"
        return kept, excluded, info
    sato_labels, n_sato = ndi.label(sato[roi], structure=connectivity)
    sizes = np.bincount(sato_labels.ravel())
    sizes[0] = 0
    anchor_label = int(np.argmax(sizes))
    anchor_size = int(sizes[anchor_label])
    info.update(
        sato_components=int(n_sato),
        main_sato_label=anchor_label,
        main_sato_voxels=anchor_size,
        main_sato_size_ties=int(np.count_nonzero(sizes == anchor_size)),
    )
    overlap = np.bincount(labels[sato_labels == anchor_label], minlength=n_model + 1)
    del sato_labels
    overlap[0] = 0
    info["main_sato_overlap_before"] = int(overlap.sum())
    if not n_model:
        info["reason"] = "empty_model"
        return kept, excluded, info
    kept_label = int(np.argmax(overlap))
    best_overlap = int(overlap[kept_label])
    if not best_overlap:
        info["reason"] = "no_main_sato_overlap"
        return kept, excluded, info
    local_kept = labels == kept_label
    kept[roi] = local_kept
    excluded[roi] &= ~local_kept
    retained = int(local_kept.sum())
    info.update(
        reason="retained_main_tree",
        kept_label=kept_label,
        kept_voxels=retained,
        removed_voxels=info["input_voxels"] - retained,
        components_after=1,
        main_sato_overlap_voxels=best_overlap,
        main_sato_overlap_removed=int(overlap.sum()) - best_overlap,
        final_overlap_ties=int(np.count_nonzero(overlap == best_overlap)),
    )
    return kept, excluded, info
