"""
Memory-efficient branch smoothing using morphological closing.

This module provides functions for smoothing 3D branch structures using
morphological operations in a memory-optimized way.
"""

import numpy as np
from skimage.morphology import closing
from tqdm import tqdm


def float_ball(radius):
    """
    Create a 3D ball structuring element with float radius.

    Parameters
    ----------
    radius : float
        Radius of the ball structuring element

    Returns
    -------
    ndarray
        Boolean array with spherical structuring element
    """
    r = int(np.ceil(radius))
    z, y, x = np.ogrid[-r : r + 1, -r : r + 1, -r : r + 1]
    return x**2 + y**2 + z**2 <= radius**2


def _expand_bbox(bbox, shape, padding):
    return tuple(
        slice(max(0, axis.start - padding), min(shape[i], axis.stop + padding))
        for i, axis in enumerate(bbox)
    )


def find_label_bboxes(label_image, max_label=None, slice_chunk_size=16):
    """Find label bounding boxes with memory proportional to one slice chunk."""
    if max_label is None:
        max_label = int(label_image.max())
    if max_label == 0:
        return []

    minimums = np.full((max_label + 1, label_image.ndim), label_image.shape)
    maximums = np.full((max_label + 1, label_image.ndim), -1)

    for start in range(0, label_image.shape[0], slice_chunk_size):
        stop = min(start + slice_chunk_size, label_image.shape[0])
        chunk = label_image[start:stop]
        coordinates = np.argwhere(chunk)
        if len(coordinates) == 0:
            continue
        labels = chunk[tuple(coordinates.T)].astype(np.intp, copy=False)
        coordinates[:, 0] += start
        for axis in range(label_image.ndim):
            np.minimum.at(minimums[:, axis], labels, coordinates[:, axis])
            np.maximum.at(maximums[:, axis], labels, coordinates[:, axis])

    return [
        (
            None
            if maximums[label].max() < 0
            else tuple(
                slice(minimums[label, axis], maximums[label, axis] + 1)
                for axis in range(label_image.ndim)
            )
        )
        for label in range(1, max_label + 1)
    ]


def apply_smoothing_by_node_order(
    airways_graph,
    branches_mask,
    node_order,
    thick_mult=2,
    output_dtype=int,
    verbose=False,
):
    """
    Smooth branch masks using morphological closing on bounding box regions.

    Process each branch inside its padded bounding box rather than allocating
    a separate full-volume mask for every branch.

    Parameters
    ----------
    airways_graph : nx.Graph
        Graph with 'mask' and 'size' edge attributes
    branches_mask : ndarray
        3D label mask of airway branches
    node_order : list
        Top-down node order
    thick_mult : float, default=2
        Multiplier for thickness to determine closing radius

    Returns
    -------
    np.ndarray
        Smoothed binary mask of the airway tree
    """
    new_smooth = np.zeros_like(branches_mask, dtype=output_dtype)
    selem_cache = {}

    max_label = int(branches_mask.max())
    branch_bboxes = find_label_bboxes(branches_mask, max_label=max_label)
    unique_label_set = {
        label for label, bbox in enumerate(branch_bboxes, start=1) if bbox is not None
    }

    processing_queue = []
    for node in node_order:
        for neighbor in airways_graph.neighbors(node):
            edge_data = airways_graph.get_edge_data(node, neighbor)
            if edge_data is None:
                continue

            mask_id = edge_data.get("mask")
            thickness = edge_data.get("size")

            if mask_id is None or thickness is None or mask_id not in unique_label_set:
                continue

            radius = round(thickness * thick_mult, 1)
            processing_queue.append((mask_id, radius))

    # Graph traversal visits each edge from both endpoints.
    processing_queue = list(dict.fromkeys(processing_queue))

    for mask_id, radius in tqdm(
        processing_queue,
        desc="Closing branches",
        disable=not verbose,
    ):
        base_bbox = branch_bboxes[mask_id - 1]
        if base_bbox is None:
            continue
        bbox = _expand_bbox(base_bbox, branches_mask.shape, padding=int(radius) + 10)

        if radius not in selem_cache:
            selem_cache[radius] = float_ball(radius)

        roi_mask = branches_mask[bbox] == mask_id
        closed_roi = closing(roi_mask, selem_cache[radius])
        np.logical_or(new_smooth[bbox], closed_roi, out=new_smooth[bbox])

    return new_smooth
