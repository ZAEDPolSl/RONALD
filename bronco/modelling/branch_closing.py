import numpy as np
from skimage.morphology import closing
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


def float_ball(radius):
    r = int(np.ceil(radius))
    z, y, x = np.ogrid[-r : r + 1, -r : r + 1, -r : r + 1]
    mask = x**2 + y**2 + z**2 <= radius**2
    return mask


def _close_branch(args):
    branch_mask, selem = args
    return closing(branch_mask, selem)


def apply_smoothing_by_node_order(airways_graph, branches_mask, node_order, thick_mult=3):
    """
    Smooth branch masks using morphological closing with a float-compatible ball structuring element.

    Parameters:
        airways_graph (nx.Graph): Graph with 'mask' and 'size' (or 'normalized_thickness') edge attributes.
        branches_mask (ndarray): 3D label mask of airway branches.
        node_order (list): Top-down node order.

    Returns:
        np.ndarray: Smoothed binary mask of the airway tree.
    """
    new_smooth = np.zeros_like(branches_mask, dtype=bool)
    selem_cache = {}

    # Precompute all unique branch masks (skip label 0 and empty masks)
    unique_labels = np.unique(branches_mask)
    branch_masks = {
        label: (branches_mask == label)
        for label in unique_labels
        if label != 0 and np.any(branches_mask == label)
    }

    # Prepare tasks for parallel processing
    tasks = []
    for node in node_order:
        for neighbor in airways_graph.neighbors(node):
            edge_data = airways_graph.get_edge_data(node, neighbor)
            if edge_data is None:
                continue
            mask = edge_data.get("mask")
            thickness = edge_data.get("size")
            if mask is None or thickness is None or mask not in branch_masks:
                continue
            radius = round(thickness * thick_mult, 1)
            if radius not in selem_cache:
                selem_cache[radius] = float_ball(radius)
            selem = selem_cache[radius]
            branch_mask = branch_masks[mask]
            tasks.append((branch_mask, selem))

    # Parallel morphological closing
    with ThreadPoolExecutor() as executor:
        for closed_branch in tqdm(executor.map(_close_branch, tasks), total=len(tasks)):
            np.logical_or(new_smooth, closed_branch, out=new_smooth)

    return new_smooth.astype(int)
