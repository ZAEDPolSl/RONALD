import numpy as np
from skimage.morphology import closing
from tqdm import tqdm

def float_ball(radius):
    r = int(np.ceil(radius))
    z, y, x = np.ogrid[-r:r+1, -r:r+1, -r:r+1]
    mask = x**2 + y**2 + z**2 <= radius**2
    return mask


def apply_smoothing_by_node_order(airways_graph, branches_mask, node_order):
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

    for node in tqdm(node_order):
        for neighbor in airways_graph.neighbors(node):
            edge_data = airways_graph.get_edge_data(node, neighbor)
            if edge_data is None:
                continue

            mask = edge_data.get("mask")
            thickness = edge_data.get("size")  # or "normalized_thickness"
            if mask is None or thickness is None:
                continue

            radius = round(thickness * 3, 1)
            if radius not in selem_cache:
                selem_cache[radius] = float_ball(radius)
            selem = selem_cache[radius]

            branch_mask = (branches_mask == mask)
            if not np.any(branch_mask):
                continue

            closed_branch = closing(branch_mask, selem)
            np.logical_or(new_smooth, closed_branch, out=new_smooth)

    return new_smooth.astype(int)
