from tqdm import tqdm
import numpy as np
from scipy.spatial import KDTree

from bronco.modelling.visvalingam import visvalingam_whyatt_3d


def closest_edge_indices(branch, edge_list, tol=1e-6):
    min_distances = np.full(branch.shape[0], np.inf)
    min_edge_indices = np.full(branch.shape[0], -1)

    # First pass: Exact coordinate matching
    for edge_idx, edge in enumerate(edge_list):
        # Create coordinate hash for fast lookups
        edge_coords = {tuple(coord) for coord in edge}
        # Vectorized check for branch points in this edge
        in_edge = np.array([tuple(pt) in edge_coords for pt in branch], dtype=bool)
        # Force assignment for exact matches
        min_edge_indices[in_edge] = edge_idx
        min_distances[in_edge] = 0

    # Second pass: Geometric proximity for non-exact matches
    kdtrees = [KDTree(edge) for edge in edge_list]
    for edge_idx, kdtree in tqdm(enumerate(kdtrees), total=len(kdtrees)):
        dists, _ = kdtree.query(branch)
        # Mask for points not already assigned and closer than tolerance
        update_mask = (dists < min_distances) & (min_distances > tol)
        min_distances[update_mask] = dists[update_mask]
        min_edge_indices[update_mask] = edge_idx

    return min_edge_indices


def segment_branch(branch: np.ndarray) -> np.ndarray:
    """Segment a branch by using Visvalingam-Whyatt algorithm.

    Parameters
    ----------
    branch : np.ndarray
        A 2D array of shape (n_points, 3) representing the branch.
    Returns
    -------
    np.ndarray
        A 1D array of indices of the points that were kept.
    """
    points = visvalingam_whyatt_3d(branch, 10.0)
    # get the indices of the points that were kept
    indices = np.where(np.isin(branch, points).all(axis=1))[0]
    return indices


def assign_edge_number(graph):
    edge_list = []
    for u, v, data in graph.edges(data=True):
        edge_list.append(np.array(graph.edges[u, v]["pts"]))
        graph.edges[u, v]["mask"] = len(edge_list)
    return edge_list, graph


def assign_branch(image, graph):
    edge_list, graph = assign_edge_number(graph)
    checkpoints = np.argwhere(image == 1)
    # assign the branch to checkpoints
    edge_indices = closest_edge_indices(checkpoints, edge_list)
    # Create a new image mask with the same shape as the input image
    new_image = np.zeros_like(image, dtype=int)
    # Vectorized assignment to the new_image array
    new_image[checkpoints[:, 0], checkpoints[:, 1], checkpoints[:, 2]] = (
        edge_indices + 1
    )
    return new_image, graph
