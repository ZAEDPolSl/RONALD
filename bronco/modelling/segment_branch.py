from tqdm import tqdm
import numpy as np
from scipy.spatial import KDTree

from bronco.modelling.visvalingam import visvalingam_whyatt_3d


def closest_edge_indices(branch, edge_list):
    min_distances = np.full(branch.shape[0], np.inf)
    min_edge_indices = np.full(branch.shape[0], -1)

    # Precompute KD-Trees for each edge array
    kdtrees = [KDTree(edge) for edge in edge_list]

    # Loop through each KD-Tree with a progress bar
    for edge_idx, kdtree in tqdm(enumerate(kdtrees), total=len(kdtrees), desc="Processing edges"):
        # Query the nearest distance for all branch points at once
        closest_distances, _ = kdtree.query(branch)
        
        # Update minimum distances and corresponding edge indices
        update_mask = closest_distances < min_distances
        min_distances[update_mask] = closest_distances[update_mask]
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
    points = visvalingam_whyatt_3d(branch, 8.0)
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
    new_image[checkpoints[:, 0], checkpoints[:, 1], checkpoints[:, 2]] = edge_indices + 1
 
    return new_image, graph
    