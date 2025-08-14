"""
Kimimaro-based graph preparation for airway tree skeletonization.

This module provides an alternative to prepare_graph using kimimaro skeletonization,
which can capture more detailed branching structures in airway trees.
"""

import networkx as nx
import SimpleITK as sitk
import numpy as np
import scipy.ndimage as ndi
import kimimaro
from bronco.modelling.prepare_graph import keep_largest_component


def _kimimaro_to_graph_structure(skeleton):
    """
    Convert kimimaro skeleton to NetworkX graph with same structure as prepare_graph.
    Returns graph with nodes having 'pts' and 'o' attributes like build_sknw.
    """
    graph = nx.Graph()

    for skel_id, skel in skeleton.items():
        vertices = skel.vertices
        edges = skel.edges

        # Add vertices as nodes with proper structure
        for i, vertex in enumerate(vertices):
            node_id = f"{skel_id}_{i}"
            pts = np.array([vertex], dtype=np.int16)
            o = vertex.astype(np.uint16)
            graph.add_node(node_id, pts=pts, o=o)

        # Add edges with pts along the path
        for edge in edges:
            node1_id = f"{skel_id}_{edge[0]}"
            node2_id = f"{skel_id}_{edge[1]}"

            pos1 = vertices[edge[0]]
            pos2 = vertices[edge[1]]

            edge_pts = _interpolate_3d_path(pos1, pos2)
            graph.add_edge(node1_id, node2_id, pts=edge_pts, weight=len(edge_pts))

    return graph


def _interpolate_3d_path(p1, p2, num_points=None):
    """
    Create interpolated points along a 3D path between two points.
    Similar to how build_sknw creates edge pts.
    """
    p1, p2 = np.array(p1), np.array(p2)
    distance = np.linalg.norm(p2 - p1)

    if num_points is None:
        num_points = max(int(distance) + 1, 2)

    if num_points < 2:
        return np.array([p1, p2], dtype=np.int16)

    t_values = np.linspace(0, 1, num_points)
    points = np.array([p1 + t * (p2 - p1) for t in t_values], dtype=np.int16)

    return points


def _preprocess_for_kimimaro(mask):
    """Gentle preprocessing to prepare mask for kimimaro skeletonization."""
    if isinstance(mask, sitk.Image):
        mask_array = sitk.GetArrayFromImage(mask)
    else:
        mask_array = mask

    # Convert to binary if needed
    if len(np.unique(mask_array)) > 2:
        binary_mask = (mask_array > 0).astype(np.uint8)
    else:
        binary_mask = mask_array.astype(np.uint8)

    # Light preprocessing to preserve tree structure
    binary_mask = ndi.binary_fill_holes(
        binary_mask, structure=np.ones((3, 3, 3))
    ).astype(np.uint8)

    # Remove very small isolated components only
    labeled, num_features = ndi.label(binary_mask)
    if num_features > 1:
        sizes = ndi.sum(binary_mask, labeled, range(num_features + 1))
        mask_size = sizes > 50  # Keep components larger than 50 voxels
        remove_pixel = mask_size[labeled]
        binary_mask[~remove_pixel] = 0

    return binary_mask


def prepare_graph_kimimaro(mask, aggressive=True):
    """
    Create a graph from a mask using kimimaro skeletonization.

    This function provides an alternative to prepare_graph that uses kimimaro
    for skeletonization, which can capture more detailed branching structures.
    Returns a graph with the same structure as prepare_graph.

    Parameters:
    -----------
    mask : SimpleITK.Image or numpy.ndarray
        Binary mask of the structure to skeletonize
    aggressive : bool, default=True
        If True, use aggressive parameters to capture more branches

    Returns:
    --------
    nx.Graph
        Graph with nodes having 'pts' and 'o' attributes, compatible with prepare_graph
    """
    # Preprocess the mask
    binary_mask = _preprocess_for_kimimaro(mask)

    if aggressive:
        # Use the same aggressive parameters that worked well in the main script
        teasar_params = {
            "scale": 0.8,  # Very low scale for maximum detail
            "const": 100,  # Low const to preserve all branches
            "pdrf_exponent": 4,
            "pdrf_scale": 100000,
            "soma_detection_threshold": 800,  # Lower threshold
            "soma_acceptance_threshold": 1200,  # Lower acceptance
            "soma_invalidation_scale": 0.8,  # Less invalidation
            "soma_invalidation_const": 100,  # Less reduction around somas
            "max_paths": 300,  # Many more paths for complex branching
        }
        dust_threshold = 20

        # Check volume size to adapt parameters
        total_voxels = np.sum(binary_mask)
        if total_voxels < 500000:  # Smaller volume, use even more aggressive parameters
            teasar_params.update(
                {
                    "scale": 0.5,
                    "const": 50,
                    "soma_detection_threshold": 500,
                    "soma_acceptance_threshold": 800,
                    "max_paths": 500,
                }
            )
            dust_threshold = 10
    else:
        # Balanced parameters
        teasar_params = {
            "scale": 1.2,
            "const": 150,
            "pdrf_exponent": 4,
            "pdrf_scale": 100000,
            "soma_detection_threshold": 1500,
            "soma_acceptance_threshold": 2500,
            "soma_invalidation_scale": 1.0,
            "soma_invalidation_const": 200,
            "max_paths": 200,
        }
        dust_threshold = 50

    # Run kimimaro skeletonization
    skeleton = kimimaro.skeletonize(
        binary_mask,
        teasar_params=teasar_params,
        object_ids=[1],
        dust_threshold=dust_threshold,
        anisotropy=(1, 1, 1),
        fix_branching=True,
        fix_borders=True,
        progress=False,  # Silent for integration
    )

    # Check if we got enough detail, if not try alternative approach
    total_vertices = sum(len(skel.vertices) for skel in skeleton.values())
    total_voxels = np.sum(binary_mask)

    # Adaptive threshold based on volume size
    min_vertices_threshold = min(3000, total_voxels // 500)  # Scale with volume size

    if total_vertices < min_vertices_threshold and aggressive:
        # Try more aggressive parameters
        skeleton = kimimaro.skeletonize(
            binary_mask,
            teasar_params={
                "scale": 0.5,  # Even lower scale
                "const": 50,  # Even lower const
                "pdrf_exponent": 4,
                "pdrf_scale": 100000,
                "soma_detection_threshold": 500,  # Much lower
                "soma_acceptance_threshold": 800,  # Much lower
                "soma_invalidation_scale": 0.5,  # Minimal invalidation
                "soma_invalidation_const": 50,  # Minimal reduction
                "max_paths": 500,  # Maximum paths
            },
            object_ids=[1],
            dust_threshold=10,  # Keep almost everything
            anisotropy=(1, 1, 1),
            fix_branching=True,
            fix_borders=True,
            progress=False,
        )

    # Keep only the largest skeleton if multiple are found
    if len(skeleton) > 1:
        largest_skel_id = max(skeleton.keys(), key=lambda k: len(skeleton[k].vertices))
        skeleton = {largest_skel_id: skeleton[largest_skel_id]}

    # Convert to NetworkX graph with prepare_graph structure
    graph = _kimimaro_to_graph_structure(skeleton)

    # Keep only the largest connected component (like prepare_graph does)
    if graph.number_of_nodes() > 0:
        graph = keep_largest_component(graph)

    return graph
