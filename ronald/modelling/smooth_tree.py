from collections import deque

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from ronald.modelling.branch_closing import (
    apply_smoothing_by_node_order,
    find_label_bboxes,
)
from ronald.modelling.fill_gaps import fill_gaps
from ronald.modelling.model_branch import BranchAnalyser
from ronald.modelling.prepare_graph import (
    assign_thickness,
    foreground_bbox,
    prepare_graph,
    sitk_region_from_array_bbox,
)
from ronald.modelling.segment_branch import assign_branch

COMPONENT_CHUNK_DEPTH = 16


def get_node_order(graph):
    """
    Determine processing order of nodes starting from the top of the trachea.

    Uses breadth-first traversal to order nodes by their distance from the top
    of the trachea (node with highest z-coordinate).

    Parameters:
    -----------
    graph : networkx.Graph
        The airway tree graph

    Returns:
    --------
    list
        Nodes in order of processing (from trachea to peripheral branches)
    """
    # Find the top of the trachea (node with highest z-coordinate)
    trachea_top_node = max(graph.nodes(), key=lambda node: graph.nodes[node]["o"][0])

    # Initialize distances with infinity for all nodes
    distances = {node: float("inf") for node in graph.nodes()}
    distances[trachea_top_node] = 0

    # Breadth-first traversal to compute distances
    queue = deque([trachea_top_node])
    while queue:
        node = queue.popleft()
        for neighbor in graph.neighbors(node):
            if distances[neighbor] > distances[node] + 1:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)

    # Sort nodes by distance from the top of the trachea
    sorted_nodes = sorted(graph.nodes(), key=lambda node: distances[node])
    return sorted_nodes


def add_connected_airways(smooth_mask, airways_mask_arr, verbose=False):
    """
    Add only those components from airways_mask_arr that are connected to smooth_mask.
    Returns the updated mask.
    """
    if verbose:
        print("  Finding connected components in airways mask...")

    if isinstance(airways_mask_arr, sitk.Image):
        airways_array = sitk.GetArrayViewFromImage(airways_mask_arr)
        bbox = foreground_bbox(airways_array)
        if bbox is None:
            return smooth_mask.astype(np.uint8, copy=True)
        cropped_airways = sitk_region_from_array_bbox(airways_mask_arr, bbox)
        sitk_airways_mask = sitk.Cast(cropped_airways > 0, sitk.sitkUInt8)
    else:
        bbox = foreground_bbox(airways_mask_arr)
        if bbox is None:
            return smooth_mask.astype(np.uint8, copy=True)
        sitk_airways_mask = sitk.GetImageFromArray(
            airways_mask_arr[bbox].astype(np.uint8, copy=False)
        )
    cc_filter = sitk.ConnectedComponentImageFilter()
    labeled = cc_filter.Execute(sitk_airways_mask)
    labeled_np = sitk.GetArrayViewFromImage(labeled)

    smooth_mask_bool = smooth_mask.astype(bool, copy=False)
    smooth_roi = smooth_mask_bool[bbox]
    connected_labels = np.unique(labeled_np[smooth_roi])
    connected_labels = connected_labels[connected_labels != 0]

    if verbose:
        print(
            f"  Found {cc_filter.GetObjectCount()} connected components in airways mask"
        )

    final_mask = smooth_mask_bool.astype(np.uint8, copy=True)
    if len(connected_labels):
        label_lut = np.zeros(cc_filter.GetObjectCount() + 1, dtype=bool)
        label_lut[connected_labels] = True
        final_roi = final_mask[bbox]
        for start in range(0, final_roi.shape[0], COMPONENT_CHUNK_DEPTH):
            stop = min(start + COMPONENT_CHUNK_DEPTH, final_roi.shape[0])
            np.logical_or(
                final_roi[start:stop],
                label_lut[labeled_np[start:stop]],
                out=final_roi[start:stop],
            )

    if verbose:
        print(f"  Added {len(connected_labels)} connected components to final mask")

    return final_mask


def _points_for_branch(branches_mask, branch_label, branch_bboxes, endpoints):
    branch_index = branch_label - 1
    bbox = branch_bboxes[branch_index] if branch_index < len(branch_bboxes) else None
    if bbox is None:
        points = np.empty((0, 3), dtype=np.intp)
    else:
        points = np.argwhere(branches_mask[bbox] == branch_label)
        points += np.array([axis.start for axis in bbox])
    return np.unique(np.vstack((points, endpoints)), axis=0)


def model_tree(bronco_mask, airways_mask, verbose=False):
    """
    Create a smooth model of the airway tree.

    Parameters:
    -----------
    bronco_mask : SimpleITK.Image
        Bronchoscopy mask image
    airways_mask : SimpleITK.Image
        Airways mask image
    verbose : bool
        Whether to print detailed progress information

    Returns:
    --------
    numpy.ndarray
        Smoothed mask
    """
    if verbose:
        print("Preparing airway tree graph...")
    airways_graph = prepare_graph(bronco_mask)

    # Check if graph has too many nodes and switch to kimimaro if needed
    num_nodes = airways_graph.number_of_nodes()
    if num_nodes > 211:
        from ronald.modelling.kimimaro_graph import prepare_graph_kimimaro

        if verbose:
            print(
                f"Graph has {num_nodes} nodes (>211), switching to kimimaro for better handling..."
            )
        else:
            print("Switching to kimimaro graph...")
        airways_graph = prepare_graph_kimimaro(bronco_mask, aggressive=True)
        if verbose:
            print(f"Kimimaro graph has {airways_graph.number_of_nodes()} nodes")

    if verbose:
        print("Converting mask images to numpy arrays...")
    bronco_mask_arr = sitk.GetArrayViewFromImage(bronco_mask)
    smooth_mask = np.zeros_like(bronco_mask_arr, dtype=bool)

    if verbose:
        print("Computing node order...")
    node_order = get_node_order(airways_graph)
    if verbose:
        print(f"Ordered {len(node_order)} nodes from trachea to periphery")
    node_to_order = {node: idx for idx, node in enumerate(node_order)}

    if verbose:
        print("Assigning branches to graph...")
    branch_label_dtype = np.min_scalar_type(max(1, airways_graph.number_of_edges()))
    branches_mask, airways_graph = assign_branch(
        bronco_mask_arr, airways_graph, label_dtype=branch_label_dtype
    )
    max_branch_label = int(branches_mask.max())
    branch_bboxes = find_label_bboxes(branches_mask, max_label=max_branch_label)

    if verbose:
        print(f"Processing {len(node_order)} nodes for tree modeling...")

    processed_branches = 0
    total_branches = airways_graph.number_of_edges()

    for node in tqdm(node_order, desc="Modeling tree branches", disable=not verbose):
        for neighbor in airways_graph.neighbors(node):
            if node_to_order[neighbor] < node_to_order[node]:
                continue

            processed_branches += 1
            if verbose and processed_branches % 10 == 0:
                print(f"  Processed {processed_branches}/{total_branches} branches")

            edge = airways_graph.get_edge_data(node, neighbor)
            edge_points = edge["pts"]
            if not (edge_points[0] == airways_graph.nodes()[node]["o"]).all():
                edge_points = edge_points[::-1]
            branch_label = edge["mask"]
            endpoints = np.array(
                [
                    airways_graph.nodes[node]["o"],
                    airways_graph.nodes[neighbor]["o"],
                ]
            )
            branch_points = _points_for_branch(
                branches_mask, branch_label, branch_bboxes, endpoints
            )

            analyser = BranchAnalyser(verbose=verbose)
            branch_points, upper_ellipse, lower_ellipse, thickness, branch_gaps = (
                analyser.smooth_branch_points(edge_points, branch_points)
            )

            airways_graph.nodes[neighbor]["ellipse"] = lower_ellipse
            airways_graph.nodes[neighbor]["thickness"] = thickness
            if len(branch_points):
                smooth_mask[tuple(branch_points.T)] = True
            for lower, upper in branch_gaps:
                fill_gaps(lower, upper, smooth_mask, cast_to_int=False)
            if node_to_order[node] != 0 and "ellipse" in airways_graph.nodes[node]:
                prev_ellipse = airways_graph.nodes[node]["ellipse"]
                fill_gaps(prev_ellipse, upper_ellipse, smooth_mask, cast_to_int=False)

    if verbose:
        print("Adding connected airways...")
    smooth_mask = add_connected_airways(smooth_mask, airways_mask, verbose=verbose)

    if verbose:
        print("Assigning thickness to nodes...")
    airways_graph = assign_thickness(airways_graph, node_order)

    if verbose:
        print("Reassigning branches after smoothing...")
    branches_mask, airways_graph = assign_branch(
        smooth_mask, airways_graph, label_dtype=branch_label_dtype
    )

    if verbose:
        print("Applying final smoothing...")
    smooth = apply_smoothing_by_node_order(
        airways_graph,
        branches_mask,
        node_order,
        thick_mult=2,
        output_dtype=np.uint8,
        verbose=verbose,
    )

    if verbose:
        print(f"Tree modeling complete, final mask has {np.sum(smooth)} active voxels")

    return smooth


def smooth_tree(bronco_mask, airways_mask, verbose=False):
    """
    Create a smoothed airway tree model from mask images.

    This is the main function to call for creating a smoothed airway tree.

    Parameters:
    -----------
    bronco_mask : SimpleITK.Image
        Bronchoscopy mask image
    airways_mask : SimpleITK.Image
        Airways mask image
    verbose : bool
        Whether to print detailed progress information
    Returns:
    --------
    SimpleITK.Image
        Smoothed airway tree image
    """
    if verbose:
        print("Starting tree smoothing process...")

    # Run the model_tree function to get the smoothed mask
    smooth = model_tree(bronco_mask, airways_mask, verbose=verbose)

    if verbose:
        print("Converting smooth mask to SimpleITK image...")

    # Convert numpy array to SimpleITK image
    sitk_smooth = sitk.GetImageFromArray(smooth)
    sitk_smooth = sitk.Cast(sitk_smooth, sitk.sitkUInt16)

    # Ensure masks used in the final intersection are the same type.
    bronco_mask = sitk.Cast(bronco_mask, sitk.sitkUInt16)

    # Copy metadata from input image
    sitk_smooth.CopyInformation(bronco_mask)

    # Mask with original airway mask to ensure no leakage
    sitk_smooth = sitk_smooth & bronco_mask

    if verbose:
        print("Tree smoothing completed successfully")

    return sitk_smooth
