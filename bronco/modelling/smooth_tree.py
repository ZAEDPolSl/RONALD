import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from bronco.modelling.fill_gaps import fill_gaps
from bronco.modelling.model_branch import BranchAnalyser
from bronco.modelling.prepare_graph import (
    prepare_graph,
    assign_thickness,
)
from bronco.modelling.kimimaro_graph import prepare_graph_kimimaro
from bronco.modelling.segment_branch import assign_branch
from bronco.modelling.branch_closing import apply_smoothing_by_node_order


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
    queue = [trachea_top_node]
    while queue:
        node = queue.pop(0)
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

    # Label connected components in airways_mask_arr
    sitk_airways_mask = sitk.GetImageFromArray(airways_mask_arr.astype(np.uint8))
    cc_filter = sitk.ConnectedComponentImageFilter()
    labeled = cc_filter.Execute(sitk_airways_mask)
    labeled_np = sitk.GetArrayFromImage(labeled)

    # Find all labels except background
    labels = np.unique(labeled_np)
    labels = labels[labels != 0]

    if verbose:
        print(f"  Found {len(labels)} connected components in airways mask")

    # Find overlap for each label (vectorized)
    smooth_mask_bool = smooth_mask.astype(bool)
    final_mask = smooth_mask.copy()
    connected_count = 0

    for label in tqdm(labels, desc="Processing components", disable=not verbose):
        comp = labeled_np == label
        # Vectorized overlap check
        if np.any(smooth_mask_bool & comp):
            final_mask = np.logical_or(final_mask, comp)
            connected_count += 1

    if verbose:
        print(f"  Added {connected_count} connected components to final mask")

    return final_mask.astype(np.uint8)


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
    bronco_mask_arr = sitk.GetArrayFromImage(bronco_mask)
    airways_mask_arr = sitk.GetArrayFromImage(airways_mask)
    tree_mask = np.zeros_like(bronco_mask_arr)
    smooth_mask = np.zeros_like(bronco_mask_arr)

    if verbose:
        print("Computing node order...")
    node_order = get_node_order(airways_graph)
    if verbose:
        print(f"Ordered {len(node_order)} nodes from trachea to periphery")
    node_to_order = {node: idx for idx, node in enumerate(node_order)}

    if verbose:
        print("Assigning branches to graph...")
    branches_mask, airways_graph = assign_branch(bronco_mask_arr, airways_graph)

    if verbose:
        print(f"Processing {len(node_order)} nodes for tree modeling...")

    # Track progress for processed branches
    processed_branches = 0
    total_branches = sum(
        1 for node in node_order for _ in airways_graph.neighbors(node)
    )

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
            current_branch_mask = edge["mask"]
            curr_mask = (branches_mask == current_branch_mask).astype(int)

            coord1 = tuple(airways_graph.nodes()[node]["o"])
            coord2 = tuple(airways_graph.nodes()[neighbor]["o"])
            curr_mask[coord1] = 1
            curr_mask[coord2] = 1
            analyser = BranchAnalyser(verbose=verbose)
            branch_mask, upper_ellipse, lower_ellipse, thickness = (
                analyser.smooth_branch(edge_points, curr_mask)
            )

            airways_graph.nodes[neighbor]["ellipse"] = lower_ellipse
            airways_graph.nodes[neighbor]["thickness"] = thickness
            np.logical_or(tree_mask, branch_mask, out=tree_mask)
            np.logical_or(smooth_mask, tree_mask, out=smooth_mask)
            if node_to_order[node] != 0 and "ellipse" in airways_graph.nodes[node]:
                prev_ellipse = airways_graph.nodes[node]["ellipse"]
                smooth_mask = fill_gaps(prev_ellipse, upper_ellipse, smooth_mask)

            # Try to free some memory periodically
            if processed_branches % 50 == 0:
                try:
                    import gc

                    gc.collect()
                except:
                    pass

    if verbose:
        print("Adding connected airways...")
    # Only add airways_mask_arr components that overlap with smooth_mask
    smooth_mask = add_connected_airways(smooth_mask, airways_mask_arr, verbose=verbose)

    if verbose:
        print("Assigning thickness to nodes...")
    airways_graph = assign_thickness(airways_graph, node_order)

    if verbose:
        print("Reassigning branches after smoothing...")
    branches_mask, airways_graph = assign_branch(smooth_mask, airways_graph)

    if verbose:
        print("Applying final smoothing...")
    smooth = apply_smoothing_by_node_order(
        airways_graph,
        branches_mask,
        node_order,
        thick_mult=2,  # Default value
    )

    if verbose:
        print(f"Tree modeling complete, final mask has {np.sum(smooth)} active voxels")

    return smooth


def smooth_tree(bronco_mask, airways_mask, verbose=True, skip_final_smoothing=False):
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
    skip_final_smoothing : bool
        Whether to skip the final smoothing step (useful for very large trees that cause OOM)

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

    # Ensure all masks are same type
    bronco_mask = sitk.Cast(bronco_mask, sitk.sitkUInt16)
    airways_mask = sitk.Cast(airways_mask, sitk.sitkUInt16)

    # Copy metadata from input image
    sitk_smooth.CopyInformation(bronco_mask)

    # Mask with original bronco mask to ensure no leakage
    sitk_smooth = sitk_smooth & bronco_mask

    if verbose:
        print("Tree smoothing completed successfully")

    return sitk_smooth
