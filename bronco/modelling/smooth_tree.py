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
    trachea_top_node = max(graph.nodes(), key=lambda node: graph.nodes[node]["o"][0])
    distances = {node: float("inf") for node in graph.nodes()}
    distances[trachea_top_node] = 0

    queue = [trachea_top_node]
    while queue:
        node = queue.pop(0)
        for neighbor in graph.neighbors(node):
            if distances[neighbor] > distances[node] + 1:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)

    # Sort nodes by distance
    sorted_nodes = sorted(graph.nodes(), key=lambda node: distances[node])
    return sorted_nodes


def add_connected_airways(smooth_mask, airways_mask_arr):
    """
    Add only those components from airways_mask_arr that are connected to smooth_mask.
    Returns the updated mask.
    """
    # Label connected components in airways_mask_arr
    sitk_airways_mask = sitk.GetImageFromArray(airways_mask_arr.astype(np.uint8))
    cc_filter = sitk.ConnectedComponentImageFilter()
    labeled = cc_filter.Execute(sitk_airways_mask)
    labeled_np = sitk.GetArrayFromImage(labeled)

    # Find all labels except background
    labels = np.unique(labeled_np)
    labels = labels[labels != 0]

    # Find overlap for each label (vectorized)
    smooth_mask_bool = smooth_mask.astype(bool)
    final_mask = smooth_mask.copy()
    for label in labels:
        comp = labeled_np == label
        # Vectorized overlap check
        if np.any(smooth_mask_bool & comp):
            final_mask = np.logical_or(final_mask, comp)

    return final_mask.astype(np.uint8)


def model_tree(bronco_mask, airways_mask):
    airways_graph = prepare_graph(bronco_mask)

    # Check if graph has too many nodes and switch to kimimaro if needed
    num_nodes = airways_graph.number_of_nodes()
    if num_nodes > 1000:
        print(
            f"Graph has {num_nodes} nodes (>1000), switching to kimimaro for better handling..."
        )
        airways_graph = prepare_graph_kimimaro(bronco_mask, aggressive=True)
        print(f"Kimimaro graph has {airways_graph.number_of_nodes()} nodes")

    bronco_mask_arr = sitk.GetArrayFromImage(bronco_mask)
    airways_mask_arr = sitk.GetArrayFromImage(airways_mask)
    tree_mask = np.zeros_like(bronco_mask_arr)
    smooth_mask = np.zeros_like(bronco_mask_arr)
    node_order = get_node_order(airways_graph)
    node_to_order = {node: idx for idx, node in enumerate(node_order)}
    branches_mask, airways_graph = assign_branch(bronco_mask_arr, airways_graph)

    print(f"Processing {len(node_order)} nodes for tree modeling...")
    for node in tqdm(node_order, desc="Modeling tree"):
        for neighbor in airways_graph.neighbors(node):
            if node_to_order[neighbor] < node_to_order[node]:
                continue
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
            analyser = BranchAnalyser()
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
    print("Adding connected airways...")
    # Only add airways_mask_arr components that overlap with smooth_mask
    smooth_mask = add_connected_airways(smooth_mask, airways_mask_arr)
    airways_graph = assign_thickness(airways_graph, node_order)
    branches_mask, airways_graph = assign_branch(smooth_mask, airways_graph)

    print("Applying final smoothing...")
    smooth = apply_smoothing_by_node_order(airways_graph, branches_mask, node_order)
    return smooth


def smooth_tree(bronco_mask, airways_mask):
    smooth = model_tree(bronco_mask, airways_mask)
    sitk_smooth = sitk.GetImageFromArray(smooth)
    sitk_smooth = sitk.Cast(sitk_smooth, sitk.sitkUInt16)
    bronco_mask = sitk.Cast(bronco_mask, sitk.sitkUInt16)
    airways_mask = sitk.Cast(airways_mask, sitk.sitkUInt16)
    sitk_smooth.CopyInformation(bronco_mask)
    sitk_smooth = sitk_smooth & bronco_mask
    return sitk_smooth
