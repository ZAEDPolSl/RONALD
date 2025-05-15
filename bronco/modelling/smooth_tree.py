import numpy as np
import SimpleITK as sitk
from skimage.morphology import skeletonize
from tqdm import tqdm
from bronco.modelling.model_branch import BranchAnalyser
from bronco.modelling.segment_branch import assign_branch
from bronco.modelling.fill_gaps import fill_gaps
from bronco.modelling.prepare_graph import prepare_graph

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


def model_tree(bronco_mask):
    airways_graph = prepare_graph(bronco_mask)
    bronco_mask_arr = sitk.GetArrayFromImage(bronco_mask)
    tree_mask = np.zeros_like(bronco_mask_arr)
    smooth_mask = np.zeros_like(bronco_mask_arr)
    node_order = get_node_order(airways_graph)
    branches_mask, airways_graph, min_dist_img = assign_branch(
        bronco_mask_arr, airways_graph
    )

    for node in tqdm(node_order):
        for neighbor in airways_graph.neighbors(node):
            # if neighbor is before node in node_order, then it was already processed
            if node_order.index(neighbor) < node_order.index(node):
                continue
            # get the edge between node and neighbor
            edge = airways_graph.get_edge_data(node, neighbor)
            # get the points in the edge
            edge_points = edge["pts"]
            # check the order of the points - if it starts at node, it is correct
            if not (edge_points[0] == airways_graph.nodes()[node]["o"]).all():
                edge_points = edge_points[::-1]
            current_branch_mask = edge["mask"]
            curr_mask = (branches_mask == current_branch_mask).astype(int)

            # Extract coordinates
            coord1 = tuple(airways_graph.nodes()[node]["o"])
            coord2 = tuple(airways_graph.nodes()[neighbor]["o"])

            # Set the corresponding points to 1 in the mask
            curr_mask[coord1] = 1
            curr_mask[coord2] = 1
            analyser = BranchAnalyser()  # segments=True by default
            branch_mask, upper_ellipse, lower_ellipse = analyser.smooth_branch(
                edge_points, curr_mask
            )

            airways_graph.nodes[neighbor]["ellipse"] = lower_ellipse
            # add the branch to the tree_mask
            tree_mask = np.logical_or(tree_mask, branch_mask)
            smooth_mask = np.logical_or(smooth_mask, tree_mask)
            if node_order.index(node) != 0 and "ellipse" in airways_graph.nodes[node]:
                prev_ellipse = airways_graph.nodes[node]["ellipse"]
                smooth_mask = fill_gaps(prev_ellipse, upper_ellipse, smooth_mask)
    return tree_mask.astype(int), smooth_mask


def smooth_tree(bronco_mask):
    tree_mask, smooth_mask = model_tree(bronco_mask)
    sitk_tree_mask = sitk.GetImageFromArray(tree_mask)
    sitk_tree_mask = sitk.Cast(sitk_tree_mask, sitk.sitkUInt16)
    sitk_tree_mask.CopyInformation(bronco_mask)

    sitk_smoothed_tree = sitk.GetImageFromArray(smooth_mask)
    sitk_smoothed_tree = sitk.Cast(sitk_smoothed_tree, sitk.sitkUInt16)
    sitk_smoothed_tree.CopyInformation(bronco_mask)

    return sitk_smoothed_tree, sitk_tree_mask
