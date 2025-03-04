import numpy as np
import SimpleITK as sitk
from skimage.morphology import skeletonize_3d
from tqdm import tqdm
from bronco.external.sknw import build_sknw
from bronco.modelling.model_branch import smooth_branch


def get_skeleton(mask):
    airways = sitk.GetArrayFromImage(mask)
    skeleton = skeletonize_3d(airways)
    skeleton = skeleton.astype(int)
    sitk_skeleton = sitk.GetImageFromArray(skeleton)
    sitk_skeleton.CopyInformation(mask)
    _sitk_skeleton = sitk.BinaryFillhole(sitk.Cast(sitk_skeleton > 0, sitk.sitkUInt8))
    _sitk_skeleton = sitk.Cast(
        (
            _sitk_skeleton
            - sitk.BinaryMorphologicalOpening(_sitk_skeleton, kernelRadius=(1, 1, 1))
        )
        > 0,
        sitk.sitkUInt8,
    )
    skeleton = sitk.GetArrayFromImage(_sitk_skeleton)
    airways_graph = build_sknw(skeleton, iso=False, ring=False, full=True)
    return airways_graph


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
    # TODO: get the top of the trachea if necessary

    airways_graph = get_skeleton(bronco_mask)
    bronco_mask_arr = sitk.GetArrayFromImage(bronco_mask)
    tree_mask = np.zeros_like(bronco_mask_arr)
    node_order = get_node_order(airways_graph)
    max_oval = {node: None for node in node_order}
    for node in tqdm(node_order):
        for neighbor in airways_graph.neighbors(node):
            # if neighbor is before node in node_order, then it was already processed
            if node_order.index(neighbor) < node_order.index(node):
                continue
            current_oval = max_oval[node]
            # get the edge between node and neighbor
            edge = airways_graph.get_edge_data(node, neighbor)
            # get the points in the edge
            edge_points = edge["pts"]
            # check the order of the points - if it starts at node, it is correct
            if not (edge_points[0] == airways_graph.nodes()[node]["o"]).all():
                edge_points = edge_points[::-1]
            # smooth the branch
            branch_mask, lower_oval = smooth_branch(
                edge_points, bronco_mask_arr, previous_oval=current_oval
            )
            # update the max_oval
            max_oval[neighbor] = lower_oval
            # add the branch to the tree_mask
            tree_mask = np.logical_or(tree_mask, branch_mask)
    # TODO: connect the branches
    return tree_mask.astype(int)


def smooth_tree(bronco_mask):
    tree_mask = model_tree(bronco_mask)
    sitk_tree_mask = sitk.GetImageFromArray(tree_mask)
    sitk_tree_mask = sitk.Cast(sitk_tree_mask, sitk.sitkUInt16)
    sitk_tree_mask.CopyInformation(bronco_mask)

    np_mask = sitk.GetArrayFromImage(bronco_mask)
    smoothed_tree = np.logical_and(tree_mask, np_mask).astype(int)
    sitk_smoothed_tree = sitk.GetImageFromArray(smoothed_tree)
    sitk_smoothed_tree = sitk.Cast(sitk_smoothed_tree, sitk.sitkUInt16)
    sitk_smoothed_tree.CopyInformation(bronco_mask)

    return sitk_smoothed_tree, sitk_tree_mask
