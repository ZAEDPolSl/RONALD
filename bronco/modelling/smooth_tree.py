import numpy as np
import SimpleITK as sitk
from skimage.morphology import skeletonize, closing, ball
from tqdm import tqdm
from bronco.external.sknw import build_sknw
from bronco.modelling.model_branch import smooth_branch
from bronco.modelling.segment_branch import assign_branch


def get_skeleton(mask):
    airways = sitk.GetArrayFromImage(mask)
    skeleton = skeletonize(airways)
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
    airways_graph = get_skeleton(bronco_mask)
    bronco_mask_arr = sitk.GetArrayFromImage(bronco_mask)
    tree_mask = np.zeros_like(bronco_mask_arr)
    node_order = get_node_order(airways_graph)
    # define branch_mask and get modified airways_graph
    branches_mask, airways_graph, min_dist_img = assign_branch(
        bronco_mask_arr, airways_graph
    )
    # min_dist_img = sitk.GetImageFromArray(min_dist_img)
    # min_dist_img.CopyInformation(bronco_mask)
    # sitk.WriteImage(min_dist_img, "distance_map.nrrd")

    # b_mask = sitk.GetImageFromArray(branches_mask)
    # b_mask.CopyInformation(bronco_mask)
    # sitk.WriteImage(b_mask, "branches_mask.nrrd")
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
            # branch_mask = smooth_branch(edge_points, bronco_mask_arr)
            branch_mask, points_up, points_down = smooth_branch(
                edge_points, curr_mask, True
            )
            # points down should be assigned to the lower node
            airways_graph.nodes()[neighbor]["ellipse"] = points_down
            prev_down = airways_graph.nodes()[node]["ellipse"]

            # points up - use with points down of the upper node
            tree_mask = np.logical_or(tree_mask, branch_mask)
    return tree_mask.astype(int)


def smooth_tree(bronco_mask):
    tree_mask = model_tree(bronco_mask)
    sitk_tree_mask = sitk.GetImageFromArray(tree_mask)
    sitk_tree_mask = sitk.Cast(sitk_tree_mask, sitk.sitkUInt16)
    sitk_tree_mask.CopyInformation(bronco_mask)

    sitk_smoothed_tree = sitk_tree_mask
    return sitk_smoothed_tree, sitk_tree_mask
