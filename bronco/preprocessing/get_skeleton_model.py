import os
import pickle
import itertools
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
from collections import defaultdict

from skimage.measure import label
from skimage.morphology import skeletonize_3d
from sklearn.metrics.pairwise import cosine_similarity

import bronco.external.sknw as sknw
from bronco.utils_bronchi import dilation_by_slice


def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def skeleton_graph_analysis(sitk_lungs, sitk_gmm_seg, ii=None, return_binary=True, path_save=None):
    lungs = sitk.GetArrayFromImage(sitk_lungs)
    gmm_seg = sitk.GetArrayFromImage(sitk_gmm_seg)

    lungs = np.moveaxis(lungs, 0, 2)
    gmm_seg = np.moveaxis(gmm_seg, 0, 2)

    _gmm_seg = gmm_seg.copy()
    _lungs = lungs.copy()

    _lungs[_lungs != _lungs.min()] = 1
    _lungs[_lungs == _lungs.min()] = 0

    _lungs = _lungs.astype(np.uint8)

    _gmm_seg[_lungs == 0] = 0

    labelled_volume = label(_gmm_seg)
    uniq, counts = np.unique(labelled_volume, return_counts=True)

    # Pair ids with their counts in the volume
    labels = list(zip(uniq[1:], counts[1:]))
    labels = sorted(labels, key=lambda x: x[1])

    # Get two largest volumes - left bronchi and right bronchi
    id_bronchi_1, id_bronchi_2 = labels[-1][0], labels[-2][0]
    id_bronchi_3, id_bronchi_4 = labels[-3][0], labels[-4][0]
    id_bronchi_5, id_bronchi_6 = labels[-5][0], labels[-6][0]

    _gmm_seg[(labelled_volume != id_bronchi_1) & (labelled_volume != id_bronchi_2) &
             (labelled_volume != id_bronchi_3) & (labelled_volume != id_bronchi_4) &
             (labelled_volume != id_bronchi_5) & (labelled_volume != id_bronchi_6)] = 0

    _gmm_seg[(labelled_volume == id_bronchi_1) | (labelled_volume == id_bronchi_2) |
             (labelled_volume == id_bronchi_3) | (labelled_volume == id_bronchi_4) |
             (labelled_volume == id_bronchi_5) | (labelled_volume == id_bronchi_6)] = 1
    '''number_of_elems = 1
    largest_bronchi_elems = np.zeros_like(_gmm_seg)
    for l in labels[::-1]:
        if l[1] > 100:
            largest_bronchi_elems[labelled_volume == l[0]] = 1
            number_of_elems += 1
    _gmm_seg = largest_bronchi_elems'''

    skeleton = skeletonize_3d(_gmm_seg)

    if path_save is not None and ii is not None:
        _sitk_gmm_seg = sitk.Cast(sitk.GetImageFromArray(np.moveaxis(_gmm_seg, 2, 0)), sitk.sitkUInt8)
        _sitk_gmm_seg.CopyInformation(sitk_lungs)
        ii.write(_sitk_gmm_seg, os.path.join(path_save, 'bronchovascular_bundle_scaffolding'))
        _sitk_skeleton = sitk.Cast(sitk.GetImageFromArray(np.moveaxis(skeleton, 2, 0)), sitk.sitkUInt8)
        _sitk_skeleton.CopyInformation(sitk_lungs)
        ii.write(_sitk_skeleton, os.path.join(path_save, "skeleton_3d_without_borders"))

    no_orthogonal_directions = 2

    skeleton_graph = sknw.build_sknw(skeleton)
    graph = skeleton_graph.copy()

    # network_plot_3D(graph, 120)
    edges_unpacked = list(graph.edges())
    nodes_unpacked = list(graph.nodes())

    def normalize_vec(v):
        return v / np.sqrt(np.sum(v ** 2))

    base_directions = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])

    edge_ids_dict = dict()
    dominant_direction_branch_dict = defaultdict(list)
    skeleton_labelled = skeleton.copy()
    skeleton_labelled = skeleton_labelled.astype(np.int32)

    i = 2
    edge_id = 2

    for i in tqdm(range(len(edges_unpacked)), desc="Searching dominant directions"):
        edge_node_ids = edges_unpacked[i]
        start_node_id = edge_node_ids[0]
        end_node_id = edge_node_ids[1]

        if start_node_id != end_node_id:
            edge_points = graph[start_node_id][end_node_id]['pts']
            starting_node_point = graph.nodes()[start_node_id]['o']
            end_node_point = graph.nodes()[end_node_id]['o']

            directional_vector = np.array(end_node_point) - np.array(starting_node_point)

            normalized_dir_vector = normalize_vec(directional_vector)

            similarity_coeffs = np.abs(cosine_similarity(normalized_dir_vector[:, np.newaxis].T, base_directions))

            # orthogonal_direction = base_directions[np.argmin(similarity_coeffs)]
            orthogonal_axes = np.argpartition(similarity_coeffs, no_orthogonal_directions).flatten()[
                              :no_orthogonal_directions]

            orthogonal_directions = base_directions[orthogonal_axes]

            dominant_direction_axis = np.argmax(similarity_coeffs)

            dominant_direction_branch_dict[dominant_direction_axis].append(edge_node_ids)

            edge_ids_dict[edge_node_ids] = edge_id
            edge_id += 1

            x, y, z = zip(*edge_points)
            skeleton_labelled[x, y, z] = edge_id

    dominant_direction_edge_id_dict = defaultdict(list)
    for dominant_direction, edge_node_ids in dominant_direction_branch_dict.items():
        for edge_node_id in edge_node_ids:
            dominant_direction_edge_id_dict[dominant_direction].append(edge_ids_dict[edge_node_id])

    if path_save is not None and ii is not None:
        save_object(edge_ids_dict, f'{str(path_save)}/edge_ids_dict.pck')
        save_object(dominant_direction_branch_dict, f'{str(path_save)}/dominant_direction_branch_dict.pck')
        _sitk_skeleton = sitk.Cast(sitk.GetImageFromArray(np.moveaxis(skeleton_labelled, 2, 0)), sitk.sitkInt16)
        _sitk_skeleton.CopyInformation(sitk_lungs)
        ii.write(_sitk_skeleton, f'{str(path_save)}/skeleton_labelled')
        save_object(dominant_direction_edge_id_dict, f'{str(path_save)}/dominant_direction_edge_id_dict.pck')

    number_set = [-1, 0, 1]
    combs = list(itertools.product(number_set, repeat=2))
    combs.remove((0, 0))

    print(f"Possible combinations: {combs}")

    skeleton_grown = skeleton_labelled.copy()
    mask_labelled = _gmm_seg.copy()
    mask_labelled = mask_labelled.astype(np.int32)
    gmm_original_mask = _gmm_seg.copy()
    gmm_condition = np.equal(gmm_original_mask, 1)
    mask = np.greater(skeleton_grown, 1)
    mask_labelled[mask] = skeleton_labelled[mask]

    print(f"Total number of labelled edged: {len(np.unique(mask_labelled))}")

    axes = [0, 1, 2]

    growth_range = 20
    print(f"Growing with range {growth_range}...")
    prog_bar = tqdm(total=growth_range, desc="Growing edges")
    for i in range(growth_range):
        for main_direction in axes:
            axes = [0, 1, 2]
            axes.remove(main_direction)
            edge_ids_considered = dominant_direction_edge_id_dict[main_direction]

            #         print(f"Len of edge ids considered in this iteration: {len(edge_ids_considered)}")

            labelled_direction_mask_iter = mask_labelled.copy()
            direction_mask = ~np.isin(labelled_direction_mask_iter, edge_ids_considered)
            labelled_direction_mask_iter[direction_mask] = 0

            #         print(f"Number of unique ids in this iteration: {len(np.unique(labelled_direction_mask_iter))-1}")

            for directions in combs:
                directions = list(directions)

                shifted_labelled_direction_mask_iter = labelled_direction_mask_iter.copy()
                shifted_labelled_direction_mask_iter = np.roll(shifted_labelled_direction_mask_iter,
                                                               shift=directions.pop(), axis=axes[0])
                shifted_labelled_direction_mask_iter = np.roll(shifted_labelled_direction_mask_iter,
                                                               shift=directions.pop(), axis=axes[1])

                shifted_mask = np.greater(shifted_labelled_direction_mask_iter, 1)
                free_voxels = np.equal(mask_labelled, 1)

                final_mask = np.logical_and(shifted_mask, free_voxels)

                mask_labelled[final_mask] = shifted_labelled_direction_mask_iter[final_mask]
        prog_bar.update()

    branch_id_to_direction_dict = dict()

    mask_labelled[mask_labelled == 1] = 0
    if return_binary:
        # mask binarization
        mask_labelled[mask_labelled > 0] = 1
    mask_labelled = np.array(mask_labelled, dtype=np.int16)
    mask_labelled = np.moveaxis(mask_labelled, 2, 0)
    sitk_mask_labelled = sitk.GetImageFromArray(mask_labelled)
    sitk_mask_labelled.CopyInformation(sitk_lungs)

    for main_direction, branch_edges_ids in dominant_direction_branch_dict.items():

        for branch_edge_id in branch_edges_ids:
            branch_id = edge_ids_dict[branch_edge_id]

            direction = main_direction
            branch_id_to_direction_dict[branch_id] = direction

    if path_save is not None and ii is not None:
        save_object(branch_id_to_direction_dict, f'{str(path_save)}/branch_id_to_direction_dict.pck')

    return sitk_mask_labelled
