"""Sato-guided elliptical tubes expanded into the filled airway walls.

The Sato mask supplies skeleton, PCA and recovery. Filled walls supply fitting
support and the final output boundary. Only the connected main tree is returned.
"""

from collections import Counter
from contextlib import contextmanager

import networkx as nx
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from ronald.modelling.branch_closing import (
    apply_smoothing_by_node_order,
    find_label_bboxes,
)
from ronald.modelling.component_connections import connect_sato_components
from ronald.modelling.connected_tree import retain_main_connected_tree
from ronald.modelling.graph_chains import merge_degree_two_chains
from ronald.modelling.local_cross_section import LocalCrossSectionSampler
from ronald.modelling.local_tube_joins import fill_local_tube_join
from ronald.modelling.model_branch import BranchAnalyser
from ronald.modelling.prepare_graph import (
    assign_thickness,
    foreground_bbox,
    prepare_graph,
    sitk_region_from_array_bbox,
)
from ronald.modelling.segment_branch import assign_branch
from ronald.modelling.sato import calculate_sato_mask

COMPONENT_CHUNK_DEPTH = 16


def get_node_order(graph):
    """Traverse each Sato component from its highest-z node, largest first."""
    ordered = []
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    for component in components:
        sub = graph.subgraph(component)
        root = max(sub.nodes(), key=lambda node: sub.nodes[node]["o"][0])
        ordered.extend(nx.bfs_tree(sub, root).nodes())
    return ordered


def restore_connected_sato(smooth_mask, sato_mask, verbose=False):
    """Restore whole 6-connected Sato components overlapping the fitted model.

    Main-tree connectivity is enforced after closing and supported bridges.
    Merely adjacent source components are not restored here.
    """
    if verbose:
        print("  Finding connected components in Sato mask...")

    if isinstance(sato_mask, sitk.Image):
        sato_array = sitk.GetArrayViewFromImage(sato_mask)
        bbox = foreground_bbox(sato_array)
        if bbox is None:
            return smooth_mask.astype(np.uint8, copy=True)
        cropped_sato = sitk_region_from_array_bbox(sato_mask, bbox)
        sitk_sato_mask = sitk.Cast(cropped_sato > 0, sitk.sitkUInt8)
    else:
        bbox = foreground_bbox(sato_mask)
        if bbox is None:
            return smooth_mask.astype(np.uint8, copy=True)
        sitk_sato_mask = sitk.GetImageFromArray(
            sato_mask[bbox].astype(np.uint8, copy=False)
        )
    cc_filter = sitk.ConnectedComponentImageFilter()
    labeled = cc_filter.Execute(sitk_sato_mask)
    labeled_np = sitk.GetArrayViewFromImage(labeled)

    smooth_mask_bool = smooth_mask.astype(bool, copy=False)
    smooth_roi = smooth_mask_bool[bbox]
    connected_labels = np.unique(labeled_np[smooth_roi])
    connected_labels = connected_labels[connected_labels != 0]

    if verbose:
        print(f"  Found {cc_filter.GetObjectCount()} connected components in Sato mask")

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


def _points_for_branch(
    branches_mask, branch_label, branch_bboxes, endpoints, *, support_mask=None
):
    branch_index = branch_label - 1
    bbox = branch_bboxes[branch_index] if branch_index < len(branch_bboxes) else None
    if bbox is None:
        points = np.empty((0, 3), dtype=np.intp)
    else:
        selected = branches_mask[bbox] == branch_label
        if support_mask is not None:
            selected &= support_mask[bbox] > 0
        points = np.argwhere(selected)
        points += np.array([axis.start for axis in bbox])
    return np.unique(np.vstack((points, endpoints)), axis=0)


@contextmanager
def _limited_native_threads(max_threads):
    """Temporarily cap SimpleITK and BLAS/OpenMP worker threads."""
    if max_threads is None:
        yield
        return
    if isinstance(max_threads, bool) or not isinstance(max_threads, int):
        raise TypeError("max_threads must be a positive integer or None")
    if max_threads < 1:
        raise ValueError("max_threads must be at least 1")

    previous_sitk_threads = sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(max_threads)
    try:
        try:
            from threadpoolctl import threadpool_limits
        except ImportError:
            yield
        else:
            with threadpool_limits(limits=max_threads):
                yield
    finally:
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(previous_sitk_threads)


def _validate_masks(walls_filled, sato_mask):
    """Validate aligned, binary 3-D images with Sato contained in the walls."""
    for name, image in (("walls_filled", walls_filled), ("sato_mask", sato_mask)):
        if not isinstance(image, sitk.Image):
            raise TypeError(f"{name} must be a SimpleITK.Image")
        if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
            raise ValueError(f"{name} must be a scalar 3D image")
    if sato_mask.GetSize() != walls_filled.GetSize():
        raise ValueError("sato_mask size does not match walls_filled")
    for attribute in ("Spacing", "Origin", "Direction"):
        actual = getattr(sato_mask, f"Get{attribute}")()
        expected = getattr(walls_filled, f"Get{attribute}")()
        if not np.allclose(actual, expected, rtol=0, atol=1e-6):
            raise ValueError(
                f"sato_mask {attribute.lower()} does not match walls_filled"
            )
    walls = sitk.GetArrayViewFromImage(walls_filled)
    sato = sitk.GetArrayViewFromImage(sato_mask)
    for start in range(0, walls.shape[0], COMPONENT_CHUNK_DEPTH):
        block = slice(start, start + COMPONENT_CHUNK_DEPTH)
        for name, array in (("walls_filled", walls), ("sato_mask", sato)):
            if np.any((array[block] != 0) & (array[block] != 1)):
                raise ValueError(f"{name} must be binary (0 and 1)")
        if np.any((sato[block] != 0) & (walls[block] == 0)):
            raise ValueError("sato_mask foreground must be a subset of walls_filled")


def model_tree(
    walls_filled, sato_mask, verbose=False, max_workers=1, *, diagnostics=None
):
    """Fit and join tubes, recover connected Sato, then keep the main tree.

    Input images must share a grid and contain binary values. Sato must be a
    subset of walls_filled. An optional diagnostics observer receives stages
    and selected geometry without changing the output. Returns a uint8 array.
    """
    _validate_masks(walls_filled, sato_mask)
    walls = sitk.GetArrayViewFromImage(walls_filled)
    sato = sitk.GetArrayViewFromImage(sato_mask)
    if not np.any(sato):
        empty = np.zeros(walls.shape, dtype=np.uint8)
        if diagnostics is not None:
            for stage in (
                "02_after_joins",
                "03_after_sato_recovery",
                "04_after_closing",
                "05_after_component_connections",
                "06_main_connected_tree",
            ):
                diagnostics.record_stage(stage, empty)
        return empty
    if verbose:
        print("Preparing Sato skeleton forest...")
    graph = prepare_graph(sato_mask, keep_all_components=True)
    if graph.number_of_nodes() > 211:
        from ronald.modelling.kimimaro_graph import prepare_graph_kimimaro

        if verbose:
            print("Using kimimaro for the detailed skeleton forest...")
        graph = prepare_graph_kimimaro(
            sato_mask, aggressive=True, keep_all_components=True
        )
    if diagnostics is not None and hasattr(diagnostics, "record_graph"):
        diagnostics.record_graph("skeleton_forest", graph)
    if graph.number_of_edges() == 0:
        raise ValueError("Sato skeleton has no modelable edges")
    graph = merge_degree_two_chains(graph)
    node_order = get_node_order(graph)
    node_to_order = {node: index for index, node in enumerate(node_order)}
    label_dtype = np.min_scalar_type(max(1, graph.number_of_edges()))
    workers = -1 if max_workers is None else max_workers
    branches_mask, graph = assign_branch(
        walls, graph, label_dtype=label_dtype, workers=workers
    )
    branch_bboxes = find_label_bboxes(branches_mask, max_label=int(branches_mask.max()))
    wall_sampler = LocalCrossSectionSampler(walls)
    sato_sampler = LocalCrossSectionSampler(sato)
    smooth_mask = np.zeros_like(walls, dtype=bool)
    join_diagnostics = []

    def join(first, second, junction=None):
        record = {}
        fill_local_tube_join(
            first,
            second,
            smooth_mask,
            walls,
            junction=junction,
            cast_to_int=False,
            diagnostics=record,
        )
        join_diagnostics.append(record)

    safeguard_counts = Counter()
    endpoint_only_pca_branches = 0
    processed_branches = 0
    for node in tqdm(node_order, desc="Modeling tree branches", disable=not verbose):
        for neighbor in graph.neighbors(node):
            if node_to_order[node] > node_to_order[neighbor]:
                continue
            processed_branches += 1
            edge = graph.get_edge_data(node, neighbor)
            edge_points = edge["pts"]
            if not (edge_points[0] == graph.nodes[node]["o"]).all():
                edge_points = edge_points[::-1]
            branch_label = edge["mask"]
            endpoints = np.array([graph.nodes[node]["o"], graph.nodes[neighbor]["o"]])
            points = _points_for_branch(
                branches_mask, branch_label, branch_bboxes, endpoints
            )
            pca_points = _points_for_branch(
                branches_mask, branch_label, branch_bboxes, endpoints, support_mask=sato
            )
            if len(pca_points) == len(np.unique(endpoints, axis=0)):
                endpoint_only_pca_branches += 1
            analyser = BranchAnalyser(
                verbose=verbose,
                capture_geometry=diagnostics is not None,
                local_cross_section_sampler=wall_sampler,
                sato_cross_section_sampler=sato_sampler,
            )
            kept, upper, lower, thickness, gaps = analyser.smooth_branch_points(
                edge_points, points, pca_points=pca_points
            )
            if diagnostics is not None:
                diagnostics.record_cylinder_points(kept)
                diagnostics.record_branch(branch_label, node, neighbor, analyser)
            safeguard_counts.update(analyser.fit_diagnostics)
            graph.nodes[neighbor]["ellipse"] = lower
            graph.nodes[neighbor]["thickness"] = thickness
            if len(kept):
                smooth_mask[tuple(kept.T)] = True
            for lower_base, upper_base in gaps:
                join(lower_base, upper_base)
            if node_to_order[node] != 0 and "ellipse" in graph.nodes[node]:
                previous = graph.nodes[node]["ellipse"]
                if len(previous) and len(upper):
                    join(previous, upper, junction=graph.nodes[node]["o"])

    if diagnostics is not None:
        if hasattr(diagnostics, "record_join_diagnostics"):
            diagnostics.record_join_diagnostics(join_diagnostics)
        diagnostics.record_stage("02_after_joins", smooth_mask)
    if verbose:
        print(
            f"PCA support only at graph endpoints: {endpoint_only_pca_branches}/{processed_branches} branches"
        )
        print(f"PCA/ellipse safeguards: {dict(safeguard_counts)}")
    smooth_mask = restore_connected_sato(smooth_mask, sato_mask, verbose=verbose)
    if diagnostics is not None:
        diagnostics.record_stage("03_after_sato_recovery", smooth_mask)
    graph = assign_thickness(graph, node_order)
    branches_mask, graph = assign_branch(
        smooth_mask, graph, label_dtype=label_dtype, workers=workers
    )
    smooth = apply_smoothing_by_node_order(
        graph,
        branches_mask,
        node_order,
        thick_mult=2,
        output_dtype=np.uint8,
        verbose=verbose,
    )
    if diagnostics is not None:
        diagnostics.record_stage("04_after_closing", smooth)
    connected_graph, bridges, connection_report = connect_sato_components(
        graph,
        walls,
        sato,
        spacing_zyx=sato_mask.GetSpacing()[::-1],
        attachment_mask=smooth & (walls > 0),
    )
    smooth = (smooth > 0) | (bridges > 0)
    smooth &= walls > 0
    if diagnostics is not None:
        if hasattr(diagnostics, "record_connections"):
            diagnostics.record_connections(connected_graph, bridges, connection_report)
        diagnostics.record_stage("05_after_component_connections", smooth)
    smooth, excluded, connected_report = retain_main_connected_tree(smooth, sato)
    if diagnostics is not None:
        if hasattr(diagnostics, "record_excluded_components"):
            diagnostics.record_excluded_components(excluded, connected_report)
        diagnostics.record_stage("06_main_connected_tree", smooth)
    if verbose:
        print(f"Tree modeling complete: {np.count_nonzero(smooth)} voxels")
    return smooth.astype(np.uint8, copy=False)


def smooth_tree(walls_filled, verbose=False, max_threads=1, *, diagnostics=None):
    """Calculate Sato and return its connected airway tree with supported walls.

    ``walls_filled`` is a binary 3-D SimpleITK image. Sato is calculated internally
    from that binary mask; responses at or above the GMM-derived threshold supply
    skeleton, PCA samples and connected-component recovery. Filled walls supply local ellipse support
    and the final output limit. No original-airway mask is needed.

    ``max_threads`` bounds native workers for both Sato and modeling (default 1;
    None keeps library defaults). An optional diagnostics observer receives
    masks, selected geometry and, through ``record_sato(mask, metadata)``, the
    Sato input and its threshold settings. The output is uint16 on the input grid.
    """
    with _limited_native_threads(max_threads):
        sato_mask, metadata = calculate_sato_mask(walls_filled, max_workers=max_threads)
        if diagnostics is not None and hasattr(diagnostics, "record_sato"):
            diagnostics.record_sato(sato_mask, metadata)
        smooth = model_tree(
            walls_filled,
            sato_mask,
            verbose=verbose,
            max_workers=max_threads,
            diagnostics=diagnostics,
        )
    result = sitk.GetImageFromArray(smooth.astype(np.uint16, copy=False))
    result.CopyInformation(walls_filled)
    return result
