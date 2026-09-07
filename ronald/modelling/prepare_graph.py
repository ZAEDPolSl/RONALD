import networkx as nx
import numpy as np
import SimpleITK as sitk
from skimage.morphology import skeletonize

from ronald.external.sknw import build_sknw


def foreground_bbox(array):
    """Return the tight nonzero bounding box without a full-size temporary."""
    occupied_axes = [
        np.flatnonzero(
            np.any(array, axis=tuple(i for i in range(array.ndim) if i != axis))
        )
        for axis in range(array.ndim)
    ]
    if any(len(occupied) == 0 for occupied in occupied_axes):
        return None
    return tuple(
        slice(int(occupied[0]), int(occupied[-1]) + 1) for occupied in occupied_axes
    )


def sitk_region_from_array_bbox(image, bbox):
    """Crop a SimpleITK image using a NumPy-order (z, y, x) bounding box."""
    index = [axis.start for axis in bbox[::-1]]
    size = [axis.stop - axis.start for axis in bbox[::-1]]
    return sitk.RegionOfInterest(image, size=size, index=index)


def expand_bbox(bbox, shape, padding=1):
    return tuple(
        slice(max(0, axis.start - padding), min(shape[i], axis.stop + padding))
        for i, axis in enumerate(bbox)
    )


def _translate_graph_coordinates(graph, offset):
    """Translate ROI-local graph coordinates back to full-volume coordinates."""
    if not np.any(offset):
        return

    for _, data in graph.nodes(data=True):
        data["pts"] += offset.astype(data["pts"].dtype)
        data["o"] += offset.astype(data["o"].dtype)
    for _, _, data in graph.edges(data=True):
        data["pts"] += offset.astype(data["pts"].dtype)


def keep_largest_component(graph):
    largest_cc = max(nx.connected_components(graph), key=len)
    return graph.subgraph(largest_cc).copy()


def make_bfs_tree(graph, root):
    bfs_edges = list(nx.bfs_edges(graph, root))
    bfs_nodes = set([root]) | {v for _, v in bfs_edges}

    tree = nx.Graph()

    for node in bfs_nodes:
        tree.add_node(node, **graph.nodes[node])

    for u, v in bfs_edges:
        if graph.has_edge(u, v):
            tree.add_edge(u, v, **graph.edges[u, v])
        elif graph.has_edge(v, u):
            tree.add_edge(u, v, **graph.edges[v, u])
        else:
            tree.add_edge(u, v)

    return tree


def clean_airways_graph(graph):
    graph = keep_largest_component(graph)
    root = max(graph.nodes(), key=lambda n: graph.nodes[n]["o"][0])
    graph = make_bfs_tree(graph, root)
    return graph


def get_skeleton(mask):
    """Build a skeleton graph inside the foreground ROI."""
    airways_full = sitk.GetArrayViewFromImage(mask)
    bbox = foreground_bbox(airways_full)
    if bbox is None:
        airways = airways_full
        offset = np.zeros(airways_full.ndim, dtype=np.int64)
    else:
        # Preserve full-volume boundary conditions for radius-1 morphology.
        bbox = expand_bbox(bbox, airways_full.shape, padding=1)
        airways = airways_full[bbox]
        offset = np.array([axis.start for axis in bbox])

    skeleton_array = skeletonize(airways).astype(np.uint8, copy=False)
    sitk_skeleton = sitk.GetImageFromArray(skeleton_array)
    sitk_skeleton.SetSpacing(mask.GetSpacing())
    sitk_skeleton.SetDirection(mask.GetDirection())
    sitk_skeleton.SetOrigin(
        mask.TransformIndexToPhysicalPoint(tuple(int(value) for value in offset[::-1]))
    )
    cleaned_skeleton = sitk.BinaryFillhole(sitk.Cast(sitk_skeleton > 0, sitk.sitkUInt8))
    cleaned_skeleton = sitk.Cast(
        (
            cleaned_skeleton
            - sitk.BinaryMorphologicalOpening(cleaned_skeleton, kernelRadius=(1, 1, 1))
        )
        > 0,
        sitk.sitkUInt8,
    )
    skeleton_array = sitk.GetArrayFromImage(cleaned_skeleton)
    graph = build_sknw(skeleton_array, iso=False, ring=False, full=True)
    _translate_graph_coordinates(graph, offset)
    return graph


def prepare_graph(mask):
    mask = keep_largest_component_mask(mask)
    graph = get_skeleton(mask)
    return clean_airways_graph(graph)


def assign_thickness(G, node_order):
    node_to_order = {node: index for index, node in enumerate(node_order)}
    small_diameters = []

    for u, v in G.edges():
        # Determine the lower node based on order
        lower_node = u if node_to_order[v] < node_to_order[u] else v

        minor_len = G.nodes[lower_node].get("thickness")
        if minor_len is not None:
            small_diameters.append(minor_len)

    max_diameter = max(small_diameters) if small_diameters else 1.0

    for u, v, data in G.edges(data=True):
        lower_node = u if node_to_order[v] < node_to_order[u] else v

        minor_len = G.nodes[lower_node].get("thickness")
        if minor_len is not None:
            data["size"] = minor_len / max_diameter
        else:
            data["size"] = 0.0  # or None

    return G


def keep_largest_component_mask(mask):
    mask_array = sitk.GetArrayViewFromImage(mask)
    bbox = foreground_bbox(mask_array)
    if bbox is None:
        return mask

    # Connectivity is unchanged by removing an all-background border. Running
    # the global filter only on this ROI substantially lowers its working set.
    cropped_mask = sitk_region_from_array_bbox(mask, bbox)
    cc = sitk.ConnectedComponent(cropped_mask)
    # ``cc`` stays alive until this function returns, making this read-only
    # view safe while avoiding a second full uint32 label volume.
    arr = sitk.GetArrayViewFromImage(cc)
    counts = np.bincount(arr.ravel())
    if len(counts) <= 1:
        return mask
    counts[0] = 0
    main_label = int(np.argmax(counts))
    largest_array = np.zeros(mask_array.shape, dtype=np.uint8)
    np.equal(arr, main_label, out=largest_array[bbox])
    largest = sitk.GetImageFromArray(largest_array)
    largest.CopyInformation(mask)
    return largest
