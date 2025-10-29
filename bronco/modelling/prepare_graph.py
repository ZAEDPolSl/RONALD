import networkx as nx
import SimpleITK as sitk
import numpy as np
from skimage.morphology import skeletonize

from bronco.external.sknw import build_sknw


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


def prepare_graph(mask):
    mask = keep_largest_component_mask(mask)
    skeleton = get_skeleton(mask)
    skeleton = clean_airways_graph(skeleton)
    return skeleton


def assign_thickness(G, node_order):
    small_diameters = []

    for u, v in G.edges():
        # Determine the lower node based on order
        lower_node = u if node_order.index(v) < node_order.index(u) else v

        minor_len = G.nodes[lower_node].get("thickness")
        if minor_len is not None:
            small_diameters.append(minor_len)

    max_diameter = max(small_diameters) if small_diameters else 1.0

    for u, v, data in G.edges(data=True):
        lower_node = u if node_order.index(v) < node_order.index(u) else v

        minor_len = G.nodes[lower_node].get("thickness")
        if minor_len is not None:
            data["size"] = minor_len / max_diameter
        else:
            data["size"] = 0.0  # or None

    return G


def keep_largest_component_mask(mask):
    cc = sitk.ConnectedComponent(mask)
    arr = sitk.GetArrayFromImage(cc)
    labels, counts = np.unique(arr, return_counts=True)
    mask_nonzero = labels != 0
    labels = labels[mask_nonzero]
    counts = counts[mask_nonzero]
    if len(counts) == 0:
        return mask
    main_label = labels[np.argmax(counts)]
    largest = sitk.GetImageFromArray((arr == main_label).astype(np.uint8))
    largest.CopyInformation(mask)
    return largest
