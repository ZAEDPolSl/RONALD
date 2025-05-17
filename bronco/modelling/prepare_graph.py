import networkx as nx
import SimpleITK as sitk
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
    skeleton = get_skeleton(mask)
    print(skeleton.nodes()[0])
    skeleton = clean_airways_graph(skeleton)
    print(skeleton.nodes()[0])
    return skeleton
