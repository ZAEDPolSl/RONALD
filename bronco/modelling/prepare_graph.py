import networkx as nx
import SimpleITK as sitk

from bronco.external.sknw import build_sknw

def keep_largest_component(graph):
    largest_cc = max(nx.connected_components(graph), key=len)
    return graph.subgraph(largest_cc).copy()


def make_bfs_tree(graph, root):
    return nx.bfs_tree(graph, root).to_undirected()


def clean_airways_graph(graph):
    graph = keep_largest_component(graph)
    root = max(graph.nodes(), key=lambda n: graph.nodes[n]['o'][0])
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
    skeleton = clean_airways_graph(skeleton)
    return skeleton
