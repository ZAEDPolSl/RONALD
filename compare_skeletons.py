import argparse
import csv
import json
import os

import networkx as nx
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import convolve, label
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree, distance_matrix
from skimage.morphology import skeletonize

from bronco.external.sknw import build_sknw
from ctools import ImageInstance


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare skeleton graphs extracted from binary masks."
    )
    parser.add_argument(
        "masks",
        nargs="*",
        help="Optional single pair supplied as: mask_a mask_b",
    )
    parser.add_argument(
        "--pair",
        nargs=2,
        action="append",
        metavar=("MASK_A", "MASK_B"),
        help="Repeat this option to compare several pairs in one run",
    )
    parser.add_argument(
        "--pairs-file",
        type=str,
        help="Text/CSV/TSV file with two columns per row: mask_a, mask_b",
    )
    parser.add_argument(
        "--align-second-to-first",
        action="store_true",
        help="Resample the second mask to the first mask geometry before comparison",
    )
    parser.add_argument(
        "--node-match-radius-mm",
        type=float,
        default=3.0,
        help="Maximum distance for matching graph nodes such as branch points or endpoints",
    )
    parser.add_argument(
        "--point-match-radius-mm",
        type=float,
        default=2.0,
        help="Distance threshold for tolerant skeleton point coverage metrics",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        help="Optional path to save per-pair metrics as a CSV file",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        help="Optional path to save per-pair metrics as JSON",
    )
    parser.add_argument(
        "--describe-metrics",
        action="store_true",
        help="Print the metric definitions and exit",
    )
    return parser.parse_args()


def read_binary_mask(path):
    image = ImageInstance().read(path)
    if image is None:
        raise ValueError(f"Could not read image from: {path}")
    if image.GetDimension() != 3:
        raise ValueError(f"Expected a 3D mask at {path}, got {image.GetDimension()}D")
    return sitk.Cast(image > 0, sitk.sitkUInt8)


def images_have_same_geometry(image_a, image_b, atol=1e-5):
    return (
        image_a.GetSize() == image_b.GetSize()
        and np.allclose(image_a.GetSpacing(), image_b.GetSpacing(), atol=atol)
        and np.allclose(image_a.GetOrigin(), image_b.GetOrigin(), atol=atol)
        and np.allclose(image_a.GetDirection(), image_b.GetDirection(), atol=atol)
    )


def get_pairs(args):
    pairs = []

    if args.masks:
        if len(args.masks) != 2:
            raise ValueError(
                "Positional arguments must contain exactly two paths: mask_a mask_b"
            )
        pairs.append((args.masks[0], args.masks[1]))

    if args.pair:
        pairs.extend((mask_a, mask_b) for mask_a, mask_b in args.pair)

    if args.pairs_file:
        with open(args.pairs_file, "r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue

                if "," in stripped:
                    parts = [part.strip() for part in stripped.split(",")]
                elif "\t" in stripped:
                    parts = [part.strip() for part in stripped.split("\t")]
                else:
                    parts = stripped.split()

                if len(parts) < 2:
                    raise ValueError(
                        f"Invalid pair definition in {args.pairs_file}:{line_number}"
                    )
                pairs.append((parts[0], parts[1]))

    if not pairs:
        raise ValueError(
            "Provide one pair as positional arguments, repeat --pair, or use --pairs-file."
        )

    return pairs


def resample_like_reference(moving_image, reference_image):
    resampled = sitk.Resample(
        moving_image,
        reference_image,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )
    resampled.CopyInformation(reference_image)
    return resampled


def image_points_to_physical(indices_zyx, image):
    if len(indices_zyx) == 0:
        return np.empty((0, 3), dtype=np.float64)

    indices_xyz = indices_zyx[:, ::-1].astype(np.float64)
    spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
    origin = np.asarray(image.GetOrigin(), dtype=np.float64)
    direction = np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3)
    scaled_indices = indices_xyz * spacing
    return origin + scaled_indices @ direction.T


def get_skeleton_image(mask):
    mask_array = sitk.GetArrayFromImage(mask) > 0
    skeleton = skeletonize(mask_array).astype(np.uint8)
    skeleton_image = sitk.GetImageFromArray(skeleton)
    skeleton_image.CopyInformation(mask)
    skeleton_image = sitk.BinaryFillhole(
        sitk.Cast(skeleton_image > 0, sitk.sitkUInt8)
    )
    skeleton_image = sitk.Cast(
        (
            skeleton_image
            - sitk.BinaryMorphologicalOpening(skeleton_image, kernelRadius=(1, 1, 1))
        )
        > 0,
        sitk.sitkUInt8,
    )
    skeleton_image.CopyInformation(mask)
    return skeleton_image


def build_skeleton_graph(skeleton_image):
    skeleton_array = sitk.GetArrayFromImage(skeleton_image) > 0
    if not np.any(skeleton_array):
        return nx.Graph()
    return build_sknw(skeleton_array.astype(np.uint8), iso=False, ring=False, full=True)


def skeleton_points_physical(skeleton_image):
    skeleton = sitk.GetArrayFromImage(skeleton_image) > 0
    indices_zyx = np.argwhere(skeleton)
    return image_points_to_physical(indices_zyx, skeleton_image)


def classify_node(degree):
    if degree == 0:
        return "isolated"
    if degree == 1:
        return "endpoint"
    if degree == 2:
        return "connector"
    return "branchpoint"


def extract_nodes(graph, reference_image):
    nodes = []
    for node_id in graph.nodes():
        degree = int(graph.degree(node_id))
        position_zyx = np.asarray(graph.nodes[node_id]["o"], dtype=np.float64).reshape(1, 3)
        position_mm = image_points_to_physical(position_zyx, reference_image)[0]
        nodes.append(
            {
                "id": int(node_id),
                "degree": degree,
                "kind": classify_node(degree),
                "position_mm": position_mm,
            }
        )
    return nodes


def connected_component_centroids(mask, reference_image):
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labeled, num_components = label(mask.astype(np.uint8), structure=structure)
    components = []
    for component_id in range(1, num_components + 1):
        component_points_zyx = np.argwhere(labeled == component_id)
        centroid_zyx = component_points_zyx.mean(axis=0, keepdims=True)
        centroid_mm = image_points_to_physical(centroid_zyx, reference_image)[0]
        components.append(
            {
                "position_mm": centroid_mm,
                "size_voxels": int(len(component_points_zyx)),
            }
        )
    return components


def extract_key_nodes_from_skeleton(skeleton_image):
    skeleton = sitk.GetArrayFromImage(skeleton_image) > 0
    if not np.any(skeleton):
        return {"endpoints": [], "branchpoints": [], "isolated": []}

    kernel = np.ones((3, 3, 3), dtype=np.int16)
    kernel[1, 1, 1] = 0
    neighbor_count = convolve(skeleton.astype(np.int16), kernel, mode="constant", cval=0)

    isolated_mask = skeleton & (neighbor_count == 0)
    endpoint_mask = skeleton & (neighbor_count == 1)
    branchpoint_mask = skeleton & (neighbor_count >= 3)

    return {
        "endpoints": connected_component_centroids(endpoint_mask, skeleton_image),
        "branchpoints": connected_component_centroids(branchpoint_mask, skeleton_image),
        "isolated": connected_component_centroids(isolated_mask, skeleton_image),
    }


def edge_length_mm(edge_points_zyx, reference_image):
    points_mm = image_points_to_physical(np.asarray(edge_points_zyx), reference_image)
    if len(points_mm) < 2:
        return 0.0
    return float(np.linalg.norm(points_mm[1:] - points_mm[:-1], axis=1).sum())


def graph_summary(graph, reference_image):
    nodes = extract_nodes(graph, reference_image)
    edge_lengths = [
        edge_length_mm(data["pts"], reference_image)
        for _, _, data in graph.edges(data=True)
    ]
    degree_histogram = {
        "degree_0": 0,
        "degree_1": 0,
        "degree_2": 0,
        "degree_3plus": 0,
    }

    for node in nodes:
        if node["degree"] == 0:
            degree_histogram["degree_0"] += 1
        elif node["degree"] == 1:
            degree_histogram["degree_1"] += 1
        elif node["degree"] == 2:
            degree_histogram["degree_2"] += 1
        else:
            degree_histogram["degree_3plus"] += 1

    return {
        "node_count": int(graph.number_of_nodes()),
        "edge_count": int(graph.number_of_edges()),
        "component_count": int(nx.number_connected_components(graph))
        if graph.number_of_nodes() > 0
        else 0,
        "endpoint_count": int(sum(node["kind"] == "endpoint" for node in nodes)),
        "branchpoint_count": int(sum(node["kind"] == "branchpoint" for node in nodes)),
        "connector_count": int(sum(node["kind"] == "connector" for node in nodes)),
        "isolated_count": int(sum(node["kind"] == "isolated" for node in nodes)),
        "total_branch_length_mm": float(sum(edge_lengths)),
        "mean_branch_length_mm": float(np.mean(edge_lengths)) if edge_lengths else 0.0,
        **degree_histogram,
    }


def normalized_degree_histogram(summary):
    counts = np.array(
        [
            summary["degree_0"],
            summary["degree_1"],
            summary["degree_2"],
            summary["degree_3plus"],
        ],
        dtype=np.float64,
    )
    total = counts.sum()
    if total == 0:
        return counts
    return counts / total


def degree_distribution_similarity(summary_a, summary_b):
    hist_a = normalized_degree_histogram(summary_a)
    hist_b = normalized_degree_histogram(summary_b)
    return float(1.0 - 0.5 * np.abs(hist_a - hist_b).sum())


def safe_ratio_similarity(value_a, value_b):
    if value_a == 0 and value_b == 0:
        return 1.0
    if value_a == 0 or value_b == 0:
        return 0.0
    return float(min(value_a, value_b) / max(value_a, value_b))


def harmonic_mean(value_a, value_b):
    if value_a is None or value_b is None:
        return None
    if value_a + value_b == 0:
        return 0.0
    return float(2.0 * value_a * value_b / (value_a + value_b))


def match_point_sets(points_a, points_b, radius_mm):
    num_a = len(points_a)
    num_b = len(points_b)

    if num_a == 0 or num_b == 0:
        return {
            "matches": [],
            "matched_count": 0,
            "mean_distance_mm": None,
            "max_distance_mm": None,
        }

    pairwise_distances = distance_matrix(points_a, points_b)
    dummy_cost = radius_mm
    invalid_cost = radius_mm * 1000.0 + 1.0
    size = num_a + num_b
    cost_matrix = np.full((size, size), dummy_cost, dtype=np.float64)
    valid_distances = np.where(pairwise_distances <= radius_mm, pairwise_distances, invalid_cost)
    cost_matrix[:num_a, :num_b] = valid_distances
    cost_matrix[num_a:, num_b:] = 0.0

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    for row, col in zip(row_ind, col_ind):
        if row < num_a and col < num_b and pairwise_distances[row, col] <= radius_mm:
            matches.append((int(row), int(col), float(pairwise_distances[row, col])))

    distances = [distance for _, _, distance in matches]
    return {
        "matches": matches,
        "matched_count": len(matches),
        "mean_distance_mm": float(np.mean(distances)) if distances else None,
        "max_distance_mm": float(np.max(distances)) if distances else None,
    }


def node_match_metrics(nodes_a, nodes_b, radius_mm, label):
    points_a = np.asarray([node["position_mm"] for node in nodes_a], dtype=np.float64)
    points_b = np.asarray([node["position_mm"] for node in nodes_b], dtype=np.float64)
    match_result = match_point_sets(points_a, points_b, radius_mm)

    matched_count = match_result["matched_count"]
    count_a = len(nodes_a)
    count_b = len(nodes_b)
    recall = None if count_a == 0 else float(matched_count / count_a)
    precision = None if count_b == 0 else float(matched_count / count_b)

    return {
        f"{label}_count_a": count_a,
        f"{label}_count_b": count_b,
        f"{label}_matched_count": matched_count,
        f"{label}_recall": recall,
        f"{label}_precision": precision,
        f"{label}_f1": harmonic_mean(recall, precision),
        f"{label}_mean_match_distance_mm": match_result["mean_distance_mm"],
        f"{label}_max_match_distance_mm": match_result["max_distance_mm"],
    }


def node_metrics(skeleton_a, skeleton_b, radius_mm):
    key_nodes_a = extract_key_nodes_from_skeleton(skeleton_a)
    key_nodes_b = extract_key_nodes_from_skeleton(skeleton_b)
    metrics = {}
    metrics.update(
        node_match_metrics(
            key_nodes_a["branchpoints"],
            key_nodes_b["branchpoints"],
            radius_mm,
            "branchpoint",
        )
    )
    metrics.update(
        node_match_metrics(
            key_nodes_a["endpoints"],
            key_nodes_b["endpoints"],
            radius_mm,
            "endpoint",
        )
    )
    metrics.update(
        node_match_metrics(
            key_nodes_a["branchpoints"] + key_nodes_a["endpoints"],
            key_nodes_b["branchpoints"] + key_nodes_b["endpoints"],
            radius_mm,
            "key_node",
        )
    )
    metrics["isolated_count_a"] = len(key_nodes_a["isolated"])
    metrics["isolated_count_b"] = len(key_nodes_b["isolated"])
    return metrics


def point_distance_metrics(skeleton_a, skeleton_b, point_match_radius_mm):
    points_a = skeleton_points_physical(skeleton_a)
    points_b = skeleton_points_physical(skeleton_b)

    if len(points_a) == 0 or len(points_b) == 0:
        return {
            "skeleton_point_count_a": int(len(points_a)),
            "skeleton_point_count_b": int(len(points_b)),
            "point_coverage_a_by_b": None,
            "point_coverage_b_by_a": None,
            "point_f1": None,
            "avg_min_distance_a_to_b_mm": None,
            "avg_min_distance_b_to_a_mm": None,
            "chamfer_distance_mm": None,
            "hd95_distance_mm": None,
        }

    tree_a = cKDTree(points_a)
    tree_b = cKDTree(points_b)
    dist_a_to_b = tree_b.query(points_a, k=1)[0]
    dist_b_to_a = tree_a.query(points_b, k=1)[0]

    coverage_a = float(np.mean(dist_a_to_b <= point_match_radius_mm))
    coverage_b = float(np.mean(dist_b_to_a <= point_match_radius_mm))
    hd95 = float(
        max(np.percentile(dist_a_to_b, 95), np.percentile(dist_b_to_a, 95))
    )

    return {
        "skeleton_point_count_a": int(len(points_a)),
        "skeleton_point_count_b": int(len(points_b)),
        "point_coverage_a_by_b": coverage_a,
        "point_coverage_b_by_a": coverage_b,
        "point_f1": harmonic_mean(coverage_a, coverage_b),
        "avg_min_distance_a_to_b_mm": float(dist_a_to_b.mean()),
        "avg_min_distance_b_to_a_mm": float(dist_b_to_a.mean()),
        "chamfer_distance_mm": float(
            np.concatenate([dist_a_to_b, dist_b_to_a]).mean()
        ),
        "hd95_distance_mm": hd95,
    }


def graph_similarity_metrics(summary_a, summary_b):
    return {
        "node_count_similarity": safe_ratio_similarity(
            summary_a["node_count"], summary_b["node_count"]
        ),
        "edge_count_similarity": safe_ratio_similarity(
            summary_a["edge_count"], summary_b["edge_count"]
        ),
        "endpoint_count_similarity": safe_ratio_similarity(
            summary_a["endpoint_count"], summary_b["endpoint_count"]
        ),
        "branchpoint_count_similarity": safe_ratio_similarity(
            summary_a["branchpoint_count"], summary_b["branchpoint_count"]
        ),
        "total_branch_length_similarity": safe_ratio_similarity(
            summary_a["total_branch_length_mm"], summary_b["total_branch_length_mm"]
        ),
        "degree_distribution_similarity": degree_distribution_similarity(
            summary_a, summary_b
        ),
    }


def round_nested(data, digits=6):
    if isinstance(data, dict):
        return {key: round_nested(value, digits) for key, value in data.items()}
    if isinstance(data, list):
        return [round_nested(value, digits) for value in data]
    if isinstance(data, float):
        return round(data, digits)
    return data


def metric_descriptions():
    return {
        "branchpoint_precision": (
            "Fraction of branch points in graph B that can be matched to branch points in graph A within --node-match-radius-mm."
        ),
        "branchpoint_recall": (
            "Fraction of branch points in graph A recovered by graph B within --node-match-radius-mm."
        ),
        "branchpoint_f1": (
            "Harmonic mean of branchpoint precision and recall. This is a strong topology-aware similarity metric."
        ),
        "endpoint_precision": (
            "Fraction of endpoints in graph B that can be matched to endpoints in graph A within --node-match-radius-mm."
        ),
        "endpoint_recall": (
            "Fraction of endpoints in graph A recovered by graph B within --node-match-radius-mm."
        ),
        "endpoint_f1": (
            "Harmonic mean of endpoint precision and recall."
        ),
        "key_node_f1": (
            "Same idea as above, but for endpoints and branch points together."
        ),
        "branchpoint_mean_match_distance_mm": (
            "Average distance between matched branch points."
        ),
        "endpoint_mean_match_distance_mm": (
            "Average distance between matched endpoints."
        ),
        "point_coverage_a_by_b": (
            "Fraction of skeleton points in A that lie within --point-match-radius-mm of skeleton B."
        ),
        "point_coverage_b_by_a": (
            "Fraction of skeleton points in B that lie within --point-match-radius-mm of skeleton A."
        ),
        "point_f1": (
            "Tolerance-based overlap score for skeleton points. More robust than Dice for slightly shifted trees."
        ),
        "chamfer_distance_mm": (
            "Average nearest-neighbour distance between both skeleton point sets. Lower is better."
        ),
        "hd95_distance_mm": (
            "95th percentile Hausdorff-style distance. Lower is better and less sensitive than max Hausdorff."
        ),
        "degree_distribution_similarity": (
            "Similarity of node degree distributions, from 0 to 1."
        ),
        "total_branch_length_similarity": (
            "Ratio-based similarity of total branch length, from 0 to 1."
        ),
        "node_count_similarity": (
            "Ratio-based similarity of total node counts, from 0 to 1."
        ),
        "edge_count_similarity": (
            "Ratio-based similarity of total edge counts, from 0 to 1."
        ),
    }


def print_metric_descriptions():
    print("Metrics used")
    for name, description in metric_descriptions().items():
        print(f"  {name}: {description}")


def compare_pair(mask_a_path, mask_b_path, align_second_to_first, node_match_radius_mm, point_match_radius_mm):
    mask_a = read_binary_mask(mask_a_path)
    mask_b = read_binary_mask(mask_b_path)

    aligned = False
    if not images_have_same_geometry(mask_a, mask_b):
        if not align_second_to_first:
            raise ValueError(
                "Input masks do not share the same geometry. "
                "Use --align-second-to-first to resample the second mask."
            )
        mask_b = resample_like_reference(mask_b, mask_a)
        aligned = True

    skeleton_a = get_skeleton_image(mask_a)
    skeleton_b = get_skeleton_image(mask_b)
    graph_a = build_skeleton_graph(skeleton_a)
    graph_b = build_skeleton_graph(skeleton_b)
    graph_summary_a = graph_summary(graph_a, skeleton_a)
    graph_summary_b = graph_summary(graph_b, skeleton_b)

    pair_metrics = {
        "mask_a": os.path.abspath(mask_a_path),
        "mask_b": os.path.abspath(mask_b_path),
        "aligned_second_to_first": aligned,
        "size": list(mask_a.GetSize()),
        "spacing_mm": list(mask_a.GetSpacing()),
        "node_match_radius_mm": float(node_match_radius_mm),
        "point_match_radius_mm": float(point_match_radius_mm),
    }
    pair_metrics.update(point_distance_metrics(skeleton_a, skeleton_b, point_match_radius_mm))
    pair_metrics.update(node_metrics(skeleton_a, skeleton_b, node_match_radius_mm))
    pair_metrics.update(graph_similarity_metrics(graph_summary_a, graph_summary_b))

    for key, value in graph_summary_a.items():
        pair_metrics[f"{key}_a"] = value
    for key, value in graph_summary_b.items():
        pair_metrics[f"{key}_b"] = value

    return round_nested(pair_metrics)


def print_pair_summary(index, metrics):
    print(f"[pair {index}]")
    print(f"  mask_a: {metrics['mask_a']}")
    print(f"  mask_b: {metrics['mask_b']}")
    print(f"  key_node_f1: {metrics['key_node_f1']}")
    print(f"  branchpoint_f1: {metrics['branchpoint_f1']}")
    print(f"  endpoint_f1: {metrics['endpoint_f1']}")
    print(f"  point_f1: {metrics['point_f1']}")
    print(f"  chamfer_distance_mm: {metrics['chamfer_distance_mm']}")
    print(f"  hd95_distance_mm: {metrics['hd95_distance_mm']}")
    print(f"  degree_distribution_similarity: {metrics['degree_distribution_similarity']}")
    print(f"  total_branch_length_similarity: {metrics['total_branch_length_similarity']}")


def print_aggregate_summary(results):
    if len(results) < 2:
        return

    keys = [
        "key_node_f1",
        "branchpoint_f1",
        "endpoint_f1",
        "point_f1",
        "chamfer_distance_mm",
        "hd95_distance_mm",
        "degree_distribution_similarity",
        "total_branch_length_similarity",
    ]

    print("")
    print("Aggregate")
    for key in keys:
        values = [result[key] for result in results if result[key] is not None]
        if not values:
            continue
        print(
            f"  {key}: mean={round(float(np.mean(values)), 6)}, "
            f"std={round(float(np.std(values)), 6)}"
        )


def save_csv(results, output_csv):
    fieldnames = list(results[0].keys())
    with open(output_csv, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def main():
    args = parse_args()

    if args.describe_metrics:
        print_metric_descriptions()
        return

    pairs = get_pairs(args)
    results = []
    for index, (mask_a_path, mask_b_path) in enumerate(pairs, start=1):
        metrics = compare_pair(
            mask_a_path,
            mask_b_path,
            args.align_second_to_first,
            args.node_match_radius_mm,
            args.point_match_radius_mm,
        )
        results.append(metrics)
        print_pair_summary(index, metrics)
        if index != len(pairs):
            print("")

    print_aggregate_summary(results)

    if args.output_csv:
        save_csv(results, args.output_csv)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=2)


if __name__ == "__main__":
    main()
