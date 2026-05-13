from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from skimage.morphology import skeletonize

from bronco.external.sknw import build_sknw

DEFAULT_CALIBER_THRESHOLDS_MM = {
    "small": 2.0,
    "large": 5.0,
}


def normalize_caliber_thresholds(
    caliber_thresholds_mm: dict[str, object] | None = None,
) -> dict[str, dict[str, float]]:
    if caliber_thresholds_mm is None:
        caliber_thresholds_mm = DEFAULT_CALIBER_THRESHOLDS_MM

    small = caliber_thresholds_mm["small"]
    large = caliber_thresholds_mm["large"]

    if isinstance(small, dict) and isinstance(large, dict):
        small_max = float(small["max"])
        large_min = float(large["min"])
        normalized = {
            "small": {"max": small_max},
            "medium": {"min": small_max, "max": large_min},
            "large": {"min": large_min},
        }
    else:
        small_max = float(small)
        large_min = float(large)
        normalized = {
            "small": {"max": small_max},
            "medium": {"min": small_max, "max": large_min},
            "large": {"min": large_min},
        }

    if not (
        normalized["small"]["max"] <= normalized["medium"]["min"]
        <= normalized["medium"]["max"]
        <= normalized["large"]["min"]
    ):
        raise ValueError(
            "Invalid caliber thresholds. Expected "
            "small <= large."
        )

    return normalized


def image_points_to_physical(indices_zyx: np.ndarray, image: sitk.Image) -> np.ndarray:
    if len(indices_zyx) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    indices_xyz = indices_zyx[:, ::-1].astype(np.float64)
    spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
    origin = np.asarray(image.GetOrigin(), dtype=np.float64)
    direction = np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3)
    scaled_indices = indices_xyz * spacing
    return origin + scaled_indices @ direction.T


def get_skeleton_image(mask: sitk.Image) -> sitk.Image:
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


def get_thickness_image(mask: sitk.Image) -> sitk.Image:
    vessel_arr = sitk.GetArrayFromImage(mask) > 0
    distance_image = sitk.SignedMaurerDistanceMap(
        sitk.Cast(mask > 0, sitk.sitkUInt8),
        insideIsPositive=True,
        squaredDistance=False,
        useImageSpacing=True,
    )
    distance_arr = sitk.GetArrayFromImage(distance_image).astype(np.float32)

    thickness_arr = np.zeros_like(distance_arr, dtype=np.float32)
    voxel_width_mm = float(min(mask.GetSpacing()))

    # Maurer distance is zero on boundary voxels; use a half-voxel radius only
    # for nonpositive samples so positive interior distances stay unchanged.
    radius_arr = np.where(
        distance_arr[vessel_arr] > 0,
        distance_arr[vessel_arr],
        0.5 * voxel_width_mm,
    )
    thickness_arr[vessel_arr] = 2.0 * radius_arr

    thickness_image = sitk.GetImageFromArray(thickness_arr)
    thickness_image.CopyInformation(mask)
    return sitk.Cast(thickness_image, sitk.sitkFloat32)


def build_skeleton_graph(skeleton_image: sitk.Image):
    skeleton_array = sitk.GetArrayFromImage(skeleton_image) > 0
    if not np.any(skeleton_array):
        return None
    # Use traced skeleton voxels directly for quantitative measurements.
    # `full=True` replaces branch endpoints with node centroids, which is
    # convenient for display but can move samples off the true skeleton path.
    return build_sknw(skeleton_array.astype(np.uint8), iso=False, ring=False, full=False)


def graph_component_count(graph) -> int:
    if graph is None or graph.number_of_nodes() == 0:
        return 0
    visited = set()
    components = 0
    for node in graph.nodes():
        if node in visited:
            continue
        components += 1
        stack = [node]
        visited.add(node)
        while stack:
            current = stack.pop()
            for neighbor in graph.neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
    return components


def mask_component_count(mask: sitk.Image) -> int:
    cc = sitk.ConnectedComponent(sitk.Cast(mask > 0, sitk.sitkUInt8))
    cc_arr = sitk.GetArrayFromImage(cc)
    if cc_arr.size == 0:
        return 0
    return int(cc_arr.max())


def edge_length_mm(edge_points_zyx: np.ndarray, reference_image: sitk.Image) -> float:
    points_mm = image_points_to_physical(np.asarray(edge_points_zyx), reference_image)
    if len(points_mm) < 2:
        return 0.0
    return float(np.linalg.norm(points_mm[1:] - points_mm[:-1], axis=1).sum())


def branch_curvatures_per_mm(points_mm: np.ndarray) -> np.ndarray:
    if len(points_mm) < 3:
        return np.zeros(0, dtype=np.float64)
    segments = points_mm[1:] - points_mm[:-1]
    segment_lengths = np.linalg.norm(segments, axis=1)
    valid = (segment_lengths[:-1] > 0) & (segment_lengths[1:] > 0)
    if not np.any(valid):
        return np.zeros(0, dtype=np.float64)

    u1 = segments[:-1][valid] / segment_lengths[:-1][valid, None]
    u2 = segments[1:][valid] / segment_lengths[1:][valid, None]
    dots = np.clip(np.sum(u1 * u2, axis=1), -1.0, 1.0)
    angles = np.arccos(dots)
    step = 0.5 * (segment_lengths[:-1][valid] + segment_lengths[1:][valid])
    return np.divide(angles, step, out=np.zeros_like(angles), where=step > 0)


def numeric_summary(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "p10": None,
            "p25": None,
            "p75": None,
            "p90": None,
        }
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
    }


def caliber_bin_fractions(
    values: np.ndarray,
    caliber_thresholds_mm: dict[str, dict[str, float]] | None = None,
) -> dict[str, float]:
    thresholds = normalize_caliber_thresholds(caliber_thresholds_mm)
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"small": 0.0, "medium": 0.0, "large": 0.0}

    small_max = thresholds["small"]["max"]
    medium_min = thresholds["medium"]["min"]
    medium_max = thresholds["medium"]["max"]
    large_min = thresholds["large"]["min"]

    return {
        "small": float(np.mean(values < small_max)),
        "medium": float(
            np.mean((values >= medium_min) & (values < medium_max))
        ),
        "large": float(np.mean(values >= large_min)),
    }


def compute_branch_metrics(
    graph,
    reference_image: sitk.Image,
    thickness_image: sitk.Image,
) -> list[dict[str, float | int]]:
    if graph is None:
        return []

    thickness_arr = sitk.GetArrayFromImage(thickness_image).astype(np.float32)
    branch_rows: list[dict[str, float | int]] = []

    for branch_id, (u, v, data) in enumerate(graph.edges(data=True), start=1):
        points_zyx = np.asarray(data["pts"], dtype=np.int32)
        points_mm = image_points_to_physical(points_zyx, reference_image)
        path_length_mm = edge_length_mm(points_zyx, reference_image)

        if len(points_mm) >= 2:
            straight_length_mm = float(np.linalg.norm(points_mm[-1] - points_mm[0]))
        else:
            straight_length_mm = 0.0

        if straight_length_mm > 0:
            tortuosity = float(path_length_mm / straight_length_mm)
        else:
            tortuosity = 1.0 if path_length_mm > 0 else 0.0

        branch_curvature = branch_curvatures_per_mm(points_mm)
        branch_thickness = thickness_arr[
            points_zyx[:, 0],
            points_zyx[:, 1],
            points_zyx[:, 2],
        ]
        branch_thickness = branch_thickness[branch_thickness > 0]

        branch_rows.append(
            {
                "branch_id": int(branch_id),
                "node_u": int(u),
                "node_v": int(v),
                "point_count": int(len(points_zyx)),
                "path_length_mm": float(path_length_mm),
                "straight_length_mm": float(straight_length_mm),
                "tortuosity": float(tortuosity),
                "mean_curvature_per_mm": float(np.mean(branch_curvature))
                if branch_curvature.size
                else 0.0,
                "max_curvature_per_mm": float(np.max(branch_curvature))
                if branch_curvature.size
                else 0.0,
                "mean_thickness_mm": float(np.mean(branch_thickness))
                if branch_thickness.size
                else 0.0,
                "median_thickness_mm": float(np.median(branch_thickness))
                if branch_thickness.size
                else 0.0,
                "min_thickness_mm": float(np.min(branch_thickness))
                if branch_thickness.size
                else 0.0,
                "max_thickness_mm": float(np.max(branch_thickness))
                if branch_thickness.size
                else 0.0,
            }
        )

    return branch_rows


def write_branch_metrics_csv(path: Path, branch_rows: list[dict[str, float | int]]) -> None:
    fieldnames = [
        "branch_id",
        "node_u",
        "node_v",
        "point_count",
        "path_length_mm",
        "straight_length_mm",
        "tortuosity",
        "mean_curvature_per_mm",
        "max_curvature_per_mm",
        "mean_thickness_mm",
        "median_thickness_mm",
        "min_thickness_mm",
        "max_thickness_mm",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(branch_rows)


def write_graph_tables(
    output_dir: Path,
    graph,
    index_offset_zyx: tuple[int, int, int] = (0, 0, 0),
) -> dict[str, str]:
    nodes_path = output_dir / "graph_nodes.csv"
    node_points_path = output_dir / "graph_node_points.csv"
    edges_path = output_dir / "graph_edges.csv"
    edge_points_path = output_dir / "graph_edge_points.csv"

    if graph is None:
        for path, fieldnames in (
            (nodes_path, ["node_id", "degree", "point_count"]),
            (node_points_path, ["node_id", "point_index", "z", "y", "x"]),
            (edges_path, ["edge_id", "node_u", "node_v", "point_count"]),
            (edge_points_path, ["edge_id", "node_u", "node_v", "point_index", "z", "y", "x"]),
        ):
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
        return {
            "graph_nodes_csv": str(nodes_path),
            "graph_node_points_csv": str(node_points_path),
            "graph_edges_csv": str(edges_path),
            "graph_edge_points_csv": str(edge_points_path),
        }

    with nodes_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["node_id", "degree", "point_count"],
        )
        writer.writeheader()
        for node_id in graph.nodes():
            writer.writerow(
                {
                    "node_id": int(node_id),
                    "degree": int(graph.degree(node_id)),
                    "point_count": int(len(graph.nodes[node_id]["pts"])),
                }
            )

    with node_points_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["node_id", "point_index", "z", "y", "x"],
        )
        writer.writeheader()
        for node_id in graph.nodes():
            points_zyx = np.asarray(graph.nodes[node_id]["pts"], dtype=np.int32)
            for point_index, point in enumerate(points_zyx):
                point = point + np.asarray(index_offset_zyx, dtype=np.int32)
                writer.writerow(
                    {
                        "node_id": int(node_id),
                        "point_index": int(point_index),
                        "z": int(point[0]),
                        "y": int(point[1]),
                        "x": int(point[2]),
                    }
                )

    with edges_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["edge_id", "node_u", "node_v", "point_count"],
        )
        writer.writeheader()
        for edge_id, (u, v, data) in enumerate(graph.edges(data=True), start=1):
            writer.writerow(
                {
                    "edge_id": int(edge_id),
                    "node_u": int(u),
                    "node_v": int(v),
                    "point_count": int(len(data["pts"])),
                }
            )

    with edge_points_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["edge_id", "node_u", "node_v", "point_index", "z", "y", "x"],
        )
        writer.writeheader()
        for edge_id, (u, v, data) in enumerate(graph.edges(data=True), start=1):
            points_zyx = np.asarray(data["pts"], dtype=np.int32)
            for point_index, point in enumerate(points_zyx):
                point = point + np.asarray(index_offset_zyx, dtype=np.int32)
                writer.writerow(
                    {
                        "edge_id": int(edge_id),
                        "node_u": int(u),
                        "node_v": int(v),
                        "point_index": int(point_index),
                        "z": int(point[0]),
                        "y": int(point[1]),
                        "x": int(point[2]),
                    }
                )

    return {
        "graph_nodes_csv": str(nodes_path),
        "graph_node_points_csv": str(node_points_path),
        "graph_edges_csv": str(edges_path),
        "graph_edge_points_csv": str(edge_points_path),
    }


def summarize_reporting(
    image: sitk.Image,
    lungs_binary: sitk.Image,
    vessel_mask: sitk.Image,
    skeleton_image: sitk.Image,
    graph,
    branch_rows: list[dict[str, float | int]],
    caliber_thresholds_mm: dict[str, dict[str, float]] | None = None,
) -> dict[str, object]:
    thresholds = normalize_caliber_thresholds(caliber_thresholds_mm)
    vessel_arr = sitk.GetArrayFromImage(vessel_mask) > 0
    lung_arr = sitk.GetArrayFromImage(lungs_binary) > 0
    skeleton_arr = sitk.GetArrayFromImage(skeleton_image) > 0

    voxel_volume_mm3 = float(np.prod(image.GetSpacing()))
    lung_volume_mm3 = float(np.count_nonzero(lung_arr) * voxel_volume_mm3)
    vessel_volume_voxels = int(np.count_nonzero(vessel_arr))
    vessel_volume_mm3 = float(vessel_volume_voxels * voxel_volume_mm3)

    thickness_image = get_thickness_image(vessel_mask)
    thickness_arr = sitk.GetArrayFromImage(thickness_image).astype(np.float32)

    skeleton_thickness_mm = thickness_arr[skeleton_arr]
    vessel_thickness_mm = thickness_arr[vessel_arr]

    branch_lengths_mm = np.asarray(
        [row["path_length_mm"] for row in branch_rows],
        dtype=np.float64,
    )
    branch_tortuosity = np.asarray(
        [row["tortuosity"] for row in branch_rows],
        dtype=np.float64,
    )
    branch_curvature = np.asarray(
        [row["mean_curvature_per_mm"] for row in branch_rows],
        dtype=np.float64,
    )
    branch_mean_thickness = np.asarray(
        [row["mean_thickness_mm"] for row in branch_rows],
        dtype=np.float64,
    )

    small_max = thresholds["small"]["max"]
    large_min = thresholds["large"]["min"]
    small_branch_length_mm = float(
        branch_lengths_mm[branch_mean_thickness < small_max].sum()
    )
    large_branch_length_mm = float(
        branch_lengths_mm[branch_mean_thickness >= large_min].sum()
    )
    total_branch_length_mm = float(branch_lengths_mm.sum())

    endpoint_count = 0
    junction_count = 0
    if graph is not None:
        endpoint_count = int(sum(graph.degree(node) == 1 for node in graph.nodes()))
        junction_count = int(sum(graph.degree(node) >= 3 for node in graph.nodes()))

    return {
        "volume": {
            "vessel_voxel_count": vessel_volume_voxels,
            "vessel_volume_mm3": vessel_volume_mm3,
            "vessel_volume_ml": float(vessel_volume_mm3 / 1000.0),
            "lung_volume_mm3": lung_volume_mm3,
            "lung_volume_ml": float(lung_volume_mm3 / 1000.0),
        },
        "length": {
            "total_skeleton_length_mm": total_branch_length_mm,
            "total_skeleton_length_cm": float(total_branch_length_mm / 10.0),
        },
        "thickness_mm": numeric_summary(skeleton_thickness_mm),
        "tortuosity": {
            "mean": float(np.mean(branch_tortuosity)) if branch_tortuosity.size else None,
            "median": float(np.median(branch_tortuosity)) if branch_tortuosity.size else None,
            "max": float(np.max(branch_tortuosity)) if branch_tortuosity.size else None,
        },
        "curvature_per_mm": {
            "mean": float(np.mean(branch_curvature)) if branch_curvature.size else None,
            "median": float(np.median(branch_curvature)) if branch_curvature.size else None,
            "max": float(np.max(branch_curvature)) if branch_curvature.size else None,
        },
        "branching": {
            "branch_count": int(len(branch_rows)),
            "endpoint_count": endpoint_count,
            "junction_count": junction_count,
            "mask_connected_components": mask_component_count(vessel_mask),
            "skeleton_connected_components": graph_component_count(graph),
        },
        "caliber_distribution": {
            "thresholds_mm": thresholds,
            "skeleton_point_fraction": caliber_bin_fractions(
                skeleton_thickness_mm,
                thresholds,
            ),
            "vessel_voxel_fraction": caliber_bin_fractions(
                vessel_thickness_mm,
                thresholds,
            ),
        },
        "small_vessel": {
            "small_skeleton_length_mm": small_branch_length_mm,
            "small_skeleton_length_fraction": float(
                small_branch_length_mm / total_branch_length_mm
            )
            if total_branch_length_mm > 0
            else 0.0,
            "large_skeleton_length_mm": large_branch_length_mm,
            "large_skeleton_length_fraction": float(
                large_branch_length_mm / total_branch_length_mm
            )
            if total_branch_length_mm > 0
            else 0.0,
            "small_skeleton_length_mm_per_ml_lung": float(
                small_branch_length_mm / (lung_volume_mm3 / 1000.0)
            )
            if lung_volume_mm3 > 0
            else 0.0,
        },
    }
