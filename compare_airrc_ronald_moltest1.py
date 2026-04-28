from __future__ import annotations

import argparse
import gc
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None

RONALD_MASK_NAME = "bronco_final.nrrd"
MASK_EXTENSIONS = (".nrrd", ".nii.gz", ".nii", ".mha", ".mhd")
AIRRC_AIRWAY_LABELS = {1, 2}
AIRRC_VESSEL_LABELS = {3, 4}
RONALD_AIRWAY_LABELS = {1, 2}
RONALD_VESSEL_LABELS = {3}


@dataclass
class PatientPair:
    patient_id: str
    ronald_mask_path: Path
    airrc_mask_path: Path


def normalize_patient_id(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "", value)
    return value


def strip_known_extensions(path: Path) -> str:
    name = path.name
    for suffix in (".nii.gz", ".nrrd", ".nii", ".mha", ".mhd"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def normalize_airrc_stem(path: Path) -> str:
    stem = strip_known_extensions(path)
    stem = re.sub(r"(_pred|_mask|_labels?)$", "", stem, flags=re.IGNORECASE)
    return normalize_patient_id(stem)


def geometry_signature(image: sitk.Image) -> tuple:
    return (
        tuple(image.GetSize()),
        tuple(round(x, 6) for x in image.GetSpacing()),
        tuple(round(x, 6) for x in image.GetOrigin()),
        tuple(round(x, 6) for x in image.GetDirection()),
    )


def geometry_signature_from_path(path: Path) -> tuple:
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    return (
        tuple(int(x) for x in reader.GetSize()),
        tuple(round(float(x), 6) for x in reader.GetSpacing()),
        tuple(round(float(x), 6) for x in reader.GetOrigin()),
        tuple(round(float(x), 6) for x in reader.GetDirection()),
    )


def resample_to_reference(moving: sitk.Image, reference: sitk.Image) -> sitk.Image:
    if geometry_signature(moving) == geometry_signature(reference):
        return moving
    return sitk.Resample(
        moving,
        reference,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        moving.GetPixelID(),
    )


def load_label_array(path: Path) -> tuple[sitk.Image, np.ndarray]:
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)
    return image, array


def binary_from_labels(array: np.ndarray, labels: set[int]) -> np.ndarray:
    return np.isin(array, list(labels))


def skeletonize_3d(mask: np.ndarray, backend: str = "sitk") -> np.ndarray:
    if mask.ndim != 3:
        raise ValueError(f"Expected a 3D mask, got shape {mask.shape}")
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)

    backend = backend.lower().strip()
    if backend == "sitk":
        img = sitk.GetImageFromArray(mask.astype(np.uint8))
        skel = sitk.BinaryThinning(img)
        return sitk.GetArrayFromImage(skel) > 0

    if backend == "skimage":
        from skimage.morphology import skeletonize

        return skeletonize(mask.astype(bool))

    raise ValueError(f"Unsupported skeleton backend: {backend}")


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a_sum = int(a.sum())
    b_sum = int(b.sum())
    if a_sum == 0 and b_sum == 0:
        return 1.0
    denom = a_sum + b_sum
    if denom == 0:
        return 0.0
    return float(2 * np.logical_and(a, b).sum() / denom)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def containment_fraction(source: np.ndarray, target: np.ndarray) -> float:
    source_count = int(source.sum())
    if source_count == 0:
        return math.nan
    return float(np.logical_and(source, target).sum() / source_count)


def _skeleton_points_mm(mask: np.ndarray, spacing_xyz: tuple[float, float, float]) -> np.ndarray:
    points_zyx = np.argwhere(mask)
    if len(points_zyx) == 0:
        return np.empty((0, 3), dtype=np.float64)
    points_xyz = points_zyx[:, [2, 1, 0]].astype(np.float64)
    spacing = np.asarray(spacing_xyz, dtype=np.float64)
    return points_xyz * spacing


def skeleton_distance_metrics(
    a_skeleton: np.ndarray,
    b_skeleton: np.ndarray,
    spacing_xyz: tuple[float, float, float],
) -> dict[str, float]:
    if cKDTree is None:
        return {
            "mean_min_distance_airrc_to_ronald_mm": math.nan,
            "mean_min_distance_ronald_to_airrc_mm": math.nan,
            "chamfer_distance_mm": math.nan,
            "hd95_distance_mm": math.nan,
            "hausdorff_distance_mm": math.nan,
        }

    pts_a = _skeleton_points_mm(a_skeleton, spacing_xyz)
    pts_b = _skeleton_points_mm(b_skeleton, spacing_xyz)

    if len(pts_a) == 0 and len(pts_b) == 0:
        return {
            "mean_min_distance_airrc_to_ronald_mm": 0.0,
            "mean_min_distance_ronald_to_airrc_mm": 0.0,
            "chamfer_distance_mm": 0.0,
            "hd95_distance_mm": 0.0,
            "hausdorff_distance_mm": 0.0,
        }
    if len(pts_a) == 0 or len(pts_b) == 0:
        return {
            "mean_min_distance_airrc_to_ronald_mm": math.nan,
            "mean_min_distance_ronald_to_airrc_mm": math.nan,
            "chamfer_distance_mm": math.nan,
            "hd95_distance_mm": math.nan,
            "hausdorff_distance_mm": math.nan,
        }

    tree_a = cKDTree(pts_a)
    tree_b = cKDTree(pts_b)
    d_a_to_b = tree_b.query(pts_a, k=1)[0]
    d_b_to_a = tree_a.query(pts_b, k=1)[0]
    all_min_d = np.concatenate([d_a_to_b, d_b_to_a])

    return {
        "mean_min_distance_airrc_to_ronald_mm": float(np.mean(d_a_to_b)),
        "mean_min_distance_ronald_to_airrc_mm": float(np.mean(d_b_to_a)),
        "chamfer_distance_mm": float((np.mean(d_a_to_b) + np.mean(d_b_to_a)) / 2.0),
        "hd95_distance_mm": float(np.percentile(all_min_d, 95)),
        "hausdorff_distance_mm": float(np.max(all_min_d)),
    }


def find_ronald_masks(root: Path, mask_name: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for patient_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        mask_path = patient_dir / mask_name
        if mask_path.exists():
            out[normalize_patient_id(patient_dir.name)] = mask_path
    return out


def iter_mask_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        lower = path.name.lower()
        if lower.endswith(MASK_EXTENSIONS):
            yield path


def build_airrc_name_index(root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in iter_mask_files(root):
        keys = {
            normalize_airrc_stem(path),
            normalize_patient_id(path.parent.name),
        }
        for key in keys:
            if not key:
                continue
            index.setdefault(key, []).append(path)
    return index


def build_geometry_index(root: Path) -> tuple[dict[tuple, list[Path]], pd.DataFrame]:
    index: dict[tuple, list[Path]] = {}
    rows: list[dict[str, object]] = []
    for path in iter_mask_files(root):
        try:
            signature = geometry_signature_from_path(path)
            index.setdefault(signature, []).append(path)
            rows.append(
                {
                    "mask_path": str(path),
                    "size": signature[0],
                    "spacing": signature[1],
                    "origin": signature[2],
                    "direction": signature[3],
                    "status": "ok",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "mask_path": str(path),
                    "size": None,
                    "spacing": None,
                    "origin": None,
                    "direction": None,
                    "status": f"error:{exc}",
                }
            )
    return index, pd.DataFrame(rows)


def load_pairing_csv(path: Path | None) -> dict[str, PatientPair]:
    if path is None:
        return {}
    df = pd.read_csv(path)
    required = {"patient_id", "airrc_path", "ronald_path"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in pairing CSV: {sorted(missing)}")
    pairs: dict[str, PatientPair] = {}
    for row in df.to_dict("records"):
        patient_id = str(row["patient_id"]).strip()
        pairs[normalize_patient_id(patient_id)] = PatientPair(
            patient_id=patient_id,
            ronald_mask_path=Path(str(row["ronald_path"])),
            airrc_mask_path=Path(str(row["airrc_path"])),
        )
    return pairs


def build_pairs(
    ronald_root: Path,
    airrc_root: Path,
    pairing_csv: Path | None,
    ronald_mask_name: str,
) -> tuple[list[PatientPair], pd.DataFrame, pd.DataFrame]:
    manual_pairs = load_pairing_csv(pairing_csv)
    ronald = find_ronald_masks(ronald_root, ronald_mask_name)
    airrc_name_index = build_airrc_name_index(airrc_root)
    airrc_geometry_index: dict[tuple, list[Path]] | None = None
    airrc_geometry_debug = pd.DataFrame()

    pairs: list[PatientPair] = []
    debug_rows: list[dict[str, object]] = []

    for patient_key, ronald_path in sorted(ronald.items()):
        patient_id = Path(ronald_path).parent.name
        if patient_key in manual_pairs:
            pair = manual_pairs[patient_key]
            if pair.ronald_mask_path != ronald_path:
                pair = PatientPair(
                    patient_id=pair.patient_id,
                    ronald_mask_path=ronald_path,
                    airrc_mask_path=pair.airrc_mask_path,
                )
            if pair.airrc_mask_path.exists():
                pairs.append(pair)
                debug_rows.append(
                    {
                        "patient_id": pair.patient_id,
                        "pairing_mode": "manual",
                        "ronald_mask_path": str(pair.ronald_mask_path),
                        "airrc_mask_path": str(pair.airrc_mask_path),
                        "status": "paired",
                    }
                )
            else:
                debug_rows.append(
                    {
                        "patient_id": pair.patient_id,
                        "pairing_mode": "manual",
                        "ronald_mask_path": str(pair.ronald_mask_path),
                        "airrc_mask_path": str(pair.airrc_mask_path),
                        "status": "missing_airrc_path",
                    }
                )
            continue

        name_candidates = airrc_name_index.get(patient_key, [])
        if len(name_candidates) == 1:
            pairs.append(
                PatientPair(
                    patient_id=patient_id,
                    ronald_mask_path=ronald_path,
                    airrc_mask_path=name_candidates[0],
                )
            )
            debug_rows.append(
                {
                    "patient_id": patient_id,
                    "pairing_mode": "auto_name",
                    "ronald_mask_path": str(ronald_path),
                    "airrc_mask_path": str(name_candidates[0]),
                    "status": "paired",
                }
            )
            continue

        if len(name_candidates) > 1:
            debug_rows.append(
                {
                    "patient_id": patient_id,
                    "pairing_mode": "auto_name",
                    "ronald_mask_path": str(ronald_path),
                    "airrc_mask_path": " | ".join(str(path) for path in name_candidates),
                    "status": "ambiguous_name_match",
                }
            )
            continue

        if airrc_geometry_index is None:
            airrc_geometry_index, airrc_geometry_debug = build_geometry_index(airrc_root)

        try:
            ronald_signature = geometry_signature_from_path(ronald_path)
        except Exception as exc:
            debug_rows.append(
                {
                    "patient_id": patient_id,
                    "pairing_mode": "auto_geometry",
                    "ronald_mask_path": str(ronald_path),
                    "airrc_mask_path": "",
                    "status": f"ronald_geometry_error:{exc}",
                }
            )
            continue

        geometry_candidates = airrc_geometry_index.get(ronald_signature, [])
        if len(geometry_candidates) == 1:
            pairs.append(
                PatientPair(
                    patient_id=patient_id,
                    ronald_mask_path=ronald_path,
                    airrc_mask_path=geometry_candidates[0],
                )
            )
            debug_rows.append(
                {
                    "patient_id": patient_id,
                    "pairing_mode": "auto_geometry_exact",
                    "ronald_mask_path": str(ronald_path),
                    "airrc_mask_path": str(geometry_candidates[0]),
                    "status": "paired",
                }
            )
        elif len(geometry_candidates) > 1:
            debug_rows.append(
                {
                    "patient_id": patient_id,
                    "pairing_mode": "auto_geometry_exact",
                    "ronald_mask_path": str(ronald_path),
                    "airrc_mask_path": " | ".join(str(path) for path in geometry_candidates),
                    "status": "ambiguous_geometry_match",
                }
            )
        else:
            debug_rows.append(
                {
                    "patient_id": patient_id,
                    "pairing_mode": "auto_geometry_exact",
                    "ronald_mask_path": str(ronald_path),
                    "airrc_mask_path": "",
                    "status": "no_match",
                }
            )

    return pairs, pd.DataFrame(debug_rows), airrc_geometry_debug


def compare_skeletons_for_pair(
    pair: PatientPair,
    skeleton_backend: str = "sitk",
) -> list[dict[str, object]]:
    airrc_img, airrc_arr = load_label_array(pair.airrc_mask_path)
    ronald_img, ronald_arr = load_label_array(pair.ronald_mask_path)
    if geometry_signature(airrc_img) != geometry_signature(ronald_img):
        ronald_img = resample_to_reference(ronald_img, airrc_img)
        ronald_arr = sitk.GetArrayFromImage(ronald_img)
        resampled = True
    else:
        resampled = False

    comparisons = [
        ("vessels", AIRRC_VESSEL_LABELS, RONALD_VESSEL_LABELS),
        ("airways_walls", AIRRC_AIRWAY_LABELS, RONALD_AIRWAY_LABELS),
    ]

    rows: list[dict[str, object]] = []
    for structure_name, airrc_labels, ronald_labels in comparisons:
        airrc_mask = binary_from_labels(airrc_arr, airrc_labels)
        ronald_mask = binary_from_labels(ronald_arr, ronald_labels)
        airrc_skeleton = skeletonize_3d(airrc_mask, backend=skeleton_backend)
        ronald_skeleton = skeletonize_3d(ronald_mask, backend=skeleton_backend)
        intersection = np.logical_and(airrc_skeleton, ronald_skeleton)
        distance_metrics = skeleton_distance_metrics(
            airrc_skeleton,
            ronald_skeleton,
            tuple(float(v) for v in airrc_img.GetSpacing()),
        )
        rows.append(
            {
                "patient_id": pair.patient_id,
                "structure": structure_name,
                "airrc_mask_path": str(pair.airrc_mask_path),
                "ronald_mask_path": str(pair.ronald_mask_path),
                "resampled_ronald_to_airrc": resampled,
                "airrc_mask_voxels": int(airrc_mask.sum()),
                "ronald_mask_voxels": int(ronald_mask.sum()),
                "mask_dice": dice(airrc_mask, ronald_mask),
                "mask_jaccard": jaccard(airrc_mask, ronald_mask),
                "airrc_skeleton_voxels": int(airrc_skeleton.sum()),
                "ronald_skeleton_voxels": int(ronald_skeleton.sum()),
                "skeleton_intersection_voxels": int(intersection.sum()),
                "skeleton_dice": dice(airrc_skeleton, ronald_skeleton),
                "skeleton_jaccard": jaccard(airrc_skeleton, ronald_skeleton),
                "airrc_skeleton_inside_ronald_mask_frac": containment_fraction(
                    airrc_skeleton, ronald_mask
                ),
                "ronald_skeleton_inside_airrc_mask_frac": containment_fraction(
                    ronald_skeleton, airrc_mask
                ),
                **distance_metrics,
            }
        )
    return rows


def discover_coordinate_csvs(root: Path) -> list[Path]:
    csvs: list[Path] = []
    for path in sorted(root.rglob("*.csv")):
        try:
            header = pd.read_csv(path, nrows=0).columns.tolist()
        except Exception:
            continue
        header_set = set(header)
        has_patient = any(
            column in header_set
            for column in ["Patient", "Patient Name", "patient", "patient_name"]
        )
        has_world = {"X_world", "Y_world", "Z_world"}.issubset(header_set)
        has_index = {"X_index", "Y_index", "Z_index"}.issubset(header_set)
        if has_patient and (has_world or has_index):
            csvs.append(path)
    return csvs


def get_patient_column(columns: Iterable[str]) -> str | None:
    for column in ["Patient", "Patient Name", "patient", "patient_name"]:
        if column in columns:
            return column
    return None


def sample_label_at_row(
    mask_image: sitk.Image, row: pd.Series
) -> tuple[int | None, str, str]:
    columns = set(row.index)
    if {"X_world", "Y_world", "Z_world"}.issubset(columns):
        try:
            point = (
                float(row["X_world"]),
                float(row["Y_world"]),
                float(row["Z_world"]),
            )
            index = mask_image.TransformPhysicalPointToIndex(point)
            size = mask_image.GetSize()
            if all(0 <= index[i] < size[i] for i in range(3)):
                label = int(mask_image[index])
                return label, "world", "ok"
            return None, "world", "out_of_bounds"
        except Exception as exc:
            world_error = f"world_error:{exc}"
        else:
            world_error = "world_failed"
    else:
        world_error = "world_missing"

    if {"X_index", "Y_index", "Z_index"}.issubset(columns):
        try:
            x = int(round(float(row["X_index"])))
            y = int(round(float(row["Y_index"])))
            z = int(round(float(row["Z_index"])))
            size = mask_image.GetSize()
            if 0 <= x < size[0] and 0 <= y < size[1] and 0 <= z < size[2]:
                label = int(mask_image[(x, y, z)])
                return label, "index", "ok"
            return None, "index", "out_of_bounds"
        except Exception as exc:
            return None, "index", f"index_error:{exc}"
    return None, "none", world_error


def evaluate_coordinate_csvs(
    csv_paths: list[Path], pairs: list[PatientPair]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_lookup = {normalize_patient_id(pair.patient_id): pair for pair in pairs}
    mask_cache: dict[str, sitk.Image] = {}
    detailed_rows: list[dict[str, object]] = []

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        patient_column = get_patient_column(df.columns)
        if patient_column is None:
            continue
        for row_index, row in df.iterrows():
            patient_value = str(row.get(patient_column, "")).strip()
            patient_key = normalize_patient_id(patient_value)
            pair = pair_lookup.get(patient_key)
            if pair is None:
                detailed_rows.append(
                    {
                        "source_csv": csv_path.name,
                        "row_index": row_index,
                        "patient_id": patient_value,
                        "airrc_mask_path": "",
                        "status": "no_patient_pair",
                        "sampling_method": "none",
                        "airrc_label": np.nan,
                        "inside_airrc_any": np.nan,
                        "inside_airrc_airways_walls": np.nan,
                        "inside_airrc_vessels": np.nan,
                        "class_group": row.get("Class_group", ""),
                    }
                )
                continue

            cache_key = str(pair.airrc_mask_path)
            if cache_key not in mask_cache:
                mask_cache[cache_key] = sitk.ReadImage(str(pair.airrc_mask_path))
            mask_image = mask_cache[cache_key]
            label, sampling_method, status = sample_label_at_row(mask_image, row)
            detailed_rows.append(
                {
                    "source_csv": csv_path.name,
                    "row_index": row_index,
                    "patient_id": pair.patient_id,
                    "airrc_mask_path": str(pair.airrc_mask_path),
                    "status": status,
                    "sampling_method": sampling_method,
                    "airrc_label": label,
                    "inside_airrc_any": (
                        int(label in AIRRC_AIRWAY_LABELS.union(AIRRC_VESSEL_LABELS))
                        if label is not None
                        else np.nan
                    ),
                    "inside_airrc_airways_walls": (
                        int(label in AIRRC_AIRWAY_LABELS)
                        if label is not None
                        else np.nan
                    ),
                    "inside_airrc_vessels": (
                        int(label in AIRRC_VESSEL_LABELS)
                        if label is not None
                        else np.nan
                    ),
                    "class_group": row.get("Class_group", ""),
                    "X_world": row.get("X_world", np.nan),
                    "Y_world": row.get("Y_world", np.nan),
                    "Z_world": row.get("Z_world", np.nan),
                    "X_index": row.get("X_index", np.nan),
                    "Y_index": row.get("Y_index", np.nan),
                    "Z_index": row.get("Z_index", np.nan),
                }
            )

    detailed_df = pd.DataFrame(detailed_rows)
    if detailed_df.empty:
        return detailed_df, pd.DataFrame()

    valid_df = detailed_df[detailed_df["status"] == "ok"].copy()
    if valid_df.empty:
        return detailed_df, pd.DataFrame()

    summary = (
        valid_df.groupby(["source_csv", "patient_id"], as_index=False)
        .agg(
            total_points=("row_index", "count"),
            inside_airrc_any=("inside_airrc_any", "sum"),
            inside_airrc_airways_walls=("inside_airrc_airways_walls", "sum"),
            inside_airrc_vessels=("inside_airrc_vessels", "sum"),
        )
        .sort_values(by=["source_csv", "patient_id"])
    )
    summary["outside_airrc_parenchyma"] = (
        summary["total_points"] - summary["inside_airrc_any"]
    )
    summary["inside_airrc_any_fraction"] = (
        summary["inside_airrc_any"] / summary["total_points"]
    )

    overall_summary = pd.DataFrame(
        [
            {
                "source_csv": "__all__",
                "patient_id": "__all__",
                "total_points": int(summary["total_points"].sum()),
                "inside_airrc_any": int(summary["inside_airrc_any"].sum()),
                "inside_airrc_airways_walls": int(
                    summary["inside_airrc_airways_walls"].sum()
                ),
                "inside_airrc_vessels": int(summary["inside_airrc_vessels"].sum()),
            }
        ]
    )
    overall_summary["outside_airrc_parenchyma"] = (
        overall_summary["total_points"] - overall_summary["inside_airrc_any"]
    )
    overall_summary["inside_airrc_any_fraction"] = (
        overall_summary["inside_airrc_any"] / overall_summary["total_points"]
    )

    summary = pd.concat([summary, overall_summary], ignore_index=True)
    return detailed_df, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Moltest 1 AirRC vs RONALD masks and evaluate coordinate hits against AirRC masks.",
    )
    parser.add_argument(
        "--ronald-root",
        type=Path,
        default=Path("/mnt/pmanas/Ania/phd-data/moltest-1/masks"),
        help="Root directory containing RONALD Moltest 1 patient folders with bronco_final.nrrd.",
    )
    parser.add_argument(
        "--airrc-root",
        type=Path,
        default=Path("/mnt/pmanas/Ania/bronco-pretrain/AirRC/Moltest1_masks"),
        help="Root directory containing AirRC Moltest 1 masks.",
    )
    parser.add_argument(
        "--pairing-csv",
        type=Path,
        default=None,
        help="Optional CSV with explicit patient_id, airrc_path, ronald_path columns.",
    )
    parser.add_argument(
        "--coordinate-csv-root",
        type=Path,
        default=Path("/mnt/pmanas/Ania/phd-data/moltest-1"),
        help="Directory containing pyradiomics/coordinate CSVs.",
    )
    parser.add_argument(
        "--coordinate-csvs",
        type=Path,
        nargs="*",
        default=None,
        help="Optional explicit list of coordinate CSVs. If omitted, CSVs are auto-discovered.",
    )
    parser.add_argument(
        "--ronald-mask-name",
        type=str,
        default=RONALD_MASK_NAME,
        help="Mask filename under each RONALD patient directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./outputs/moltest1_airrc_ronald_comparison"),
        help="Directory where CSV outputs will be written.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit of paired patients for quick debugging.",
    )
    parser.add_argument(
        "--skeleton-backend",
        type=str,
        default="sitk",
        choices=["sitk", "skimage"],
        help="Skeletonization backend. 'sitk' is more stable for long runs.",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=25,
        help="Write progress CSV every N paired patients.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from progress CSV and skip already processed patients.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pairs, pairing_debug, airrc_geometry_debug = build_pairs(
        ronald_root=args.ronald_root,
        airrc_root=args.airrc_root,
        pairing_csv=args.pairing_csv,
        ronald_mask_name=args.ronald_mask_name,
    )
    if args.limit is not None:
        pairs = pairs[: args.limit]

    pairing_debug.to_csv(
        args.output_dir / "airrc_ronald_pairing_debug.csv", index=False
    )
    airrc_geometry_debug.to_csv(
        args.output_dir / "airrc_geometry_debug.csv", index=False
    )
    if len(pairs) == 0:
        raise RuntimeError(
            "No paired AirRC/RONALD masks found. Check --airrc-root and inspect airrc_ronald_pairing_debug.csv / airrc_geometry_debug.csv."
        )

    progress_csv = args.output_dir / "moltest1_airrc_ronald_skeleton_comparison.progress.csv"
    final_csv = args.output_dir / "moltest1_airrc_ronald_skeleton_comparison.csv"

    skeleton_rows: list[dict[str, object]] = []
    processed_keys: set[str] = set()

    if args.resume and progress_csv.exists():
        previous = pd.read_csv(progress_csv)
        if not previous.empty and "patient_id" in previous.columns:
            skeleton_rows = previous.to_dict("records")
            processed_keys = {
                normalize_patient_id(v)
                for v in previous["patient_id"].dropna().astype(str).tolist()
            }

    run_pairs = [
        pair for pair in pairs if normalize_patient_id(pair.patient_id) not in processed_keys
    ]

    for idx, pair in enumerate(tqdm(run_pairs, desc="Skeleton comparison"), start=1):
        try:
            skeleton_rows.extend(
                compare_skeletons_for_pair(
                    pair,
                    skeleton_backend=args.skeleton_backend,
                )
            )
        except Exception as exc:
            skeleton_rows.append(
                {
                    "patient_id": pair.patient_id,
                    "structure": "error",
                    "airrc_mask_path": str(pair.airrc_mask_path),
                    "ronald_mask_path": str(pair.ronald_mask_path),
                    "resampled_ronald_to_airrc": np.nan,
                    "airrc_mask_voxels": np.nan,
                    "ronald_mask_voxels": np.nan,
                    "mask_dice": np.nan,
                    "mask_jaccard": np.nan,
                    "airrc_skeleton_voxels": np.nan,
                    "ronald_skeleton_voxels": np.nan,
                    "skeleton_intersection_voxels": np.nan,
                    "skeleton_dice": np.nan,
                    "skeleton_jaccard": np.nan,
                    "airrc_skeleton_inside_ronald_mask_frac": np.nan,
                    "ronald_skeleton_inside_airrc_mask_frac": np.nan,
                    "mean_min_distance_airrc_to_ronald_mm": np.nan,
                    "mean_min_distance_ronald_to_airrc_mm": np.nan,
                    "chamfer_distance_mm": np.nan,
                    "hd95_distance_mm": np.nan,
                    "hausdorff_distance_mm": np.nan,
                    "error": str(exc),
                }
            )

        if args.flush_every > 0 and idx % args.flush_every == 0:
            pd.DataFrame(skeleton_rows).to_csv(progress_csv, index=False)
            gc.collect()

    pd.DataFrame(skeleton_rows).to_csv(progress_csv, index=False)
    pd.DataFrame(skeleton_rows).to_csv(final_csv, index=False)

    coordinate_csvs = (
        args.coordinate_csvs
        if args.coordinate_csvs
        else discover_coordinate_csvs(args.coordinate_csv_root)
    )
    if len(coordinate_csvs) == 0:
        raise RuntimeError(
            "No coordinate CSVs found. Pass --coordinate-csvs explicitly or check --coordinate-csv-root."
        )

    detailed_df, summary_df = evaluate_coordinate_csvs(list(coordinate_csvs), pairs)
    detailed_df.to_csv(
        args.output_dir / "moltest1_airrc_coordinate_hits_detailed.csv", index=False
    )
    summary_df.to_csv(
        args.output_dir / "moltest1_airrc_coordinate_hits_summary.csv", index=False
    )

    csv_inventory = pd.DataFrame(
        [{"coordinate_csv": str(path)} for path in coordinate_csvs]
    )
    csv_inventory.to_csv(
        args.output_dir / "moltest1_coordinate_csv_inventory.csv", index=False
    )

    print(f"Paired patients: {len(pairs)}")
    print(f"Skeleton backend: {args.skeleton_backend}")
    if args.resume:
        print(f"Resumed from progress: {len(processed_keys)} patients already processed")
    print(f"Coordinate CSVs: {len(coordinate_csvs)}")
    print(f"Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
