from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path

import SimpleITK as sitk
from ctools import ImageInstance
from tqdm import tqdm

from bronco.segmentation.mediastinum_segmentation import mediastinum_segmentation
from bronco.segmentation.vessel_segmentation import vessel_segmentation
from bronco.vessel_metrics import summarize_reporting, get_skeleton_image, build_skeleton_graph
from bronco.vessel_metrics import compute_branch_metrics, write_branch_metrics_csv
from bronco.vessel_metrics import normalize_caliber_thresholds, get_thickness_image
from bronco.vessel_metrics import write_graph_tables


def derive_case_name(path: Path) -> str:
    if path.name.endswith(".nii.gz"):
        return path.name[:-7]
    return path.stem


def flatten_dict(data: dict, prefix: str = "") -> dict[str, object]:
    flat: dict[str, object] = {}
    for key, value in data.items():
        new_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_dict(value, new_key))
        else:
            flat[new_key] = value
    return flat


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    config["caliber_thresholds_mm"] = normalize_caliber_thresholds(
        config["caliber_thresholds_mm"]
    )

    studies = config.get("studies")
    if not isinstance(studies, list) or not studies:
        raise ValueError("Config must contain a non-empty 'studies' list.")

    for index, study in enumerate(studies, start=1):
        if "image" not in study or "lung_mask" not in study:
            raise ValueError(
                f"Study {index} must contain both 'image' and 'lung_mask' paths."
            )

    return config


def resolve_output_root(config: dict, config_path: Path, cli_output_dir: Path | None) -> Path:
    if cli_output_dir is not None:
        return cli_output_dir
    if "output_dir" in config:
        return Path(config["output_dir"])
    return config_path.parent / "mri_vessel_reports"


def mask_bounding_box_xyz(mask: sitk.Image, padding: int = 0):
    mask_u8 = sitk.Cast(mask > 0, sitk.sitkUInt8)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask_u8)
    if not stats.HasLabel(1):
        return None

    x, y, z, size_x, size_y, size_z = stats.GetBoundingBox(1)
    image_size = mask.GetSize()
    start = [
        max(0, int(x) - int(padding)),
        max(0, int(y) - int(padding)),
        max(0, int(z) - int(padding)),
    ]
    stop = [
        min(int(image_size[0]), int(x + size_x) + int(padding)),
        min(int(image_size[1]), int(y + size_y) + int(padding)),
        min(int(image_size[2]), int(z + size_z) + int(padding)),
    ]
    size = [stop[i] - start[i] for i in range(3)]
    return tuple(start), tuple(size)


def crop_to_bbox(image: sitk.Image, bbox_xyz):
    index_xyz, size_xyz = bbox_xyz
    return sitk.RegionOfInterest(image, size=size_xyz, index=index_xyz)


def paste_crop_into_reference(
    cropped: sitk.Image,
    reference: sitk.Image,
    bbox_xyz,
    pixel_id: int | None = None,
) -> sitk.Image:
    pixel_id = cropped.GetPixelID() if pixel_id is None else pixel_id
    output = sitk.Image(reference.GetSize(), pixel_id)
    output.CopyInformation(reference)
    return sitk.Paste(
        output,
        cropped,
        sourceSize=cropped.GetSize(),
        sourceIndex=(0, 0, 0),
        destinationIndex=bbox_xyz[0],
    )


def read_image(path: Path) -> sitk.Image:
    reader = ImageInstance(show_exceptions=True)
    image = reader.read(str(path))
    if image is None:
        raise ValueError(f"Unable to read image from {path}.")
    return image


def run_case(
    image_path: Path,
    lung_mask_path: Path,
    case_output_dir: Path,
    caliber_thresholds_mm: dict[str, object],
    case_name: str,
) -> dict[str, object]:
    masks_dir = case_output_dir / "masks"
    centerlines_dir = case_output_dir / "centerlines"
    metrics_dir = case_output_dir / "metrics"
    for path in (masks_dir, centerlines_dir, metrics_dir):
        path.mkdir(parents=True, exist_ok=True)

    def set_stage(progress_bar, message: str) -> None:
        progress_bar.set_postfix_str(message)
        print(f"[{case_name}] {message}", flush=True)

    with tqdm(total=11, desc=f"{case_name}", unit="step", leave=False) as progress:
        set_stage(progress, "read image")
        image = read_image(image_path)
        progress.update(1)

        set_stage(progress, "read lung mask")
        lungs = read_image(lung_mask_path)
        lungs_binary = sitk.Cast(lungs > 0, sitk.sitkUInt8)
        progress.update(1)

        set_stage(progress, "mediastinum")
        mediastinum = mediastinum_segmentation(lungs_binary)
        progress.update(1)

        mediastinum_path = masks_dir / "mediastinum_mask.nrrd"
        vessel_mask_path = masks_dir / "vessel_mask.nrrd"
        skeleton_path = centerlines_dir / "skeleton.nrrd"
        vessel_metrics_json_path = metrics_dir / "vessel_metrics.json"
        branch_metrics_csv_path = metrics_dir / "branch_metrics.csv"

        set_stage(progress, "vessel segmentation")
        vessel_mask = vessel_segmentation(
            image,
            lungs_binary,
            sitk_mediastinum=mediastinum,
            mode="mri",
            check_mediastinum_connectivity=True,
        )
        progress.update(1)

        set_stage(progress, "write early masks")
        sitk.WriteImage(mediastinum, str(mediastinum_path))
        sitk.WriteImage(vessel_mask, str(vessel_mask_path))
        progress.update(1)

        metric_bbox_xyz = mask_bounding_box_xyz(vessel_mask, padding=6)
        if metric_bbox_xyz is None:
            metric_bbox_xyz = mask_bounding_box_xyz(lungs_binary, padding=6)
        image_metrics = crop_to_bbox(image, metric_bbox_xyz)
        lungs_metrics = crop_to_bbox(lungs_binary, metric_bbox_xyz)
        vessel_mask_metrics = crop_to_bbox(vessel_mask, metric_bbox_xyz)
        metric_offset_zyx = tuple(reversed(metric_bbox_xyz[0]))

        del image
        del lungs
        del lungs_binary
        del mediastinum
        gc.collect()

        set_stage(progress, "skeletonize")
        skeleton_image_metrics = get_skeleton_image(vessel_mask_metrics)
        skeleton_image = paste_crop_into_reference(
            skeleton_image_metrics,
            vessel_mask,
            metric_bbox_xyz,
            pixel_id=sitk.sitkUInt8,
        )
        sitk.WriteImage(skeleton_image, str(skeleton_path))
        del skeleton_image
        progress.update(1)

        set_stage(progress, "graph")
        graph = build_skeleton_graph(skeleton_image_metrics)
        progress.update(1)

        set_stage(progress, "thickness + branches")
        thickness_image = get_thickness_image(vessel_mask_metrics)
        branch_rows = compute_branch_metrics(
            graph,
            skeleton_image_metrics,
            thickness_image,
        )
        progress.update(1)

        set_stage(progress, "write graph tables")
        write_branch_metrics_csv(branch_metrics_csv_path, branch_rows)
        graph_outputs = write_graph_tables(
            centerlines_dir,
            graph,
            index_offset_zyx=metric_offset_zyx,
        )
        progress.update(1)

        del thickness_image
        gc.collect()

        set_stage(progress, "summarize metrics")
        vessel_metrics = summarize_reporting(
            image=image_metrics,
            lungs_binary=lungs_metrics,
            vessel_mask=vessel_mask_metrics,
            skeleton_image=skeleton_image_metrics,
            graph=graph,
            branch_rows=branch_rows,
            caliber_thresholds_mm=caliber_thresholds_mm,
        )
        progress.update(1)

        set_stage(progress, "write report")
        report = {
            "inputs": {
                "image": str(image_path),
                "lung_mask": str(lung_mask_path),
            },
            "outputs": {
                "mediastinum_mask": str(mediastinum_path),
                "vessel_mask": str(vessel_mask_path),
                "skeleton": str(skeleton_path),
                "vessel_metrics_json": str(vessel_metrics_json_path),
                "branch_metrics_csv": str(branch_metrics_csv_path),
                **graph_outputs,
            },
            "statistics": vessel_metrics,
        }
        vessel_metrics_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        progress.update(1)

        del image_metrics
        del lungs_metrics
        del vessel_mask_metrics
        del vessel_mask
        del skeleton_image_metrics
        del graph
        del branch_rows
        gc.collect()
    return report


def write_study_metrics_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "MRI vessel reporting pipeline driven by a JSON config. "
            "The config provides caliber thresholds and a list of image/lung-mask pairs."
        )
    )
    parser.add_argument("--config", type=Path, required=True, help="Input JSON config.")
    parser.add_argument("--output-dir", type=Path, help="Optional output root directory.")
    args = parser.parse_args()

    config = load_config(args.config)
    output_root = resolve_output_root(config, args.config, args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    caliber_thresholds_mm = config["caliber_thresholds_mm"]
    studies = config["studies"]

    reports: list[dict[str, object]] = []
    flat_rows: list[dict[str, object]] = []

    for index, study in enumerate(tqdm(studies, desc="Studies", unit="study"), start=1):
        image_path = Path(study["image"])
        lung_mask_path = Path(study["lung_mask"])
        case_name = study.get("name") or derive_case_name(image_path)
        case_output_dir = output_root / case_name

        report = run_case(
            image_path=image_path,
            lung_mask_path=lung_mask_path,
            case_output_dir=case_output_dir,
            caliber_thresholds_mm=caliber_thresholds_mm,
            case_name=case_name,
        )
        reports.append(report)

        flat_row = {
            "study_index": index,
            "study_name": case_name,
            **flatten_dict(report["inputs"], "inputs"),
            **flatten_dict(report["outputs"], "outputs"),
            **flatten_dict(report["statistics"], "statistics"),
        }
        flat_rows.append(flat_row)

    reports_json_path = output_root / "reports.json"
    study_metrics_csv_path = output_root / "study_metrics.csv"
    reports_json_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    write_study_metrics_csv(study_metrics_csv_path, flat_rows)

    summary = {
        "config": str(args.config),
        "output_root": str(output_root),
        "reports_json": str(reports_json_path),
        "study_metrics_csv": str(study_metrics_csv_path),
        "study_count": len(reports),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
