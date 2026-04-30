#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.spatial import cKDTree
from tqdm import tqdm


DEFAULT_MALIGNANT_GROUPS = ["guzek", "zmiana podejrzana", "zmiany zapalne"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Create per-change AIRRC BVB statistics using voxel-distance matching."
    )
    p.add_argument(
        "--nodules-csv",
        type=Path,
        default=Path("/mnt/pmanas/Ania/copm/pyradiomics_results_MOLTEST 1.csv"),
        help="CSV with at least: Patient, X_index, Y_index, Z_index, Class_group",
    )
    p.add_argument(
        "--airrc-root",
        type=Path,
        default=Path("/mnt/pmanas/Ania/bronco-pretrain/AirRC/Moltest1_masks"),
        help="Folder with AIRRC masks (e.g. <Patient>_pred.nrrd)",
    )
    p.add_argument(
        "--airrc-pattern",
        type=str,
        default="{patient}_pred.nrrd",
        help="Mask filename pattern inside --airrc-root",
    )
    p.add_argument(
        "--ct-root",
        type=Path,
        default=Path("/mnt/pmanas/Wojtek/MIRASLC/Dane surowe/Moltest 1/NRRD"),
        help="Folder with source CT volumes used to report neighborhood max HU",
    )
    p.add_argument(
        "--ct-pattern",
        type=str,
        default="{patient}.nrrd",
        help="CT filename pattern inside --ct-root",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=Path("/mnt/pmanas/Ania/copm/airrc_stats_per_change.csv"),
        help="Output CSV path (per change)",
    )
    p.add_argument(
        "--malignant-groups",
        nargs="+",
        default=DEFAULT_MALIGNANT_GROUPS,
        help="Class_group values treated as malignant/suspicious",
    )
    p.add_argument(
        "--bvb-labels",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        help="AIRRC labels treated as BVB (default: 1 2 3 4)",
    )
    p.add_argument(
        "--match-radius",
        type=float,
        default=1.1,
        help="Euclidean voxel radius for match tolerance (default: 1.1)",
    )
    return p.parse_args()


def safe_read(path: Path):
    if not path.exists():
        return None
    return sitk.ReadImage(str(path))


def offsets_within_radius(radius: float):
    max_delta = int(np.floor(radius)) + 1
    offsets = []
    for dz in range(-max_delta, max_delta + 1):
        for dy in range(-max_delta, max_delta + 1):
            for dx in range(-max_delta, max_delta + 1):
                if np.sqrt(dx * dx + dy * dy + dz * dz) <= radius:
                    offsets.append((dz, dy, dx))
    return offsets


def patient_cache(mask_img, ct_img, bvb_labels, offsets):
    mask_arr = sitk.GetArrayFromImage(mask_img)
    ct_arr = sitk.GetArrayFromImage(ct_img)
    if mask_arr.shape != ct_arr.shape:
        raise ValueError(f"Shape mismatch: mask {mask_arr.shape} vs ct {ct_arr.shape}")

    bvb_mask = np.isin(mask_arr, list(bvb_labels))
    bvb_coords = np.argwhere(bvb_mask)
    bvb_tree = cKDTree(bvb_coords) if len(bvb_coords) else None

    return {
        "mask_arr": mask_arr,
        "ct_arr": ct_arr,
        "bvb_mask": bvb_mask,
        "bvb_tree": bvb_tree,
        "offsets": offsets,
    }


def local_ball(mask_arr, ct_arr, point, offsets):
    z, y, x = point
    coords = []
    labels = []
    hu = []
    zdim, ydim, xdim = mask_arr.shape

    for dz, dy, dx in offsets:
        zz, yy, xx = z + dz, y + dy, x + dx
        if 0 <= zz < zdim and 0 <= yy < ydim and 0 <= xx < xdim:
            coords.append((zz, yy, xx))
            labels.append(mask_arr[zz, yy, xx])
            hu.append(ct_arr[zz, yy, xx])

    if not coords:
        return np.empty((0, 3), dtype=int), np.array([]), np.array([])

    return np.asarray(coords, dtype=int), np.asarray(labels), np.asarray(hu, dtype=float)


def main():
    args = parse_args()
    df = pd.read_csv(args.nodules_csv, index_col=0)

    required_cols = {"Patient", "X_index", "Y_index", "Z_index", "Class_group"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {args.nodules_csv}: {sorted(missing)}")

    malignant_set = set(args.malignant_groups)
    bvb_labels = set(args.bvb_labels)
    offsets = offsets_within_radius(args.match_radius)

    out_rows = []
    errors = []
    patient_data = {}

    for row_idx, row in tqdm(df.iterrows(), total=len(df), desc="AIRRC per-change"):
        patient = str(row["Patient"])
        x = int(row["X_index"])
        y = int(row["Y_index"])
        z = int(row["Z_index"])
        class_group = str(row["Class_group"])

        result = {
            "source_row": int(row_idx),
            "patient": patient,
            "x_index": x,
            "y_index": y,
            "z_index": z,
            "class_group": class_group,
            "is_malignant_group": int(class_group in malignant_set),
            "status": "ok",
            "match_radius_vox": float(args.match_radius),
            "detected": np.nan,
            "all_voxels_bvb_neighborhood": np.nan,
            "max_hu_neighborhood": np.nan,
            "max_hu_non_bvb_neighborhood": np.nan,
            "max_bvb_label_neighborhood": np.nan,
        }

        if patient not in patient_data:
            mask_path = args.airrc_root / args.airrc_pattern.format(patient=patient)
            ct_path = args.ct_root / args.ct_pattern.format(patient=patient)
            mask_img = safe_read(mask_path)
            ct_img = safe_read(ct_path)

            if mask_img is None:
                patient_data[patient] = {"error": f"missing_mask: {mask_path}"}
            elif ct_img is None:
                patient_data[patient] = {"error": f"missing_ct: {ct_path}"}
            else:
                try:
                    patient_data[patient] = patient_cache(mask_img, ct_img, bvb_labels, offsets)
                except Exception as exc:
                    patient_data[patient] = {"error": str(exc)}

        pdata = patient_data[patient]
        if "error" in pdata:
            result["status"] = pdata["error"].split(":", 1)[0]
            errors.append(f"{patient}: {pdata['error']}")
            out_rows.append(result)
            continue

        mask_arr = pdata["mask_arr"]
        ct_arr = pdata["ct_arr"]
        bvb_tree = pdata["bvb_tree"]

        if not (0 <= z < mask_arr.shape[0] and 0 <= y < mask_arr.shape[1] and 0 <= x < mask_arr.shape[2]):
            result["status"] = "coord_out_of_bounds"
            errors.append(f"Out-of-bounds coord for {patient}: ({x},{y},{z})")
            out_rows.append(result)
            continue

        ball_coords, ball_labels, ball_hu = local_ball(mask_arr, ct_arr, (z, y, x), offsets)
        if len(ball_labels) == 0:
            result["status"] = "empty_neighborhood"
            out_rows.append(result)
            continue

        bvb_present = np.isin(ball_labels, list(bvb_labels))
        non_bvb_present = ~bvb_present

        result["all_voxels_bvb_neighborhood"] = int(np.all(bvb_present))
        result["max_hu_neighborhood"] = float(np.max(ball_hu))
        result["max_bvb_label_neighborhood"] = int(np.max(ball_labels[bvb_present])) if np.any(bvb_present) else 0
        result["max_hu_non_bvb_neighborhood"] = float(np.max(ball_hu[non_bvb_present])) if np.any(non_bvb_present) else np.nan

        if bvb_tree is None:
            result["detected"] = 0
        else:
            result["detected"] = int(len(bvb_tree.query_ball_point((z, y, x), r=args.match_radius)) > 0)

        out_rows.append(result)

    out_df = pd.DataFrame(out_rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)

    if errors:
        log_path = args.output_csv.with_suffix(args.output_csv.suffix + ".log")
        log_path.write_text("\n".join(errors) + "\n", encoding="utf-8")

    print(f"Saved: {args.output_csv}")
    print(f"Rows: {len(out_df)}")
    if errors:
        print(f"Warnings/errors: {len(errors)} (see log file)")


if __name__ == "__main__":
    main()
