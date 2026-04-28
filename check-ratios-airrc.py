#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm


DEFAULT_MALIGNANT_GROUPS = ["guzek", "zmiana podejrzana", "zmiany zapalne"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Create per-change AIRRC BVB cut-off CSV (neighborhood based)."
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
        "--neighborhood-radius",
        type=int,
        default=2,
        help="Neighborhood radius in voxels for BVB hit check (default: 2 => +/-2)",
    )
    return p.parse_args()


def safe_read(path: Path):
    if not path.exists():
        return None
    return sitk.ReadImage(str(path))


def main():
    args = parse_args()
    df = pd.read_csv(args.nodules_csv, index_col=0)

    required_cols = {"Patient", "X_index", "Y_index", "Z_index", "Class_group"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {args.nodules_csv}: {sorted(missing)}")

    malignant_set = set(args.malignant_groups)
    bvb_labels = set(args.bvb_labels)
    r = int(args.neighborhood_radius)

    out_rows = []
    errors = []
    mask_cache = {}
    ct_cache = {}

    for row_idx, row in tqdm(df.iterrows(), total=len(df), desc="AIRRC per-change"):
        patient = str(row["Patient"]) 
        x = int(row["X_index"])
        y = int(row["Y_index"])
        z = int(row["Z_index"])
        class_group = str(row["Class_group"])

        mask_path = args.airrc_root / args.airrc_pattern.format(patient=patient)
        ct_path = args.ct_root / args.ct_pattern.format(patient=patient)

        if patient not in mask_cache:
            mask_cache[patient] = safe_read(mask_path)
        if patient not in ct_cache:
            ct_cache[patient] = safe_read(ct_path)

        mask_img = mask_cache[patient]
        ct_img = ct_cache[patient]

        result = {
            "source_row": int(row_idx),
            "patient": patient,
            "x_index": x,
            "y_index": y,
            "z_index": z,
            "class_group": class_group,
            "is_malignant_group": int(class_group in malignant_set),
            "status": "ok",
            "all_voxels_bvb_neighborhood": np.nan,
            "max_hu_neighborhood": np.nan,
            "max_hu_non_bvb_neighborhood": np.nan,
            "detected": np.nan,
        }

        if mask_img is None:
            result["status"] = "missing_mask"
            errors.append(f"Missing mask: {mask_path}")
            out_rows.append(result)
            continue

        if ct_img is None:
            result["status"] = "missing_ct"
            errors.append(f"Missing CT: {ct_path}")
            out_rows.append(result)
            continue

        mask_arr = sitk.GetArrayFromImage(mask_img)
        ct_arr = sitk.GetArrayFromImage(ct_img)

        if mask_arr.shape != ct_arr.shape:
            result["status"] = "shape_mismatch"
            errors.append(f"Shape mismatch for {patient}: mask {mask_arr.shape} vs ct {ct_arr.shape}")
            out_rows.append(result)
            continue

        if not (0 <= z < mask_arr.shape[0] and 0 <= y < mask_arr.shape[1] and 0 <= x < mask_arr.shape[2]):
            result["status"] = "coord_out_of_bounds"
            errors.append(f"Out-of-bounds coord for {patient}: ({x},{y},{z})")
            out_rows.append(result)
            continue

        z0, z1 = max(0, z - r), min(mask_arr.shape[0], z + r + 1)
        y0, y1 = max(0, y - r), min(mask_arr.shape[1], y + r + 1)
        x0, x1 = max(0, x - r), min(mask_arr.shape[2], x + r + 1)

        mask_nb = mask_arr[z0:z1, y0:y1, x0:x1]
        ct_nb = ct_arr[z0:z1, y0:y1, x0:x1]

        bvb_mask = np.isin(mask_nb, list(bvb_labels))
        non_bvb_mask = ~bvb_mask

        result["max_hu_neighborhood"] = float(np.max(ct_nb))

        all_bvb = bool(np.all(bvb_mask))
        result["all_voxels_bvb_neighborhood"] = int(all_bvb)

        if all_bvb:
            # fully BVB neighborhood -> not detected
            result["detected"] = 0
        else:
            # any non-BVB voxel in neighborhood means candidate remains visible
            result["detected"] = 1
            result["max_hu_non_bvb_neighborhood"] = float(np.max(ct_nb[non_bvb_mask]))

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
