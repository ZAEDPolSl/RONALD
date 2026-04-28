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
        description="Create AIRRC cut-off ratio CSV analogous to check-ratios.py"
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
        "--output-csv",
        type=Path,
        default=Path("/mnt/pmanas/Ania/copm/airrc_stats.csv"),
        help="Output CSV path",
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

    patients = df.Patient.unique()
    malignant_set = set(args.malignant_groups)
    bvb_labels = set(args.bvb_labels)

    rows = []
    errors = []

    for patient in tqdm(patients, total=len(patients), desc="AIRRC ratios"):
        lumps = np.array(df.loc[df.Patient == patient, ["X_index", "Y_index", "Z_index"]])
        lump_types = df.loc[df.Patient == patient, "Class_group"].tolist()

        mask_path = args.airrc_root / args.airrc_pattern.format(patient=patient)
        mask_img = safe_read(mask_path)

        row = {
            "patient": patient,
            "malignant_cut_off": 0,
            "benign_cut_off": 0,
            "malignant_left": 0,
            "benign_left": 0,
        }

        if mask_img is None:
            errors.append(f"Missing mask: {mask_path}")
            rows.append(row)
            continue

        mask_arr = sitk.GetArrayFromImage(mask_img)

        try:
            for lump_type, lump in zip(lump_types, lumps):
                x, y, z = [int(v) for v in lump]
                inside = False

                if (
                    0 <= z < mask_arr.shape[0]
                    and 0 <= y < mask_arr.shape[1]
                    and 0 <= x < mask_arr.shape[2]
                ):
                    label_val = int(mask_arr[z, y, x])
                    inside = label_val in bvb_labels
                else:
                    errors.append(f"Out-of-bounds coord for {patient}: ({x},{y},{z})")

                is_malignant = lump_type in malignant_set
                if inside:
                    if is_malignant:
                        row["malignant_cut_off"] += 1
                    else:
                        row["benign_cut_off"] += 1
                else:
                    if is_malignant:
                        row["malignant_left"] += 1
                    else:
                        row["benign_left"] += 1

        except Exception as exc:
            errors.append(f"{patient}: {exc}")

        rows.append(row)

    out_df = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)

    if errors:
        log_path = args.output_csv.with_suffix(args.output_csv.suffix + ".log")
        log_path.write_text("\n".join(errors) + "\n", encoding="utf-8")

    print(f"Saved: {args.output_csv}")
    print(f"Patients: {len(rows)}")
    if errors:
        print(f"Warnings/errors: {len(errors)} (see log file)")


if __name__ == "__main__":
    main()
