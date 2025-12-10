"""
Dataset preparation and data loading for SWIN UNETR training.
Handles multiple file formats (DICOM, NIFTI, NRRD) using ImageInstance.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

from ctools import ImageInstance


# Dataset configurations
DATASETS_CONFIG = {
    "Moltest 1": {
        "studies_path": "/mnt/pmanas/Wojtek/MIRASLC/Dane surowe/Moltest 1/NRRD",
        "masks_path": "/mnt/pmanas/Ania/phd-data/moltest-1/masks",
    },
    "Moltest 2": {
        "studies_path": "/mnt/pmanas/Wojtek/MIRASLC/Dane surowe/Moltest 2/",
        "masks_path": "/mnt/pmanas/Ania/phd-data/moltest-2/masks",
    },
    "Luna 2016": {
        "studies_path": "/mnt/pmanas/Ania/phd-data/luna-2016/raw/",
        "masks_path": "/mnt/pmanas/Ania/phd-data/luna-2016/masks",
    },
    "OPPRP": {
        "studies_path": "/mnt/pmanas/OPPRP/",
        "masks_path": "/mnt/pmanas/Ania/phd-data/OPPRP/masks",
    },
    "NLST USA": {
        "studies_path": "/mnt/pmanas/Ania/phd-data/nlst-usa/raw/manifest-NLST_allCT/",
        "masks_path": "/mnt/pmanas/Ania/phd-data/nlst-usa/masks",
    },
    "DUKE": {
        "studies_path": "/mnt/pmanas/Ania/phd-data/DUKE/raw/",
        "masks_path": "/mnt/pmanas/Ania/phd-data/DUKE/masks/",
    },
    "Moltest 2_400": {
        "studies_path": "/mnt/pmanas/Ania/phd-data/moltest_2_400/annotations_paths.csv",
        "masks_path": "/mnt/pmanas/Ania/phd-data/moltest_2_400/masks/",
    },
}


def get_patient_name(filename: str, dataset: str) -> str:
    """Extract patient name from filename based on dataset format."""
    if dataset in ["Moltest 1"]:
        patient_name = filename.split("/")[-1].split(".")[0]
    elif dataset in ["Moltest 2", "OPPRP", "NLST USA"]:
        # For directories, just take the last component (patient folder name)
        patient_name = filename.rstrip("/").split("/")[-1]
    elif dataset in ["Luna 2016"]:
        patient_name = filename.split("/")[-1][:-4]
    elif dataset in ["DUKE"]:
        patient_name = filename.split("/")[-1][:-7]
    elif dataset in ["Moltest 2_400"]:
        patient_name = filename.rstrip("/").split("/")[-1]
    return patient_name


def get_dataset_files(dataset_name: str) -> List[str]:
    """
    Get list of all patient files/directories for a dataset.

    Args:
        dataset_name: Name of the dataset from DATASETS_CONFIG

    Returns:
        List of file paths or directory paths for patients
    """
    if dataset_name not in DATASETS_CONFIG:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    studies_path = DATASETS_CONFIG[dataset_name]["studies_path"]
    p = Path(studies_path).absolute()

    if dataset_name in ["Moltest 1"]:
        filenames = sorted([str(path) for path in p.rglob("*.nrrd")])
    elif dataset_name in ["Moltest 2_400"]:
        df = pd.read_csv(studies_path, sep=";")
        df = df.dropna(subset=["folder_path"])
        filenames = sorted(list(df["folder_path"]))
    elif dataset_name in ["Moltest 2", "OPPRP"]:
        subdirs = [x for x in p.iterdir() if x.is_dir()]
        filenames = sorted(
            [
                str(patient_dir)
                for subdir in subdirs
                for patient_dir in subdir.iterdir()
                if patient_dir.is_dir()
            ]
        )
    elif dataset_name in ["Luna 2016"]:
        subdirs = [path for path in p.rglob("*subset*") if path.is_dir()]
        filenames = sorted(
            [str(x) for subdir in subdirs for x in subdir.rglob("*.mhd")]
        )
    elif dataset_name == "NLST USA":
        files_df = pd.read_csv(studies_path + "metadata.csv")
        filenames = (
            studies_path
            + files_df["File Location"].str.split(pat="/", n=1).str[-1]
            + "/"
        ).tolist()
    elif dataset_name == "DUKE":
        subdirs = [path for path in p.rglob("*DLCS_subset*") if path.is_dir()]
        filenames = sorted(
            [str(x) for subdir in subdirs for x in subdir.rglob("*.nii.gz")],
            reverse=True,
        )
    else:
        filenames = []

    return filenames


def create_datalist(
    dataset_names: List[str],
    output_json: str = "bronco_dataset.json",
    val_split: float = 0.2,
    test_split: float = 0.1,
    random_seed: int = 42,
    mask_filename: str = "bronco_final.nrrd",
) -> Dict[str, List[Dict[str, str]]]:
    """
    Create a MONAI-compatible dataset JSON file with train/val/test splits.

    Args:
        dataset_names: List of dataset names to include
        output_json: Path to save the JSON file
        val_split: Fraction of data for validation
        test_split: Fraction of data for testing
        random_seed: Random seed for reproducibility
        mask_filename: Name of the mask file to look for (default: bronco_final.nrrd)

    Returns:
        Dictionary with train/val/test splits
    """
    all_data = []

    for dataset_name in dataset_names:
        print(f"Processing dataset: {dataset_name}")

        filenames = get_dataset_files(dataset_name)
        masks_path = DATASETS_CONFIG[dataset_name]["masks_path"]

        for filename in filenames:
            patient_name = get_patient_name(filename, dataset_name)

            # Determine the image path
            if dataset_name in ["Moltest 2", "Moltest 2_400"]:
                # For DICOM series, use the DICOM directory
                image_path = str(Path(filename) / "DICOM")
            else:
                image_path = filename

            # Check if mask exists
            mask_path = os.path.join(masks_path, patient_name, mask_filename)

            if os.path.exists(mask_path):
                all_data.append(
                    {
                        "image": image_path,
                        "label": mask_path,
                        "patient_id": patient_name,
                        "dataset": dataset_name,
                    }
                )

    print(f"Total patients with masks: {len(all_data)}")

    if len(all_data) == 0:
        raise ValueError("No valid data found! Check if masks exist.")

    # Split data into train/val/test
    train_val, test_data = train_test_split(
        all_data, test_size=test_split, random_state=random_seed
    )

    train_data, val_data = train_test_split(
        train_val, test_size=val_split / (1 - test_split), random_state=random_seed
    )

    datalist = {
        "training": train_data,
        "validation": val_data,
        "testing": test_data,
    }

    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Save to JSON file
    with open(output_json, "w") as f:
        json.dump(datalist, f, indent=2)

    print(f"Dataset saved to {output_json}")

    return datalist


def verify_data_integrity(datalist: Dict[str, List[Dict[str, str]]]) -> Dict[str, int]:
    """
    Verify that all images and labels in the datalist are accessible.

    Args:
        datalist: Dictionary with train/val/test splits

    Returns:
        Dictionary with counts of valid/invalid files per split
    """
    ii = ImageInstance()
    results = {}

    for split_name, split_data in datalist.items():
        valid_count = 0
        invalid_files = []

        print(f"\nVerifying {split_name} split...")

        for item in split_data:
            image_path = item["image"]
            label_path = item["label"]
            patient_id = item.get("patient_id", "unknown")

            try:
                # Try to load image
                _ = ii.read(image_path)

                # Try to load label
                _ = ii.read(label_path)

                valid_count += 1
            except Exception as e:
                invalid_files.append({"patient_id": patient_id, "error": str(e)})

        results[split_name] = {
            "total": len(split_data),
            "valid": valid_count,
            "invalid": len(invalid_files),
        }

        print(f"  Valid: {valid_count}/{len(split_data)}")

        if invalid_files:
            print(f"  Invalid files: {invalid_files[:5]}")  # Show first 5

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create dataset for SWIN UNETR training"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        choices=list(DATASETS_CONFIG.keys()),
        help="Dataset names to include",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="bronco_dataset.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split fraction",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.1,
        help="Test split fraction",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--mask-filename",
        type=str,
        default="bronco_final.nrrd",
        help="Mask filename to use",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify data integrity after creation",
    )

    args = parser.parse_args()

    # Create datalist
    datalist = create_datalist(
        dataset_names=args.datasets,
        output_json=args.output,
        val_split=args.val_split,
        test_split=args.test_split,
        random_seed=args.seed,
        mask_filename=args.mask_filename,
    )

    # Optionally verify
    if args.verify:
        print("\n" + "=" * 50)
        print("Verifying data integrity...")
        print("=" * 50)
        results = verify_data_integrity(datalist)

        print("\n" + "=" * 50)
        print("Verification Summary:")
        print("=" * 50)
        for split_name, counts in results.items():
            print(f"{split_name}: {counts['valid']}/{counts['total']} valid files")
