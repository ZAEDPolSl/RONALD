import os
from pathlib import Path
import argparse
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
import pandas as pd

from ctools import ImageInstance


datasets = {
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
        "studies_path": "/mnt/pmanas//Ania/phd-data/nlst-usa/raw/manifest-NLST_allCT/",
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


def get_patient_name(filename, dataset):
    if dataset in ["Moltest 1"]:
        patient_name = filename.split("/")[-1].split(".")[0]
    elif dataset in ["Moltest 2", "OPPRP", "NLST USA"]:
        # For directories, just take the last component (patient folder name)
        patient_name = filename.rstrip("/").split("/")[-1]
    elif dataset in ["Luna 2016"]:
        patient_name = filename.split("/")[-1][:-4]
    elif dataset in ["DUKE"]:
        patient_name = filename.split("/")[-1][:-7]
    return patient_name


def create_bronco_final(mask_path):
    """
    Creates bronco_final.nrrd by combining airways, airways_final, and vessels_final.

    Values in bronco_final:
    - 1: ones from airways.nrrd
    - 2: ones from airways_final.nrrd (where airways is 0)
    - 3: ones from vessels_final.nrrd

    Args:
        mask_path: Path to the patient's mask directory
    """
    ii = ImageInstance()

    # Define file paths
    airways_path = os.path.join(mask_path, "airways.nrrd")
    airways_final_path = os.path.join(mask_path, "airways_final.nrrd")
    vessels_final_path = os.path.join(mask_path, "vessels_final.nrrd")
    bronco_final_path = os.path.join(mask_path, "bronco_final.nrrd")

    # Check if bronco_final already exists
    if os.path.exists(bronco_final_path):
        print(f"Skipping {mask_path}: bronco_final.nrrd already exists")
        return "exists"

    # Check if all three files exist
    if not all(
        [
            os.path.exists(airways_path),
            os.path.exists(airways_final_path),
            os.path.exists(vessels_final_path),
        ]
    ):
        missing = []
        if not os.path.exists(airways_path):
            missing.append("airways.nrrd")
        if not os.path.exists(airways_final_path):
            missing.append("airways_final.nrrd")
        if not os.path.exists(vessels_final_path):
            missing.append("vessels_final.nrrd")
        print(f"Skipping {mask_path}: missing {', '.join(missing)}")
        return False

    # Load the three masks
    try:
        airways = ii.read(airways_path)
        if airways is None:
            raise ValueError(f"Failed to read {airways_path}")

        airways_final = ImageInstance().read(airways_final_path)
        if airways_final is None:
            raise ValueError(f"Failed to read {airways_final_path}")

        vessels_final = ImageInstance().read(vessels_final_path)
        if vessels_final is None:
            raise ValueError(f"Failed to read {vessels_final_path}")
    except Exception as e:
        print(f"Error loading masks in {mask_path}: {e}")
        return False

    # Convert to numpy arrays
    airways_array = sitk.GetArrayFromImage(airways)
    airways_final_array = sitk.GetArrayFromImage(airways_final)
    vessels_final_array = sitk.GetArrayFromImage(vessels_final)

    # Create the combined mask
    # Start with zeros
    bronco_final_array = np.zeros_like(airways_array, dtype=np.uint8)

    # Add vessels_final as 3
    bronco_final_array[vessels_final_array > 0] = 3

    # Add airways_final as 2 (where airways is 0)
    bronco_final_array[airways_final_array > 0] = 2

    # Overwrite with airways as 1 (this overwrites airways_final where they overlap)
    bronco_final_array[airways_array > 0] = 1

    # Convert back to SimpleITK image
    bronco_final = sitk.GetImageFromArray(bronco_final_array)
    bronco_final.CopyInformation(airways)

    # Save the result
    ii.write(
        bronco_final, bronco_final_path, description="Bronco Final", forced_mode="file"
    )

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create bronco_final.nrrd from airways and vessels masks"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(datasets.keys()),
        help="Dataset name",
    )
    parser.add_argument(
        "--start", type=int, default=0, help="Index to start processing from"
    )
    args = parser.parse_args()

    dataset = args.dataset
    start_index = args.start
    studies_path = datasets[dataset]["studies_path"]
    masks_path = datasets[dataset]["masks_path"]
    p = Path(studies_path).absolute()

    if dataset in ["Moltest 1"]:
        filenames = sorted([str(path) for path in p.rglob("*.nrrd")])
    elif dataset in ["Moltest 2_400"]:
        df = pd.read_csv(studies_path, sep=";")
        df = df.dropna(subset=["folder_path"])
        filenames = sorted(list(df["folder_path"]))
    elif dataset in ["Moltest 2", "OPPRP"]:
        subdirs = [x for x in p.iterdir() if x.is_dir()]
        filenames = sorted(
            [
                str(patient_dir)
                for subdir in subdirs
                for patient_dir in subdir.iterdir()
                if patient_dir.is_dir()
            ]
        )
    elif dataset in ["Luna 2016"]:
        subdirs = [path for path in p.rglob("*subset*") if path.is_dir()]
        filenames = sorted(
            [str(x) for subdir in subdirs for x in subdir.rglob("*.mhd")]
        )
    elif dataset == "NLST USA":
        files_df = pd.read_csv(studies_path + "metadata.csv")
        filenames = (
            studies_path
            + files_df["File Location"].str.split(pat="/", n=1).str[-1]
            + "/"
        )
    elif dataset == "DUKE":
        subdirs = [path for path in p.rglob("*DLCS_subset*") if path.is_dir()]
        filenames = sorted(
            [str(x) for subdir in subdirs for x in subdir.rglob("*.nii.gz")],
            reverse=True,
        )

    success_count = 0
    skip_count = 0
    exists_count = 0
    error_count = 0

    for i, filename in enumerate(tqdm(filenames)):
        if i >= start_index:
            patient_name = get_patient_name(filename, dataset)
            save_path = os.path.join(masks_path, patient_name)

            try:
                result = create_bronco_final(save_path)
                if result == "exists":
                    exists_count += 1
                elif result:
                    success_count += 1
                else:
                    skip_count += 1
            except Exception as e:
                error_count += 1
                with open(f"{dataset}_bronco_final.log", "a") as f:
                    f.write(f"{patient_name}: {str(e)}\n")
                print(f"Error processing {patient_name}: {e}")

    print(f"\nProcessing complete:")
    print(f"  Successfully created: {success_count}")
    print(f"  Already exists: {exists_count}")
    print(f"  Skipped (missing files): {skip_count}")
    print(f"  Errors: {error_count}")
