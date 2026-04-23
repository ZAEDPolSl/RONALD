import os

# os.environ["OPENBLAS_NUM_THREADS"] = "16"
# os.environ["OMP_NUM_THREADS"] = "16"
# os.environ["MKL_NUM_THREADS"] = "16"
# os.environ["NUMEXPR_NUM_THREADS"] = "16"

import pickle
from pathlib import Path

import SimpleITK as sitk
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import argparse

from ctools import ImageInstance
from bronco.segmentation import (
    lungs_segmentation,
    lobes_segmentation,
    mediastinum_segmentation,
    vessel_segmentation,
)

# from bronco.modelling.smooth_tree import smooth_tree


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
        "studies_path": "/Volumes/pma/Ania/phd-data/nlst-usa/raw/manifest-NLST_allCT/",
        "masks_path": "/Volumes/pma/Ania/phd-data/nlst-usa/masks",
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


def pipeline(patient_path, airway_path, cache_path):
    ii = ImageInstance()
    try:
        sitk_image = ii.read(patient_path)
    except Exception as e:
        print(f"Error reading {patient_path}: {e}")
        return

    # Ensure cache_path exists
    os.makedirs(cache_path, exist_ok=True)

    path_lungs = os.path.join(airway_path, "lungs.nrrd")
    path_mediastinum = os.path.join(cache_path, "mediastinum.nrrd")
    path_lobes = os.path.join(cache_path, "lobes.nrrd")

    if not os.path.exists(path_lungs):
        sitk_lungs = lungs_segmentation(sitk_image)
        sitk_lungs.CopyInformation(sitk_image)
        ii.write(sitk_lungs, path_lungs, description="Lungs", forced_mode="file")
    else:
        sitk_lungs = ImageInstance().read(path_lungs)
        sitk_lungs.CopyInformation(sitk_image)
    vessel_path = os.path.join(cache_path, "vessels_final.nrrd")
    # mediastinum
    if not os.path.exists(path_mediastinum):
        sitk_mediastinum = mediastinum_segmentation(sitk_lungs)
        ii.write(
            sitk_mediastinum,
            path_mediastinum,
            description="Mediastinum",
            forced_mode="file",
        )
    else:
        sitk_mediastinum = ImageInstance().read(path_mediastinum)
        sitk_mediastinum.CopyInformation(sitk_image)

    # lobes
    if not os.path.exists(path_lobes):
        sitk_lobes = lobes_segmentation(sitk_image)
        ii.write(sitk_lobes, path_lobes, description="Lobes", forced_mode="file")
    else:
        sitk_lobes = ImageInstance().read(path_lobes)
        sitk_lobes.CopyInformation(sitk_image)

    # sitk_vessels = vessel_segmentation(
    #     sitk_image, sitk_lungs, sitk_lobes, sitk_mediastinum
    # )
    # ii.write(sitk_vessels, vessel_path, description="Vessels", forced_mode="file")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bronco pipeline runner")
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

    for i, filename in enumerate(tqdm(filenames)):
        if i >= start_index:
            patient_name = get_patient_name(filename, dataset)
            save_path = os.path.join(masks_path, patient_name)
            try:
                if dataset in ["Moltest 2", "Moltest 2_400"]:
                    # ImageInstance will handle finding the correct series with most slices
                    patient_series = str(Path(filename) / "DICOM")
                    pipeline(patient_series, save_path, save_path)
                else:
                    pipeline(filename, save_path, save_path)
            except Exception as e:
                with open(f"{dataset}_bronco.log", "a") as f:
                    f.write(f"{patient_name}: {str(e)}\n")
                print(f"Error processing {patient_name}: {e}")
