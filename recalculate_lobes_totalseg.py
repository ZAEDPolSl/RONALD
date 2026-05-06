import argparse
from pathlib import Path
import re
import unicodedata

import pandas as pd
import SimpleITK as sitk
from ctools import ImageInstance

from bronco.segmentation import lobes_segmentation


def resolve_first_existing_path(*candidates: str) -> Path:
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    return Path(candidates[0])


MOLTEST1_STUDIES_PATH = resolve_first_existing_path(
    "/Volumes/pma/Wojtek/MIRASLC/Dane surowe/Moltest 1/NRRD",
    "/mnt/pmanas/Wojtek/MIRASLC/Dane surowe/Moltest 1/NRRD",
)
MOLTEST1_MASKS_PATH = resolve_first_existing_path(
    "/Volumes/pma/Ania/phd-data/moltest-1/masks",
    "/mnt/pmanas/Ania/phd-data/moltest-1/masks",
)
DEFAULT_PATIENTS_CSV = resolve_first_existing_path(
    "/Volumes/pma/Ania/phd-data/moltest-1/pyradiomics_results_MOLTEST 1.csv",
    "/mnt/pmanas/Ania/phd-data/moltest-1/pyradiomics_results_MOLTEST 1.csv",
)


def normalize_patient_id(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = value.lower().strip()
    return re.sub(r"[^a-z0-9]+", "", value)


def get_patient_column(columns) -> str | None:
    preferred_columns = [
        "Patient",
        "Patient Name",
        "patient",
        "patient_name",
        "patient_id",
        "PatientID",
    ]
    for column in preferred_columns:
        if column in columns:
            return column

    normalized_columns = {
        normalize_patient_id(column): column for column in columns if not str(column).startswith("Unnamed:")
    }
    for candidate in ("patient", "patientname", "patientid"):
        if candidate in normalized_columns:
            return normalized_columns[candidate]
    return None


def load_allowed_patients(csv_path: Path) -> set[str]:
    df = pd.read_csv(csv_path)
    patient_column = get_patient_column(df.columns)
    if patient_column is None:
        raise ValueError(
            f"Could not find a patient column in {csv_path}. "
            "Expected one of: Patient, Patient Name, patient, patient_name, patient_id, PatientID."
        )

    allowed_patients = {
        normalize_patient_id(value)
        for value in df[patient_column].dropna().astype(str)
        if str(value).strip()
    }
    if not allowed_patients:
        raise ValueError(f"No patient IDs found in column '{patient_column}' of {csv_path}")
    return allowed_patients


def recalculate_lobes(patient_path, masks_root, overwrite, device, fast):
    patient_name = patient_path.stem
    save_dir = masks_root / patient_name
    output_path = save_dir / "lobes_totalseg.nrrd"

    if output_path.exists() and not overwrite:
        print(f"Skipping {patient_name}: lobes_totalseg.nrrd already exists")
        return "exists"

    save_dir.mkdir(parents=True, exist_ok=True)

    sitk_image = ImageInstance().read(str(patient_path))
    sitk_lobes = lobes_segmentation(
        sitk_image,
        config={
            "device": device,
            "fast": fast,
        },
    )
    sitk.WriteImage(sitk_lobes, str(output_path))
    return "success"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recalculate Moltest 1 lobe masks with TotalSegmentator"
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", type=str, default="gpu")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument(
        "--all-patients",
        action="store_true",
        help="Process all Moltest 1 patients and ignore the radiomics CSV filter.",
    )
    parser.add_argument(
        "--patients-csv",
        type=Path,
        default=DEFAULT_PATIENTS_CSV,
        help=(
            "CSV used to restrict processing to listed patients. "
            f"Defaults to {DEFAULT_PATIENTS_CSV}"
        ),
    )
    args = parser.parse_args()

    patient_paths = sorted(MOLTEST1_STUDIES_PATH.rglob("*.nrrd"))
    if not args.all_patients and args.patients_csv is not None:
        allowed_patients = load_allowed_patients(args.patients_csv)
        patient_paths = [
            patient_path
            for patient_path in patient_paths
            if normalize_patient_id(patient_path.stem) in allowed_patients
        ]
        print(
            f"Restricted to {len(patient_paths)} patients from CSV: {args.patients_csv}"
        )
    elif args.all_patients:
        print("Processing all Moltest 1 patients without CSV filtering")

    success_count = 0
    exists_count = 0
    error_count = 0

    for index, patient_path in enumerate(patient_paths):
        if index < args.start:
            continue
        if args.limit is not None and (index - args.start) >= args.limit:
            break

        try:
            result = recalculate_lobes(
                patient_path,
                MOLTEST1_MASKS_PATH,
                args.overwrite,
                args.device,
                args.fast,
            )
            if result == "exists":
                exists_count += 1
            else:
                success_count += 1
        except Exception as exc:
            error_count += 1
            with open("Moltest 1_lobes_totalseg.log", "a") as log_file:
                log_file.write(f"{patient_path.stem}: {str(exc)}\n")
            print(f"Error processing {patient_path.stem}: {exc}")

    print("\nProcessing complete:")
    print(f"  Successfully created: {success_count}")
    print(f"  Already exists: {exists_count}")
    print(f"  Errors: {error_count}")
