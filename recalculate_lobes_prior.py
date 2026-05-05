import argparse
from pathlib import Path

from ctools import ImageInstance

from bronco.segmentation import lobes_segmentation


MOLTEST1_STUDIES_PATH = Path("/mnt/pmanas/Wojtek/MIRASLC/Dane surowe/Moltest 1/NRRD")
MOLTEST1_MASKS_PATH = Path("/mnt/pmanas/Ania/phd-data/moltest-1/masks")


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
    ImageInstance().write(
        sitk_lobes,
        str(output_path),
        description="Lobes TotalSegmentator",
        forced_mode="file",
    )
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
    args = parser.parse_args()

    patient_paths = sorted(MOLTEST1_STUDIES_PATH.rglob("*.nrrd"))

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
