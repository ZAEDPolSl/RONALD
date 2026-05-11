import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from .lobes_segmentation import LOBE_ROIS


LEFT_LOBE_ROIS = {
    "lung_upper_lobe_left",
    "lung_lower_lobe_left",
}


def _compose_lung_mask(output_dir, reference_image, binary):
    lung_array = np.zeros(
        sitk.GetArrayFromImage(reference_image).shape,
        dtype=np.uint8,
    )

    for roi_name in LOBE_ROIS:
        roi_path = Path(output_dir) / f"{roi_name}.nii.gz"
        if not roi_path.exists():
            raise FileNotFoundError(
                f"TotalSegmentator did not produce the expected mask: {roi_path}"
            )

        roi_image = sitk.ReadImage(str(roi_path))
        roi_array = sitk.GetArrayFromImage(roi_image) > 0
        if binary:
            lung_array[roi_array] = 1
        else:
            lung_array[roi_array] = 1 if roi_name in LEFT_LOBE_ROIS else 2

    sitk_lungs = sitk.GetImageFromArray(lung_array)
    sitk_lungs.CopyInformation(reference_image)
    return sitk.Cast(sitk_lungs, sitk.sitkUInt8)


def lungs_segmentation(sitk_image, config=None, binary=True, gpu_id=None):
    config = {} if config is None else dict(config)

    try:
        from totalsegmentator.python_api import totalsegmentator
    except ImportError as exc:
        raise ImportError(
            "TotalSegmentator is required for lungs_segmentation(). "
            "Install it with `pip install TotalSegmentator`."
        ) from exc

    device = config.get("device")
    if device is None:
        device = "gpu" if gpu_id is None else f"gpu:{gpu_id}"

    fast = bool(config.get("fast", False))
    quiet = bool(config.get("quiet", True))
    verbose = bool(config.get("verbose", False))

    with tempfile.TemporaryDirectory(prefix="totalseg_lungs_") as temp_dir:
        temp_dir_path = Path(temp_dir)
        input_path = temp_dir_path / "input.nii.gz"
        output_dir = temp_dir_path / "lungs"

        sitk.WriteImage(sitk_image, str(input_path), True)
        output_dir.mkdir(parents=True, exist_ok=True)

        totalsegmentator(
            input=input_path,
            output=output_dir,
            task="total",
            roi_subset=LOBE_ROIS,
            ml=False,
            fast=fast,
            quiet=quiet,
            verbose=verbose,
            device=device,
        )

        return _compose_lung_mask(output_dir, sitk_image, binary)
