import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk


LOBE_ROIS = [
    "lung_upper_lobe_left",
    "lung_lower_lobe_left",
    "lung_upper_lobe_right",
    "lung_middle_lobe_right",
    "lung_lower_lobe_right",
]


def _compose_lobe_labels(output_dir, reference_image):
    label_array = np.zeros(
        sitk.GetArrayFromImage(reference_image).shape,
        dtype=np.uint8,
    )

    for label_value, roi_name in enumerate(LOBE_ROIS, start=1):
        roi_path = Path(output_dir) / f"{roi_name}.nii.gz"
        if not roi_path.exists():
            raise FileNotFoundError(
                f"TotalSegmentator did not produce the expected mask: {roi_path}"
            )
        roi_image = sitk.ReadImage(str(roi_path))
        roi_array = sitk.GetArrayFromImage(roi_image) > 0
        label_array[roi_array] = label_value

    sitk_lobes = sitk.GetImageFromArray(label_array)
    sitk_lobes.CopyInformation(reference_image)
    return sitk.Cast(sitk_lobes, sitk.sitkUInt8)


def lobes_segmentation(sitk_image, config=None):
    config = {} if config is None else dict(config)

    try:
        from totalsegmentator.python_api import totalsegmentator
    except ImportError as exc:
        raise ImportError(
            "TotalSegmentator is required for lobes_segmentation(). "
            "Install it with `pip install TotalSegmentator`."
        ) from exc

    device = config.get("device", "gpu")
    fast = bool(config.get("fast", False))
    quiet = bool(config.get("quiet", True))
    verbose = bool(config.get("verbose", False))

    with tempfile.TemporaryDirectory(prefix="totalseg_lobes_") as temp_dir:
        temp_dir_path = Path(temp_dir)
        input_path = temp_dir_path / "input.nii.gz"
        output_dir = temp_dir_path / "lobes"

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

        return _compose_lobe_labels(output_dir, sitk_image)
