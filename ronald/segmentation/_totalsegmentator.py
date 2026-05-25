import numpy as np
import SimpleITK as sitk

from ctools import nifti_to_sitk, sitk_to_nifti


def get_total_roi_labels(roi_names):
    from totalsegmentator.map_to_binary import class_map

    total_labels = {name: label for label, name in class_map["total"].items()}
    missing = [roi_name for roi_name in roi_names if roi_name not in total_labels]
    if missing:
        raise KeyError(f"Missing TotalSegmentator ROI labels: {missing}")
    return {roi_name: total_labels[roi_name] for roi_name in roi_names}


def segment_total_roi_subset(sitk_image, roi_subset, config=None, gpu_id=None):
    config = {} if config is None else dict(config)

    try:
        from totalsegmentator.python_api import totalsegmentator
    except ImportError as exc:
        raise ImportError(
            "TotalSegmentator is required for segmentation. "
            "Install it with `pip install TotalSegmentator`."
        ) from exc

    device = config.get("device")
    if device is None:
        device = "gpu" if gpu_id is None else f"gpu:{gpu_id}"

    nifti_image = sitk_to_nifti(sitk_image)
    segmentation_nifti = totalsegmentator(
        input=nifti_image,
        output=None,
        task="total",
        roi_subset=list(roi_subset),
        ml=True,
        fast=bool(config.get("fast", False)),
        quiet=bool(config.get("quiet", True)),
        verbose=bool(config.get("verbose", False)),
        skip_saving=True,
        device=device,
    )

    segmentation_sitk = nifti_to_sitk(segmentation_nifti)
    segmentation_sitk.CopyInformation(sitk_image)
    return sitk.Cast(segmentation_sitk, sitk.sitkUInt8)
