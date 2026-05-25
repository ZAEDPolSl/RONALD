import numpy as np
import SimpleITK as sitk

from ._totalsegmentator import get_total_roi_labels, segment_total_roi_subset


LOBE_ROIS = [
    "lung_upper_lobe_left",
    "lung_lower_lobe_left",
    "lung_upper_lobe_right",
    "lung_middle_lobe_right",
    "lung_lower_lobe_right",
]


def lobes_segmentation(sitk_image, config=None):
    sitk_total_lobes = segment_total_roi_subset(sitk_image, LOBE_ROIS, config=config)
    total_labels = get_total_roi_labels(LOBE_ROIS)
    total_lobes = sitk.GetArrayFromImage(sitk_total_lobes)

    label_array = np.zeros(total_lobes.shape, dtype=np.uint8)
    for label_value, roi_name in enumerate(LOBE_ROIS, start=1):
        label_array[total_lobes == total_labels[roi_name]] = label_value

    sitk_lobes = sitk.GetImageFromArray(label_array)
    sitk_lobes.CopyInformation(sitk_image)
    return sitk.Cast(sitk_lobes, sitk.sitkUInt8)
