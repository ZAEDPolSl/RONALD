import numpy as np
import SimpleITK as sitk

from ._totalsegmentator import get_total_roi_labels, segment_total_roi_subset
from .lobes_segmentation import LOBE_ROIS


LEFT_LOBE_ROIS = {
    "lung_upper_lobe_left",
    "lung_lower_lobe_left",
}


def lungs_segmentation(sitk_image, config=None, binary=True, gpu_id=None):
    sitk_total_lobes = segment_total_roi_subset(
        sitk_image,
        LOBE_ROIS,
        config=config,
        gpu_id=gpu_id,
    )
    total_lobes = sitk.GetArrayFromImage(sitk_total_lobes)

    if binary:
        lungs = (total_lobes > 0).astype(np.uint8)
    else:
        lungs = np.zeros(total_lobes.shape, dtype=np.uint8)
        total_labels = get_total_roi_labels(LOBE_ROIS)
        left_labels = {total_labels[roi_name] for roi_name in LEFT_LOBE_ROIS}
        lungs[np.isin(total_lobes, list(left_labels))] = 1
        lungs[(total_lobes > 0) & ~np.isin(total_lobes, list(left_labels))] = 2

    sitk_lungs = sitk.GetImageFromArray(lungs)
    sitk_lungs.CopyInformation(sitk_image)
    return sitk.Cast(sitk_lungs, sitk.sitkUInt8)
