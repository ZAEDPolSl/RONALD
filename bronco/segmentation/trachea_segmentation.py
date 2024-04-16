import numpy as np
import SimpleITK as sitk
from skimage.measure import label
from bronco.utils import erosion_by_slice

from bronco.segmentation.mediastinum_segmentation import mediastinum_segmentation


def trachea_main_bronchus_segmentation(sitk_image, sitk_lungs, return_all=False):
    # segment bronchi in mediastinum
    sitk_mediastinum = mediastinum_segmentation(sitk_lungs)
    sitk_mediastinum_view = sitk.Mask(sitk_image, sitk_mediastinum)
    sitk_stats = sitk.StatisticsImageFilter()
    sitk_stats.Execute(sitk_mediastinum_view)
    _min = sitk_stats.GetMinimum()
    seed_range = (_min, _min // 3)
    sitk_trachea = sitk.BinaryThreshold(sitk_mediastinum_view, seed_range[0], seed_range[1])

    # clean the bronchi
    sitk_trachea = erosion_by_slice(sitk_trachea)

    # get bronchi only
    labelled_trachea, components = get_connected_components(sitk.GetArrayFromImage(sitk_trachea))
    largest_component = components[-1]
    labelled_trachea[labelled_trachea != largest_component[0]] = 0
    labelled_trachea[labelled_trachea == largest_component[0]] = 1
    labelled_trachea = np.array(labelled_trachea, dtype=np.uint8)
    sitk_trachea_main_bronchus = sitk.GetImageFromArray(labelled_trachea)
    sitk_trachea_main_bronchus.CopyInformation(sitk_trachea)
    if return_all:
        return sitk_trachea_main_bronchus, sitk_mediastinum
    else:
        return sitk_trachea_main_bronchus


def get_connected_components(image):
    labelled_image = label(image)
    uniq, counts = np.unique(labelled_image, return_counts=True)

    # Pair ids with their counts in the volume
    labels = list(zip(uniq[1:], counts[1:]))
    labels = sorted(labels, key=lambda x: x[1])
    return labelled_image, labels