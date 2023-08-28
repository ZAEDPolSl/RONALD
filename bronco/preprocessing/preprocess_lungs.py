import numpy as np
import SimpleITK as sitk
from skimage.measure import label
from skimage.morphology import convex_hull_image
from bronco.utils_bronchi import erosion_by_slice


def mediastinum_segmentation(sitk_segmentation, sitk_image=None):
    min = sitk.MinimumMaximumImageFilter()
    min.Execute(sitk_segmentation)
    min_seg = min.GetMinimum()
    # create mask
    sitk_segmentation = (sitk_segmentation > min_seg)
    # closing
    closing = sitk.BinaryMorphologicalClosingImageFilter()
    sitk_segmentation = closing.Execute(sitk_segmentation)
    # convex hull
    lung_region = sitk.GetArrayFromImage(sitk_segmentation)
    for axial in range(lung_region.shape[0]):
        if np.sum(lung_region[axial]) != 0:
            lung_region[axial] = convex_hull_image(lung_region[axial])
    sitk_lung_region = sitk.GetImageFromArray(lung_region)
    sitk_lung_region.CopyInformation(sitk_segmentation)
    sitk_segmentation.CopyInformation(sitk_segmentation)
    # opening
    sitk_segmentation = (sitk_lung_region - sitk_segmentation)
    opening = sitk.BinaryMorphologicalOpeningImageFilter()
    opening.SetKernelRadius(3)
    sitk_segmentation = opening.Execute(sitk_segmentation)
    # processing for view
    if sitk_image is not None:
        sitk_mediastinum = sitk.Mask(sitk_image, sitk_segmentation, outsideValue=-1024)
    else:
        sitk_mediastinum = sitk_segmentation
    return sitk_mediastinum


def get_lungs_with_mediastinum(sitk_lungs):
    lung_region = sitk.GetArrayFromImage(sitk_lungs)
    for axial in range(lung_region.shape[0]):
        if np.sum(lung_region[axial]) != 0:
            lung_region[axial] = convex_hull_image(lung_region[axial])
    sitk_lung_region = sitk.GetImageFromArray(lung_region)
    sitk_lung_region.CopyInformation(sitk_lungs)
    return sitk_lung_region


def get_connected_components(image):
    labelled_image = label(image)
    uniq, counts = np.unique(labelled_image, return_counts=True)

    # Pair ids with their counts in the volume
    labels = list(zip(uniq[1:], counts[1:]))
    labels = sorted(labels, key=lambda x: x[1])
    return labelled_image, labels


def preprocess_lungs(sitk_image, sitk_lungs, retain_main_bronchi=True):
    # binarize mask
    print("Preparing data...")
    lungs = sitk.GetArrayFromImage(sitk_lungs)
    if np.min(lungs) != 0 or np.max(lungs) != 1:
        _min, _max = np.min(lungs), np.max(lungs)
        lungs[lungs > _min] = 1
        lungs[lungs == _min] = 0
    _sitk_lungs = sitk.GetImageFromArray(lungs)
    _sitk_lungs.CopyInformation(sitk_lungs)

    if retain_main_bronchi:
        print("Retaining main bronchi in the lung mask...")
        # segment bronchi in mediastinum
        sitk_mediastinum = mediastinum_segmentation(_sitk_lungs)
        sitk_mediastinum_view = sitk.Mask(sitk_image, sitk_mediastinum)
        sitk_stats = sitk.StatisticsImageFilter()
        sitk_stats.Execute(sitk_mediastinum_view)
        _min = sitk_stats.GetMinimum()
        seed_range = (_min, _min // 3)
        sitk_bronchi = sitk.BinaryThreshold(sitk_mediastinum_view, seed_range[0], seed_range[1])

        # clean the bronchi
        sitk_bronchi = erosion_by_slice(sitk_bronchi)

        # get bronchi only
        labelled_bronchi, components = get_connected_components(sitk.GetArrayFromImage(sitk_bronchi))
        largest_component = components[-1]
        labelled_bronchi[labelled_bronchi != largest_component[0]] = 0
        labelled_bronchi[labelled_bronchi == largest_component[0]] = 1
        labelled_bronchi = np.array(labelled_bronchi, dtype=np.uint8)
        sitk_labelled_bronchi = sitk.GetImageFromArray(labelled_bronchi)
        sitk_labelled_bronchi.CopyInformation(sitk_bronchi)

        # enlarge it to prevent erosion of the nearby structures
        _sitk_bronchi = sitk.BinaryDilate(sitk_labelled_bronchi, kernelRadius=(20, 20, 20))
        _sitk_erosion_dummy = sitk.Cast(_sitk_lungs, sitk.sitkUInt8) + sitk.Cast(_sitk_bronchi, sitk.sitkUInt8)
    else:
        _sitk_erosion_dummy = sitk_lungs
    # erode edges of lungs
    print("Preprocessing the lungs area...")
    _sitk_erosion_dummy = erosion_by_slice(_sitk_erosion_dummy)
    _sitk_lungs = _sitk_erosion_dummy * sitk.Cast(_sitk_lungs, sitk.sitkUInt8)

    return _sitk_lungs
