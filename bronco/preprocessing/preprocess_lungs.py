import numpy as np
from time import time
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


def shortest_distance_to_point(array, point):
    """
    Calculate the shortest distance from a 3D NumPy array to a point in 3D space.
    """
    # Create a grid of coordinates matching the shape of the array
    '''x_coords, y_coords, z_coords = np.meshgrid(
        np.arange(array.shape[0]),
        np.arange(array.shape[1]),
        np.arange(array.shape[2])
    )'''
    # Calculate the Euclidean distance from each point in the array to the target point
    # distances = np.sqrt((x_coords - point[0])**2 + (y_coords - point[1])**2 + (z_coords - point[2])**2)
    array_cords = np.argwhere(array == 1)
    distances = np.linalg.norm(array_cords - point)
    # Find the minimum distance
    shortest_distance = np.min(distances)
    return shortest_distance


def get_connected_components(image):
    labelled_image = label(image)
    uniq, counts = np.unique(labelled_image, return_counts=True)

    # Pair ids with their counts in the volume
    labels = list(zip(uniq[1:], counts[1:]))
    labels = sorted(labels, key=lambda x: x[1])
    return labelled_image, labels


def min_max_normalize(data):
    min_val = min(data) + 1e-13
    max_val = max(data)

    normalized_data = [(x - min_val) / (max_val - min_val) for x in data]

    return normalized_data


def get_larges_central_sorted_connected_components(image):
    labelled_image = label(image)
    uniq, count = np.unique(labelled_image, return_counts=True)
    uniq, count = uniq[1:], count[1:]

    labels = list(zip(uniq, count))
    labels = sorted(labels, key=lambda x: x[1])
    uniq, count = zip(*labels)

    # Pair ids with their counts in the volume
    labels = []
    counts = []
    distances = []
    point_center = np.array(image.shape) // 2
    # point_center[0] = image.shape[0]
    for l, c in list(zip(uniq[-5:], count[-5:])):
        im = np.array(np.equal(labelled_image, l), dtype=int)
        dist = shortest_distance_to_point(im, point_center)
        labels.append(l)
        counts.append(c)
        distances.append(dist)
    counts = min_max_normalize(counts)
    distances = min_max_normalize(distances)
    # reverse distances, in counts 1 == good in distances 0 == good -> now in rev_distance 1 == good
    rev_distances = [np.abs(1 - d) for d in distances]
    # importance is a mean of the pixel counts and reverse distance from the center of the closest pixel
    importance = [(2 * c + rd) / (1 + 2) for c, rd in zip(counts, rev_distances)]
    labels = list(zip(labels, importance))
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
        seed_range = (-1024, -960)
        sitk_bronchi = sitk.BinaryThreshold(sitk_mediastinum_view, seed_range[0], seed_range[1])

        # clean the bronchi
        # sitk_bronchi = erosion_by_slice(sitk_bronchi)

        # get bronchi only
        labelled_bronchi, components = get_larges_central_sorted_connected_components(sitk.GetArrayFromImage(sitk_bronchi))
        largest_component = components[-1]
        labelled_bronchi[labelled_bronchi != largest_component[0]] = 0
        labelled_bronchi[labelled_bronchi == largest_component[0]] = 1
        labelled_bronchi = np.array(labelled_bronchi, dtype=np.uint8)
        sitk_labelled_bronchi = sitk.GetImageFromArray(labelled_bronchi)
        sitk_labelled_bronchi.CopyInformation(sitk_bronchi)

        # enlarge it to prevent erosion of the nearby structures
        sitk_labelled_bronchi = sitk.BinaryMorphologicalClosing(sitk_labelled_bronchi)
        _sitk_bronchi = sitk.BinaryDilate(sitk_labelled_bronchi, kernelRadius=(20, 20, 20))
        _sitk_erosion_dummy = sitk.Cast(_sitk_lungs, sitk.sitkUInt8) + sitk.Cast(_sitk_bronchi, sitk.sitkUInt8)
    else:
        _sitk_erosion_dummy = sitk_lungs
        sitk_labelled_bronchi = None
        sitk_mediastinum = None
    # erode edges of lungs
    print("Preprocessing the lungs area...")
    _sitk_erosion_dummy = erosion_by_slice(_sitk_erosion_dummy)
    _sitk_lungs = _sitk_erosion_dummy * sitk.Cast(_sitk_lungs, sitk.sitkUInt8)

    return _sitk_lungs, sitk_labelled_bronchi, sitk_mediastinum
