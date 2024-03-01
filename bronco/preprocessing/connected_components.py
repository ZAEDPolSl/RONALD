import numpy as np
from tqdm import tqdm
import SimpleITK as sitk


def convex_hull_3d(sitk_mask):
    from skimage.morphology import convex_hull_image
    # closing
    closing = sitk.BinaryMorphologicalClosingImageFilter()
    sitk_segmentation = closing.Execute(sitk_mask)
    # convex hull
    lung_region = sitk.GetArrayFromImage(sitk_segmentation)
    for axial in range(lung_region.shape[0]):
        lung_region[axial] = convex_hull_image(lung_region[axial])
    sitk_lung_region = sitk.GetImageFromArray(lung_region)
    sitk_lung_region.CopyInformation(sitk_mask)
    sitk_segmentation.CopyInformation(sitk_mask)
    return sitk.Cast((sitk_lung_region + sitk_segmentation) > 0, sitk.sitkUInt8)


def find_largest_connected_component(image):
    # Use ConnectedComponent to label connected components
    labeled_image = sitk.ConnectedComponent(image)

    # Use LabelShapeStatisticsImageFilter to get the size of each component
    stats_filter = sitk.LabelShapeStatisticsImageFilter()
    stats_filter.Execute(labeled_image)

    # Find the label with the largest size
    max_label = max(stats_filter.GetLabels(), key=lambda x: stats_filter.GetPhysicalSize(x))

    # Create a binary mask for the largest connected component
    largest_component_mask = sitk.BinaryThreshold(labeled_image, lowerThreshold=max_label, upperThreshold=max_label)

    return largest_component_mask


def find_most_similar_connected_component(target_image, reference_mask):
    reference_array = sitk.GetArrayFromImage(reference_mask)

    target_connected_components = sitk.ConnectedComponent(target_image)

    target_connected_components_mask = target_connected_components * sitk.Cast(reference_mask, sitk.sitkUInt32)

    max_iou = 0
    most_similar_component = None

    list_uniques = np.unique(sitk.GetArrayFromImage(target_connected_components_mask))

    for label in tqdm(list_uniques):
        label = int(label)
        target_component = sitk.BinaryThreshold(target_connected_components, label, label)
        target_component_array = sitk.GetArrayFromImage(target_component)

        iou = calculate_iou(target_component_array, reference_array)

        if iou > max_iou:
            max_iou = iou
            most_similar_component = target_component
    most_similar_component.CopyInformation(target_image)
    return most_similar_component, max_iou


def calculate_iou(segmentation1, segmentation2):
    intersection = np.logical_and(segmentation1, segmentation2)
    union = np.logical_or(segmentation1, segmentation2)
    iou = np.sum(intersection) / np.sum(union)
    return iou


def calculate_dice(segmentation1, segmentation2):
    intersection = np.logical_and(segmentation1, segmentation2)
    dice_coefficient = 2 * np.sum(intersection) / (np.sum(segmentation1) + np.sum(segmentation2))
    return dice_coefficient
