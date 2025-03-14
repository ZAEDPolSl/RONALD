import SimpleITK as sitk
from skimage.morphology import convex_hull_image


def mediastinum_segmentation(sitk_segmentation, sitk_image=None):
    min = sitk.MinimumMaximumImageFilter()
    min.Execute(sitk_segmentation)
    min_seg = min.GetMinimum()
    # create mask
    sitk_segmentation = sitk_segmentation > min_seg
    # closing
    closing = sitk.BinaryMorphologicalClosingImageFilter()
    sitk_segmentation = closing.Execute(sitk_segmentation)
    # convex hull
    lung_region = sitk.GetArrayFromImage(sitk_segmentation)
    for axial in range(lung_region.shape[0]):
        lung_region[axial] = convex_hull_image(lung_region[axial])
    sitk_lung_region = sitk.GetImageFromArray(lung_region)
    sitk_lung_region.CopyInformation(sitk_segmentation)
    sitk_segmentation.CopyInformation(sitk_segmentation)
    # opening
    sitk_segmentation = sitk_lung_region - sitk_segmentation
    opening = sitk.BinaryMorphologicalOpeningImageFilter()
    opening.SetKernelRadius(3)
    sitk_segmentation = opening.Execute(sitk_segmentation)
    # processing for view
    if sitk_image is not None:
        sitk_mediastinum = sitk.Mask(sitk_image, sitk_segmentation, outsideValue=-1024)
    else:
        sitk_mediastinum = sitk_segmentation
    return sitk_mediastinum
