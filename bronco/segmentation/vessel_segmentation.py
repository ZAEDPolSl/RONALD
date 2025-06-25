import itk
import numpy as np
import SimpleITK as sitk
from skimage.measure import regionprops


from bronco.external.sitk2itk import (
    ConvertItkImageToSimpleItkImage,
    ConvertSimpleItkImageToItkImage,
)
from bronco.segmentation.blobs_segmentation import blobs_segmentation


def vesselness_filter(sitk_image, sitk_lungs):
    sigma = 1.0
    alpha1 = 0.5
    alpha2 = 2.0

    float_sitk = sitk.Cast(sitk_image, sitk.sitkFloat32)
    itk_image = ConvertSimpleItkImageToItkImage(float_sitk, itk.F)
    itk_lungs = ConvertSimpleItkImageToItkImage(sitk_lungs, itk.F)

    itk_image = itk.cast_image_filter(
        itk_image, ttype=[type(itk_image), itk.Image[itk.F, 3]]
    )
    itk_lungs = itk.cast_image_filter(
        itk_lungs, ttype=[type(itk_lungs), itk.Image[itk.F, 3]]
    )

    multiply_filter = itk.MultiplyImageFilter[
        itk.Image[itk.F, 3], itk.Image[itk.F, 3], itk.Image[itk.F, 3]
    ].New()
    multiply_filter.SetInput1(itk_image)
    multiply_filter.SetInput2(itk_lungs)
    multiply_filter.Update()
    masked_image = multiply_filter.GetOutput()
    # Convert to float for further processing if needed
    input_image_float = itk.cast_image_filter(
        masked_image, ttype=[type(masked_image), itk.Image[itk.F, 3]]
    )
    # Compute Hessian with ITK
    hessian_image = itk.hessian_recursive_gaussian_image_filter(
        input_image_float, sigma=sigma
    )
    vesselness_filter = itk.Hessian3DToVesselnessMeasureImageFilter[
        itk.ctype("float")
    ].New()
    vesselness_filter.SetInput(hessian_image)
    vesselness_filter.SetAlpha1(alpha1)
    vesselness_filter.SetAlpha2(alpha2)
    vesselness_filter.Update()
    itk_output = vesselness_filter.GetOutput()

    multiply_filter = itk.MultiplyImageFilter[
        itk.Image[itk.F, 3], itk.Image[itk.F, 3], itk.Image[itk.F, 3]
    ].New()
    multiply_filter.SetInput1(itk_output)
    multiply_filter.SetInput2(itk_lungs)
    multiply_filter.Update()
    output_image = multiply_filter.GetOutput()

    threshold_filter = itk.BinaryThresholdImageFilter[
        itk.Image[itk.F, 3], itk.Image[itk.UC, 3]
    ].New()
    threshold_filter.SetInput(output_image)
    threshold_filter.SetLowerThreshold(30)
    threshold_filter.SetOutsideValue(0)
    threshold_filter.SetInsideValue(255)
    threshold_filter.Update()
    output_image = threshold_filter.GetOutput()
    # Convert Hessian ITK image back to SimpleITK
    direction = float_sitk.GetDirection()
    sitk_vessels = ConvertItkImageToSimpleItkImage(output_image, 8, direction)
    return sitk_vessels


def vessel_segmentation(
    sitk_image, sitk_lungs, sitk_lobes=None, sitk_mediastinum=None, binary=True
):
    sitk_vessels = vesselness_filter(sitk_image, sitk_lungs)
    if sitk_mediastinum is None:
        from bronco.segmentation import mediastinum_segmentation

        sitk_mediastinum = mediastinum_segmentation(sitk_lungs)
    if sitk_lobes is None:
        from bronco.segmentation import lobe_segmentation

        sitk_lobes = lobe_segmentation(sitk_image)

    lungs = sitk.GetArrayFromImage(sitk_lungs)
    vessels = sitk.GetArrayFromImage(sitk_vessels)
    mediastinum = sitk.GetArrayFromImage(sitk_mediastinum)
    lobes = sitk.GetArrayFromImage(sitk_lobes)

    blood_system = np.logical_or(mediastinum, vessels).astype(int)
    blood_regions = regionprops(blood_system)
    sorted_regions = sorted(blood_regions, key=lambda x: x.area)
    first_region_mask = np.zeros_like(blood_system, dtype=int)
    first_region_mask[tuple(sorted_regions[0].coords.T)] = 1
    first_region_mask[mediastinum > 1] = 0
    first_region_mask[first_region_mask > 0] = lobes[first_region_mask > 0]
    first_region_mask[lungs == 0] = 0

    if binary:
        first_region_mask[first_region_mask > 0] = 1

    vessels_connected = sitk.GetImageFromArray(first_region_mask)
    vessels_connected.CopyInformation(sitk_vessels)
    vessels_connected = sitk.Cast(vessels_connected, sitk.sitkUInt8)

    blobs = blobs_segmentation(vessels_connected, sitk_lungs)
    vessels_final = sitk.Subtract(vessels_connected, blobs)
    vessels_final = sitk.BinaryThreshold(vessels_final,
                                         lowerThreshold=1,
                                         upperThreshold=255,
                                         insideValue=255,
                                         outsideValue=0)

    return vessels_final
