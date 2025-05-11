import itk
import numpy as np
import SimpleITK as sitk

from bronco.external.sitk2itk import (
    ConvertItkImageToSimpleItkImage,
    ConvertSimpleItkImageToItkImage,
)


def vessel_segmentation(sitk_image, sitk_lungs):
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
    output_image = threshold_filter.GetOutput()
    # Convert Hessian ITK image back to SimpleITK
    direction = float_sitk.GetDirection()
    sitk_vessels = ConvertItkImageToSimpleItkImage(output_image, 8, direction)
    return sitk_vessels
