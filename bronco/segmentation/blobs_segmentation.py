import itk

from ctools import itk_to_sitk, sitk_to_itk


def blobs_segmentation(sitk_image, sitk_lungs, sigma_min=1, sigma_max=5, steps=8):
    """
    Perform blob segmentation on a 3D image using ITK.

    Parameters:
        sitk_image: SimpleITK image - Input 3D image.
        sitk_lungs: SimpleITK image - Lung mask.

    Returns:
        sitk_blob_enhanced: SimpleITK image - Blob-enhanced image.
    """
    # sitk.sitkFloat64
    Dimension = 3
    PixelType = itk.D

    itk_image = sitk_to_itk(sitk_image, PixelType)
    itk_lungs = sitk_to_itk(sitk_lungs, PixelType)
    direction = sitk_image.GetDirection()

    multiply_filter = itk.MultiplyImageFilter[
        itk.Image[PixelType, Dimension],
        itk.Image[PixelType, Dimension],
        itk.Image[PixelType, Dimension],
    ].New()
    multiply_filter.SetInput1(itk_image)
    multiply_filter.SetInput2(itk_lungs)
    multiply_filter.Update()
    masked_image = multiply_filter.GetOutput()

    ImageType = itk.Image[PixelType, Dimension]
    HessianPixelType = itk.SymmetricSecondRankTensor[PixelType, Dimension]
    HessianImageType = itk.Image[HessianPixelType, Dimension]

    # Set up the Hessian to objectness measure filter for blobs (object dimension = 0)
    objectness_filter = itk.HessianToObjectnessMeasureImageFilter[
        HessianImageType, ImageType
    ].New()
    objectness_filter.SetObjectDimension(0)  # 0 for blobs
    objectness_filter.SetBrightObject(
        True
    )  # True if blobs are bright on dark background
    objectness_filter.SetAlpha(0.2)  # Sensitivity to blob shape deviation
    objectness_filter.SetBeta(
        1.0
    )  # Sensitivity to plate-like structures (not critical for blobs)
    objectness_filter.SetGamma(20.0)  # Sensitivity to background noise
    objectness_filter.SetScaleObjectnessMeasure(
        False
    )  # Do not scale by eigenvalue magnitude

    # Set up the multi-scale Hessian-based measure filter
    multi_scale_filter = itk.MultiScaleHessianBasedMeasureImageFilter[
        ImageType, HessianImageType, ImageType
    ].New()
    multi_scale_filter.SetInput(masked_image)
    multi_scale_filter.SetHessianToMeasureFilter(objectness_filter)
    multi_scale_filter.SetSigmaMinimum(sigma_min)  # Minimum scale
    multi_scale_filter.SetSigmaMaximum(sigma_max)  # Maximum scale
    multi_scale_filter.SetNumberOfSigmaSteps(steps)  # Number of scales
    multi_scale_filter.SetSigmaStepMethodToLogarithmic()

    # Run the filter
    multi_scale_filter.Update()
    enhanced_blobs = multi_scale_filter.GetOutput()

    multiply_filter = itk.MultiplyImageFilter[
        itk.Image[PixelType, Dimension],
        itk.Image[PixelType, Dimension],
        itk.Image[PixelType, Dimension],
    ].New()
    multiply_filter.SetInput1(enhanced_blobs)
    multiply_filter.SetInput2(itk_lungs)
    multiply_filter.Update()
    enhanced_blobs = multiply_filter.GetOutput()

    # Rescale intensity for saving as 8-bit image
    OutputPixelType = itk.UC
    OutputImageType = itk.Image[OutputPixelType, Dimension]
    rescale_filter = itk.RescaleIntensityImageFilter[ImageType, OutputImageType].New()
    rescale_filter.SetInput(enhanced_blobs)
    rescale_filter.SetOutputMinimum(0)
    rescale_filter.SetOutputMaximum(255)
    rescale_filter.Update()
    itk_blobs = rescale_filter.GetOutput()

    ThresholdFilterType = itk.BinaryThresholdImageFilter[
        type(itk_blobs), itk.Image[itk.UC, 3]
    ]
    threshold_filter = ThresholdFilterType.New()
    threshold_filter.SetInput(itk_blobs)
    threshold_filter.SetLowerThreshold(126)  # strictly greater than 125
    threshold_filter.SetUpperThreshold(255)
    threshold_filter.SetInsideValue(1)
    threshold_filter.SetOutsideValue(0)
    threshold_filter.Update()
    itk_blobs = threshold_filter.GetOutput()

    sitk_blob_enhanced = itk_to_sitk(itk_blobs, 1, direction)
    return sitk_blob_enhanced
