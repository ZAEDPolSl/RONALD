import itk
import SimpleITK as sitk


def sitk_to_itk(sitk_image):
    """Convert SimpleITK image to ITK image."""
    array = sitk.GetArrayFromImage(sitk_image)
    itk_image = itk.image_view_from_array(array)
    itk_image.SetSpacing(sitk_image.GetSpacing())
    itk_image.SetOrigin(sitk_image.GetOrigin())
    itk_image.SetDirection(sitk_image.GetDirection())
    return itk_image


def itk_to_sitk(itk_image):
    """Convert ITK image (including vector images) to SimpleITK."""
    # Get numpy array from itk image
    array = itk.GetArrayViewFromImage(itk_image)
    # ITK vector images have shape: [depth, height, width, components]
    # For 3D images, shape is (z,y,x,components)
    # SimpleITK expects components as the last dimension too
    sitk_image = sitk.GetImageFromArray(array, isVector=True)
    sitk_image.SetSpacing(itk_image.GetSpacing())
    sitk_image.SetOrigin(itk_image.GetOrigin())
    sitk_image.SetDirection(itk_image.GetDirection())
    return sitk_image


def vessel_segmentation(sitk_image, sitk_lungs):
    stats = sitk.StatisticsImageFilter()
    stats.Execute(sitk_image)
    _min = stats.GetMinimum()
    # get lungs image
    sitk_image = sitk.Mask(sitk_image, sitk_lungs, outsideValue=_min - 1)

    sigma = 1.0
    alpha1 = 0.5
    alpha2 = 2.0

    itk_image = sitk_to_itk(sitk_image)

    # Compute Hessian with ITK
    hessian_filter = itk.HessianRecursiveGaussianImageFilter.New(itk_image)
    hessian_filter.SetSigma(sigma)
    hessian_filter.Update()
    hessian_itk = hessian_filter.GetOutput()

    # Convert Hessian ITK image back to SimpleITK
    hessian_sitk = itk_to_sitk(hessian_itk)

    vesselness_filter = sitk.HessianToObjectnessMeasureImageFilter()
    vesselness_filter.SetObjectDimension(1)
    vesselness_filter.SetBrightObject(True)
    vesselness_filter.SetAlpha(alpha1)
    vesselness_filter.SetBeta(alpha2)
    vesselness_filter.SetGamma(5.0)
    vesselness_image = vesselness_filter.Execute(hessian_sitk)
    masked_vesselness = sitk.Mask(vesselness_image, sitk_lungs)
    # Threshold the vesselness image to create a binary mask
    threshold = 0.3
    binary_vesselness = sitk.BinaryThreshold(masked_vesselness, threshold, 255)
    return binary_vesselness
