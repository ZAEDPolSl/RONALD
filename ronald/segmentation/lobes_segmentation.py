import numpy as np
import SimpleITK as sitk
from lungmask import mask


def lobes_segmentation(sitk_image, config=None):
    if config is not None:
        model = config["model_lobes_segmentation"]
    else:
        model = mask.get_model("LTRCLobes")
    lobes = mask.apply(sitk_image, model, batch_size=5)
    sitk_lobes = sitk.GetImageFromArray(lobes)
    sitk_lobes.CopyInformation(sitk_image)
    return sitk_lobes
