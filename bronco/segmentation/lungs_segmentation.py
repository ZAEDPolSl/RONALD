import numpy as np
import SimpleITK as sitk
from lungmask import mask


def lungs_segmentation(sitk_image, config=None, binary=True, gpu_id=None):
    if config is not None:
        model = config["model_lungs_segmentation"]
    else:
        model = mask.get_model('R231')
    lungs = mask.apply(sitk_image, model, batch_size=5)
    if binary:
        _min = np.min(lungs)
        lungs[lungs > _min] = 1
        lungs[lungs == _min] = 0
        lungs = np.array(lungs, np.uint8)
    sitk_lungs = sitk.GetImageFromArray(lungs)
    sitk_lungs.CopyInformation(sitk_image)
    return sitk_lungs
