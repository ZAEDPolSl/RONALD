from ctools.ctools.ImageInstance import ImageInstance
import numpy as np
import torch
import SimpleITK as sitk


def load_image_with_instance(path):
    """
    Loads an image using ImageInstance, returns a numpy array.
    """
    img_inst = ImageInstance(show_exceptions=False)
    img = img_inst.read(path)
    if img is None:
        raise RuntimeError(f"Could not load image: {path}")
    arr = np.array(torch.from_numpy(sitk.GetArrayFromImage(img)))
    return arr
