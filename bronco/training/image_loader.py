from ctools.ImageInstance import ImageInstance
import SimpleITK as sitk
import numpy as np


def load_image_with_instance(path):
    """
    Loads an image using ImageInstance, returns a numpy array with channel-first format.
    """

    img_inst = ImageInstance(show_exceptions=False)
    img = img_inst.read(path)
    if img is None:
        raise RuntimeError(f"Could not load image: {path}")
    arr = sitk.GetArrayFromImage(img)  # [D, H, W] or [H, W]
    if arr.ndim == 3:
        arr = arr[None, ...]  # [1, D, H, W]
    elif arr.ndim == 2:
        arr = arr[None, ...]  # [1, H, W]
    return arr.astype(np.float32)
