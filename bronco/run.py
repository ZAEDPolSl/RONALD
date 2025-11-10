import argparse
import SimpleITK as sitk
from bronco.utils import display
from bronco.io_local import ImageInstance
from bronco.segmentation import airways_segmentation, lungs_segmentation


def __handle_input(data):
    ii = None
    if type(data) == str:
        ii = ImageInstance()
        sitk_data = ii.read(data)
    elif type(data) == sitk.Image:
        sitk_data = data
    elif data is None:
        # Neutral action
        return None, None
    else:
        raise TypeError(
            f"Input should be of type str (path) or sitk.Image, not {type(data)}"
        )
    return sitk_data, ii


def run(input_image, path_output, input_lungs=None, verbose=1):
    # loading required data
    display("Loading data...", verbose)
    sitk_image, ii = __handle_input(input_image)
    sitk_lungs, _ = __handle_input(input_lungs)
    if sitk_lungs is None:
        display("Lungs segmentation...", verbose)
        sitk_lungs = lungs_segmentation(sitk_image)

    # processing
    display("Airways segmentation...", verbose)
    sitk_airways, sitk_walls, sitk_trachea, sitk_walls_filled = airways_segmentation(
        sitk_image, sitk_lungs
    )

    # saving
    if path_output is not None:
        ii.write(sitk_airways, path_output[0])
        ii.write(sitk_walls_filled, path_output[1])
    return sitk_airways
