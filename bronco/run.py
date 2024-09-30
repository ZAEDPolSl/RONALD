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
        raise TypeError(f"Input should be of type str (path) or sitk.Image, not {type(data)}")
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
    sitk_airways, sitk_walls, sitk_trachea = airways_segmentation(sitk_image, sitk_lungs)

    # saving
    if path_output is not None:
        ii.write(sitk_airways, path_output)
    return sitk_airways


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Full airways segmentation pipeline with lungs segmentation."
    )
    parser.add_argument("--input_image", required=True, help="(str|sitk.Image) Input thorax CT image or path to the "
                                                             "DICOM folder or NRRD file which contains the image.")
    parser.add_argument("--path_output", required=True, type=str, help="(str) path to the saving location of resulting"
                                                                       " airways segmentation.")
    parser.add_argument("--input_lungs", required=False, help="(str|sitk.Image) Input lungs segmentation, if not "
                                                              "provided the lungs segmentation will be "
                                                              "generated automatically.")
    parser.add_argument("--verbose", required=False, type=int, default=1,
                        help="(int) 1 or more == display messages, "
                             "0 == do not display messages.")
    args = parser.parse_args()

    _input_image = args.input_image
    _path_output = args.path_output
    _input_lungs = args.input_lungs
    _verbose = args.verbose

    _sitk_airways = run(_input_image, _path_output, _input_lungs, _verbose)
