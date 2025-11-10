import os
import SimpleITK as sitk
from bronco.segmentation import vessel_segmentation
from bronco.io_local import ImageInstance

opj = os.path.join
dir_path = os.path.dirname(os.path.realpath(__file__))
root = os.path.split(dir_path)[0]

if __name__ == "__main__":
    path_data = opj(root, "data/patient1.nrrd")
    path_lungs = opj(root, "data/lungs1.nrrd")

    path_output = opj(root, "data/vessels1.nrrd")

    ii = ImageInstance()
    sitk_image = ii.read(path_data)
    sitk_lungs = ImageInstance().read(path_lungs)

    sitk_vessels = vessel_segmentation(sitk_image, sitk_lungs)
    ii.write(sitk_vessels, path_output)
    print("Done!")
