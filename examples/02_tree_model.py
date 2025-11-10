import os
import SimpleITK as sitk
from bronco.modelling.smooth_tree import smooth_tree
from bronco.io_local import ImageInstance

opj = os.path.join
dir_path = os.path.dirname(os.path.realpath(__file__))
root = os.path.split(dir_path)[0]

if __name__ == "__main__":
    path_airways = opj(root, "data/airways1.nrrd")
    path_rough = opj(root, "data/rough_tracheobronchial1.nrrd")
    path_output = opj(root, "data/tracheobronchial_tree1.nrrd")

    ii = ImageInstance()
    sitk_airways = ii.read(path_airways)
    sitk_rough = ImageInstance().read(path_rough)

    sitk_smooth = smooth_tree(sitk_rough, sitk_airways)
    ii.write(sitk_smooth, path_output)
    print("Done!")
