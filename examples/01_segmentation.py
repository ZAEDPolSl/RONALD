import os
from bronco.run import run

opj = os.path.join
dir_path = os.path.dirname(os.path.realpath(__file__))
root = os.path.split(dir_path)[0]

if __name__ == "__main__":
    path_input = opj(root, "data/input/patient1.nrrd")
    path_supl = opj(root, "data/input/patient1_lungs.nrrd")
    path_output = opj(root, "data/output.nrrd")

    run(path_input, path_output, path_supl)
