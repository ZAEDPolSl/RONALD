import os
from ronald.run import run

opj = os.path.join
dir_path = os.path.dirname(os.path.realpath(__file__))
root = os.path.split(dir_path)[0]

if __name__ == "__main__":
    path_input = opj(root, "data/input/patient1.nrrd")
    path_supl = opj(root, "data/input/lungs1.nrrd")
    output_paths = [
        opj(root, "data/airways1.nrrd"),
        opj(root, "data/rough_tracheobronchial1.nrrd"),
    ]

    run(path_input, output_paths, path_supl)
