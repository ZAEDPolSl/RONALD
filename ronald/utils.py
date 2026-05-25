import pickle
import numpy as np


def display(text, verbose):
    if verbose > 0:
        print(text)


def save_object(obj, filename):
    with open(filename, "wb") as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def get_gmm_metadata(gmm):

    weights = gmm.weights_
    means = gmm.means_
    covars = gmm.covariances_

    weights = list(weights)
    means = list(means.squeeze())
    covars = list(covars.squeeze())
    stds = list(np.sqrt(covars))

    gmm_list = [
        {"mean": mean, "weight": weight, "std": std}
        for mean, std, weight in zip(means, stds, weights)
    ]

    gmm_list = sorted(gmm_list, key=lambda x: x["mean"])

    return gmm_list


def solve(m1, m2, std1, std2):
    a = 1 / (2 * std1**2) - 1 / (2 * std2**2)
    b = m2 / (std2**2) - m1 / (std1**2)
    c = m1**2 / (2 * std1**2) - m2**2 / (2 * std2**2) - np.log(std2 / std1)
    return np.roots([a, b, c])
