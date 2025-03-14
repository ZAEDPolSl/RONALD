import numpy as np

from bronco.modelling.visvalingam import visvalingam_whyatt_3d


def segment_branch(branch):
    """Segment a branch by using Visvalingam-Whyatt algorithm.

    Parameters
    ----------
    branch : np.ndarray
        A 2D array of shape (n_points, 3) representing the branch.
    Returns
    -------
    np.ndarray
        A 1D array of indices of the points that were kept.
    """
    points = visvalingam_whyatt_3d(branch, 8.0)
    # get the indices of the points that were kept
    indices = np.where(np.isin(branch, points).all(axis=1))[0]
    return indices
