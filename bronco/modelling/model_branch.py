import numpy as np
import math

from bronco.modelling.oval_construction import find_ellipse
from bronco.modelling.cone_construction import is_point_in_cylinder


def smooth_branch(branch, image, mode=None, previous_oval=None, eps=1e-10):
    """
    Smooth a branch by fitting an ellipse to each point and checking if the point is in the cylinder.

    Parameters:
        branch: numpy array of shape (N, 3) - Points in the branch.
        image: numpy array of shape (I, J, K) - 3D binary image
        mode: str - Mode for smoothing the branch. Options are "5percent" or "full".
        previous_oval: tuple - Tuple of the previous ellipse major and minor axis
        eps: float - Small value to avoid division by zero.
    Returns:
        smooth_branch: numpy array of shape (I, J, K) - 3D binary image
    """
    if previous_oval is not None:
        max_elipse_major = previous_oval[0]
        max_elipse_minor = previous_oval[1]
    if mode == "5percent":
        first_5_percent = math.ceil(branch.shape[0] * 0.05)
        last_5_percent = math.floor(branch.shape[0] * 0.95 - 1)
        point1 = branch[first_5_percent]
        plane_normal1 = branch[first_5_percent + 1] - point1
        point2 = branch[last_5_percent]
        plane_normal2 = point2 - branch[last_5_percent - 1]

    else:
        point1 = branch[0]
        plane_normal1 = branch[1] - point1
        point2 = branch[-1]
        plane_normal2 = point2 - branch[-2]

    ellipse1 = find_ellipse(image, plane_normal1, point1)
    ellipse2 = find_ellipse(image, plane_normal2, point2)
    if previous_oval is not None:
        # check if the new ellipse is bigger than the previous one
        if np.linalg.norm(ellipse1[1]) > np.linalg.norm(
            max_elipse_major
        ) and np.linalg.norm(max_elipse_minor) > 0:
            # normalize the new ellipse to the previous one
            new_major_axis = (
                ellipse1[1]
                / (np.linalg.norm(ellipse1[1]) + eps)
                * (np.linalg.norm(max_elipse_major) + eps)
            )
            new_minor_axis = (
                ellipse1[2]
                / (np.linalg.norm(ellipse1[2]) + eps)
                * (np.linalg.norm(max_elipse_minor) + eps)
            )
            ellipse1 = ellipse1[0], new_major_axis, new_minor_axis
        if np.linalg.norm(ellipse2[1]) > np.linalg.norm(
            max_elipse_major
        ) and np.linalg.norm(max_elipse_minor) > 0:
            new_major_axis = (
                ellipse2[1]
                / (np.linalg.norm(ellipse2[1]) + eps)
                * (np.linalg.norm(max_elipse_major) + eps)
            )
            new_minor_axis = (
                ellipse2[2]
                / (np.linalg.norm(ellipse2[2]) + eps)
                * (np.linalg.norm(max_elipse_minor) + eps)
            )
            ellipse2 = ellipse2[0], new_major_axis, new_minor_axis
    x, y, z = np.meshgrid(
        np.arange(image.shape[0]),
        np.arange(image.shape[1]),
        np.arange(image.shape[2]),
        indexing="ij",
    )
    points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    inside_cylinder = is_point_in_cylinder(
        points,
        point1, #ellipse1[0],
        point2, #ellipse2[0],
        ellipse1[1],
        ellipse1[2],
        ellipse2[1],
        ellipse2[2],
    )

    # inside cylinder back to 3D
    inside_cylinder = inside_cylinder.reshape(image.shape)
    return inside_cylinder, (ellipse2[1], ellipse2[2])
