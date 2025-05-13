import numpy as np
from skimage.measure import label
from skimage.util import label_points
from sklearn.decomposition import PCA

from bronco.modelling.cone_construction import is_point_in_cylinder
from bronco.modelling.densify import densify_point_cloud
from bronco.modelling.ellipse import find_ellipse as f_el
from bronco.modelling.segment_branch import segment_branch


def between_endpoints(points, c1, c2, eps=1e-10):
    # Compute cylinder axis vector and normalize
    axis_vector = c2 - c1
    height = np.linalg.norm(axis_vector)
    axis_unit_vector = axis_vector / (height + eps)

    # Project points onto cylinder's axis
    vector_to_points = points - c1
    projection_lengths = np.dot(vector_to_points, axis_unit_vector)

    # Check if projections are within height bounds
    height_bounds_check = (0 <= projection_lengths) & (projection_lengths <= height)
    return points[height_bounds_check]


def prepare_branch_svd(points):
    branch_points = densify_point_cloud(points, factor=50)
    svd = PCA(n_components=3)
    svd.fit(branch_points)
    transformed_points = svd.transform(branch_points)
    return svd, transformed_points

def separate_branch(transformed_points, transformed_endpoints, segments=False):
    if segments:
        transformed_points = between_endpoints(
            transformed_points, transformed_endpoints[0, :], transformed_endpoints[1, :]
        )

    # Extract the first principal component values of endpoints
    first_endpoint_value = transformed_endpoints[0, 0]  # First endpoint on first axis
    second_endpoint_value = transformed_endpoints[1, 0]  # Second endpoint on first axis

    # Find points matching the same height on the first SVD axis
    tol = (transformed_points[:, 0].max() - transformed_points[:, 0].min()) / 100
    mask_first = np.isclose(transformed_points[:, 0], first_endpoint_value, atol=tol)
    mask_second = np.isclose(transformed_points[:, 0], second_endpoint_value, atol=tol)
    # Extract the corresponding original points
    points_same_as_first = transformed_points[mask_first]
    points_same_as_second = transformed_points[mask_second]
    ellipse1 = f_el(points_same_as_first, transformed_endpoints[0, :])
    ellipse2 = f_el(points_same_as_second, transformed_endpoints[1, :])
    return [ellipse1, ellipse2]


def check_ellipse_convergence(ellipse, ellipse_constraint, eps=1e-10, to_higher=True):
    """
    Check if the ellipse is converging to the constraint ellipse. If not, resize the ellipse to the constraint ellipse.

    Parameters:
        ellipse: numpy array of shape (3,3) - Ellipse parameters (center, major axis, minor axis)
        ellipse_constraint: numpy array of shape (3,3) - Ellipse constraint parameters (center, major axis, minor axis)
        eps: float - Small value to avoid division by zero.
    Returns:
        ellipse: tuple - Ellipse parameters (center, major axis, minor axis)
    """
    if (
        np.linalg.norm(ellipse[1]) > np.linalg.norm(ellipse_constraint[1])
        and np.linalg.norm(ellipse_constraint[2]) > 0
        and to_higher
    ):
        new_major_axis = (
            ellipse[1]
            / (np.linalg.norm(ellipse[1]) + eps)
            * (np.linalg.norm(ellipse_constraint[1]) + eps)
        )
        new_minor_axis = (
            ellipse[2]
            / (np.linalg.norm(ellipse[2]) + eps)
            * (np.linalg.norm(ellipse_constraint[2]) + eps)
        )
        ellipse = ellipse[0], new_major_axis, new_minor_axis
    elif np.linalg.norm(ellipse[1]) == 0 or np.linalg.norm(ellipse[2]) == 0:
        ellipse = ellipse[0], ellipse_constraint[1], ellipse_constraint[2]
    return ellipse


def analyse_segment(points, ellipses, eps=1e-10):
    """
    Analyse a branch by fitting cylinder to each point and checking if the point is in the cylinder.

    Parameters:
        points: np.ndarray - numpy array of shape (N, 3) - Points in the 3D image coordinate system.
        ellipses: list - list of numpy arrays of shape (3,3) - Ellipse parameters (center, major axis, minor axis)
        eps: float - Small value to avoid division by zero.
    Returns:
        inside_cylinder: numpy array of shape (N, 3) - Points inside the cylinder.
    """
    ellipse1 = ellipses[0]
    ellipse2 = ellipses[1]

    ellipse2 = check_ellipse_convergence(ellipse2, ellipse1, eps=eps)
    ellipse1 = check_ellipse_convergence(ellipse1, ellipse2, eps=eps, to_higher=False)

    inside_cylinder, on_first_base, on_second_base = is_point_in_cylinder(
        points,
        ellipse1[0],
        ellipse2[0],
        ellipse1[1],
        ellipse1[2],
        ellipse2[1],
        ellipse2[2],
    )
    return inside_cylinder, on_first_base, on_second_base


def smooth_branch(branch, image, segments=False, eps=1e-10):
    """
    Smooth a branch by fitting an ellipse to each point and checking if the point is in the cylinder.

    Parameters:
        branch: np.ndarray - numpy array of shape (N, 3) - Points in the branch.
        image: numpy array of shape (I, J, K) - 3D binary image
        segments: bool - If True, divide the branch into segments.
        eps: float - Small value to avoid division by zero.
    Returns:
        smooth_branch: numpy array of shape (I, J, K) - 3D binary image
    """
    best_cylinder = np.zeros(image.shape, dtype=int)

    points = np.argwhere(image == 1)
    svd, densified_points =  prepare_branch_svd(points)
    transformed_points = svd.transform(points)

    if segments:
        indices_options = segment_branch(branch, adaptive=True)
    else:
        indices_options = [np.array([0, -1])]
    # for each consecutive pair of indices, analyse the segment
    for indices in indices_options:
        smooth_cylinder = np.zeros(image.shape, dtype=int)
        for i in range(len(indices) - 1):
            if (
                np.all(branch[indices[i] + 1] == branch[indices[i + 1] - 1])
                or indices[i] != 0
            ):
                start_idx = indices[i]
                end_idx = indices[i + 1]
            else:
                start_idx = indices[i] + 1
                end_idx = indices[i + 1] - 1

            transformed_endpoints = svd.transform(np.array([branch[start_idx], branch[end_idx]]))
            ellipses = separate_branch(densified_points, transformed_endpoints, segments)

            inside_cylinder, _on_first_base, on_second_base = analyse_segment(transformed_points, ellipses, eps=eps)
            # ellipses 0 and previous ellipse
            # previous ellipse is in a different coordinate system - get the lower ellipse indices beforehand from svd and then use in og coord system
            cyl = points[inside_cylinder]
            if i == 0:
                on_first_base = _on_first_base
            cyl_mask = np.zeros(image.shape, dtype=int)
            cyl_mask[cyl[:, 0], cyl[:, 1], cyl[:, 2]] = 1
            smooth_cylinder = np.logical_or(smooth_cylinder, cyl_mask)
        if np.sum(smooth_cylinder) > np.sum(best_cylinder):
            best_cylinder = smooth_cylinder
            second_base = points[on_second_base]
            first_base = points[on_first_base]
        elif np.sum(smooth_cylinder) == 0:
            continue
        else: break
    return best_cylinder.astype(int), first_base, second_base
