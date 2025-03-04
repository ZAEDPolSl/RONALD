import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
from skimage.measure import label, regionprops


def points_in_plane(points, plane_normal, plane_point):
    """
    Check if points are in the plane defined by a normal vector and a point on the plane.

    Parameters:
        points: numpy array of shape (N, 3) - Array of 3D points.
        plane_normal: numpy array of shape (3,) - Normal vector of the plane.
        plane_point: numpy array of shape (3,) - Point on the plane.

    Returns:
        numpy array of bools - True if point is in the plane; False otherwise.
    """
    # Compute vectors from plane point to each point
    vectors = points - plane_point

    # Project onto plane normal
    projection_lengths = np.dot(vectors, plane_normal)

    return np.isclose(projection_lengths, 0)


def fit_plane(points, plane_point):
    """
    Fit a plane to a set of 3D points using SVD.

    Parameters:
        points: numpy array of shape (N, 3) - Array of 3D points.

    Returns:
        plane_point: numpy array (3,) - A point on the plane (centroid).
        plane_normal: numpy array (3,) - Normal vector of the plane.
    """
    # Compute centroid
    # plane_point = np.mean(points, axis=0)

    # Perform SVD to find the normal vector
    _, _, vh = np.linalg.svd(points - plane_point)
    plane_normal = vh[2]  # Normal is the last row of V^T

    return plane_normal


def project_points_onto_plane(points, plane_point, eps=1e-10):
    """
    Project 3D points onto a plane defined by a normal vector and a point on the plane.

    Parameters:
        points: numpy array of shape (N, 3) - Array of 3D points.
        eps: float - Small value to avoid division by zero.
    Returns:
        projected_points: numpy array of shape (N, 2) - Projected points in the plane's local coordinates.
    """
    # Compute vectors from plane point to each point
    plane_normal = fit_plane(points, plane_point)
    vectors = points - plane_point

    # Project onto plane
    projected_vectors = vectors - np.dot(vectors, plane_normal)[:, None] * plane_normal

    # Define local coordinate system in the plane
    # Find a vector perpendicular to plane_normal
    if not np.allclose(plane_normal, [0, 0, 1]):
        perp_vector = np.cross(plane_normal, [0, 0, 1.0])
    else:
        perp_vector = np.cross(plane_normal, [0, 1.0, 0])
    perp_vector /= np.linalg.norm(perp_vector) + eps

    # Find another perpendicular vector (orthonormal basis)
    second_perp_vector = np.cross(plane_normal, perp_vector)

    # Project onto local coordinates
    x_local = np.dot(projected_vectors, perp_vector)
    y_local = np.dot(projected_vectors, second_perp_vector)

    return np.column_stack((x_local, y_local))


def create_local_mask(points):
    """
    Create a binary mask of points in the local coordinate system.

    Parameters:
        points: numpy array of shape (N, 2) - Array of 2D points in the local coordinate system.

    Returns:
        mask: numpy array of shape (H, W) - Binary mask of points.
    """
    # Find the bounding box of the points
    min_x, min_y = np.min(np.round(points).astype(int), axis=0)
    max_x, max_y = np.max(np.round(points).astype(int), axis=0)

    # Create a mask of zeros
    mask = np.zeros((int(max_y - min_y) + 1, int(max_x - min_x) + 1), dtype=np.uint8)

    # Convert points to integer coordinates
    points_int = np.round(points).astype(int)

    # Shift points to start from (0, 0)
    points_int_shifted = points_int - np.array([min_x, min_y])

    # Fill in the mask at the integer coordinates
    mask[points_int_shifted[:, 1], points_int_shifted[:, 0]] = 1

    return mask


def get_mask_for_ellipse_fit(image, plane_normal, plane_point):
    """
    Get mask in the plane for fitting an ellipse.

    Parameters:
        image: numpy array of shape (I, J, K) - 3D binary image
        plane_normal: numpy array of shape (3,) - Normal vector of the plane.
        plane_point: numpy array of shape (3,) - Point on the plane.

    Returns:
        local_mask: numpy array of shape (H, W) - Binary mask of points in the plane.
    """
    # get the coordinates of the points in the image
    points = np.argwhere(image == 1)
    # Check which points are in the plane
    in_plane = points_in_plane(points, plane_normal, plane_point)

    in_plane_projection = project_points_onto_plane(points[in_plane], plane_point)

    # get the mask for the local plane
    local_mask = create_local_mask(in_plane_projection)
    return local_mask


def fit_ellipse_to_mask(mask):
    """
    Fit an ellipse to a binary mask.

    Parameters:
        mask: numpy array of shape (H, W) - Binary mask.

    Returns:
        ellipse_params: numpy array of shape (5,) - Ellipse parameters (center_x, center_y, semi_major_axis, semi_minor_axis, rotation_angle).
    """
    # get regionprops from skimage
    label_img = label(mask)
    regions = regionprops(label_img)
    # use the largest region
    region = regions[0]
    # get the ellipse parameters in the numpy array
    ellipse_params = np.array(
        [
            region.centroid[1],
            region.centroid[0],
            region.major_axis_length / 2,
            region.minor_axis_length / 2,
            region.orientation,
        ]
    )
    return ellipse_params


def transform_ellipse_to_3d(
    plane_normal,
    ellipse_center_2d,
    semi_major_axis,
    semi_minor_axis,
    rotation_angle,
    eps=1e-10,
):
    """
    Transform 2D ellipse parameters to 3D space.

    Parameters:
        plane_normal: numpy array of shape (3,) - Normal vector of the plane.
        ellipse_center_2d: tuple (h, k) - Center of the ellipse in the plane's local coordinates.
        semi_major_axis: float - Semi-major axis length.
        semi_minor_axis: float - Semi-minor axis length.
        rotation_angle: float - Rotation angle of the ellipse in radians.
        eps: float - Small value to avoid division by zero.
    Returns:
        ellipse_center_3d: numpy array of shape (3,) - Center of the ellipse in 3D coordinates.
        major_axis_vector_3d: numpy array of shape (3,) - Semi-major axis vector in 3D.
        minor_axis_vector_3d: numpy array of shape (3,) - Semi-minor axis vector in 3D.
    """
    # Convert inputs to numpy arrays
    plane_normal = np.array(plane_normal)
    plane_normal /= np.linalg.norm(plane_normal) + eps

    # Define local coordinate system in the plane
    # Find a vector perpendicular to plane_normal
    if not np.allclose(plane_normal, [0, 0, 1]):
        perp_vector = np.cross(plane_normal, [0.0, 0.0, 1.0])
    else:
        perp_vector = np.cross(plane_normal, [0.0, 1.0, 0.0])
    perp_vector /= np.linalg.norm(perp_vector) + eps

    # Find another perpendicular vector (orthonormal basis)
    second_perp_vector = np.cross(plane_normal, perp_vector)

    rot = R.from_rotvec(rotation_angle * plane_normal)  # Rotation around normal
    major_vector = rot.apply(perp_vector) * semi_major_axis / 2
    minor_vector = rot.apply(second_perp_vector) * semi_minor_axis / 2

    center_3d = (
        ellipse_center_2d[0] * perp_vector + ellipse_center_2d[1] * second_perp_vector
    )  # Map 2D coordinates into 3D plane
    return center_3d, major_vector, minor_vector


def get_ellipse_params(mask, plane_normal, plane_point):
    """
    Get ellipse parameters for fitting an ellipse to points in the plane.

    Parameters:
        mask: numpy array of shape (H, W) - Binary mask of points in the plane.
        plane_normal: numpy array of shape (3,) - Normal vector of the plane.
        plane_point: numpy array of shape (3,) - Point on the plane.

    Returns:
        ellipse_center_3d: numpy array of shape (3,) - Center of the ellipse in 3D coordinates.
        major_axis_vector_3d: numpy array of shape (3,) - Semi-major axis vector in 3D.
        minor_axis_vector_3d: numpy array of shape (3,) - Semi-minor axis vector in 3D.
    """
    # Fit ellipse to points
    h, k, a, b, theta = fit_ellipse_to_mask(mask)
    ellipse_center, major_axis_vector_3d, minor_axis_vector_3d = (
        transform_ellipse_to_3d(plane_normal, plane_point, (h, k), a, b, theta)
    )
    return ellipse_center, major_axis_vector_3d, minor_axis_vector_3d


def find_ellipse(image, plane_normal, plane_point):
    """
    Find an ellipse in the plane defined by a normal vector and a point on the plane.

    Parameters:
        image: numpy array of shape (I, J, K) - 3D binary image
        plane_normal: numpy array of shape (3,) - Normal vector of the plane.
        plane_point: numpy array of shape (3,) - Point on the plane.

    Returns:
        ellipse_center_3d: numpy array of shape (3,) - Center of the ellipse in 3D coordinates.
        major_axis_vector_3d: numpy array of shape (3,) - Semi-major axis vector in 3D.
        minor_axis_vector_3d: numpy array of shape (3,) - Semi-minor axis vector in 3D.
    """
    # Get mask for ellipse fitting
    mask = get_mask_for_ellipse_fit(image, plane_normal, plane_point)

    # Get ellipse parameters
    ellipse_center_3d, major_axis_vector_3d, minor_axis_vector_3d = get_ellipse_params(
        mask, plane_normal, plane_point
    )

    return ellipse_center_3d, major_axis_vector_3d, minor_axis_vector_3d
