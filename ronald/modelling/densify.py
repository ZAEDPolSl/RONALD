import numpy as np
from scipy.spatial import KDTree


def densify_point_cloud(
    points: np.ndarray, factor=2, pair_chunk_size=4096
) -> np.ndarray:
    """
    Densify a point cloud by interpolating between close pairs of points.
    """
    if len(points) < 2 or factor < 2:
        return points.copy()

    tree = KDTree(points)
    k = min(10, len(points))
    _, neighbors = tree.query(points, k=k)

    # Preserve the legacy iteration order: point, neighbour, interpolation fraction.
    source_indices = np.repeat(np.arange(len(points)), k - 1)
    neighbor_indices = neighbors[:, 1:].reshape(-1)
    valid = (neighbor_indices > source_indices) & (neighbor_indices < len(points))
    source_indices = source_indices[valid]
    neighbor_indices = neighbor_indices[valid]

    if len(source_indices) == 0:
        return points.copy()

    interpolation_dtype = np.result_type(points.dtype, 1.0 / factor)
    fractions_float64 = np.arange(1, factor, dtype=np.float64) / factor
    fractions = fractions_float64.astype(interpolation_dtype, copy=False)
    complementary_fractions = (1.0 - fractions_float64).astype(
        interpolation_dtype, copy=False
    )
    result = np.empty(
        (len(points) + len(source_indices) * len(fractions), 3),
        dtype=interpolation_dtype,
    )
    result[: len(points)] = points

    output_start = len(points)
    for chunk_start in range(0, len(source_indices), pair_chunk_size):
        chunk_end = min(chunk_start + pair_chunk_size, len(source_indices))
        starts = points[source_indices[chunk_start:chunk_end]]
        stops = points[neighbor_indices[chunk_start:chunk_end]]
        interpolated = (
            starts[:, None, :] * complementary_fractions[None, :, None]
            + stops[:, None, :] * fractions[None, :, None]
        ).reshape(-1, 3)
        result[output_start : output_start + len(interpolated)] = interpolated
        output_start += len(interpolated)

    return result
