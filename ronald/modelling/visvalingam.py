import numpy as np
import heapq


def calculate_triangle_area(a, b, c):
    ab = b - a
    ac = c - a
    return 0.5 * np.linalg.norm(np.cross(ab, ac))


def visvalingam_whyatt_3d(points: np.ndarray, epsilon=0.51) -> np.ndarray:
    """
    Simplify a polyline in 3D using the Visvalingam-Whyatt algorithm with NumPy arrays.

    Args:
        points (numpy.ndarray): Ordered array of 3D points.
        epsilon (float): Minimum effective area threshold.

    Returns:
        numpy.ndarray: Simplified array of 3D points.
    """
    if len(points) <= 2:
        return points

    points = points.copy()
    N = len(points)
    mask = np.ones(N, dtype=bool)
    heap = []
    prev = np.arange(N)
    next = np.arange(N)

    # Set up prev/next pointers
    prev[1:] = np.arange(N - 1)
    next[:-1] = np.arange(1, N)
    prev[0] = -1
    next[-1] = -1

    # Compute initial areas and heap
    areas = np.full(N, np.inf)
    for i in range(1, N - 1):
        areas[i] = calculate_triangle_area(points[i - 1], points[i], points[i + 1])
        heapq.heappush(heap, (areas[i], i))

    while heap:
        area, idx = heapq.heappop(heap)
        if not mask[idx]:
            continue
        if area >= epsilon:
            break
        mask[idx] = False

        # Update neighbors
        i_prev = prev[idx]
        i_next = next[idx]
        if i_prev != -1 and i_next != -1:
            areas[idx] = np.inf
            areas[i_prev] = (
                calculate_triangle_area(
                    points[prev[i_prev]], points[i_prev], points[i_next]
                )
                if prev[i_prev] != -1
                else np.inf
            )
            areas[i_next] = (
                calculate_triangle_area(
                    points[i_prev], points[i_next], points[next[i_next]]
                )
                if next[i_next] != -1
                else np.inf
            )
            heapq.heappush(heap, (areas[i_prev], i_prev))
            heapq.heappush(heap, (areas[i_next], i_next))
            next[i_prev] = i_next
            prev[i_next] = i_prev
        elif i_prev != -1:
            next[i_prev] = i_next
        elif i_next != -1:
            prev[i_next] = i_prev

        # Early exit if only two points remain
        if np.count_nonzero(mask) <= 2:
            break

    return points[mask]
