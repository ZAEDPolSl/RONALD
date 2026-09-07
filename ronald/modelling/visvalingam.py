import heapq

import numpy as np


def calculate_triangle_area(a, b, c):
    ab = b - a
    ac = c - a
    return 0.5 * np.linalg.norm(np.cross(ab, ac))


def _indices_at_thresholds(points, thresholds):
    """Run one elimination sequence and snapshot it at each threshold."""
    point_count = len(points)
    mask = np.ones(point_count, dtype=bool)
    remaining_count = point_count
    heap = []
    prev = np.arange(point_count)
    next = np.arange(point_count)
    prev[1:] = np.arange(point_count - 1)
    next[:-1] = np.arange(1, point_count)
    prev[0] = -1
    next[-1] = -1

    for i in range(1, point_count - 1):
        area = calculate_triangle_area(points[i - 1], points[i], points[i + 1])
        heapq.heappush(heap, (area, i))

    results = {}
    for threshold in sorted(set(thresholds)):
        while remaining_count > 2 and heap:
            area, idx = heapq.heappop(heap)
            if not mask[idx]:
                continue
            if area >= threshold:
                # Resume from this item when processing the next threshold.
                heapq.heappush(heap, (area, idx))
                break

            mask[idx] = False
            remaining_count -= 1
            previous_idx = prev[idx]
            next_idx = next[idx]
            if previous_idx != -1 and next_idx != -1:
                previous_area = (
                    calculate_triangle_area(
                        points[prev[previous_idx]],
                        points[previous_idx],
                        points[next_idx],
                    )
                    if prev[previous_idx] != -1
                    else np.inf
                )
                next_area = (
                    calculate_triangle_area(
                        points[previous_idx], points[next_idx], points[next[next_idx]]
                    )
                    if next[next_idx] != -1
                    else np.inf
                )
                heapq.heappush(heap, (previous_area, previous_idx))
                heapq.heappush(heap, (next_area, next_idx))
                next[previous_idx] = next_idx
                prev[next_idx] = previous_idx
            elif previous_idx != -1:
                next[previous_idx] = next_idx
            elif next_idx != -1:
                prev[next_idx] = previous_idx

        results[threshold] = np.flatnonzero(mask)

    return [results[threshold] for threshold in thresholds]


def visvalingam_whyatt_3d(
    points: np.ndarray, epsilon=0.51, return_indices=False
) -> np.ndarray:
    """Simplify an ordered 3D polyline with the Visvalingam-Whyatt algorithm."""
    if len(points) <= 2:
        return np.arange(len(points)) if return_indices else points

    indices = _indices_at_thresholds(points, [epsilon])[0]
    return indices if return_indices else points[indices]


def visvalingam_whyatt_3d_many(points, epsilons, return_indices=False):
    """Simplify once and return results for multiple area thresholds."""
    thresholds = list(epsilons)
    if not thresholds:
        return []
    if len(points) <= 2:
        result = np.arange(len(points)) if return_indices else points
        return [result.copy() for _ in thresholds]

    index_sets = _indices_at_thresholds(points, thresholds)
    if return_indices:
        return index_sets
    return [points[indices] for indices in index_sets]
