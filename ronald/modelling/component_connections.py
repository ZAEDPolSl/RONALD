"""Conservative, opt-in reconnection of separate Sato skeleton components.

Distances/radii are physical when spacing_zyx is in mm. Nothing is discarded:
rejected fragments remain in the returned graph. The bridge mask is intended
for union *after* original-forest modelling, since a gap has no Sato to fit.
"""

from collections import Counter
import networkx as nx
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from .local_cross_section import LocalCrossSectionSampler
from .sato_envelope import fit_sato_guided_ellipse


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else np.zeros(3)


def _frame(tangent):
    tangent = _unit(tangent)
    axis = np.eye(3)[np.argmin(np.abs(tangent))]
    first = _unit(np.cross(tangent, axis))
    return np.array([tangent, first, np.cross(tangent, first)])


def _ordered(graph, u, v, data):
    points = np.asarray(data["pts"], dtype=float)
    start, end = np.asarray(graph.nodes[u]["o"]), np.asarray(graph.nodes[v]["o"])
    if np.linalg.norm(points[-1] - start) < np.linalg.norm(points[0] - start):
        points = points[::-1]
    points = np.vstack((start, points, end))
    return points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-8]]


def _endpoint_direction(graph, node, spacing, lookback=3.0):
    neighbor = next(iter(graph.neighbors(node)))
    points = _ordered(graph, node, neighbor, graph.edges[node, neighbor]) * spacing
    distances = np.cumsum(np.r_[0.0, np.linalg.norm(np.diff(points, axis=0), axis=1)])
    index = min(max(1, np.searchsorted(distances, lookback)), len(points) - 1)
    return _unit(points[0] - points[index])


def _sample(mask, positions):
    indices = np.rint(positions).astype(int)
    valid = np.all((indices >= 0) & (indices < np.asarray(mask.shape)), axis=1)
    result = np.zeros(len(indices), dtype=bool)
    result[valid] = mask[tuple(indices[valid].T)] != 0
    return result


def _ellipse(sampler, center, tangent, spacing):
    # The voxel-plane normal is a covector: multiplying by spacing keeps the
    # transformed sampled plane perpendicular to the physical tangent.
    voxel_frame = _frame(tangent * spacing)
    points, diagnostic = sampler.sample(center, voxel_frame)
    frame = _frame(tangent)
    projected = ((points - center) * spacing) @ frame.T
    ellipse, fit = fit_sato_guided_ellipse(projected, None, np.zeros(3))
    if ellipse is not None:
        axes = np.array(ellipse[1:]) @ frame
        radii = np.linalg.norm(axes, axis=1)
        return (
            axes / radii[:, None],
            radii,
            {"method": "sato_ellipse", "section": diagnostic, "fit": fit},
        )
    # Explicit conservative fallback, computed in a tiny local patch only.
    half = np.ceil(6.0 / spacing).astype(int) + 2
    voxel = np.rint(center).astype(int)
    lower = np.maximum(0, voxel - half)
    upper = np.minimum(sampler.mask.shape, voxel + half + 1)
    crop = sampler.mask[tuple(slice(a, b) for a, b in zip(lower, upper))] != 0
    if crop.size and np.all(voxel >= lower) and np.all(voxel < upper):
        distance = ndi.distance_transform_edt(np.pad(crop, 1), sampling=spacing)
        radius = float(distance[tuple(voxel - lower + 1)])
    else:
        radius = 0.0
    radius = max(0.5 * float(spacing.min()), min(radius, 2.0))
    return (
        frame[1:],
        np.full(2, radius),
        {"method": "edt_circle_fallback", "reason": fit["reason"], "radius": radius},
    )


def _hermite(start, end, outgoing, incoming, step):
    length = np.linalg.norm(end - start)
    times = np.linspace(0.0, 1.0, max(3, int(np.ceil(length * 2 / step)) + 1))
    t = times[:, None]
    path = (
        (2 * t**3 - 3 * t**2 + 1) * start
        + (t**3 - 2 * t**2 + t) * outgoing * length
        + (-2 * t**3 + 3 * t**2) * end
        + (t**3 - t**2) * incoming * length
    )
    tangent = np.gradient(path, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1)[:, None], 1e-12)
    return path, tangent, times


def _tube(path, tangents, times, axes0, radii0, axes1, radii1, spacing, shape):
    # Align equivalent ellipse axes before interpolation; do not twist 90deg
    # merely because eigendecomposition swapped equivalent radius labels.
    if abs(axes0[0] @ axes1[1]) > abs(axes0[0] @ axes1[0]):
        axes1, radii1 = axes1[::-1].copy(), radii1[::-1].copy()
    if axes0[0] @ axes1[0] < 0:
        axes1 = -axes1
    radius = max(float(max(radii0)), float(max(radii1)))
    lower = np.maximum(
        0, np.floor((path.min(0) - radius - spacing.max()) / spacing).astype(int)
    )
    upper = np.minimum(
        shape, np.ceil((path.max(0) + radius + spacing.max()) / spacing).astype(int) + 1
    )
    grid = np.indices(tuple(upper - lower)).reshape(3, -1).T + lower
    physical = grid * spacing
    selected = np.zeros(len(grid), dtype=bool)
    slab = 0.65 * float(spacing.max())
    for center, normal, t in zip(path, tangents, times):
        major = (1 - t) * axes0[0] + t * axes1[0]
        major = _unit(major - (major @ normal) * normal)
        if np.linalg.norm(major) < 0.5:
            major = _frame(normal)[1]
        minor = np.cross(normal, major)
        radii = (1 - t) * radii0 + t * radii1
        delta = physical - center
        selected |= (np.abs(delta @ normal) <= slab) & (
            (delta @ major / radii[0]) ** 2 + (delta @ minor / radii[1]) ** 2 <= 1.0
        )
    return grid[selected]


def _split_target(graph, source_edge, point, new_node):
    candidates = [
        (u, v, d)
        for u, v, d in graph.edges(data=True)
        if d.get("_connection_source_edge") == source_edge
    ]
    if not candidates:
        u, v = source_edge
        candidates = [(u, v, graph.edges[u, v])]
    u, v, data = min(
        candidates,
        key=lambda item: np.min(
            np.linalg.norm(np.asarray(item[2]["pts"]) - point, axis=1)
        ),
    )
    for node in (u, v):
        if np.linalg.norm(graph.nodes[node]["o"] - point) < 1e-6:
            return node
    points = _ordered(graph, u, v, data)
    index = int(np.argmin(np.linalg.norm(points - point, axis=1)))
    point = points[index]
    graph.add_node(
        new_node, o=point.copy(), pts=point[None].copy(), component_attachment=True
    )
    graph.remove_edge(u, v)
    for a, b, part in (
        (u, new_node, points[: index + 1]),
        (new_node, v, points[index:]),
    ):
        attrs = dict(data, pts=part.copy(), _connection_source_edge=source_edge)
        attrs["weight"] = float(np.linalg.norm(np.diff(part, axis=0), axis=1).sum())
        graph.add_edge(a, b, **attrs)
    return new_node


class _MaskComponents:
    """26-connected labels in a tight foreground ROI, avoiding a full label volume."""

    def __init__(self, mask):
        occupied = [
            np.flatnonzero(np.any(mask, axis=tuple(j for j in range(3) if j != i)))
            for i in range(3)
        ]
        if any(len(axis) == 0 for axis in occupied):
            self.low = np.zeros(3, dtype=int)
            self.labels = np.zeros((0, 0, 0), dtype=np.uint32)
        else:
            self.low = np.array([axis[0] for axis in occupied])
            high = np.array([axis[-1] + 1 for axis in occupied])
            region = tuple(slice(a, b) for a, b in zip(self.low, high))
            self.labels, _ = ndi.label(
                mask[region] != 0, structure=np.ones((3, 3, 3)), output=np.uint32
            )

    def values(self, positions):
        indices = np.rint(positions).astype(int) - self.low
        inside = np.all(
            (indices >= 0) & (indices < np.array(self.labels.shape)), axis=1
        )
        values = np.zeros(len(indices), dtype=np.uint32)
        values[inside] = self.labels[tuple(indices[inside].T)]
        return values

    def endpoint_id(self, point, spacing):
        """One nearest label only; reject a background center with tied labels."""
        center = self.values(np.asarray(point)[None])[0]
        if center:
            return int(center)
        # A rounded node/section center can fall just outside its component.
        # Search a bounded two-voxel neighborhood, never authorize every nearby CC.
        offsets = np.indices((5, 5, 5)).reshape(3, -1).T - 2
        positions = np.rint(point).astype(int) + offsets
        values = self.values(positions)
        distances = np.linalg.norm((positions - point) * spacing, axis=1)
        eligible = (values != 0) & (distances <= 2 * spacing.max())
        if not eligible.any():
            return None
        distance = distances[eligible].min()
        closest = set(values[eligible & (distances <= distance + 1e-8)])
        return int(next(iter(closest))) if len(closest) == 1 else None

    def touching_ids(self, low, dilated):
        """IDs overlapping this bridge's 26-neighbor footprint."""
        high = low + np.array(dilated.shape)
        begin = np.maximum(low, self.low)
        end = np.minimum(high, self.low + np.array(self.labels.shape))
        if np.any(end <= begin):
            return set()
        label_region = tuple(
            slice(a, b) for a, b in zip(begin - self.low, end - self.low)
        )
        local_region = tuple(slice(a, b) for a, b in zip(begin - low, end - low))
        values = self.labels[label_region][dilated[local_region]]
        return set(np.unique(values)) - {0}


def connect_sato_components(
    graph,
    walls_mask,
    sato_mask,
    *,
    attachment_mask=None,
    spacing_zyx=(1.0, 1.0, 1.0),
    max_gap=8.0,
    min_forward_cos=0.7,
    min_wall_fraction=0.9,
    min_tube_wall_fraction=0.75,
    ambiguity_margin=0.15,
    max_candidates=8,
):
    """Return ``(graph_copy, clipped_bridge_mask_uint8, JSON_diagnostics)``.

    Each smaller component can attach once to a larger component (ranked by
    physical skeleton length). Candidate receiving edge interiors are allowed.
    No extra connection is made within an already connected component. Only a
    few nearby edges are tested per endpoint. Tangents, wall occupancy, endpoint
    survival, post-clipping connectivity, and ambiguity are independently tested.
    Ellipse radii are interpolated from Sato endpoint sections without expansion.
    A bounded circular EDT fallback is explicitly recorded if a fit fails.
    """
    walls = np.asarray(walls_mask)
    sato = np.asarray(sato_mask)
    attachment = sato if attachment_mask is None else np.asarray(attachment_mask)
    spacing = np.asarray(spacing_zyx, dtype=float)
    if walls.ndim != 3 or walls.shape != sato.shape or walls.shape != attachment.shape:
        raise ValueError("masks must have matching 3D shapes")
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("spacing_zyx must contain three positive finite values")
    if not np.isfinite(max_gap) or max_gap <= 0 or max_candidates < 1:
        raise ValueError("invalid candidate distance/count")
    if any(
        not 0 <= value <= 1
        for value in (
            min_forward_cos,
            min_wall_fraction,
            min_tube_wall_fraction,
            ambiguity_margin,
        )
    ):
        raise ValueError("fraction thresholds must be in [0, 1]")
    if graph.is_directed() or graph.is_multigraph():
        raise ValueError("a simple undirected forest graph is required")
    result = graph.copy()
    bridge_mask = np.zeros(walls.shape, dtype=np.uint8)
    diagnostics = {
        "accepted": [],
        "rejections": {},
        "parameters": {
            "max_gap": max_gap,
            "spacing_zyx": spacing.tolist(),
            "min_forward_cos": min_forward_cos,
            "min_wall_fraction": min_wall_fraction,
            "min_tube_wall_fraction": min_tube_wall_fraction,
            "ambiguity_margin": ambiguity_margin,
            "max_candidates": max_candidates,
        },
        "components_before": nx.number_connected_components(graph),
    }
    if not graph.number_of_edges():
        diagnostics["components_after"] = diagnostics["components_before"]
        return result, bridge_mask, diagnostics
    components = list(nx.connected_components(graph))
    size = lambda cc: sum(
        np.linalg.norm(
            np.diff(_ordered(graph, u, v, d) * spacing, axis=0), axis=1
        ).sum()
        for u, v, d in graph.subgraph(cc).edges(data=True)
    )
    components.sort(key=size, reverse=True)
    membership = {n: rank for rank, cc in enumerate(components) for n in cc}
    positions, records = [], []
    for u, v, data in graph.edges(data=True):
        points = _ordered(graph, u, v, data)
        physical = points * spacing
        for i, point in enumerate(points):
            tangent = _unit(
                physical[min(i + 2, len(points) - 1)] - physical[max(0, i - 2)]
            )
            positions.append(point)
            records.append((u, v, tangent))
    positions = np.asarray(positions)
    tree = cKDTree(positions * spacing)
    sampler = LocalCrossSectionSampler(sato, max_half_size=32.0)
    sato_components = _MaskComponents(sato)
    attachment_components = (
        sato_components if attachment_mask is None else _MaskComponents(attachment)
    )
    accepted_voxels = []
    rejected = Counter()
    next_node = (
        max(graph.nodes, default=-1) + 1
        if all(isinstance(n, (int, np.integer)) for n in graph)
        else None
    )
    for rank, component in enumerate(components[1:], 1):
        proposals = []
        for node in component:
            if graph.degree(node) != 1:
                continue
            origin = np.asarray(graph.nodes[node]["o"], dtype=float)
            outgoing = _endpoint_direction(graph, node, spacing)
            nearby = tree.query_ball_point(origin * spacing, max_gap)
            nearby.sort(
                key=lambda index: np.linalg.norm((positions[index] - origin) * spacing)
            )
            edges_seen = set()
            # Source geometry is fixed across receiving candidates. Fit lazily
            # after the cheap rejection checks, once per endpoint and call.
            source_ellipse = None
            for index in nearby:
                u, v, incoming = records[index]
                if membership[u] >= rank or (u, v) in edges_seen:
                    continue
                edges_seen.add((u, v))
                if len(edges_seen) > max_candidates:
                    break
                target = positions[index]
                delta = (target - origin) * spacing
                distance = float(np.linalg.norm(delta))
                alignment = float(outgoing @ _unit(delta))
                if distance < 0.5 * spacing.min() or alignment < min_forward_cos:
                    rejected["direction_or_zero_gap"] += 1
                    continue
                if incoming @ delta < 0:
                    incoming = -incoming
                path, tangents, times = _hermite(
                    origin * spacing,
                    target * spacing,
                    outgoing,
                    incoming,
                    0.4 * spacing.min(),
                )
                length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
                if length > 1.5 * max_gap:
                    rejected["curved_path_too_long"] += 1
                    continue
                support = float(_sample(walls, path / spacing).mean())
                if support < min_wall_fraction:
                    rejected["centerline_wall_support"] += 1
                    continue
                allowed_ids = []
                for component_labels in (sato_components, attachment_components):
                    source_id = component_labels.endpoint_id(origin, spacing)
                    target_id = component_labels.endpoint_id(target, spacing)
                    allowed_ids.append(
                        None
                        if source_id is None or target_id is None
                        else {source_id, target_id}
                    )
                if any(ids is None for ids in allowed_ids):
                    rejected["clipped_bridge_does_not_connect_attachments"] += 1
                    continue
                if source_ellipse is None:
                    source_ellipse = _ellipse(sampler, origin, outgoing, spacing)
                axes0, radii0, fit0 = source_ellipse
                axes1, radii1, fit1 = _ellipse(sampler, target, incoming, spacing)
                # Huge sheet-derived ellipses are not safe bridge evidence.
                if max(max(radii0), max(radii1)) > max_gap:
                    rejected["oversized_endpoint_section"] += 1
                    continue
                voxels = _tube(
                    path,
                    tangents,
                    times,
                    axes0,
                    radii0,
                    axes1,
                    radii1,
                    spacing,
                    walls.shape,
                )
                inside = walls[tuple(voxels.T)] != 0
                fraction = float(inside.mean()) if len(inside) else 0.0
                if fraction < min_tube_wall_fraction:
                    rejected["tube_wall_support"] += 1
                    continue
                voxels = voxels[inside]
                if not len(voxels):
                    # Zero support thresholds are valid, but an empty clipped
                    # tube cannot connect either attachment.
                    rejected["clipped_bridge_does_not_connect_attachments"] += 1
                    continue
                low, high = voxels.min(0), voxels.max(0) + 1
                local = np.zeros(tuple(high - low), dtype=bool)
                local[tuple((voxels - low).T)] = True
                labels, _ = ndi.label(local, structure=np.ones((3, 3, 3)))
                contacts = attachment[tuple(voxels.T)] != 0
                ids = labels[tuple((voxels - low).T)]
                distance0 = np.linalg.norm((voxels - origin) * spacing, axis=1)
                distance1 = np.linalg.norm((voxels - target) * spacing, axis=1)
                first = (
                    contacts
                    & (distance0 <= min(max(radii0), 2 * spacing.max()) + spacing.max())
                    & (distance0 < distance1)
                )
                last = (
                    contacts
                    & (distance1 <= min(max(radii1), 2 * spacing.max()) + spacing.max())
                    & (distance1 < distance0)
                )
                common = set(ids[first]) & set(ids[last])
                if not common:
                    rejected["clipped_bridge_does_not_connect_attachments"] += 1
                    continue
                voxels = voxels[np.isin(ids, list(common))]
                # Topology is defined by VOLUME contact, including face/edge/corner
                # adjacency: a tube need not contain a third skeleton to merge it.
                contact_low, contact_high = voxels.min(0) - 1, voxels.max(0) + 2
                contact_region = np.zeros(tuple(contact_high - contact_low), dtype=bool)
                contact_region[tuple((voxels - contact_low).T)] = True
                contact_region = ndi.binary_dilation(
                    contact_region, structure=np.ones((3, 3, 3))
                )
                third_volume = False
                for name, component_labels, permitted in zip(
                    ("sato", "attachment"),
                    (sato_components, attachment_components),
                    allowed_ids,
                ):
                    if (
                        component_labels.touching_ids(contact_low, contact_region)
                        - permitted
                    ):
                        rejected["touches_third_" + name + "_component"] += 1
                        third_volume = True
                        break
                if third_volume:
                    continue
                # A previous accepted bridge may be the only visible connection
                # to a third merged component, so also inspect its actual voxels.
                touches_previous_third = False
                for (
                    previous_source,
                    previous_target,
                    previous_voxels,
                ) in accepted_voxels:
                    if nx.has_path(result, node, previous_source) or nx.has_path(
                        result, u, previous_source
                    ):
                        continue
                    if _sample(contact_region, previous_voxels - contact_low).any():
                        touches_previous_third = True
                        break
                if touches_previous_third:
                    rejected["touches_third_previous_bridge_component"] += 1
                    continue
                # The skeleton guard also catches graph components that occupy
                # an already merged input mask component.

                clipped = np.zeros(tuple(high - low), dtype=bool)
                clipped[tuple((voxels - low).T)] = True
                midpoint = (low + high - 1) * spacing / 2
                nearby_skeleton = tree.query_ball_point(
                    midpoint, np.linalg.norm((high - low) * spacing) / 2 + spacing.max()
                )
                third = [
                    i
                    for i in nearby_skeleton
                    if membership[records[i][0]] not in (rank, membership[u])
                ]
                if third and _sample(clipped, positions[third] - low).any():
                    rejected["touches_third_skeleton_component"] += 1
                    continue
                score = distance / max_gap + (1 - alignment) + 0.25 * (1 - support)
                record = {
                    "source_node": int(node)
                    if isinstance(node, (int, np.integer))
                    else str(node),
                    "source_component": rank,
                    "target_component": membership[u],
                    "start_zyx": origin.tolist(),
                    "end_zyx": target.tolist(),
                    "path_zyx": (path / spacing).tolist(),
                    "gap_length": distance,
                    "path_length": length,
                    "forward_cos": alignment,
                    "wall_fraction": support,
                    "tube_wall_fraction": fraction,
                    "source_ellipse": fit0,
                    "target_ellipse": fit1,
                    "bridge_voxels": int(len(voxels)),
                    "score": score,
                    "allowed_sato_component_ids": sorted(allowed_ids[0]),
                    "allowed_attachment_component_ids": sorted(allowed_ids[1]),
                }
                proposals.append((score, node, (u, v), target, voxels, record))
        proposals.sort(key=lambda p: p[0])
        if not proposals:
            continue
        best = proposals[0]
        # Equivalent nearby receiving samples are not distinct alternatives.
        rivals = [
            p
            for p in proposals[1:]
            if np.linalg.norm((p[3] - best[3]) * spacing) > 2 * spacing.max()
        ]
        if rivals and rivals[0][0] - best[0] < ambiguity_margin:
            rejected["ambiguous_attachment"] += 1
            continue
        _, source, edge, target, voxels, record = best
        if nx.has_path(result, source, edge[0]):
            rejected["already_connected"] += 1
            continue
        new_node = (
            next_node
            if next_node is not None
            else ("component_attachment", len(diagnostics["accepted"]))
        )
        target_node = _split_target(result, edge, target, new_node)
        if target_node == new_node and next_node is not None:
            next_node += 1
        result.add_edge(
            source,
            target_node,
            pts=np.asarray(record["path_zyx"]),
            weight=record["path_length"],
            component_bridge=True,
        )
        bridge_mask[tuple(voxels.T)] = 1
        diagnostics["accepted"].append(record)
        accepted_voxels.append((source, edge[0], voxels.copy()))
    diagnostics["rejections"] = dict(rejected)
    diagnostics["components_after"] = nx.number_connected_components(result)
    diagnostics["bridge_voxels"] = int(bridge_mask.sum())
    return result, bridge_mask, diagnostics
