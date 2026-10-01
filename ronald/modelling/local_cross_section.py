"""Dense local mask sections, independent of graph-branch ownership labels."""

import numpy as np
from scipy import ndimage as ndi


_BOUNDARY_AXES = tuple(
    np.array([(bits >> i) & 1 for i in range(3)], dtype=bool) for bits in range(1, 8)
)
for _axes in _BOUNDARY_AXES:
    _axes.setflags(write=False)


class LocalCrossSectionSampler:
    """Sample a center-connected short slab in an existing orthonormal PCA frame.

    Coordinates are z,y,x voxel coordinates, matching branch modelling. A
    half-voxel transverse grid supplies dense support without interpolating
    between branch-owned points. Nearest-neighbor sampling treats each source
    voxel as a unit cell. The three slab planes are projected onto the center
    plane before selecting its 8-connected foreground component. This is a slab
    projection, not exact containment in the single center plane.
    """

    def __init__(self, mask, *, pitch=0.5, initial_half_size=8.0, max_half_size=128.0):
        self.mask = np.asarray(mask)
        if self.mask.ndim != 3 or any(n == 0 for n in self.mask.shape):
            raise ValueError("mask must be a nonempty 3D array")
        if not np.isfinite(pitch) or pitch <= 0:
            raise ValueError("pitch must be positive and finite")
        if (
            not np.isfinite(initial_half_size)
            or not np.isfinite(max_half_size)
            or initial_half_size <= 0
            or max_half_size < initial_half_size
        ):
            raise ValueError("invalid sampling patch sizes")
        self.pitch = float(pitch)
        self.initial_half_size = float(initial_half_size)
        self.max_half_size = float(max_half_size)
        # Coordinates only: occupancy must be sampled again if the mask changes.
        self._grid_cache = {}

    def sample(self, center_zyx, frame_rows, half_width=0.5):
        center = np.asarray(center_zyx, dtype=float)
        frame = np.asarray(frame_rows, dtype=float)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError("center must be a finite 3-vector")
        if (
            frame.shape != (3, 3)
            or not np.isfinite(frame).all()
            or not np.allclose(frame @ frame.T, np.eye(3), atol=1e-7, rtol=0)
        ):
            raise ValueError("frame rows must be orthonormal")
        if not np.isfinite(half_width) or half_width < 0:
            raise ValueError("half_width must be finite and nonnegative")
        offsets = [0.0] if half_width == 0 else [-half_width, 0.0, half_width]
        diagnostic = dict(
            status="failed",
            reason=None,
            pitch=self.pitch,
            slab_half_width=float(half_width),
            plane_offsets=offsets,
            center_zyx=center.tolist(),
            expansions=0,
        )
        size = self.initial_half_size
        shape = np.asarray(self.mask.shape)
        while True:
            steps = int(np.ceil(size / self.pitch))
            key = (steps, self.pitch)
            grid = self._grid_cache.get(key)
            if grid is None:
                axis = np.arange(-steps, steps + 1, dtype=float) * self.pitch
                u, v = np.meshgrid(axis, axis, indexing="ij")
                for coordinates in (axis, u, v):
                    coordinates.setflags(write=False)
                grid = self._grid_cache[key] = (axis, u, v)
            axis, u, v = grid
            plane = (
                center[:, None, None]
                + frame[1, :, None, None] * u
                + frame[2, :, None, None] * v
            )
            foreground = np.zeros(u.shape, dtype=bool)
            for offset in offsets:
                # Use closed voxel cells on exact cell faces. A point on a
                # foreground/background interface belongs to the foreground
                # boundary even when nearest-neighbor tie-breaking picks the
                # background voxel. This avoids one-sided support at grid ties.
                shifted = plane + offset * frame[0, :, None, None] + 0.5
                nearest_integer = np.rint(shifted)
                ties = np.abs(shifted - nearest_integer) <= 1e-9
                indices = np.floor(shifted).astype(np.intp)
                indices[ties] = nearest_integer[ties].astype(np.intp)
                valid = np.all(
                    (indices >= 0) & (indices < shape[:, None, None]), axis=0
                )
                foreground[valid] |= self.mask[tuple(indices[:, valid])] != 0
                # Exact voxel-face ties are sparse for oblique sections. Avoid
                # scanning the whole plane seven times for alternate cell owners.
                boundary_sites = np.flatnonzero(ties.any(axis=0))
                if not len(boundary_sites):
                    continue
                boundary_ties = ties.reshape(3, -1)[:, boundary_sites]
                boundary_indices = indices.reshape(3, -1)[:, boundary_sites]
                for axes in _BOUNDARY_AXES:
                    on_boundary = np.all(boundary_ties[axes], axis=0)
                    if not on_boundary.any():
                        continue
                    alternate = boundary_indices[:, on_boundary] - axes[:, None]
                    inside = np.all(
                        (alternate >= 0) & (alternate < shape[:, None]), axis=0
                    )
                    values = np.zeros(int(on_boundary.sum()), dtype=bool)
                    values[inside] = self.mask[tuple(alternate[:, inside])] != 0
                    foreground.ravel()[boundary_sites[on_boundary]] |= values
            diagnostic.update(
                patch_half_size=float(axis[-1]), foreground_points=int(foreground.sum())
            )
            if not foreground[steps, steps]:
                diagnostic["reason"] = "center_not_in_slab_foreground"
                return np.empty((0, 3)), diagnostic
            labels, _ = ndi.label(foreground, structure=np.ones((3, 3), dtype=np.uint8))
            selected = labels == labels[steps, steps]
            touches = bool(
                selected[0].any()
                or selected[-1].any()
                or selected[:, 0].any()
                or selected[:, -1].any()
            )
            diagnostic.update(
                component_points=int(selected.sum()), touches_patch_edge=touches
            )
            if not touches:
                diagnostic.update(status="accepted", reason="accepted")
                return plane[:, selected].T.copy(), diagnostic
            if size >= self.max_half_size:
                diagnostic["reason"] = "component_exceeds_patch_limit"
                return np.empty((0, 3)), diagnostic
            size = min(2 * size, self.max_half_size)
            diagnostic["expansions"] += 1
