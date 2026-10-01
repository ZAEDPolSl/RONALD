"""Fixed-center Sato enclosure with proportional growth to local wall support.

Both inputs are already-local cross-sections in the caller's PCA frame, whose
coordinate zero is longitudinal. The wall convex hull bounds expansion; when
a sampling pitch is supplied, foreground-grid support also bounds every added
annulus site. Callers must still clip the resulting tube to the actual wall mask.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, QhullError


def fit_sato_guided_ellipse(
    sato_points, walls_points, endpoint, *, wall_sample_pitch=None
):
    """Return ``(ellipse_or_None, JSON-safe diagnostics)`` without moving center.

    Maximize log(det(Q)) subject to p.T @ Q @ p <= 1 for every Sato point.
    This gives the minimum-area centered seed ellipse, including outliers and
    attached sheets. Three Cholesky parameters and analytic Jacobians keep
    precision Q positive definite in a bounded SLSQP solve. Hull vertices are
    sufficient constraints because a positive quadratic is convex.

    Preserve seed orientation/aspect and take the largest uniform expansion
    contained in the local wall convex hull. Never shrink a seed that crosses
    the walls. Unusable walls retain the seed, with an explicit diagnostic.
    If wall_sample_pitch is supplied, walls_points must contain all foreground
    sites of the regular transverse sampling grid centered on endpoint. Inspect
    every site in the proposed added annulus, including internal holes, and stop
    before the first background site. This is sampled containment: sub-grid gaps
    below the supplied pitch are unresolved, and seed/background overlap is
    retained for final clipping. Missing or degenerate Sato support fails; there
    is no free-center fallback.
    """
    diagnostic = dict(
        status="fallback",
        reason=None,
        center=None,
        input_count=0,
        unique_count=0,
        constraint_count=0,
        iterations=0,
        solver_converged=False,
        radii=None,
        seed_radii=None,
        expansion_factor=1.0,
        no_expansion_reason=None,
        clipping_needed=None,
        sato_max_quadratic=None,
        wall_hull_planes=0,
        convex_envelope_only=wall_sample_pitch is None,
    )
    if wall_sample_pitch is not None:
        if not np.isfinite(wall_sample_pitch) or wall_sample_pitch <= 0:
            raise ValueError("wall_sample_pitch must be positive and finite")

    def failure(reason):
        diagnostic["reason"] = reason
        return None, diagnostic

    try:
        sato = np.asarray(sato_points, dtype=float)
        center = np.asarray(endpoint, dtype=float)
    except (TypeError, ValueError):
        return failure("invalid_input")
    if sato.ndim != 2 or sato.shape[1] != 3 or center.shape != (3,):
        return failure("invalid_shape")
    diagnostic["input_count"] = len(sato)
    if not np.all(np.isfinite(sato)) or not np.all(np.isfinite(center)):
        return failure("nonfinite_input")
    diagnostic["center"] = center.tolist()
    local = np.unique(sato[:, 1:], axis=0) - center[1:]
    diagnostic["unique_count"] = len(local)
    if len(local) < 3:
        return failure("insufficient_sato_points")
    scale = float(np.max(np.abs(local)))
    if not np.isfinite(scale) or scale <= 0:
        return failure("degenerate_sato_support")
    normalized = local / scale
    try:
        hull = ConvexHull(normalized)
    except QhullError:
        return failure("degenerate_sato_support")
    vertices = normalized[hull.vertices]
    diagnostic["constraint_count"] = len(vertices)
    # Whiten the centered second moment to condition very elongated sections.
    # This invertible change of coordinates adds only a constant to log(det(Q)).
    try:
        _, singular, directions = np.linalg.svd(vertices, full_matrices=False)
    except np.linalg.LinAlgError:
        return failure("linear_algebra_error")
    if singular.min() <= 0:
        return failure("degenerate_sato_support")
    whitening = directions.T * (np.sqrt(len(vertices)) / singular)
    vertices = vertices @ whitening
    x, y = vertices.T

    def factor(parameters):
        with np.errstate(over="ignore", invalid="ignore"):
            return np.exp(parameters[0]), parameters[1], np.exp(parameters[2])

    def constraints(parameters):
        a, b, c = factor(parameters)
        with np.errstate(over="ignore", invalid="ignore"):
            return 1 - (a * x + b * y) ** 2 - (c * y) ** 2

    def jacobian(parameters):
        a, b, c = factor(parameters)
        term = a * x + b * y
        with np.errstate(over="ignore", invalid="ignore"):
            return -2 * np.column_stack((term * a * x, term * y, (c * y) ** 2))

    initial = -np.log(np.max(np.linalg.norm(vertices, axis=1)) * 1.01)
    try:
        result = minimize(
            lambda p: -2 * (p[0] + p[2]),
            [initial, 0.0, initial],
            jac=lambda p: np.array([-2.0, 0.0, -2.0]),
            method="SLSQP",
            constraints=dict(type="ineq", fun=constraints, jac=jacobian),
            options=dict(ftol=1e-10, maxiter=100),
        )
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return failure("optimizer_error")
    diagnostic.update(
        iterations=int(result.nit),
        solver_converged=bool(result.success),
        optimizer_message=str(result.message),
    )
    if not result.success:
        return failure("optimizer_did_not_converge")
    a, b, c = factor(result.x)
    lower = np.array([[a, 0.0], [b, c]])
    precision = whitening @ (lower @ lower.T) @ whitening.T
    quadratic = np.einsum("ij,jk,ik->i", normalized, precision, normalized)
    if not np.all(np.isfinite(precision)) or not np.all(np.isfinite(quadratic)):
        return failure("nonfinite_fit")
    max_quadratic = float(np.max(quadratic))
    diagnostic["optimizer_max_quadratic"] = max_quadratic
    if max_quadratic > 1 + 1e-7:
        return failure("sato_enclosure_violation")
    # Correct only floating point solver tolerance; discard no Sato points.
    if max_quadratic > 1:
        precision /= max_quadratic * (1 + 4 * np.finfo(float).eps)
    try:
        values, axes = np.linalg.eigh(precision)
    except np.linalg.LinAlgError:
        return failure("linear_algebra_error")
    if values.min() <= 0 or not np.all(np.isfinite(values)):
        return failure("degenerate_ellipse")
    normalized_radii = 1 / np.sqrt(values)
    seed_radii = normalized_radii * scale
    shape = (axes * normalized_radii**2) @ axes.T
    diagnostic["seed_sato_max_quadratic"] = float(
        np.max(np.einsum("ij,jk,ik->i", normalized, precision, normalized))
    )

    expansion = 1.0
    wall_reason = None
    try:
        walls = np.asarray(walls_points, dtype=float)
    except (TypeError, ValueError):
        walls = np.empty((0, 3))
        wall_reason = "invalid_walls_input"
    if wall_reason is None:
        if walls_points is None or walls.size == 0:
            wall_reason = "walls_unavailable"
        elif walls.ndim != 2 or walls.shape[1] != 3:
            wall_reason = "invalid_walls_shape"
        elif not np.all(np.isfinite(walls)):
            wall_reason = "nonfinite_walls_input"
        else:
            wall_local = np.unique(walls[:, 1:], axis=0) - center[1:]
            if len(wall_local) < 3:
                wall_reason = "degenerate_walls_support"
            else:
                try:
                    wall_hull = ConvexHull(wall_local / scale)
                except QhullError:
                    wall_reason = "degenerate_walls_support"
                else:
                    normals = wall_hull.equations[:, :2]
                    distances = -wall_hull.equations[:, 2]
                    diagnostic["wall_hull_planes"] = len(distances)
                    if np.any(distances <= 0):
                        wall_reason = "center_not_strictly_inside_wall_hull"
                        diagnostic["clipping_needed"] = True
                    else:
                        support = np.sqrt(
                            np.einsum("ij,jk,ik->i", normals, shape, normals)
                        )
                        wall_limit = float(np.min(distances / support))
                        if not np.isfinite(wall_limit):
                            wall_reason = "nonfinite_wall_expansion"
                        else:
                            diagnostic["wall_supported_scale"] = wall_limit
                            diagnostic["clipping_needed"] = bool(wall_limit < 1.0)
                            expansion = max(1.0, wall_limit)
                            if wall_limit < 1.0:
                                wall_reason = "sato_seed_crosses_wall_hull"
                            elif wall_limit == 1.0:
                                wall_reason = "seed_already_at_wall_limit"
    if wall_sample_pitch is not None:
        diagnostic["actual_wall_support"] = dict(
            status="no_growth_proposed",
            pitch=float(wall_sample_pitch),
            sampling_tolerance=float(wall_sample_pitch / np.sqrt(2.0)),
            hull_expansion_factor=float(expansion),
        )
        if expansion > 1.0:
            expansion, actual = _sampled_wall_expansion(
                wall_local, precision / scale**2, expansion, float(wall_sample_pitch)
            )
            diagnostic["actual_wall_support"].update(actual)
            if actual["status"] != "accepted" or actual.get("limited", False):
                wall_reason = actual["reason"]
    radii = seed_radii * expansion
    ellipse = (
        center.copy(),
        np.r_[0.0, axes[:, 0] * radii[0]],
        np.r_[0.0, axes[:, 1] * radii[1]],
    )
    if not np.all(np.isfinite(np.asarray(ellipse))) or np.any(radii <= 0):
        return failure("invalid_ellipse")
    diagnostic.update(
        status="sato_guided",
        reason="accepted",
        radii=radii.tolist(),
        seed_radii=seed_radii.tolist(),
        expansion_factor=float(expansion),
        no_expansion_reason=wall_reason,
        sato_max_quadratic=diagnostic["seed_sato_max_quadratic"] / expansion**2,
        area=float(np.pi * np.prod(radii)),
    )
    return ellipse, diagnostic


def _sampled_wall_expansion(wall_local, seed_precision, hull_expansion, pitch):
    """Largest growth before a missing sample, never removing the seed.

    All raster sites in the growth annulus are tested, rather than just the
    final ellipse outline. Sample-center occupancy leaves at most half a cell
    diagonal of spatial discretization uncertainty. The existing convex-hull
    bound is retained as an additional outer limit.
    """
    grid = wall_local / pitch
    rounded = np.rint(grid)
    if not np.allclose(grid, rounded, atol=1e-6, rtol=0):
        return 1.0, dict(
            status="invalid_grid", reason="walls_not_on_sampling_grid", limited=True
        )
    indices = rounded.astype(np.int64)
    shape = np.linalg.inv(seed_precision)
    bounds = (
        np.ceil(hull_expansion * np.sqrt(np.diag(shape)) / pitch).astype(np.int64) + 1
    )
    dimensions = 2 * bounds + 1
    if np.prod(dimensions, dtype=float) > 4_000_000:
        return 1.0, dict(
            status="grid_limit", reason="wall_support_grid_limit", limited=True
        )
    supported = np.zeros(tuple(dimensions), dtype=bool)
    keep = np.all((indices >= -bounds) & (indices <= bounds), axis=1)
    supported[tuple((indices[keep] + bounds).T)] = True
    u = np.arange(-bounds[0], bounds[0] + 1) * pitch
    v = np.arange(-bounds[1], bounds[1] + 1) * pitch
    quadratic = (
        seed_precision[0, 0] * u[:, None] ** 2
        + 2 * seed_precision[0, 1] * u[:, None] * v[None, :]
        + seed_precision[1, 1] * v[None, :] ** 2
    )
    # Treat numerical error at the protected seed outline as part of the seed.
    annulus = (quadratic > 1.0 + 1e-8) & (quadratic <= hull_expansion**2 + 1e-8)
    missing = annulus & ~supported
    count = int(missing.sum())
    diagnostic = dict(
        status="accepted",
        reason="sampled_walls_support_growth",
        tested_annulus_sites=int(annulus.sum()),
        unsupported_annulus_sites=count,
        unsupported_seed_sites=int(((quadratic <= 1.0 + 1e-8) & ~supported).sum()),
        limited=bool(count),
    )
    expansion = hull_expansion
    if count:
        first = float(np.min(quadratic[missing]))
        expansion = max(1.0, min(hull_expansion, np.sqrt(first) * (1.0 - 1e-9)))
        diagnostic.update(
            reason="actual_wall_background_limits_growth",
            first_unsupported_scale=float(np.sqrt(first)),
        )
    diagnostic["supported_expansion_factor"] = float(expansion)
    return expansion, diagnostic
