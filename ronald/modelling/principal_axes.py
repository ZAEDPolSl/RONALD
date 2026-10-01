import numpy as np


class PrincipalAxes3D:
    """Minimal three-dimensional PCA used by branch modelling.

    This reproduces scikit-learn PCA's ``covariance_eigh`` path, including its
    deterministic component sign convention, without importing the full
    machine-learning stack for a fixed three-feature decomposition.
    """

    def fit(self, values):
        values = np.asarray(values)
        if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] < 3:
            raise ValueError("PCA requires an array with shape (n >= 3, 3)")

        sample_count = values.shape[0]
        self.mean_ = np.mean(values, axis=0).reshape(-1)

        covariance = values.T @ values
        covariance -= (
            sample_count * self.mean_.reshape(-1, 1) * self.mean_.reshape(1, -1)
        )
        covariance /= sample_count - 1

        _, eigenvectors = np.linalg.eigh(covariance)
        eigenvectors = np.flip(eigenvectors, axis=1)

        components = eigenvectors.T
        max_columns = np.argmax(np.abs(components), axis=1)
        signs = np.sign(components[np.arange(3), max_columns])
        components *= signs[:, None]
        self.components_ = np.array(components, copy=True)
        return self

    def reordered_for_direction(self, direction):
        """Return this fitted PCA with its most aligned axis first.

        The guide selects and orients an existing eigenvector; it never
        replaces that vector or rotates the PCA frame. Remaining axes keep
        their original relative order and sign. The source fit is unchanged.
        Coordinates in this frame can be obtained from source coordinates as
        ``coordinates[:, result.axis_order_] * result.axis_signs_``.
        """
        direction = np.asarray(direction, dtype=float)
        if direction.shape != (3,) or not np.all(np.isfinite(direction)):
            raise ValueError("direction must be a finite, nonzero three-vector")
        scale = np.max(np.abs(direction))
        if scale == 0:
            raise ValueError("direction must be a finite, nonzero three-vector")
        # Scale first so both very small and very large finite guides work.
        unit_direction = direction / scale
        unit_direction /= np.linalg.norm(unit_direction)
        alignment = self.components_ @ unit_direction
        selected = int(np.argmax(np.abs(alignment)))
        order = np.array([selected] + [axis for axis in range(3) if axis != selected])
        signs = np.ones(3)
        if alignment[selected] < 0:
            signs[0] = -1.0

        result = type(self)()
        result.mean_ = self.mean_.copy()
        result.components_ = self.components_[order].copy() * signs[:, None]
        result.axis_order_ = order
        result.axis_signs_ = signs
        return result

    def transform(self, values):
        values = np.asarray(values)
        transformed = values @ self.components_.T
        transformed -= self.mean_.reshape(1, -1) @ self.components_.T
        return transformed

    def inverse_transform(self, values):
        values = np.asarray(values)
        return values @ self.components_ + self.mean_
