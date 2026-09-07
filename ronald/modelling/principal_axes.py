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

    def transform(self, values):
        values = np.asarray(values)
        transformed = values @ self.components_.T
        transformed -= self.mean_.reshape(1, -1) @ self.components_.T
        return transformed

    def inverse_transform(self, values):
        values = np.asarray(values)
        return values @ self.components_ + self.mean_
