import numpy as np
from ronald.modelling.ellipse_recovery import ellipse_is_valid


def test_validity_rejects_line_support_and_nonfinite_geometry():
    e = (np.zeros(3), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    line = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]])
    assert ellipse_is_valid(e)
    assert not ellipse_is_valid(e, line)
    assert not ellipse_is_valid((np.full(3, np.nan), e[1], e[2]))
