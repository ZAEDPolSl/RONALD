"""The growth check sees holes/concavities, preserving the seed and its shape."""

import json
import numpy as np
import pytest

from ronald.modelling.sato_envelope import fit_sato_guided_ellipse


def ring(a=1.0, b=1.0, rotation=0.0):
    t = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    xy = np.column_stack((a * np.cos(t), b * np.sin(t)))
    rot = np.array(
        [[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]]
    )
    return np.column_stack((np.zeros(len(t)), xy @ rot.T))


def wall_grid(radius=5.0, pitch=0.5, remove=None):
    axis = np.arange(-radius, radius + pitch / 2, pitch)
    u, v = np.meshgrid(axis, axis, indexing="ij")
    keep = np.ones(u.shape, bool)
    if remove:
        keep &= ~remove(u, v)
    return np.column_stack((np.zeros(keep.sum()), u[keep], v[keep]))


def test_internal_hole_blocks_growth_even_when_final_outline_is_supported():
    # The final radius-5 outline avoids this hole; checking just that outline fails.
    walls = wall_grid(remove=lambda u, v: (u == 2.0) & (v == 0.0))
    _, old = fit_sato_guided_ellipse(ring(), walls, np.zeros(3))
    ellipse, new = fit_sato_guided_ellipse(
        ring(), walls, np.zeros(3), wall_sample_pitch=0.5
    )
    assert old["expansion_factor"] == pytest.approx(5.0, rel=1e-6)
    assert new["expansion_factor"] == pytest.approx(2.0, rel=1e-6)
    assert new["expansion_factor"] < 2.0
    assert new["actual_wall_support"]["unsupported_annulus_sites"] == 1
    assert new["actual_wall_support"]["limited"]
    assert new["no_expansion_reason"] == "actual_wall_background_limits_growth"
    assert ellipse is not None
    json.dumps(new, allow_nan=False)


def test_concavity_limits_growth_without_changing_sato_proportions():
    walls = wall_grid(remove=lambda u, v: (u >= 2.0) & (np.abs(v) <= 0.5))
    sato = ring(1.5, 0.75, 0.4)
    ellipse, result = fit_sato_guided_ellipse(
        sato, walls, np.zeros(3), wall_sample_pitch=0.5
    )
    assert result["actual_wall_support"]["limited"]
    assert result["radii"][0] / result["radii"][1] == pytest.approx(2.0, rel=1e-6)
    vectors = np.column_stack([v[1:] for v in ellipse[1:]])
    inverse = np.linalg.inv(vectors @ vectors.T)
    assert (
        np.max(np.einsum("ij,jk,ik->i", sato[:, 1:], inverse, sato[:, 1:]))
        <= 1.0 + 1e-8
    )


def test_background_already_in_seed_never_shrinks_it():
    walls = wall_grid(remove=lambda u, v: (u >= 1.0) & (np.abs(v) <= 0.5))
    _, result = fit_sato_guided_ellipse(
        ring(2.0, 2.0), walls, np.zeros(3), wall_sample_pitch=0.5
    )
    assert min(result["radii"]) >= 2.0 - 1e-7
    assert 1.0 <= result["expansion_factor"] < 1.3
    assert result["actual_wall_support"]["unsupported_seed_sites"] > 0


def test_full_support_matches_legacy_and_does_not_mutate_inputs():
    walls, sato, center = wall_grid(), ring(), np.zeros(3)
    saved = walls.copy()
    _, old = fit_sato_guided_ellipse(sato, walls, center)
    _, new = fit_sato_guided_ellipse(sato, walls, center, wall_sample_pitch=0.5)
    assert new["radii"] == pytest.approx(old["radii"])
    assert not new["actual_wall_support"]["limited"]
    assert np.array_equal(saved, walls)


def test_translated_grid_and_center():
    offset = np.array([19.0, 2.13, -7.87])
    walls = wall_grid(remove=lambda u, v: (u == 2.0) & (v == 0.0)) + offset
    _, result = fit_sato_guided_ellipse(
        ring() + offset, walls, offset, wall_sample_pitch=0.5
    )
    assert result["expansion_factor"] == pytest.approx(2.0, rel=1e-6)


def test_non_grid_wall_points_conservatively_keep_seed():
    _, result = fit_sato_guided_ellipse(
        ring(), ring(5.0, 5.0), np.zeros(3), wall_sample_pitch=0.5
    )
    assert result["expansion_factor"] == 1.0
    assert result["no_expansion_reason"] == "walls_not_on_sampling_grid"


@pytest.mark.parametrize("pitch", [0.0, -1.0, np.nan, np.inf])
def test_invalid_pitch_rejected(pitch):
    with pytest.raises(ValueError, match="wall_sample_pitch"):
        fit_sato_guided_ellipse(
            ring(), wall_grid(), np.zeros(3), wall_sample_pitch=pitch
        )
