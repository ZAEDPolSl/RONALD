import networkx as nx
import numpy as np
from ronald.modelling.component_connections import connect_sato_components


def fixture(offset=0):
    shape = (25, 25, 35)
    zz, yy, xx = np.indices(shape)
    sato = np.zeros(shape, dtype=np.uint8)
    graph = nx.Graph()
    # Main tree is longer; endpoint of detached fragment points toward it.
    for u, v, start, end in [
        (0, 1, (12, 12, 2), (12, 12, 15)),
        (2, 3, (12, 12, 20), (12, 12, 27)),
    ]:
        points = np.linspace(
            start, end, int(np.linalg.norm(np.subtract(end, start))) + 1
        )
        graph.add_node(u, o=points[0], pts=points[:1])
        graph.add_node(v, o=points[-1], pts=points[-1:])
        graph.add_edge(u, v, pts=points, weight=len(points) - 1)
        sato[
            ((zz - 12) ** 2 + (yy - 12) ** 2 <= 4) & (xx >= start[2]) & (xx <= end[2])
        ] = 1
    walls = (((zz - 12) ** 2 + (yy - 12) ** 2 <= 9) & (xx >= 2) & (xx <= 27)).astype(
        np.uint8
    )
    return graph, walls, sato


def test_connects_supported_gap_and_preserves_inputs():
    graph, walls, sato = fixture()
    copy = sato.copy()
    result, bridges, diag = connect_sato_components(graph, walls, sato)
    assert len(diag["accepted"]) == 1
    assert nx.is_tree(result)
    assert nx.number_connected_components(graph) == 2
    assert np.array_equal(sato, copy)
    assert not np.any(bridges & (walls == 0))
    assert bridges[12, 12, 18]
    assert diag["accepted"][0]["source_ellipse"]["method"] == "sato_ellipse"


def test_rejects_background_gap():
    graph, walls, sato = fixture()
    walls[:, :, 16:20] = 0
    result, bridges, diag = connect_sato_components(graph, walls, sato)
    assert not bridges.any()
    assert nx.number_connected_components(result) == 2
    assert diag["rejections"]["centerline_wall_support"]


def test_endpoint_must_survive_in_attachment_mask():
    graph, walls, sato = fixture()
    attachment = sato.copy()
    attachment[:, :, 20:] = 0
    _, bridges, diag = connect_sato_components(
        graph, walls, sato, attachment_mask=attachment
    )
    assert not bridges.any()
    assert diag["rejections"]["clipped_bridge_does_not_connect_attachments"]


def test_no_connection_within_single_tree():
    graph, walls, sato = fixture()
    graph.add_edge(1, 2, pts=np.array([[12, 12, 15], [12, 12, 20]]))
    result, bridges, diag = connect_sato_components(graph, walls, sato)
    assert not bridges.any()
    assert result.number_of_edges() == graph.number_of_edges()


def test_physical_spacing_controls_distance():
    graph, walls, sato = fixture()
    _, bridges, diag = connect_sato_components(
        graph, walls, sato, spacing_zyx=(1, 1, 2), max_gap=8
    )
    assert not bridges.any()


def test_rejects_sideways_nearest_branch():
    graph, walls, sato = fixture()
    # source points away from the nearby tree instead of continuing toward it
    graph.nodes[2]["o"] = np.array([12, 19, 20.0])
    graph.nodes[3]["o"] = np.array([12, 12, 20.0])
    graph.edges[2, 3]["pts"] = np.array([[12, y, 20] for y in range(19, 11, -1)])
    _, bridges, _ = connect_sato_components(graph, walls, sato)
    assert not bridges.any()


def test_attaches_to_edge_interior_and_splits_it():
    shape = (31, 31, 31)
    z, y, x = np.indices(shape)
    graph = nx.Graph()
    for u, v, points in [
        (0, 1, np.array([[15, y, 15] for y in range(3, 28)])),
        (2, 3, np.array([[15, 15, x] for x in range(20, 27)])),
    ]:
        graph.add_node(u, o=points[0])
        graph.add_node(v, o=points[-1])
        graph.add_edge(u, v, pts=points)
    sato = (
        (((z - 15) ** 2 + (x - 15) ** 2 <= 4) & (y >= 3) & (y <= 27))
        | (((z - 15) ** 2 + (y - 15) ** 2 <= 4) & (x >= 20) & (x <= 26))
    ).astype(np.uint8)
    walls = (
        (((z - 15) ** 2 + (x - 15) ** 2 <= 9) & (y >= 3) & (y <= 27))
        | (((z - 15) ** 2 + (y - 15) ** 2 <= 9) & (x >= 15) & (x <= 26))
    ).astype(np.uint8)
    result, bridges, diag = connect_sato_components(
        graph, walls, sato, min_tube_wall_fraction=0.5
    )
    assert len(diag["accepted"]) == 1
    assert len(result) == 5
    assert nx.is_tree(result)
    assert bridges.any()


def test_rejects_ambiguous_receiving_branches():
    graph, walls, sato = fixture()
    # Two equally plausible larger receivers in the same tree, 4 voxels apart.
    for node in (0, 1):
        graph.nodes[node]["o"] = graph.nodes[node]["o"] + np.array([0, 2, 0])
    graph.edges[0, 1]["pts"] = graph.edges[0, 1]["pts"] + np.array([0, 2, 0])
    points = np.array([[12, 10, x] for x in range(2, 16)])
    graph.add_node(4, o=points[0])
    graph.add_node(5, o=points[-1])
    graph.add_edge(4, 5, pts=points)
    graph.add_edge(0, 4, pts=np.array([[12, 14, 2], [12, 10, 2]]))
    z, y, x = np.indices(sato.shape)
    sato = (
        (((z - 12) ** 2 + (y - 14) ** 2 <= 2) & (x >= 2) & (x <= 15))
        | (((z - 12) ** 2 + (y - 10) ** 2 <= 2) & (x >= 2) & (x <= 15))
        | (((z - 12) ** 2 + (y - 12) ** 2 <= 2) & (x >= 20) & (x <= 27))
    ).astype(np.uint8)
    # Match the joining graph edge with a real mask connection so the two
    # alternatives belong to the same receiving volume component.
    sato[12, 10:15, 2] = 1
    walls = np.ones(sato.shape, dtype=np.uint8)
    _, bridges, diag = connect_sato_components(graph, walls, sato)
    assert not bridges.any()
    assert diag["rejections"]["ambiguous_attachment"]


def test_clipping_must_leave_a_continuous_bridge():
    graph, walls, sato = fixture()
    walls[:, :, 18] = 0
    _, bridges, diag = connect_sato_components(
        graph, walls, sato, min_wall_fraction=0.5, min_tube_wall_fraction=0.5
    )
    assert not bridges.any()
    assert diag["rejections"]["clipped_bridge_does_not_connect_attachments"]


def _third_volume_fixture(y=16):
    graph, walls, sato = fixture()
    points = np.array([[12, y, x] for x in range(17, 19)])
    graph.add_node(4, o=points[0])
    graph.add_node(5, o=points[-1])
    graph.add_edge(4, 5, pts=points)
    z, yy, x = np.indices(sato.shape)
    third = ((z - 12) ** 2 + (yy - y) ** 2 <= 4) & (x >= 17) & (x <= 18)
    sato[third] = 1
    walls[:] = 1
    return graph, walls, sato


def test_rejects_touching_third_sato_volume_without_touching_its_skeleton():
    graph, walls, sato = _third_volume_fixture(16)
    # Third skeleton is four voxels away; its radius-2 surface meets the bridge.
    _, bridges, diag = connect_sato_components(graph, walls, sato)
    assert diag["rejections"]["touches_third_sato_component"]
    assert not any(r["source_component"] == 1 for r in diag["accepted"])
    # The short third branch might independently be accepted elsewhere, but
    # it must not be swallowed by the proposed main gap bridge.
    assert not bridges[12, 12, 18]


def test_rejects_26_neighbor_tangent_contact_with_third_volume():
    graph, walls, sato = _third_volume_fixture(17)
    _, bridges, diag = connect_sato_components(graph, walls, sato)
    assert diag["rejections"]["touches_third_sato_component"]
    assert not bridges[12, 12, 18]


def test_attachment_volume_guard_applies_when_not_present_in_sato():
    graph, walls, sato = fixture()
    z, y, x = np.indices(sato.shape)
    attachment = sato.copy()
    attachment[((z - 12) ** 2 + (y - 16) ** 2 <= 4) & (x >= 17) & (x <= 18)] = 1
    walls[:] = 1
    _, bridges, diag = connect_sato_components(
        graph, walls, sato, attachment_mask=attachment
    )
    assert not bridges.any()
    assert diag["rejections"]["touches_third_attachment_component"]


def test_sato_guard_still_applies_when_attachment_already_connected():
    graph, walls, sato = _third_volume_fixture(16)
    _, bridges, diag = connect_sato_components(
        graph, walls, sato, attachment_mask=np.ones(sato.shape, dtype=np.uint8)
    )
    assert diag["rejections"]["touches_third_sato_component"]
    assert not bridges[12, 12, 18]


def test_background_endpoint_id_does_not_authorize_multiple_nearby_components():
    from ronald.modelling.component_connections import _MaskComponents

    mask = np.zeros((7, 7, 7), dtype=np.uint8)
    mask[3, 2, 3] = 1
    mask[3, 4, 3] = 1
    labels = _MaskComponents(mask)
    assert labels.endpoint_id(np.array([3.0, 3.0, 3.0]), np.ones(3)) is None
    assert (
        labels.endpoint_id(np.array([3.0, 2.1, 3.0]), np.ones(3))
        == labels.values([[3, 2, 3]])[0]
    )


def test_rejects_contact_with_an_unrelated_previously_accepted_bridge():
    # Two straight gaps cross, but none of the original four foreground arms
    # reaches the crossing. Original-mask CC checks alone cannot catch this.
    graph = nx.Graph()
    sato = np.zeros((25, 31, 31), dtype=np.uint8)
    arrays = [
        np.array([[12, 16, x] for x in range(1, 13)]),
        np.array([[12, y, 16] for y in range(1, 13)]),
        np.array([[12, 16, x] for x in range(20, 28)]),
        np.array([[12, y, 16] for y in range(20, 28)]),
    ]
    for index, points in enumerate(arrays):
        u, v = 2 * index, 2 * index + 1
        graph.add_node(u, o=points[0])
        graph.add_node(v, o=points[-1])
        graph.add_edge(u, v, pts=points)
        sato[tuple(points.T)] = 1
    _, bridges, diag = connect_sato_components(
        graph, np.ones_like(sato), sato, max_gap=9.0, min_forward_cos=0.9
    )
    assert len(diag["accepted"]) == 1
    assert diag["rejections"]["touches_third_previous_bridge_component"]


def test_zero_support_thresholds_reject_empty_clipped_bridge():
    graph, walls, sato = fixture()
    walls[:] = 0
    result, bridges, diagnostic = connect_sato_components(
        graph, walls, sato, min_wall_fraction=0.0, min_tube_wall_fraction=0.0
    )
    assert not bridges.any()
    assert nx.number_connected_components(result) == 2
    assert diagnostic["rejections"]["clipped_bridge_does_not_connect_attachments"]


def test_source_ellipse_reused_for_multiple_targets_and_reset_per_call(monkeypatch):
    import ronald.modelling.component_connections as connections

    graph, walls, sato = fixture()
    # Split the receiving branch into two edges: the detached endpoint must
    # evaluate both x=15 and x=12, using the same source section at x=20.
    points = graph.edges[0, 1]["pts"]
    graph.remove_edge(0, 1)
    graph.add_node(4, o=points[10], pts=points[10:11])
    graph.add_edge(0, 4, pts=points[:11])
    graph.add_edge(4, 1, pts=points[10:])
    original = connections._ellipse
    calls = []

    def recording_ellipse(sampler, center, tangent, spacing):
        calls.append(tuple(center))
        return original(sampler, center, tangent, spacing)

    monkeypatch.setattr(connections, "_ellipse", recording_ellipse)
    result, bridges, diagnostic = connections.connect_sato_components(
        graph, walls, sato
    )
    assert len(diagnostic["accepted"]) == 1
    assert nx.is_tree(result)
    assert calls.count((12.0, 12.0, 20.0)) == 1
    assert calls.count((12.0, 12.0, 15.0)) == 1
    assert calls.count((12.0, 12.0, 12.0)) == 1
    # Cache lifetime is one endpoint within one invocation, never stale across
    # masks or repeated runs; mask and serialized diagnostics remain identical.
    _, repeated, repeated_diagnostic = connections.connect_sato_components(
        graph, walls, sato
    )
    assert calls.count((12.0, 12.0, 20.0)) == 2
    np.testing.assert_array_equal(repeated, bridges)
    assert repeated_diagnostic == diagnostic
