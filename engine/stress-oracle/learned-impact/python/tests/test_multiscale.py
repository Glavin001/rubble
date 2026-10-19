import pytest
import torch

from learned_impact.multiscale import (
    EdgeMemory, MultiscaleProcessor, RecurrentGraphProcessor, align_memory,
    build_hierarchy, HierarchyCache,
)


def chain(n):
    return torch.stack((torch.arange(n - 1), torch.arange(1, n)), -1)


def test_hierarchy_does_not_create_paths_across_a_fracture():
    edges = chain(17)
    connected = torch.ones(16, dtype=torch.bool)
    connected[7] = False
    group = torch.arange(17) >= 8
    for level in build_hierarchy(torch.arange(17), edges, connected):
        next_group = torch.zeros(level.size, dtype=torch.bool)
        for p in range(level.size):
            members = group[level.parent == p]
            assert (members == members[0]).all()
            next_group[p] = members[0]
        if len(level.endpoints):
            a, b = level.endpoints.T
            assert (next_group[a] == next_group[b]).all()
        group = next_group


def test_multiscale_reaches_distant_nodes_that_two_local_passes_cannot():
    torch.manual_seed(3)
    n, width = 64, 8
    endpoints = chain(n)
    connected = torch.ones(n - 1, dtype=torch.bool)
    node = torch.randn(n, width, requires_grad=True)
    edge = torch.randn(n - 1, width)
    processor = MultiscaleProcessor(width)
    hierarchy = build_hierarchy(torch.arange(n), endpoints, connected)
    local, _ = processor(node, edge, endpoints, connected, [])
    distant, _ = processor(node, edge, endpoints, connected, hierarchy)
    local_grad = torch.autograd.grad(local[-1, 0], node, retain_graph=True)[0][0]
    distant_grad = torch.autograd.grad(distant[-1, 0], node)[0][0]
    assert local_grad.abs().sum() == 0
    assert distant_grad.abs().sum() > 1e-9


def test_unrelated_island_does_not_change_predictions_even_with_different_depth():
    torch.manual_seed(4)
    n, extra, width = 7, 64, 8
    a = chain(n)
    edges = torch.cat((a, chain(extra) + n))
    node, edge = torch.randn(n + extra, width), torch.randn(len(edges), width)
    processor = MultiscaleProcessor(width)
    def predict(x, e, endpoints):
        alive = torch.ones(len(endpoints), dtype=torch.bool)
        hierarchy = build_hierarchy(torch.arange(len(x)), endpoints, alive)
        return processor(x, e, endpoints, alive, hierarchy)
    first = predict(node[:n], edge[:n-1], a)
    combined = predict(node, edge, edges)
    torch.testing.assert_close(first[0], combined[0][:n], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(first[1], combined[1][:n-1], atol=2e-6, rtol=2e-6)


def test_stable_identity_preserves_node_and_edge_permutation_equivariance():
    torch.manual_seed(5)
    n, width = 13, 8
    edges, node, edge = chain(n), torch.randn(n, width), torch.randn(n-1, width)
    mask = torch.ones(n-1, dtype=torch.bool)
    mask[5] = False
    ids = torch.arange(n) * 3 + 10
    processor = MultiscaleProcessor(width)
    expected = processor(node, edge, edges, mask, build_hierarchy(ids, edges, mask))
    order, edge_order = torch.randperm(n), torch.randperm(n-1)
    inverse = torch.argsort(order)
    perm_edges = inverse[edges[edge_order]]
    result = processor(node[order], edge[edge_order], perm_edges, mask[edge_order],
                       build_hierarchy(ids[order], perm_edges, mask[edge_order]))
    torch.testing.assert_close(expected[0][order], result[0], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(expected[1][edge_order], result[1], atol=2e-6, rtol=2e-6)


def test_memory_follows_bond_identity_and_retirement_with_gradients():
    hidden = torch.randn(3, 4, requires_grad=True)
    state = EdgeMemory('wall', .01, torch.tensor([9, 2, 7]), hidden)
    result = align_memory(state, torch.tensor([7, 12, 9]), scene='wall', time=.01,
                          width=4, like=hidden)
    torch.testing.assert_close(result[0], hidden[2])
    torch.testing.assert_close(result[2], hidden[0])
    assert result[1].abs().sum() == 0
    result.sum().backward()
    assert hidden.grad[1].abs().sum() == 0
    assert (hidden.grad[[0, 2]] == 1).all()
    retired = EdgeMemory('wall', .02, torch.tensor([7, 12, 9]), result)
    fresh = align_memory(retired, torch.tensor([2]), scene='wall', time=.02, width=4, like=hidden)
    assert fresh.abs().sum() == 0


@pytest.mark.parametrize('scene,time', [('other', .01), ('wall', .011), ('wall', float('nan'))])
def test_memory_rejects_scene_or_clock_reuse(scene, time):
    hidden = torch.zeros(1, 4)
    state = EdgeMemory('wall', .01, torch.tensor([2]), hidden)
    with pytest.raises(ValueError, match='scene/clock'):
        align_memory(state, state.ids, scene=scene, time=time, width=4, like=hidden)


def test_recurrent_step_carries_history_without_hidden_module_mutation():
    torch.manual_seed(7)
    model = RecurrentGraphProcessor(8, 4)
    edges = chain(3)
    mask = torch.ones(2, dtype=torch.bool)
    node, edge = torch.randn(3, 8), torch.randn(2, 8)
    kwargs = dict(scene='beam', dt=.01)
    hierarchy = build_hierarchy(torch.arange(3), edges, mask)
    args = (node, edge, edges, mask, hierarchy, torch.tensor([4, 8]))
    _, first, state = model(*args, time=0., **kwargs)
    _, repeated, repeated_state = model(*args, time=0., **kwargs)
    torch.testing.assert_close(first, repeated, atol=0, rtol=0)
    torch.testing.assert_close(state.values, repeated_state.values, atol=0, rtol=0)
    _, second, _ = model(*args, time=.01, state=state, **kwargs)
    assert not torch.allclose(first, second)
    assert state.values.grad_fn is not None
    assert state.detach().values.grad_fn is None


def test_empty_edges_and_isolated_nodes_remain_valid():
    model = RecurrentGraphProcessor(8, 4)
    edges = torch.empty(0, 2, dtype=torch.long)
    mask = torch.empty(0, dtype=torch.bool)
    levels = build_hierarchy(torch.arange(5), edges, mask)
    node, edge, memory = model(torch.randn(5, 8), torch.empty(0, 8), edges, mask,
                               levels, torch.empty(0, dtype=torch.long),
                               scene='debris', time=0., dt=.01)
    assert node.shape == (5, 8) and edge.shape == (0, 8)
    assert memory.values.shape == (0, 4) and torch.isfinite(node).all()


def test_topology_cache_reuses_only_exact_topology_and_invalidates_in_place_changes():
    ids, edges = torch.arange(9), chain(9)
    connected = torch.ones(8, dtype=torch.bool)
    cache = HierarchyCache()
    first = cache.get(ids, edges, connected, 8)
    assert cache.get(ids.clone(), edges.clone(), connected.clone(), 8) is first
    assert cache.builds == 1
    connected[3] = False
    changed = cache.get(ids, edges, connected, 8)
    assert cache.builds == 2 and changed is not first
    fresh = build_hierarchy(ids, edges, connected, 8)
    for a, b in zip(changed, fresh):
        assert torch.equal(a.parent, b.parent) and torch.equal(a.endpoints, b.endpoints)
    cache.get(ids, edges, connected, 2)
    assert cache.builds == 3
