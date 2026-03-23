import numpy as np
import pytest

from openpi.training import composable_sampler as _sampler


def _make_leaves(
    lengths: tuple[int, int, int] = (20, 20, 20),
    *,
    shuffle: bool = False,
    seed_base: int = 100,
) -> list[_sampler.LeafNode]:
    offsets = np.cumsum([0, *lengths[:-1]]).tolist()
    leaves: list[_sampler.LeafNode] = []
    for i, (length, offset) in enumerate(zip(lengths, offsets, strict=True)):
        leaves.append(
            _sampler.LeafNode(
                length=length,
                offset=int(offset),
                shuffle=shuffle,
                seed=seed_base + i,
                name=f"leaf_{i}",
            )
        )
    return leaves


def _child_ids_from_indices(indices: list[int], lengths: tuple[int, int, int]) -> list[int]:
    bounds = np.cumsum(lengths).tolist()
    out: list[int] = []
    for idx in indices:
        if idx < bounds[0]:
            out.append(0)
        elif idx < bounds[1]:
            out.append(1)
        else:
            out.append(2)
    return out


def _counts_by_child(indices: list[int], lengths: tuple[int, int, int]) -> np.ndarray:
    child_ids = _child_ids_from_indices(indices, lengths)
    return np.bincount(np.asarray(child_ids, dtype=np.int64), minlength=3)


def test_normalize_weights_valid_and_invalid() -> None:
    p = _sampler.normalize_weights([2.0, 1.0, 1.0])
    assert np.isclose(p.sum(), 1.0)
    assert np.allclose(p, np.asarray([0.5, 0.25, 0.25]))

    with pytest.raises(ValueError):
        _sampler.normalize_weights([])
    with pytest.raises(ValueError):
        _sampler.normalize_weights([0.0, 0.0])
    with pytest.raises(ValueError):
        _sampler.normalize_weights([1.0, -1.0])


def test_largest_remainder_allocate_basic_and_tie_break() -> None:
    q = _sampler.largest_remainder_allocate(10, [0.51, 0.29, 0.20])
    assert q.tolist() == [5, 3, 2]
    assert int(q.sum()) == 10

    # 1/3 tie remainder, stable tie-break should favor lower index.
    q_tie = _sampler.largest_remainder_allocate(1, [1.0, 1.0, 1.0])
    assert q_tie.tolist() == [1, 0, 0]

    q_zero = _sampler.largest_remainder_allocate(0, [0.2, 0.8])
    assert q_zero.tolist() == [0, 0]

    with pytest.raises(ValueError):
        _sampler.largest_remainder_allocate(-1, [1.0, 1.0])


def test_permutation_traversal_sequential_and_state_restore() -> None:
    t = _sampler.PermutationTraversal(length=5, shuffle=False, seed=1)
    first = [t.next_index() for _ in range(7)]
    assert first == [0, 1, 2, 3, 4, 0, 1]

    t2 = _sampler.PermutationTraversal(length=5, shuffle=True, seed=123)
    t2.set_epoch(2)
    warmup = [t2.next_index() for _ in range(4)]
    state = t2.state_dict()
    cont1 = [t2.next_index() for _ in range(6)]

    t3 = _sampler.PermutationTraversal(length=5, shuffle=True, seed=999)
    t3.load_state_dict(state)
    cont2 = [t3.next_index() for _ in range(6)]
    assert warmup != cont1
    assert cont1 == cont2


def test_leaf_node_offset_budget_and_errors() -> None:
    leaf = _sampler.LeafNode(length=5, offset=10, shuffle=False, seed=0)
    assert list(leaf.plan(6)) == [10, 11, 12, 13, 14, 10]
    assert list(leaf.plan(0)) == []
    with pytest.raises(ValueError):
        list(leaf.plan(-1))


def test_largest_remainder_strategy_quota_and_prefix_smoothness() -> None:
    lengths = (20, 20, 20)
    leaves = _make_leaves(lengths, shuffle=False)
    for leaf in leaves:
        leaf.set_epoch(0)

    strategy = _sampler.LargestRemainderStrategy()
    rng = np.random.default_rng(7)
    budget = 30
    weights = _sampler.normalize_weights([0.5, 0.3, 0.2])
    out = list(strategy.plan(leaves, weights, budget, rng))
    counts = _counts_by_child(out, lengths)
    assert counts.tolist() == [15, 9, 6]
    assert len(out) == budget

    prefix_counts = _counts_by_child(out[:10], lengths)
    assert prefix_counts.tolist() == [5, 3, 2]


def test_weighted_random_temperature_effect_and_invalid_temperature() -> None:
    lengths = (2000, 2000, 2000)
    leaves = _make_leaves(lengths, shuffle=False)
    for leaf in leaves:
        leaf.set_epoch(0)

    weights = _sampler.normalize_weights([0.7, 0.2, 0.1])
    budget = 900

    cold = _sampler.WeightedRandomStrategy(temperature=0.5)
    neutral = _sampler.WeightedRandomStrategy(temperature=1.0)
    hot = _sampler.WeightedRandomStrategy(temperature=2.0)
    rng = np.random.default_rng(1234)
    out_neutral = list(cold.plan(leaves, weights, budget, rng))
    counts_cold = _counts_by_child(out_neutral, lengths)

    for leaf in leaves:
        leaf.set_epoch(0)
    rng = np.random.default_rng(1234)
    out_mid = list(neutral.plan(leaves, weights, budget, rng))
    counts_mid = _counts_by_child(out_mid, lengths)

    for leaf in leaves:
        leaf.set_epoch(0)
    rng = np.random.default_rng(1234)
    out_hot = list(hot.plan(leaves, weights, budget, rng))
    counts_hot = _counts_by_child(out_hot, lengths)

    assert len(out_neutral) == budget == len(out_mid) == len(out_hot)
    # Colder temperature should further favor the largest-weight child.
    assert counts_cold[0] > counts_mid[0] > counts_hot[0]

    with pytest.raises(ValueError):
        _sampler.WeightedRandomStrategy(temperature=0.0)


def test_round_robin_strategy_uniform_and_weighted_cyclic() -> None:
    lengths = (20, 20, 20)
    leaves = _make_leaves(lengths, shuffle=False)
    for leaf in leaves:
        leaf.set_epoch(0)

    strategy = _sampler.RoundRobinStrategy()
    rng = np.random.default_rng(0)

    uniform_out = list(strategy.plan(leaves, _sampler.normalize_weights([1, 1, 1]), 12, rng))
    uniform_child_ids = _child_ids_from_indices(uniform_out, lengths)
    assert uniform_child_ids == [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]

    for leaf in leaves:
        leaf.set_epoch(0)
    weighted_out = list(strategy.plan(leaves, _sampler.normalize_weights([0.5, 0.3, 0.2]), 10, rng))
    weighted_child_ids = _child_ids_from_indices(weighted_out, lengths)
    assert _counts_by_child(weighted_out, lengths).tolist() == [5, 3, 2]
    assert weighted_child_ids == [0, 1, 2, 0, 1, 2, 0, 1, 0, 0]


def test_shuffle_sequential_strategy_runs_in_contiguous_blocks() -> None:
    lengths = (20, 20, 20)
    leaves = _make_leaves(lengths, shuffle=False)
    for leaf in leaves:
        leaf.set_epoch(0)

    strategy = _sampler.ShuffleSequentialStrategy()
    out = list(strategy.plan(leaves, _sampler.normalize_weights([0.5, 0.3, 0.2]), 10, np.random.default_rng(5)))
    child_ids = _child_ids_from_indices(out, lengths)

    # At most 2 transitions for 3 blocks.
    transitions = sum(a != b for a, b in zip(child_ids, child_ids[1:], strict=False))
    assert transitions <= 2
    assert _counts_by_child(out, lengths).tolist() == [5, 3, 2]


def test_stratified_random_strategy_round_uniqueness() -> None:
    lengths = (20, 20, 20)
    leaves = _make_leaves(lengths, shuffle=False)
    for leaf in leaves:
        leaf.set_epoch(0)

    strategy = _sampler.StratifiedRandomStrategy()
    out = list(strategy.plan(leaves, _sampler.normalize_weights([1, 1, 1]), 12, np.random.default_rng(9)))
    child_ids = _child_ids_from_indices(out, lengths)
    assert _counts_by_child(out, lengths).tolist() == [4, 4, 4]
    for i in range(0, 12, 3):
        assert set(child_ids[i : i + 3]) == {0, 1, 2}


def test_mixnode_default_strategy_and_state_restore() -> None:
    lengths = (30, 30, 30)
    leaves = _make_leaves(lengths, shuffle=True, seed_base=300)
    node = _sampler.MixNode(children=leaves, weights=[0.5, 0.3, 0.2], seed=42, name="node")
    node.set_epoch(2)

    assert isinstance(node.strategy, _sampler.LargestRemainderStrategy)

    pre = list(node.plan(10))
    state = node.state_dict()
    post_ref = list(node.plan(12))

    leaves2 = _make_leaves(lengths, shuffle=True, seed_base=300)
    node2 = _sampler.MixNode(children=leaves2, weights=[0.5, 0.3, 0.2], seed=1, name="node2")
    node2.load_state_dict(state)
    post_loaded = list(node2.plan(12))
    assert len(pre) == 10
    assert post_ref == post_loaded


def test_compose_and_from_dataset_lengths_and_len() -> None:
    leaves = _make_leaves((10, 10, 10), shuffle=False)
    for leaf in leaves:
        leaf.set_epoch(0)

    nodes = [
        _sampler.Compose.largest_remainder(leaves, weights=[0.5, 0.3, 0.2], seed=1),
        _sampler.Compose.weighted_random(leaves, weights=[0.5, 0.3, 0.2], temperature=1.2, seed=1),
        _sampler.Compose.round_robin(leaves, weights=[0.5, 0.3, 0.2], seed=1),
        _sampler.Compose.shuffle_sequential(leaves, weights=[0.5, 0.3, 0.2], seed=1),
        _sampler.Compose.stratified_random(leaves, weights=[0.5, 0.3, 0.2], seed=1),
    ]
    for node in nodes:
        node.set_epoch(0)
        out = list(node.plan(15))
        assert len(out) == 15

    sampler_default = _sampler.ComposableSampler.from_dataset_lengths(
        dataset_lengths=[10, 10, 10],
        num_samples=17,
        seed=123,
    )
    sampler_custom = _sampler.ComposableSampler.from_dataset_lengths(
        dataset_lengths=[10, 10, 10],
        num_samples=17,
        seed=123,
        strategy=_sampler.RoundRobinStrategy(),
    )
    assert len(sampler_default) == 17
    assert len(list(iter(sampler_default))) == 17
    assert len(list(iter(sampler_custom))) == 17


def test_builder_helpers_with_custom_strategies() -> None:
    episodes = [
        _sampler.EpisodeSpec(length=5, offset=0, weight=1.0, name="ep0"),
        _sampler.EpisodeSpec(length=5, offset=5, weight=1.0, name="ep1"),
    ]
    ep_mix = _sampler.build_episode_mix_node(
        episodes,
        seed=7,
        strategy=_sampler.RoundRobinStrategy(),
        name="ep_mix",
    )
    ep_mix.set_epoch(0)
    assert len(list(ep_mix.plan(8))) == 8

    root = _sampler.build_source_task_hierarchy(
        {
            "s1": {"t1": [_sampler.EpisodeSpec(length=6, offset=0)]},
            "s2": {"t2": [_sampler.EpisodeSpec(length=6, offset=6)]},
        },
        seed=5,
        source_strategy=_sampler.RoundRobinStrategy(),
        task_strategy=_sampler.ShuffleSequentialStrategy(),
        episode_strategy=_sampler.StratifiedRandomStrategy(),
    )
    root.set_epoch(1)
    assert len(list(root.plan(10))) == 10


@pytest.mark.parametrize(
    ("strategy", "expected_type"),
    [
        (_sampler.LargestRemainderStrategy(), _sampler.LargestRemainderStrategy),
        (_sampler.WeightedRandomStrategy(temperature=1.3), _sampler.WeightedRandomStrategy),
        (_sampler.RoundRobinStrategy(), _sampler.RoundRobinStrategy),
        (_sampler.ShuffleSequentialStrategy(), _sampler.ShuffleSequentialStrategy),
        (_sampler.StratifiedRandomStrategy(), _sampler.StratifiedRandomStrategy),
    ],
)
def test_strategy_state_restore_and_clone(strategy: _sampler.MixStrategy, expected_type: type) -> None:
    restored = _sampler.mix_strategy_from_state(strategy.state_dict())
    cloned = _sampler.clone_mix_strategy(strategy)
    assert isinstance(restored, expected_type)
    assert isinstance(cloned, expected_type)
    assert cloned is not strategy


def test_mixnode_and_resolve_weights_edge_cases() -> None:
    leaves = _make_leaves((10, 10, 10), shuffle=False)
    node_bad_weights = _sampler.MixNode(children=leaves, weights=[0.5, 0.5], seed=1)
    with pytest.raises(ValueError):
        list(node_bad_weights.plan(5))

    node = _sampler.MixNode(children=leaves, weights=[0.5, 0.3, 0.2], seed=1)
    with pytest.raises(ValueError):
        list(node.plan(-1))


@pytest.mark.parametrize("seed", [0, 1, 2, 7, 42])
def test_weighted_random_matches_multinomial_quota_per_seed(seed: int) -> None:
    lengths = (4000, 4000, 4000)
    leaves = _make_leaves(lengths, shuffle=False, seed_base=500)
    for leaf in leaves:
        leaf.set_epoch(0)

    strategy = _sampler.WeightedRandomStrategy(temperature=1.0)
    weights = _sampler.normalize_weights([0.6, 0.3, 0.1])
    budget = 1200
    rng = np.random.default_rng(seed)
    out = list(strategy.plan(leaves, weights, budget, rng))
    counts = _counts_by_child(out, lengths)

    # 4-sigma guard band against rare multinomial fluctuation.
    expected = np.asarray([0.6, 0.3, 0.1]) * budget
    sigma = np.sqrt(budget * np.asarray([0.6, 0.3, 0.1]) * (1 - np.asarray([0.6, 0.3, 0.1])))
    assert np.all(np.abs(counts - expected) <= 4.0 * sigma + 3.0)


@pytest.mark.parametrize("epoch", [0, 1, 3])
def test_sampler_reproducible_with_same_seed_and_epoch(epoch: int) -> None:
    sampler_a = _sampler.ComposableSampler.from_dataset_lengths(
        dataset_lengths=[30, 30, 30],
        weights=[0.5, 0.3, 0.2],
        num_samples=60,
        seed=2024,
        strategy=_sampler.WeightedRandomStrategy(temperature=0.8),
    )
    sampler_b = _sampler.ComposableSampler.from_dataset_lengths(
        dataset_lengths=[30, 30, 30],
        weights=[0.5, 0.3, 0.2],
        num_samples=60,
        seed=2024,
        strategy=_sampler.WeightedRandomStrategy(temperature=0.8),
    )
    sampler_a.set_epoch(epoch)
    sampler_b.set_epoch(epoch)
    assert list(sampler_a) == list(sampler_b)


def test_mixnode_state_restore_keeps_leaf_and_rng_progress() -> None:
    lengths = (50, 50, 50)
    leaves_a = _make_leaves(lengths, shuffle=True, seed_base=900)
    leaves_b = _make_leaves(lengths, shuffle=True, seed_base=900)
    node_a = _sampler.MixNode(
        children=leaves_a,
        weights=[0.5, 0.3, 0.2],
        strategy=_sampler.StratifiedRandomStrategy(),
        seed=123,
    )
    node_b = _sampler.MixNode(
        children=leaves_b,
        weights=[0.5, 0.3, 0.2],
        strategy=_sampler.StratifiedRandomStrategy(),
        seed=999,
    )
    node_a.set_epoch(4)
    node_b.set_epoch(4)

    _ = list(node_a.plan(25))
    state = node_a.state_dict()
    node_b.load_state_dict(state)

    next_a = list(node_a.plan(40))
    next_b = list(node_b.plan(40))
    assert next_a == next_b


def test_nested_compose_source_distribution_sanity() -> None:
    # Inner mix over sources 0/1, then outer mix with source 2.
    leaves = [
        _sampler.LeafNode(length=400, offset=0, shuffle=False, seed=10),
        _sampler.LeafNode(length=400, offset=400, shuffle=False, seed=11),
        _sampler.LeafNode(length=400, offset=800, shuffle=False, seed=12),
    ]
    for leaf in leaves:
        leaf.set_epoch(0)

    inner = _sampler.Compose.weighted_random(
        leaves[:2],
        weights=[0.7, 0.3],
        temperature=1.0,
        seed=100,
        name="inner",
    )
    root = _sampler.Compose.round_robin(
        [inner, leaves[2]],
        weights=[0.8, 0.2],
        seed=101,
        name="root",
    )
    root.set_epoch(0)
    out = list(root.plan(1000))
    counts = _counts_by_child(out, (400, 400, 400))
    # Outer (0.8,0.2) and inner (0.7,0.3) => approximately (0.56,0.24,0.20).
    props = counts / counts.sum()
    assert np.allclose(props, np.asarray([0.56, 0.24, 0.20]), atol=0.08)

