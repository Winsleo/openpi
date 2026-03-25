from __future__ import annotations

import dataclasses
import itertools
import logging
from typing import Any

from _pytest.logging import LogCaptureFixture
import jax
import numpy as np
import pytest

from openpi.models import pi0_config
from openpi.training import composable_sampler as _composable_sampler
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

# ---------------------------------------------------------------------------
# Shared fixtures (composable / debug TrainConfig)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def debug_config() -> _config.TrainConfig:
    """One debug TrainConfig per module; tests only read it and use dataclasses.replace."""
    return _config.get_config("debug")


def _assert_action_batches_shape(
    cfg: _config.TrainConfig, batches: list[tuple[Any, Any]], *, expected_len: int | None = None
) -> None:
    if expected_len is not None:
        assert len(batches) == expected_len
    bs = cfg.batch_size
    for _, actions in batches:
        assert actions.shape == (bs, cfg.model.action_horizon, cfg.model.action_dim)


def _two_leaf_sampler(
    mix_strategy: str,
    *,
    seed: int = 0,
    num_samples: int = 64,
    weights: tuple[float, float] = (0.5, 0.5),
) -> _config.ComposableSamplerConfig:
    """Build a 2-leaf ComposableSamplerConfig; set temperature only for weighted_random."""
    kwargs: dict[str, Any] = {
        "children": (_config.FakeDataConfig(), _config.FakeDataConfig()),
        "mix_strategy": mix_strategy,
        "weights": list(weights),
        "seed": seed,
        "num_samples": num_samples,
        "shuffle_within_leaf": False,
    }
    if mix_strategy == _composable_sampler.MIX_STRATEGY_WEIGHTED_RANDOM:
        kwargs["temperature"] = 1.3
    return _config.ComposableSamplerConfig(**kwargs)


def _replace_composable(
    base: _config.TrainConfig,
    composable: _config.ComposableNode,
    **train_overrides: Any,
) -> _config.TrainConfig:
    return dataclasses.replace(base, composable_data=composable, **train_overrides)


def _nested_sampler_config(
    *,
    inner_mix: str,
    outer_mix: str,
    inner_weights: tuple[float, float],
    outer_weights: tuple[float, float],
    inner_temp: float | None,
    num_samples: int,
) -> _config.ComposableSamplerConfig:
    """Inner (two fakes) + outer (inner + one fake) composable sampler tree."""
    inner_kw: dict[str, Any] = {
        "children": (_config.FakeDataConfig(), _config.FakeDataConfig()),
        "mix_strategy": inner_mix,
        "weights": list(inner_weights),
        "seed": 1,
        "shuffle_within_leaf": False,
    }
    if inner_temp is not None:
        inner_kw["temperature"] = inner_temp
    inner = _config.ComposableSamplerConfig(**inner_kw)

    outer_kw: dict[str, Any] = {
        "children": (inner, _config.FakeDataConfig()),
        "mix_strategy": outer_mix,
        "weights": list(outer_weights),
        "seed": 2,
        "num_samples": num_samples,
        "shuffle_within_leaf": False,
    }
    if outer_mix == _composable_sampler.MIX_STRATEGY_WEIGHTED_RANDOM:
        outer_kw["temperature"] = 1.0
    return _config.ComposableSamplerConfig(**outer_kw)


# ---------------------------------------------------------------------------
# TorchDataLoader (non-composable)
# ---------------------------------------------------------------------------


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_loops_for_extra_batches():
    """``num_batches`` > one epoch of data: _batched_iter rewinds the underlying loader."""
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=10)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_build_transformed_training_dataset_behavior_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    model = pi0_config.Pi0Config(action_dim=7, action_horizon=4, max_token_len=32)
    dc = _config.DataConfig(repo_id="x/y", behavior_dataset_root="/tmp/b1k")

    def fake_behavior(data_cfg: _config.DataConfig, ah: int) -> _data_loader.Dataset:
        assert data_cfg is dc
        assert ah == 4
        return _data_loader.FakeDataset(model, num_samples=3)

    def fake_torch(*_args: Any, **_kwargs: Any) -> _data_loader.Dataset:
        raise AssertionError("create_torch_dataset should not run when behavior_dataset_root is set")

    monkeypatch.setattr(_data_loader, "create_behavior_dataset", fake_behavior)
    monkeypatch.setattr(_data_loader, "create_torch_dataset", fake_torch)

    out = _data_loader._build_transformed_training_dataset(dc, 4, model, skip_norm_stats=True)
    assert len(out) == 3


def test_build_transformed_training_dataset_torch_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    model = pi0_config.Pi0Config(action_dim=7, action_horizon=4, max_token_len=32)
    dc = _config.DataConfig(repo_id="fake")

    def fake_behavior(*_args: Any, **_kwargs: Any) -> _data_loader.Dataset:
        raise AssertionError("create_behavior_dataset should not run without behavior_dataset_root")

    def fake_torch(data_cfg: _config.DataConfig, ah: int, mc: pi0_config.Pi0Config) -> _data_loader.Dataset:
        assert data_cfg is dc
        assert ah == 4
        assert mc is model
        return _data_loader.FakeDataset(mc, num_samples=5)

    monkeypatch.setattr(_data_loader, "create_behavior_dataset", fake_behavior)
    monkeypatch.setattr(_data_loader, "create_torch_dataset", fake_torch)

    out = _data_loader._build_transformed_training_dataset(dc, 4, model, skip_norm_stats=True)
    assert len(out) == 5


# ---------------------------------------------------------------------------
# ComposableSampler in data loader — flat (all mix strategies)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mix_strategy", sorted(_composable_sampler.COMPOSABLE_MIX_STRATEGIES))
def test_composable_sampler_in_dataloader_flat(
    debug_config: _config.TrainConfig, mix_strategy: str
) -> None:
    """ComposableSamplerConfig: ConcatDataset + MixNode; every built-in mix_strategy."""
    sampler_cfg = _two_leaf_sampler(mix_strategy, seed=0, num_samples=64)
    comp = _config.ComposableDataConfig(
        composition_strategy="random",
        children=[sampler_cfg],
        weights=[1.0],
        seed=123,
    )
    cfg = _replace_composable(debug_config, comp)
    loader = _data_loader.create_composable_data_loader(
        cfg, comp, skip_norm_stats=True, num_batches=2, shuffle=False
    )
    assert loader.data_config().repo_id == "fake"
    batches = list(loader)
    _assert_action_batches_shape(cfg, batches, expected_len=2)


# ---------------------------------------------------------------------------
# Nested ComposableSamplerConfig (parametrized: default + pi05-style)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    (
        "inner_mix",
        "outer_mix",
        "inner_weights",
        "outer_weights",
        "inner_temp",
        "num_batches",
    ),
    [
        pytest.param(
            _composable_sampler.MIX_STRATEGY_LARGEST_REMAINDER,
            _composable_sampler.MIX_STRATEGY_ROUND_ROBIN,
            (1.0, 1.0),
            (1.0, 1.0),
            None,
            1,
            id="inner_lr_outer_rr",
        ),
        pytest.param(
            _composable_sampler.MIX_STRATEGY_WEIGHTED_RANDOM,
            _composable_sampler.MIX_STRATEGY_LARGEST_REMAINDER,
            (0.6, 0.4),
            (2.0, 1.0),
            1.0,
            2,
            id="pi05_style_wr_lr",
        ),
        pytest.param(
            _composable_sampler.MIX_STRATEGY_STRATIFIED_RANDOM,
            _composable_sampler.MIX_STRATEGY_SHUFFLE_SEQUENTIAL,
            (1.0, 1.0),
            (1.0, 1.0),
            None,
            1,
            id="inner_stratified_outer_shuffle_seq",
        ),
    ],
)
def test_composable_sampler_in_dataloader_nested(
    debug_config: _config.TrainConfig,
    inner_mix: str,
    outer_mix: str,
    inner_weights: tuple[float, float],
    outer_weights: tuple[float, float],
    inner_temp: float | None,
    num_batches: int,
) -> None:
    """Nested sampler tree: DFS leaves + isomorphic MixNode; multiple strategy pairs."""
    outer = _nested_sampler_config(
        inner_mix=inner_mix,
        outer_mix=outer_mix,
        inner_weights=inner_weights,
        outer_weights=outer_weights,
        inner_temp=inner_temp,
        num_samples=48 if num_batches == 1 else 64,
    )

    comp = _config.ComposableDataConfig(
        composition_strategy="proportional",
        children=[outer],
        weights=[1.0],
        seed=99,
    )
    cfg = _replace_composable(debug_config, comp)
    loader = _data_loader.create_composable_data_loader(
        cfg, comp, skip_norm_stats=True, num_batches=num_batches, shuffle=False
    )
    batches = list(loader)
    _assert_action_batches_shape(cfg, batches, expected_len=num_batches)


# ---------------------------------------------------------------------------
# create_data_loader dispatch + composable roots
# ---------------------------------------------------------------------------


def test_composable_sampler_via_create_data_loader(debug_config: _config.TrainConfig) -> None:
    """create_data_loader dispatches to composable path when composable_data is set."""
    sampler_cfg = _config.ComposableSamplerConfig(
        children=[_config.FakeDataConfig()],
        mix_strategy=_composable_sampler.MIX_STRATEGY_LARGEST_REMAINDER,
        num_samples=8,
        shuffle_within_leaf=False,
    )
    comp = _config.ComposableDataConfig(
        composition_strategy="random",
        children=[sampler_cfg],
        weights=[1.0],
    )
    cfg = _replace_composable(debug_config, comp)
    loader = _data_loader.create_data_loader(cfg, skip_norm_stats=True, num_batches=1, shuffle=False)
    batches = list(loader)
    assert len(batches) == 1


def test_create_data_loader_composable_root_is_sampler(debug_config: _config.TrainConfig) -> None:
    """composable_data root may be a bare ComposableSamplerConfig."""
    sampler_cfg = _config.ComposableSamplerConfig(
        children=[_config.FakeDataConfig()],
        mix_strategy=_composable_sampler.MIX_STRATEGY_LARGEST_REMAINDER,
        num_samples=16,
        shuffle_within_leaf=False,
    )
    cfg = _replace_composable(debug_config, sampler_cfg)
    loader = _data_loader.create_data_loader(cfg, skip_norm_stats=True, num_batches=1, shuffle=False)
    batches = list(loader)
    _assert_action_batches_shape(cfg, batches, expected_len=1)


def test_create_data_loader_composable_root_is_data_factory(debug_config: _config.TrainConfig) -> None:
    """composable_data root may be a bare DataConfigFactory."""
    cfg = _replace_composable(debug_config, _config.FakeDataConfig())
    loader = _data_loader.create_data_loader(cfg, skip_norm_stats=True, num_batches=2, shuffle=False)
    batches = list(loader)
    _assert_action_batches_shape(cfg, batches, expected_len=2)


# ---------------------------------------------------------------------------
# ComposableDataConfig composition_strategy (batch-level)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("composition_strategy", ["random", "proportional", "round_robin"])
def test_composable_data_config_batch_strategies(
    debug_config: _config.TrainConfig, composition_strategy: str
) -> None:
    """Leaf loaders each get ``num_batches``; the composed iterator may yield more steps."""
    comp = _config.ComposableDataConfig(
        composition_strategy=composition_strategy,
        children=[_config.FakeDataConfig(), _config.FakeDataConfig()],
        weights=[0.6, 0.4],
        seed=7,
    )
    cfg = _replace_composable(debug_config, comp)
    loader = _data_loader.create_composable_data_loader(
        cfg, comp, skip_norm_stats=True, num_batches=2, shuffle=False
    )
    batches = list(itertools.islice(loader, 2))
    _assert_action_batches_shape(cfg, batches, expected_len=2)


def test_composable_data_config_inbatch(debug_config: _config.TrainConfig) -> None:
    """inbatch splits local batch across child loaders; output batch still matches global batch_size."""
    comp = _config.ComposableDataConfig(
        composition_strategy="inbatch",
        children=[_config.FakeDataConfig(), _config.FakeDataConfig()],
        weights=[1.0, 1.0],
        seed=11,
    )
    cfg = _replace_composable(debug_config, comp)
    loader = _data_loader.create_composable_data_loader(
        cfg, comp, skip_norm_stats=True, num_batches=2, shuffle=False
    )
    batches = list(itertools.islice(loader, 2))
    _assert_action_batches_shape(cfg, batches, expected_len=2)


# ---------------------------------------------------------------------------
# Errors + logging
# ---------------------------------------------------------------------------


def test_composable_sampler_num_samples_below_batch_raises(debug_config: _config.TrainConfig) -> None:
    sampler_cfg = _config.ComposableSamplerConfig(
        children=[_config.FakeDataConfig()],
        mix_strategy=_composable_sampler.MIX_STRATEGY_LARGEST_REMAINDER,
        num_samples=1,
    )
    comp = _config.ComposableDataConfig(
        composition_strategy="random",
        children=[sampler_cfg],
        weights=[1.0],
    )
    cfg = _replace_composable(debug_config, comp, batch_size=8)
    with pytest.raises(ValueError, match="num_samples"):
        _data_loader.create_composable_data_loader(cfg, comp, skip_norm_stats=True, num_batches=1, shuffle=False)


def test_composable_sampler_concat_shorter_than_batch_raises(debug_config: _config.TrainConfig) -> None:
    sampler_cfg = _config.ComposableSamplerConfig(
        children=[_config.FakeDataConfig()],
        mix_strategy=_composable_sampler.MIX_STRATEGY_LARGEST_REMAINDER,
        num_samples=1024,
    )
    comp = _config.ComposableDataConfig(
        composition_strategy="random",
        children=[sampler_cfg],
        weights=[1.0],
    )
    cfg = _replace_composable(debug_config, comp, batch_size=2048)
    with pytest.raises(ValueError, match="ConcatDataset length"):
        _data_loader.create_composable_data_loader(cfg, comp, skip_norm_stats=True, num_batches=1, shuffle=False)


def test_composable_sampler_shuffle_true_logs_debug(
    debug_config: _config.TrainConfig, caplog: LogCaptureFixture
) -> None:
    sampler_cfg = _config.ComposableSamplerConfig(
        children=[_config.FakeDataConfig()],
        mix_strategy=_composable_sampler.MIX_STRATEGY_LARGEST_REMAINDER,
        num_samples=16,
    )
    comp = _config.ComposableDataConfig(
        composition_strategy="random",
        children=[sampler_cfg],
        weights=[1.0],
    )
    cfg = _replace_composable(debug_config, comp)
    with caplog.at_level(logging.DEBUG):
        _data_loader.create_composable_data_loader(
            cfg, comp, skip_norm_stats=True, num_batches=1, shuffle=True
        )
    assert "shuffle=True is ignored" in caplog.text


# ---------------------------------------------------------------------------
# ComposableSamplerConfig __post_init__ validation
# ---------------------------------------------------------------------------


def test_composable_sampler_config_temperature_only_for_weighted_random() -> None:
    with pytest.raises(ValueError, match="temperature"):
        _config.ComposableSamplerConfig(
            children=[_config.FakeDataConfig()],
            mix_strategy=_composable_sampler.MIX_STRATEGY_LARGEST_REMAINDER,
            temperature=1.0,
        )


def test_composable_sampler_config_empty_children() -> None:
    with pytest.raises(ValueError, match="at least one child"):
        _config.ComposableSamplerConfig(
            children=(),
            mix_strategy=_composable_sampler.MIX_STRATEGY_LARGEST_REMAINDER,
        )


def test_composable_sampler_config_weights_length_mismatch() -> None:
    with pytest.raises(ValueError, match="weights length"):
        _config.ComposableSamplerConfig(
            children=[_config.FakeDataConfig(), _config.FakeDataConfig()],
            mix_strategy=_composable_sampler.MIX_STRATEGY_LARGEST_REMAINDER,
            weights=[1.0],
        )


def test_composable_sampler_config_rejects_composable_data_child() -> None:
    nested_data = _config.ComposableDataConfig(
        composition_strategy="random",
        children=[_config.FakeDataConfig()],
        weights=[1.0],
    )
    with pytest.raises(ValueError, match="ComposableDataConfig"):
        _config.ComposableSamplerConfig(
            children=[nested_data],
            mix_strategy=_composable_sampler.MIX_STRATEGY_LARGEST_REMAINDER,
        )


# ---------------------------------------------------------------------------
# Determinism: same TrainConfig + composable tree -> identical batches
# ---------------------------------------------------------------------------


def _collect_action_arrays(batches: list[tuple[Any, Any]]) -> list[np.ndarray]:
    return [np.asarray(actions) for _, actions in batches]


def test_composable_loader_determinism_same_config(debug_config: _config.TrainConfig) -> None:
    """Two loaders built with the same config produce identical action tensors."""
    sampler_cfg = _two_leaf_sampler(
        _composable_sampler.MIX_STRATEGY_LARGEST_REMAINDER,
        seed=42,
        num_samples=32,
    )
    comp = _config.ComposableDataConfig(
        composition_strategy="random",
        children=[sampler_cfg],
        weights=[1.0],
        seed=2025,
    )
    cfg = _replace_composable(debug_config, comp, seed=2025)

    def _make_loader():
        return _data_loader.create_composable_data_loader(
            cfg, comp, skip_norm_stats=True, num_batches=3, shuffle=False
        )

    a = _collect_action_arrays(list(_make_loader()))
    b = _collect_action_arrays(list(_make_loader()))
    assert len(a) == len(b) == 3
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)


# ---------------------------------------------------------------------------
# Legacy integration tests
# ---------------------------------------------------------------------------


def test_with_fake_dataset(debug_config: _config.TrainConfig) -> None:
    loader = _data_loader.create_data_loader(debug_config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == debug_config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (
            debug_config.batch_size,
            debug_config.model.action_horizon,
            debug_config.model.action_dim,
        )


# def test_with_real_dataset() -> None:
#     config = _config.get_config("pi0_aloha_sim")
#     config = dataclasses.replace(config, batch_size=4)

#     try:
#         loader = _data_loader.create_data_loader(
#             config,
#             skip_norm_stats=True,
#             num_batches=2,
#             shuffle=True,
#         )
#     except Exception as e:
#         msg = str(e).lower()
#         if any(
#             x in msg
#             for x in ("connection", "network", "no such file", "max retries", "huggingface")
#         ):
#             pytest.skip(f"Real dataset unavailable: {e}")
#         raise

#     assert loader.data_config().repo_id == config.data.repo_id

#     batches = list(loader)

#     assert len(batches) == 2

#     for _, actions in batches:
#         assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)
