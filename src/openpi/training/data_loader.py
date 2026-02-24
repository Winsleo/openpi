from collections.abc import Iterator, Sequence
import logging
import multiprocessing
import os
import typing
from typing import Any, Literal, Protocol, SupportsIndex, TypeVar
from pathlib import Path

import jax
import jax.numpy as jnp
import lerobot.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.training.composable_dataloader as composable
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_behavior_dataset(data_config: _config.DataConfig, action_horizon: int) -> Dataset:
    """Create a dataset for training."""
    from omnigibson.learning.datas.lerobot_dataset import BehaviorLeRobotDataset
    
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    dataset = BehaviorLeRobotDataset(
        repo_id=data_config.repo_id,
        root=data_config.behavior_dataset_root,
        tasks=["picking_up_trash"],
        modalities=["rgb"],
        local_only=True,
        delta_timestamps={
            key: [t / 30.0 for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
        episodes=data_config.episodes_index,
        chunk_streaming_using_keyframe=True,
        shuffle=True,
    )
    print("*********************************************")

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset.meta.tasks)])

    return dataset


def create_torch_dataset(
    data_config: _config.DataConfig,
    action_horizon: int, 
    model_config: _model.BaseModelConfig,
    root: str | Path | None = None,
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=root)
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        root=root,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        datasets=data_config.datasets,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    data_config: _config.DataConfig | None = None,
    data_config_factory: _config.DataConfigFactory | None = None,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        data_config: Optional pre-configured data config. If None, will be created from config.
        data_config_factory: Optional factory to create data config. If provided and data_config
            is None, used instead of config.data.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    
    Note:
        If config.composable_data is set, this function will automatically use
        the composable data loader for mixed dataset training.
    """
    # Check if composable/mixed dataset training is configured
    if config.composable_data is not None:
        logging.info("Using composable data loader for mixed dataset training")
        return create_composable_data_loader(
            config,
            config.composable_data,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
        )
    
    if data_config is None:
        factory = data_config_factory or config.data
        data_config = factory.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.num_batches is not None:
        num_batches = data_config.num_batches

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    elif data_config.behavior_dataset_root is not None:
        # Handle behavior datasets by reusing create_behavior_data_loader
        return create_behavior_data_loader(
            config,
            data_config=data_config,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
        )
    else:
        return create_torch_data_loader(
            data_config,
            model_config=config.model,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            num_workers=config.num_workers,
            seed=config.seed,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )


def create_behavior_data_loader(
    config: _config.TrainConfig,
    *,
    data_config: _config.DataConfig | None = None,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    if data_config is None:
        data_config = config.data.create(config.assets_dirs, config.model)
    dataset = create_behavior_dataset(data_config, action_horizon=config.model.action_horizon)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=config.batch_size // jax.process_count(),
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    @property
    def dataset(self) -> torch.utils.data.Dataset:
        return self._data_loader.dataset

    @property
    def sharding(self) -> jax.sharding.Sharding | None:
        return self._sharding

    def __iter__(self):
        if self._sharding is not None:
            transform = lambda batch: jax.tree.map(
                lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch
            )
        else:
            transform = lambda batch: jax.tree.map(torch.as_tensor, batch)
        yield from _batched_iter(self._data_loader, self._num_batches, transform)

    def __len__(self) -> int:
        """Return the number of batches.

        Returns:
            If num_batches was specified, returns that value.
            Otherwise, returns the number of batches in one epoch.
        """
        if self._num_batches is not None:
            return self._num_batches
        return len(self._data_loader)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


def _batched_iter(source, num_batches: int | None, transform_fn):
    """Iterate over *source* with an optional batch-count limit, looping if needed.

    When *num_batches* is ``None`` the source is iterated exactly once.
    Otherwise the source is re-iterated as needed until *num_batches*
    batches have been yielded.  Each raw batch is passed through
    *transform_fn* before being yielded.
    """
    epoch = 0
    num_items = 0
    while True:
        if num_batches is None and epoch > 0:
            return
        data_iter = iter(source)
        while True:
            if num_batches is not None and num_items >= num_batches:
                return
            try:
                batch = next(data_iter)
            except StopIteration:
                epoch += 1
                break
            num_items += 1
            yield transform_fn(batch)


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    @property
    def dataset(self) -> DroidRldsDataset:
        return self._dataset

    @property
    def sharding(self) -> jax.sharding.Sharding | None:
        return self._sharding

    def __iter__(self):
        transform = lambda batch: jax.tree.map(
            lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch
        )
        yield from _batched_iter(self._dataset, self._num_batches, transform)


class BaseDataLoaderAdapter:
    """Common base for adapters that bridge low-level loaders to the DataLoader protocol.

    Shared functionality: ``data_config``, ``__len__``, ``dataset``,
    ``dataloader``, and ``sharding`` properties.  Subclasses only need to
    implement ``__iter__``.
    """

    def __init__(self, data_config: _config.DataConfig, inner):
        self._data_config = data_config
        self._inner = inner

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __len__(self) -> int:
        return len(self._inner)

    @property
    def dataset(self):
        return getattr(self._inner, "dataset", None)

    @property
    def sharding(self) -> jax.sharding.Sharding | None:
        return getattr(self._inner, "sharding", None)

    @property
    def inner(self):
        return self._inner

    @property
    def unwrapped(self):
        """Unwrap all intermediate wrappers and return the innermost loader."""
        return getattr(self.inner, "unwrapped", self.inner)


class DataLoaderImpl(BaseDataLoaderAdapter):
    def __iter__(self):
        for batch in self._inner:
            yield _model.Observation.from_dict(batch), batch["actions"]


# =============================================================================
# Composable DataLoader Support
# =============================================================================

class ComposableDataLoaderWrapper(BaseDataLoaderAdapter):
    """Wrapper that makes ComposableDataLoader compatible with openpi's DataLoader protocol.

    Converts output batches from the composable dataloader to the
    ``(Observation, Actions)`` format expected by the training loop.
    """

    def __init__(
        self,
        composable_loader: composable.ComposableDataLoader,
        primary_data_config: _config.DataConfig,
        return_source: bool = False,
    ):
        super().__init__(primary_data_config, composable_loader)
        self._return_source = return_source

    def __iter__(self):
        for batch in self._inner:
            obs, actions, source = self._parse_batch(batch)
            if self._return_source and source is not None:
                yield source, (obs, actions)
            else:
                yield obs, actions

    def _parse_batch(self, batch) -> tuple[_model.Observation, Any, str | None]:
        """Parse batch into (Observation, actions, source_tag)."""
        source = None
        # Tagged batch: (source_name, payload)
        if isinstance(batch, tuple) and len(batch) == 2 and isinstance(batch[0], str):
            source, batch = batch
        # Tuple batch: (Observation or dict, actions)
        if isinstance(batch, tuple) and len(batch) == 2:
            first, actions = batch
            obs = first if isinstance(first, _model.Observation) else _model.Observation.from_dict(first)
        else:
            # Raw dict batch
            obs, actions = _model.Observation.from_dict(batch), batch["actions"]
        return obs, actions, source


def _build_composable_from_node(
    config: _config.TrainConfig,
    node: _config.ComposableNode,
    *,
    sharding: jax.sharding.Sharding | None,
    shuffle: bool,
    skip_norm_stats: bool,
    num_batches: int | None,
    return_source: bool = False,
    _is_root: bool = True,
    _depth: int = 0,
) -> tuple[composable.BaseDataLoader, _config.DataConfig]:
    """Recursively build a composable loader from a config node.

    Args:
        return_source: If True, wrap intermediate nodes with SourceTaggedDataLoader
            to enable hierarchical source names.
        _depth: Internal parameter tracking nesting depth for logging indentation.
    
    Returns:
        (loader, primary_data_config) where primary_data_config is from the first leaf.
    """
    indent = "  " * _depth
    
    if isinstance(node, _config.DataConfigFactory):
        train_config = _create_single_dataset_config(config, node)
        loader = create_data_loader(
            train_config,
            data_config_factory=node,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
        )
        data_config = loader.data_config()
        logging.info(f"{indent}  +-- Dataset: {data_config.repo_id}")
        return loader, data_config

    # node is ComposableDataConfig — extract node-level num_batches if set
    if node.num_batches is not None:
        num_batches = node.num_batches

    inputs = node.children
    if not inputs:
        raise ValueError("ComposableDataConfig has no children")

    child_loaders: list[composable.BaseDataLoader] = []
    primary_data_config: _config.DataConfig | None = None

    for i, child in enumerate(inputs):
        sub_loader, sub_data_config = _build_composable_from_node(
            config,
            child,
            sharding=sharding,
            shuffle=shuffle,
            skip_norm_stats=skip_norm_stats,
            num_batches=num_batches,
            return_source=return_source,
            _is_root=False,
            _depth=_depth + 1,
        )
        child_loaders.append(sub_loader)
        if primary_data_config is None:
            primary_data_config = sub_data_config

    strategy = node.composition_strategy
    weights = node.weights
    pattern = node.pattern
    task_names = node.task_names
    stop_strategy = node.stop_strategy
    num_loaders = len(child_loaders)

    # Single child: passthrough directly, no composition needed.
    if num_loaders == 1:
        composed = child_loaders[0]
    elif strategy == "random":
        composed = composable.Compose.random(*child_loaders, weights=weights, stop_strategy=stop_strategy)
    elif strategy == "proportional":
        composed = composable.Compose.proportional(*child_loaders, ratios=weights, stop_strategy=stop_strategy)
    elif strategy == "round_robin":
        composed = composable.Compose.round_robin(*child_loaders, stop_strategy=stop_strategy)
    elif strategy == "alternating":
        composed = composable.Compose.alternating(*child_loaders, pattern=pattern, stop_strategy=stop_strategy)
    elif strategy == "tagged":
        names = task_names or _unique_source_names(inputs)
        loaders_dict = dict(zip(names, child_loaders))
        composed = composable.Compose.tagged(loaders_dict, sampling_strategy="random", stop_strategy=stop_strategy)
    elif strategy == "dynamic":
        composed = composable.Compose.dynamic(*child_loaders, initial_weights=weights, stop_strategy=stop_strategy)
    elif strategy == "inbatch":
        samples_per_loader = _compute_samples_per_loader(weights, num_loaders, config.batch_size)
        logging.info(
            f"inbatch: num_loaders={num_loaders}, batch_size={config.batch_size}, "
            f"samples_per_loader={samples_per_loader}"
        )
        composed = composable.Compose.inbatch(
            *child_loaders,
            samples_per_loader=samples_per_loader,
            random_sample=node.inbatch_random_sample,
            stop_strategy=stop_strategy,
        )
    else:
        raise ValueError(f"Unknown composition strategy: {strategy}")

    level_marker = "[ROOT]" if _is_root else "  |--"
    if num_loaders == 1:
        logging.info(f"{indent}{level_marker} passthrough(children=1)")
    else:
        weights_info = f" weights={list(weights)}" if weights else ""
        stop_info = f" stop={stop_strategy}" if stop_strategy != composable.LONGEST else ""
        logging.info(
            f"{indent}{level_marker} Compose.{strategy}(children={num_loaders}{weights_info}{stop_info})"
        )

    # Wrap with RefreshableDataLoader if refresh_every is set at this level
    # (must be done before SourceTagged so that source tags remain consistent across epochs)
    if node.refresh_every is not None:
        composed = composable.RefreshableDataLoader(
            composed,
            on_refresh=node.on_refresh,
            refresh_every=node.refresh_every,
            num_epochs=node.num_epochs,
        )
        epochs_str = f"{node.num_epochs}" if node.num_epochs else "inf"
        logging.info(f"{indent}   -> RefreshableDataLoader(epochs={epochs_str}, refresh_every={node.refresh_every})")

    # Wrap intermediate nodes with SourceTaggedDataLoader only when return_source
    # is enabled. Root wrapping is handled by create_composable_data_loader.
    if return_source and not _is_root:
        source_names = _unique_source_names(inputs)
        composed = composable.SourceTaggedDataLoader(composed, source_names)
        logging.info(f"{indent}   -> SourceTaggedDataLoader(sources={source_names})")

    assert primary_data_config is not None
    return composed, primary_data_config


def _unique_source_names(inputs: Sequence[_config.ComposableNode]) -> list[str]:
    """Generate unique source names for child nodes, adding suffixes for duplicates.

    For DataConfigFactory leaves, uses repo_id if available.
    For ComposableDataConfig nodes, uses 'group_<index>'.
    Duplicates at the same level get '_0', '_1', etc. suffixes.
    """
    from collections import Counter

    # Collect base names
    base_names = [
        getattr(node, "repo_id", None) or f"dataset_{i}"
        if isinstance(node, _config.DataConfigFactory)
        else f"group_{i}"
        for i, node in enumerate(inputs)
    ]

    # Add suffixes for duplicates
    counts = Counter(base_names)
    seen: dict[str, int] = {}
    result = []
    for name in base_names:
        if counts[name] > 1:
            idx = seen.setdefault(name, 0)
            seen[name] += 1
            result.append(f"{name}_{idx}")
        else:
            result.append(name)
    return result


def _compute_samples_per_loader(
    weights: Sequence[float] | None,
    num_loaders: int,
    batch_size: int,
) -> list[int]:
    """Compute per-loader sample counts for inbatch mixing."""
    if weights is not None:
        w = np.array(weights, dtype=np.float64)
        w /= w.sum()
        samples = (w * batch_size).astype(np.int64)
        # Distribute remainder to last loader
        samples[-1] += batch_size - int(samples.sum())
        return samples.tolist()
    # Equal split with remainder to last
    base = batch_size // num_loaders
    samples = [base] * num_loaders
    samples[-1] += batch_size - sum(samples)
    return samples


def create_composable_data_loader(
    config: _config.TrainConfig,
    composable_config: _config.ComposableDataConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = True,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a composable data loader that combines multiple datasets (with nesting).

    Builds the loader tree recursively from ``ComposableDataConfig``:
    each node is either a dataset leaf (``DataConfigFactory``) or a nested
    ``ComposableDataConfig`` with its own children.

    Args:
        config: Training configuration.
        composable_config: Root node config, including children / strategy / weights.
        sharding: Optional JAX sharding for distributed loading.
        shuffle: Whether to shuffle at the leaf dataset level.
        num_batches: Optional total batch count for proportional strategy at root.
        skip_norm_stats: Whether to skip normalization statistics.

    Returns:
        A composed ``DataLoader`` instance.

    Example:
        >>> composable_cfg = ComposableDataConfig(
        ...     composition_strategy="proportional",
        ...     children=[
        ...         ComposableDataConfig(
        ...             composition_strategy="random",
        ...             children=[dataset_a, dataset_b],
        ...             weights=[0.6, 0.4],
        ...         ),
        ...         dataset_c,
        ...     ],
        ...     weights=[2, 1],
        ... )
        >>> loader = create_composable_data_loader(config, composable_cfg)
    """
    seed = composable_config.seed
    return_source = composable_config.return_source
    
    logging.info("=" * 70)
    logging.info("Building composable data loader tree...")
    if seed is not None:
        composable.set_seed(seed)
        logging.info(f"Set random seed: {seed}")
    
    composed, primary_data_config = _build_composable_from_node(
        config,
        composable_config,
        sharding=sharding,
        shuffle=shuffle,
        skip_norm_stats=skip_norm_stats,
        num_batches=num_batches,
        return_source=return_source,
        _depth=0,
    )

    strategy = composable_config.composition_strategy
    children = composable_config.children

    # Wrap root with SourceTaggedDataLoader if return_source is enabled
    if return_source and strategy != "tagged":
        source_names = composable_config.task_names or _unique_source_names(children)
        composed = composable.SourceTaggedDataLoader(composed, source_names)
        logging.info(f"[TAG] Root SourceTaggedDataLoader (sources={source_names})")

    logging.info("=" * 70)
    logging.info(f"[SUCCESS] Composable data loader created successfully")
    logging.info(f"  Primary dataset: {primary_data_config.repo_id}")
    logging.info(f"  Total children: {len(children)}")
    if composable_config.weights:
        logging.info(f"  Root weights: {list(composable_config.weights)}")
    if composable_config.refresh_every is not None:
        epochs_str = f"{composable_config.num_epochs}" if composable_config.num_epochs else "inf"
        logging.info(f"  Refresh config: every {composable_config.refresh_every} epochs (total: {epochs_str})")
    logging.info("=" * 70)

    return ComposableDataLoaderWrapper(composed, primary_data_config, return_source=return_source)


def _create_single_dataset_config(
    base_config: _config.TrainConfig,
    data_factory: _config.DataConfigFactory,
) -> _config.TrainConfig:
    """Create a TrainConfig for a single dataset within a composed loader.
    
    This helper creates a modified TrainConfig that uses the specified
    data factory while inheriting other settings from the base config.
    
    Important: composable_data is set to None to prevent infinite recursion
    when create_data_loader is called for individual datasets.
    """
    import dataclasses
    return dataclasses.replace(base_config, data=data_factory, composable_data=None)
