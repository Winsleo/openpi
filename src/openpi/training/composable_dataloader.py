"""Composable DataLoader Module.

This module provides flexible DataLoader composition strategies for complex
multi-dataset training scenarios. All composable loaders implement the
BaseDataLoader protocol and can be freely nested.

Supported composition strategies:
- Round-Robin: Cycle through loaders sequentially
- Random/Weighted: Sample from loaders with optional weights
- Proportional: Fixed ratio allocation of batches
- Alternating: Custom pattern-based alternation
- Task-Tagged: Add source labels to batches
- Dynamic Scheduling: Adjust weights based on training feedback
- Curriculum Learning: Gradually introduce harder data
- In-Batch Mixing: Combine samples from multiple sources in one batch

Example:
    >>> # Simple weighted mixing
    >>> loader = Compose.random(loader_a, loader_b, weights=[0.7, 0.3])
    >>> 
    >>> # Nested composition
    >>> inner = Compose.random(loader_a, loader_b)
    >>> outer = Compose.proportional(inner, loader_c, ratios=[2, 1])
    >>>
    >>> # Set seed for reproducibility
    >>> set_seed(42)
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterator, Sequence
import itertools
from typing import Dict, Optional, Protocol, TypeVar, Union, runtime_checkable
import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader, Dataset as TorchDataset
from openpi.training.pytree_utils import slice_data, concat_data


# =============================================================================
# Random Number Generator Management
# =============================================================================

# Module-level random number generator using NumPy's modern Generator API
_rng: np.random.Generator = np.random.default_rng()


def set_seed(seed: Optional[int] = None) -> np.random.Generator:
    """Set the random seed for reproducibility.
    
    This function resets the module-level random number generator with the
    given seed. All composable DataLoaders use this generator for shuffling
    and sampling operations.
    
    Args:
        seed: Random seed. If None, uses a non-deterministic seed.
        
    Returns:
        The new random number generator instance.
        
    Examples:
        >>> set_seed(42)  # For reproducible results
        >>> loader = Compose.random(loader_a, loader_b)
        >>> 
        >>> # Reset with a different seed
        >>> set_seed(123)
    """
    global _rng
    _rng = np.random.default_rng(seed)
    return _rng


def get_rng() -> np.random.Generator:
    """Get the current random number generator.
    
    Returns:
        The module-level numpy random Generator instance.
        
    Examples:
        >>> rng = get_rng()
        >>> rng.random()  # Generate a random number
    """
    return _rng

__all__ = [
    # Random seed management
    "set_seed",
    "get_rng",
    # Protocols and base classes
    "BaseDataLoader",
    "ComposableDataLoader",
    "AnyDataLoader",
    # Concrete implementations
    "RoundRobinDataLoader",
    "RandomMixDataLoader",
    "ProportionalMixDataLoader",
    "AlternatingDataLoader",
    "TaskTaggedDataLoader",
    "SourceTaggedDataLoader",
    "DynamicScheduleDataLoader",
    "InBatchMixDataLoader",
    "CurriculumDataLoader",
    # Factory
    "Compose",
]

T_co = TypeVar("T_co", covariant=True)


# =============================================================================
# Protocol Definition
# =============================================================================

@runtime_checkable
class BaseDataLoader(Protocol[T_co]):
    """Minimal DataLoader protocol requiring only iteration.
    
    This protocol defines the base interface that all DataLoaders must satisfy.
    Any object implementing __iter__ automatically conforms to this protocol.
    
    The @runtime_checkable decorator enables isinstance() checks at runtime.
    
    Examples:
        >>> isinstance(torch_loader, BaseDataLoader)
        True
        >>> isinstance(composable_loader, BaseDataLoader)
        True
    """

    def __iter__(self) -> Iterator[T_co]:
        """Return an iterator over batches."""
        ...


# Type alias for any valid DataLoader
# Using BaseDataLoader Protocol ensures compatibility with:
# - torch.utils.data.DataLoader
# - OpenPI's TorchDataLoader, RLDSDataLoader, DataLoaderImpl
# - All ComposableDataLoader subclasses
# - Any object implementing __iter__
AnyDataLoader = BaseDataLoader


# =============================================================================
# Abstract Base Class
# =============================================================================

class ComposableDataLoader(ABC, BaseDataLoader):
    """Abstract base class for composable DataLoaders.
    
    This class provides the foundation for all composition strategies.
    Subclasses must implement __iter__ and __len__.
    
    Key features:
    - Explicitly inherits BaseDataLoader protocol
    - Supports arbitrary nesting of composable loaders
    - Compatible with PyTorch DataLoader
    - Framework-agnostic design
    - ``dataloaders``: child loader storage
    - ``last_loader_idx``: tracks which child produced the last batch
    - Default ``__len__`` (sum of child lengths, override if needed)
    - ``_create_iterators`` / ``_normalize_weights`` helpers

    Inheritance hierarchy:
        BaseDataLoader (Protocol)
            ↑
        ComposableDataLoader (ABC)
            ↑
        RoundRobinDataLoader, RandomMixDataLoader, ...
    
    Examples:
        >>> mixed = RandomMixDataLoader([loader_a, loader_b], weights=[0.7, 0.3])
        >>> assert isinstance(mixed, BaseDataLoader)
        >>> for batch in mixed:
        ...     process(batch)
    """

    def __init__(self, dataloaders: Sequence[AnyDataLoader]):
        self.dataloaders = list(dataloaders)
        self._num_loaders = len(self.dataloaders)
        self._last_loader_idx: Optional[Union[int, str]] = None

    @property
    def last_loader_idx(self) -> Optional[Union[int, str]]:
        """Index (or label) of the loader that produced the last batch."""
        return self._last_loader_idx

    # ------------------------------------------------------------------
    # Helpers available to all subclasses
    # ------------------------------------------------------------------

    def _create_iterators(self) -> list:
        """Create fresh iterators from all child loaders."""
        return [iter(ld) for ld in self.dataloaders]

    @staticmethod
    def _normalize_weights(weights: Sequence[float]) -> np.ndarray:
        """Normalize weights to a probability distribution."""
        w = np.asarray(weights, dtype=np.float64)
        return w / w.sum()

    @staticmethod
    def _weighted_choice(
        num_loaders: int,
        base_weights: np.ndarray,
        active: np.ndarray,
        buf: np.ndarray,
    ) -> int:
        """Pick a loader index via weighted random choice over active loaders.

        Args:
            num_loaders: Total number of loaders.
            base_weights: Base weight array (not mutated).
            active: Boolean mask of active loaders.
            buf: Pre-allocated float64 buffer (len == num_loaders).

        Returns:
            Selected loader index, or -1 if no active loaders.
        """
        np.multiply(base_weights, active, out=buf)
        total = buf.sum()
        if total == 0:
            return -1
        buf /= total
        return int(_rng.choice(num_loaders, p=buf))

    # ------------------------------------------------------------------
    # Default implementations (override in subclasses if needed)
    # ------------------------------------------------------------------

    @abstractmethod
    def __iter__(self) -> Iterator:
        """Return an iterator over batches."""
        ...

    def __len__(self) -> int:
        """Default: sum of all child loader lengths."""
        return sum(len(ld) for ld in self.dataloaders)

    @staticmethod
    def is_dataloader(obj) -> bool:
        """Check if an object satisfies the BaseDataLoader protocol."""
        return isinstance(obj, BaseDataLoader)


# =============================================================================
# Round-Robin DataLoader
# =============================================================================

class RoundRobinDataLoader(ComposableDataLoader):
    """Cycle through DataLoaders sequentially, taking one batch from each.
    
    This loader iterates through all child loaders in order, yielding one
    batch from each before cycling back to the first.
    
    Args:
        dataloaders: Sequence of DataLoaders to cycle through.
    
    Examples:
        >>> loader = RoundRobinDataLoader([loader_a, loader_b, loader_c])
        >>> # Yields: batch_a, batch_b, batch_c, batch_a, batch_b, ...
    """

    def __init__(self, dataloaders: Sequence[AnyDataLoader]):
        super().__init__(dataloaders)

    def __iter__(self):
        iterators = self._create_iterators()
        # Use boolean mask instead of removing elements (preserves original indices)
        active = [True] * self._num_loaders
        active_count = self._num_loaders
        idx = 0

        while active_count > 0:
            # Skip inactive loaders
            while not active[idx]:
                idx = (idx + 1) % self._num_loaders
            try:
                self._last_loader_idx = idx
                yield next(iterators[idx])
            except StopIteration:
                active[idx] = False
                active_count -= 1
            idx = (idx + 1) % self._num_loaders


# =============================================================================
# Random/Weighted Mix DataLoader
# =============================================================================

class RandomMixDataLoader(ComposableDataLoader):
    """Sample batches randomly from DataLoaders with optional weights.
    
    This is the simplest approach for multi-dataset weighted sampling.
    Each batch is drawn from a randomly selected loader based on weights.
    
    Args:
        dataloaders: Sequence of DataLoaders to sample from.
        weights: Optional sampling weights. If None, uniform weights are used.
    
    Examples:
        >>> # 70% probability from loader_a, 30% from loader_b
        >>> loader = RandomMixDataLoader(
        ...     [loader_a, loader_b],
        ...     weights=[0.7, 0.3]
        ... )
    """

    def __init__(
        self,
        dataloaders: Sequence[AnyDataLoader],
        weights: Optional[Sequence[float]] = None,
    ):
        super().__init__(dataloaders)
        self.weights = self._normalize_weights(
            weights if weights is not None else [1.0] * self._num_loaders
        )

    def __iter__(self):
        iterators = self._create_iterators()
        active = np.ones(self._num_loaders, dtype=bool)
        buf = np.empty(self._num_loaders, dtype=np.float64)

        while active.any():
            idx = self._weighted_choice(self._num_loaders, self.weights, active, buf)
            if idx < 0:
                break
            try:
                self._last_loader_idx = idx
                yield next(iterators[idx])
            except StopIteration:
                active[idx] = False


# =============================================================================
# Proportional Mix DataLoader
# =============================================================================

class ProportionalMixDataLoader(ComposableDataLoader):
    """Allocate a fixed number of batches to each DataLoader by ratio.
    
    Unlike RandomMixDataLoader which samples probabilistically, this loader
    guarantees exact batch counts per loader based on the specified ratios.
    
    Uses an efficient online sampling algorithm (O(1) memory per step) instead
    of pre-generating the full schedule.
    
    Args:
        dataloaders: Sequence of DataLoaders.
        ratios: Ratio of batches for each loader. If None, equal ratios.
        max_batches: Total number of batches. If None, uses sum of loader lengths.
    
    Examples:
        >>> # Exactly 75% from loader_a, 25% from loader_b
        >>> loader = ProportionalMixDataLoader(
        ...     [loader_a, loader_b],
        ...     ratios=[3, 1],
        ...     max_batches=100
        ... )
    """

    def __init__(
        self,
        dataloaders: Sequence[AnyDataLoader],
        ratios: Optional[Sequence[float]] = None,
        max_batches: Optional[int] = None,
    ):
        super().__init__(dataloaders)
        ratios_arr = self._normalize_weights(
            ratios if ratios is not None else [1.0] * self._num_loaders
        )
        total_batches = max_batches or sum(len(ld) for ld in self.dataloaders)
        self.batches_per_loader = (ratios_arr * total_batches).astype(int)
        # Distribute remainder to last loader
        self.batches_per_loader[-1] += total_batches - int(self.batches_per_loader.sum())
        self._total_batches = int(self.batches_per_loader.sum())

    def __iter__(self):
        iterators = self._create_iterators()
        remaining = self.batches_per_loader.astype(np.float64)
        buf = np.empty(self._num_loaders, dtype=np.float64)

        for _ in range(self._total_batches):
            total = remaining.sum()
            if total <= 0:
                break
            np.divide(remaining, total, out=buf)
            idx = int(_rng.choice(self._num_loaders, p=buf))
            remaining[idx] -= 1

            try:
                batch = next(iterators[idx])
            except StopIteration:
                iterators[idx] = iter(self.dataloaders[idx])
                batch = next(iterators[idx])

            self._last_loader_idx = idx
            yield batch

    def __len__(self):
        return self._total_batches


# =============================================================================
# Alternating DataLoader
# =============================================================================

class AlternatingDataLoader(ComposableDataLoader):
    """Alternate between DataLoaders using a custom pattern.
    
    Args:
        dataloaders: Sequence of DataLoaders.
        pattern: Index pattern for alternation. If None, cycles through indices.
    
    Examples:
        >>> # Pattern: A, A, B, A, A, B, ...
        >>> loader = AlternatingDataLoader(
        ...     [loader_a, loader_b],
        ...     pattern=[0, 0, 1]
        ... )
    """

    def __init__(
        self,
        dataloaders: Sequence[AnyDataLoader],
        pattern: Optional[Sequence[int]] = None,
    ):
        super().__init__(dataloaders)
        self.pattern = list(pattern) if pattern else list(range(self._num_loaders))

    def __iter__(self):
        iterators = self._create_iterators()
        pattern_cycle = itertools.cycle(self.pattern)
        total = len(self)
        yielded = 0

        while yielded < total:
            idx = next(pattern_cycle)
            try:
                self._last_loader_idx = idx
                yield next(iterators[idx])
                yielded += 1
            except StopIteration:
                continue


# =============================================================================
# Task-Tagged DataLoader
# =============================================================================

class TaskTaggedDataLoader(ComposableDataLoader):
    """Add task/source labels to each batch for multi-task learning.
    
    Yields (task_name, batch) tuples, enabling task-specific processing
    in the training loop.
    
    Args:
        dataloaders: Dictionary mapping task names to DataLoaders.
        sampling_strategy: Either 'random' or 'round_robin'.
        propagate_tags: If True, preserve nested tags as "parent/child".
    
    Examples:
        >>> loader = TaskTaggedDataLoader({
        ...     'task_a': loader_a,
        ...     'task_b': loader_b,
        ... }, sampling_strategy='random')
        >>> 
        >>> for task_name, batch in loader:
        ...     print(f"Batch from {task_name}")
    """

    def __init__(
        self,
        dataloaders: Dict[str, AnyDataLoader],
        sampling_strategy: str = 'random',
        propagate_tags: bool = False,
    ):
        # TaskTaggedDataLoader uses a dict; store list of values for base class
        super().__init__(list(dataloaders.values()))
        self._dataloaders_dict = dataloaders
        self.task_names = list(dataloaders.keys())
        self._task_to_idx = {name: i for i, name in enumerate(self.task_names)}
        self.sampling_strategy = sampling_strategy
        self.propagate_tags = propagate_tags

    def __iter__(self):
        iterators = {name: iter(ld) for name, ld in self._dataloaders_dict.items()}
        task_cycle = itertools.cycle(self.task_names) if self.sampling_strategy == 'round_robin' else None

        active = [True] * self._num_loaders
        active_count = self._num_loaders
        active_list = self.task_names.copy()

        while active_count > 0:
            if self.sampling_strategy == 'random':
                task = str(_rng.choice(active_list))
            elif task_cycle is not None:
                task = next(task_cycle)
                while not active[self._task_to_idx[task]]:
                    task = next(task_cycle)
            else:
                raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")

            task_idx = self._task_to_idx[task]
            try:
                batch = next(iterators[task])
                self._last_loader_idx = task_idx

                if self.propagate_tags and isinstance(batch, tuple) and len(batch) == 2 and isinstance(batch[0], str):
                    yield f"{task}/{batch[0]}", batch[1]
                else:
                    yield task, batch
            except StopIteration:
                active[task_idx] = False
                active_count -= 1
                active_list = [self.task_names[i] for i in range(self._num_loaders) if active[i]]


# =============================================================================
# Source-Tagged Wrapper
# =============================================================================

class SourceTaggedDataLoader(ComposableDataLoader):
    """Wrap a DataLoader and tag each batch with its source name.

    This wrapper is used at any level of a composed data loader tree to
    attach source names. When stacked, it builds hierarchical source names
    like ``"group_a/dataset_1"``.

    Args:
        dataloader: The DataLoader to wrap.
        source_names: Names for each child loader (by index).

    Notes:
        - Maps ``last_loader_idx`` to ``source_names``.
        - If the inner loader already yields ``(tag, batch)``, this wrapper
          prefixes the resolved parent name: ``f"{parent}/{tag}"``.
    """

    def __init__(self, dataloader: ComposableDataLoader, source_names: Sequence[str]):
        # SourceTaggedDataLoader wraps a single loader, not a list
        super().__init__([dataloader])
        self.dataloader = dataloader
        self.source_names = list(source_names)

    @property
    def last_loader_idx(self) -> Optional[Union[int, str]]:
        """Forward last_loader_idx from wrapped loader."""
        return getattr(self.dataloader, "last_loader_idx", self._last_loader_idx)

    def _resolve_source_name(self, idx: Optional[Union[int, str]]) -> str:
        if isinstance(idx, str):
            return idx
        if isinstance(idx, int) and 0 <= idx < len(self.source_names):
            return self.source_names[idx]
        if idx is None and len(self.source_names) == 1:
            return self.source_names[0]
        return "unknown" if idx is None else f"dataset_{idx}"

    def __iter__(self):
        for batch in self.dataloader:
            idx = getattr(self.dataloader, "last_loader_idx", None)
            self._last_loader_idx = idx
            parent_name = self._resolve_source_name(idx)

            # Already tagged: (tag, payload) -> prefix with parent name
            if isinstance(batch, tuple) and len(batch) == 2 and isinstance(batch[0], (int, str)):
                tag, payload = batch
                if parent_name != "unknown" and isinstance(tag, str) and not tag.startswith(f"{parent_name}/"):
                    tag = f"{parent_name}/{tag}"
                yield tag, payload
            else:
                # Untagged batch: attach parent name directly.
                yield parent_name, batch

    def __len__(self):
        return len(self.dataloader)


# =============================================================================
# Dynamic Schedule DataLoader
# =============================================================================

class DynamicScheduleDataLoader(ComposableDataLoader):
    """Dynamically adjust sampling weights based on training feedback.
    
    This loader enables adaptive sampling strategies where weights are
    updated during training based on performance metrics like loss.
    
    Args:
        dataloaders: Sequence of DataLoaders.
        initial_weights: Starting weights. If None, uniform weights.
        enable_tracking: If True, track statistics and allow weight updates.
    
    Attributes:
        weights: Current sampling weights (numpy array).
        last_loader_idx: Index of the loader that produced the last batch.
    
    Examples:
        >>> loader = DynamicScheduleDataLoader(
        ...     [loader_a, loader_b],
        ...     initial_weights=[0.5, 0.5]
        ... )
        >>> 
        >>> for batch in loader:
        ...     loss = train_step(batch)
        ...     loader.update_weights(loader.last_loader_idx, loss)
    """

    def __init__(
        self,
        dataloaders: Sequence[AnyDataLoader],
        initial_weights: Optional[Sequence[float]] = None,
        enable_tracking: bool = True,
    ):
        super().__init__(dataloaders)
        self.weights = self._normalize_weights(
            initial_weights if initial_weights is not None else [1.0] * self._num_loaders
        )
        self.enable_tracking = enable_tracking
        self.loader_performance: Dict[int, list] = defaultdict(list)
        self.batches_from_loader = [0] * self._num_loaders

    def update_weights(
        self, 
        loader_idx: int, 
        performance: float, 
        learning_rate: float = 0.1,
    ):
        """Update sampling weights based on performance feedback.
        
        The current strategy increases weights for loaders with lower
        performance values (e.g., lower loss = better = higher weight).
        
        Args:
            loader_idx: Index of the DataLoader.
            performance: Performance metric (lower is better, e.g., loss).
            learning_rate: Rate of weight adjustment.
        """
        if not self.enable_tracking:
            return
        self.loader_performance[loader_idx].append(performance)

        avg_perfs = np.array([
            np.mean(self.loader_performance.get(i, [0.5]))
            for i in range(self._num_loaders)
        ])
        if avg_perfs.max() > avg_perfs.min():
            normalized = (avg_perfs - avg_perfs.min()) / (avg_perfs.max() - avg_perfs.min())
            target = (1 - normalized)
            target /= target.sum()
            self.weights = (1 - learning_rate) * self.weights + learning_rate * target

    def __iter__(self):
        iterators = self._create_iterators()
        active = np.ones(self._num_loaders, dtype=bool)
        buf = np.empty(self._num_loaders, dtype=np.float64)

        while active.any():
            idx = self._weighted_choice(self._num_loaders, self.weights, active, buf)
            if idx < 0:
                break
            try:
                self._last_loader_idx = idx
                yield next(iterators[idx])
                if self.enable_tracking:
                    self.batches_from_loader[idx] += 1
            except StopIteration:
                active[idx] = False

    def get_statistics(self) -> dict:
        """Get usage statistics.
        
        Returns:
            Dictionary containing batches_per_loader, current_weights,
            and performance_history.
        """
        return {
            'batches_per_loader': self.batches_from_loader,
            'current_weights': self.weights.tolist(),
            'performance_history': dict(self.loader_performance),
        }


# =============================================================================
# In-Batch Mix DataLoader
# =============================================================================

def _get_batch_size(data) -> int:
    """Get batch size from data (typically actions/labels which are always arrays)."""
    if hasattr(data, 'shape'):
        return data.shape[0]
    if hasattr(data, '__len__'):
        return len(data)
    raise TypeError(f"Cannot determine batch size for type {type(data)}")


class InBatchMixDataLoader(ComposableDataLoader):
    """Mix samples from multiple DataLoaders within a single batch.
    
    Supports multiple batch formats:
    - Standard PyTorch: (tensor_data, tensor_labels)
    - OpenPI format: (Observation, Actions)
    - Tagged batches: (tag, (data, labels))
    
    Args:
        dataloaders: Sequence of DataLoaders.
        samples_per_loader: Number of samples from each loader per batch.
        total_batch_size: Alternative to samples_per_loader; divided equally.
        random_sample: If True, randomly sample from each batch when it's larger
            than needed. If False, take the first N samples. Default False.
    
    Examples:
        >>> # Each batch contains 16 samples from each loader
        >>> loader = InBatchMixDataLoader(
        ...     [loader_a, loader_b],
        ...     samples_per_loader=[16, 16]
        ... )
    """

    def __init__(
        self,
        dataloaders: Sequence[AnyDataLoader],
        samples_per_loader: Optional[Sequence[int]] = None,
        total_batch_size: Optional[int] = None,
        random_sample: bool = False,
    ):
        super().__init__(dataloaders)
        self._random_sample = random_sample
        self._sharding = next(
            (getattr(ld, "sharding", None) for ld in self.dataloaders if getattr(ld, "sharding", None) is not None),
            None,
        )
        # Lazy-load apply_sharding only when sharding is actually used
        self._apply_sharding = None
        if self._sharding is not None:
            from openpi.training.pytree_utils import apply_sharding
            self._apply_sharding = apply_sharding

        if samples_per_loader is None:
            if total_batch_size is None:
                total_batch_size = sum(getattr(ld, 'batch_size', 32) for ld in dataloaders)
            samples_per_loader = [total_batch_size // self._num_loaders] * self._num_loaders
            samples_per_loader[-1] = total_batch_size - sum(samples_per_loader[:-1])

        self.samples_per_loader = list(samples_per_loader)
        self.total_batch_size = sum(samples_per_loader)

    @staticmethod
    def _extract_batch(batch) -> Optional[tuple]:
        """Extract (data, labels) from various batch formats."""
        if not isinstance(batch, tuple) or len(batch) != 2:
            return None
        first, second = batch
        if isinstance(first, (int, str)) and isinstance(second, tuple) and len(second) == 2:
            # Tagged batch: (tag, (data, labels))
            return second
        # Regular batch: (data, labels)
        return batch

    def _slice_batch(self, data, labels, num_samples: int):
        """Slice data and labels to num_samples."""
        batch_size = _get_batch_size(labels)
        if batch_size <= num_samples:
            return data, labels
        if self._random_sample:
            indices = _rng.choice(batch_size, size=num_samples, replace=False)
        else:
            indices = np.arange(num_samples)
        return slice_data(data, indices), slice_data(labels, indices)

    def __iter__(self):
        iterators = self._create_iterators()
        data_parts: list = [None] * self._num_loaders
        label_parts: list = [None] * self._num_loaders

        while True:
            all_valid = True
            for idx in range(self._num_loaders):
                try:
                    batch = next(iterators[idx])
                except StopIteration:
                    return
                extracted = self._extract_batch(batch)
                if extracted is None:
                    all_valid = False
                    break
                data_parts[idx], label_parts[idx] = self._slice_batch(
                    *extracted, self.samples_per_loader[idx]
                )

            if not all_valid:
                continue

            combined_data = concat_data(data_parts)
            combined_labels = concat_data(label_parts)
            if self._apply_sharding is not None:
                combined_data = self._apply_sharding(combined_data, self._sharding)
                combined_labels = self._apply_sharding(combined_labels, self._sharding)

            self._last_loader_idx = "mixed"
            yield combined_data, combined_labels

    def __len__(self):
        return min(len(ld) for ld in self.dataloaders)


# =============================================================================
# Curriculum Learning DataLoader
# =============================================================================

class CurriculumDataLoader(ComposableDataLoader):
    """Gradually introduce harder data through training stages.
    
    Implements curriculum learning by controlling which DataLoaders are
    active at each stage of training.
    
    Args:
        dataloaders: Sequence of DataLoaders (typically ordered by difficulty).
        stages: List of lists, where each inner list contains indices of
            active loaders for that stage.
        batches_per_stage: Number of batches to yield at each stage.
    
    Attributes:
        current_stage: Current stage index (0-based).
        batches_in_stage: Number of batches yielded in current stage.
    
    Examples:
        >>> loader = CurriculumDataLoader(
        ...     [easy_loader, medium_loader, hard_loader],
        ...     stages=[
        ...         [0],        # Stage 1: easy only
        ...         [0, 1],     # Stage 2: easy + medium
        ...         [0, 1, 2],  # Stage 3: all
        ...     ],
        ...     batches_per_stage=[100, 200, 300]
        ... )
    """

    def __init__(
        self,
        dataloaders: Sequence[AnyDataLoader],
        stages: Sequence[Sequence[int]],
        batches_per_stage: Sequence[int],
    ):
        super().__init__(dataloaders)
        self.stages = [list(s) for s in stages]
        self.batches_per_stage = list(batches_per_stage)
        self.current_stage = 0
        self.batches_in_stage = 0

    def __iter__(self):
        iterators = self._create_iterators()
        total = len(self)
        yielded = 0

        while yielded < total:
            idx = int(_rng.choice(self.stages[self.current_stage]))
            try:
                self._last_loader_idx = idx
                yield next(iterators[idx])
                yielded += 1
                self.batches_in_stage += 1
                if (self.batches_in_stage >= self.batches_per_stage[self.current_stage]
                        and self.current_stage < len(self.stages) - 1):
                    self.current_stage += 1
                    self.batches_in_stage = 0
            except StopIteration:
                # Restart exhausted loader
                iterators[idx] = iter(self.dataloaders[idx])

    def __len__(self):
        return sum(self.batches_per_stage)


# =============================================================================
# Composition Factory
# =============================================================================

class Compose:
    """Factory class providing a fluent API for creating composed DataLoaders.
    
    Examples:
        >>> # Weighted random mixing
        >>> loader = Compose.random(loader_a, loader_b, weights=[0.7, 0.3])
        >>> 
        >>> # Round-robin cycling
        >>> loader = Compose.round_robin(loader_a, loader_b, loader_c)
        >>> 
        >>> # Proportional allocation
        >>> loader = Compose.proportional(loader_a, loader_b, ratios=[3, 1])
        >>> 
        >>> # Task tagging
        >>> loader = Compose.tagged({'task_a': loader_a, 'task_b': loader_b})
        >>> 
        >>> # Nested composition
        >>> inner = Compose.random(loader_a, loader_b, weights=[0.6, 0.4])
        >>> outer = Compose.proportional(inner, loader_c, ratios=[2, 1])
    """
    
    @staticmethod
    def round_robin(*loaders: AnyDataLoader) -> RoundRobinDataLoader:
        """Create a round-robin DataLoader."""
        return RoundRobinDataLoader(list(loaders))
    
    @staticmethod
    def random(
        *loaders: AnyDataLoader, 
        weights: Optional[Sequence[float]] = None
    ) -> RandomMixDataLoader:
        """Create a weighted random mixing DataLoader."""
        return RandomMixDataLoader(list(loaders), weights=weights)
    
    @staticmethod
    def proportional(
        *loaders: AnyDataLoader,
        ratios: Optional[Sequence[float]] = None,
        max_batches: Optional[int] = None
    ) -> ProportionalMixDataLoader:
        """Create a proportional allocation DataLoader."""
        return ProportionalMixDataLoader(list(loaders), ratios=ratios, max_batches=max_batches)
    
    @staticmethod
    def alternating(
        *loaders: AnyDataLoader,
        pattern: Optional[Sequence[int]] = None
    ) -> AlternatingDataLoader:
        """Create an alternating pattern DataLoader."""
        return AlternatingDataLoader(list(loaders), pattern=pattern)
    
    @staticmethod
    def tagged(
        loaders_dict: Dict[str, AnyDataLoader],
        sampling_strategy: str = 'random',
        propagate_tags: bool = False
    ) -> TaskTaggedDataLoader:
        """Create a task-tagged DataLoader."""
        return TaskTaggedDataLoader(loaders_dict, sampling_strategy, propagate_tags)
    
    @staticmethod
    def dynamic(
        *loaders: AnyDataLoader,
        initial_weights: Optional[Sequence[float]] = None,
        enable_tracking: bool = True
    ) -> DynamicScheduleDataLoader:
        """Create a dynamic scheduling DataLoader."""
        return DynamicScheduleDataLoader(list(loaders), initial_weights, enable_tracking)
    
    @staticmethod
    def curriculum(
        *loaders: AnyDataLoader,
        stages: Sequence[Sequence[int]],
        batches_per_stage: Sequence[int]
    ) -> CurriculumDataLoader:
        """Create a curriculum learning DataLoader."""
        return CurriculumDataLoader(list(loaders), stages, batches_per_stage)
    
    @staticmethod
    def inbatch(
        *loaders: AnyDataLoader,
        samples_per_loader: Optional[Sequence[int]] = None,
        random_sample: bool = False
    ) -> InBatchMixDataLoader:
        """Create an in-batch mixing DataLoader."""
        return InBatchMixDataLoader(list(loaders), samples_per_loader, random_sample=random_sample)


# =============================================================================
# Examples and Testing
# =============================================================================

def _create_test_loaders() -> Dict[str, TorchDataLoader]:
    """Create test DataLoaders for examples."""
    
    class SimpleDataset(TorchDataset):
        def __init__(self, size: int, label: str = ""):
            self.size = size
            self.label = label
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            return {"data": torch.randn(10), "label": self.label, "idx": idx}
    
    return {
        "easy": TorchDataLoader(SimpleDataset(100, "easy"), batch_size=8),
        "medium": TorchDataLoader(SimpleDataset(80, "medium"), batch_size=8),
        "hard": TorchDataLoader(SimpleDataset(60, "hard"), batch_size=8),
    }


def example_basic_usage():
    """Example 1: Basic composition patterns."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    loaders = _create_test_loaders()
    easy, medium, hard = loaders["easy"], loaders["medium"], loaders["hard"]
    
    # 1.1 Weighted random mixing
    print("\n[1.1] Random Mix - 70% easy, 30% medium")
    mixed = Compose.random(easy, medium, weights=[0.7, 0.3])
    it = iter(mixed)
    batches = [next(it) for _ in range(5)]
    print(f"  5 batches from: {[b['label'][0] for b in batches]}")
    
    # 1.2 Round-robin
    print("\n[1.2] Round Robin - cycle through easy, medium, hard")
    robin = Compose.round_robin(easy, medium, hard)
    it = iter(robin)
    batches = [next(it) for _ in range(6)]
    print(f"  6 batches from: {[b['label'][0] for b in batches]}")
    
    # 1.3 Proportional
    print("\n[1.3] Proportional - easy:hard = 3:1, 20 batches total")
    prop = Compose.proportional(easy, hard, ratios=[3, 1], max_batches=20)
    print(f"  Allocation: easy={prop.batches_per_loader[0]}, hard={prop.batches_per_loader[1]}")


def example_nested_composition():
    """Example 2: Nested DataLoader composition."""
    print("\n" + "=" * 60)
    print("Example 2: Nested Composition")
    print("=" * 60)
    
    loaders = _create_test_loaders()
    easy, medium, hard = loaders["easy"], loaders["medium"], loaders["hard"]
    
    print("\nStructure: Proportional( Random(easy, medium), hard )")
    print("           Outer: 2:1 ratio")
    print("           Inner: 60:40 weighted random")
    
    inner = Compose.random(easy, medium, weights=[0.6, 0.4])
    outer = Compose.proportional(inner, hard, ratios=[2, 1], max_batches=30)
    
    print(f"\n  Total batches: {len(outer)}")
    print(f"  Allocation: inner={outer.batches_per_loader[0]}, hard={outer.batches_per_loader[1]}")
    
    count = sum(1 for _ in outer)
    print(f"  Actual iteration: {count} batches")


def example_dynamic_scheduling():
    """Example 3: Dynamic weight adjustment based on loss."""
    print("\n" + "=" * 60)
    print("Example 3: Dynamic Scheduling")
    print("=" * 60)
    
    loaders = _create_test_loaders()
    easy, hard = loaders["easy"], loaders["hard"]
    
    dynamic = DynamicScheduleDataLoader(
        [easy, hard],
        initial_weights=[0.5, 0.5],
        enable_tracking=True
    )
    
    print("\nSimulating training with dynamic weight adjustment:")
    print(f"  Initial weights: {dynamic.weights.tolist()}")
    
    for i, batch in enumerate(dynamic):
        if i >= 10:
            break
        
        # Simulate: easy data has low loss, hard data has high loss
        idx = dynamic.last_loader_idx
        fake_loss = 0.2 if idx == 0 else 0.8
        
        dynamic.update_weights(idx, fake_loss)
        
        if i % 3 == 0:
            print(f"  Step {i}: loader={idx}, loss={fake_loss:.1f}, "
                  f"weights={dynamic.weights.round(2).tolist()}")
    
    print(f"\n  Final weights: {dynamic.weights.round(2).tolist()}")
    print("  (easy weight increased due to lower loss)")


def example_curriculum_learning():
    """Example 4: Curriculum learning with staged data introduction."""
    print("\n" + "=" * 60)
    print("Example 4: Curriculum Learning")
    print("=" * 60)
    
    loaders = _create_test_loaders()
    easy, medium, hard = loaders["easy"], loaders["medium"], loaders["hard"]
    
    curriculum = CurriculumDataLoader(
        [easy, medium, hard],
        stages=[
            [0],        # Stage 1: easy only
            [0, 1],     # Stage 2: easy + medium
            [0, 1, 2],  # Stage 3: all
        ],
        batches_per_stage=[5, 5, 5]
    )
    
    print("\nCurriculum design:")
    print("  Stage 1 (5 batches): easy only")
    print("  Stage 2 (5 batches): easy + medium")
    print("  Stage 3 (5 batches): easy + medium + hard")
    
    print(f"\nIteration:")
    for i, batch in enumerate(curriculum):
        stage = curriculum.current_stage + 1
        label = batch["label"][0]
        if i % 3 == 0 or label == "hard":
            print(f"  Batch {i:2d}: Stage {stage}, data={label}")


def example_task_tagging():
    """Example 5: Task-tagged batches for multi-task learning."""
    print("\n" + "=" * 60)
    print("Example 5: Task Tagging")
    print("=" * 60)
    
    loaders = _create_test_loaders()
    
    tagged = TaskTaggedDataLoader(
        {"task_A": loaders["easy"], "task_B": loaders["hard"]},
        sampling_strategy="round_robin"
    )
    
    print("\nTagged batches:")
    for i, (tag, batch) in enumerate(tagged):
        if i >= 6:
            break
        print(f"  Batch {i}: tag='{tag}', data={batch['label'][0]}")


def example_training_loop():
    """Example 6: Complete training loop with composed DataLoader."""
    print("\n" + "=" * 60)
    print("Example 6: Training Loop")
    print("=" * 60)
    
    loaders = _create_test_loaders()
    
    # Build composition: dynamic source mixing with proportional target
    source_mix = Compose.dynamic(
        loaders["easy"], loaders["medium"],
        initial_weights=[0.5, 0.5],
        enable_tracking=False
    )
    
    train_loader = Compose.proportional(
        source_mix, loaders["hard"],
        ratios=[7, 3],
        max_batches=20
    )
    
    print("\nComposition structure:")
    print("  Proportional(")
    print("    Dynamic(easy, medium),  # source domain")
    print("    hard                    # target domain")
    print("  )  # ratio 7:3")
    
    print(f"\nTraining ({len(train_loader)} batches):")
    
    for epoch in range(2):
        total_loss = 0.0
        for batch in train_loader:
            fake_loss = _rng.random()
            total_loss += fake_loss
        
        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch+1}: avg_loss = {avg_loss:.3f}")
    
    print("\nTraining complete!")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ComposableDataLoader Examples")
    print("=" * 60)
    set_seed(42)
    example_basic_usage()
    example_nested_composition()
    example_dynamic_scheduling()
    example_curriculum_learning()
    example_task_tagging()
    example_training_loop()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
    print("\nQuick Reference:")
    print("  Compose.random(a, b, weights=[0.7, 0.3])  # weighted random")
    print("  Compose.round_robin(a, b, c)              # round-robin")
    print("  Compose.proportional(a, b, ratios=[3, 1]) # proportional")
    print("  Compose.dynamic(a, b)                     # dynamic scheduling")
    print("  Compose.curriculum(a, b, c, ...)          # curriculum learning")
    print("  Compose.tagged({'x': a, 'y': b})          # task tagging")
