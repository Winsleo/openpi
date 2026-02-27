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
from collections.abc import Callable, Iterator, Sequence
import itertools
import logging
from typing import Any, Optional, Protocol, TypeVar, Union, runtime_checkable
import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader, Dataset as TorchDataset
from openpi.training.loader_ident import get_loader_ident
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
    # Batch-count sentinels
    "LONGEST",
    "SHORTEST",
    # Composition strategies
    "VALID_COMPOSITION_STRATEGIES",
    # Protocols and base classes
    "BaseDataLoader",
    "ComposableDataLoader",
    "MultiSourceDataLoader",
    "SingleLoaderWrapper",
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
    "RefreshableDataLoader",
    # Factory
    "Compose",
]

# =============================================================================
# Stop Strategy Sentinels
# =============================================================================

LONGEST: str = "longest"
"""Use as ``stop_strategy=LONGEST`` — iteration continues until **every** child
loader is fully traversed at least once.  Shorter loaders restart as needed."""

SHORTEST: str = "shortest"
"""Use as ``stop_strategy=SHORTEST`` — iteration stops as soon as the first
child loader would exhaust.  No child loader needs to restart."""

# =============================================================================
# Composition Strategy Constants
# =============================================================================

VALID_COMPOSITION_STRATEGIES = frozenset({
    "random",
    "proportional",
    "round_robin",
    "alternating",
    "tagged",
    "dynamic",
    "inbatch",
})
"""Valid composition strategy names for ComposableDataConfig.

These correspond to the factory methods in the Compose class:
- "random" → Compose.random()
- "proportional" → Compose.proportional()
- "round_robin" → Compose.round_robin()
- "alternating" → Compose.alternating()
- "tagged" → Compose.tagged()
- "dynamic" → Compose.dynamic()
- "inbatch" → Compose.inbatch()
"""

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
# Utility Functions
# =============================================================================

def _normalize_weights(weights: Sequence[float]) -> np.ndarray:
    """Normalize *weights* to a probability distribution (sums to 1).

    Raises:
        ValueError: If any weight is negative or all weights are zero.
    """
    w = np.asarray(weights, dtype=np.float64)
    if (w < 0).any():
        raise ValueError(f"Weights must be non-negative, got {weights}")
    total = w.sum()
    if total == 0:
        raise ValueError("Weights must not all be zero")
    return w / total


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


# =============================================================================
# Abstract Base Class
# =============================================================================

class ComposableDataLoader(ABC, BaseDataLoader):
    """Minimal abstract base class for all composable DataLoaders.

    Defines only the iteration contract (``__iter__``, ``__len__``) and
    ``last_loader_idx`` tracking.  Concrete behaviour is added by the two
    intermediate bases:

    - ``MultiSourceDataLoader`` — for loaders that compose *multiple*
      child DataLoaders (round-robin, random mix, proportional, …).
    - ``SingleLoaderWrapper`` — for loaders that wrap *one* child
      (source-tagging, refreshable epochs, …).

    Inheritance hierarchy::

        BaseDataLoader (Protocol)
            ↑
        ComposableDataLoader (ABC)
            ├── MultiSourceDataLoader
            │   ├── RoundRobinDataLoader
            │   ├── RandomMixDataLoader
            │   └── …
            └── SingleLoaderWrapper
                ├── SourceTaggedDataLoader
                └── RefreshableDataLoader
    """

    def __init__(self):
        self._last_loader_idx: Optional[Union[int, str, dict[int, int]]] = None

    @property
    def last_loader_idx(self) -> Optional[Union[int, str, dict[int, int]]]:
        """Index, label, or composition dict of the loader that produced the last batch.

        - ``int`` — single-source loaders (index of the child).
        - ``str`` — e.g. ``"mixed"`` when no further detail is needed.
        - ``dict[int, int]`` — in-batch mix: ``{child_idx: sample_count}``.
        """
        return self._last_loader_idx

    @abstractmethod
    def __iter__(self) -> Iterator:
        """Return an iterator over batches."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of batches."""
        ...

    @staticmethod
    def is_dataloader(obj) -> bool:
        """Check if an object satisfies the BaseDataLoader protocol."""
        return isinstance(obj, BaseDataLoader)


# =============================================================================
# Multi-Source Base Class
# =============================================================================

class MultiSourceDataLoader(ComposableDataLoader):
    """Intermediate base for loaders that compose *multiple* child DataLoaders.

    Provides:

    - ``dataloaders`` / ``_num_loaders`` — child storage and count.
    - ``_stop_strategy`` — ``LONGEST`` or ``SHORTEST`` termination control.
    - ``_create_iterators()`` — convenience to build fresh iterators.
    - A default ``__len__`` (sum of child lengths; override if needed).
    """

    def __init__(
        self,
        dataloaders: Sequence[AnyDataLoader],
        stop_strategy: str = LONGEST,
    ):
        super().__init__()
        self.dataloaders = list(dataloaders)
        if not self.dataloaders:
            raise ValueError("dataloaders must not be empty")
        if stop_strategy not in (LONGEST, SHORTEST):
            raise ValueError(
                f"stop_strategy must be LONGEST or SHORTEST, got {stop_strategy!r}"
            )
        self._num_loaders = len(self.dataloaders)
        self._stop_strategy = stop_strategy

    def _create_iterators(self) -> list:
        """Create fresh iterators from all child loaders."""
        return [iter(ld) for ld in self.dataloaders]

    def __len__(self) -> int:
        """Default length estimate based on stop strategy.

        LONGEST: sum of all child lengths (every loader fully consumed).
        SHORTEST: min child length × number of loaders (stop when first exhausts).
        """
        if self._stop_strategy == SHORTEST:
            return min(len(ld) for ld in self.dataloaders) * self._num_loaders
        return sum(len(ld) for ld in self.dataloaders)

    @property
    def sharding(self):
        """Return the first non-None sharding found among child loaders."""
        for ld in self.dataloaders:
            s = getattr(ld, "sharding", None)
            if s is not None:
                return s
        return None


# =============================================================================
# Single-Loader Wrapper Base Class
# =============================================================================

class SingleLoaderWrapper(ComposableDataLoader):
    """Intermediate base for loaders that wrap a *single* child DataLoader.

    ``last_loader_idx`` automatically delegates to the inner loader,
    enabling transparent pass-through of source tracking in nested
    compositions.
    """

    def __init__(self, dataloader: AnyDataLoader):
        super().__init__()
        self._inner = dataloader

    @property
    def inner(self) -> AnyDataLoader:
        """The wrapped inner DataLoader."""
        return self._inner

    @inner.setter
    def inner(self, dataloader: AnyDataLoader) -> None:
        self._inner = dataloader

    @property
    def last_loader_idx(self) -> Optional[Union[int, str, dict[int, int]]]:
        """Delegate to the inner loader's ``last_loader_idx``."""
        return getattr(self._inner, "last_loader_idx", self._last_loader_idx)

    @property
    def dataset(self):
        return getattr(self._inner, "dataset", None)

    @property
    def sharding(self):
        return getattr(self._inner, "sharding", None)

    @property
    def unwrapped(self):
        """Unwrap all intermediate wrappers and return the innermost loader."""
        return getattr(self.inner, "unwrapped", self.inner)


# =============================================================================
# Round-Robin DataLoader
# =============================================================================

class RoundRobinDataLoader(MultiSourceDataLoader):
    """Cycle through DataLoaders sequentially, taking one batch from each.
    
    This loader iterates through all child loaders in order, yielding one
    batch from each before cycling back to the first.
    
    Args:
        dataloaders: Sequence of DataLoaders to cycle through.
        stop_strategy: ``LONGEST`` (default) — continue until all loaders
            exhaust.  ``SHORTEST`` — stop as soon as any loader exhausts.
    
    Examples:
        >>> loader = RoundRobinDataLoader([loader_a, loader_b, loader_c])
        >>> # Yields: batch_a, batch_b, batch_c, batch_a, batch_b, ...
    """

    def __init__(self, dataloaders: Sequence[AnyDataLoader], stop_strategy: str = LONGEST):
        super().__init__(dataloaders, stop_strategy)

    def __iter__(self):
        iterators = self._create_iterators()
        active = [True] * self._num_loaders
        active_count = self._num_loaders
        idx = 0

        while active_count > 0:
            while not active[idx]:
                idx = (idx + 1) % self._num_loaders
            try:
                batch = next(iterators[idx])
            except StopIteration:
                if self._stop_strategy == SHORTEST:
                    return
                active[idx] = False
                active_count -= 1
                idx = (idx + 1) % self._num_loaders
                continue
            self._last_loader_idx = idx
            yield batch
            idx = (idx + 1) % self._num_loaders


# =============================================================================
# Random/Weighted Mix DataLoader
# =============================================================================

class RandomMixDataLoader(MultiSourceDataLoader):
    """Sample batches randomly from DataLoaders with optional weights.
    
    This is the simplest approach for multi-dataset weighted sampling.
    Each batch is drawn from a randomly selected loader based on weights.
    
    Args:
        dataloaders: Sequence of DataLoaders to sample from.
        weights: Optional sampling weights. If None, uniform weights are used.
        stop_strategy: ``LONGEST`` (default) — continue until all loaders
            exhaust.  ``SHORTEST`` — stop as soon as any loader exhausts.
    
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
        stop_strategy: str = LONGEST,
    ):
        super().__init__(dataloaders, stop_strategy)
        self.weights = _normalize_weights(
            weights if weights is not None else [1.0] * self._num_loaders
        )

    def __iter__(self):
        iterators = self._create_iterators()
        active = np.ones(self._num_loaders, dtype=bool)
        buf = np.empty(self._num_loaders, dtype=np.float64)

        while active.any():
            idx = _weighted_choice(self._num_loaders, self.weights, active, buf)
            if idx < 0:
                break
            try:
                self._last_loader_idx = idx
                yield next(iterators[idx])
            except StopIteration:
                if self._stop_strategy == SHORTEST:
                    return
                active[idx] = False

    def __len__(self) -> int:
        draws = [len(ld) / w for ld, w in zip(self.dataloaders, self.weights)]
        if self._stop_strategy == SHORTEST:
            return int(np.floor(min(draws)))
        return int(np.ceil(max(draws)))


# =============================================================================
# Proportional Mix DataLoader
# =============================================================================

class ProportionalMixDataLoader(MultiSourceDataLoader):
    """Allocate a fixed number of batches to each DataLoader by ratio.
    
    Unlike RandomMixDataLoader which samples probabilistically, this loader
    guarantees exact batch counts per loader based on the specified ratios.
    
    Uses an efficient online sampling algorithm (O(1) memory per step) instead
    of pre-generating the full schedule.
    
    ``stop_strategy`` controls total iteration length:
    
    - ``LONGEST`` (default) — every child loader is traversed **at least
      once**.  ``total = ceil(max(len(ld_i) / ratio_i))``.  Shorter loaders
      restart as needed.
    - ``SHORTEST`` — **no** child loader needs to restart.  Stops as soon as
      the first loader would exhaust.
      ``total = floor(min(len(ld_i) / ratio_i))``.
    
    Args:
        dataloaders: Sequence of DataLoaders.
        ratios: Ratio of batches for each loader. If None, equal ratios.
        stop_strategy: ``LONGEST`` (default) or ``SHORTEST``.
    
    Examples:
        >>> # LONGEST (default): all loaders fully covered
        >>> # loader_a(100) ratio 0.75, loader_b(50) ratio 0.25
        >>> # total = ceil(max(100/0.75, 50/0.25)) = 200
        >>> loader = ProportionalMixDataLoader(
        ...     [loader_a, loader_b], ratios=[3, 1],
        ... )
        >>>
        >>> # SHORTEST: stop when the first loader would exhaust
        >>> # total = floor(min(100/0.75, 50/0.25)) = 133
        >>> loader = ProportionalMixDataLoader(
        ...     [loader_a, loader_b], ratios=[3, 1],
        ...     stop_strategy=SHORTEST,
        ... )
    """

    def __init__(
        self,
        dataloaders: Sequence[AnyDataLoader],
        ratios: Optional[Sequence[float]] = None,
        stop_strategy: str = LONGEST,
    ):
        super().__init__(dataloaders, stop_strategy)
        ratios_arr = _normalize_weights(
            ratios if ratios is not None else [1.0] * self._num_loaders
        )
        lengths_over_ratios = [len(ld) / r for ld, r in zip(self.dataloaders, ratios_arr)]

        if stop_strategy == LONGEST:
            # Every child loader traversed at least once (shortest restarts)
            total_batches = int(np.ceil(max(lengths_over_ratios)))
        else:
            # SHORTEST: No child loader needs to restart (stop at first exhaustion)
            total_batches = int(np.floor(min(lengths_over_ratios)))
        self.batches_per_loader = (ratios_arr * total_batches).astype(int)
        # Distribute remainder to last loader
        self.batches_per_loader[-1] += total_batches - int(self.batches_per_loader.sum())
        self._total_batches = int(self.batches_per_loader.sum())

    def __iter__(self):
        iterators = self._create_iterators()
        remaining = self.batches_per_loader.astype(np.float64)
        buf = np.empty(self._num_loaders, dtype=np.float64)
        dead = np.zeros(self._num_loaders, dtype=bool)

        for _ in range(self._total_batches):
            # Zero out dead loaders so they are never selected
            remaining[dead] = 0
            total = remaining.sum()
            if total <= 0:
                break
            np.divide(remaining, total, out=buf)
            idx = int(_rng.choice(self._num_loaders, p=buf))
            remaining[idx] -= 1

            try:
                batch = next(iterators[idx])
            except StopIteration:
                # Restart exhausted loader
                iterators[idx] = iter(self.dataloaders[idx])
                try:
                    batch = next(iterators[idx])
                except StopIteration:
                    # Loader is empty even after restart — mark dead
                    dead[idx] = True
                    continue

            self._last_loader_idx = idx
            yield batch

    def __len__(self):
        return self._total_batches


# =============================================================================
# Alternating DataLoader
# =============================================================================

class AlternatingDataLoader(MultiSourceDataLoader):
    """Alternate between DataLoaders using a custom pattern.
    
    Args:
        dataloaders: Sequence of DataLoaders.
        pattern: Index pattern for alternation. If None, cycles through indices.
        stop_strategy: ``LONGEST`` (default) — continue until all
            pattern-referenced loaders exhaust.  ``SHORTEST`` — stop as
            soon as any loader exhausts.
    
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
        stop_strategy: str = LONGEST,
    ):
        super().__init__(dataloaders, stop_strategy)
        self.pattern = list(pattern) if pattern else list(range(self._num_loaders))

    def __iter__(self):
        iterators = self._create_iterators()
        pattern_cycle = itertools.cycle(self.pattern)
        active = [True] * self._num_loaders
        total = len(self)
        yielded = 0
        consecutive_skips = 0

        while yielded < total:
            if consecutive_skips >= len(self.pattern):
                break
            idx = next(pattern_cycle)
            if not active[idx]:
                consecutive_skips += 1
                continue
            try:
                batch = next(iterators[idx])
            except StopIteration:
                if self._stop_strategy == SHORTEST:
                    return
                active[idx] = False
                consecutive_skips += 1
                continue
            self._last_loader_idx = idx
            yield batch
            yielded += 1
            consecutive_skips = 0

    def __len__(self) -> int:
        """Estimate total batches based on stop strategy.

        For ``LONGEST``, returns the sum of all child loader lengths (every
        loader fully consumed).  For ``SHORTEST``, estimates batches until the
        first loader exhausts, accounting for the pattern frequency.
        """
        if self._stop_strategy == SHORTEST:
            from collections import Counter
            freq = Counter(self.pattern)
            # Batches until the first loader would exhaust, scaled by pattern frequency
            return min(
                len(self.dataloaders[idx]) * len(self.pattern) // count
                for idx, count in freq.items()
                if idx < self._num_loaders
            )
        return sum(len(ld) for ld in self.dataloaders)


# =============================================================================
# Task-Tagged DataLoader
# =============================================================================

class TaskTaggedDataLoader(MultiSourceDataLoader):
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
        dataloaders: dict[str, AnyDataLoader],
        sampling_strategy: str = 'random',
        propagate_tags: bool = False,
        stop_strategy: str = LONGEST,
    ):
        # TaskTaggedDataLoader uses a dict; store list of values for base class
        super().__init__(list(dataloaders.values()), stop_strategy)
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
                if self._stop_strategy == SHORTEST:
                    return
                active[task_idx] = False
                active_count -= 1
                active_list = [self.task_names[i] for i in range(self._num_loaders) if active[i]]

    def __len__(self) -> int:
        if self.sampling_strategy == 'round_robin':
            return super().__len__()
        # 'random' with uniform weights: estimate via len(ld) / (1/N) = len(ld) * N
        draws = [len(ld) * self._num_loaders for ld in self.dataloaders]
        if self._stop_strategy == SHORTEST:
            return min(draws)
        return max(draws)


# =============================================================================
# Dynamic Schedule DataLoader
# =============================================================================

class DynamicScheduleDataLoader(RandomMixDataLoader):
    """Extend :class:`RandomMixDataLoader` with dynamic weight updates.

    Sampling behaviour is identical to ``RandomMixDataLoader``; this subclass
    adds the ability to adjust weights at runtime based on training feedback
    (e.g. per-loader loss) and to track per-loader statistics.

    Args:
        dataloaders: Sequence of DataLoaders.
        initial_weights: Starting weights. If None, uniform weights.
        enable_tracking: If True, track statistics and allow weight updates.
        stop_strategy: ``LONGEST`` (default) or ``SHORTEST``.

    Attributes:
        weights: Current sampling weights (numpy array, inherited).
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
        stop_strategy: str = LONGEST,
    ):
        super().__init__(dataloaders, weights=initial_weights, stop_strategy=stop_strategy)
        self.enable_tracking = enable_tracking
        self.loader_performance: dict[int, list] = defaultdict(list)
        self.batches_from_loader = [0] * self._num_loaders

    def update_weights(
        self,
        loader_idx: int,
        performance: float,
        learning_rate: float = 0.1,
    ):
        """Update sampling weights based on performance feedback.

        Increases weights for loaders with lower performance values
        (e.g. lower loss = better = higher weight).

        Args:
            loader_idx: Index of the DataLoader.
            performance: Performance metric (lower is better, e.g. loss).
            learning_rate: Rate of weight adjustment.
        """
        if not self.enable_tracking:
            return
        self.loader_performance[loader_idx].append(performance)

        avg_perfs = np.array([
            np.mean(self.loader_performance[i] or [0.5])
            for i in range(self._num_loaders)
        ])
        if avg_perfs.max() > avg_perfs.min():
            normalized = (avg_perfs - avg_perfs.min()) / (avg_perfs.max() - avg_perfs.min())
            target = (1 - normalized)
            target /= target.sum()
            self.weights = (1 - learning_rate) * self.weights + learning_rate * target

    def __iter__(self):
        for batch in super().__iter__():
            if self.enable_tracking:
                self.batches_from_loader[self._last_loader_idx] += 1
            yield batch

    def get_statistics(self) -> dict:
        """Get usage statistics.

        Returns:
            Dictionary with batches_per_loader, current_weights,
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


class InBatchMixDataLoader(MultiSourceDataLoader):
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
        stop_strategy: ``SHORTEST`` (default) — stop when any loader exhausts.
            ``LONGEST`` — restart exhausted loaders so every loader is fully
            consumed; total iterations = ``max(len(ld))``.
    
    After each yielded batch, ``last_loader_idx`` is set to a dict
    ``{child_index: actual_sample_count}`` describing the composition,
    e.g. ``{0: 16, 1: 8}``.  This is compatible with
    ``SourceTaggedDataLoader`` which resolves the indices to source names.
    
    Examples:
        >>> loader = InBatchMixDataLoader(
        ...     [loader_a, loader_b],
        ...     samples_per_loader=[16, 16],
        ... )
        >>> for batch in loader:
        ...     print(loader.last_loader_idx)  # e.g. {0: 16, 1: 16}
    """

    def __init__(
        self,
        dataloaders: Sequence[AnyDataLoader],
        samples_per_loader: Optional[Sequence[int]] = None,
        total_batch_size: Optional[int] = None,
        random_sample: bool = False,
        stop_strategy: str = SHORTEST,
    ):
        super().__init__(dataloaders, stop_strategy)
        self._random_sample = random_sample
        self._apply_sharding = None
        if self.sharding is not None:
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

    def _slice_batch(self, data, labels, num_samples: int) -> tuple:
        """Slice data and labels to at most *num_samples*.

        Returns:
            ``(sliced_data, sliced_labels, actual_count)`` where
            *actual_count* is ``min(batch_size, num_samples)``.
        """
        batch_size = _get_batch_size(labels)
        if batch_size <= num_samples:
            return data, labels, batch_size
        if self._random_sample:
            indices = _rng.choice(batch_size, size=num_samples, replace=False)
        else:
            indices = np.arange(num_samples)
        return slice_data(data, indices), slice_data(labels, indices), num_samples

    def __iter__(self):
        iterators = self._create_iterators()
        data_parts: list = [None] * self._num_loaders
        label_parts: list = [None] * self._num_loaders
        counts: list = [0] * self._num_loaders
        is_longest = self._stop_strategy == LONGEST
        max_iters = max(len(ld) for ld in self.dataloaders) if is_longest else float('inf')
        iters_done = 0

        while iters_done < max_iters:
            for idx in range(self._num_loaders):
                try:
                    batch = next(iterators[idx])
                except StopIteration:
                    if not is_longest:
                        return  # SHORTEST: stop immediately
                    # LONGEST: restart exhausted loader
                    iterators[idx] = iter(self.dataloaders[idx])
                    try:
                        batch = next(iterators[idx])
                    except StopIteration:
                        return  # Empty even after restart
                extracted = self._extract_batch(batch)
                if extracted is None:
                    raise ValueError(
                        f"InBatchMixDataLoader: loader {idx} yielded an incompatible "
                        f"batch format (expected a (data, labels) tuple, got "
                        f"{type(batch).__name__}). All child loaders must yield "
                        f"(data, labels) or (tag, (data, labels)) tuples."
                    )
                data_parts[idx], label_parts[idx], counts[idx] = (
                    self._slice_batch(*extracted, self.samples_per_loader[idx])
                )

            combined_data = concat_data(data_parts)
            combined_labels = concat_data(label_parts)
            if self._apply_sharding is not None:
                combined_data = self._apply_sharding(combined_data, self.sharding)
                combined_labels = self._apply_sharding(combined_labels, self.sharding)

            self._last_loader_idx = {i: counts[i] for i in range(self._num_loaders)}
            yield combined_data, combined_labels
            iters_done += 1

    def __len__(self):
        if self._stop_strategy == LONGEST:
            return max(len(ld) for ld in self.dataloaders)
        return min(len(ld) for ld in self.dataloaders)


# =============================================================================
# Curriculum Learning DataLoader
# =============================================================================

class CurriculumDataLoader(MultiSourceDataLoader):
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
                batch = next(iterators[idx])
            except StopIteration:
                # Restart exhausted loader
                iterators[idx] = iter(self.dataloaders[idx])
                try:
                    batch = next(iterators[idx])
                except StopIteration:
                    raise RuntimeError(
                        f"CurriculumDataLoader: loader {idx} is empty even after "
                        f"restart (stage {self.current_stage}). All loaders used "
                        f"in stages must contain data."
                    )
            self._last_loader_idx = idx
            yield batch
            yielded += 1
            self.batches_in_stage += 1
            if (self.batches_in_stage >= self.batches_per_stage[self.current_stage]
                    and self.current_stage < len(self.stages) - 1):
                self.current_stage += 1
                self.batches_in_stage = 0

    def __len__(self):
        return sum(self.batches_per_stage)


# =============================================================================
# Source-Tagged Wrapper
# =============================================================================

class SourceTaggedDataLoader(SingleLoaderWrapper):
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
        super().__init__(dataloader)
        self.source_names = list(source_names)

    def _resolve_source_name(self, idx: Optional[Union[int, str, dict[int, int]]]) -> str:
        """Resolve a ``last_loader_idx`` value to a human-readable name.

        Handles three forms:
        - ``int``  → looked up in ``source_names``.
        - ``str``  → returned as-is.
        - ``dict`` → in-batch mix composition ``{child_idx: count}``;
          each key is recursively resolved and formatted as
          ``"mixed(name_a:16,name_b:8)"``.
        """
        if isinstance(idx, dict):
            parts = ", ".join(
                f"{self._resolve_source_name(k)}:{v}"
                for k, v in idx.items()
            )
            return f"mixed({parts})"
        if isinstance(idx, str):
            return idx
        if isinstance(idx, int) and 0 <= idx < len(self.source_names):
            return self.source_names[idx]
        if idx is None and len(self.source_names) == 1:
            return self.source_names[0]
        return "unknown" if idx is None else f"dataset_{idx}"

    def __iter__(self):
        for batch in self._inner:
            parent_name = self._resolve_source_name(self.last_loader_idx)

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
        return len(self._inner)


# =============================================================================
# Refreshable DataLoader Wrapper
# =============================================================================

class RefreshableDataLoader(SingleLoaderWrapper):
    """Wrap a DataLoader and trigger a user-specified refresh callback after specified epochs.

    This is useful for scenarios like re-shuffling data, rotating through
    dataset shards, updating sampling weights, or refreshing remote dataset
    connections between epochs.

    The wrapper iterates through the inner loader normally. When the inner
    loader is exhausted and the epoch interval condition is met, ``on_refresh``
    is called before the next iteration begins. If ``num_epochs`` is set, the
    loader will automatically restart for that many epochs; if ``None``, the
    loader iterates indefinitely (useful for infinite training loops).

    The callback receives ``(epoch, wrapper)`` where *wrapper* is this
    ``RefreshableDataLoader`` instance. You can modify the inner loader in
    three ways:

    1. **In-place mutation**: ``wrapper.inner.dataset = new_dataset``
    2. **Property assignment**: ``wrapper.inner = new_loader``
    3. **Return a new loader**: the return value, if not ``None``, replaces
       the inner loader automatically.

    Note:
        The refresh callback is **not** invoked after the final epoch
        (there is no subsequent epoch to prepare for). This means the
        actual number of refresh calls is
        ``floor((num_epochs - 1) / refresh_every)`` for finite runs.

    Args:
        dataloader: The DataLoader to wrap.
        on_refresh: Callback ``(epoch, wrapper) -> Optional[AnyDataLoader]``.
            *epoch* is 1-based. If the callback returns a DataLoader, the
            inner loader is replaced; otherwise only in-place / property
            mutations take effect.
        refresh_every: Invoke ``on_refresh`` every this many epochs.
            Defaults to 1 (refresh after every epoch). For example, if
            ``refresh_every=3``, the callback fires after epochs 3, 6, 9, …
            (1-indexed) — provided a subsequent epoch exists.
        num_epochs: Number of epochs to iterate. If ``None``, iterate
            indefinitely (never stops). If set, the inner loader will be
            re-iterated ``num_epochs`` times.

    Attributes:
        epoch: Current epoch number (1-based), updated at the start of
            each epoch. 0 before iteration begins.

    Examples:
        >>> # 1. In-place mutation
        >>> def reshuffle(epoch, wrapper):
        ...     wrapper.inner.dataset.shuffle()
        >>>
        >>> # 2. Property assignment
        >>> def swap_loader(epoch, wrapper):
        ...     wrapper.inner = build_new_loader(epoch)
        >>>
        >>> # 3. Return value replacement (refresh_every=3, num_epochs=10)
        >>> #    Refreshes fire after epochs 3, 6, 9 (3 times).
        >>> #    Epoch 10 is the last — no refresh afterwards.
        >>> def rotate_shard(epoch, wrapper):
        ...     return build_loader_for_shard(epoch % num_shards)
        >>>
        >>> loader = RefreshableDataLoader(
        ...     train_loader,
        ...     on_refresh=rotate_shard,
        ...     refresh_every=3,
        ...     num_epochs=10,
        ... )
        >>> for batch in loader:
        ...     train_step(batch)
    """

    def __init__(
        self,
        dataloader: AnyDataLoader,
        on_refresh: Optional[Callable[..., Any]] = None,
        refresh_every: int = 1,
        num_epochs: Optional[int] = None,
    ):
        super().__init__(dataloader)
        self._on_refresh = on_refresh
        self._num_epochs = num_epochs
        if refresh_every < 1:
            raise ValueError(f"refresh_every must be >= 1, got {refresh_every}")
        self._refresh_every = refresh_every
        self.epoch: int = 0

    def __iter__(self):
        epoch_iter = range(1, self._num_epochs + 1) if self._num_epochs is not None else itertools.count(1)
        for ep in epoch_iter:
            self.epoch = ep
            for batch in self._inner:
                yield batch
            has_next = self._num_epochs is None or ep < self._num_epochs
            if has_next and ep % self._refresh_every == 0:
                RefreshableDataLoader.default_refresh(ep, self)
                if self._on_refresh is not None:
                    result = self._on_refresh(ep, self)
                    if result is not None:
                        self._inner = result
        RefreshableDataLoader.default_refresh(self._num_epochs, self)

    def __len__(self):
        if self._num_epochs is None:
            raise TypeError(
                "RefreshableDataLoader with num_epochs=None (infinite) has no len(). "
                "Set num_epochs to a finite value to use len()."
            )
        return self._num_epochs * len(self._inner)

    @staticmethod
    def default_refresh(epoch: int, wrapper) -> None:
        ident = get_loader_ident(wrapper)
        total = wrapper._num_epochs
        if total is not None:
            logging.info("Epoch %s/%s complete\nLoader tree:\n%s", epoch, total, ident)
        else:
            logging.info("Epoch %s complete\nLoader tree:\n%s", epoch, ident)


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
    def round_robin(
        *loaders: AnyDataLoader,
        stop_strategy: str = LONGEST,
    ) -> RoundRobinDataLoader:
        """Create a round-robin DataLoader."""
        return RoundRobinDataLoader(list(loaders), stop_strategy=stop_strategy)
    
    @staticmethod
    def random(
        *loaders: AnyDataLoader, 
        weights: Optional[Sequence[float]] = None,
        stop_strategy: str = LONGEST,
    ) -> RandomMixDataLoader:
        """Create a weighted random mixing DataLoader."""
        return RandomMixDataLoader(list(loaders), weights=weights, stop_strategy=stop_strategy)
    
    @staticmethod
    def proportional(
        *loaders: AnyDataLoader,
        ratios: Optional[Sequence[float]] = None,
        stop_strategy: str = LONGEST,
    ) -> ProportionalMixDataLoader:
        """Create a proportional allocation DataLoader.

        ``stop_strategy`` accepts ``LONGEST`` (default) or ``SHORTEST``.
        """
        return ProportionalMixDataLoader(list(loaders), ratios=ratios, stop_strategy=stop_strategy)
    
    @staticmethod
    def alternating(
        *loaders: AnyDataLoader,
        pattern: Optional[Sequence[int]] = None,
        stop_strategy: str = LONGEST,
    ) -> AlternatingDataLoader:
        """Create an alternating pattern DataLoader."""
        return AlternatingDataLoader(list(loaders), pattern=pattern, stop_strategy=stop_strategy)
    
    @staticmethod
    def tagged(
        loaders_dict: dict[str, AnyDataLoader],
        sampling_strategy: str = 'random',
        propagate_tags: bool = False,
        stop_strategy: str = LONGEST,
    ) -> TaskTaggedDataLoader:
        """Create a task-tagged DataLoader."""
        return TaskTaggedDataLoader(
            loaders_dict, sampling_strategy, propagate_tags, stop_strategy=stop_strategy,
        )
    
    @staticmethod
    def dynamic(
        *loaders: AnyDataLoader,
        initial_weights: Optional[Sequence[float]] = None,
        enable_tracking: bool = True,
        stop_strategy: str = LONGEST,
    ) -> DynamicScheduleDataLoader:
        """Create a dynamic scheduling DataLoader."""
        return DynamicScheduleDataLoader(
            list(loaders), initial_weights, enable_tracking, stop_strategy=stop_strategy,
        )
    
    @staticmethod
    def curriculum(
        *loaders: AnyDataLoader,
        stages: Sequence[Sequence[int]],
        batches_per_stage: Sequence[int],
    ) -> CurriculumDataLoader:
        """Create a curriculum learning DataLoader."""
        return CurriculumDataLoader(list(loaders), stages, batches_per_stage)
    
    @staticmethod
    def inbatch(
        *loaders: AnyDataLoader,
        samples_per_loader: Optional[Sequence[int]] = None,
        random_sample: bool = False,
        stop_strategy: str = SHORTEST,
    ) -> InBatchMixDataLoader:
        """Create an in-batch mixing DataLoader."""
        return InBatchMixDataLoader(
            list(loaders), samples_per_loader,
            random_sample=random_sample, stop_strategy=stop_strategy,
        )

    @staticmethod
    def refreshable(
        loader: AnyDataLoader,
        on_refresh: Optional[Callable[..., None]] = None,
        refresh_every: int = 1,
        num_epochs: Optional[int] = None,
    ) -> RefreshableDataLoader:
        """Create a refreshable DataLoader that calls on_refresh at specified epoch intervals."""
        return RefreshableDataLoader(
            loader, on_refresh=on_refresh, refresh_every=refresh_every, num_epochs=num_epochs
        )


# =============================================================================
# Examples and Testing
# =============================================================================

def _create_test_loaders() -> dict[str, TorchDataLoader]:
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
    print("\n[1.3] Proportional - easy:hard = 3:1 (LONGEST)")
    prop = Compose.proportional(easy, hard, ratios=[3, 1])
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
    outer = Compose.proportional(inner, hard, ratios=[2, 1])
    
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
