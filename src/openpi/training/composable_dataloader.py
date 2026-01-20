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
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterator
import itertools
import random
from typing import Dict, Optional, Protocol, Sequence, TypeVar, Union, runtime_checkable

import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader, Dataset as TorchDataset

__all__ = [
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
        """Check if an object satisfies the BaseDataLoader protocol.
        
        Args:
            obj: Object to check.
            
        Returns:
            True if obj implements __iter__, False otherwise.
        """
        return isinstance(obj, BaseDataLoader)


# Type alias for any valid DataLoader
# Using BaseDataLoader Protocol ensures compatibility with:
# - torch.utils.data.DataLoader
# - OpenPI's TorchDataLoader, RLDSDataLoader, DataLoaderImpl
# - All ComposableDataLoader subclasses
# - Any object implementing __iter__
AnyDataLoader = BaseDataLoader


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
        self.dataloaders = list(dataloaders)
        self.num_loaders = len(dataloaders)
    
    def __iter__(self):
        iterators = [iter(loader) for loader in self.dataloaders]
        loader_idx = 0
        
        while iterators:
            try:
                batch = next(iterators[loader_idx])
                yield batch
                loader_idx = (loader_idx + 1) % len(iterators)
            except StopIteration:
                iterators.pop(loader_idx)
                if iterators:
                    loader_idx = loader_idx % len(iterators)
    
    def __len__(self):
        return sum(len(loader) for loader in self.dataloaders)


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
        weights: Optional[Sequence[float]] = None
    ):
        self.dataloaders = list(dataloaders)
        
        if weights is None:
            weights = [1.0] * len(dataloaders)
        total = sum(weights)
        self.weights = np.array([w / total for w in weights])
    
    def __iter__(self):
        iterators = [iter(loader) for loader in self.dataloaders]
        active_indices = list(range(len(iterators)))
        
        while active_indices:
            probs = self.weights[active_indices]
            probs = probs / probs.sum()
            idx = np.random.choice(active_indices, p=probs)
            
            try:
                batch = next(iterators[idx])
                yield batch
            except StopIteration:
                active_indices.remove(idx)
    
    def __len__(self):
        return sum(len(loader) for loader in self.dataloaders)


# =============================================================================
# Proportional Mix DataLoader
# =============================================================================

class ProportionalMixDataLoader(ComposableDataLoader):
    """Allocate a fixed number of batches to each DataLoader by ratio.
    
    Unlike RandomMixDataLoader which samples probabilistically, this loader
    guarantees exact batch counts per loader based on the specified ratios.
    
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
        max_batches: Optional[int] = None
    ):
        self.dataloaders = list(dataloaders)
        
        if ratios is None:
            ratios = [1.0] * len(dataloaders)
        
        total_batches = max_batches or sum(len(loader) for loader in dataloaders)
        ratios = np.array(ratios) / sum(ratios)
        self.batches_per_loader = (ratios * total_batches).astype(int)
        
        # Distribute remainder to last loader
        diff = total_batches - self.batches_per_loader.sum()
        if diff > 0:
            self.batches_per_loader[-1] += diff
    
    def __iter__(self):
        iterators = [iter(loader) for loader in self.dataloaders]
        
        # Build shuffled schedule
        schedule = []
        for idx, num_batches in enumerate(self.batches_per_loader):
            schedule.extend([idx] * num_batches)
        random.shuffle(schedule)
        
        for loader_idx in schedule:
            try:
                batch = next(iterators[loader_idx])
                yield batch
            except StopIteration:
                # Restart exhausted loader
                iterators[loader_idx] = iter(self.dataloaders[loader_idx])
                batch = next(iterators[loader_idx])
                yield batch
    
    def __len__(self):
        return int(self.batches_per_loader.sum())


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
        pattern: Optional[Sequence[int]] = None
    ):
        self.dataloaders = list(dataloaders)
        self.pattern = list(pattern) if pattern else list(range(len(dataloaders)))
    
    def __iter__(self):
        iterators = [iter(loader) for loader in self.dataloaders]
        pattern_cycle = itertools.cycle(self.pattern)
        
        total_batches = sum(len(loader) for loader in self.dataloaders)
        batches_yielded = 0
        
        while batches_yielded < total_batches:
            loader_idx = next(pattern_cycle)
            try:
                batch = next(iterators[loader_idx])
                yield batch
                batches_yielded += 1
            except StopIteration:
                continue
    
    def __len__(self):
        return sum(len(loader) for loader in self.dataloaders)


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
        propagate_tags: bool = False
    ):
        self.dataloaders = dataloaders
        self.task_names = list(dataloaders.keys())
        self.sampling_strategy = sampling_strategy
        self.propagate_tags = propagate_tags
    
    def __iter__(self):
        iterators = {name: iter(loader) for name, loader in self.dataloaders.items()}
        
        if self.sampling_strategy == 'round_robin':
            task_cycle = itertools.cycle(self.task_names)
        
        active_tasks = set(self.task_names)
        
        while active_tasks:
            if self.sampling_strategy == 'random':
                task = random.choice(list(active_tasks))
            elif self.sampling_strategy == 'round_robin':
                task = next(task_cycle)
                while task not in active_tasks:
                    task = next(task_cycle)
            else:
                raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")
            
            try:
                batch = next(iterators[task])
                
                # Handle nested tagged batches
                if self.propagate_tags and isinstance(batch, tuple) and len(batch) == 2:
                    if isinstance(batch[0], str):
                        nested_task, nested_batch = batch
                        yield f"{task}/{nested_task}", nested_batch
                    else:
                        yield task, batch
                else:
                    yield task, batch
                    
            except StopIteration:
                active_tasks.remove(task)
    
    def __len__(self):
        return sum(len(loader) for loader in self.dataloaders.values())


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
        enable_tracking: bool = True
    ):
        self.dataloaders = list(dataloaders)
        
        if initial_weights is None:
            initial_weights = [1.0] * len(dataloaders)
        self.weights = np.array(initial_weights) / sum(initial_weights)
        
        self.enable_tracking = enable_tracking
        self.loader_performance: Dict[int, list] = defaultdict(list)
        self.batches_from_loader = [0] * len(dataloaders)
        self._last_loader_idx: Optional[int] = None
    
    @property
    def last_loader_idx(self) -> Optional[int]:
        """Get the index of the loader that produced the last batch.
        
        Returns:
            Loader index, or None if iteration hasn't started.
        """
        return self._last_loader_idx
    
    def update_weights(
        self, 
        loader_idx: int, 
        performance: float, 
        learning_rate: float = 0.1
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
            for i in range(len(self.dataloaders))
        ])
        
        if avg_perfs.max() > avg_perfs.min():
            normalized = (avg_perfs - avg_perfs.min()) / (avg_perfs.max() - avg_perfs.min())
            # Lower performance (loss) = higher weight
            target_weights = 1 - normalized
            target_weights = target_weights / target_weights.sum()
            
            self.weights = (1 - learning_rate) * self.weights + learning_rate * target_weights
    
    def __iter__(self):
        iterators = [iter(loader) for loader in self.dataloaders]
        active_indices = list(range(len(iterators)))
        self._last_loader_idx = None
        
        while active_indices:
            probs = self.weights[active_indices]
            probs = probs / probs.sum()
            idx = np.random.choice(active_indices, p=probs)
            
            try:
                batch = next(iterators[idx])
                self._last_loader_idx = idx
                if self.enable_tracking:
                    self.batches_from_loader[idx] += 1
                yield batch
            except StopIteration:
                active_indices.remove(idx)
    
    def __len__(self):
        return sum(len(loader) for loader in self.dataloaders)
    
    def get_statistics(self) -> dict:
        """Get usage statistics.
        
        Returns:
            Dictionary containing batches_per_loader, current_weights,
            and performance_history.
        """
        return {
            'batches_per_loader': self.batches_from_loader,
            'current_weights': self.weights.tolist(),
            'performance_history': dict(self.loader_performance)
        }


# =============================================================================
# In-Batch Mix DataLoader
# =============================================================================

class InBatchMixDataLoader(ComposableDataLoader):
    """Mix samples from multiple DataLoaders within a single batch.
    
    Useful for contrastive learning or scenarios requiring samples from
    different sources in every batch.
    
    Args:
        dataloaders: Sequence of DataLoaders.
        samples_per_loader: Number of samples from each loader per batch.
        total_batch_size: Alternative to samples_per_loader; divided equally.
    
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
        total_batch_size: Optional[int] = None
    ):
        self.dataloaders = list(dataloaders)
        
        if samples_per_loader is None:
            if total_batch_size is None:
                total_batch_size = sum(
                    getattr(loader, 'batch_size', 32) 
                    for loader in dataloaders
                )
            samples_per_loader = [total_batch_size // len(dataloaders)] * len(dataloaders)
            samples_per_loader[-1] = total_batch_size - sum(samples_per_loader[:-1])
        
        self.samples_per_loader = list(samples_per_loader)
        self.total_batch_size = sum(samples_per_loader)
    
    def __iter__(self):
        iterators = [iter(loader) for loader in self.dataloaders]
        
        while True:
            mixed_batch_data = []
            mixed_batch_labels = []
            
            for idx, num_samples in enumerate(self.samples_per_loader):
                try:
                    batch = next(iterators[idx])
                    
                    # Handle tagged batches
                    if isinstance(batch, tuple) and len(batch) == 2:
                        if isinstance(batch[0], (int, str)):
                            _, actual_batch = batch
                            data, labels = actual_batch
                        else:
                            data, labels = batch
                    else:
                        continue
                    
                    if len(data) >= num_samples:
                        mixed_batch_data.append(data[:num_samples])
                        mixed_batch_labels.append(labels[:num_samples])
                    else:
                        mixed_batch_data.append(data)
                        mixed_batch_labels.append(labels)
                        
                except StopIteration:
                    return
            
            if mixed_batch_data:
                combined_data = torch.cat(mixed_batch_data, dim=0)
                combined_labels = torch.cat(mixed_batch_labels, dim=0)
                
                # Shuffle combined batch
                indices = torch.randperm(len(combined_data))
                yield combined_data[indices], combined_labels[indices]
    
    def __len__(self):
        return min(len(loader) for loader in self.dataloaders)


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
        batches_per_stage: Sequence[int]
    ):
        self.dataloaders = list(dataloaders)
        self.stages = [list(stage) for stage in stages]
        self.batches_per_stage = list(batches_per_stage)
        self.current_stage = 0
        self.batches_in_stage = 0
    
    def __iter__(self):
        iterators = [iter(loader) for loader in self.dataloaders]
        total_batches = sum(self.batches_per_stage)
        batches_yielded = 0
        
        while batches_yielded < total_batches:
            active_loaders = self.stages[self.current_stage]
            loader_idx = random.choice(active_loaders)
            
            try:
                batch = next(iterators[loader_idx])
                yield batch
                batches_yielded += 1
                self.batches_in_stage += 1
                
                # Check for stage transition
                if (self.batches_in_stage >= self.batches_per_stage[self.current_stage] and
                    self.current_stage < len(self.stages) - 1):
                    self.current_stage += 1
                    self.batches_in_stage = 0
                    
            except StopIteration:
                # Restart exhausted loader
                iterators[loader_idx] = iter(self.dataloaders[loader_idx])
    
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
        samples_per_loader: Optional[Sequence[int]] = None
    ) -> InBatchMixDataLoader:
        """Create an in-batch mixing DataLoader."""
        return InBatchMixDataLoader(list(loaders), samples_per_loader)


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
            fake_loss = random.random()
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
