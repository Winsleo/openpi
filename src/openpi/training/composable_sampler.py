"""Composable hierarchical sampler with pluggable mix strategies.

Architecture (bottom-up):
  LeafNode          - maps a local index range to global indices via a TraversalPolicy.
  MixNode           - distributes a budget across children using a MixStrategy,
                      then merges their output streams.
  MixStrategy       - protocol that decides quota allocation and ordering
                      (5 built-in implementations).
  Compose           - convenience factory for building MixNode trees.
  ComposableSampler - thin torch Sampler wrapper around any SamplerNode root
                      (mirrors the composable DataLoader pattern).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional, Protocol, Sequence
import heapq
import numpy as np
from torch.utils.data import Sampler
from openpi.training.composable_dataloader import normalize_weights


__all__ = [
    # Protocols
    "SamplerNode",
    "MixStrategy",
    "TraversalPolicy",
    # Types
    "WeightSpec",
    "MIX_STRATEGY_LARGEST_REMAINDER",
    "MIX_STRATEGY_WEIGHTED_RANDOM",
    "MIX_STRATEGY_ROUND_ROBIN",
    "MIX_STRATEGY_SHUFFLE_SEQUENTIAL",
    "MIX_STRATEGY_STRATIFIED_RANDOM",
    "COMPOSABLE_MIX_STRATEGIES",
    "mix_strategy_from_name",
    # Core utilities
    "largest_remainder_allocate",
    # Interleaver
    "SmoothQuotaInterleaver",
    # Strategies
    "LargestRemainderStrategy",
    "WeightedRandomStrategy",
    "RoundRobinStrategy",
    "ShuffleSequentialStrategy",
    "StratifiedRandomStrategy",
    "mix_strategy_from_state",
    "clone_mix_strategy",
    # Traversal
    "PermutationTraversal",
    # Nodes
    "LeafNode",
    "MixNode",
    # Sampler
    "ComposableSampler",
    # Factory
    "Compose",
    # Helpers
    "EpisodeSpec",
    "build_episode_mix_node",
    "build_source_task_hierarchy",
]


# Public type alias: weights can be static or epoch-dependent.
WeightSpec = Sequence[float] | Callable[[int], Sequence[float]]

# Mix strategy identifiers: ComposableSamplerConfig.mix_strategy, state_dict["type"], MixNode default names.
MIX_STRATEGY_LARGEST_REMAINDER = "largest_remainder"
MIX_STRATEGY_WEIGHTED_RANDOM = "weighted_random"
MIX_STRATEGY_ROUND_ROBIN = "round_robin"
MIX_STRATEGY_SHUFFLE_SEQUENTIAL = "shuffle_sequential"
MIX_STRATEGY_STRATIFIED_RANDOM = "stratified_random"

COMPOSABLE_MIX_STRATEGIES: frozenset[str] = frozenset(
    {
        MIX_STRATEGY_LARGEST_REMAINDER,
        MIX_STRATEGY_WEIGHTED_RANDOM,
        MIX_STRATEGY_ROUND_ROBIN,
        MIX_STRATEGY_SHUFFLE_SEQUENTIAL,
        MIX_STRATEGY_STRATIFIED_RANDOM,
    }
)


def largest_remainder_allocate(total: int, weights: Sequence[float]) -> np.ndarray:
    """
    Allocate integer quotas summing exactly to `total` with the
    Largest Remainder / Hamilton method.

    Example:
        total = 10, weights = [0.51, 0.29, 0.20]
        raw   = [5.1, 2.9, 2.0]
        base  = [5,   2,   2]
        rem   = 1 -> allocate to index 1 => [5, 3, 2]
    """
    if total < 0:
        raise ValueError("total must be >= 0")

    w = normalize_weights(weights)
    raw = w * float(total)
    base = np.floor(raw).astype(np.int64)
    remain = int(total - int(base.sum()))

    if remain == 0:
        return base

    frac = raw - base
    # Descending fractional part, then ascending index for stable tie-breaking
    order = np.lexsort((np.arange(len(frac)), -frac))
    base[order[:remain]] += 1
    return base


def seed_to_rng(seed: Optional[int]) -> np.random.Generator:
    """Create a numpy RNG from an optional seed."""
    return np.random.default_rng(seed)


# ============================================================
# Interleaver
# ============================================================

class SmoothQuotaInterleaver:
    """Smoothly interleave child streams according to exact quotas.

    Maintains (consumed_i / quota_i) for each active child and always
    takes from the child with the smallest ratio (most behind its target).
    Gives much stabler prefix proportions than random shuffling.
    """

    def interleave(self, streams: Sequence[Iterator[int]], quotas: Sequence[int]) -> Iterator[int]:
        q = np.asarray(quotas, dtype=np.int64)
        total = int(q.sum())
        if total == 0:
            return

        consumed = np.zeros(len(q), dtype=np.int64)

        # Min-heap on progress ratio = consumed / quota
        heap: list[tuple[float, int]] = []
        for i, qi in enumerate(q):
            if qi > 0:
                heapq.heappush(heap, (0.0, i))

        for _ in range(total):
            if not heap:
                return
            _, i = heapq.heappop(heap)
            yield next(streams[i])
            consumed[i] += 1
            if consumed[i] < q[i]:
                progress = consumed[i] / float(q[i])
                heapq.heappush(heap, (progress, i))


# ============================================================
# Node protocol
# ============================================================

class SamplerNode(Protocol):
    def set_epoch(self, epoch: int) -> None:
        ...

    def plan(self, budget: int) -> Iterator[int]:
        ...

    def state_dict(self) -> dict[str, Any]:
        ...

    def load_state_dict(self, state: dict[str, Any]) -> None:
        ...


# ============================================================
# Mix strategies
# ============================================================

class MixStrategy(Protocol):
    """How an internal MixNode distributes budget and orders child output."""

    def plan(
        self,
        children: Sequence["SamplerNode"],
        weights: np.ndarray,
        budget: int,
        rng: np.random.Generator,
    ) -> Iterator[int]:
        ...

    def state_dict(self) -> dict[str, Any]:
        ...

    def load_state_dict(self, state: dict[str, Any]) -> None:
        ...


class LargestRemainderStrategy:
    def __init__(self, interleaver: Optional[SmoothQuotaInterleaver] = None) -> None:
        self._interleaver = interleaver or SmoothQuotaInterleaver()

    def plan(
        self,
        children: Sequence["SamplerNode"],
        weights: np.ndarray,
        budget: int,
        rng: np.random.Generator,
    ) -> Iterator[int]:
        budget = int(budget)
        if budget <= 0:
            return

        quotas = largest_remainder_allocate(budget, weights)
        streams = [child.plan(int(q)) for child, q in zip(children, quotas)]
        yield from self._interleaver.interleave(streams, quotas.tolist())

    def state_dict(self) -> dict[str, Any]:
        return {"type": MIX_STRATEGY_LARGEST_REMAINDER}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        _ = state  # stateless


class WeightedRandomStrategy:
    def __init__(self, *, temperature: float = 1.0) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature = float(temperature)

    def plan(
        self,
        children: Sequence["SamplerNode"],
        weights: np.ndarray,
        budget: int,
        rng: np.random.Generator,
    ) -> Iterator[int]:
        budget = int(budget)
        if budget <= 0:
            return

        if self.temperature != 1.0:
            scaled = np.asarray(weights, dtype=np.float64) ** (1.0 / self.temperature)
            total = scaled.sum()
            if total <= 0:
                raise ValueError("scaled weights must not all be zero")
            probs = scaled / total
        else:
            probs = weights

        quotas = rng.multinomial(budget, probs)
        streams = [child.plan(int(q)) for child, q in zip(children, quotas)]
        schedule = np.repeat(np.arange(len(children), dtype=np.int64), quotas)
        rng.shuffle(schedule)
        for i in schedule:
            yield next(streams[int(i)])

    def state_dict(self) -> dict[str, Any]:
        return {"type": MIX_STRATEGY_WEIGHTED_RANDOM, "temperature": self.temperature}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.temperature = float(state.get("temperature", 1.0))


class RoundRobinStrategy:
    def plan(
        self,
        children: Sequence["SamplerNode"],
        weights: np.ndarray,
        budget: int,
        rng: np.random.Generator,
    ) -> Iterator[int]:
        _ = rng  # deterministic ordering by cyclic pointer
        budget = int(budget)
        if budget <= 0:
            return

        quotas = largest_remainder_allocate(budget, weights)
        streams = [child.plan(int(q)) for child, q in zip(children, quotas)]
        remaining = quotas.copy()
        # Maintain a compact list of active children to avoid repeated scans.
        active: list[int] = [i for i, q in enumerate(remaining) if int(q) > 0]
        pos = 0
        for _ in range(budget):
            if not active:
                break
            idx = int(active[pos])
            yield next(streams[idx])
            remaining[idx] -= 1
            if int(remaining[idx]) <= 0:
                # Current child is exhausted; remove it and keep the position.
                active.pop(pos)
                if not active:
                    break
                if pos >= len(active):
                    pos = 0
            else:
                pos = (pos + 1) % len(active)

    def state_dict(self) -> dict[str, Any]:
        return {"type": MIX_STRATEGY_ROUND_ROBIN}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        _ = state  # stateless


class ShuffleSequentialStrategy:
    def plan(
        self,
        children: Sequence["SamplerNode"],
        weights: np.ndarray,
        budget: int,
        rng: np.random.Generator,
    ) -> Iterator[int]:
        budget = int(budget)
        if budget <= 0:
            return

        quotas = largest_remainder_allocate(budget, weights)
        streams = [child.plan(int(q)) for child, q in zip(children, quotas)]
        order = rng.permutation(len(children))
        for i in order:
            yield from streams[int(i)]

    def state_dict(self) -> dict[str, Any]:
        return {"type": MIX_STRATEGY_SHUFFLE_SEQUENTIAL}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        _ = state  # stateless


class StratifiedRandomStrategy:
    def plan(
        self,
        children: Sequence["SamplerNode"],
        weights: np.ndarray,
        budget: int,
        rng: np.random.Generator,
    ) -> Iterator[int]:
        budget = int(budget)
        if budget <= 0:
            return

        quotas = largest_remainder_allocate(budget, weights)
        streams = [child.plan(int(q)) for child, q in zip(children, quotas)]
        remaining = quotas.copy()
        left = budget
        while left > 0:
            active = np.where(remaining > 0)[0]
            order = rng.permutation(active)
            for i in order:
                ii = int(i)
                yield next(streams[ii])
                remaining[ii] -= 1
                left -= 1

    def state_dict(self) -> dict[str, Any]:
        return {"type": MIX_STRATEGY_STRATIFIED_RANDOM}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        _ = state  # stateless


_STRATEGY_REGISTRY: dict[str, Callable[[dict[str, Any]], MixStrategy]] = {
    MIX_STRATEGY_LARGEST_REMAINDER: lambda s: LargestRemainderStrategy(),
    MIX_STRATEGY_WEIGHTED_RANDOM: lambda s: WeightedRandomStrategy(temperature=float(s.get("temperature", 1.0))),
    MIX_STRATEGY_ROUND_ROBIN: lambda s: RoundRobinStrategy(),
    MIX_STRATEGY_SHUFFLE_SEQUENTIAL: lambda s: ShuffleSequentialStrategy(),
    MIX_STRATEGY_STRATIFIED_RANDOM: lambda s: StratifiedRandomStrategy(),
}


def register_strategy(type_name: str, factory: Callable[[dict[str, Any]], MixStrategy]) -> None:
    """Register a custom MixStrategy factory for state_dict restoration.

    ``factory`` receives the full state dict and must return a ready-to-use
    strategy instance (``load_state_dict`` is still called afterwards).
    """
    _STRATEGY_REGISTRY[type_name] = factory


def mix_strategy_from_state(state: dict[str, Any]) -> MixStrategy:
    t = state.get("type")
    factory = _STRATEGY_REGISTRY.get(t)  # type: ignore[arg-type]
    if factory is None:
        raise ValueError(f"unknown strategy type: {t!r}")
    s = factory(state)
    s.load_state_dict(state)
    return s


def mix_strategy_from_name(
    name: str,
    *,
    temperature: float | None = None,
) -> MixStrategy:
    """Build a built-in MixStrategy from a config-style name (see COMPOSABLE_MIX_STRATEGIES)."""
    if name not in COMPOSABLE_MIX_STRATEGIES:
        raise ValueError(f"Unhandled mix_strategy: {name!r}")
    state: dict[str, Any] = {"type": name}
    if name == MIX_STRATEGY_WEIGHTED_RANDOM:
        state["temperature"] = 1.0 if temperature is None else float(temperature)
    return mix_strategy_from_state(state)


def clone_mix_strategy(strategy: Optional[MixStrategy]) -> Optional[MixStrategy]:
    if strategy is None:
        return None
    return mix_strategy_from_state(strategy.state_dict())


# ============================================================
# Leaf traversal policies
# ============================================================

class TraversalPolicy(Protocol):
    def set_epoch(self, epoch: int) -> None:
        ...

    def next_index(self) -> int:
        ...

    def state_dict(self) -> dict[str, Any]:
        ...

    def load_state_dict(self, state: dict[str, Any]) -> None:
        ...


class PermutationTraversal:
    """Coverage-first traversal over local indices ``[0, length)``.

    Yields a full permutation each epoch; when exhausted wraps around and
    draws a fresh permutation (using the same RNG, so the sequence is
    deterministic given a fixed seed + epoch).  With ``shuffle=False``
    the "permutation" is simply the identity order.
    """

    def __init__(
        self,
        length: int,
        *,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        if length <= 0:
            raise ValueError("length must be > 0")
        self.length = int(length)
        self.shuffle = bool(shuffle)
        self.base_seed = seed
        self.epoch = 0
        self._cycle = 0
        self._rng = seed_to_rng(seed)
        self._perm = self._make_perm()
        self._ptr = 0

    def _make_perm(self) -> np.ndarray:
        if self.shuffle:
            return self._rng.permutation(self.length)
        return np.arange(self.length, dtype=np.int64)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._cycle = 0
        seed = None if self.base_seed is None else self.base_seed + self.epoch
        self._rng = seed_to_rng(seed)
        self._perm = self._make_perm()
        self._ptr = 0

    def _refresh_cycle(self) -> None:
        self._cycle += 1
        # Advance RNG naturally; no need to reseed every cycle
        self._perm = self._make_perm()
        self._ptr = 0

    def next_index(self) -> int:
        if self._ptr >= self.length:
            self._refresh_cycle()
        idx = int(self._perm[self._ptr])
        self._ptr += 1
        return idx

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": "permutation",
            "length": self.length,
            "shuffle": self.shuffle,
            "base_seed": self.base_seed,
            "epoch": self.epoch,
            "cycle": self._cycle,
            "perm": self._perm.tolist(),
            "ptr": self._ptr,
            "rng_state": self._rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.length = int(state["length"])
        self.shuffle = bool(state["shuffle"])
        self.base_seed = state["base_seed"]
        self.epoch = int(state["epoch"])
        self._cycle = int(state["cycle"])
        self._rng = np.random.default_rng()
        self._rng.bit_generator.state = state["rng_state"]
        self._perm = np.asarray(state["perm"], dtype=np.int64)
        self._ptr = int(state["ptr"])


# ============================================================
# Leaf node
# ============================================================

class LeafNode:
    """
    Leaf node over a contiguous local index space mapped to global index space.

    The node itself is agnostic to what those indices mean:
    - a child dataset in ConcatDataset
    - an episode bucket
    - a shard
    - a window list
    """

    def __init__(
        self,
        length: int,
        offset: int,
        *,
        traversal: Optional[TraversalPolicy] = None,
        shuffle: bool = True,
        seed: Optional[int] = None,
        name: Optional[str] = None,
    ) -> None:
        if length <= 0:
            raise ValueError("length must be > 0")
        self.length = int(length)
        self.offset = int(offset)
        self.name = name or f"leaf_{offset}_{length}"
        self.traversal = traversal or PermutationTraversal(
            length=self.length,
            shuffle=shuffle,
            seed=seed,
        )

    def set_epoch(self, epoch: int) -> None:
        self.traversal.set_epoch(epoch)

    def plan(self, budget: int) -> Iterator[int]:
        budget = int(budget)
        if budget < 0:
            raise ValueError("budget must be >= 0")
        for _ in range(budget):
            yield self.offset + self.traversal.next_index()

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": "leaf",
            "length": self.length,
            "offset": self.offset,
            "name": self.name,
            "traversal": self.traversal.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.length = int(state["length"])
        self.offset = int(state["offset"])
        self.name = state["name"]
        self.traversal.load_state_dict(state["traversal"])


# ============================================================
# Weight schedulers
# ============================================================


def resolve_weights(weights: WeightSpec, epoch: int, expected_len: int) -> np.ndarray:
    w = weights(epoch) if callable(weights) else weights
    if len(w) != expected_len:
        raise ValueError(
            f"weights length must match number of children: "
            f"got {len(w)} vs expected {expected_len}"
        )
    return normalize_weights(w)


# ============================================================
# Internal mix node
# ============================================================

class MixNode:
    """Internal node that delegates budget allocation and ordering to a MixStrategy.

    Responsibilities:
      1) Resolve this epoch's child weights (static or callable).
      2) Forward to the plugged-in MixStrategy for quota + ordering.
      3) Propagate epoch / state_dict to children.
    """

    def __init__(
        self,
        children: Sequence[SamplerNode],
        weights: WeightSpec,
        *,
        strategy: Optional[MixStrategy] = None,
        seed: Optional[int] = None,
        name: Optional[str] = None,
    ) -> None:
        if len(children) == 0:
            raise ValueError("children must not be empty")
        self.children = list(children)
        self.weights = weights
        self.name = name or "mix"
        self.strategy: MixStrategy = strategy or LargestRemainderStrategy()
        self.base_seed = seed
        self._rng = seed_to_rng(seed)
        self.epoch = 0

    def current_weights(self) -> np.ndarray:
        return resolve_weights(self.weights, self.epoch, len(self.children))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        seed = None if self.base_seed is None else self.base_seed + self.epoch
        self._rng = seed_to_rng(seed)
        for child in self.children:
            child.set_epoch(epoch)

    def plan(self, budget: int) -> Iterator[int]:
        budget = int(budget)
        if budget < 0:
            raise ValueError("budget must be >= 0")
        if budget == 0:
            return

        w = self.current_weights()
        yield from self.strategy.plan(self.children, w, budget, self._rng)

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": "mix",
            "name": self.name,
            "epoch": self.epoch,
            "base_seed": self.base_seed,
            "rng_state": self._rng.bit_generator.state,
            "strategy": self.strategy.state_dict(),
            "children": [child.state_dict() for child in self.children],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.name = state["name"]
        self.epoch = int(state["epoch"])
        self.base_seed = state.get("base_seed")
        self._rng = np.random.default_rng()
        if "rng_state" in state:
            self._rng.bit_generator.state = state["rng_state"]
        else:
            seed = None if self.base_seed is None else self.base_seed + self.epoch
            self._rng = seed_to_rng(seed)
        if "strategy" in state:
            self.strategy = mix_strategy_from_state(state["strategy"])
        child_states = state["children"]
        if len(child_states) != len(self.children):
            raise ValueError("child state length mismatch")
        for child, child_state in zip(self.children, child_states):
            child.load_state_dict(child_state)


# ============================================================
# Public sampler
# ============================================================

class ComposableSampler(Sampler[int]):
    """
    General-purpose hierarchical sampler for map-style datasets.

    Default use:
        sampler = ComposableSampler.from_dataset_lengths(
            dataset_lengths=[len(ds1), len(ds2), len(ds3)],
            weights=[0.5, 0.3, 0.2],
            num_samples=sum(map(len, [ds1, ds2, ds3])),
            seed=42,
        )

    Advanced use:
        root = MixNode(...)
        sampler = ComposableSampler(root=root, num_samples=10000)
    """

    def __init__(
        self,
        *,
        root: SamplerNode,
        num_samples: int,
    ) -> None:
        self.root = root
        self.num_samples = int(num_samples)
        if self.num_samples <= 0:
            raise ValueError("num_samples must be > 0")
        self.epoch = 0
        self.root.set_epoch(0)

    @classmethod
    def from_dataset_lengths(
        cls,
        dataset_lengths: Sequence[int],
        *,
        weights: Optional[Sequence[float]] = None,
        num_samples: Optional[int] = None,
        seed: Optional[int] = None,
        shuffle_within_leaf: bool = True,
        strategy: Optional[MixStrategy] = None,
    ) -> "ComposableSampler":
        lengths = np.asarray(list(dataset_lengths), dtype=np.int64)
        if lengths.ndim != 1 or lengths.size == 0 or np.any(lengths <= 0):
            raise ValueError("dataset_lengths must be a non-empty 1D array of positive integers")

        offsets = np.cumsum(np.concatenate(([0], lengths[:-1])), dtype=np.int64)
        leaves = [
            LeafNode(
                length=int(lengths[i]),
                offset=int(offsets[i]),
                shuffle=shuffle_within_leaf,
                seed=None if seed is None else seed + i * 10007,
                name=f"dataset_{i}",
            )
            for i in range(len(lengths))
        ]
        if weights is None:
            weights = [1.0] * len(leaves)

        root = MixNode(
            children=leaves,
            weights=weights,
            strategy=strategy,
            seed=seed,
            name="root",
        )
        if num_samples is None:
            num_samples = int(lengths.sum())
        return cls(root=root, num_samples=num_samples)

    def __iter__(self) -> Iterator[int]:
        yield from self.root.plan(self.num_samples)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.root.set_epoch(epoch)

    def state_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "num_samples": self.num_samples,
            "root": self.root.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.epoch = int(state["epoch"])
        self.num_samples = int(state["num_samples"])
        self.root.load_state_dict(state["root"])


# ============================================================
# Optional helpers for building deeper hierarchies
# ============================================================

@dataclass
class EpisodeSpec:
    length: int
    offset: int
    weight: float = 1.0
    name: Optional[str] = None


def build_episode_mix_node(
    episodes: Sequence[EpisodeSpec],
    *,
    seed: Optional[int] = None,
    shuffle_within_episode: bool = True,
    strategy: Optional[MixStrategy] = None,
    name: str = "episode_mix",
) -> MixNode:
    """
    Build:
        task
          ├── episode_1
          ├── episode_2
          └── ...
    """
    leaves: list[LeafNode] = []
    weights: list[float] = []

    for i, ep in enumerate(episodes):
        if ep.length <= 0:
            raise ValueError("episode length must be > 0")
        leaves.append(
            LeafNode(
                length=ep.length,
                offset=ep.offset,
                shuffle=shuffle_within_episode,
                seed=None if seed is None else seed + i * 4099,
                name=ep.name or f"episode_{i}",
            )
        )
        weights.append(float(ep.weight))

    return MixNode(children=leaves, weights=weights, strategy=strategy, seed=seed, name=name)


def build_source_task_hierarchy(
    source_to_task_to_episodes: dict[str, dict[str, Sequence[EpisodeSpec]]],
    *,
    source_weights: Optional[dict[str, float]] = None,
    task_weights: Optional[dict[str, dict[str, float]]] = None,
    seed: Optional[int] = None,
    source_strategy: Optional[MixStrategy] = None,
    task_strategy: Optional[MixStrategy] = None,
    episode_strategy: Optional[MixStrategy] = None,
) -> MixNode:
    """
    Build a 3-level hierarchy:
        root
          ├── source
          │     ├── task
          │     │     ├── episode
          │     │     └── ...
          │     └── ...
          └── ...
    """
    source_nodes: list[MixNode] = []
    source_ws: list[float] = []

    source_names = list(source_to_task_to_episodes.keys())
    for s_idx, source_name in enumerate(source_names):
        task_map = source_to_task_to_episodes[source_name]
        task_names = list(task_map.keys())

        task_nodes: list[MixNode] = []
        task_ws: list[float] = []

        for t_idx, task_name in enumerate(task_names):
            task_node = build_episode_mix_node(
                task_map[task_name],
                seed=None if seed is None else seed + s_idx * 100003 + t_idx * 1009,
                shuffle_within_episode=True,
                strategy=clone_mix_strategy(episode_strategy),
                name=f"{source_name}/{task_name}",
            )
            task_nodes.append(task_node)

            if task_weights is None:
                task_ws.append(1.0)
            else:
                task_ws.append(float(task_weights.get(source_name, {}).get(task_name, 1.0)))

        source_node = MixNode(
            children=task_nodes,
            weights=task_ws,
            strategy=clone_mix_strategy(task_strategy),
            seed=None if seed is None else seed + s_idx * 700001,
            name=source_name,
        )
        source_nodes.append(source_node)

        if source_weights is None:
            source_ws.append(1.0)
        else:
            source_ws.append(float(source_weights.get(source_name, 1.0)))

    return MixNode(
        children=source_nodes,
        weights=source_ws,
        strategy=clone_mix_strategy(source_strategy),
        seed=seed,
        name="root",
    )


def _uniform_weights(children: Sequence[SamplerNode]) -> Sequence[float]:
    return [1.0] * len(children)


class Compose:
    @staticmethod
    def _make(
        children: Sequence[SamplerNode],
        weights: Optional[WeightSpec],
        *,
        strategy: MixStrategy,
        seed: Optional[int],
        name: Optional[str],
        default_name: str,
    ) -> MixNode:
        return MixNode(
            children=children,
            weights=weights if weights is not None else _uniform_weights(children),
            strategy=strategy,
            seed=seed,
            name=name or default_name,
        )

    @staticmethod
    def largest_remainder(
        children: Sequence[SamplerNode],
        weights: Optional[WeightSpec] = None,
        *,
        seed: Optional[int] = None,
        name: Optional[str] = None,
    ) -> MixNode:
        return Compose._make(
            children,
            weights,
            strategy=LargestRemainderStrategy(),
            seed=seed,
            name=name,
            default_name=MIX_STRATEGY_LARGEST_REMAINDER,
        )

    @staticmethod
    def weighted_random(
        children: Sequence[SamplerNode],
        weights: Optional[WeightSpec] = None,
        *,
        temperature: float = 1.0,
        seed: Optional[int] = None,
        name: Optional[str] = None,
    ) -> MixNode:
        return Compose._make(
            children,
            weights,
            strategy=WeightedRandomStrategy(temperature=temperature),
            seed=seed,
            name=name,
            default_name=MIX_STRATEGY_WEIGHTED_RANDOM,
        )

    @staticmethod
    def round_robin(
        children: Sequence[SamplerNode],
        weights: Optional[WeightSpec] = None,
        *,
        seed: Optional[int] = None,
        name: Optional[str] = None,
    ) -> MixNode:
        return Compose._make(
            children,
            weights,
            strategy=RoundRobinStrategy(),
            seed=seed,
            name=name,
            default_name=MIX_STRATEGY_ROUND_ROBIN,
        )

    @staticmethod
    def shuffle_sequential(
        children: Sequence[SamplerNode],
        weights: Optional[WeightSpec] = None,
        *,
        seed: Optional[int] = None,
        name: Optional[str] = None,
    ) -> MixNode:
        return Compose._make(
            children,
            weights,
            strategy=ShuffleSequentialStrategy(),
            seed=seed,
            name=name,
            default_name=MIX_STRATEGY_SHUFFLE_SEQUENTIAL,
        )

    @staticmethod
    def stratified_random(
        children: Sequence[SamplerNode],
        weights: Optional[WeightSpec] = None,
        *,
        seed: Optional[int] = None,
        name: Optional[str] = None,
    ) -> MixNode:
        return Compose._make(
            children,
            weights,
            strategy=StratifiedRandomStrategy(),
            seed=seed,
            name=name,
            default_name=MIX_STRATEGY_STRATIFIED_RANDOM,
        )


# ============================================================
# Runnable examples (python -m openpi.training.composable_sampler)
# ============================================================

def _example_flat() -> None:
    """Flat mix over 3 datasets with Largest-Remainder (default)."""
    sampler = ComposableSampler.from_dataset_lengths(
        dataset_lengths=[1000, 500, 250],
        weights=[0.5, 0.3, 0.2],
        num_samples=1750,
        seed=42,
    )
    sampler.set_epoch(0)
    idxs = list(sampler)
    assert len(idxs) == 1750
    # Reproducible: same seed + epoch always yields the same sequence.
    sampler.set_epoch(0)
    assert list(sampler) == idxs


def _example_strategy_comparison() -> None:
    """Compare prefix proportions across all five strategies.

    With weights [0.5, 0.3, 0.2] and budget=30 the expected quotas are
    [15, 9, 6].  The strategies differ in *ordering*, not in total counts.
    """
    import numpy as np

    lengths = (300, 300, 300)
    weights = [0.5, 0.3, 0.2]
    budget = 30

    def make_leaves() -> list[LeafNode]:
        offsets = [0, 300, 600]
        return [LeafNode(length=n, offset=o, shuffle=False, seed=i)
                for i, (n, o) in enumerate(zip(lengths, offsets))]

    def child_counts(idxs: list[int]) -> list[int]:
        bounds = [300, 600, 900]
        counts = [0, 0, 0]
        for x in idxs:
            counts[next(j for j, b in enumerate(bounds) if x < b)] += 1
        return counts

    # Deterministic strategies produce exact quotas [15, 9, 6].
    for name, strategy in [
        (MIX_STRATEGY_LARGEST_REMAINDER, LargestRemainderStrategy()),
        (MIX_STRATEGY_ROUND_ROBIN, RoundRobinStrategy()),
        (MIX_STRATEGY_SHUFFLE_SEQUENTIAL, ShuffleSequentialStrategy()),
        (MIX_STRATEGY_STRATIFIED_RANDOM, StratifiedRandomStrategy()),
    ]:
        leaves = make_leaves()
        for leaf in leaves:
            leaf.set_epoch(0)
        node = MixNode(children=leaves, weights=weights, strategy=strategy, seed=0)
        node.set_epoch(0)
        counts = child_counts(list(node.plan(budget)))
        assert counts == [15, 9, 6], f"{name}: unexpected counts {counts}"

    # WeightedRandom uses multinomial sampling; check totals and rough proportions.
    leaves = make_leaves()
    for leaf in leaves:
        leaf.set_epoch(0)
    node = MixNode(children=leaves, weights=weights, strategy=WeightedRandomStrategy(), seed=0)
    node.set_epoch(0)
    counts = child_counts(list(node.plan(budget)))
    assert sum(counts) == budget
    assert counts[0] >= counts[1] >= counts[2], f"weighted_random order unexpected: {counts}"


def _example_nested_compose() -> None:
    """Nested composition: temperature-scaled inner mix + round-robin outer."""
    leaves = [
        LeafNode(length=100, offset=0,   seed=1, name="l0"),
        LeafNode(length=100, offset=100, seed=2, name="l1"),
        LeafNode(length=100, offset=200, seed=3, name="l2"),
    ]
    for leaf in leaves:
        leaf.set_epoch(0)

    # Inner node mixes l0/l1 with temperature sharpening toward l0.
    inner = Compose.weighted_random(
        leaves[:2], weights=[0.7, 0.3], temperature=0.5, seed=10, name="inner"
    )
    # Outer node mixes inner/l2 with round-robin.
    root = Compose.round_robin([inner, leaves[2]], weights=[0.8, 0.2], seed=20, name="root")
    root.set_epoch(0)

    idxs = list(root.plan(200))
    assert len(idxs) == 200
    # l2 (offset 200-299) should contribute ~20 % of samples.
    l2_count = sum(1 for x in idxs if 200 <= x < 300)
    assert 25 <= l2_count <= 55, f"unexpected l2 count: {l2_count}"


def _example_hierarchical() -> None:
    """3-level source → task → episode hierarchy."""
    root = build_source_task_hierarchy(
        {
            "base": {
                "open_door": [
                    EpisodeSpec(length=120, offset=0,   weight=1.0, name="base/open_door/ep0"),
                    EpisodeSpec(length=90,  offset=120, weight=1.0, name="base/open_door/ep1"),
                ],
                "close_door": [
                    EpisodeSpec(length=80, offset=210, weight=1.0, name="base/close_door/ep0"),
                ],
            },
            "dagger": {
                "open_door": [
                    EpisodeSpec(length=40, offset=290, weight=2.0, name="dagger/open_door/recovery0"),
                ],
                "insert_cloth": [
                    EpisodeSpec(length=70, offset=330, weight=1.0, name="dagger/insert_cloth/ep0"),
                ],
            },
        },
        source_weights={"base": 0.8, "dagger": 0.2},
        task_weights={
            "base":   {"open_door": 1.0, "close_door": 1.0},
            "dagger": {"open_door": 2.0, "insert_cloth": 1.0},
        },
        seed=123,
    )
    sampler = ComposableSampler(root=root, num_samples=400)
    sampler.set_epoch(3)
    idxs = list(sampler)
    assert len(idxs) == 400
    # All indices must fall within the known global range [0, 400).
    assert all(0 <= x < 400 for x in idxs)


if __name__ == "__main__":
    _example_flat()
    _example_strategy_comparison()
    _example_nested_compose()
    _example_hierarchical()
    print("All examples passed.")