from __future__ import annotations

"""Hierarchical index sampler for map-style datasets.

Build trees of :class:`LeafNode` (per-dataset indices) and :class:`MixNode` (weighted mixing),
then wrap the root in :class:`ComposableSampler` for PyTorch. The :class:`Compose` namespace is
the supported entry point.

Layers (bottom-up):

- :class:`TraversalPolicy` — how a leaf walks its local index range.
- :class:`LeafNode` — maps local indices to global indices via ``offset``.
- :class:`MixNode` — splits a sample budget across children using a :class:`MixStrategy`.
- :class:`ComposableSampler` — ``torch.utils.data.Sampler`` over the root node.

Typical usage::

    # Flat mix over concatenated datasets
    sampler = Compose.flat([1000, 500], weights=[0.7, 0.3], seed=42)

    # Custom tree
    leaves = [Compose.leaf(100, 0, seed=1), Compose.leaf(200, 100, seed=2)]
    root = Compose.mix(leaves, [0.6, 0.4], strategy="round_robin")
    sampler = Compose.sampler(root, 300)

    # Episode list
    node = Compose.episodes(specs, strategy="weighted_random", temperature=0.5)

    # Three-level tree: source -> task -> episode
    root = Compose.hierarchy(data, source_weights={...}, episode_strategy="weighted_random")

``strategy`` and ``traversal`` accept:

- ``str`` — registry name (e.g. ``"largest_remainder"``, ``"permutation"``).
- A concrete instance (e.g. ``WeightedRandomStrategy(temperature=0.5)``).
- ``None`` — use the default for that API.
- For ``traversal`` only: a callable ``(length) -> TraversalPolicy``.
"""

from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional, Protocol, Sequence
import math

import numpy as np
from torch.utils.data import Sampler

from openpi.training.mixing import (
    COMPOSABLE_QUOTA_ALLOCATOR_TYPES,
    COMPOSABLE_QUOTA_SCHEDULER_TYPES,
    QuotaAllocator,
    QuotaScheduler,
    LargestRemainderAllocator,
    MultinomialAllocator,
    SmoothInterleaveSchedule,
    RandomShuffleSchedule,
    RoundRobinSchedule,
    ShuffleSequentialSchedule,
    StratifiedRandomSchedule,
    quota_allocator_from_registry,
    quota_scheduler_from_registry,
    restore_quota_and_schedule,
    register_quota_allocator,
    register_quota_scheduler,
    registered_quota_allocator_types,
    registered_quota_scheduler_types,
    largest_remainder_allocate,
    normalize_weights,
)


__all__ = [
    # ── Unified entry point (recommended) ────────────────────
    "Compose",
    # ── Core types ───────────────────────────────────────────
    "SamplerNode",
    "TraversalPolicy",
    "MixStrategy",
    "LeafNode",
    "MixNode",
    "ComposableSampler",
    "EpisodeSpec",
    # ── Strategy constants (for string-based construction) ───
    "MIX_STRATEGY_LARGEST_REMAINDER",
    "MIX_STRATEGY_WEIGHTED_RANDOM",
    "MIX_STRATEGY_ROUND_ROBIN",
    "MIX_STRATEGY_SHUFFLE_SEQUENTIAL",
    "MIX_STRATEGY_STRATIFIED_RANDOM",
    "COMPOSABLE_MIX_STRATEGIES",
    # ── Traversal constants ──────────────────────────────────
    "TRAVERSAL_AFFINE_PERMUTATION",
    "TRAVERSAL_PERMUTATION",
    "TRAVERSAL_RANDOM_WITH_REPLACEMENT",
    "TRAVERSAL_BUFFERED_SHUFFLE",
    "COMPOSABLE_TRAVERSAL_POLICIES",
    # ── Named strategy classes (for direct instantiation) ────
    "LargestRemainderStrategy",
    "WeightedRandomStrategy",
    "RoundRobinStrategy",
    "ShuffleSequentialStrategy",
    "StratifiedRandomStrategy",
    # ── Named traversal classes (for direct instantiation) ───
    "AffinePermutationTraversal",
    "PermutationTraversal",
    "RandomWithReplacementTraversal",
    "BufferedShuffleTraversal",
    # ── Utilities ────────────────────────────────────────────
    "WeightSpec",
    "largest_remainder_allocate",
    "make_traversal",
    "mix_strategy_from_name",
    "resolve_weights",
    "normalize_weights",
    # ── Advanced (state restore, custom registries) ──────────
    "CompositeMixStrategy",
    "custom_mix_strategy",
    "mix_strategy_from_state",
    "clone_mix_strategy",
    "traversal_from_state",
    "clone_traversal",
    "register_strategy",
    "register_traversal_policy",
    "QuotaAllocator",
    "QuotaScheduler",
    "COMPOSABLE_QUOTA_SCHEDULER_TYPES",
    "quota_scheduler_from_registry",
    "registered_quota_scheduler_types",
    "mix_strategy_from_registry",
    "traversal_from_registry",
    "register_quota_allocator",
    "register_quota_scheduler",
]


# Public type aliases.
WeightSpec = Sequence[float] | Callable[[int], Sequence[float]]
TraversalFactory = Callable[[int], "TraversalPolicy"]  # (length) -> policy

# Traversal identifiers.
TRAVERSAL_AFFINE_PERMUTATION       = "affine_permutation"
TRAVERSAL_PERMUTATION              = "permutation"
TRAVERSAL_RANDOM_WITH_REPLACEMENT  = "random_with_replacement"
TRAVERSAL_BUFFERED_SHUFFLE         = "buffered_shuffle"

COMPOSABLE_TRAVERSAL_POLICIES: frozenset[str] = frozenset({
    TRAVERSAL_AFFINE_PERMUTATION,
    TRAVERSAL_PERMUTATION,
    TRAVERSAL_RANDOM_WITH_REPLACEMENT,
    TRAVERSAL_BUFFERED_SHUFFLE,
})

# Mix strategy identifiers used in state_dicts and mix_strategy_from_name().
MIX_STRATEGY_LARGEST_REMAINDER = "largest_remainder"
MIX_STRATEGY_WEIGHTED_RANDOM   = "weighted_random"
MIX_STRATEGY_ROUND_ROBIN       = "round_robin"
MIX_STRATEGY_SHUFFLE_SEQUENTIAL= "shuffle_sequential"
MIX_STRATEGY_STRATIFIED_RANDOM = "stratified_random"

COMPOSABLE_MIX_STRATEGIES: frozenset[str] = frozenset({
    MIX_STRATEGY_LARGEST_REMAINDER,
    MIX_STRATEGY_WEIGHTED_RANDOM,
    MIX_STRATEGY_ROUND_ROBIN,
    MIX_STRATEGY_SHUFFLE_SEQUENTIAL,
    MIX_STRATEGY_STRATIFIED_RANDOM,
})


# ============================================================
# Utility
# ============================================================

def seed_to_rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)


def _epoch_rng(base_seed: Optional[int], epoch: int) -> np.random.Generator:
    """Derive a per-epoch RNG from a base seed (centralises the ternary pattern)."""
    return seed_to_rng(None if base_seed is None else base_seed + epoch)


# ============================================================
# Protocols
# ============================================================

class SamplerNode(Protocol):
    def set_epoch(self, epoch: int) -> None: ...
    def plan(self, budget: int) -> Iterator[int]: ...
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state: dict[str, Any]) -> None: ...


class MixStrategy(Protocol):
    """How a MixNode distributes a budget across children and orders their output.

    Named implementations compose QuotaAllocator + QuotaScheduler; typical use is a
    *Strategy class, mix_strategy_from_name, or MixNode(..., strategy=...).
    """

    def plan(
        self,
        children: Sequence["SamplerNode"],
        weights: np.ndarray,
        budget: int,
        rng: np.random.Generator,
    ) -> Iterator[int]: ...

    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state: dict[str, Any]) -> None: ...


class TraversalPolicy(Protocol):
    """How a LeafNode advances through its local index space."""

    def set_epoch(self, epoch: int) -> None: ...
    def next_index(self) -> int: ...
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state: dict[str, Any]) -> None: ...


# ============================================================
# Public MixStrategy implementations
# ============================================================

class BaseMixStrategy:
    """Internal base: implements plan() via an allocator + scheduler pair."""

    _allocator: QuotaAllocator
    _scheduler: QuotaScheduler

    def plan(
        self,
        children: Sequence[SamplerNode],
        weights: np.ndarray,
        budget: int,
        rng: np.random.Generator,
    ) -> Iterator[int]:
        budget = int(budget)
        if budget <= 0:
            return
        quotas = np.asarray(self._allocator.allocate(weights, budget, rng), dtype=np.int64)
        streams = [child.plan(int(q)) for child, q in zip(children, quotas)]
        yield from self._scheduler.schedule(streams, quotas, rng)


class CompositeMixStrategy(BaseMixStrategy):
    """Public advanced MixStrategy built from a QuotaAllocator + QuotaScheduler.

    This preserves the simple named wrappers for common use, while enabling
    custom mixing behaviour without exposing MixNode internals.
    """

    def __init__(
        self,
        *,
        quota_allocator: QuotaAllocator,
        quota_scheduler: QuotaScheduler,
    ) -> None:
        self._allocator = quota_allocator  # type: ignore[assignment]
        self._scheduler = quota_scheduler  # type: ignore[assignment]

    @property
    def quota_allocator(self) -> QuotaAllocator:
        return self._allocator  # type: ignore[return-value]

    @property
    def quota_scheduler(self) -> QuotaScheduler:
        return self._scheduler  # type: ignore[return-value]

    def __repr__(self) -> str:
        return (
            f"CompositeMixStrategy(quota_allocator={self._allocator!r}, "
            f"quota_scheduler={self._scheduler!r})"
        )

    def state_dict(self) -> dict[str, Any]:
        scheduler_state = self._scheduler.state_dict()
        return {
            "type": "composite",
            "quota_allocator": self._allocator.state_dict(),
            "quota_scheduler": scheduler_state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        qa_state = state["quota_allocator"]
        quota_scheduler_state = state.get("quota_scheduler")
        if not isinstance(quota_scheduler_state, dict):
            raise ValueError("composite strategy state missing quota_scheduler")
        qa, sp = restore_quota_and_schedule(
            qa_state,
            quota_scheduler_state,
            error_prefix="Cannot restore composite strategy",
        )
        self._allocator = qa
        self._scheduler = sp


def custom_mix_strategy(
    quota_allocator: QuotaAllocator,
    quota_scheduler: QuotaScheduler,
) -> CompositeMixStrategy:
    """Build a custom MixStrategy from public allocator + scheduler components."""
    return CompositeMixStrategy(
        quota_allocator=quota_allocator,
        quota_scheduler=quota_scheduler,
    )


class LargestRemainderStrategy(CompositeMixStrategy):
    """Exact proportional allocation with smooth interleaving (default strategy).

    Guarantees integer quotas summing to budget, with the best possible
    approximation of the target weights. Child streams are interleaved so
    each stays as close to its fraction as possible throughout the epoch.
    """

    def __init__(self) -> None:
        super().__init__(quota_allocator=LargestRemainderAllocator(), quota_scheduler=SmoothInterleaveSchedule())

    def __repr__(self) -> str:
        return "LargestRemainderStrategy()"

    def state_dict(self) -> dict[str, Any]:
        return {"type": MIX_STRATEGY_LARGEST_REMAINDER}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        pass  # stateless


class WeightedRandomStrategy(CompositeMixStrategy):
    """Multinomial quota allocation with random interleaving.

    Quotas are drawn from a multinomial distribution, so they vary each epoch
    (unbiased in expectation). A temperature < 1 sharpens the distribution
    toward the highest-weight child; > 1 flattens it toward uniform.

    Args:
        temperature: Sharpens (< 1) or flattens (> 1) the weight distribution.
    """

    def __init__(self, *, temperature: float = 1.0) -> None:
        super().__init__(quota_allocator=MultinomialAllocator(temperature=temperature), quota_scheduler=RandomShuffleSchedule())

    def __repr__(self) -> str:
        return f"WeightedRandomStrategy(temperature={self.temperature!r})"

    @property
    def temperature(self) -> float:
        return self._allocator.temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        value = float(value)
        if value <= 0:
            raise ValueError("temperature must be > 0")
        self._allocator.temperature = value

    def state_dict(self) -> dict[str, Any]:
        return {"type": MIX_STRATEGY_WEIGHTED_RANDOM, "temperature": self.temperature}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._allocator = MultinomialAllocator(temperature=float(state.get("temperature", 1.0)))
        self._scheduler = RandomShuffleSchedule()


class RoundRobinStrategy(CompositeMixStrategy):
    """Exact proportional allocation with strict round-robin interleaving."""

    def __init__(self) -> None:
        super().__init__(quota_allocator=LargestRemainderAllocator(), quota_scheduler=RoundRobinSchedule())

    def __repr__(self) -> str:
        return "RoundRobinStrategy()"

    def state_dict(self) -> dict[str, Any]:
        return {"type": MIX_STRATEGY_ROUND_ROBIN}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        pass


class ShuffleSequentialStrategy(CompositeMixStrategy):
    """Exact proportional allocation; child streams are consumed one at a time
    in a randomly shuffled order (each child's block is contiguous)."""

    def __init__(self) -> None:
        super().__init__(quota_allocator=LargestRemainderAllocator(), quota_scheduler=ShuffleSequentialSchedule())

    def __repr__(self) -> str:
        return "ShuffleSequentialStrategy()"

    def state_dict(self) -> dict[str, Any]:
        return {"type": MIX_STRATEGY_SHUFFLE_SEQUENTIAL}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        pass


class StratifiedRandomStrategy(CompositeMixStrategy):
    """Exact proportional allocation with stratified random interleaving.

    Each round, all active children are permuted and one sample is taken from
    each. This gives stronger short-term balance than LargestRemainder while
    still being random.
    """

    def __init__(self) -> None:
        super().__init__(quota_allocator=LargestRemainderAllocator(), quota_scheduler=StratifiedRandomSchedule())

    def __repr__(self) -> str:
        return "StratifiedRandomStrategy()"

    def state_dict(self) -> dict[str, Any]:
        return {"type": MIX_STRATEGY_STRATIFIED_RANDOM}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        pass


# ── Strategy registry ─────────────────────────────────────────────────────────

_STRATEGY_REGISTRY: dict[str, Callable[[dict[str, Any]], MixStrategy]] = {
    MIX_STRATEGY_LARGEST_REMAINDER:  lambda s: LargestRemainderStrategy(),
    MIX_STRATEGY_WEIGHTED_RANDOM:    lambda s: WeightedRandomStrategy(temperature=float(s.get("temperature", 1.0))),
    MIX_STRATEGY_ROUND_ROBIN:        lambda s: RoundRobinStrategy(),
    MIX_STRATEGY_SHUFFLE_SEQUENTIAL: lambda s: ShuffleSequentialStrategy(),
    MIX_STRATEGY_STRATIFIED_RANDOM:  lambda s: StratifiedRandomStrategy(),
    "composite":                    lambda s: CompositeMixStrategy(
        quota_allocator=LargestRemainderAllocator(),
        quota_scheduler=SmoothInterleaveSchedule(),
    ),
}


def register_strategy(
    type_name: str,
    factory: Callable[[dict[str, Any]], MixStrategy],
) -> None:
    """Register a custom MixStrategy factory for state_dict restoration."""
    _STRATEGY_REGISTRY[type_name] = factory


def registered_mix_strategy_types() -> frozenset[str]:
    """Keys currently in the mix-strategy registry (includes :func:`register_strategy` adds)."""
    return frozenset(_STRATEGY_REGISTRY.keys())


def mix_strategy_from_state(state: dict[str, Any]) -> MixStrategy:
    """Reconstruct a MixStrategy from its state dict (including legacy formats)."""
    t = state.get("type")

    # New format: type key directly names the strategy.
    factory = _STRATEGY_REGISTRY.get(t)  # type: ignore[arg-type]
    if factory is not None:
        strategy = factory(state)
        strategy.load_state_dict(state)
        return strategy

    # Composite strategy with nested quota allocator + quota stream scheduler.
    if t == "composite" and "quota_allocator" in state and "quota_scheduler" in state:
        return _strategy_from_legacy_composite(state)

    # Top-level composite payload.
    if "quota_allocator" in state and "quota_scheduler" in state:
        return _strategy_from_legacy_composite(state)

    raise ValueError(
        f"unknown strategy type: {t!r}; registered: {sorted(registered_mix_strategy_types())}"
    )


def _strategy_from_legacy_composite(state: dict[str, Any]) -> MixStrategy:
    """Reconstruct a MixStrategy from a composite state dict.

    Accepts the unified ``quota_scheduler`` field.
    """
    qa_state = state["quota_allocator"]
    sp_state = state.get("quota_scheduler")
    if not isinstance(sp_state, dict):
        raise ValueError("composite state missing quota_scheduler")
    qa, sp = restore_quota_and_schedule(
        qa_state,
        sp_state,
        error_prefix="Cannot restore legacy composite",
    )
    # Map back to the closest named strategy where possible.
    type_map = {
        ("largest_remainder", "smooth_interleave"): LargestRemainderStrategy,
        ("largest_remainder", "round_robin"):        RoundRobinStrategy,
        ("largest_remainder", "shuffle_sequential"): ShuffleSequentialStrategy,
        ("largest_remainder", "stratified_random"):  StratifiedRandomStrategy,
        ("multinomial",       "random_shuffle"):      lambda: WeightedRandomStrategy(
            temperature=float(qa_state.get("temperature", 1.0))
        ),
    }
    key = (qa_state.get("type"), sp_state.get("type"))
    cls = type_map.get(key)
    if cls is not None:
        s = cls()
        s.load_state_dict(state)
        return s
    # Fall through: return a generic BaseMixStrategy with the restored pair.
    class _RestoredStrategy(BaseMixStrategy):
        def state_dict(self):
            return {"type": "composite",
                    "quota_allocator": qa.state_dict(),
                    "quota_scheduler": sp.state_dict()}
        def load_state_dict(self, s): pass

    inst = _RestoredStrategy()
    inst._allocator = qa
    inst._scheduler = sp
    return inst


def mix_strategy_from_name(
    name: str,
    *,
    temperature: Optional[float] = None,
) -> MixStrategy:
    """Build a named built-in MixStrategy, optionally with parameters.

    Args:
        name: One of COMPOSABLE_MIX_STRATEGIES.
        temperature: Only used for ``weighted_random``.
    """
    if name not in COMPOSABLE_MIX_STRATEGIES:
        raise ValueError(
            f"unknown mix strategy: {name!r}. Choose from {sorted(COMPOSABLE_MIX_STRATEGIES)} "
            f"(registered keys: {sorted(registered_mix_strategy_types())})"
        )
    state: dict[str, Any] = {"type": name}
    if name == MIX_STRATEGY_WEIGHTED_RANDOM and temperature is not None:
        state["temperature"] = float(temperature)
    return mix_strategy_from_state(state)


def mix_strategy_from_registry(
    *,
    strategy_name: Optional[str] = None,
    quota_type: Optional[str] = None,
    quota_scheduler_type: Optional[str] = None,
    quota_state: Optional[dict[str, Any]] = None,
    quota_scheduler_state: Optional[dict[str, Any]] = None,
    temperature: Optional[float] = None,
) -> MixStrategy:
    """Build a :class:`MixStrategy` from registry keys (unified entry for mix layer).

    Either:

    - ``strategy_name``: one of :data:`COMPOSABLE_MIX_STRATEGIES` (same as :func:`mix_strategy_from_name`).
        - ``quota_type`` and ``quota_scheduler_type``: composite strategy from
            :data:`COMPOSABLE_QUOTA_ALLOCATOR_TYPES` x :data:`COMPOSABLE_QUOTA_SCHEDULER_TYPES`
      (same as :func:`custom_mix_strategy`).

    ``temperature`` applies only when ``strategy_name`` is ``weighted_random``; for ``multinomial``
    allocation use *quota_state* or :func:`quota_allocator_from_registry` ``temperature=``.
    """
    if strategy_name is not None:
        if quota_type is not None or quota_scheduler_type is not None:
            raise ValueError(
                "mix_strategy_from_registry: pass either strategy_name or (quota_type, quota_scheduler_type), not both"
            )
        return mix_strategy_from_name(strategy_name, temperature=temperature)
    if quota_type is not None and quota_scheduler_type is not None:
        qa = quota_allocator_from_registry(quota_type, quota_state or {})
        sp = quota_scheduler_from_registry(
            quota_scheduler_type,
            quota_scheduler_state or {},
        )
        return custom_mix_strategy(qa, sp)
    if quota_type is not None or quota_scheduler_type is not None:
        raise ValueError(
            "quota_type and quota_scheduler_type must both be set for a composite mix strategy"
        )
    raise ValueError(
        "mix_strategy_from_registry requires strategy_name=... or quota_type=... and quota_scheduler_type=...; "
        f"registered mix keys: {sorted(registered_mix_strategy_types())}"
    )


def clone_mix_strategy(strategy: Optional[MixStrategy]) -> Optional[MixStrategy]:
    if strategy is None:
        return None
    return mix_strategy_from_state(strategy.state_dict())


# ============================================================
# Traversal policies
# ============================================================

class AffinePermutationTraversal:
    """Exact-coverage permutation with O(1) memory using an affine map.

    Each cycle uses x_t = (a·t + b) mod length where gcd(a, length) = 1,
    guaranteeing every index in [0, length) appears exactly once.
    Parameters (a, b) are re-randomised each cycle for different orderings.

    Prefer over PermutationTraversal when length ≥ ~500 000 (saves ≥ 4 MB RAM).
    Use make_traversal() to auto-select based on length.
    """

    MEMORY_BREAK_EVEN = 500_000

    def __init__(self, length: int, *, seed: Optional[int] = None) -> None:
        if length <= 0:
            raise ValueError("length must be > 0")
        self.length = int(length)
        self.base_seed = seed
        self.epoch = 0
        self._cycle = 0
        self._rng = seed_to_rng(seed)
        self._a = 1
        self._b = 0
        self._t = 0
        self._reset_cycle_params()

    def __repr__(self) -> str:
        return f"AffinePermutationTraversal(length={self.length!r}, base_seed={self.base_seed!r})"

    def _sample_coprime(self) -> int:
        if self.length == 1:
            return 1
        while True:
            a = int(self._rng.integers(1, self.length))
            if math.gcd(a, self.length) == 1:
                return a

    def _reset_cycle_params(self) -> None:
        self._a = self._sample_coprime()
        self._b = int(self._rng.integers(0, self.length))
        self._t = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._cycle = 0
        self._rng = _epoch_rng(self.base_seed, epoch)
        self._reset_cycle_params()

    def next_index(self) -> int:
        if self._t >= self.length:
            self._cycle += 1
            self._reset_cycle_params()
        idx = (self._a * self._t + self._b) % self.length
        self._t += 1
        return int(idx)

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": TRAVERSAL_AFFINE_PERMUTATION,
            "length": self.length, "base_seed": self.base_seed,
            "epoch": self.epoch, "cycle": self._cycle,
            "a": self._a, "b": self._b, "t": self._t,
            "rng_state": self._rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.length = int(state["length"]); self.base_seed = state["base_seed"]
        self.epoch = int(state["epoch"]); self._cycle = int(state["cycle"])
        self._a = int(state["a"]); self._b = int(state["b"]); self._t = int(state["t"])
        self._rng = np.random.default_rng()
        self._rng.bit_generator.state = state["rng_state"]


class PermutationTraversal:
    """Exact-coverage traversal: yields a full shuffled permutation each cycle.

    O(n) memory. For large leaves (≥ 500 000 samples) prefer AffinePermutationTraversal.
    With shuffle=False yields 0, 1, 2, … in order (useful for debugging).
    """

    def __init__(self, length: int, *, shuffle: bool = True, seed: Optional[int] = None) -> None:
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

    def __repr__(self) -> str:
        return f"PermutationTraversal(length={self.length!r}, shuffle={self.shuffle!r}, base_seed={self.base_seed!r})"

    def _make_perm(self) -> np.ndarray:
        return self._rng.permutation(self.length) if self.shuffle else np.arange(self.length, dtype=np.int64)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._cycle = 0
        self._rng = _epoch_rng(self.base_seed, epoch)
        self._perm = self._make_perm()
        self._ptr = 0

    def next_index(self) -> int:
        if self._ptr >= self.length:
            self._cycle += 1
            self._perm = self._make_perm()
            self._ptr = 0
        idx = int(self._perm[self._ptr])
        self._ptr += 1
        return idx

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": TRAVERSAL_PERMUTATION,
            "length": self.length, "shuffle": self.shuffle, "base_seed": self.base_seed,
            "epoch": self.epoch, "cycle": self._cycle,
            "perm": self._perm.tolist(), "ptr": self._ptr,
            "rng_state": self._rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.length = int(state["length"]); self.shuffle = bool(state["shuffle"])
        self.base_seed = state["base_seed"]; self.epoch = int(state["epoch"])
        self._cycle = int(state["cycle"])
        self._rng = np.random.default_rng()
        self._rng.bit_generator.state = state["rng_state"]
        self._perm = np.asarray(state["perm"], dtype=np.int64)
        self._ptr = int(state["ptr"])


class RandomWithReplacementTraversal:
    """IID random sampling with replacement over [0, length).

    No coverage guarantee: the same index can appear many times before others.
    Appropriate when iid sampling semantics are required exactly.
    """

    def __init__(self, length: int, *, seed: Optional[int] = None) -> None:
        if length <= 0:
            raise ValueError("length must be > 0")
        self.length = int(length)
        self.base_seed = seed
        self.epoch = 0
        self._rng = seed_to_rng(seed)

    def __repr__(self) -> str:
        return f"RandomWithReplacementTraversal(length={self.length!r}, base_seed={self.base_seed!r})"

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._rng = _epoch_rng(self.base_seed, epoch)

    def next_index(self) -> int:
        return int(self._rng.integers(0, self.length))

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": TRAVERSAL_RANDOM_WITH_REPLACEMENT,
            "length": self.length, "base_seed": self.base_seed, "epoch": self.epoch,
            "rng_state": self._rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.length = int(state["length"]); self.base_seed = state["base_seed"]
        self.epoch = int(state["epoch"])
        self._rng = np.random.default_rng()
        self._rng.bit_generator.state = state["rng_state"]


class BufferedShuffleTraversal:
    """Approximate shuffle via a fixed-size reservoir buffer.

    Visits every index exactly once per pass, but only keeps buffer_size indices
    in memory instead of the full permutation. Larger buffers approach a full
    permutation; smaller buffers reduce memory at the cost of local ordering.
    """

    def __init__(self, length: int, *, buffer_size: int, seed: Optional[int] = None) -> None:
        if length <= 0:
            raise ValueError("length must be > 0")
        if buffer_size <= 0:
            raise ValueError("buffer_size must be > 0")
        self.length = int(length)
        self.buffer_size = int(min(buffer_size, length))
        self.base_seed = seed
        self.epoch = 0
        self._cycle = 0
        self._rng = seed_to_rng(seed)
        self._cursor = 0
        self._buffer: list[int] = []
        self._fill_initial()

    def __repr__(self) -> str:
        return f"BufferedShuffleTraversal(length={self.length!r}, buffer_size={self.buffer_size!r}, base_seed={self.base_seed!r})"

    def _fill_initial(self) -> None:
        self._cursor = 0
        self._buffer = []
        while len(self._buffer) < self.buffer_size and self._cursor < self.length:
            self._buffer.append(self._cursor)
            self._cursor += 1

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._cycle = 0
        self._rng = _epoch_rng(self.base_seed, epoch)
        self._fill_initial()

    def next_index(self) -> int:
        if not self._buffer:
            self._cycle += 1
            self._fill_initial()
        j = int(self._rng.integers(0, len(self._buffer)))
        idx = self._buffer[j]
        if self._cursor < self.length:
            self._buffer[j] = self._cursor
            self._cursor += 1
        else:
            self._buffer.pop(j)
        return idx

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": TRAVERSAL_BUFFERED_SHUFFLE,
            "length": self.length, "buffer_size": self.buffer_size,
            "base_seed": self.base_seed, "epoch": self.epoch, "cycle": self._cycle,
            "cursor": self._cursor, "buffer": list(self._buffer),
            "rng_state": self._rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.length = int(state["length"]); self.buffer_size = int(state["buffer_size"])
        self.base_seed = state["base_seed"]; self.epoch = int(state["epoch"])
        self._cycle = int(state["cycle"]); self._cursor = int(state["cursor"])
        self._buffer = list(state["buffer"])
        self._rng = np.random.default_rng()
        self._rng.bit_generator.state = state["rng_state"]


# ── Traversal registry ────────────────────────────────────────────────────────

_TRAVERSAL_POLICY_REGISTRY: dict[str, Callable[[dict[str, Any]], TraversalPolicy]] = {
    TRAVERSAL_AFFINE_PERMUTATION: lambda s: AffinePermutationTraversal(
        length=int(s["length"]), seed=s.get("base_seed")),
    TRAVERSAL_PERMUTATION: lambda s: PermutationTraversal(
        length=int(s["length"]), shuffle=bool(s.get("shuffle", True)), seed=s.get("base_seed")),
    TRAVERSAL_RANDOM_WITH_REPLACEMENT: lambda s: RandomWithReplacementTraversal(
        length=int(s["length"]), seed=s.get("base_seed")),
    TRAVERSAL_BUFFERED_SHUFFLE: lambda s: BufferedShuffleTraversal(
        length=int(s["length"]), buffer_size=int(s.get("buffer_size", s["length"])),
        seed=s.get("base_seed")),
}


def register_traversal_policy(
    type_name: str,
    factory: Callable[[dict[str, Any]], TraversalPolicy],
) -> None:
    """Register a custom TraversalPolicy factory for state restoration."""
    _TRAVERSAL_POLICY_REGISTRY[type_name] = factory


def registered_traversal_policy_types() -> frozenset[str]:
    """Keys currently in the traversal-policy registry (includes :func:`register_traversal_policy` adds)."""
    return frozenset(_TRAVERSAL_POLICY_REGISTRY.keys())


def traversal_from_state(state: dict[str, Any]) -> TraversalPolicy:
    """Reconstruct a TraversalPolicy from its state dict."""
    t = state.get("type")
    factory = _TRAVERSAL_POLICY_REGISTRY.get(t)  # type: ignore[arg-type]
    if factory is None:
        raise ValueError(
            f"unknown traversal policy type: {t!r}; registered: {sorted(registered_traversal_policy_types())}"
        )
    traversal = factory(state)
    traversal.load_state_dict(state)
    return traversal


def clone_traversal(traversal: Optional[TraversalPolicy]) -> Optional[TraversalPolicy]:
    if traversal is None:
        return None
    return traversal_from_state(traversal.state_dict())


def make_traversal(
    length: int,
    *,
    seed: Optional[int] = None,
    policy: Optional[str] = None,
    shuffle: bool = True,
    buffer_size: Optional[int] = None,
) -> TraversalPolicy:
    """Build a TraversalPolicy, auto-selecting the best one when policy=None.

    Auto-selection (policy=None):
      length < 500 000  -> PermutationTraversal  (vectorised numpy, fastest)
      length ≥ 500 000  -> AffinePermutationTraversal  (O(1) memory, exact coverage)

    Both guarantee every index appears exactly once per cycle.

    Args:
        length: Leaf size.
        seed: RNG seed for reproducibility.
        policy: Override auto-selection. One of :func:`registered_traversal_policy_types` keys.
        shuffle: Only used for ``permutation`` policy; ignored otherwise.
        buffer_size: Required for ``buffered_shuffle``; ignored otherwise.

    Example::

        # Auto-select (recommended):
        traversal_factory=lambda length: make_traversal(length, seed=42)

        # Explicit policy:
        traversal_factory=lambda length: make_traversal(
            length, policy=TRAVERSAL_BUFFERED_SHUFFLE, buffer_size=1024
        )
    """
    if policy is None:
        if length >= AffinePermutationTraversal.MEMORY_BREAK_EVEN:
            return AffinePermutationTraversal(length=length, seed=seed)
        return PermutationTraversal(length=length, shuffle=shuffle, seed=seed)

    if policy not in _TRAVERSAL_POLICY_REGISTRY:
        raise ValueError(
            f"unknown traversal policy: {policy!r}. Registered: {sorted(registered_traversal_policy_types())}"
        )
    if policy == TRAVERSAL_AFFINE_PERMUTATION:
        return AffinePermutationTraversal(length=length, seed=seed)
    if policy == TRAVERSAL_PERMUTATION:
        return PermutationTraversal(length=length, shuffle=shuffle, seed=seed)
    if policy == TRAVERSAL_RANDOM_WITH_REPLACEMENT:
        return RandomWithReplacementTraversal(length=length, seed=seed)
    if policy == TRAVERSAL_BUFFERED_SHUFFLE:
        if buffer_size is None:
            raise ValueError("buffer_size must be provided for buffered_shuffle traversal")
        return BufferedShuffleTraversal(length=length, buffer_size=buffer_size, seed=seed)
    # Custom policy registered via register_traversal_policy
    state: dict[str, Any] = {
        "type": policy,
        "length": int(length),
        "shuffle": shuffle,
        "base_seed": seed,
    }
    if buffer_size is not None:
        state["buffer_size"] = buffer_size
    return traversal_from_state(state)


def traversal_from_registry(
    type_name: str,
    length: int,
    *,
    seed: Optional[int] = None,
    shuffle: bool = True,
    buffer_size: Optional[int] = None,
) -> TraversalPolicy:
    """Build a :class:`TraversalPolicy` from a :func:`registered_traversal_policy_types` key.

    This is the traversal-layer counterpart to :func:`mix_strategy_from_name` / registry-based
    mix construction. It delegates to :func:`make_traversal` with ``policy=type_name`` (no auto
    affine/permutation switching unless *type_name* is chosen accordingly).
    """
    if type_name not in _TRAVERSAL_POLICY_REGISTRY:
        raise ValueError(
            f"unknown traversal policy: {type_name!r}. Registered: {sorted(registered_traversal_policy_types())}"
        )
    return make_traversal(
        int(length),
        seed=seed,
        policy=type_name,
        shuffle=shuffle,
        buffer_size=buffer_size,
    )


# ============================================================
# Leaf node
# ============================================================

class LeafNode:
    """Leaf node: maps a local index range [0, length) to global indices via offset."""

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
        self.traversal = traversal or PermutationTraversal(length=self.length, shuffle=shuffle, seed=seed)

    def set_epoch(self, epoch: int) -> None:
        self.traversal.set_epoch(epoch)

    def plan(self, budget: int) -> Iterator[int]:
        if budget < 0:
            raise ValueError("budget must be >= 0")
        for _ in range(int(budget)):
            yield self.offset + self.traversal.next_index()

    def state_dict(self) -> dict[str, Any]:
        return {"type": "leaf", "length": self.length, "offset": self.offset,
                "name": self.name, "traversal": self.traversal.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.length = int(state["length"]); self.offset = int(state["offset"])
        self.name = state["name"]
        self.traversal = traversal_from_state(state["traversal"])


# ============================================================
# Weight resolver
# ============================================================

def resolve_weights(weights: WeightSpec, epoch: int, expected_len: int) -> np.ndarray:
    """Normalize static or epoch-dependent weights to sum to 1; length must match *expected_len*."""
    w = weights(epoch) if callable(weights) else weights
    if len(w) != expected_len:
        raise ValueError(f"weights length must match children: got {len(w)}, expected {expected_len}")
    return normalize_weights(w)


# ============================================================
# MixNode
# ============================================================

class MixNode:
    """Internal node: distributes a budget across children using a MixStrategy.

    Args:
        children: Child SamplerNodes.
        weights: Sampling weights (static list or epoch-callable).
        strategy: How to allocate and order samples. Defaults to
            LargestRemainderStrategy (exact proportional, smooth interleave).
        seed: RNG seed; advanced per epoch.
        name: Optional label for debugging / state_dict identification.
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
        if not children:
            raise ValueError("children must not be empty")
        self.children = list(children)
        self.weights = weights
        self.name = name or "mix"
        self._strategy: MixStrategy = strategy if strategy is not None else LargestRemainderStrategy()
        self.base_seed = seed
        self._rng = seed_to_rng(seed)
        self.epoch = 0

    @property
    def strategy(self) -> MixStrategy:
        return self._strategy

    def current_weights(self) -> np.ndarray:
        return resolve_weights(self.weights, self.epoch, len(self.children))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._rng = _epoch_rng(self.base_seed, epoch)
        for child in self.children:
            child.set_epoch(epoch)

    def plan(self, budget: int) -> Iterator[int]:
        budget = int(budget)
        if budget < 0:
            raise ValueError("budget must be >= 0")
        if budget == 0:
            return
        yield from self._strategy.plan(self.children, self.current_weights(), budget, self._rng)

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": "mix",
            "name": self.name,
            "epoch": self.epoch,
            "base_seed": self.base_seed,
            "rng_state": self._rng.bit_generator.state,
            "strategy": self._strategy.state_dict(),
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
            self._rng = _epoch_rng(self.base_seed, self.epoch)

        # Restore strategy from the unified state format.
        if "strategy" in state:
            self._strategy = mix_strategy_from_state(state["strategy"])
        elif "quota_allocator" in state and "quota_scheduler" in state:
            self._strategy = _strategy_from_legacy_composite(state)
        # else: keep existing strategy (no-op for partial updates).

        child_states = state["children"]
        if len(child_states) != len(self.children):
            raise ValueError(f"child state length mismatch: {len(child_states)} vs {len(self.children)}")
        for child, child_state in zip(self.children, child_states):
            child.load_state_dict(child_state)


# ============================================================
# Public sampler
# ============================================================

class ComposableSampler(Sampler[int]):
    """General-purpose hierarchical sampler for map-style datasets.

    Simple usage::

        sampler = ComposableSampler.from_dataset_lengths(
            dataset_lengths=[len(ds1), len(ds2), len(ds3)],
            weights=[0.5, 0.3, 0.2],
            num_samples=sum(map(len, [ds1, ds2, ds3])),
            seed=42,
        )

    Advanced usage (custom mix strategy and traversal)::

        sampler = ComposableSampler.from_dataset_lengths(
            dataset_lengths=[1000, 2_000_000],
            weights=[0.3, 0.7],
            strategy=WeightedRandomStrategy(temperature=0.5),
            traversal_factory=lambda length: make_traversal(length, seed=0),
            seed=42,
        )
    """

    def __init__(self, *, root: SamplerNode, num_samples: int) -> None:
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
        shuffle: bool = True,
        strategy: str | MixStrategy | None = None,
        temperature: Optional[float] = None,
        traversal: str | TraversalPolicy | Callable[[int], TraversalPolicy] | None = None,
        buffer_size: Optional[int] = None,
        # Backward-compat aliases (prefer the unified names above).
        shuffle_within_leaf: Optional[bool] = None,
        traversal_factory: Optional[TraversalFactory] = None,
    ) -> "ComposableSampler":
        """Build a flat sampler over multiple datasets.

        Delegates to :meth:`Compose.flat`; see :class:`Compose` for parameter semantics.

        Backward-compatible aliases ``shuffle_within_leaf`` (-> *shuffle*) and
        ``traversal_factory`` (-> *traversal*) are accepted but deprecated.
        """
        if shuffle_within_leaf is not None:
            shuffle = shuffle_within_leaf
        if traversal_factory is not None:
            if traversal is not None:
                raise ValueError("pass traversal or traversal_factory, not both")
            traversal = traversal_factory
        return Compose.flat(
            dataset_lengths,
            weights=weights,
            num_samples=num_samples,
            seed=seed,
            shuffle=shuffle,
            strategy=strategy,
            temperature=temperature,
            traversal=traversal,
            buffer_size=buffer_size,
        )

    def __iter__(self) -> Iterator[int]:
        yield from self.root.plan(self.num_samples)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.root.set_epoch(epoch)

    def state_dict(self) -> dict[str, Any]:
        return {"epoch": self.epoch, "num_samples": self.num_samples, "root": self.root.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.epoch = int(state["epoch"])
        self.num_samples = int(state["num_samples"])
        self.root.load_state_dict(state["root"])


# ============================================================
# EpisodeSpec (for :meth:`Compose.episodes` / :meth:`Compose.hierarchy`)
# ============================================================

@dataclass
class EpisodeSpec:
    length: int
    offset: int
    weight: float = 1.0
    name: Optional[str] = None


def _uniform_weights(children: Sequence[SamplerNode]) -> Sequence[float]:
    return [1.0] * len(children)


# ============================================================
# Unified resolution helpers (single source of truth for strategy/traversal)
# ============================================================

def _resolve_strategy(
    strategy: str | MixStrategy | None = None,
    *,
    temperature: Optional[float] = None,
) -> MixStrategy:
    """Resolve a strategy specification to a concrete :class:`MixStrategy`.

    Accepts:
      - ``None``  -> :class:`LargestRemainderStrategy` (default)
      - ``str``   -> registry lookup via :func:`mix_strategy_from_name`
      - instance  -> used as-is

    ``temperature`` only applies when *strategy* is ``"weighted_random"`` (string).
    """
    if strategy is None:
        return LargestRemainderStrategy()
    if isinstance(strategy, str):
        return mix_strategy_from_name(strategy, temperature=temperature)
    if temperature is not None:
        raise ValueError("temperature is only valid when strategy is a string name")
    return strategy


def _resolve_traversal(
    traversal: str | TraversalPolicy | Callable[[int], TraversalPolicy] | None,
    length: int,
    *,
    seed: Optional[int] = None,
    shuffle: bool = True,
    buffer_size: Optional[int] = None,
) -> TraversalPolicy:
    """Resolve a traversal specification to a concrete :class:`TraversalPolicy`.

    Accepts:
      - ``None``     -> auto-select via :func:`make_traversal`
      - ``str``      -> registry lookup via :func:`traversal_from_registry`
      - instance (has ``next_index``) -> used as-is
      - callable     -> treated as ``(length) -> TraversalPolicy`` factory
    """
    if traversal is None:
        return make_traversal(int(length), seed=seed, shuffle=shuffle, buffer_size=buffer_size)
    if isinstance(traversal, str):
        return traversal_from_registry(traversal, int(length), seed=seed, shuffle=shuffle, buffer_size=buffer_size)
    if hasattr(traversal, 'next_index'):
        return traversal
    if callable(traversal):
        return traversal(int(length))
    raise TypeError(
        f"traversal must be str, TraversalPolicy, callable, or None; got {type(traversal).__name__}"
    )


class Compose:
    """Unified construction interface for composable hierarchical samplers.

    Every method accepts **two polymorphic parameters** for maximum convenience:

    ``strategy`` (mix layer):
      - ``None``  -> :class:`LargestRemainderStrategy` (default)
      - ``str``   -> registry key (``"largest_remainder"``, ``"weighted_random"``, …)
      - :class:`MixStrategy` instance -> used as-is

    ``traversal`` (leaf layer):
      - ``None``     -> auto-select via :func:`make_traversal` (length-based)
      - ``str``      -> registry key (``"permutation"``, ``"affine_permutation"``, …)
      - ``callable`` -> factory ``(length) -> TraversalPolicy``
      - :class:`TraversalPolicy` instance -> used as-is

    Core methods::

        Compose.leaf(100, 0, traversal="permutation", seed=1)
        Compose.mix(leaves, [0.7, 0.3], strategy="round_robin")
        Compose.flat([1000, 500], weights=[0.7, 0.3], seed=42)
        Compose.sampler(root, num_samples=300)
        Compose.episodes([EpisodeSpec(100, 0), ...], strategy="weighted_random", temperature=0.5)
        Compose.hierarchy({...}, source_weights={...}, strategy="largest_remainder", seed=42)
    """

    @staticmethod
    def leaf(
        length: int,
        offset: int,
        *,
        traversal: str | TraversalPolicy | Callable[[int], TraversalPolicy] | None = None,
        shuffle: bool = True,
        seed: Optional[int] = None,
        name: Optional[str] = None,
        buffer_size: Optional[int] = None,
    ) -> LeafNode:
        """Build a :class:`LeafNode` with the given traversal.

        Args:
            length: Number of samples in this leaf.
            offset: Global index offset for this leaf.
            traversal: How to walk the leaf's index space (see class docstring).
            shuffle: Passed to :func:`make_traversal` when *traversal* is ``None`` or to
                :func:`traversal_from_registry` when *traversal* is a ``str``.
            seed: RNG seed for the traversal.
            name: Optional label for debugging / state_dict.
            buffer_size: Required when traversal is ``"buffered_shuffle"``.
        """
        tr = _resolve_traversal(traversal, length, seed=seed, shuffle=shuffle, buffer_size=buffer_size)
        return LeafNode(length=length, offset=offset, traversal=tr, name=name)

    @staticmethod
    def mix(
        children: Sequence[SamplerNode],
        weights: WeightSpec | None = None,
        *,
        strategy: str | MixStrategy | None = None,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        name: Optional[str] = None,
    ) -> MixNode:
        """Build a :class:`MixNode` that distributes budget across *children*.

        Args:
            children: Child :class:`SamplerNode` instances.
            weights: Sampling weights (static or epoch-callable). Uniform if ``None``.
            strategy: How to allocate and interleave samples (see class docstring).
            temperature: Only for ``strategy="weighted_random"``; sharpens (< 1) or flattens (> 1).
            seed: RNG seed.
            name: Optional label.
        """
        resolved = _resolve_strategy(strategy, temperature=temperature)
        return MixNode(
            children=children,
            weights=weights if weights is not None else _uniform_weights(children),
            strategy=resolved,
            seed=seed,
            name=name or "mix",
        )

    @staticmethod
    def sampler(root: SamplerNode, num_samples: int) -> "ComposableSampler":
        """Wrap a root node as a ``torch.utils.data.Sampler``."""
        return ComposableSampler(root=root, num_samples=num_samples)

    @staticmethod
    def flat(
        dataset_lengths: Sequence[int],
        *,
        weights: Optional[Sequence[float]] = None,
        num_samples: Optional[int] = None,
        seed: Optional[int] = None,
        shuffle: bool = True,
        strategy: str | MixStrategy | None = None,
        temperature: Optional[float] = None,
        traversal: str | TraversalPolicy | Callable[[int], TraversalPolicy] | None = None,
        buffer_size: Optional[int] = None,
    ) -> "ComposableSampler":
        """Build a flat sampler over concatenated datasets.

        Args:
            dataset_lengths: Number of samples in each dataset.
            weights: Sampling weights; uniform if ``None``.
            num_samples: Total samples per epoch; ``sum(dataset_lengths)`` if ``None``.
            seed: Base RNG seed. Per-leaf seeds are derived automatically.
            shuffle: Passed to traversal when *traversal* is ``None`` or a string.
            strategy: Mix strategy (see class docstring).
            temperature: Only for ``strategy="weighted_random"``.
            traversal: Leaf traversal (see class docstring).
            buffer_size: For ``traversal="buffered_shuffle"``.

        Example::

            sampler = Compose.flat(
                [len(ds1), len(ds2)],
                weights=[0.7, 0.3],
                strategy="weighted_random",
                temperature=0.5,
                seed=42,
            )
        """
        lengths = np.asarray(list(dataset_lengths), dtype=np.int64)
        if lengths.ndim != 1 or lengths.size == 0 or np.any(lengths <= 0):
            raise ValueError("dataset_lengths must be a non-empty 1D array of positive integers")
        offsets = np.cumsum(np.concatenate(([0], lengths[:-1])), dtype=np.int64)
        leaves = [
            Compose.leaf(
                int(lengths[i]),
                int(offsets[i]),
                traversal=traversal,
                shuffle=shuffle,
                seed=None if seed is None else seed + i * 10007,
                name=f"dataset_{i}",
                buffer_size=buffer_size,
            )
            for i in range(len(lengths))
        ]
        root = Compose.mix(
            leaves,
            weights=weights if weights is not None else [1.0] * len(leaves),
            strategy=strategy,
            temperature=temperature,
            seed=seed,
            name="root",
        )
        ns = int(lengths.sum()) if num_samples is None else num_samples
        return ComposableSampler(root=root, num_samples=ns)

    @staticmethod
    def episodes(
        episodes: Sequence[EpisodeSpec],
        *,
        strategy: str | MixStrategy | None = None,
        temperature: Optional[float] = None,
        traversal: str | TraversalPolicy | Callable[[int], TraversalPolicy] | None = None,
        shuffle: bool = True,
        buffer_size: Optional[int] = None,
        seed: Optional[int] = None,
        name: str = "episode_mix",
    ) -> MixNode:
        """Build a :class:`MixNode` over a list of episode leaves.

        Args:
            episodes: Episode specs (length, offset, weight, name).
            strategy: Mix strategy for combining episodes (see class docstring).
            temperature: Only for ``strategy="weighted_random"``.
            traversal: Leaf traversal for each episode (see class docstring).
            shuffle: Passed to traversal.
            buffer_size: For ``traversal="buffered_shuffle"``.
            seed: Base RNG seed; per-episode seeds are derived.
            name: Label for the :class:`MixNode`.

        Example::

            node = Compose.episodes(
                [EpisodeSpec(100, 0, name="ep0"), EpisodeSpec(50, 100)],
                strategy="weighted_random",
                temperature=0.5,
                seed=42,
            )
        """
        if not episodes:
            raise ValueError("episodes must not be empty")
        resolved = _resolve_strategy(strategy, temperature=temperature)
        leaves: list[LeafNode] = []
        ws: list[float] = []
        for i, ep in enumerate(episodes):
            if ep.length <= 0:
                raise ValueError(f"episode {i} has non-positive length: {ep.length}")
            tr = _resolve_traversal(
                traversal, ep.length,
                seed=None if seed is None else seed + i * 4099,
                shuffle=shuffle,
                buffer_size=buffer_size,
            )
            leaves.append(LeafNode(
                length=ep.length, offset=ep.offset,
                traversal=tr, name=ep.name or f"episode_{i}",
            ))
            ws.append(float(ep.weight))
        return MixNode(children=leaves, weights=ws, strategy=resolved, seed=seed, name=name)

    @staticmethod
    def hierarchy(
        source_to_task_to_episodes: dict[str, dict[str, Sequence[EpisodeSpec]]],
        *,
        source_weights: Optional[dict[str, float]] = None,
        task_weights: Optional[dict[str, dict[str, float]]] = None,
        strategy: str | MixStrategy | None = None,
        source_strategy: str | MixStrategy | None = None,
        task_strategy: str | MixStrategy | None = None,
        episode_strategy: str | MixStrategy | None = None,
        temperature: Optional[float] = None,
        episode_temperature: Optional[float] = None,
        traversal: str | TraversalPolicy | Callable[[int], TraversalPolicy] | None = None,
        shuffle: bool = True,
        buffer_size: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> MixNode:
        """Build a 3-level source -> task -> episode hierarchy.

        Per-level strategy resolution: ``source_strategy`` / ``task_strategy`` /
        ``episode_strategy`` override the shared ``strategy`` default. Similarly,
        ``episode_temperature`` overrides ``temperature`` for the episode level.

        Args:
            source_to_task_to_episodes: Nested ``{source: {task: [EpisodeSpec, ...]}}``.
            source_weights: Per-source sampling weights (uniform if ``None``).
            task_weights: ``{source: {task: weight}}`` (uniform if ``None``).
            strategy: Shared default mix strategy for all levels.
            source_strategy / task_strategy / episode_strategy: Per-level overrides.
            temperature: Shared temperature (for ``"weighted_random"``).
            episode_temperature: Override temperature for episode level.
            traversal: Leaf traversal for all episodes.
            shuffle / buffer_size: Passed to traversal.
            seed: Base RNG seed; per-node seeds are derived.

        Example::

            root = Compose.hierarchy(
                {"base":   {"open_door": [EpisodeSpec(120, 0)],
                            "close":     [EpisodeSpec(80, 120)]},
                 "dagger": {"open_door": [EpisodeSpec(40, 200, weight=2.0)]}},
                source_weights={"base": 0.8, "dagger": 0.2},
                episode_strategy="weighted_random",
                episode_temperature=0.5,
                seed=123,
            )
        """
        src_resolved = _resolve_strategy(
            source_strategy if source_strategy is not None else strategy,
            temperature=temperature,
        )
        tsk_resolved = _resolve_strategy(
            task_strategy if task_strategy is not None else strategy,
            temperature=temperature,
        )
        ep_resolved = _resolve_strategy(
            episode_strategy if episode_strategy is not None else strategy,
            temperature=episode_temperature if episode_temperature is not None else temperature,
        )

        source_nodes: list[MixNode] = []
        source_ws: list[float] = []

        for s_idx, source_name in enumerate(source_to_task_to_episodes):
            task_map = source_to_task_to_episodes[source_name]
            task_nodes: list[MixNode] = []
            task_ws: list[float] = []

            for t_idx, task_name in enumerate(task_map):
                task_seed = None if seed is None else seed + s_idx * 100003 + t_idx * 1009
                task_node = Compose.episodes(
                    task_map[task_name],
                    strategy=clone_mix_strategy(ep_resolved),
                    traversal=traversal,
                    shuffle=shuffle,
                    buffer_size=buffer_size,
                    seed=task_seed,
                    name=f"{source_name}/{task_name}",
                )
                task_nodes.append(task_node)
                task_ws.append(
                    float((task_weights or {}).get(source_name, {}).get(task_name, 1.0))
                )

            source_node = MixNode(
                children=task_nodes,
                weights=task_ws,
                strategy=clone_mix_strategy(tsk_resolved),
                seed=None if seed is None else seed + s_idx * 700001,
                name=source_name,
            )
            source_nodes.append(source_node)
            source_ws.append(float((source_weights or {}).get(source_name, 1.0)))

        return MixNode(
            children=source_nodes,
            weights=source_ws,
            strategy=clone_mix_strategy(src_resolved),
            seed=seed,
            name="root",
        )


# ============================================================
# Examples / self-tests (unified Compose API)
# ============================================================

def _example_traversal_exactness() -> None:
    for traversal in [
        AffinePermutationTraversal(length=17, seed=0),
        PermutationTraversal(length=17, shuffle=True, seed=0),
        BufferedShuffleTraversal(length=17, buffer_size=5, seed=0),
    ]:
        traversal.set_epoch(0)
        vals = [traversal.next_index() for _ in range(17)]
        assert sorted(vals) == list(range(17)), type(traversal).__name__


def _example_flat() -> None:
    """Compose.flat — flat mix over 3 datasets; reproducible."""
    sampler = Compose.flat(
        [1000, 500, 250],
        weights=[0.5, 0.3, 0.2],
        num_samples=1750,
        seed=42,
    )
    sampler.set_epoch(0)
    idxs = list(sampler)
    assert len(idxs) == 1750
    sampler.set_epoch(0)
    assert list(sampler) == idxs


def _example_flat_with_strategy() -> None:
    """Compose.flat with string strategy + temperature."""
    sampler = Compose.flat(
        [100, 200],
        weights=[0.4, 0.6],
        strategy="weighted_random",
        temperature=0.8,
        seed=7,
    )
    sampler.set_epoch(0)
    assert len(list(sampler)) == 300


def _example_strategy_comparison() -> None:
    """Compose.mix with different strategy strings; quotas [15, 9, 6]."""
    lengths = (300, 300, 300)
    weights = [0.5, 0.3, 0.2]
    budget = 30

    def make_leaves() -> list[LeafNode]:
        offsets = [0, 300, 600]
        return [Compose.leaf(n, o, traversal="permutation", shuffle=False, seed=i)
                for i, (n, o) in enumerate(zip(lengths, offsets, strict=True))]

    def child_counts(idxs: list[int]) -> list[int]:
        bounds = [300, 600, 900]
        counts = [0, 0, 0]
        for x in idxs:
            counts[next(j for j, b in enumerate(bounds) if x < b)] += 1
        return counts

    for name in ["largest_remainder", "round_robin", "shuffle_sequential", "stratified_random"]:
        leaves = make_leaves()
        node = Compose.mix(leaves, weights, strategy=name, seed=0)
        node.set_epoch(0)
        counts = child_counts(list(node.plan(budget)))
        assert counts == [15, 9, 6], f"{name}: unexpected counts {counts}"

    leaves = make_leaves()
    node = Compose.mix(leaves, weights, strategy="weighted_random", seed=0)
    node.set_epoch(0)
    counts = child_counts(list(node.plan(budget)))
    assert sum(counts) == budget
    assert counts[0] >= counts[1] >= counts[2], f"weighted_random unexpected: {counts}"


def _example_nested_compose() -> None:
    """Nested: Compose.mix with string strategies at different levels."""
    leaves = [
        Compose.leaf(100, 0, seed=1, name="l0"),
        Compose.leaf(100, 100, seed=2, name="l1"),
        Compose.leaf(100, 200, seed=3, name="l2"),
    ]

    inner = Compose.mix(
        leaves[:2], weights=[0.7, 0.3],
        strategy="weighted_random", temperature=0.5, seed=10, name="inner",
    )
    root = Compose.mix(
        [inner, leaves[2]], weights=[0.8, 0.2],
        strategy="round_robin", seed=20, name="root",
    )
    root.set_epoch(0)

    idxs = list(root.plan(200))
    assert len(idxs) == 200
    l2_count = sum(1 for x in idxs if 200 <= x < 300)
    assert 25 <= l2_count <= 55, f"unexpected l2 count: {l2_count}"


def _example_compose_leaf_traversal() -> None:
    """Compose.leaf with string traversal types."""
    leaves = [
        Compose.leaf(30, 0, seed=11, name="leaf_a"),
        Compose.leaf(30, 30, traversal="permutation", shuffle=False, seed=12, name="leaf_b"),
        Compose.leaf(40, 60, traversal="buffered_shuffle", buffer_size=7, seed=13, name="leaf_c"),
    ]
    for leaf in leaves:
        leaf.set_epoch(0)
    n = Compose.mix(leaves, weights=[0.4, 0.3, 0.3], strategy="round_robin", seed=3)
    n.set_epoch(0)
    assert len(list(n.plan(30))) == 30

    n2 = Compose.mix(leaves[:2], strategy="stratified_random", seed=4)
    n2.set_epoch(0)
    assert len(list(n2.plan(18))) == 18


def _example_make_traversal_auto() -> None:
    small = make_traversal(100, seed=0)
    large = make_traversal(1_000_000, seed=0)
    assert isinstance(small, PermutationTraversal)
    assert isinstance(large, AffinePermutationTraversal)
    for t in [small, large]:
        t.set_epoch(0)
        seen = {t.next_index() for _ in range(t.length)}
        assert seen == set(range(t.length))


def _example_episodes() -> None:
    """Compose.episodes — episode group with unified params."""
    ep = Compose.episodes(
        [EpisodeSpec(4, 0), EpisodeSpec(4, 4)],
        strategy="round_robin",
        traversal="permutation",
        shuffle=False,
        seed=3,
    )
    ep.set_epoch(0)
    assert len(list(ep.plan(8))) == 8


def _example_hierarchy() -> None:
    """Compose.hierarchy — 3-level hierarchy with per-level strategy overrides."""
    root = Compose.hierarchy(
        {
            "base":   {"open_door": [EpisodeSpec(120, 0), EpisodeSpec(90, 120)],
                       "close_door": [EpisodeSpec(80, 210)]},
            "dagger": {"open_door": [EpisodeSpec(40, 290, weight=2.0)],
                       "insert":    [EpisodeSpec(70, 330)]},
        },
        source_weights={"base": 0.8, "dagger": 0.2},
        task_weights={
            "base":   {"open_door": 1.0, "close_door": 1.0},
            "dagger": {"open_door": 2.0, "insert": 1.0},
        },
        episode_strategy="weighted_random",
        episode_temperature=0.5,
        seed=123,
    )
    sampler = Compose.sampler(root, 400)
    sampler.set_epoch(3)
    idxs = list(sampler)
    assert len(idxs) == 400
    assert all(0 <= x < 400 for x in idxs)


def _example_hierarchy_full() -> None:
    """Compose.hierarchy — full named episodes."""
    root = Compose.hierarchy(
        {
            "base": {
                "open_door": [
                    EpisodeSpec(length=120, offset=0, weight=1.0, name="base/open_door/ep0"),
                    EpisodeSpec(length=90, offset=120, weight=1.0, name="base/open_door/ep1"),
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
    sampler = Compose.sampler(root, 400)
    sampler.set_epoch(3)
    idxs = list(sampler)
    assert len(idxs) == 400
    assert all(0 <= x < 400 for x in idxs)


def _example_compose_sampler_flat() -> None:
    """Compose.leaf + Compose.mix + Compose.sampler pipeline."""
    leaves = [
        Compose.leaf(10, 0, seed=1, name="a"),
        Compose.leaf(10, 10, seed=2, name="b"),
    ]
    root = Compose.mix(leaves, weights=[0.6, 0.4], seed=0)
    sampler = Compose.sampler(root, 20)
    sampler.set_epoch(0)
    assert len(list(sampler)) == 20
    flat = Compose.flat([5, 5], num_samples=10, seed=99)
    assert len(flat) == 10


def _example_state_dict_roundtrip() -> None:
    sampler = Compose.flat(
        [100, 200], weights=[0.4, 0.6],
        strategy="weighted_random", temperature=0.8, seed=7,
    )
    sampler.set_epoch(2)
    list(sampler)
    sd = sampler.state_dict()
    sampler2 = Compose.flat(
        [100, 200], weights=[0.4, 0.6],
        strategy="weighted_random", temperature=0.8, seed=7,
    )
    sampler2.load_state_dict(sd)
    sampler.set_epoch(3); sampler2.set_epoch(3)
    assert list(sampler) == list(sampler2)


if __name__ == "__main__":
    _example_traversal_exactness()
    _example_flat()
    _example_flat_with_strategy()
    _example_strategy_comparison()
    _example_nested_compose()
    _example_compose_leaf_traversal()
    _example_make_traversal_auto()
    _example_episodes()
    _example_hierarchy()
    _example_hierarchy_full()
    _example_compose_sampler_flat()
    _example_state_dict_roundtrip()
    print("All examples passed.")
