"""Mixing runtime core shared by composable dataloaders and samplers.

This module centralizes the building blocks used to mix multiple sources:

- Weight normalization and quota allocation helpers.
- Online source schedulers used by composable dataloaders.
- Quota schedulers used by composable samplers.
- Registry and state-restore helpers shared across both layers.
- Runtime stream-consumption glue such as ``schedule_and_consume``.

The scope intentionally includes both allocation and execution helpers because
they are tightly coupled to the scheduler protocols defined here.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
import heapq
from typing import Any, Callable, Optional, Protocol, TypeVar

import numpy as np


T = TypeVar("T")


def extract_type_from_state(obj: Any, fallback: str) -> str:
    getter = getattr(obj, "state_dict", None)
    if callable(getter):
        state = getter()
        if isinstance(state, dict) and "type" in state:
            return str(state["type"])
    return fallback


def normalize_weights(weights: Sequence[float]) -> np.ndarray:
    """Normalize non-negative weights to sum to 1.

    Raises:
        ValueError: If weights are empty, multi-dimensional, contain negative
            values, or all weights sum to 0.
    """
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1 or w.size == 0:
        raise ValueError("weights must be a non-empty 1D sequence")
    if np.any(w < 0):
        raise ValueError(f"weights must be non-negative, got {weights}")
    s = float(w.sum())
    if s <= 0:
        raise ValueError("weights must not all be zero")
    return w / s


def largest_remainder_allocate(total: int, weights: Sequence[float]) -> np.ndarray:
    """Allocate integer quotas summing exactly to *total*.

    Uses the Hamilton / Largest-Remainder method with deterministic tie-breaks
    by lower index.
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
    order = np.lexsort((np.arange(len(frac)), -frac))
    base[order[:remain]] += 1
    return base


def _active_weighted_choice(
    num_loaders: int,
    base_weights: np.ndarray,
    active: np.ndarray,
    buf: np.ndarray,
    rng: np.random.Generator,
) -> int:
    np.multiply(base_weights, active, out=buf)
    total = buf.sum()
    if total == 0:
        return -1
    buf /= total
    return int(rng.choice(num_loaders, p=buf))


class _TypedStatelessStateMixin:
    """Common state serialization for stateless registry-backed helpers."""

    TYPE_NAME: str

    def state_dict(self) -> dict[str, Any]:
        return {"type": self.TYPE_NAME}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        _ = state


class QuotaAllocator(Protocol):
    """Allocate integer quotas for weighted mixtures."""

    def allocate(
        self,
        weights: np.ndarray,
        budget: int,
        rng: np.random.Generator,
    ) -> np.ndarray: ...

    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state: dict[str, Any]) -> None: ...


class LargestRemainderAllocator(_TypedStatelessStateMixin):
    """Stateless quota allocator based on Hamilton / Largest-Remainder."""

    TYPE_NAME = "largest_remainder"

    def allocate(
        self,
        weights: np.ndarray,
        budget: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        _ = rng
        return largest_remainder_allocate(int(budget), weights)


class MultinomialAllocator:
    """Stochastic quota allocator using multinomial draws."""

    def __init__(self, *, temperature: float = 1.0) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature = float(temperature)

    def allocate(
        self,
        weights: np.ndarray,
        budget: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        probs = np.asarray(weights, dtype=np.float64)
        if self.temperature != 1.0:
            probs = probs ** (1.0 / self.temperature)
            total = probs.sum()
            if total <= 0:
                raise ValueError("scaled weights must not all be zero")
            probs = probs / total
        return rng.multinomial(int(budget), probs)

    def state_dict(self) -> dict[str, Any]:
        return {"type": "multinomial", "temperature": self.temperature}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.temperature = float(state.get("temperature", 1.0))


class IndexDrivenQuotaScheduler:
    """Base class providing `schedule` for index-driven quota planners."""

    def iter_indices(
        self,
        quotas: np.ndarray,
        rng: np.random.Generator,
    ) -> Iterator[int]:
        raise NotImplementedError

    def schedule(
        self,
        streams: Sequence[Iterator[T]],
        quotas: np.ndarray,
        rng: np.random.Generator,
    ) -> Iterator[T]:
        for i in self.iter_indices(quotas, rng):
            yield next(streams[i])


class OnlineSourceScheduler(Protocol):
    """Choose the next source index in an online, step-by-step loop."""

    def choose_index(
        self,
        weights: np.ndarray,
        active: np.ndarray,
        rng: np.random.Generator,
        buf: np.ndarray,
    ) -> int: ...

    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state: dict[str, Any]) -> None: ...


def schedule_and_consume(
    streams: list[Iterator[T]],
    weights: np.ndarray | Sequence[float] | Sequence[int],
    rng: np.random.Generator,
    scheduler: OnlineSourceScheduler,
    *,
    max_steps: Optional[int] = None,
    decrement_after_selection: bool = False,
    restart_stream: Optional[Callable[[int], Iterator[T]]] = None,
    stop_on_exhausted: bool = False,
) -> Iterator[tuple[int, T]]:
    """Online source selection plus stream consumption for DataLoader-style mixing.

    Unlike quota schedulers that precompute a full source plan, this helper asks
    the online scheduler for the next source at each step, then immediately
    consumes one element from that stream.
    """
    current = np.asarray(weights, dtype=np.float64).copy()
    exhausted = np.zeros(len(current), dtype=bool)
    buf = np.empty(len(current), dtype=np.float64)
    steps = 0

    while True:
        if max_steps is not None and steps >= max_steps:
            return

        if decrement_after_selection:
            active = (current > 0) & ~exhausted
        else:
            active = ~exhausted
        if not active.any():
            return

        idx = scheduler.choose_index(current, active, rng, buf)
        if idx < 0:
            return

        if decrement_after_selection:
            current[idx] = max(0.0, current[idx] - 1.0)

        try:
            item = next(streams[idx])
        except StopIteration:
            if stop_on_exhausted:
                return
            if restart_stream is not None:
                streams[idx] = restart_stream(idx)
                try:
                    item = next(streams[idx])
                except StopIteration:
                    exhausted[idx] = True
                    continue
            else:
                exhausted[idx] = True
                continue

        steps += 1
        yield idx, item


class ProportionalRandomScheduler(IndexDrivenQuotaScheduler, _TypedStatelessStateMixin):
    """Dual-role scheduler for proportional random source selection.

    - As an OnlineSourceScheduler, it picks the next active source.
    - As a QuotaIndexScheduler, it turns remaining quotas into an index plan.
    """

    TYPE_NAME = "proportional_random"

    def choose_index(
        self,
        weights: np.ndarray,
        active: np.ndarray,
        rng: np.random.Generator,
        buf: np.ndarray,
    ) -> int:
        return _active_weighted_choice(len(weights), weights, active, buf, rng)

    def iter_indices(
        self,
        quotas: np.ndarray,
        rng: np.random.Generator,
    ) -> Iterator[int]:
        remaining = np.asarray(quotas, dtype=np.float64).copy()
        active = remaining > 0
        buf = np.empty(len(remaining), dtype=np.float64)
        while active.any():
            idx = self.choose_index(remaining, active, rng, buf)
            if idx < 0:
                return
            remaining[idx] -= 1
            if remaining[idx] <= 0:
                active[idx] = False
            yield int(idx)


class ChunkedRoundRobinScheduler:
    """Stay on the same active source for a chunk of steps before switching."""

    def __init__(self, chunk_size: int = 1, start_idx: int = 0, step: int = 0) -> None:
        self.chunk_size = max(1, int(chunk_size))
        self._next_idx = max(0, int(start_idx))
        self.step = max(0, int(step))

    def choose_index(
        self,
        weights: np.ndarray,
        active: np.ndarray,
        rng: np.random.Generator,
        buf: np.ndarray,
    ) -> int:
        _ = (weights, rng, buf)
        active_idx = np.flatnonzero(active)
        if active_idx.size == 0:
            return -1
        pos = int(np.searchsorted(active_idx, self._next_idx, side="left"))
        if pos >= active_idx.size:
            pos = 0
        idx = int(active_idx[pos])
        self.step += 1
        if self.step % self.chunk_size == 0:
            next_pos = (pos + 1) % active_idx.size
            self._next_idx = int(active_idx[next_pos])
        else:
            self._next_idx = idx
        return idx

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": "chunked_round_robin",
            "chunk_size": self.chunk_size,
            "next_idx": self._next_idx,
            "step": self.step,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.chunk_size = max(1, int(state.get("chunk_size", state.get("k", self.chunk_size))))
        self._next_idx = max(0, int(state.get("next_idx", self._next_idx)))
        self.step = max(0, int(state.get("step", self.step)))


class LeastServedScheduler:
    """Deficit-style scheduler that compensates short-horizon under-service."""

    def __init__(self, deficits: Sequence[float] | None = None) -> None:
        self._deficits = np.asarray(deficits, dtype=np.float64).copy() if deficits is not None else np.zeros(0, dtype=np.float64)

    def choose_index(
        self,
        weights: np.ndarray,
        active: np.ndarray,
        rng: np.random.Generator,
        buf: np.ndarray,
    ) -> int:
        _ = rng
        if self._deficits.shape != weights.shape:
            self._deficits = np.zeros(len(weights), dtype=np.float64)
        np.copyto(buf, weights)
        buf[~active] = 0.0
        total = float(buf.sum())
        if total <= 0:
            return -1
        buf /= total
        self._deficits += buf
        self._deficits[~active] = -np.inf
        idx = int(np.argmax(self._deficits))
        if not active[idx]:
            return -1
        self._deficits[idx] -= 1.0
        self._deficits[~active] = 0.0
        return idx

    def state_dict(self) -> dict[str, Any]:
        return {"type": "least_served", "deficits": self._deficits.tolist()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        deficits = state.get("deficits", [])
        self._deficits = np.asarray(deficits, dtype=np.float64).copy()


_QUOTA_ALLOCATOR_REGISTRY: dict[str, Callable[[dict[str, Any]], QuotaAllocator]] = {
    "largest_remainder": lambda s: LargestRemainderAllocator(),
    "multinomial": lambda s: MultinomialAllocator(temperature=float(s.get("temperature", 1.0))),
}


_ONLINE_SOURCE_SCHEDULER_REGISTRY: dict[str, Callable[[dict[str, Any]], OnlineSourceScheduler]] = {
    "proportional_random": lambda s: ProportionalRandomScheduler(),
    "round_robin_online": lambda s: ChunkedRoundRobinScheduler(chunk_size=1, start_idx=int(s.get("next_idx", 0))),
    "chunked_round_robin": lambda s: ChunkedRoundRobinScheduler(
        chunk_size=int(s.get("chunk_size", s.get("k", 1))),
        start_idx=int(s.get("next_idx", 0)),
        step=int(s.get("step", 0)),
    ),
    "least_served": lambda s: LeastServedScheduler(deficits=s.get("deficits")),
}


def register_quota_allocator(
    type_name: str,
    factory: Callable[[dict[str, Any]], QuotaAllocator],
) -> None:
    _QUOTA_ALLOCATOR_REGISTRY[type_name] = factory


def registered_quota_allocator_types() -> frozenset[str]:
    return frozenset(_QUOTA_ALLOCATOR_REGISTRY.keys())


COMPOSABLE_QUOTA_ALLOCATOR_TYPES: frozenset[str] = frozenset(_QUOTA_ALLOCATOR_REGISTRY.keys())


def quota_allocator_from_registry(
    type_name: str,
    state: Optional[dict[str, Any]] = None,
    *,
    temperature: Optional[float] = None,
) -> QuotaAllocator:
    extra_state = {"temperature": float(temperature)} if temperature is not None else None
    return _build_from_registry(
        _QUOTA_ALLOCATOR_REGISTRY,
        type_name,
        state,
        kind_label="quota allocator",
        registered_keys=registered_quota_allocator_types,
        extra_state=extra_state,
    )


COMPOSABLE_ONLINE_SCHEDULER_TYPES: frozenset[str] = frozenset(_ONLINE_SOURCE_SCHEDULER_REGISTRY.keys())


def register_online_scheduler(
    name: str,
    factory: Callable[[dict[str, Any]], OnlineSourceScheduler],
) -> None:
    _ONLINE_SOURCE_SCHEDULER_REGISTRY[name] = factory


def registered_online_scheduler_types() -> frozenset[str]:
    return frozenset(_ONLINE_SOURCE_SCHEDULER_REGISTRY.keys())


def online_scheduler_from_name(
    name: str,
    state: Optional[dict[str, Any]] = None,
) -> OnlineSourceScheduler:
    return _build_from_registry(
        _ONLINE_SOURCE_SCHEDULER_REGISTRY,
        name,
        state,
        kind_label="online scheduler",
        registered_keys=registered_online_scheduler_types,
    )


def smooth_interleave_indices(quotas: np.ndarray | Sequence[int]) -> Iterator[int]:
    """Yield source indices using smooth quota interleaving."""
    q = np.asarray(quotas, dtype=np.int64)
    total = int(q.sum())
    if total == 0:
        return
    consumed = np.zeros(len(q), dtype=np.int64)
    heap: list[tuple[float, int]] = []
    for i, qi in enumerate(q):
        if qi > 0:
            heapq.heappush(heap, (0.0, i))
    for _ in range(total):
        if not heap:
            return
        _, i = heapq.heappop(heap)
        yield i
        consumed[i] += 1
        if consumed[i] < q[i]:
            heapq.heappush(heap, (consumed[i] / float(q[i]), i))


class QuotaScheduler(Protocol):
    """Consume child streams according to already computed quotas."""

    def schedule(
        self,
        streams: Sequence[Iterator[T]],
        quotas: np.ndarray,
        rng: np.random.Generator,
    ) -> Iterator[T]: ...

    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state: dict[str, Any]) -> None: ...


class SmoothInterleaveSchedule(IndexDrivenQuotaScheduler, _TypedStatelessStateMixin):
    """QuotaScheduler adapter via smooth_interleave_indices."""

    TYPE_NAME = "smooth_interleave"

    def iter_indices(
        self,
        quotas: np.ndarray,
        rng: np.random.Generator,
    ) -> Iterator[int]:
        _ = rng
        yield from smooth_interleave_indices(np.asarray(quotas, dtype=np.int64))


class RandomShuffleSchedule(IndexDrivenQuotaScheduler, _TypedStatelessStateMixin):
    TYPE_NAME = "random_shuffle"

    def iter_indices(
        self,
        quotas: np.ndarray,
        rng: np.random.Generator,
    ) -> Iterator[int]:
        q = np.asarray(quotas, dtype=np.int64)
        order = np.repeat(np.arange(len(q), dtype=np.int64), q)
        rng.shuffle(order)
        for i in order:
            yield int(i)


class RoundRobinSchedule(IndexDrivenQuotaScheduler, _TypedStatelessStateMixin):
    TYPE_NAME = "round_robin"

    def iter_indices(
        self,
        quotas: np.ndarray,
        rng: np.random.Generator,
    ) -> Iterator[int]:
        _ = rng
        remaining = np.asarray(quotas, dtype=np.int64).copy()
        active = [i for i, q in enumerate(remaining) if int(q) > 0]
        pos = 0
        for _ in range(int(remaining.sum())):
            if not active:
                break
            idx = int(active[pos])
            yield idx
            remaining[idx] -= 1
            if int(remaining[idx]) <= 0:
                active.pop(pos)
                if not active:
                    break
                if pos >= len(active):
                    pos = 0
            else:
                pos = (pos + 1) % len(active)


class ShuffleSequentialSchedule(IndexDrivenQuotaScheduler, _TypedStatelessStateMixin):
    TYPE_NAME = "shuffle_sequential"

    def iter_indices(
        self,
        quotas: np.ndarray,
        rng: np.random.Generator,
    ) -> Iterator[int]:
        q = np.asarray(quotas, dtype=np.int64)
        order = rng.permutation(len(q))
        for i in order:
            for _ in range(int(q[int(i)])):
                yield int(i)


class StratifiedRandomSchedule(IndexDrivenQuotaScheduler, _TypedStatelessStateMixin):
    TYPE_NAME = "stratified_random"

    def iter_indices(
        self,
        quotas: np.ndarray,
        rng: np.random.Generator,
    ) -> Iterator[int]:
        remaining = np.asarray(quotas, dtype=np.int64).copy()
        left = int(remaining.sum())
        while left > 0:
            active = np.where(remaining > 0)[0]
            for i in rng.permutation(active):
                ii = int(i)
                yield ii
                remaining[ii] -= 1
                left -= 1
                if left <= 0:
                    return


_QUOTA_SCHEDULER_REGISTRY: dict[str, Callable[[dict[str, Any]], QuotaScheduler]] = {
    "proportional_random": lambda s: ProportionalRandomScheduler(),
    "smooth_interleave": lambda s: SmoothInterleaveSchedule(),
    "random_shuffle": lambda s: RandomShuffleSchedule(),
    "round_robin": lambda s: RoundRobinSchedule(),
    "shuffle_sequential": lambda s: ShuffleSequentialSchedule(),
    "stratified_random": lambda s: StratifiedRandomSchedule(),
}


TRegistryItem = TypeVar("TRegistryItem")


def _build_from_registry(
    registry: dict[str, Callable[[dict[str, Any]], TRegistryItem]],
    type_name: str,
    state: Optional[dict[str, Any]],
    *,
    kind_label: str,
    registered_keys: Callable[[], frozenset[str]],
    extra_state: Optional[dict[str, Any]] = None,
) -> TRegistryItem:
    if type_name not in registry:
        raise ValueError(
            f"unknown {kind_label} type: {type_name!r}; "
            f"registered: {sorted(registered_keys())}"
        )
    merged: dict[str, Any] = {"type": type_name, **(state or {})}
    if extra_state:
        merged.update(extra_state)
    obj = registry[type_name](merged)
    loader = getattr(obj, "load_state_dict", None)
    if callable(loader):
        loader(merged)
    return obj


def register_quota_scheduler(
    type_name: str,
    factory: Callable[[dict[str, Any]], QuotaScheduler],
) -> None:
    _QUOTA_SCHEDULER_REGISTRY[type_name] = factory


def registered_quota_scheduler_types() -> frozenset[str]:
    return frozenset(_QUOTA_SCHEDULER_REGISTRY.keys())


COMPOSABLE_QUOTA_SCHEDULER_TYPES: frozenset[str] = frozenset(_QUOTA_SCHEDULER_REGISTRY.keys())


def quota_scheduler_from_registry(
    type_name: str,
    state: Optional[dict[str, Any]] = None,
) -> QuotaScheduler:
    return _build_from_registry(
        _QUOTA_SCHEDULER_REGISTRY,
        type_name,
        state,
        kind_label="quota scheduler",
        registered_keys=registered_quota_scheduler_types,
    )


def restore_quota_and_schedule(
    quota_state: dict[str, Any],
    quota_scheduler_state: dict[str, Any],
    *,
    error_prefix: str,
) -> tuple[QuotaAllocator, QuotaScheduler]:
    """Restore quota allocator and quota scheduler from state payloads."""
    quota_type = str(quota_state.get("type", ""))
    quota_scheduler_type = str(quota_scheduler_state.get("type", ""))
    try:
        quota = quota_allocator_from_registry(quota_type, quota_state)
        scheduler = quota_scheduler_from_registry(
            quota_scheduler_type,
            quota_scheduler_state,
        )
    except ValueError as exc:
        raise ValueError(
            f"{error_prefix}: qa={quota_state}, qs={quota_scheduler_state}"
        ) from exc
    return quota, scheduler