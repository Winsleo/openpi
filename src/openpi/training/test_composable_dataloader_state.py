from __future__ import annotations

import itertools

import numpy as np

from openpi.training.composable_dataloader import DynamicScheduleDataLoader
from openpi.training.composable_dataloader import LONGEST
from openpi.training.composable_dataloader import Compose
from openpi.training.composable_dataloader import ProportionalMixDataLoader
from openpi.training.composable_dataloader import RandomMixDataLoader
from openpi.training.mixing import OnlineSourceScheduler
from openpi.training.mixing import QuotaAllocator
from openpi.training.mixing import online_scheduler_from_name
from openpi.training.mixing import quota_allocator_from_registry
from openpi.training.mixing import register_online_scheduler
from openpi.training.mixing import register_quota_allocator
from openpi.training.mixing import registered_online_scheduler_types
from openpi.training.mixing import registered_quota_allocator_types


class _SeqLoader:
    def __init__(self, size: int, label: str) -> None:
        self._size = int(size)
        self._label = label

    def __iter__(self):
        for i in range(self._size):
            yield f"{self._label}-{i}"

    def __len__(self) -> int:
        return self._size


def _take(loader, n: int) -> list[str]:
    return list(itertools.islice(iter(loader), n))


def test_random_mix_state_roundtrip_reproduces_sequence() -> None:
    a = _SeqLoader(20, "a")
    b = _SeqLoader(20, "b")

    loader1 = RandomMixDataLoader([a, b], weights=[0.7, 0.3], stop_strategy=LONGEST, seed=123)
    state = loader1.state_dict()

    loader2 = RandomMixDataLoader([a, b], weights=[0.5, 0.5], stop_strategy=LONGEST, seed=999)
    loader2.load_state_dict(state)

    assert _take(loader1, 15) == _take(loader2, 15)


def test_proportional_state_roundtrip_reproduces_epoch() -> None:
    a = _SeqLoader(16, "a")
    b = _SeqLoader(12, "b")

    loader1 = ProportionalMixDataLoader([a, b], ratios=[3, 1], stop_strategy=LONGEST, seed=321)
    state = loader1.state_dict()

    loader2 = ProportionalMixDataLoader([a, b], ratios=[1, 1], stop_strategy=LONGEST, seed=222)
    loader2.load_state_dict(state)

    assert list(loader1) == list(loader2)
    assert loader2.batches_per_loader.tolist() == loader1.batches_per_loader.tolist()


def test_random_mix_longest_drops_exhausted_loader_without_restart() -> None:
    class FirstActiveSchedule(OnlineSourceScheduler):
        def choose_index(self, weights, active, rng, buf) -> int:
            _ = (weights, rng, buf)
            active_idx = np.where(active)[0]
            return int(active_idx[0]) if active_idx.size > 0 else -1

        def state_dict(self) -> dict:
            return {"type": "first_active"}

        def load_state_dict(self, state: dict) -> None:
            _ = state

    a = _SeqLoader(2, "a")
    b = _SeqLoader(4, "b")
    loader = RandomMixDataLoader([a, b], weights=[0.5, 0.5], stop_strategy=LONGEST, online_scheduler=FirstActiveSchedule())

    assert list(loader) == ["a-0", "a-1", "b-0", "b-1", "b-2", "b-3"]


def test_proportional_longest_restarts_exhausted_loader_to_fill_quota() -> None:
    class FirstActiveSchedule(OnlineSourceScheduler):
        def choose_index(self, weights, active, rng, buf) -> int:
            _ = (weights, rng, buf)
            active_idx = np.where(active)[0]
            return int(active_idx[0]) if active_idx.size > 0 else -1

        def state_dict(self) -> dict:
            return {"type": "first_active"}

        def load_state_dict(self, state: dict) -> None:
            _ = state

    class FrontLoadedQuota(QuotaAllocator):
        def allocate(self, weights, budget: int, rng) -> np.ndarray:
            _ = (weights, budget, rng)
            return np.asarray([3, 1], dtype=np.int64)

        def state_dict(self) -> dict:
            return {"type": "front_loaded"}

        def load_state_dict(self, state: dict) -> None:
            _ = state

    a = _SeqLoader(1, "a")
    b = _SeqLoader(3, "b")
    loader = ProportionalMixDataLoader(
        [a, b],
        ratios=[1, 1],
        stop_strategy=LONGEST,
        quota_allocator=FrontLoadedQuota(),
        online_scheduler=FirstActiveSchedule(),
    )

    assert list(loader) == ["a-0", "a-0", "a-0", "b-0"]


def test_dynamic_schedule_state_roundtrip_preserves_tracking_state() -> None:
    a = _SeqLoader(12, "a")
    b = _SeqLoader(12, "b")
    loader1 = DynamicScheduleDataLoader([a, b], initial_weights=[0.5, 0.5], seed=7)

    it = iter(loader1)
    for _ in range(4):
        try:
            next(it)
        except StopIteration:
            break
        idx = int(loader1.last_loader_idx)
        perf = 0.2 if idx == 0 else 0.8
        loader1.update_weights(idx, perf)

    state = loader1.state_dict()

    loader2 = DynamicScheduleDataLoader([a, b], initial_weights=[0.9, 0.1], seed=99)
    loader2.load_state_dict(state)

    np.testing.assert_allclose(loader2.weights, loader1.weights, atol=1e-12)
    assert loader2.batches_from_loader == loader1.batches_from_loader
    assert dict(loader2.loader_performance) == dict(loader1.loader_performance)


def test_compose_accepts_strategy_name_strings() -> None:
    a = _SeqLoader(10, "a")
    b = _SeqLoader(10, "b")

    rnd = Compose.random(a, b, weights=[0.6, 0.4], online_scheduler="proportional_random", seed=1)
    prop = Compose.proportional(
        a,
        b,
        ratios=[3, 1],
        quota_allocator="largest_remainder",
        online_scheduler="proportional_random",
        seed=2,
    )

    assert isinstance(rnd, RandomMixDataLoader)
    assert isinstance(prop, ProportionalMixDataLoader)
    assert len(_take(rnd, 5)) == 5
    assert int(prop.batches_per_loader.sum()) == len(prop)


def test_builtin_online_scheduler_types_include_deterministic_options() -> None:
    assert {"proportional_random", "round_robin_online", "chunked_round_robin", "least_served"} <= registered_online_scheduler_types()


def test_custom_online_scheduler_registry_roundtrip() -> None:
    class FirstActiveSchedule(OnlineSourceScheduler):
        def choose_index(self, weights, active, rng, buf) -> int:
            _ = (weights, rng, buf)
            active_idx = np.where(active)[0]
            return int(active_idx[0]) if active_idx.size > 0 else -1

        def state_dict(self) -> dict:
            return {"type": "first_active"}

        def load_state_dict(self, state: dict) -> None:
            _ = state

    register_online_scheduler("first_active", lambda s: FirstActiveSchedule())
    assert "first_active" in registered_online_scheduler_types()

    a = _SeqLoader(8, "a")
    b = _SeqLoader(8, "b")
    loader = RandomMixDataLoader([a, b], weights=[0.5, 0.5], online_scheduler="first_active", seed=11)
    out = _take(loader, 6)
    assert all(x.startswith("a-") for x in out)

    st = loader.state_dict()
    loader2 = RandomMixDataLoader([a, b], weights=[0.5, 0.5], seed=12)
    loader2.load_state_dict(st)
    out2 = _take(loader2, 6)
    assert all(x.startswith("a-") for x in out2)


def test_online_scheduler_from_name_applies_state_payload() -> None:
    policy = online_scheduler_from_name("chunked_round_robin", {"chunk_size": 2, "step": 3, "next_idx": 1})
    assert policy.state_dict()["chunk_size"] == 2
    assert policy.state_dict()["step"] == 3


def test_round_robin_online_produces_strict_rotation() -> None:
    a = _SeqLoader(4, "a")
    b = _SeqLoader(4, "b")
    c = _SeqLoader(4, "c")

    loader = RandomMixDataLoader(
        [a, b, c],
        weights=[0.2, 0.3, 0.5],
        stop_strategy=LONGEST,
        online_scheduler="round_robin_online",
        seed=17,
    )

    assert _take(loader, 6) == ["a-0", "b-0", "c-0", "a-1", "b-1", "c-1"]


def test_chunked_round_robin_reduces_source_switching() -> None:
    a = _SeqLoader(5, "a")
    b = _SeqLoader(5, "b")

    loader = RandomMixDataLoader(
        [a, b],
        weights=[0.5, 0.5],
        stop_strategy=LONGEST,
        online_scheduler="chunked_round_robin",
        online_scheduler_state={"chunk_size": 2},
        seed=19,
    )

    assert _take(loader, 6) == ["a-0", "a-1", "b-0", "b-1", "a-2", "a-3"]


def test_least_served_prioritizes_short_window_fairness() -> None:
    a = _SeqLoader(6, "a")
    b = _SeqLoader(6, "b")

    loader = RandomMixDataLoader(
        [a, b],
        weights=[0.75, 0.25],
        stop_strategy=LONGEST,
        online_scheduler="least_served",
        seed=23,
    )

    assert _take(loader, 8) == ["a-0", "a-1", "b-0", "a-2", "a-3", "a-4", "b-1", "a-5"]


def test_compose_random_accepts_online_scheduler_state() -> None:
    class FirstNSchedule(OnlineSourceScheduler):
        def __init__(self, first_n: int = 1) -> None:
            self.first_n = max(1, int(first_n))
            self.step = 0

        def choose_index(self, weights, active, rng, buf) -> int:
            _ = (weights, rng, buf)
            active_idx = np.where(active)[0]
            if active_idx.size == 0:
                return -1
            if self.step < self.first_n and active_idx.size > 0:
                idx = int(active_idx[0])
            else:
                idx = int(active_idx[-1])
            self.step += 1
            return idx

        def state_dict(self) -> dict:
            return {"type": "first_n", "first_n": self.first_n, "step": self.step}

        def load_state_dict(self, state: dict) -> None:
            self.first_n = max(1, int(state.get("first_n", self.first_n)))
            self.step = int(state.get("step", self.step))

    register_online_scheduler("first_n", lambda s: FirstNSchedule(first_n=int(s.get("first_n", 1))))

    a = _SeqLoader(10, "a")
    b = _SeqLoader(10, "b")
    loader = Compose.random(
        a,
        b,
        online_scheduler="first_n",
        online_scheduler_state={"first_n": 3},
        seed=5,
    )
    out = _take(loader, 6)
    assert out[:3] == ["a-0", "a-1", "a-2"]


def test_registered_online_scheduler_types_tracks_new_registrations() -> None:
    register_online_scheduler("first_active_new", lambda s: type("_Sched", (), {
        "choose_index": lambda self, weights, active, rng, buf: int(np.where(active)[0][0]) if np.where(active)[0].size > 0 else -1,
        "state_dict": lambda self: {"type": "first_active_new"},
        "load_state_dict": lambda self, state: None,
    })())

    assert "first_active_new" in registered_online_scheduler_types()


def test_quota_allocator_from_registry_applies_state_payload() -> None:
    class FrontLoadedQuota(QuotaAllocator):
        def __init__(self, front: int = 1) -> None:
            self.front = max(1, int(front))

        def allocate(self, weights, budget: int, rng) -> np.ndarray:
            _ = (weights, rng)
            out = np.zeros(2, dtype=np.int64)
            out[0] = min(int(budget), self.front)
            out[1] = int(budget) - out[0]
            return out

        def state_dict(self) -> dict:
            return {"type": "front_loaded", "front": self.front}

        def load_state_dict(self, state: dict) -> None:
            self.front = max(1, int(state.get("front", self.front)))

    register_quota_allocator("front_loaded", lambda s: FrontLoadedQuota(front=int(s.get("front", 1))))
    assert "front_loaded" in registered_quota_allocator_types()

    qa = quota_allocator_from_registry("front_loaded", {"front": 3})
    assert qa.state_dict()["front"] == 3


def test_compose_proportional_accepts_quota_state() -> None:
    class FrontLoadedQuota2(QuotaAllocator):
        def __init__(self, front: int = 1) -> None:
            self.front = max(1, int(front))

        def allocate(self, weights, budget: int, rng) -> np.ndarray:
            _ = (weights, rng)
            out = np.zeros(2, dtype=np.int64)
            out[0] = min(int(budget), self.front)
            out[1] = int(budget) - out[0]
            return out

        def state_dict(self) -> dict:
            return {"type": "front_loaded2", "front": self.front}

        def load_state_dict(self, state: dict) -> None:
            self.front = max(1, int(state.get("front", self.front)))

    register_quota_allocator("front_loaded2", lambda s: FrontLoadedQuota2(front=int(s.get("front", 1))))

    a = _SeqLoader(10, "a")
    b = _SeqLoader(10, "b")
    loader = Compose.proportional(
        a,
        b,
        ratios=[1, 1],
        quota_allocator="front_loaded2",
        quota_state={"front": 2},
        online_scheduler="proportional_random",
        seed=13,
    )
    assert loader.batches_per_loader.tolist() == [2, len(loader) - 2]
