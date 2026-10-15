import random
from datetime import datetime

import gevent
import pytest


class DummyEvent:
    creation_time = None
    type = "event"
    id = 1

    def __init__(self, creation_time, event_id=1):
        self.creation_time = creation_time
        self.id = event_id


class DummyEvents(list):
    def count(self):
        return len(self)


class DummyExperiment:
    replay_path = "/replay"
    _events = DummyEvents()
    replayed = []
    finished = False
    started = False

    def log(self, *args, **kw):
        pass

    def replay_started(self):
        return self.started

    def events_for_replay(self, *args, **kwargs):
        return self._events

    def replay_event(self, event):
        self.replayed.append(
            {"orig_time": event.creation_time, "replay_time": datetime.now()}
        )

    def replay_finish(self):
        self.finished = True


@pytest.mark.slow
class TestReplayBackend:
    allowed_jitter = 0.05

    def launch_replay_task(self, exp):
        from dallinger.experiment_server.replay import ReplayBackend

        task = ReplayBackend(exp)
        gevent.spawn(task)
        return task

    def wait_to_finish(self, exp):
        loop_count = 0
        while exp.finished is False:
            gevent.sleep(2)
            loop_count += 1
            if loop_count > 10:
                raise RuntimeError("Replay did not finish in reasonable time")

    def test_replay_waits_to_start(self):
        exp = DummyExperiment()
        assert exp.started is False
        assert exp.finished is False

        self.launch_replay_task(exp)

        # Wait for task to `run` in background
        gevent.sleep(2)
        assert exp.finished is False

        # Start the task
        exp.started = True
        gevent.sleep(2)
        assert exp.finished is True

    def test_replay_event_timing(self):
        exp = DummyExperiment()

        # 10 events ~ 1 second apart with some random jitter
        exp._events = DummyEvents(
            [
                DummyEvent(datetime(2010, 1, 1, 0, 0, t, int(random.random() * 1000)))
                for t in range(10)
            ]
        )

        exp.started = True
        self.launch_replay_task(exp)

        self.wait_to_finish(exp)

        replayed = exp.replayed
        assert len(replayed) == 10

        base_offset = (
            replayed[0]["replay_time"] - replayed[0]["orig_time"]
        ).total_seconds()

        for rp in replayed:
            time_diff = (rp["replay_time"] - rp["orig_time"]).total_seconds()
            assert abs(time_diff - base_offset) <= self.allowed_jitter

    def _backend(self, exp):
        from dallinger.experiment_server.replay import ReplayBackend

        return ReplayBackend(exp)

    def test_empty_events_finishes_and_converges(self):
        exp = DummyExperiment()
        exp._events = DummyEvents()
        exp.started = True
        backend = self._backend(exp)

        backend()

        assert exp.finished is True
        assert backend.generation.finished is True
        assert backend.generation.confirmed_count == 0
        assert backend.generation.closed is True
        from dallinger.replay import get_generation

        assert get_generation(backend.generation.generation_id) is None

    def test_successful_run_confirms_every_event(self):
        exp = DummyExperiment()
        exp.replayed = []
        exp._events = DummyEvents(
            [
                DummyEvent(datetime(2010, 1, 1, 0, 0, 0), event_id=i)
                for i in range(5)
            ]
        )
        exp.started = True
        backend = self._backend(exp)

        backend()

        assert exp.finished is True
        assert backend.generation.confirmed_count == 5
        assert backend.generation.last_position.index == 4
        assert backend.generation.last_position.event_id == 4
        assert backend.generation.failed is False
        assert len(exp.replayed) == 5

    def test_event_hook_failure_preserves_position_and_skips_finish(self):
        exp = DummyExperiment()
        exp.replayed = []
        events = DummyEvents(
            [
                DummyEvent(datetime(2010, 1, 1, 0, 0, 0), event_id=i)
                for i in range(5)
            ]
        )
        exp._events = events
        exp.started = True
        backend = self._backend(exp)

        def replay_event(event):
            if event.id == 2:
                raise ValueError("hook failed on event 2")
            exp.replayed.append(event.id)

        exp.replay_event = replay_event

        with pytest.raises(ValueError, match="hook failed on event 2"):
            backend()

        generation = backend.generation
        assert exp.finished is False
        assert generation.failed is True
        assert generation.failure_stage == "event"
        assert generation.failure_position.index == 2
        assert generation.failure_position.event_id == 2
        # Only events before the failure were confirmed.
        assert generation.confirmed_count == 2
        assert exp.replayed == [0, 1]
        # The failed generation converges instead of lingering.
        assert generation.closed is True
        from dallinger.replay import active_generations

        assert generation not in active_generations()

    def test_finish_hook_failure_is_recorded(self):
        exp = DummyExperiment()
        exp._events = DummyEvents()
        exp.started = True

        def failing_finish():
            raise RuntimeError("finish hook failed")

        exp.replay_finish = failing_finish
        backend = self._backend(exp)

        with pytest.raises(RuntimeError, match="finish hook failed"):
            backend()

        assert backend.generation.failed is True
        assert backend.generation.failure_stage == "finish"
        assert backend.generation.finished is False
        assert backend.generation.closed is True

    def test_each_backend_has_distinct_generation(self):
        exp_a = DummyExperiment()
        exp_a._events = DummyEvents()
        exp_a.started = True
        exp_b = DummyExperiment()
        exp_b._events = DummyEvents()
        exp_b.started = True
        backend_a = self._backend(exp_a)
        backend_b = self._backend(exp_b)

        assert (
            backend_a.generation.generation_id
            != backend_b.generation.generation_id
        )

        backend_a()
        backend_b()

        assert backend_a.generation.finished is True
        assert backend_b.generation.finished is True

