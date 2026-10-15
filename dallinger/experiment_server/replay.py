import logging
import time

import gevent

from dallinger.replay import ReplayGeneration
from dallinger.utils import get_base_url

logger = logging.getLogger(__name__)


class ReplayBackend:
    """Replay backend which replays `events` from a completed experiment run.

    This is started during launch and delegates `event` selection and
    publication to the experiment class. `Events` are any objects with a
    creation_time attribute.

    Each run owns a lightweight :class:`~dallinger.replay.ReplayGeneration`
    that tracks resumable progress. Events are confirmed one at a time, so a
    failure preserves the diagnostic position; the main session is rolled
    back and ``replay_finish`` is only called when every event and user hook
    succeeds.
    """

    def __init__(self, experiment):
        self.experiment = experiment
        self.generation = ReplayGeneration.lightweight(experiment)

    def __call__(self):
        gevent.sleep(0.200)

        try:
            logger.info(
                "Replay ready: {}{}".format(
                    get_base_url(), self.experiment.replay_path
                )
            )
        except RuntimeError:
            # config not loaded, we may be in unit tests
            pass

        try:
            while not self.experiment.replay_started():
                gevent.sleep(0.01)

            self.experiment.log("Looping through replayable data", key="replay")
            timestamp = self.timestamp
            events = self.experiment.events_for_replay()

            if not events.count():
                self._finish()
                return

            first_timestamp = timestamp(events[0].creation_time)
            self.experiment.log(
                "Found {} messages to replay starting from {}".format(
                    events.count(), events[0].creation_time
                ),
                key="replay",
            )
            start = time.time()
            for index, event in enumerate(events):
                event_offset = timestamp(event.creation_time) - first_timestamp
                cur_offset = time.time() - start
                if event_offset >= cur_offset:
                    if (event_offset - cur_offset) > 1:
                        self.experiment.log(
                            "Waiting {} seconds to replay {} {}".format(
                                event_offset - cur_offset, event.type, event.id
                            ),
                            key="replay",
                        )
                    gevent.sleep(event_offset - cur_offset)
                try:
                    self.experiment.replay_event(event)
                except Exception as exc:
                    # Record the diagnostic position and roll back the main
                    # session; the outer handler converges the generation and
                    # re-raises, so replay_finish is never published.
                    self.generation.record_failure(
                        "event", event=event, index=index, exc=exc
                    )
                    raise
                self.generation.confirm(event)

            self.experiment.log(
                "Replayed {} events in {} seconds (original duration {} "
                "seconds)".format(
                    events.count(),
                    time.time() - start,
                    timestamp(event.creation_time) - first_timestamp,
                ),
                key="replay",
            )
            self._finish()
        except BaseException:
            # Any failure (including an event hook) converges this
            # generation: the run stops and confirmed events are not replayed.
            self.generation.converge()
            raise

    def _finish(self):
        """Publish replay_finish only if no event or hook failed."""
        if self.generation.failed:
            return
        try:
            self.experiment.replay_finish()
        except Exception as exc:
            self.generation.record_failure("finish", exc=exc)
            raise
        self.generation.finished = True
        self.generation.converge()

    @staticmethod
    def timestamp(dt):
        """Generate a microsecond accurate timestamp from a datetime."""
        return time.mktime(dt.timetuple()) + dt.microsecond / 1e6
