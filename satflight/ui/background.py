"""Slow work off the UI thread, newest request only.

A :class:`Latest` runs one job at a time on a daemon thread. Submitting
while a job runs replaces whatever was waiting (a queue one deep), so a
form that changes ten times while a plan is flown costs one more plan, not
ten. Results come back through :meth:`Latest.poll`, called from the UI
thread, tagged with the key they were asked for so stale ones can be told
apart. Jobs must not touch live UI or simulation state: take what they need
on the UI thread before submitting.

A pure-Python job holds the GIL, and by default the interpreter hands it
over only every 5 ms: each time the UI thread waits on a C call, the worker
takes 5 ms, and a frame that waits often stretches to 0.2 s or more. While
a job runs the switch interval is cut to 0.2 ms, which keeps frames near
their usual time; the job then takes about as long as the frames leave it.
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable


class Latest:
    """One worker thread; only the most recent waiting job is run."""

    SWITCH_INTERVAL = 0.0002        # s, while a job runs (Python's default is 0.005)

    def __init__(self):
        self._lock = threading.Lock()
        self._waiting = None             # (key, job) not started yet
        self._done = None                # (key, result, error) not collected yet
        self._thread = None

    def submit(self, key, job: Callable[[], object]):
        """Run ``job()`` in the background, dropping any job still waiting."""
        with self._lock:
            self._waiting = (key, job)
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, daemon=True,
                                                name="satflight-background")
                self._thread.start()

    def _run(self):
        default = sys.getswitchinterval()
        sys.setswitchinterval(min(default, self.SWITCH_INTERVAL))
        while True:
            with self._lock:
                if self._waiting is None:
                    self._thread = None
                    sys.setswitchinterval(default)
                    return
                key, job = self._waiting
                self._waiting = None
            try:
                out = (key, job(), None)
            except Exception as exc:     # reported to the UI, never raised in the thread
                out = (key, None, exc)
            with self._lock:
                self._done = out

    def poll(self):
        """``(key, result, error)`` of the last finished job, once, or None."""
        with self._lock:
            done, self._done = self._done, None
        return done

    @property
    def busy(self) -> bool:
        """True while a job runs or waits."""
        with self._lock:
            return self._thread is not None
