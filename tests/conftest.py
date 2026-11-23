"""Suite-wide guards: a test that hangs fails with the stack showing where it
was stuck (and the run goes on), instead of freezing the run or CI."""

import _thread
import os
import sys
import threading
import traceback

import pytest

# seconds one test may take; SATFLIGHT_TEST_TIMEOUT=0 switches the watchdog off
TIMEOUT = float(os.environ.get("SATFLIGHT_TEST_TIMEOUT", "300"))


def pytest_configure(config):
    config.addinivalue_line("markers", "time_limit(seconds): this test's own watchdog limit, "
                                       "in place of SATFLIGHT_TEST_TIMEOUT")


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    """Interrupt a test that runs longer than ``TIMEOUT`` (or its own
    ``time_limit`` mark) and report it as a failure carrying the main thread's
    stack at that moment."""
    if TIMEOUT <= 0:
        return (yield)
    mark = item.get_closest_marker("time_limit")
    limit = float(mark.args[0]) if mark else TIMEOUT
    main = threading.main_thread().ident
    stuck: list[str] = []

    def expire():
        stuck.append("".join(traceback.format_stack(sys._current_frames().get(main))))
        _thread.interrupt_main()

    timer = threading.Timer(limit, expire)
    timer.daemon = True
    timer.start()
    try:
        return (yield)
    except KeyboardInterrupt:
        if stuck:                  # ours, not Ctrl+C
            pytest.fail(f"ran over {limit:g} s; it was here:\n{stuck[0]}", pytrace=False)
        raise
    finally:
        timer.cancel()
