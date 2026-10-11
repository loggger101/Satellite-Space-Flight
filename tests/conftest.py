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


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    """Interrupt a test that runs longer than ``TIMEOUT`` and report it as a
    failure carrying the main thread's stack at that moment."""
    if TIMEOUT <= 0:
        return (yield)
    main = threading.main_thread().ident
    stuck: list[str] = []

    def expire():
        stuck.append("".join(traceback.format_stack(sys._current_frames().get(main))))
        _thread.interrupt_main()

    timer = threading.Timer(TIMEOUT, expire)
    timer.daemon = True
    timer.start()
    try:
        return (yield)
    except KeyboardInterrupt:
        if stuck:                  # ours, not Ctrl+C
            pytest.fail(f"ran over {TIMEOUT:g} s; it was here:\n{stuck[0]}", pytrace=False)
        raise
    finally:
        timer.cancel()
