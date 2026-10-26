import faulthandler
import logging
import os
import threading
import time


class TrainingWatchdog:
    """Terminate a training process that stops reporting progress."""

    def __init__(self, timeout_minutes=30, rank=0):
        self.timeout_seconds = float(timeout_minutes) * 60.0
        self.rank = rank
        self._last_heartbeat = time.monotonic()
        self._last_phase = "startup"
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        if self.timeout_seconds <= 0:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="training-watchdog",
            daemon=True,
        )
        self._thread.start()

    def beat(self, phase):
        with self._lock:
            self._last_heartbeat = time.monotonic()
            self._last_phase = phase

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _run(self):
        while not self._stop_event.wait(timeout=min(self.timeout_seconds, 60.0)):
            with self._lock:
                elapsed = time.monotonic() - self._last_heartbeat
                phase = self._last_phase

            if elapsed < self.timeout_seconds:
                continue

            logging.critical(
                "Training watchdog timed out on rank %s after %.1f minutes; "
                "last progress: %s",
                self.rank,
                elapsed / 60.0,
                phase,
            )
            faulthandler.dump_traceback(all_threads=True)
            os._exit(124)


def get_watchdog_timeout_minutes(default=30):
    value = os.environ.get("SWINFACE_WATCHDOG_TIMEOUT_MINUTES")
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logging.warning(
            "Ignoring invalid SWINFACE_WATCHDOG_TIMEOUT_MINUTES=%r; using %.1f",
            value,
            default,
        )
        return default