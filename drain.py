"""Thread-safe admission and drain state for bounded graceful termination."""

from __future__ import annotations

import math
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Condition


class ServiceDraining(RuntimeError):
    """Raised when work arrives after the service has closed admission."""


@dataclass(frozen=True)
class DrainSnapshot:
    accepting: bool
    in_flight: int

    @property
    def drained(self) -> bool:
        return not self.accepting and self.in_flight == 0


class DrainController:
    """Atomically close admission and wait for admitted work to finish."""

    def __init__(self) -> None:
        self._condition = Condition()
        self._accepting = True
        self._in_flight = 0

    def snapshot(self) -> DrainSnapshot:
        with self._condition:
            return DrainSnapshot(self._accepting, self._in_flight)

    @contextmanager
    def admit(self) -> Iterator[None]:
        with self._condition:
            if not self._accepting:
                raise ServiceDraining("prediction admission is closed")
            self._in_flight += 1
        try:
            yield
        finally:
            with self._condition:
                self._in_flight -= 1
                if self._in_flight == 0:
                    self._condition.notify_all()

    def begin_draining(self) -> DrainSnapshot:
        """Close admission idempotently before readiness is reported unhealthy."""
        with self._condition:
            self._accepting = False
            return DrainSnapshot(self._accepting, self._in_flight)

    def wait_for_idle(self, timeout_seconds: float) -> DrainSnapshot:
        """Wait up to the caller's shutdown budget for admitted work to finish."""
        if isinstance(timeout_seconds, bool) or not math.isfinite(timeout_seconds):
            raise ValueError("timeout_seconds must be a finite non-negative number")
        if timeout_seconds < 0:
            raise ValueError("timeout_seconds must be a finite non-negative number")

        deadline = time.monotonic() + timeout_seconds
        with self._condition:
            while self._in_flight:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(timeout=remaining)
            return DrainSnapshot(self._accepting, self._in_flight)
