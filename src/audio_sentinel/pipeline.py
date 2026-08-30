from dataclasses import dataclass


@dataclass(frozen=True)
class WindowPlan:
    short_seconds: float = 1.0
    medium_seconds: float = 5.0
    long_seconds: float = 10.0


DEFAULT_WINDOW_PLAN = WindowPlan()

