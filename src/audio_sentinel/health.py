"""Small, dependency-free project health checks used by tests and the API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from audio_sentinel.config import AudioSentinelSettings, load_settings


class ProjectHealth(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: Literal["ok", "degraded"]
    project_root: str
    directories: dict[str, bool]
    missing_directories: tuple[str, ...]


def check_project_health(settings: AudioSentinelSettings | None = None) -> ProjectHealth:
    """Report whether the project directories expected by Phase 0 are present."""

    settings = settings or load_settings()
    directories = {name: path.is_dir() for name, path in settings.paths.directory_map.items()}
    missing_directories = tuple(name for name, exists in directories.items() if not exists)
    return ProjectHealth(
        status="ok" if not missing_directories else "degraded",
        project_root=str(settings.paths.root),
        directories=directories,
        missing_directories=missing_directories,
    )
