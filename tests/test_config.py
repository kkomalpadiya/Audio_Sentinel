from pathlib import Path

import pytest
from pydantic import ValidationError

from audio_sentinel.config import AudioSentinelSettings, AudioSettings, find_project_root, load_settings


def test_settings_create_only_managed_directories(temporary_settings: AudioSentinelSettings) -> None:
    assert not temporary_settings.paths.raw_data.exists()

    temporary_settings.ensure_directories()

    assert all(path.is_dir() for path in temporary_settings.paths.directory_map.values())


def test_resolve_within_rejects_path_escape(temporary_settings: AudioSentinelSettings) -> None:
    temporary_settings.ensure_directories()

    assert temporary_settings.paths.resolve_within(temporary_settings.paths.raw_data, "fsd50k/clip.wav") == (
        temporary_settings.paths.raw_data / "fsd50k" / "clip.wav"
    ).resolve()

    with pytest.raises(ValueError, match="stay within"):
        temporary_settings.paths.resolve_within(temporary_settings.paths.raw_data, "../outside.wav")


def test_audio_settings_require_ordered_unique_windows() -> None:
    with pytest.raises(ValidationError, match="unique and ordered"):
        AudioSettings(window_seconds=(5.0, 1.0, 1.0))


def test_load_settings_discovers_the_project_root(project_root: Path) -> None:
    assert find_project_root(project_root / "src" / "audio_sentinel") == project_root
    assert load_settings(project_root).paths.root == project_root.resolve()
