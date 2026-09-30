from audio_sentinel.config import AudioSentinelSettings
from audio_sentinel.health import check_project_health
from audio_sentinel.main import health, project_status


def test_health_check_reports_missing_directories(temporary_settings: AudioSentinelSettings) -> None:
    health_report = check_project_health(temporary_settings)

    assert health_report.status == "degraded"
    assert set(health_report.missing_directories) == set(temporary_settings.paths.directory_map)

    temporary_settings.ensure_directories()
    assert check_project_health(temporary_settings).status == "ok"


def test_api_health_and_project_status_are_available() -> None:
    assert health()["status"] == "ok"
    status = project_status()
    assert status["current_focus"] == (
        "The offline MVP has a repeatable, consent-gated demonstration workflow."
    )
    assert status["next_step"] == (
        "Define authorized-device enrollment and streaming interfaces (A10.1)."
    )
