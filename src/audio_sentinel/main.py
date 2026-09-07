from fastapi import FastAPI

from audio_sentinel.config import load_settings
from audio_sentinel.health import check_project_health

app = FastAPI(title="Audio Sentinel")


@app.get("/health")
def health() -> dict[str, object]:
    return check_project_health().model_dump()


@app.get("/project/status")
def project_status() -> dict[str, object]:
    settings = load_settings()
    return {
        "project_root": str(settings.paths.root),
        "datasets_dir": str(settings.paths.raw_data),
        "current_focus": "Offline audio preparation is implemented and integration-tested.",
        "next_step": "Define Log-Mel feature contract and metadata schema (A2.1).",
    }
