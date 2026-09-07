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
        "current_focus": "YAMNet baseline selected with a verified, versioned acoustic label mapping.",
        "next_step": "Implement isolated acoustic-model loader with version metadata (B3.1).",
    }
