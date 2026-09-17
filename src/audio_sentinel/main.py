from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from audio_sentinel.config import load_settings
from audio_sentinel.demo import WEB_DIRECTORY, router as demo_router
from audio_sentinel.health import check_project_health

app = FastAPI(title="Audio Sentinel")
app.include_router(demo_router)
app.mount("/demo/assets", StaticFiles(directory=WEB_DIRECTORY), name="demo-assets")


@app.get("/", include_in_schema=False)
def home() -> RedirectResponse:
    return RedirectResponse("/demo")


@app.get("/health")
def health() -> dict[str, object]:
    return check_project_health().model_dump()


@app.get("/project/status")
def project_status() -> dict[str, object]:
    settings = load_settings()
    return {
        "project_root": str(settings.paths.root),
        "datasets_dir": str(settings.paths.raw_data),
        "current_focus": "Pinned YAMNet v1 produces verified raw patch scores over prepared waveform windows.",
        "next_step": "Implement event aggregation across overlapping windows (B3.2).",
    }
