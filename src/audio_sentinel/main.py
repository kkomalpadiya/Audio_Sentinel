from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from audio_sentinel.api import router as evaluation_router
from audio_sentinel.config import load_settings
from audio_sentinel.demo import WEB_DIRECTORY, router as demo_router
from audio_sentinel.health import check_project_health

app = FastAPI(title="Audio Sentinel")
app.include_router(evaluation_router)
app.include_router(demo_router)
app.mount("/demo/assets", StaticFiles(directory=WEB_DIRECTORY), name="demo-assets")


@app.exception_handler(RequestValidationError)
def request_validation_error(
    _request: Request, _error: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"code": "invalid_request", "error": "Request body failed validation."},
    )


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
        "current_focus": "Offline evaluation is available through local CLI and API boundaries.",
        "next_step": "Add CLI and report integration tests (B8.3).",
    }
