from pathlib import Path

from fastapi import FastAPI

from audio_sentinel.config import Paths

app = FastAPI(title="Audio Sentinel")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/project/status")
def project_status() -> dict[str, object]:
    paths = Paths.from_root(Path(__file__).resolve().parents[2])
    return {
        "project_root": str(paths.root),
        "datasets_dir": str(paths.raw_data),
        "current_focus": "dataset selection and offline pipeline setup",
        "next_step": "Create a dataset registry and download scripts for the approved shortlist.",
    }

