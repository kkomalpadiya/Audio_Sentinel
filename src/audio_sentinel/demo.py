"""Local presentation UI backed by the real audio-to-feature service.

Only deterministic generated signals are accepted. No microphone, uploaded audio,
model inference, or project dataset is accessed by this demonstration.
"""

from pathlib import Path
from threading import Lock
from typing import Literal
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict


WEB_DIRECTORY = Path(__file__).with_name("web")
router = APIRouter(prefix="/demo", tags=["Presentation demo"])
_run_lock = Lock()
_logger = logging.getLogger(__name__)


class DemoRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sample: Literal["tone", "sweep", "pulses"] = "sweep"


@router.get("", include_in_schema=False)
def demo_page() -> FileResponse:
    return FileResponse(WEB_DIRECTORY / "index.html")


def build_demo(sample: str) -> dict[str, object]:
    """Compute, persist, reload, and serialize actual float32 feature values."""
    import base64
    from datetime import UTC, datetime
    import hashlib
    from tempfile import TemporaryDirectory
    from time import perf_counter

    import numpy as np
    import soundfile as sf

    from audio_sentinel.config import AudioSentinelSettings
    from audio_sentinel.contracts import ConsentRecord
    from audio_sentinel.feature_persistence import load_log_mel
    from audio_sentinel.feature_pipeline import AudioFeatureService
    from audio_sentinel.interfaces import InputAudio

    if sample not in {"tone", "sweep", "pulses"}:
        raise ValueError("Unknown generated sample")
    started = perf_counter()
    source_rate, duration = 48_000, 1.6
    times = np.arange(round(source_rate * duration)) / source_rate
    if sample == "tone":
        signal = 0.2 * np.sin(2 * np.pi * 1000 * times)
        description = "A steady 1,000 Hz tone produces a horizontal frequency band."
    elif sample == "sweep":
        slope = (6000 - 250) / duration
        signal = 0.2 * np.sin(2 * np.pi * (250 * times + slope * times ** 2 / 2))
        description = "A 250–6,000 Hz sweep produces a rising frequency band."
    else:
        signal = np.zeros_like(times)
        noise = np.random.default_rng(42).normal(size=len(times))
        for start in (0.15, 0.55, 0.95, 1.35):
            relative = times - start
            active = (relative >= 0) & (relative < 0.12)
            signal[active] = noise[active] * np.exp(-relative[active] / 0.025)
        signal *= 0.25 / max(float(np.max(np.abs(signal))), 1e-9)
        description = "Four generated broadband pulses produce vertical bursts of energy."

    with TemporaryDirectory(prefix="audio-sentinel-panel-") as directory:
        root = Path(directory)
        settings = AudioSentinelSettings.from_project_root(root)
        settings.paths.raw_data.mkdir(parents=True)
        source = settings.paths.raw_data / "generated.wav"
        sf.write(source, np.column_stack((signal, signal * 0.5)), source_rate, subtype="PCM_16")
        source_bytes = source.read_bytes()
        source_digest = hashlib.sha256(source_bytes).hexdigest()
        consent = ConsentRecord(
            consent_id="generated-panel-demo", status="granted",
            processing_scope="acoustic_only", device_authorized=True,
            granted_at=datetime.now(UTC),
        )
        clip = InputAudio("panel-" + sample, source, consent)
        service = AudioFeatureService(settings, "generated-panel-signal")
        first = service.prepare(clip)
        repeated = service.prepare(clip)
        prepared, rate = sf.read(first.prepared.audio.audio_path, dtype="float32")
        # Min/max envelopes preserve short peaks while keeping the waveform small.
        envelope = [[float(part.min()), float(part.max())]
                    for part in np.array_split(prepared, min(640, len(prepared))) if len(part)]
        windows = []
        for feature in first.features:
            loaded = load_log_mel(settings.paths,
                                 feature.metadata_path.relative_to(settings.paths.processed_data))
            window, recipe = loaded.metadata.source.window, loaded.metadata.settings
            windows.append({
                "id": window.window_id,
                "duration_seconds": window.window_seconds,
                "start_seconds": window.start_sample / rate,
                "end_seconds": window.end_sample / rate,
                "padding_seconds": window.padding_samples / rate,
                "shape": list(loaded.values.shape),
                "values": loaded.values.tolist(),
                "min_db": float(loaded.values.min()),
                "max_db": float(loaded.values.max()),
                "frame_times_seconds": ((np.arange(loaded.values.shape[1]) * recipe.hop_length
                                         + recipe.n_fft / 2) / rate).tolist(),
            })
        unchanged = source_digest == hashlib.sha256(source.read_bytes()).hexdigest()
        if not unchanged or not repeated.reused:
            raise RuntimeError("Pipeline integrity or repeat-reuse check failed")
        result = {
            "sample": sample, "description": description,
            "source": {"sample_rate_hz": source_rate, "channels": 2,
                       "duration_seconds": duration, "sha256": source_digest,
                       "audio_url": "data:audio/wav;base64," + base64.b64encode(source_bytes).decode("ascii")},
            "prepared": {"sample_rate_hz": rate, "channels": 1, "duration_seconds": len(prepared) / rate,
                         "rms_dbfs": float(20 * np.log10(max(float(np.sqrt(np.mean(prepared ** 2))), 1e-10))),
                         "envelope": envelope,
                         "audio_url": "data:audio/wav;base64," + base64.b64encode(
                             first.prepared.audio.audio_path.read_bytes()).decode("ascii")},
            "recipe": settings.log_mel.model_dump(mode="json"),
            "windows": windows,
            "checks": {"source_unchanged": unchanged, "repeat_reused": repeated.reused,
                       "feature_bundles_verified": len(windows)},
            "inference_implemented": False,
        }
    result["checks"]["temporary_files_removed"] = not root.exists()
    result["elapsed_seconds"] = round(perf_counter() - started, 3)
    return result


EVALUATION_RUN_RELATIVE_PATH = Path("outputs") / "a9_1_evaluation" / "evaluation-run.json"


def load_stored_evaluation_run() -> dict[str, object]:
    """Return the checked-in A9.1 held-out run after verifying its content id.

    Read-only: nothing is recomputed. The document is already privacy-minimized
    (no paths, audio, or transcript text); only display fields are returned.
    """
    from audio_sentinel.config import load_settings
    from audio_sentinel.evaluation_manifest import EvaluationRunDocument, _run_id

    path = load_settings().paths.root / EVALUATION_RUN_RELATIVE_PATH
    document = EvaluationRunDocument.model_validate_json(path.read_bytes())
    if document.run_id != _run_id(document):
        raise ValueError("Stored evaluation run failed its content-identity check")
    return {
        "run_id": document.run_id,
        "created_at": document.created_at.isoformat(),
        "manifest_id": document.manifest_id,
        "decision_status": document.decision_status,
        "positive_outcomes": [item.value for item in document.positive_outcomes],
        "metrics": document.metrics.model_dump(mode="json"),
        "cases": [
            case.model_dump(
                mode="json",
                include={
                    "case_id", "categories", "expected_positive", "status",
                    "observed_positive", "observed_outcome", "risk_score",
                    "risk_severity", "error_code",
                },
            )
            for case in document.cases
        ],
        "source_file": EVALUATION_RUN_RELATIVE_PATH.as_posix(),
        "notification_delivery": document.notification_delivery,
    }


@router.get("/api/evaluation-run")
def stored_evaluation_run() -> dict[str, object]:
    try:
        return load_stored_evaluation_run()
    except FileNotFoundError as error:
        raise HTTPException(404, "The stored evaluation run is not present in this checkout.") from error
    except Exception as error:
        _logger.exception("Stored evaluation run could not be verified")
        raise HTTPException(500, "The stored evaluation run could not be verified.") from error


@router.post("/api/run")
def run_demo(request: DemoRequest) -> dict[str, object]:
    if not _run_lock.acquire(blocking=False):
        raise HTTPException(409, "A demonstration is already running. Please try again shortly.")
    try:
        return build_demo(request.sample)
    except Exception as error:
        _logger.exception("Panel demonstration failed")
        raise HTTPException(500, "Audio processing failed. Check the server terminal, then retry.") from error
    finally:
        _run_lock.release()
