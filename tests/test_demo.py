"""Exercise the presentation API against real saved and reloaded features."""

import base64
from importlib import import_module
from io import BytesIO

from fastapi.testclient import TestClient
import numpy as np
import pytest
import soundfile as sf

from audio_sentinel import demo, feature_persistence
from audio_sentinel.main import app


@pytest.fixture
def client():
    with TestClient(app) as session:
        yield session


def test_demo_page_and_offline_assets(client):
    response = client.get("/")
    assert response.status_code == 200 and response.url.path == "/demo"
    assert "Log-Mel spectrogram" in response.text
    for asset, content_type in [("style.css", "text/css"), ("app.js", "javascript")]:
        response = client.get("/demo/assets/" + asset)
        assert response.status_code == 200
        assert content_type in response.headers["content-type"]
    assert client.get("/health").json()["status"] == "ok"


@pytest.mark.parametrize("sample", ["tone", "sweep", "pulses"])
def test_demo_returns_actual_saved_features_and_playable_audio(client, monkeypatch, sample):
    # Load the service before spying on the demo's explicit reloads. Its internal
    # verification must retain its original dependency even in isolated test runs.
    import_module("audio_sentinel.feature_pipeline")
    original_loader = feature_persistence.load_log_mel
    saved_arrays = []

    def capture(*args, **kwargs):
        loaded = original_loader(*args, **kwargs)
        saved_arrays.append(loaded.values.copy())
        return loaded

    monkeypatch.setattr(feature_persistence, "load_log_mel", capture)
    response = client.post("/demo/api/run", json={"sample": sample})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["inference_implemented"] is False
    assert result["checks"] == {"source_unchanged": True, "repeat_reused": True,
                                "feature_bundles_verified": 5, "temporary_files_removed": True}
    assert [w["shape"] for w in result["windows"]] == [[64,97]] * 3 + [[64,497], [64,997]]
    assert len(saved_arrays) == 5
    for window, expected in zip(result["windows"], saved_arrays, strict=True):
        actual = np.array(window["values"], dtype=np.float32)
        np.testing.assert_array_equal(actual, expected)
        assert np.isfinite(actual).all()
        assert len(window["frame_times_seconds"]) == actual.shape[1]
        assert window["frame_times_seconds"][0] == 0.016
    assert [w["padding_seconds"] for w in result["windows"]] == [0, 0, .4, 3.4, 8.4]
    for key, expected_rate, expected_channels in [("source", 48000, 2), ("prepared", 16000, 1)]:
        samples, rate = sf.read(BytesIO(base64.b64decode(result[key]["audio_url"].split(",", 1)[1])),
                                always_2d=True)
        assert rate == expected_rate and samples.shape == (round(1.6 * rate), expected_channels)
    assert len(result["prepared"]["envelope"]) == 640
    assert all(low <= high for low, high in result["prepared"]["envelope"])
    if sample == "tone":
        assert len(set(saved_arrays[0].argmax(axis=0))) == 1
    if sample == "sweep":
        peaks = saved_arrays[0].argmax(axis=0)
        assert peaks[-1] > peaks[0] + 15
        assert (np.diff(peaks) >= 0).all()


@pytest.mark.parametrize("payload", [{"sample":"../../private.wav"}, {"sample":"sweep", "path":"file.wav"}])
def test_demo_rejects_arbitrary_inputs(client, payload):
    assert client.post("/demo/api/run", json=payload).status_code == 422


def test_demo_reports_busy_without_releasing_another_run(client):
    with demo._run_lock:
        response = client.post("/demo/api/run", json={"sample":"tone"})
        assert response.status_code == 409
        assert demo._run_lock.locked()


def test_demo_failure_releases_run_lock_and_hides_internal_details(client, monkeypatch):
    def fail(sample):
        raise ValueError("internal path details")

    monkeypatch.setattr(demo, "build_demo", fail)
    response = client.post("/demo/api/run", json={"sample":"tone"})
    assert response.status_code == 500
    assert "internal path" not in response.text
    assert not demo._run_lock.locked()
