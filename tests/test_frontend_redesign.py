"""Regression guards for the redesigned dark workspace (no upload feature here)."""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil

from fastapi.testclient import TestClient
import pytest

from audio_sentinel import demo
from audio_sentinel.config import AudioSentinelSettings
from audio_sentinel.main import app

WEB = Path(demo.__file__).with_name("web")

# Every element id the original Feature explorer script depends on.
ORIGINAL_CONTROL_IDS = [
    "demo-form", "run-button", "run-state", "status", "error", "rate-value",
    "channel-value", "window-value", "band-value", "sample-description",
    "export-button", "window-select", "shape-label", "spectrogram", "plot-detail",
    "fft-value", "hop-value", "waveform", "wave-meta", "source-audio",
    "prepared-audio", "checks",
]
VIEWS = ["overview", "explorer", "pipeline", "evaluation", "methodology"]


@pytest.fixture
def client():
    with TestClient(app) as session:
        yield session


def test_page_keeps_every_original_control_once(client):
    html = client.get("/demo").text
    for element_id in ORIGINAL_CONTROL_IDS:
        assert len(re.findall(rf'id="{element_id}"', html)) == 1, element_id
    for sample in ("tone", "sweep", "pulses"):
        assert f'name="sample" value="{sample}"' in html


def test_every_view_has_a_navigation_link_and_section(client):
    html = client.get("/demo").text
    for view in VIEWS:
        assert f'id="view-{view}"' in html
        assert f'href="#{view}" data-view="{view}"' in html
    script = client.get("/demo/assets/app.js").text
    assert all(f'"{view}"' in script for view in VIEWS)


def test_explorer_script_still_calls_the_same_backend(client):
    script = client.get("/demo/assets/app.js").text
    assert 'fetch("/demo/api/run"' in script
    assert 'fetch("/demo/api/evaluation-run")' in script
    assert "upload" not in script.lower()


def test_page_makes_no_unsupported_claims(client):
    html = client.get("/demo").text.lower()
    for phrase in ("not implemented yet", "uptime", "confidence:", "users monitored"):
        assert phrase not in html


def test_stored_evaluation_run_is_served_verified_and_unchanged(client, project_root):
    response = client.get("/demo/api/evaluation-run")
    assert response.status_code == 200, response.text
    body = response.json()
    stored = json.loads((project_root / demo.EVALUATION_RUN_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert body["run_id"] == stored["run_id"]
    assert body["metrics"] == stored["metrics"]
    assert len(body["cases"]) == len(stored["cases"]) == 32
    for served, original in zip(body["cases"], stored["cases"], strict=True):
        for key, value in served.items():
            assert original[key] == value
        assert "report_id" not in served and "audit_id" not in served
    assert body["notification_delivery"] == "not_sent"
    assert str(project_root) not in response.text


def _copy_run_to(tmp_path: Path, monkeypatch, mutate=None) -> None:
    settings = AudioSentinelSettings.from_project_root(tmp_path / "root")
    target = settings.paths.root / demo.EVALUATION_RUN_RELATIVE_PATH
    target.parent.mkdir(parents=True)
    source = Path(demo.__file__).resolve().parents[2] / demo.EVALUATION_RUN_RELATIVE_PATH
    shutil.copyfile(source, target)
    if mutate:
        document = json.loads(target.read_text(encoding="utf-8"))
        mutate(document)
        target.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr("audio_sentinel.config.load_settings", lambda: settings)


def test_tampered_evaluation_run_is_rejected(client, tmp_path, monkeypatch):
    def tamper(document):
        document["metrics"]["binary"]["true_positive"] = 16
        document["metrics"]["binary"]["false_negative"] = 0

    _copy_run_to(tmp_path, monkeypatch, tamper)
    response = client.get("/demo/api/evaluation-run")
    assert response.status_code == 500
    assert response.json()["detail"] == "The stored evaluation run could not be verified."


def test_case_tamper_breaks_content_identity(client, tmp_path, monkeypatch):
    def tamper(document):
        document["cases"][0]["categories"][1] = "relabelled"

    _copy_run_to(tmp_path, monkeypatch, tamper)
    assert client.get("/demo/api/evaluation-run").status_code == 500


def test_missing_evaluation_run_returns_not_found(client, tmp_path, monkeypatch):
    settings = AudioSentinelSettings.from_project_root(tmp_path / "empty")
    monkeypatch.setattr("audio_sentinel.config.load_settings", lambda: settings)
    response = client.get("/demo/api/evaluation-run")
    assert response.status_code == 404


def test_stylesheet_defines_design_tokens_and_responsive_rules():
    css = (WEB / "style.css").read_text(encoding="utf-8")
    for token in ("--bg:", "--panel:", "--accent:", "--nav-accent:", "--radius:"):
        assert token in css
    assert "@media (max-width: 760px)" in css
    assert "prefers-reduced-motion" in css
    assert ":focus-visible" in css
