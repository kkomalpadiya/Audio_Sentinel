"""A9.2 responsible-use documentation parity checks."""

from __future__ import annotations

from pathlib import Path

from audio_sentinel.contracts import ConsentStatus, ProcessingScope


def _document(project_root: Path) -> str:
    document = (
        project_root / "docs" / "authorized-use-and-limitations.md"
    ).read_text(encoding="utf-8")
    return " ".join(document.split())


def test_document_covers_every_consent_status_and_processing_scope(
    project_root: Path,
) -> None:
    document = _document(project_root)

    for status in ConsentStatus:
        assert f"`{status.value}`" in document
    for scope in ProcessingScope:
        assert f"`{scope.value}`" in document
    for rejection_state in ("expired", "not-yet-active", "unauthorized device"):
        assert rejection_state in document


def test_document_preserves_local_alert_and_human_review_boundary(
    project_root: Path,
) -> None:
    document = _document(project_root)

    assert "`notification_delivery=not_sent`" in document
    assert "`alert_delivery_authorized=false`" in document
    assert "pending local candidate" in document
    assert "human review remains required" in document
    assert "Audio Sentinel cannot make that decision or contact anyone" in document
    for disposition in ("`confirmed`", "`dismissed`", "`insufficient_evidence`"):
        assert disposition in document


def test_document_records_model_confidence_and_policy_limitations(
    project_root: Path,
) -> None:
    document = _document(project_root)

    for component in (
        "YAMNet acoustic model",
        "Silero VAD v6",
        "Faster-Whisper `tiny.en`",
        "English language rules",
        "Risk scoring",
        "Consensus",
    ):
        assert component in document
    assert "not incident probabilities" in document
    assert "not a calibrated word-correctness probability" in document
    assert "A score is not a probability" in document
    assert "Acoustic-only cases cannot become alerts" in document


def test_document_reports_a9_1_results_and_limits_without_overclaiming(
    project_root: Path,
) -> None:
    document = _document(project_root)

    for result in (
        "13 true positives",
        "2 false positives",
        "3 false negatives",
        "14 true negatives",
        "84.375%",
        "86.667%",
        "81.25%",
        "87.5%",
        "83.871%",
    ):
        assert result in document
    assert "does not approve a production threshold or accuracy claim" in document
    assert "not representative of deployment prevalence or environments" in document
