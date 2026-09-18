"""A3.3 tests for versioned timestamped acoustic-evidence JSON."""

from dataclasses import replace
from datetime import UTC, datetime
import hashlib
import json

import numpy as np
import pytest
import soundfile as sf
from pydantic import ValidationError

from audio_sentinel import acoustic_evidence as evidence
from audio_sentinel.acoustic_aggregation import AcousticAggregationSettings, aggregate_acoustic_events
from audio_sentinel.acoustic_inference import expected_yamnet_patches, infer_prepared_audio
from audio_sentinel.acoustic_loader import (
    AcousticModelMetadata,
    LoadedAcousticModel,
    TensorContract,
)
from audio_sentinel.acoustic_model import YAMNET, load_class_map
from audio_sentinel.config import AudioSettings
from audio_sentinel.interfaces import InputAudio
from audio_sentinel.pipeline import AudioPreparationService


NOW = datetime(2026, 9, 18, 12, 30, tzinfo=UTC)


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def numpy(self):
        return self.values


class FakeModel:
    def __call__(self, waveform):
        patches = expected_yamnet_patches(len(waveform))
        frames = YAMNET.patch_frames + (patches - 1) * 48
        scores = np.zeros((patches, YAMNET.num_classes), dtype=np.float32)
        scores[:, 390] = np.float32(0.8)
        scores[:, 420] = np.float32(0.7)
        return tuple(FakeTensor(item) for item in (
            scores,
            np.zeros((patches, 1024), dtype=np.float32),
            np.zeros((frames, YAMNET.mel_bands), dtype=np.float32),
        ))


def loaded_model() -> LoadedAcousticModel:
    metadata = AcousticModelMetadata(
        model_id=YAMNET.model_id,
        model_version="1",
        model_handle=YAMNET.model_handle,
        model_path="yamnet/1",
        artifact_sha256=YAMNET.artifact_sha256,
        class_map_sha256=YAMNET.class_map_sha256,
        label_mapping_version="1.0",
        runtime_distribution=YAMNET.runtime_distribution,
        runtime_version=YAMNET.runtime_version,
        exported_with_tensorflow="2.3.0",
        exported_with_tensorflow_git="test",
        signature_name="serving_default",
        input=TensorContract("waveform", (None,), "float32"),
        outputs=(
            TensorContract("output_0", (None, 521), "float32"),
            TensorContract("output_1", (None, 1024), "float32"),
            TensorContract("output_2", (None, 64), "float32"),
        ),
        num_classes=521,
    )
    return LoadedAcousticModel(metadata, load_class_map(), FakeModel())


@pytest.fixture
def snapshot(temporary_settings, active_consent):
    paths = temporary_settings.paths
    paths.raw_data.mkdir(parents=True)
    source_path = paths.raw_data / "source.wav"
    samples = 0.2 * np.sin(2 * np.pi * 440 * np.arange(25_600) / 16_000)
    sf.write(source_path, samples, 16_000, subtype="PCM_16")
    bundle = AudioPreparationService(temporary_settings, "synthetic").prepare(
        InputAudio("evidence-clip", source_path, active_consent),
        audio_settings=AudioSettings(
            target_sample_rate_hz=16_000,
            normalize_loudness=False,
            window_seconds=(1.0,),
            window_overlap_ratio=0.5,
        ),
        now=NOW,
    )
    inference = infer_prepared_audio(
        paths,
        bundle.manifest_path.relative_to(paths.interim_data),
        loaded_model(),
        now=NOW,
    )
    aggregation = aggregate_acoustic_events(
        inference, AcousticAggregationSettings.uniform(0.5)
    )
    return paths, bundle, inference, aggregation


def build(snapshot, **kwargs):
    return evidence.build_acoustic_evidence(snapshot[2], snapshot[3], now=NOW, **kwargs)


def save(snapshot, **kwargs):
    return evidence.save_acoustic_evidence(
        snapshot[0], snapshot[2], snapshot[3], now=NOW, **kwargs
    )


def test_builds_versioned_timestamped_sample_exact_evidence(snapshot):
    document = build(snapshot)
    assert document.schema_version == "1.0"
    assert document.document_type == "acoustic_event_candidates"
    assert document.created_at == NOW
    assert document.sample_rate_hz == 16_000
    assert document.input_window_count == 3
    assert document.input_patch_count == 5
    assert len(document.windows) == 3
    assert {item.label.value for item in document.events} == {"explosion", "siren"}
    first = document.events[0]
    assert first.start_seconds == first.start_sample / 16_000
    assert first.end_seconds == first.end_sample / 16_000
    assert all(item.start_seconds == item.start_sample / 16_000 for item in first.contributions)


def test_document_is_evidence_only_not_incident_or_probability(snapshot):
    payload = build(snapshot).model_dump(mode="json")
    serialized = json.dumps(payload)
    assert "risk_level" not in serialized
    assert "incident" not in serialized
    assert "probability" not in serialized
    assert payload["events"][0]["peak_score"] == pytest.approx(0.7)


def test_records_complete_source_model_window_and_patch_provenance(snapshot):
    paths, bundle, inference, _ = snapshot
    document = build(snapshot)
    assert document.source.preparation_manifest_sha256 == hashlib.sha256(
        bundle.manifest_path.read_bytes()
    ).hexdigest()
    assert document.source.raw_audio_sha256 == bundle.manifest.source.sha256
    assert document.model.artifact_sha256 == YAMNET.artifact_sha256
    assert document.model.label_mapping_version == "1.0"
    assert [item.window_id for item in document.windows] == [
        item.window.window_id for item in inference.windows
    ]
    assert str(paths.root) not in document.model_dump_json()
    contribution = document.events[0].contributions[0]
    assert contribution.window_audio_sha256 == document.windows[0].window_audio_sha256
    assert contribution.winning_class.index == 420


def test_evidence_id_is_stable_across_creation_times_but_recipe_sensitive(snapshot):
    first = build(snapshot)
    later = evidence.build_acoustic_evidence(
        snapshot[2], snapshot[3], now=datetime(2026, 9, 19, tzinfo=UTC)
    )
    changed_aggregation = aggregate_acoustic_events(
        snapshot[2], AcousticAggregationSettings.uniform(0.6, merge_gap_samples=1)
    )
    changed = evidence.build_acoustic_evidence(snapshot[2], changed_aggregation, now=NOW)
    assert first.evidence_id == later.evidence_id
    assert first.evidence_id != changed.evidence_id


def test_rejects_naive_creation_time(snapshot):
    with pytest.raises(evidence.AcousticEvidenceError) as error:
        evidence.build_acoustic_evidence(
            snapshot[2], snapshot[3], now=datetime(2026, 9, 18)
        )
    assert error.value.code == "invalid_time"


def test_rejects_aggregation_from_a_different_snapshot(snapshot):
    altered = replace(snapshot[2], clip_id="different-clip")
    with pytest.raises(evidence.AcousticEvidenceError) as error:
        evidence.build_acoustic_evidence(altered, snapshot[3], now=NOW)
    assert error.value.code == "aggregation_mismatch"


def test_json_round_trip_preserves_contract(snapshot):
    document = build(snapshot)
    assert evidence.AcousticEvidenceDocument.model_validate_json(
        document.model_dump_json()
    ) == document


@pytest.mark.parametrize("change", ["timestamp", "span", "score", "window_hash", "model"])
def test_contract_rejects_tampered_semantic_content(snapshot, change):
    payload = build(snapshot).model_dump(mode="json")
    if change == "timestamp":
        payload["events"][0]["start_seconds"] += 0.01
    elif change == "span":
        payload["events"][0]["end_sample"] -= 1
    elif change == "score":
        payload["events"][0]["peak_score"] = 0.99
    elif change == "window_hash":
        payload["events"][0]["contributions"][0]["window_audio_sha256"] = "0" * 64
    else:
        payload["model"]["artifact_sha256"] = "0" * 64
    with pytest.raises(ValidationError):
        evidence.AcousticEvidenceDocument.model_validate(payload)


def test_save_is_atomic_reloadable_and_repeat_save_reuses_original(snapshot):
    first = save(snapshot)
    before = first.evidence_path.read_bytes()
    second = evidence.save_acoustic_evidence(
        snapshot[0], snapshot[2], snapshot[3],
        now=datetime(2026, 9, 19, tzinfo=UTC),
    )
    relative = first.evidence_path.relative_to(snapshot[0].processed_data)
    loaded = evidence.load_acoustic_evidence(snapshot[0], relative, now=NOW)
    assert not first.reused and second.reused
    assert first.evidence_path == second.evidence_path
    assert first.evidence_path.read_bytes() == before
    assert second.evidence.created_at == NOW
    assert loaded == first.evidence
    assert {item.name for item in first.directory.iterdir()} == {"evidence.json"}


@pytest.mark.parametrize("target", ["manifest", "window"])
def test_changed_sources_invalidate_reload(snapshot, target):
    paths, bundle, _, _ = snapshot
    saved = save(snapshot)
    if target == "manifest":
        bundle.manifest_path.write_bytes(bundle.manifest_path.read_bytes() + b"\n")
    else:
        window = paths.interim_data / bundle.manifest.windows[0].audio_path
        window.write_bytes(window.read_bytes() + b"x")
    with pytest.raises(evidence.AcousticEvidenceError) as error:
        evidence.load_acoustic_evidence(
            paths, saved.evidence_path.relative_to(paths.processed_data), now=NOW
        )
    assert error.value.code == "source_changed"


def test_corrupt_existing_document_is_rejected_and_not_overwritten(snapshot):
    saved = save(snapshot)
    payload = json.loads(saved.evidence_path.read_text())
    payload["events"][0]["start_seconds"] = 9.0
    saved.evidence_path.write_text(json.dumps(payload))
    corrupt = saved.evidence_path.read_bytes()
    with pytest.raises(evidence.AcousticEvidenceError):
        save(snapshot)
    assert saved.evidence_path.read_bytes() == corrupt


@pytest.mark.parametrize("field", ["max_windows", "max_events", "max_contributions", "max_document_bytes"])
def test_resource_limits_reject_without_publishing(snapshot, field):
    limits = {
        "max_windows": 2,
        "max_events": 1,
        "max_contributions": 1,
        "max_document_bytes": 100,
    }
    policy = evidence.AcousticEvidencePolicy(**{field: limits[field]})
    with pytest.raises(evidence.AcousticEvidenceError):
        save(snapshot, policy=policy)
    parent = snapshot[0].processed_data / "acoustic-evidence"
    assert not parent.exists() or not list(parent.iterdir())


def test_path_escape_and_unexpected_bundle_entries_are_rejected(snapshot):
    saved = save(snapshot)
    with pytest.raises(evidence.AcousticEvidenceError) as escape:
        evidence.load_acoustic_evidence(snapshot[0], "../outside.json", now=NOW)
    assert escape.value.code == "invalid_path"
    (saved.directory / "unexpected.txt").write_text("x")
    with pytest.raises(evidence.AcousticEvidenceError) as inventory:
        evidence.load_acoustic_evidence(
            snapshot[0], saved.evidence_path.relative_to(snapshot[0].processed_data), now=NOW
        )
    assert inventory.value.code == "output_conflict"


def test_checked_in_schema_matches_generated_contract(project_root):
    path = project_root / "docs" / "schemas" / "v1" / "acoustic-evidence.schema.json"
    assert json.loads(path.read_text()) == evidence.acoustic_evidence_schema_documents()[
        "acoustic-evidence.schema.json"
    ]
