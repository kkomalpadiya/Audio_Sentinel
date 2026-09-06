from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from pydantic import ValidationError
import soundfile as sf

from audio_sentinel import persistence
from audio_sentinel.audio_loader import AudioLoadError
from audio_sentinel.audio_transforms import NormalizationStats, PreparedSignal
from audio_sentinel.config import AudioSettings, Paths, PersistenceSettings
from audio_sentinel.contracts import ConsentStatus, EventAnnotation
from audio_sentinel.persistence import AudioPersistenceError, save_prepared_audio
from audio_sentinel.preparation import PreparedAudioManifest, SourceAudioMetadata
from audio_sentinel.segmentation import iter_windows


NOW = datetime(2026, 9, 6, tzinfo=UTC)


@pytest.fixture
def signal(active_consent):
    frames = 25_600
    samples = np.column_stack((np.linspace(-0.7, 0.7, frames), np.linspace(0.2, -0.2, frames))).astype(np.float32)
    return PreparedSignal(
        clip_id="clip-001", consent=active_consent,
        source=SourceAudioMetadata(audio_path="synthetic/input.wav", sha256="a" * 64, format="WAV",
                                   sample_rate_hz=16_000, channels=2, num_frames=frames, size_bytes=44 + frames * 4),
        settings=AudioSettings(convert_to_mono=False), sample_rate_hz=16_000, samples=samples,
        normalization=NormalizationStats("disabled", 0, None, None, None, None),
    )


def save(signal, tmp_path, **options):
    return save_prepared_audio(signal, Paths.from_root(tmp_path), source_dataset="synthetic", now=NOW, **options)


def snapshot(directory):
    return {p.relative_to(directory).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in directory.rglob("*") if p.is_file()}


def assert_no_output(tmp_path):
    parent = tmp_path / "data/interim/prepared"
    assert not parent.exists() or list(parent.iterdir()) == []


@pytest.mark.parametrize("tail", ["pad", "drop"])
def test_round_trip_complete_inventory_and_annotations(signal, tmp_path, tail):
    signal = replace(signal, settings=signal.settings.model_copy(update={"tail_policy": tail}))
    original = signal.samples.copy()
    annotation = EventAnnotation(label="ambient", risk_level="none", start_seconds=0.1, end_seconds=1.0, confidence=1.0)
    result = save(signal, tmp_path, annotations=[annotation])
    assert not result.reused
    assert PreparedAudioManifest.model_validate_json(result.manifest_path.read_bytes()) == result.manifest
    assert result.manifest.clip.annotations == [annotation]
    assert result.manifest.source == signal.source
    assert result.manifest.settings == signal.settings
    assert result.audio.audio_path == result.directory / "audio.wav"
    assert result.audio.duration_seconds == 1.6
    assert result.audio.consent == signal.consent
    assert not result.manifest.clip.consent.raw_audio_retention_allowed
    expected_windows = list(iter_windows(signal, now=NOW))
    assert result.manifest.windows == tuple(w.record for w in expected_windows)
    pairs = [(result.manifest.clip.audio_path, signal.samples)]
    pairs += [(w.record.audio_path, w.samples) for w in expected_windows]
    for relative, samples in pairs:
        path = tmp_path / "data/interim" / relative
        info = sf.info(path)
        assert (info.format, info.subtype, info.samplerate, info.channels, info.frames) == ("WAV", "PCM_16", 16_000, 2, len(samples))
        actual, rate = sf.read(path, dtype="float32", always_2d=True)
        np.testing.assert_allclose(actual, samples, atol=1 / 65536)
    assert len(snapshot(result.directory)) == len(pairs) + 1
    assert not list(result.directory.parent.glob(".pending-*"))
    np.testing.assert_array_equal(signal.samples, original)


def test_pcm16_saturation_rounding_and_input_preservation(signal, tmp_path):
    values = np.array([-2, -1, -0.5, -0.5 / 32768, 0, 0.5 / 32768, 1.5 / 32768, 0.5, 1, 2], dtype=np.float32)
    signal.samples[:10, 0] = values
    result = save(signal, tmp_path)
    samples, _ = sf.read(result.audio.audio_path, dtype="int16", always_2d=True)
    np.testing.assert_array_equal(samples[:10, 0], [-32768, -32768, -16384, 0, 0, 0, 2, 16384, 32767, 32767])
    np.testing.assert_array_equal(signal.samples[:10, 0], values)


def test_repeat_reuses_identical_bundle_without_touching_it(signal, tmp_path):
    first = save(signal, tmp_path)
    before = snapshot(first.directory)
    second = save(signal, tmp_path)
    assert second.reused and second.directory == first.directory
    assert second.manifest == first.manifest
    assert snapshot(first.directory) == before
    assert list(first.directory.parent.iterdir()) == [first.directory]


@pytest.mark.parametrize("change", ["corrupt_wav", "missing_window", "extra_file", "extra_directory", "dataset", "consent"])
def test_conflicting_output_is_preserved(signal, tmp_path, change):
    first = save(signal, tmp_path)
    if change == "corrupt_wav":
        first.audio.audio_path.write_bytes(b"broken")
    elif change == "missing_window":
        (tmp_path / "data/interim" / first.manifest.windows[0].audio_path).unlink()
    elif change == "extra_file":
        (first.directory / "unexpected.txt").write_text("keep me")
    elif change == "extra_directory":
        (first.directory / "unexpected").mkdir()
    elif change == "consent":
        signal = replace(signal, consent=signal.consent.model_copy(update={"consent_id": "another-consent"}))
    before = snapshot(first.directory)
    with pytest.raises(AudioPersistenceError, match="Existing output") as error:
        save_prepared_audio(signal, Paths.from_root(tmp_path), source_dataset="changed" if change == "dataset" else "synthetic", now=NOW)
    assert error.value.code == "output_conflict"
    assert snapshot(first.directory) == before
    assert list(first.directory.parent.iterdir()) == [first.directory]


@pytest.mark.parametrize("field,value,code", [("max_windows", 4, "too_many_windows"), ("max_output_bytes", 50_000, "output_too_large")])
def test_limits_reject_before_writing(signal, tmp_path, field, value, code):
    with pytest.raises(AudioPersistenceError) as error:
        save(signal, tmp_path, policy=PersistenceSettings(**{field: value}))
    assert error.value.code == code
    assert_no_output(tmp_path)


def test_drop_can_save_full_clip_without_any_windows(signal, tmp_path):
    signal = replace(signal, settings=AudioSettings(convert_to_mono=False, window_seconds=(5.0,), tail_policy="drop"))
    result = save(signal, tmp_path, policy=PersistenceSettings(max_windows=0))
    assert result.manifest.windows == ()
    assert set(snapshot(result.directory)) == {"audio.wav", "manifest.json"}


@pytest.mark.parametrize("failure_at", [1, 3])
def test_disk_failure_removes_partial_staging(signal, tmp_path, monkeypatch, failure_at):
    real_write = persistence._write_verified_wav
    count = 0
    def write(path, samples, rate):
        nonlocal count
        count += 1
        if count == failure_at:
            path.write_bytes(b"partial WAV")
            raise OSError("simulated disk full")
        return real_write(path, samples, rate)
    monkeypatch.setattr(persistence, "_write_verified_wav", write)
    with pytest.raises(AudioPersistenceError) as error:
        save(signal, tmp_path)
    assert error.value.code == "write_failed"
    assert_no_output(tmp_path)


def test_manifest_validation_failure_cleans_wavs(signal, tmp_path, monkeypatch):
    original = persistence.iter_windows
    def incomplete(*args, **kwargs):
        windows = original(*args, **kwargs)
        next(windows)
        yield from windows
    monkeypatch.setattr(persistence, "iter_windows", incomplete)
    with pytest.raises(ValidationError):
        save(signal, tmp_path)
    assert_no_output(tmp_path)


def test_publish_failure_does_not_expose_partial_bundle(signal, tmp_path, monkeypatch):
    def fail(*args):
        raise PermissionError("simulated rename failure")
    monkeypatch.setattr(Path, "rename", fail)
    with pytest.raises(AudioPersistenceError) as error:
        save(signal, tmp_path)
    assert error.value.code == "write_failed"
    assert_no_output(tmp_path)


def test_competing_writer_publishes_matching_bundle(signal, tmp_path, monkeypatch):
    def publish_other(stage, destination):
        import shutil
        shutil.copytree(stage, destination)
        raise FileExistsError("another writer won")
    monkeypatch.setattr(Path, "rename", publish_other)
    result = save(signal, tmp_path)
    assert result.reused and result.manifest_path.exists()
    assert list(result.directory.parent.iterdir()) == [result.directory]


@pytest.mark.parametrize("status", ["denied", "withdrawn"])
def test_permission_denied_before_io(signal, tmp_path, status):
    signal = replace(signal, consent=signal.consent.model_copy(update={"status": ConsentStatus(status)}))
    with pytest.raises(AudioLoadError):
        save(signal, tmp_path)
    assert_no_output(tmp_path)


@pytest.mark.parametrize("reuse", [False, True])
def test_expiry_checked_at_final_boundary(signal, tmp_path, monkeypatch, reuse):
    signal = replace(signal, consent=signal.consent.model_copy(update={"expires_at": NOW + timedelta(seconds=1)}))
    result = save(signal, tmp_path) if reuse else None
    clock = [NOW]
    class Clock:
        @staticmethod
        def now(tz):
            return clock[0]
    monkeypatch.setattr(persistence, "datetime", Clock)
    if reuse:
        original = persistence._compare_existing
        def compare(*args):
            original(*args)
            clock[0] += timedelta(seconds=2)
        monkeypatch.setattr(persistence, "_compare_existing", compare)
    else:
        original = PreparedAudioManifest.model_validate_json
        def validate(*args, **kwargs):
            value = original(*args, **kwargs)
            clock[0] += timedelta(seconds=2)
            return value
        monkeypatch.setattr(PreparedAudioManifest, "model_validate_json", validate)
    # Freeze the generator's separate clock too; expiry advances only at the final boundary.
    import audio_sentinel.segmentation as segmentation
    monkeypatch.setattr(segmentation, "datetime", Clock)
    with pytest.raises(AudioLoadError) as error:
        save_prepared_audio(signal, Paths.from_root(tmp_path), source_dataset="synthetic")
    assert error.value.code == "consent_expired"
    if reuse:
        assert list(result.directory.parent.iterdir()) == [result.directory]
    else:
        assert_no_output(tmp_path)


@pytest.mark.parametrize("path_kind", ["outside", "raw", "ancestor", "root"])
def test_invalid_output_roots_do_not_write(signal, tmp_path, path_kind):
    paths = Paths.from_root(tmp_path)
    target = {"outside": tmp_path.parent / "escaped", "raw": paths.raw_data,
              "ancestor": paths.raw_data.parent, "root": paths.root}[path_kind]
    paths = paths.model_copy(update={"interim_data": target})
    with pytest.raises(AudioPersistenceError) as error:
        save_prepared_audio(signal, paths, source_dataset="synthetic", now=NOW)
    assert error.value.code == "invalid_output_path"
    assert_no_output(tmp_path)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows junction regression")
@pytest.mark.parametrize("location", ["parent", "bundle", "child"])
def test_junctions_are_rejected_without_following_them(signal, tmp_path, location):
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "keep.txt"
    sentinel.write_text("untouched")
    if location == "parent":
        junction = tmp_path / "data/interim"
        junction.parent.mkdir()
    else:
        result = save(signal, tmp_path)
        if location == "bundle":
            moved = result.directory.with_name("original")
            result.directory.rename(moved)
            junction = result.directory
        else:
            junction = result.directory / "linked"
    # mklink is used only to create the fixture; no shell deletion or moving.
    subprocess.run(["cmd", "/c", "mklink", "/J", str(junction), str(outside)], check=True, capture_output=True)
    try:
        with pytest.raises(AudioPersistenceError) as error:
            save(signal, tmp_path)
        assert error.value.code == ("invalid_output_path" if location == "parent" else "output_conflict")
        assert sentinel.read_text() == "untouched"
        assert list(outside.iterdir()) == [sentinel]
    finally:
        junction.rmdir()  # Removes only the junction itself.


def test_source_metadata_revalidated_before_writing(signal, tmp_path):
    signal = replace(signal, source=signal.source.model_copy(update={"audio_path": "../escape.wav"}))
    with pytest.raises(ValidationError):
        save(signal, tmp_path)
    assert_no_output(tmp_path)


@pytest.mark.parametrize("options", [{"max_windows": -1}, {"max_output_bytes": 0}, {"unknown": True}])
def test_invalid_persistence_policy(options):
    with pytest.raises(ValidationError):
        PersistenceSettings(**options)


def test_corrupted_write_is_detected_before_publication(signal, tmp_path, monkeypatch):
    original = sf.SoundFile.read
    def corrupt(audio, *args, **kwargs):
        data = original(audio, *args, **kwargs)
        if len(data):
            data[0, 0] ^= 1
        return data
    monkeypatch.setattr(sf.SoundFile, "read", corrupt)
    with pytest.raises(AudioPersistenceError) as error:
        save(signal, tmp_path)
    assert error.value.code == "verification_failed"
    assert_no_output(tmp_path)


@pytest.mark.parametrize("failure", ["manifest", "flush"])
def test_manifest_and_flush_io_failures_clean_staging(signal, tmp_path, monkeypatch, failure):
    if failure == "manifest":
        original = Path.open
        def fail(path, *args, **kwargs):
            if path.name == "manifest.json" and args == ("xb",):
                raise OSError("manifest write failed")
            return original(path, *args, **kwargs)
        monkeypatch.setattr(Path, "open", fail)
    else:
        def fail(fd):
            raise OSError("flush failed")
        monkeypatch.setattr(persistence.os, "fsync", fail)
    with pytest.raises(AudioPersistenceError) as error:
        save(signal, tmp_path)
    assert error.value.code == "write_failed"
    assert_no_output(tmp_path)


def test_annotation_outside_clip_fails_before_writing(signal, tmp_path):
    annotation = EventAnnotation(label="ambient", risk_level="none", start_seconds=1, end_seconds=2, confidence=1)
    with pytest.raises(ValidationError):
        save(signal, tmp_path, annotations=[annotation])
    assert_no_output(tmp_path)
