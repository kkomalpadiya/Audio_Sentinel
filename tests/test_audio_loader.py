from dataclasses import replace
from datetime import UTC, datetime, timedelta
import hashlib
import os
from pathlib import Path
import subprocess
import wave

import numpy as np
import pytest
import soundfile as sf

from audio_sentinel.audio_loader import AudioLoadError, load_audio
from audio_sentinel.config import AudioSentinelSettings, AudioSettings
from audio_sentinel.contracts import ConsentRecord, ProcessingScope
from audio_sentinel.interfaces import InputAudio


NOW = datetime(2026, 9, 5, 12, tzinfo=UTC)


@pytest.fixture
def make_clip(temporary_settings, active_consent):
    temporary_settings.ensure_directories()

    def make(samples=None, rate=16_000, format="WAV", subtype="PCM_16", name="example/clip.wav"):
        path = temporary_settings.paths.raw_data / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if samples is None:
            samples = np.array([-1, -0.5, 0, 0.25, 0.5], dtype=np.float32)
        sf.write(path, samples, rate, format=format, subtype=subtype)
        return InputAudio("clip-001", Path(name), active_consent)

    return make


def assert_rejected(code, clip, settings, **kwargs):
    with pytest.raises(AudioLoadError) as caught:
        load_audio(clip, settings, now=NOW, **kwargs)
    assert caught.value.code == code


@pytest.mark.parametrize("format,subtype", [
    ("WAV", "PCM_U8"), ("WAV", "PCM_16"), ("WAV", "PCM_24"),
    ("WAV", "PCM_32"), ("WAV", "FLOAT"), ("FLAC", "PCM_16"), ("FLAC", "PCM_24"),
])
def test_formats_scale_to_float32_and_preserve_provenance(make_clip, temporary_settings, format, subtype):
    clip = make_clip(format=format, subtype=subtype)
    source_path = temporary_settings.paths.raw_data / clip.audio_path
    original = source_path.read_bytes()
    loaded = load_audio(clip, temporary_settings, now=NOW)
    assert loaded.samples.dtype == np.float32
    assert loaded.samples.shape == (5, 1)
    np.testing.assert_allclose(loaded.samples[:, 0], [-1, -0.5, 0, 0.25, 0.5], atol=1e-7)
    assert loaded.clip_id == clip.clip_id
    assert loaded.consent == clip.consent
    assert loaded.source.format == format
    assert loaded.source.sha256 == hashlib.sha256(original).hexdigest()
    assert loaded.source.size_bytes == len(original)
    assert loaded.source.audio_path == "example/clip.wav"
    assert source_path.read_bytes() == original
    assert list(temporary_settings.paths.interim_data.iterdir()) == []


def test_stereo_and_original_rate_survive_multiple_read_blocks(make_clip, temporary_settings):
    samples = np.column_stack((np.full(70_000, 0.5), np.full(70_000, -0.25)))
    clip = make_clip(samples, rate=44_100)
    loaded = load_audio(clip, temporary_settings, now=NOW)
    np.testing.assert_array_equal(loaded.samples, samples)
    assert loaded.source.sample_rate_hz == 44_100
    assert loaded.source.num_frames == 70_000
    assert loaded.source.channels == 2


def test_wav_extensible_is_recognized_as_wav(make_clip, temporary_settings):
    clip = make_clip(format="WAVEX")
    assert load_audio(clip, temporary_settings, now=NOW).source.format == "WAV"


def test_valid_absolute_path_inside_raw_root(make_clip, temporary_settings):
    clip = make_clip()
    absolute_clip = replace(clip, audio_path=temporary_settings.paths.raw_data / clip.audio_path)
    assert load_audio(absolute_clip, temporary_settings, now=NOW).source.audio_path == "example/clip.wav"


@pytest.mark.parametrize("path", ["../outside.wav", "https://example.com/file.wav"])
def test_relative_escape_and_url_are_rejected(make_clip, temporary_settings, path):
    assert_rejected("invalid_path", replace(make_clip(), audio_path=Path(path)), temporary_settings)


def test_absolute_path_outside_root_is_rejected(make_clip, temporary_settings, tmp_path):
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"not audio")
    assert_rejected("invalid_path", replace(make_clip(), audio_path=outside), temporary_settings)


def test_symlink_escape_is_rejected(make_clip, temporary_settings, tmp_path):
    clip = make_clip()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "clip.wav").write_bytes(b"not audio")
    link = temporary_settings.paths.raw_data / "escape"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        if os.name != "nt":
            raise
        # Directory junctions exercise Windows path resolution without admin rights.
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(outside)],
            check=True, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW,
        )
    try:
        assert_rejected("invalid_path", replace(clip, audio_path=Path("escape/clip.wav")), temporary_settings)
    finally:
        if link.is_symlink():
            link.unlink()
        else:
            link.rmdir()  # Remove the junction itself, never the outside target.


def test_directory_and_missing_file_are_rejected(make_clip, temporary_settings):
    clip = make_clip()
    assert_rejected("invalid_path", replace(clip, audio_path=Path("example")), temporary_settings)
    assert_rejected("file_not_found", replace(clip, audio_path=Path("missing.wav")), temporary_settings)


@pytest.mark.parametrize("payload,code", [(b"", "empty_audio"), (b"not a recording", "invalid_audio")])
def test_empty_and_corrupt_files(make_clip, temporary_settings, payload, code):
    clip = make_clip()
    (temporary_settings.paths.raw_data / clip.audio_path).write_bytes(payload)
    assert_rejected(code, clip, temporary_settings)


def test_header_without_samples_is_rejected(make_clip, temporary_settings):
    assert_rejected("empty_audio", make_clip(np.empty(0)), temporary_settings)


def test_truncated_wav_is_rejected_before_decode(make_clip, temporary_settings, monkeypatch):
    clip = make_clip(np.zeros(100))
    path = temporary_settings.paths.raw_data / clip.audio_path
    path.write_bytes(path.read_bytes()[:-10])
    monkeypatch.setattr(sf, "SoundFile", lambda *args, **kwargs: pytest.fail("Must reject before decoding"))
    assert_rejected("invalid_audio", clip, temporary_settings)


def test_truncated_flac_is_rejected(make_clip, temporary_settings):
    clip = make_clip(np.linspace(-0.5, 0.5, 100_000), format="FLAC")
    path = temporary_settings.paths.raw_data / clip.audio_path
    path.write_bytes(path.read_bytes()[:-100])
    assert_rejected("invalid_audio", clip, temporary_settings)


def test_partial_pcm_frame_is_rejected(make_clip, temporary_settings):
    clip = make_clip()
    path = temporary_settings.paths.raw_data / clip.audio_path
    payload = bytearray(path.read_bytes()[:-1])
    payload[4:8] = (len(payload) - 8).to_bytes(4, "little")
    data_offset = payload.index(b"data")
    size = int.from_bytes(payload[data_offset + 4:data_offset + 8], "little")
    payload[data_offset + 4:data_offset + 8] = (size - 1).to_bytes(4, "little")
    path.write_bytes(payload)
    assert_rejected("invalid_audio", clip, temporary_settings)


def test_container_detection_does_not_trust_extension(make_clip, temporary_settings):
    clip = make_clip(format="AIFF", name="example/not-really-wav.wav")
    assert_rejected("unsupported_format", clip, temporary_settings)


def test_integer_pcm_scaling_against_independently_written_wave(make_clip, temporary_settings):
    clip = make_clip()
    path = temporary_settings.paths.raw_data / clip.audio_path
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(np.array([-32768, 0, 32767], dtype="<i2").tobytes())
    loaded = load_audio(clip, temporary_settings, now=NOW)
    np.testing.assert_array_equal(loaded.samples[:, 0], [-1, 0, 32767 / 32768])


@pytest.mark.parametrize("samples", [[np.nan], [np.inf], [-np.inf]])
def test_non_finite_float_audio_is_rejected(make_clip, temporary_settings, samples):
    assert_rejected("non_finite_audio", make_clip(np.array(samples), subtype="FLOAT"), temporary_settings)


def test_float_amplitudes_and_silence_are_preserved(make_clip, temporary_settings):
    clip = make_clip(np.array([0, 2, -2, 0], dtype=np.float32), subtype="FLOAT")
    np.testing.assert_array_equal(load_audio(clip, temporary_settings, now=NOW).samples[:, 0], [0, 2, -2, 0])
    silent = make_clip(np.zeros(100))
    assert not load_audio(silent, temporary_settings, now=NOW).samples.any()


@pytest.mark.parametrize("override,code", [
    ({"max_input_bytes": 1}, "input_too_large"),
    ({"max_decoded_bytes": 1}, "decoded_audio_too_large"),
    ({"max_duration_seconds": 0.0001}, "audio_too_long"),
    ({"max_input_channels": 1}, "too_many_channels"),
    ({"accepted_formats": ("FLAC",)}, "unsupported_format"),
])
def test_limits_reject_before_sample_allocation(make_clip, temporary_settings, monkeypatch, override, code):
    clip = make_clip(np.zeros((100, 2)))
    settings = AudioSentinelSettings(paths=temporary_settings.paths, audio=AudioSettings(**override))
    monkeypatch.setattr(np, "empty", lambda *args, **kwargs: pytest.fail("Must reject before allocating samples"))
    assert_rejected(code, clip, settings)


def test_exact_limits_are_accepted(make_clip, temporary_settings):
    clip = make_clip(np.zeros((100, 2)))
    size = (temporary_settings.paths.raw_data / clip.audio_path).stat().st_size
    settings = AudioSentinelSettings(paths=temporary_settings.paths, audio=AudioSettings(
        max_input_bytes=size, max_decoded_bytes=800, max_duration_seconds=100 / 16_000, max_input_channels=2,
    ))
    assert load_audio(clip, settings, now=NOW).samples.shape == (100, 2)


@pytest.mark.parametrize("rate", [4_000, 384_000])
def test_invalid_source_sample_rate(make_clip, temporary_settings, rate):
    assert_rejected("invalid_sample_rate", make_clip(rate=rate), temporary_settings)


def test_clip_that_rounds_to_zero_target_frames_is_rejected(make_clip, temporary_settings):
    assert_rejected("audio_too_short", make_clip(np.zeros(1), rate=192_000), temporary_settings)


@pytest.mark.parametrize("status", ["denied", "withdrawn"])
def test_permission_is_checked_before_file_access(make_clip, temporary_settings, monkeypatch, status):
    consent = ConsentRecord(consent_id="consent-001", status=status, processing_scope="none", device_authorized=False)
    clip = replace(make_clip(), consent=consent)
    monkeypatch.setattr(Path, "resolve", lambda *args, **kwargs: pytest.fail("Must not access a source without permission"))
    assert_rejected("consent_denied", clip, temporary_settings)


@pytest.mark.parametrize("update,code", [
    ({"granted_at": NOW + timedelta(seconds=1)}, "consent_not_yet_active"),
    ({"expires_at": NOW}, "consent_expired"),
    ({"expires_at": NOW - timedelta(seconds=1)}, "consent_expired"),
    ({"granted_at": NOW.replace(tzinfo=None)}, "invalid_consent"),
    ({"device_authorized": False}, "invalid_consent"),
])
def test_consent_time_and_device_checks(make_clip, temporary_settings, update, code):
    clip = make_clip()
    # Deliberately use unvalidated update to exercise validation at the loader boundary.
    clip = replace(clip, consent=clip.consent.model_copy(update=update))
    assert_rejected(code, clip, temporary_settings)


def test_acoustic_only_permission_is_sufficient(make_clip, temporary_settings):
    clip = make_clip()
    clip = replace(clip, consent=clip.consent.model_copy(update={"processing_scope": ProcessingScope.ACOUSTIC_ONLY}))
    assert load_audio(clip, temporary_settings, now=NOW).consent.permits_acoustic_processing


def test_invalid_clip_id_is_rejected(make_clip, temporary_settings):
    assert_rejected("invalid_clip_id", replace(make_clip(), clip_id="../name"), temporary_settings)


def test_source_change_during_decode_is_rejected(make_clip, temporary_settings, monkeypatch):
    clip = make_clip()
    path = temporary_settings.paths.raw_data / clip.audio_path
    original_read = sf.SoundFile.read

    def read_then_append(audio, *args, **kwargs):
        result = original_read(audio, *args, **kwargs)
        if len(result):
            with path.open("ab") as output:
                output.write(b"changed")
        return result

    monkeypatch.setattr(sf.SoundFile, "read", read_then_append)
    assert_rejected("source_changed", clip, temporary_settings)


def test_permission_is_rechecked_after_decode(make_clip, temporary_settings, monkeypatch):
    from audio_sentinel import audio_loader

    clip = make_clip()
    clip = replace(clip, consent=clip.consent.model_copy(update={"expires_at": NOW + timedelta(seconds=1)}))
    times = iter([NOW, NOW + timedelta(seconds=1)])

    class Clock:
        @staticmethod
        def now(tz):
            return next(times)

    monkeypatch.setattr(audio_loader, "datetime", Clock)
    with pytest.raises(AudioLoadError) as caught:
        load_audio(clip, temporary_settings)
    assert caught.value.code == "consent_expired"
