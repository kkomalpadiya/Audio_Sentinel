from datetime import UTC, datetime, timedelta
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from pydantic import ValidationError
import soundfile as sf

from audio_sentinel import feature_persistence as storage
from audio_sentinel.audio_loader import AudioLoadError
from audio_sentinel.feature_persistence import FeaturePersistenceError, FeaturePersistenceSettings, load_log_mel, save_window_log_mel
from audio_sentinel.feature_settings import LogMelSettings
from audio_sentinel.interfaces import InputAudio
from audio_sentinel.log_mel import generate_log_mel
from audio_sentinel.pipeline import AudioPreparationService


NOW = datetime(2026, 9, 7, tzinfo=UTC)


@pytest.fixture
def prepared(temporary_settings, active_consent):
    settings = temporary_settings
    settings.paths.raw_data.mkdir(parents=True)
    path = settings.paths.raw_data / "tone.wav"
    tone = 0.2*np.sin(2*np.pi*1000*np.arange(25600)/16000)
    sf.write(path,tone,16000,subtype="PCM_16")
    result = AudioPreparationService(settings,"synthetic-tone").prepare(InputAudio("tone-001",path,active_consent),now=NOW)
    return settings.paths,result


def save(prepared, index=0, **kwargs):
    paths, bundle=prepared
    return save_window_log_mel(paths,bundle.manifest_path.relative_to(paths.interim_data),
                               bundle.manifest.windows[index].window_id,kwargs.pop("settings",LogMelSettings()),
                               now=kwargs.pop("now",NOW),**kwargs)


def reload(prepared, saved, **kwargs):
    paths,_=prepared
    return load_log_mel(paths,saved.metadata_path.relative_to(paths.processed_data),now=kwargs.pop("now",NOW),**kwargs)


def no_partial(paths):
    assert not list(paths.processed_data.rglob(".pending-*"))


@pytest.mark.parametrize("index,shape",[(0,(64,97)),(2,(64,97)),(3,(64,497)),(4,(64,997))])
def test_saved_features_and_metadata_match_actual_source(prepared,index,shape):
    paths,bundle=prepared
    window=bundle.manifest.windows[index]
    source_path=paths.interim_data/window.audio_path
    before=source_path.read_bytes()
    saved=save(prepared,index)
    loaded=reload(prepared,saved)
    samples,rate=sf.read(source_path,dtype="float32",always_2d=True)
    np.testing.assert_array_equal(loaded.values,generate_log_mel(samples,rate,LogMelSettings()).values)
    assert loaded.values.shape==shape and loaded.values.dtype==np.float32
    metadata=loaded.metadata
    assert metadata==saved.metadata and not saved.reused
    assert metadata.source.window==window
    assert metadata.source.window_audio_sha256==hashlib.sha256(before).hexdigest()
    assert metadata.source.preparation_manifest_sha256==hashlib.sha256(bundle.manifest_path.read_bytes()).hexdigest()
    assert metadata.feature_sha256==hashlib.sha256(saved.feature_path.read_bytes()).hexdigest()
    assert source_path.read_bytes()==before
    assert {p.name for p in saved.directory.iterdir()}=={"features.npy","metadata.json"}
    no_partial(paths)


def test_repeat_save_preserves_bytes_times_and_creation_timestamp(prepared):
    first=save(prepared)
    before={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in first.directory.iterdir()}
    second=save(prepared,now=NOW+timedelta(seconds=30))
    assert second.reused and first.directory==second.directory
    assert second.metadata.created_at==NOW
    assert before=={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in second.directory.iterdir()}
    no_partial(prepared[0])


def test_different_recipe_has_separate_identity(prepared):
    first=save(prepared)
    second=save(prepared,settings=LogMelSettings(n_mels=32))
    assert first.directory!=second.directory
    assert reload(prepared,second).values.shape==(32,97)
    assert reload(prepared,first).values.shape==(64,97)


@pytest.mark.parametrize("change",["bytes","shape","dtype","object","trailing","missing","extra","metadata"])
def test_corrupt_existing_output_is_rejected_and_preserved(prepared,change):
    saved=save(prepared)
    if change=="bytes":
        data=bytearray(saved.feature_path.read_bytes()); data[-1]^=1; saved.feature_path.write_bytes(data)
    elif change in ("shape","dtype","object"):
        values={"shape":np.zeros((1,1),dtype=np.float32),"dtype":np.zeros((64,97),dtype=np.float64),
                "object":np.array([["no pickle loading"]],dtype=object)}[change]
        np.save(saved.feature_path,values)
    elif change=="trailing":
        with saved.feature_path.open("ab") as f: f.write(b"extra")
    elif change=="missing":
        saved.feature_path.unlink()
    elif change=="extra":
        (saved.directory/"unexpected.txt").write_text("keep")
    else:
        data=json.loads(saved.metadata_path.read_text()); data["feature_sha256"]="0"*64
        saved.metadata_path.write_text(json.dumps(data))
    before={p.name:p.read_bytes() for p in saved.directory.iterdir()}
    with pytest.raises(ValueError): reload(prepared,saved)
    with pytest.raises(ValueError): save(prepared)
    assert before=={p.name:p.read_bytes() for p in saved.directory.iterdir()}
    no_partial(prepared[0])


def test_nonfinite_array_rejected_even_with_matching_hash(prepared):
    saved=save(prepared)
    np.save(saved.feature_path,np.full((64,97),np.nan,dtype=np.float32),allow_pickle=False)
    data=json.loads(saved.metadata_path.read_text())
    data["feature_sha256"]=hashlib.sha256(saved.feature_path.read_bytes()).hexdigest()
    saved.metadata_path.write_text(json.dumps(data))
    with pytest.raises(FeaturePersistenceError,match="non-finite"):
        reload(prepared,saved)


@pytest.mark.parametrize("target",["manifest","window"])
def test_changed_sources_invalidate_reload(prepared,target):
    paths,bundle=prepared
    saved=save(prepared)
    if target=="manifest":
        with bundle.manifest_path.open("ab") as f: f.write(b"\n")
    else:
        path=paths.interim_data/bundle.manifest.windows[0].audio_path
        samples,rate=sf.read(path,dtype="int16"); samples[100]+=1
        sf.write(path,samples,rate,subtype="PCM_16")
    with pytest.raises(FeaturePersistenceError) as error: reload(prepared,saved)
    assert error.value.code=="source_changed"


@pytest.mark.parametrize("scope",["expired","denied"])
def test_consent_changes_prevent_reload_and_new_save(prepared,scope):
    _,bundle=prepared
    saved=save(prepared)
    data=json.loads(bundle.manifest_path.read_text())
    if scope=="expired": data["clip"]["consent"]["expires_at"]=NOW.isoformat()
    else:
        data["clip"]["consent"].update(status="denied",processing_scope="none",device_authorized=False)
    bundle.manifest_path.write_text(json.dumps(data))
    expected = AudioLoadError if scope == "expired" else ValidationError
    with pytest.raises(expected): reload(prepared,saved)
    with pytest.raises(expected): save(prepared)


@pytest.mark.parametrize("field",["max_metadata_bytes","max_window_bytes","max_decoded_bytes","max_output_bytes"])
def test_limits_reject_without_partial_output(prepared,field):
    with pytest.raises(FeaturePersistenceError):
        save(prepared,policy=FeaturePersistenceSettings(**{field:1}))
    no_partial(prepared[0])


@pytest.mark.parametrize("manifest",["../escape.json","/absolute.json","C:/escape.json","a\\b.json"])
def test_source_path_escape_rejected(prepared,manifest):
    paths,bundle=prepared
    with pytest.raises(FeaturePersistenceError) as error:
        save_window_log_mel(paths,manifest,bundle.manifest.windows[0].window_id,LogMelSettings(),now=NOW)
    assert error.value.code=="invalid_path"


def test_unlisted_window_rejected(prepared):
    paths,bundle=prepared
    with pytest.raises(FeaturePersistenceError,match="not listed"):
        save_window_log_mel(paths,bundle.manifest_path.relative_to(paths.interim_data),"missing-window",LogMelSettings(),now=NOW)


@pytest.mark.parametrize("failure",["write","publish","source_change"])
def test_failed_save_cleans_staging_preserves_prior_bundle_and_can_retry(prepared,monkeypatch,failure):
    paths,bundle=prepared
    original=save(prepared)
    before={p.name:p.read_bytes() for p in original.directory.iterdir()}
    manifest_bytes=bundle.manifest_path.read_bytes()
    with monkeypatch.context() as m:
        if failure=="write":
            original_save=storage.np.save
            def fail(*args,**kwargs):
                original_save(*args,**kwargs); raise OSError("injected disk error")
            m.setattr(storage.np,"save",fail)
        elif failure=="publish":
            original_rename=Path.rename
            def fail(path,target):
                if path.name.startswith(".pending-"): raise OSError("injected publication error")
                return original_rename(path,target)
            m.setattr(Path,"rename",fail)
        else:
            original_generate=storage.generate_log_mel
            def fail(*args,**kwargs):
                result=original_generate(*args,**kwargs)
                bundle.manifest_path.write_bytes(manifest_bytes+b"\n")
                return result
            m.setattr(storage,"generate_log_mel",fail)
        with pytest.raises(FeaturePersistenceError): save(prepared,index=1)
    bundle.manifest_path.write_bytes(manifest_bytes)
    assert before=={p.name:p.read_bytes() for p in original.directory.iterdir()}
    no_partial(paths)
    assert not save(prepared,index=1).reused


@pytest.mark.skipif(sys.platform!="win32",reason="Windows junction regression")
@pytest.mark.parametrize("location",["output","source"])
def test_windows_junctions_rejected(prepared,tmp_path,location):
    paths,bundle=prepared
    outside=tmp_path/"outside"; outside.mkdir()
    sentinel=outside/"keep.txt"; sentinel.write_text("untouched")
    junction=paths.processed_data if location=="output" else paths.interim_data/"linked"
    subprocess.run(["cmd","/c","mklink","/J",str(junction),str(outside)],check=True,capture_output=True)
    try:
        with pytest.raises(FeaturePersistenceError):
            if location=="output": save(prepared)
            else: save_window_log_mel(paths,"linked/manifest.json","missing",LogMelSettings(),now=NOW)
        assert sentinel.read_text()=="untouched" and list(outside.iterdir())==[sentinel]
    finally:
        junction.rmdir()


def test_output_cannot_overlap_interim_data(prepared):
    paths,bundle=prepared
    altered=paths.model_copy(update={"processed_data":paths.interim_data/"features"})
    with pytest.raises(FeaturePersistenceError,match="separate"):
        save_window_log_mel(altered,bundle.manifest_path.relative_to(paths.interim_data),bundle.manifest.windows[0].window_id,LogMelSettings(),now=NOW)


def test_huge_npy_shape_is_rejected_before_payload_allocation(prepared,monkeypatch):
    saved=save(prepared)
    with saved.feature_path.open("wb") as stream:
        np.lib.format.write_array_header_1_0(stream,dict(descr="<f4",fortran_order=False,shape=(64,2**50)))
    def unexpected(*args,**kwargs): pytest.fail("Invalid header reached array allocation")
    monkeypatch.setattr(storage.np,"frombuffer",unexpected)
    with pytest.raises(FeaturePersistenceError,match="shape"):
        reload(prepared,saved)


@pytest.mark.parametrize("failure",["rate","padding","length"])
def test_prepared_wav_properties_are_checked(prepared,failure):
    paths,bundle=prepared
    index=2 if failure=="padding" else 0
    path=paths.interim_data/bundle.manifest.windows[index].audio_path
    samples,rate=sf.read(path,dtype="int16")
    if failure=="rate": rate=8000
    elif failure=="length": samples=samples[:-1]
    else: samples[-1]=1
    sf.write(path,samples,rate,subtype="PCM_16")
    with pytest.raises(FeaturePersistenceError) as error: save(prepared,index=index)
    assert error.value.code=="source_mismatch"
    no_partial(paths)


def test_consent_expiring_during_generation_stops_publication(prepared,monkeypatch):
    paths,bundle=prepared
    data=json.loads(bundle.manifest_path.read_text())
    data["clip"]["consent"]["expires_at"]=(NOW+timedelta(seconds=1)).isoformat()
    bundle.manifest_path.write_text(json.dumps(data))
    current=[NOW]
    class Clock:
        @staticmethod
        def now(tz): return current[0]
    monkeypatch.setattr(storage,"datetime",Clock)
    generate=storage.generate_log_mel
    def delayed(*args,**kwargs):
        result=generate(*args,**kwargs)
        current[0]+=timedelta(seconds=2)
        return result
    monkeypatch.setattr(storage,"generate_log_mel",delayed)
    with pytest.raises(AudioLoadError) as error: save(prepared,now=None)
    assert error.value.code=="consent_expired"
    no_partial(paths)
    assert not list((paths.processed_data/"log-mel").iterdir())


def test_cooperating_writer_publication_race_reuses_verified_output(prepared,monkeypatch):
    import shutil
    def competing_writer(path,target):
        shutil.copytree(path,target)
        raise FileExistsError("another writer published first")
    monkeypatch.setattr(Path,"rename",competing_writer)
    saved=save(prepared)
    assert saved.reused and reload(prepared,saved).values.shape==(64,97)
    no_partial(prepared[0])
