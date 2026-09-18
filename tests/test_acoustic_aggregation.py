"""B3.2 tests for mapped-score and overlapping-window aggregation."""

from dataclasses import FrozenInstanceError, replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from audio_sentinel import acoustic_aggregation as aggregation
from audio_sentinel.acoustic_inference import (
    AcousticInferenceResult,
    AcousticWindowInference,
    expected_yamnet_patches,
)
from audio_sentinel.acoustic_loader import AcousticModelMetadata, TensorContract
from audio_sentinel.acoustic_model import LABEL_MAPPING, YAMNET
from audio_sentinel.contracts import EventAnnotation, EventLabel
from audio_sentinel.preparation import PreparedWindowRecord


def model_metadata() -> AcousticModelMetadata:
    return AcousticModelMetadata(
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


def window(window_id: str, start: int, length: int, values=None) -> AcousticWindowInference:
    patches = expected_yamnet_patches(length)
    scores = np.zeros((patches, 521), dtype=np.float32)
    for patch, index, score in values or ():
        scores[patch, index] = score
    frames = YAMNET.patch_frames + (patches - 1) * 48
    record = PreparedWindowRecord(
        window_id=window_id,
        audio_path=f"prepared/test/windows/{window_id}.wav",
        window_seconds=length / 16_000,
        start_sample=start,
        end_sample=start + length,
        padding_samples=0,
    )
    return AcousticWindowInference(
        window=record,
        window_audio_sha256=(window_id[0] * 64),
        input_num_samples=length,
        scores=scores,
        embeddings_shape=(patches, 1024),
        spectrogram_shape=(frames, 64),
    )


def result(*windows, metadata=None) -> AcousticInferenceResult:
    return AcousticInferenceResult(
        model=metadata or model_metadata(),
        preparation_manifest_path="prepared/test/manifest.json",
        preparation_manifest_sha256="a" * 64,
        raw_audio_sha256="b" * 64,
        clip_id="aggregation-clip",
        windows=tuple(windows),
    )


def policy(score=0.5, **kwargs):
    return aggregation.AcousticAggregationSettings.uniform(score, **kwargs)


def test_maps_each_label_with_max_not_sum_and_records_winning_class():
    values = [(0, 317, 0.55), (0, 390, 0.80), (0, 391, 0.70)]
    output = aggregation.aggregate_acoustic_events(result(window("c-one", 0, 9_600, values)), policy())
    event = next(item for item in output.events if item.label is EventLabel.SIREN)
    contribution = event.contributions[0]
    assert contribution.score == pytest.approx(0.8)
    assert contribution.winning_class == LABEL_MAPPING[EventLabel.SIREN][3]
    assert event.peak_score == pytest.approx(0.8)


@pytest.mark.parametrize("label", tuple(LABEL_MAPPING))
def test_every_project_label_and_all_its_mapped_classes_are_exercised(label):
    classes = LABEL_MAPPING[label]
    values = [(0, item.index, 0.55 + index * 0.05) for index, item in enumerate(classes)]
    output = aggregation.aggregate_acoustic_events(
        result(window("c-label", 0, 9_600, values)), policy()
    )
    event = next(item for item in output.events if item.label is label)
    assert event.contributions[0].winning_class == classes[-1]
    assert event.peak_score == pytest.approx(0.55 + (len(classes) - 1) * 0.05)


def test_overlapping_windows_union_into_one_event_without_score_inflation():
    values = [(0, 390, 0.60), (1, 390, 0.70)]
    first = window("c-first", 0, 16_000, values)
    second = window("d-second", 8_000, 16_000, [(0, 390, 0.80), (1, 390, 0.65)])
    output = aggregation.aggregate_acoustic_events(result(first, second), policy())
    sirens = [item for item in output.events if item.label is EventLabel.SIREN]
    assert len(sirens) == 1
    assert (sirens[0].start_sample, sirens[0].end_sample) == (0, 24_000)
    assert sirens[0].peak_score == pytest.approx(0.8)
    assert len(sirens[0].contributions) == 4
    assert sirens[0].source_window_ids == ("c-first", "d-second")


def test_contribution_and_source_window_order_is_deterministic_for_reversed_inputs():
    early = window("c-early", 0, 16_000, [(0, 390, 0.7), (1, 390, 0.75)])
    late = window("d-late", 8_000, 16_000, [(0, 390, 0.8), (1, 390, 0.85)])
    output = aggregation.aggregate_acoustic_events(result(late, early), policy())
    event = next(item for item in output.events if item.label is EventLabel.SIREN)
    assert [(item.start_sample, item.window_id, item.patch_index) for item in event.contributions] == [
        (0, "c-early", 0),
        (7_680, "c-early", 1),
        (8_000, "d-late", 0),
        (15_680, "d-late", 1),
    ]
    assert event.source_window_ids == ("c-early", "d-late")


def test_repeated_patches_from_one_window_deduplicate_source_window_ids():
    item = window("c-repeat", 0, 16_000, [(0, 390, 0.7), (1, 390, 0.8)])
    event = aggregation.aggregate_acoustic_events(result(item), policy()).events[0]
    assert len(event.contributions) == 2
    assert event.source_window_ids == ("c-repeat",)


def test_tail_patch_support_is_clipped_to_real_window_end():
    item = window("c-tail", 0, 16_000, [(1, 390, 0.8)])
    event = aggregation.aggregate_acoustic_events(result(item), policy()).events[0]
    assert (event.start_sample, event.end_sample) == (7_680, 16_000)
    assert (event.contributions[0].start_sample, event.contributions[0].end_sample) == (
        7_680, 16_000
    )


def test_equal_threshold_is_included_and_below_threshold_is_excluded():
    equal = window("c-equal", 0, 9_600, [(0, 437, 0.5)])
    below = window("d-below", 20_000, 9_600, [(0, 437, np.nextafter(np.float32(0.5), 0))])
    output = aggregation.aggregate_acoustic_events(result(equal, below), policy())
    events = [item for item in output.events if item.label is EventLabel.GLASS_BREAK]
    assert len(events) == 1 and events[0].contributions[0].window_id == "c-equal"


def test_separated_support_remains_separate_without_configured_gap():
    first = window("c-first", 0, 9_600, [(0, 420, 0.9)])
    second = window("d-second", 20_000, 9_600, [(0, 420, 0.8)])
    output = aggregation.aggregate_acoustic_events(result(first, second), policy())
    events = [item for item in output.events if item.label is EventLabel.EXPLOSION]
    assert [(item.start_sample, item.end_sample) for item in events] == [(0, 9_600), (20_000, 29_600)]


def test_exactly_adjacent_support_merges_with_default_zero_gap():
    first = window("c-first", 0, 9_600, [(0, 420, 0.9)])
    second = window("d-second", 9_600, 9_600, [(0, 420, 0.8)])
    events = aggregation.aggregate_acoustic_events(result(second, first), policy()).events
    explosions = [item for item in events if item.label is EventLabel.EXPLOSION]
    assert len(explosions) == 1
    assert (explosions[0].start_sample, explosions[0].end_sample) == (0, 19_200)


def test_gap_merge_is_transitive_across_a_chain_of_contributions():
    items = (
        window("c-first", 0, 9_600, [(0, 420, 0.9)]),
        window("d-middle", 9_700, 9_600, [(0, 420, 0.8)]),
        window("e-last", 19_400, 9_600, [(0, 420, 0.7)]),
    )
    output = aggregation.aggregate_acoustic_events(
        result(*reversed(items)), policy(merge_gap_samples=100)
    )
    explosions = [item for item in output.events if item.label is EventLabel.EXPLOSION]
    assert len(explosions) == 1
    assert (explosions[0].start_sample, explosions[0].end_sample) == (0, 29_000)


@pytest.mark.parametrize("gap,expected", [(10_399, 2), (10_400, 1)])
def test_configured_gap_has_exact_inclusive_boundary(gap, expected):
    first = window("c-first", 0, 9_600, [(0, 420, 0.9)])
    second = window("d-second", 20_000, 9_600, [(0, 420, 0.8)])
    output = aggregation.aggregate_acoustic_events(
        result(first, second), policy(merge_gap_samples=gap)
    )
    assert len([item for item in output.events if item.label is EventLabel.EXPLOSION]) == expected


def test_different_labels_never_merge_into_each_other():
    item = window("c-shared", 0, 9_600, [(0, 390, 0.8), (0, 420, 0.9)])
    output = aggregation.aggregate_acoustic_events(result(item), policy())
    assert [(event.label, event.start_sample, event.end_sample) for event in output.events] == [
        (EventLabel.EXPLOSION, 0, 9_600),
        (EventLabel.SIREN, 0, 9_600),
    ]


def test_tied_classes_choose_lowest_mapping_index_deterministically():
    item = window("c-tie", 0, 9_600, [(0, 317, 0.8), (0, 390, 0.8)])
    output = aggregation.aggregate_acoustic_events(result(item), policy())
    siren = next(event for event in output.events if event.label is EventLabel.SIREN)
    assert siren.contributions[0].winning_class.index == 317


def test_events_are_chronological_with_label_tiebreaker():
    late = window("d-late", 30_000, 9_600, [(0, 390, 0.8)])
    early = window("c-early", 0, 9_600, [(0, 437, 0.8), (0, 420, 0.8)])
    output = aggregation.aggregate_acoustic_events(result(late, early), policy())
    assert [item.label for item in output.events] == [
        EventLabel.EXPLOSION, EventLabel.GLASS_BREAK, EventLabel.SIREN
    ]


def test_summary_preserves_provenance_and_is_json_ready():
    output = aggregation.aggregate_acoustic_events(
        result(window("c-one", 0, 9_600, [(0, 390, 0.8)])), policy()
    )
    summary = json.loads(json.dumps(output.to_summary()))
    assert summary["clip_id"] == "aggregation-clip"
    assert summary["preparation_manifest_sha256"] == "a" * 64
    assert summary["raw_audio_sha256"] == "b" * 64
    assert summary["input_window_count"] == 1 and summary["input_patch_count"] == 1
    assert summary["events"][0]["contributions"][0]["winning_class"]["index"] == 390


def test_result_is_not_public_event_annotation_or_risk_decision():
    output = aggregation.aggregate_acoustic_events(
        result(window("c-one", 0, 9_600, [(0, 390, 0.8)])), policy()
    )
    assert all(not isinstance(item, EventAnnotation) for item in output.events)
    assert not hasattr(output.events[0], "risk_level")


def test_empty_inference_returns_explicit_empty_inventory():
    output = aggregation.aggregate_acoustic_events(result(), policy())
    assert output.events == () and output.input_window_count == 0 and output.input_patch_count == 0


def test_input_score_arrays_are_not_modified():
    item = window("c-one", 0, 9_600, [(0, 390, 0.8)])
    before = item.scores.copy()
    aggregation.aggregate_acoustic_events(result(item), policy())
    assert np.array_equal(item.scores, before)


def test_aggregation_result_event_and_contribution_are_immutable():
    output = aggregation.aggregate_acoustic_events(
        result(window("c-one", 0, 9_600, [(0, 390, 0.8)])), policy()
    )
    with pytest.raises(FrozenInstanceError):
        output.clip_id = "changed"
    with pytest.raises(FrozenInstanceError):
        output.events[0].peak_score = 0.1
    with pytest.raises(FrozenInstanceError):
        output.events[0].contributions[0].score = 0.1


def test_uniform_policy_is_complete_ordered_and_independent():
    first = policy(0.4)
    second = policy(0.7)
    assert tuple(item.label for item in first.thresholds) == tuple(sorted(LABEL_MAPPING, key=lambda x: x.value))
    assert all(item.minimum_score == 0.4 for item in first.thresholds)
    assert all(item.minimum_score == 0.7 for item in second.thresholds)


@pytest.mark.parametrize("change", ["missing", "duplicate", "unordered", "deferred"])
def test_policy_requires_exact_unique_ordered_mapped_label_coverage(change):
    thresholds = list(policy().thresholds)
    if change == "missing":
        thresholds.pop()
    elif change == "duplicate":
        thresholds[-1] = thresholds[0]
    elif change == "unordered":
        thresholds[0], thresholds[1] = thresholds[1], thresholds[0]
    else:
        thresholds[-1] = aggregation.AcousticLabelThreshold(
            label=EventLabel.AMBIENT, minimum_score=0.5
        )
    with pytest.raises(ValidationError):
        aggregation.AcousticAggregationSettings(thresholds=tuple(thresholds))


@pytest.mark.parametrize("score", [-0.1, 1.1, float("nan"), float("inf")])
def test_threshold_rejects_out_of_range_or_nonfinite_values(score):
    with pytest.raises(ValidationError):
        aggregation.AcousticLabelThreshold(label=EventLabel.SIREN, minimum_score=score)


@pytest.mark.parametrize("field,value", [
    ("merge_gap_samples", -1),
    ("merge_gap_samples", True),
    ("max_contributions", 0),
    ("max_contributions", True),
])
def test_policy_rejects_invalid_integer_limits(field, value):
    with pytest.raises(ValidationError):
        policy(**{field: value})


def test_contribution_limit_fails_instead_of_returning_partial_events():
    item = window("c-one", 0, 9_600, [(0, 390, 0.8), (0, 420, 0.8)])
    with pytest.raises(aggregation.AcousticAggregationError) as error:
        aggregation.aggregate_acoustic_events(result(item), policy(max_contributions=1))
    assert error.value.code == "too_many_contributions"


def test_contribution_limit_is_inclusive_at_the_exact_boundary():
    item = window("c-one", 0, 9_600, [(0, 390, 0.8), (0, 420, 0.8)])
    output = aggregation.aggregate_acoustic_events(
        result(item), policy(max_contributions=2)
    )
    assert sum(len(event.contributions) for event in output.events) == 2


@pytest.mark.parametrize("field,value", [
    ("model_id", "other"),
    ("model_version", "other"),
    ("model_handle", "other"),
    ("artifact_sha256", "0" * 64),
    ("class_map_sha256", "0" * 64),
    ("label_mapping_version", "other"),
    ("runtime_distribution", "other"),
    ("runtime_version", "other"),
    ("signature_name", "other"),
    ("input", TensorContract("audio", (None,), "float32")),
    ("outputs", (TensorContract("output_0", (None, 521), "float32"),)),
    ("num_classes", 520),
])
def test_rejects_model_metadata_drift(field, value):
    metadata = replace(model_metadata(), **{field: value})
    with pytest.raises(aggregation.AcousticAggregationError) as error:
        aggregation.aggregate_acoustic_events(result(metadata=metadata), policy())
    assert error.value.code == "model_mismatch"


def test_rejects_duplicate_window_ids():
    one = window("c-same", 0, 9_600)
    two = window("c-same", 20_000, 9_600)
    with pytest.raises(aggregation.AcousticAggregationError) as error:
        aggregation.aggregate_acoustic_events(result(one, two), policy())
    assert error.value.code == "invalid_inference"


@pytest.mark.parametrize("change", ["dtype", "shape", "nan", "low", "high", "input", "embedding", "spectrogram"])
def test_rejects_malformed_inference_arrays_and_shapes(change):
    item = window("c-one", 0, 16_000)
    if change == "dtype":
        object.__setattr__(item, "scores", item.scores.astype(np.float64))
    elif change == "shape":
        object.__setattr__(item, "scores", item.scores[:-1])
    elif change == "nan":
        item.scores[0, 0] = np.nan
    elif change == "low":
        item.scores[0, 0] = -0.1
    elif change == "high":
        item.scores[0, 0] = 1.1
    elif change == "input":
        object.__setattr__(item, "input_num_samples", 9_600)
    elif change == "embedding":
        object.__setattr__(item, "embeddings_shape", (1, 1024))
    else:
        object.__setattr__(item, "spectrogram_shape", (96, 64))
    with pytest.raises(aggregation.AcousticAggregationError) as error:
        aggregation.aggregate_acoustic_events(result(item), policy())
    assert error.value.code == "invalid_inference"


@pytest.mark.parametrize("span", [((-1, 9_600),), ((0, 9_601),), ((10, 10),)])
def test_rejects_patch_support_outside_or_empty_against_real_window(span):
    item = window("c-one", 0, 9_600)
    malformed = SimpleNamespace(
        window=item.window,
        window_audio_sha256=item.window_audio_sha256,
        input_num_samples=item.input_num_samples,
        scores=item.scores,
        embeddings_shape=item.embeddings_shape,
        spectrogram_shape=item.spectrogram_shape,
        patch_spans_samples=span,
    )
    with pytest.raises(aggregation.AcousticAggregationError) as error:
        aggregation.aggregate_acoustic_events(result(malformed), policy())
    assert error.value.code == "invalid_inference"
