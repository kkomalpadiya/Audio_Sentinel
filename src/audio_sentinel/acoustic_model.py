"""A3.1 baseline specification and vocabulary; no model runtime or inference."""

from __future__ import annotations

import csv
import hashlib
import io
from dataclasses import dataclass
from importlib.resources import files
from types import MappingProxyType
from typing import Mapping

from audio_sentinel.contracts import EventLabel


@dataclass(frozen=True)
class AcousticModelSpec:
    model_id: str
    model_handle: str
    reference_revision: str
    class_map_sha256: str
    num_classes: int
    sample_rate_hz: int
    input_layout: str
    input_dtype: str
    input_min: float
    input_max: float
    frontend: str
    patch_frames: int
    mel_bands: int
    patch_hop_samples: int
    patch_support_samples: int
    score_semantics: str


YAMNET = AcousticModelSpec(
    model_id="yamnet-tfhub-1",
    model_handle="https://tfhub.dev/google/yamnet/1",
    reference_revision="d598fb8b23d9cd2fb26b5789b8242de3f494aca7",
    class_map_sha256="cdf24d193e196d9e95912a2667051ae203e92a2ba09449218ccb40ef787c6df2",
    num_classes=521,
    sample_rate_hz=16000,
    input_layout="samples",
    input_dtype="float32",
    input_min=-1.0,
    input_max=1.0,
    frontend="model_owned_waveform_to_log_mel",
    patch_frames=96,
    mel_bands=64,
    patch_hop_samples=7680,
    patch_support_samples=15600,
    score_semantics="independent_sigmoid_uncalibrated",
)

LABEL_MAPPING_VERSION = "1.0"


@dataclass(frozen=True)
class AcousticClass:
    index: int
    mid: str
    display_name: str


# Candidate evidence only. These associations never establish an incident or risk.
LABEL_MAPPING: Mapping[EventLabel, tuple[AcousticClass, ...]] = MappingProxyType({
    EventLabel.SPEECH_PRESENT: (
        AcousticClass(0, "/m/09x0r", "Speech"),
        AcousticClass(1, "/m/0ytgt", "Child speech, kid speaking"),
        AcousticClass(2, "/m/01h8n0", "Conversation"),
        AcousticClass(3, "/m/02qldy", "Narration, monologue"),
        AcousticClass(5, "/m/0brhx", "Speech synthesizer"),
        AcousticClass(12, "/m/02rtxlg", "Whispering"),
    ),
    EventLabel.SIREN: (
        AcousticClass(317, "/m/04qvtq", "Police car (siren)"),
        AcousticClass(318, "/m/012n7d", "Ambulance (siren)"),
        AcousticClass(319, "/m/012ndj", "Fire engine, fire truck (siren)"),
        AcousticClass(390, "/m/03kmc9", "Siren"),
        AcousticClass(391, "/m/0dgbq", "Civil defense siren"),
    ),
    EventLabel.SMOKE_ALARM: (
        AcousticClass(393, "/m/01y3hg", "Smoke detector, smoke alarm"),
        AcousticClass(394, "/m/0c3f7m", "Fire alarm"),
    ),
    EventLabel.GLASS_BREAK: (AcousticClass(437, "/m/07rn7sz", "Shatter"),),
    EventLabel.GUNSHOT: (
        AcousticClass(421, "/m/032s66", "Gunshot, gunfire"),
        AcousticClass(422, "/m/04zjc", "Machine gun"),
    ),
    EventLabel.EXPLOSION: (AcousticClass(420, "/m/014zdl", "Explosion"),),
})

# Explicit coverage for the remaining v1 labels; unmatched classes have no label.
DEFERRED_LABELS: Mapping[EventLabel, str] = MappingProxyType({
    EventLabel.AMBIENT: "Requires a reviewed background decision; unmatched is not safe.",
    EventLabel.NO_SPEECH: "Requires speech gating; low acoustic speech scores are insufficient.",
    EventLabel.NON_THREATENING_SPEECH: "Requires consented speech/language analysis.",
    EventLabel.CROWD_PANIC: "Crowd and screaming alone do not establish panic.",
    EventLabel.DISTRESS_SPEECH: "Requires consented speech/language analysis.",
    EventLabel.THREATENING_SPEECH: "Requires consented speech/language analysis.",
    EventLabel.WEAPON_REFERENCE: "Requires consented speech/language analysis.",
})


def load_class_map() -> tuple[AcousticClass, ...]:
    """Read the bundled official vocabulary offline and reject changed bytes."""
    raw = files("audio_sentinel").joinpath("resources/yamnet_class_map.csv").read_bytes()
    if hashlib.sha256(raw).hexdigest() != YAMNET.class_map_sha256:
        raise ValueError("YAMNet class-map checksum mismatch")
    rows = csv.DictReader(io.StringIO(raw.decode("utf-8")))
    return tuple(
        AcousticClass(int(row["index"]), row["mid"], row["display_name"])
        for row in rows
    )


def validate_label_mapping() -> None:
    """Fail on vocabulary drift or an incomplete/ambiguous project taxonomy."""
    classes = load_class_map()
    if len(classes) != YAMNET.num_classes or tuple(c.index for c in classes) != tuple(
        range(YAMNET.num_classes)
    ):
        raise ValueError("YAMNet vocabulary must contain ordered indexes 0..520")
    if len({c.mid for c in classes}) != len(classes):
        raise ValueError("Duplicate YAMNet class identifier")
    if set(LABEL_MAPPING) & set(DEFERRED_LABELS) or (
        set(LABEL_MAPPING) | set(DEFERRED_LABELS)
    ) != set(EventLabel):
        raise ValueError("Every project label needs exactly one mapping or deferral")
    seen: set[int] = set()
    for targets in LABEL_MAPPING.values():
        if not targets:
            raise ValueError("Mapped label has no model classes")
        for target in targets:
            if not 0 <= target.index < len(classes) or classes[target.index] != target:
                raise ValueError(f"Mapping differs from official vocabulary: {target}")
            if target.index in seen:
                raise ValueError("Model class mapped more than once")
            seen.add(target.index)
