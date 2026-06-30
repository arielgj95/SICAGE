import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


CULTURE_FOLDER_TO_KEY: Dict[str, str] = {
    "indian_ted_hindi_language": "indian",
    "italian_ted_italian_language": "italian",
    "japanese_ted_japanese_language": "japanese",
    "turkish_ted_turkish_language": "turkish",
}

CULTURE_KEY_TO_FOLDER: Dict[str, str] = {
    v: k for k, v in CULTURE_FOLDER_TO_KEY.items()
}

CULTURE_DISPLAY_NAME: Dict[str, str] = {
    "indian": "Indian",
    "italian": "Italian",
    "japanese": "Japanese",
    "turkish": "Turkish",
}

CULTURE_LANGUAGE_NAME: Dict[str, str] = {
    "indian": "hindi",
    "italian": "italian",
    "japanese": "japanese",
    "turkish": "turkish",
}

LANGUAGE_NAME_TO_CODE: Dict[str, str] = {
    "english": "en",
    "hindi": "hi",
    "italian": "it",
    "japanese": "ja",
    "turkish": "tr",
}

CONDITIONS: List[str] = ["real", "no_culture", "fishr", "adversarial"]

QUESTION_DEFS: List[Dict[str, str]] = [
    {
        "id": "coherence_with_speech",
        "title": "Coherence with speech",
        "prompt": "Did the gestures match the meaning of the speech?",
    },
    {
        "id": "appropriateness",
        "title": "Appropriateness",
        "prompt": "Did the gesture shapes at different speech segments feel appropriate?",
    },
    {
        "id": "fluency",
        "title": "Fluency",
        "prompt": "Did the movement look fluid?",
    },
    {
        "id": "timing",
        "title": "Timing",
        "prompt": "Were the speed and timing of the gestures appropriate?",
    },
    {
        "id": "amount_of_gesticulation",
        "title": "Amount of gesticulation",
        "prompt": "Was the amount of gesturing sufficient?",
    },
    {
        "id": "naturalness",
        "title": "Naturalness",
        "prompt": "Did the overall upper-body motion look natural and human-like?",
    },
    {
        "id": "cultural_match",
        "title": "Cultural match",
        "prompt": "Did the gesture style fit the target culture?",
    },
]


@dataclass
class StudyPaths:
    output_dir: Path
    intro_dir: Path
    trial_dir: Path
    metadata_dir: Path
    result_dir: Path
    participant_dir: Path


def build_study_paths(output_dir: Path) -> StudyPaths:
    output_dir = output_dir.resolve()
    paths = StudyPaths(
        output_dir=output_dir,
        intro_dir=output_dir / "intro_videos",
        trial_dir=output_dir / "trial_videos",
        metadata_dir=output_dir / "metadata",
        result_dir=output_dir / "results",
        participant_dir=output_dir / "results" / "participants",
    )
    for p in (
        paths.output_dir,
        paths.intro_dir,
        paths.trial_dir,
        paths.metadata_dir,
        paths.result_dir,
        paths.participant_dir,
    ):
        p.mkdir(parents=True, exist_ok=True)
    return paths


def rel_to(base_dir: Path, path: Path) -> str:
    return str(path.resolve().relative_to(base_dir.resolve()))


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
