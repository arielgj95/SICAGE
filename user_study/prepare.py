#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import random
import re
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Helps avoid OpenMP shared-memory issues on restricted environments.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import yaml
from easydict import EasyDict
from moviepy.editor import AudioFileClip, CompositeVideoClip, VideoClip, VideoFileClip
from moviepy.video.VideoClip import ImageClip
from moviepy.video.io.bindings import mplfig_to_npimage
from PIL import Image, ImageDraw, ImageFont

from TED4CL import process_audio as pr_audio
from TED4CL.process_poses import MotionPreprocessor
from dataset import (
    extract_audio_features,
    extract_gesture_features,
    extract_text_features,
    resample_motion,
    translate_to_english,
)
from mdm_generator.diffusion.utils.model_util import (
    create_gaussian_diffusion as create_hierarchical_diffusion,
)
from mdm_generator.diffusion.utils.model_util import load_model as load_hierarchical_state
from mdm_generator.hierarchical_mdm import Hierarchical_MDM
from transformers import AutoModel, AutoTokenizer, Wav2Vec2ForCTC, Wav2Vec2Processor
from user_study.common import (
    CONDITIONS,
    CULTURE_DISPLAY_NAME,
    CULTURE_FOLDER_TO_KEY,
    CULTURE_KEY_TO_FOLDER,
    CULTURE_LANGUAGE_NAME,
    LANGUAGE_NAME_TO_CODE,
    QUESTION_DEFS,
    build_study_paths,
    dump_json,
    rel_to,
)
from user_study.subtitles import (
    build_youtube_caption_states as _build_youtube_caption_states,
    clip_subtitles as _clip_subtitles,
    parse_subtitles as _parse_subtitles,
)
from vq_vae.vqvae import VQVAE


TARGET_SR = 16000
TARGET_FPS = 15
WINDOW_DURATION_SEC = 5.0
PREFIX_SECONDS = 1.0
PREFIX_CODEBOOKS = 5
KEYPOINT_INDICES = [7, 8, 9, 14, 15, 16, 11, 12, 13]
WAV2VEC_MODEL_NAME = "voidful/wav2vec2-xlsr-multilingual-56"
LABSE_MODEL_NAME = "sentence-transformers/LaBSE"
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
LOCAL_FONT_DIR = THIS_DIR / "fonts"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = os.environ.get("SICAGE_DATASET_ROOT", str(REPO_ROOT / "data_root"))
FONT_PATHS = {
    "en": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "it": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "ja": "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "hi": "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
    "tr": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
}
FONT_CANDIDATES = {
    "en": [
        str(LOCAL_FONT_DIR / "Roboto-Regular.ttf"),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ],
    "it": [
        str(LOCAL_FONT_DIR / "Roboto-Regular.ttf"),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ],
    "tr": [
        str(LOCAL_FONT_DIR / "Roboto-Regular.ttf"),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ],
    "hi": [
        str(LOCAL_FONT_DIR / "NotoSansDevanagari-Regular.ttf"),
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
    ],
    "ja": [
        str(LOCAL_FONT_DIR / "NotoSansJP-Regular.otf"),
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
        "/usr/share/fonts/truetype/fonts-japanese-mincho.ttf",
    ],
}

def _load_audio_model_for_features():
    processor = Wav2Vec2Processor.from_pretrained(WAV2VEC_MODEL_NAME)
    model = Wav2Vec2ForCTC.from_pretrained(WAV2VEC_MODEL_NAME)
    model = model.to("cpu").eval()
    return model, processor


def _load_text_model_for_features():
    tokenizer = AutoTokenizer.from_pretrained(LABSE_MODEL_NAME)
    model = AutoModel.from_pretrained(LABSE_MODEL_NAME)
    model = model.to("cpu").eval()
    return model, tokenizer


@dataclass
class ClipCandidate:
    culture_key: str
    culture_folder: str
    speaker: str
    speaker_dir: Path
    motion_path: Path
    info_path: Path
    links_len_path: Optional[Path]
    video_path: Path
    native_subs_path: Path
    english_subs_path: Optional[Path]
    language_name: str
    scene_fps: float
    pose_fps: float
    real_start: int
    real_end: int
    motion_frames: int
    duration_sec: float


@dataclass
class SelectedClip:
    candidate: ClipCandidate
    offset_sec: float
    clip_start_abs_sec: float
    clip_end_abs_sec: float
    clip_start_scene_frame: int


@dataclass
class ModelBundle:
    name: str
    run_dir: Path
    checkpoint_path: Path
    args: argparse.Namespace
    model: Hierarchical_MDM
    diffusion: object
    device: torch.device


def _safe_load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _resolve_repo_root() -> Path:
    return REPO_ROOT


def _resolve_local_path(path_value: str, repo_root: Path) -> Path:
    p = Path(path_value).expanduser()
    if not p.is_absolute():
        p = (repo_root / p).resolve()
    return p


def _find_info_pickle(speaker_dir: Path, speaker: str) -> Optional[Path]:
    candidates = [
        speaker_dir / f"{speaker}_video_mmpose_data_output_final.pkl",
        speaker_dir / f"{speaker}_mmpose_data_output_final.pkl",
    ]
    for c in candidates:
        if c.exists():
            return c
    any_match = sorted(speaker_dir.glob("*_mmpose_data_output_final.pkl"))
    return any_match[0] if any_match else None


def _find_links_len_for_motion(motion_path: Path) -> Optional[Path]:
    stem = motion_path.stem
    all_motion_dir = motion_path.parent
    suffix = stem.replace("motion_", "")
    candidate = all_motion_dir / f"person_segments_{suffix}.pkl"
    if candidate.exists():
        return candidate
    fallback = sorted(all_motion_dir.glob("person_segments_*.pkl"))
    return fallback[0] if fallback else None


def _subtitle_path(speaker_dir: Path, speaker: str, lang_code: str) -> Path:
    return speaker_dir / f"{speaker}_subtitles_{lang_code}.txt"


def _is_valid_font_file(path: Optional[str]) -> bool:
    if not path or not os.path.isfile(path):
        return False
    try:
        with open(path, "rb") as f:
            head = f.read(8)
        if head.startswith(b"\x00\x01\x00\x00") or head.startswith(b"OTTO") or head.startswith(b"ttcf"):
            return True
        lowered = head.lower()
        if lowered.startswith(b"<!do") or lowered.startswith(b"<html"):
            return False
        return False
    except Exception:
        return False


def _pick_valid_font_path(language: str) -> Optional[str]:
    candidates = list(FONT_CANDIDATES.get(language, []))
    default_font = FONT_PATHS.get(language)
    if default_font:
        candidates.append(default_font)
    candidates.extend(FONT_CANDIDATES.get("en", []))
    for candidate in candidates:
        if _is_valid_font_file(candidate):
            return candidate
    return None


def _safe_pil_font(language: str, size: int):
    font_path = _pick_valid_font_path(language)
    if font_path:
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _make_caption_window_clip(
    top_line: str,
    bottom_line: str,
    start: float,
    end: float,
    video_size: Tuple[int, int],
    language: str = "en",
    fontsize: int = 32,
):
    w, h = video_size
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = _safe_pil_font(language, fontsize)

    raw_lines = [line for line in (top_line, bottom_line) if line]
    if not raw_lines:
        return None

    max_chars = max(18, min(32, int((w * 0.68) / max(fontsize * 0.58, 1.0))))
    lines: List[str] = []
    for line in raw_lines:
        wrapped = textwrap.wrap(
            line,
            width=max_chars,
            break_long_words=False,
            break_on_hyphens=False,
        )
        lines.extend(wrapped if wrapped else [line])

    max_display_lines = 3
    if len(lines) > max_display_lines:
        preserved = lines[: max_display_lines - 1]
        remainder = " ".join(lines[max_display_lines - 1 :])
        lines = preserved + [
            textwrap.shorten(
                remainder,
                width=max_chars,
                placeholder="...",
            )
        ]

    line_heights: List[int] = []
    line_widths: List[int] = []
    for line in lines:
        try:
            tw, th = draw.textsize(line, font=font)
        except Exception:
            bb = draw.textbbox((0, 0), line, font=font)
            tw = bb[2] - bb[0]
            th = bb[3] - bb[1]
        line_heights.append(th)
        line_widths.append(tw)
    line_gap = 10
    padding_x = 20
    padding_y = 14
    total_h = sum(line_heights) + max(0, len(lines) - 1) * line_gap
    box_w = min(int(w * 0.78), max(line_widths) + 2 * padding_x)
    box_h = total_h + 2 * padding_y

    x0 = max(10, (w - box_w) // 2)
    y0 = max(10, h - box_h - 26)
    box = [x0, y0, x0 + box_w, y0 + box_h]
    if hasattr(draw, "rounded_rectangle"):
        draw.rounded_rectangle(box, radius=14, fill=(0, 0, 0, 208))
    else:
        draw.rectangle(box, fill=(0, 0, 0, 208))

    y_text = y0 + padding_y
    for i, line in enumerate(lines):
        x_text = x0 + max(0, (box_w - line_widths[i]) // 2)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx != 0 or dy != 0:
                    draw.text((x_text + dx, y_text + dy), line, font=font, fill=(0, 0, 0, 255))
        draw.text((x_text, y_text), line, font=font, fill=(255, 255, 255, 255))
        y_text += line_heights[i] + line_gap

    arr = np.array(img)
    return ImageClip(arr).set_start(start).set_end(end).set_pos(("center", "bottom"))


def _create_parent_indices(skeleton_info: dict, keypoint_indices: Sequence[int]) -> List[int]:
    links = skeleton_info.get("skeleton_links", [])
    if not links:
        # Fallback tree for selected H36M subset:
        # 7->8->9 and shoulders/elbows/wrists branching from 8.
        links = [
            (7, 8),
            (8, 9),
            (8, 11),
            (11, 12),
            (12, 13),
            (8, 14),
            (14, 15),
            (15, 16),
        ]
    child_to_parent = {}
    for parent, child in links:
        child_to_parent[int(child)] = int(parent)

    parent_indices: List[int] = []
    for kp in keypoint_indices:
        parent = child_to_parent.get(int(kp), -1)
        if parent in keypoint_indices:
            parent_indices.append(keypoint_indices.index(parent))
        else:
            parent_indices.append(-1)
    return parent_indices


def _fit_pose_length(poses: np.ndarray, target_frames: int) -> np.ndarray:
    if poses.shape[0] == target_frames:
        return poses
    if poses.shape[0] > target_frames:
        return poses[:target_frames]
    if poses.shape[0] == 0:
        return np.zeros((target_frames, len(KEYPOINT_INDICES), 3), dtype=np.float32)
    pad = np.repeat(poses[-1:, :, :], target_frames - poses.shape[0], axis=0)
    return np.concatenate([poses, pad], axis=0)


def _render_single_skeleton_video(
    poses: np.ndarray,
    audio_path: Path,
    subtitles: Sequence[Tuple[float, float, str]],
    parent_indices: Sequence[int],
    out_path: Path,
    language: str = "en",
    fps: int = TARGET_FPS,
    line_width: float = 9.0,
    camera_elev: int = 10,
    camera_azim: int = 45,
):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8.4, 8.0), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    ax.set_position([0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    ax.view_init(elev=camera_elev, azim=camera_azim)

    all_xyz = poses.reshape(-1, 3)
    mins, maxs = all_xyz.min(axis=0), all_xyz.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max((maxs - mins).max() / 2.0, 1e-3)
    zoom_radius = max(radius * 0.58, 1e-3)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(center[0] - zoom_radius, center[0] + zoom_radius)
    ax.set_ylim(center[1] - zoom_radius, center[1] + zoom_radius)
    ax.set_zlim(center[2] - zoom_radius, center[2] + zoom_radius)

    palette = plt.get_cmap("tab10").colors
    lines = [
        ax.plot([], [], [], lw=line_width, color=palette[i % len(palette)], solid_capstyle="round")[0]
        for i in range(len(parent_indices))
    ]

    total_frames = poses.shape[0]

    def make_frame(t: float):
        idx = min(int(t * fps), total_frames - 1)
        p = poses[idx]
        for j, parent in enumerate(parent_indices):
            if parent < 0:
                continue
            lines[j].set_data([p[j, 0], p[parent, 0]], [p[j, 1], p[parent, 1]])
            lines[j].set_3d_properties([p[j, 2], p[parent, 2]])
        return mplfig_to_npimage(fig)

    audio_clip = AudioFileClip(str(audio_path))
    duration = audio_clip.duration
    core = VideoClip(make_frame, duration=duration).set_fps(fps).set_audio(audio_clip)

    subtitle_clips = []
    for s, e, top_line, bottom_line in _build_youtube_caption_states(subtitles):
        s_clamped = max(0.0, s)
        e_clamped = min(duration, e)
        if e_clamped - s_clamped < 0.08:
            continue
        clip = _make_caption_window_clip(
            top_line=top_line,
            bottom_line=bottom_line,
            start=s_clamped,
            end=e_clamped,
            video_size=core.size,
            language=language,
        )
        if clip is not None:
            subtitle_clips.append(clip)

    final = CompositeVideoClip([core] + subtitle_clips).set_duration(duration)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final.write_videofile(
        str(out_path),
        codec="libx264",
        audio_codec="aac",
        preset="medium",
        fps=fps,
        threads=max(1, os.cpu_count() or 1),
        logger=None,
    )
    final.close()
    core.close()
    audio_clip.close()
    plt.close(fig)


def _extract_intro_video(source_video: Path, out_path: Path, start_sec: float, duration_sec: float) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clip = VideoFileClip(str(source_video))
    end_sec = min(start_sec + duration_sec, clip.duration)
    if end_sec - start_sec < 1.0:
        raise ValueError(
            f"Invalid intro interval: start={start_sec:.3f}, end={end_sec:.3f}, source={source_video}"
        )
    sub = clip.subclip(start_sec, end_sec)
    sub.write_videofile(
        str(out_path),
        codec="libx264",
        audio_codec="aac",
        preset="medium",
        threads=max(1, os.cpu_count() or 1),
        logger=None,
    )
    clip.close()
    sub.close()


def _scan_candidates(dataset_root: Path, min_duration_sec: float) -> Dict[str, List[ClipCandidate]]:
    all_candidates: Dict[str, List[ClipCandidate]] = {k: [] for k in CULTURE_KEY_TO_FOLDER.keys()}
    for culture_folder, culture_key in CULTURE_FOLDER_TO_KEY.items():
        culture_dir = dataset_root / culture_folder
        if not culture_dir.is_dir():
            continue

        language_name = CULTURE_LANGUAGE_NAME[culture_key]
        native_lang_code = LANGUAGE_NAME_TO_CODE[language_name]
        for speaker_dir in sorted(culture_dir.iterdir()):
            if not speaker_dir.is_dir():
                continue
            speaker = speaker_dir.name
            all_motion_dir = speaker_dir / "all_motion_data"
            if not all_motion_dir.is_dir():
                continue

            info_path = _find_info_pickle(speaker_dir, speaker)
            if info_path is None:
                continue

            video_path = speaker_dir / f"{speaker}_video.mp4"
            if not video_path.exists():
                continue

            native_subs_path = _subtitle_path(speaker_dir, speaker, native_lang_code)
            if not native_subs_path.exists():
                continue
            english_subs_path = _subtitle_path(speaker_dir, speaker, "en")
            english_subs = english_subs_path if english_subs_path.exists() else None

            info_raw = _safe_load_pickle(info_path)
            meta_info = info_raw.get("meta_info", info_raw) if isinstance(info_raw, dict) else {}
            fallback_scene_fps = float(meta_info.get("fps", 0.0) or 0.0)
            fallback_pose_fps = float(meta_info.get("pose_fps", 0.0) or 0.0)

            for motion_path in sorted(all_motion_dir.glob("motion_*.pkl")):
                motion_data = _safe_load_pickle(motion_path)
                if not isinstance(motion_data, dict):
                    continue
                if "data" not in motion_data or "real_start" not in motion_data or "real_end" not in motion_data:
                    continue

                data = np.asarray(motion_data["data"])
                if data.ndim != 3 or data.shape[-2:] != (len(KEYPOINT_INDICES), 6):
                    continue

                scene_fps = float(motion_data.get("scene_fps", 0.0) or fallback_scene_fps)
                pose_fps = float(motion_data.get("pose_fps", 0.0) or fallback_pose_fps)
                if scene_fps <= 0 or pose_fps <= 0:
                    continue

                real_start = int(motion_data["real_start"])
                real_end = int(motion_data["real_end"])
                if real_end <= real_start:
                    continue

                duration_scene = (real_end - real_start) / scene_fps
                duration_pose = data.shape[0] / pose_fps
                duration_sec = min(duration_scene, duration_pose)
                if duration_sec < min_duration_sec:
                    continue

                candidate = ClipCandidate(
                    culture_key=culture_key,
                    culture_folder=culture_folder,
                    speaker=speaker,
                    speaker_dir=speaker_dir,
                    motion_path=motion_path,
                    info_path=info_path,
                    links_len_path=_find_links_len_for_motion(motion_path),
                    video_path=video_path,
                    native_subs_path=native_subs_path,
                    english_subs_path=english_subs,
                    language_name=language_name,
                    scene_fps=scene_fps,
                    pose_fps=pose_fps,
                    real_start=real_start,
                    real_end=real_end,
                    motion_frames=int(data.shape[0]),
                    duration_sec=float(duration_sec),
                )
                all_candidates[culture_key].append(candidate)
    return all_candidates


def _pick_diverse(
    candidates: Sequence[ClipCandidate],
    n: int,
    rng: random.Random,
    excluded_motion_paths: Optional[set] = None,
) -> List[ClipCandidate]:
    excluded_motion_paths = excluded_motion_paths or set()
    by_speaker: Dict[str, List[ClipCandidate]] = {}
    for c in candidates:
        if str(c.motion_path) in excluded_motion_paths:
            continue
        by_speaker.setdefault(c.speaker, []).append(c)

    for speaker_list in by_speaker.values():
        rng.shuffle(speaker_list)
    speaker_order = list(by_speaker.keys())
    rng.shuffle(speaker_order)

    selected: List[ClipCandidate] = []
    progress = True
    while len(selected) < n and progress:
        progress = False
        for speaker in speaker_order:
            if len(selected) >= n:
                break
            pool = by_speaker.get(speaker, [])
            if not pool:
                continue
            selected.append(pool.pop(0))
            progress = True

    if len(selected) < n:
        remaining: List[ClipCandidate] = []
        for c in candidates:
            if str(c.motion_path) in excluded_motion_paths:
                continue
            if c not in selected:
                remaining.append(c)
        rng.shuffle(remaining)
        selected.extend(remaining[: n - len(selected)])

    return selected[:n]


def _sample_clip_offset(candidate: ClipCandidate, clip_duration_sec: float, rng: random.Random) -> SelectedClip:
    max_offset = max(0.0, candidate.duration_sec - clip_duration_sec)
    offset = rng.uniform(0.0, max_offset) if max_offset > 0 else 0.0
    clip_start_abs = candidate.real_start / candidate.scene_fps + offset
    clip_end_abs = clip_start_abs + clip_duration_sec
    clip_start_scene_frame = int(round(clip_start_abs * candidate.scene_fps))
    return SelectedClip(
        candidate=candidate,
        offset_sec=float(offset),
        clip_start_abs_sec=float(clip_start_abs),
        clip_end_abs_sec=float(clip_end_abs),
        clip_start_scene_frame=clip_start_scene_frame,
    )


def _resolve_best_checkpoint(run_dir: Path, explicit_checkpoint: Optional[Path]) -> Path:
    if explicit_checkpoint is not None:
        if not explicit_checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {explicit_checkpoint}")
        return explicit_checkpoint.resolve()

    fgd_files = sorted(run_dir.rglob("validation_fgd_scores.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for fgd_path in fgd_files:
        try:
            payload = json.loads(fgd_path.read_text(encoding="utf-8"))
            ckpt = payload.get("best_checkpoint_path")
            if ckpt:
                ckpt_path = Path(ckpt)
                if ckpt_path.exists():
                    return ckpt_path.resolve()
        except Exception:
            continue

    eval_meta_files = sorted(
        run_dir.rglob("evaluation_run_metadata.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for meta_path in eval_meta_files:
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            ckpt = payload.get("selected_checkpoint_path")
            if ckpt:
                ckpt_path = Path(ckpt)
                if ckpt_path.exists():
                    return ckpt_path.resolve()
        except Exception:
            continue

    models = sorted(run_dir.glob("model*.pt"))
    if not models:
        raise FileNotFoundError(f"No checkpoint found under {run_dir}")

    def _step(path: Path) -> int:
        m = re.search(r"model(\d+)\.pt", path.name)
        return int(m.group(1)) if m else -1

    models.sort(key=_step)
    return models[-1].resolve()


def _load_run_args(run_dir: Path) -> argparse.Namespace:
    args_path = run_dir / "args.json"
    if not args_path.exists():
        raise FileNotFoundError(f"args.json not found in run dir: {run_dir}")
    with args_path.open("r", encoding="utf-8") as f:
        args_dict = json.load(f)
    return argparse.Namespace(**args_dict)


def _infer_n_train_speakers(info_path: Path) -> int:
    training_map = info_path / "training_speakers_id_map.pkl"
    if training_map.exists():
        data = _safe_load_pickle(training_map)
        if isinstance(data, dict):
            return len(data)
    return 0


def _load_model_checkpoint(
    model: Hierarchical_MDM,
    checkpoint_path: Path,
    device: torch.device,
    use_ema: bool,
    use_culture: bool,
):
    state = torch.load(str(checkpoint_path), map_location=device)
    if isinstance(state, dict) and "model_avg" in state and use_ema:
        model_state = state["model_avg"]
    elif isinstance(state, dict) and "model" in state:
        model_state = state["model"]
    else:
        model_state = state
    load_hierarchical_state(model, model_state, use_culture=use_culture)
    model.eval()


def _load_mdm_bundle(
    name: str,
    run_dir: Path,
    checkpoint_path: Path,
    repo_root: Path,
    device: torch.device,
) -> ModelBundle:
    args = _load_run_args(run_dir)

    culture_config_path = _resolve_local_path(getattr(args, "culture_config_path", "culture_encoder/config.yml"), repo_root)
    with culture_config_path.open("r", encoding="utf-8") as f:
        culture_config = EasyDict(yaml.safe_load(f))

    if getattr(args, "fishr_model_path", ""):
        culture_config.fishr_model_path = getattr(args, "fishr_model_path")
    if getattr(args, "adversarial_checkpoint_path", ""):
        culture_config.adversarial_checkpoint_path = getattr(args, "adversarial_checkpoint_path")
    if getattr(args, "adversarial_model_save_path", ""):
        culture_config.adversarial_model_save_path = getattr(args, "adversarial_model_save_path")

    info_path = _resolve_local_path(getattr(args, "info_path"), repo_root)
    n_train_speakers = _infer_n_train_speakers(info_path)

    model = Hierarchical_MDM(
        vqvae_dim=(25, 512),
        audio_onset_dim=156,
        audio_mel_dim=(156, 64),
        audio_wav2vec_dim=(50, 1024),
        text_embedding_dim=768,
        culture_embedding_dim=512,
        latent_dim=args.latent_dim,
        n_train_speakers=n_train_speakers,
        num_heads=args.heads,
        num_layers=args.layers,
        ff_size=args.ffn_size,
        dropout=0.1,
        activation="gelu",
        culture_embedder_config=culture_config,
        device=str(device),
        dataset_type="_sep_people",
        motion_prefix_len=args.motion_prefix_len,
        audio_prefix_len=args.audio_prefix_len,
        motion_mask_prob=getattr(args, "motion_mask_prob", 0.0),
        audio_mask_prob=getattr(args, "audio_mask_prob", 0.0),
        use_culture=bool(getattr(args, "use_culture", True)),
        use_adversarial=bool(getattr(args, "use_adversarial", False)),
        use_native_attention=bool(getattr(args, "use_native_attention", False)),
    ).to(device)

    diffusion = create_hierarchical_diffusion(args)
    _load_model_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
        use_ema=bool(getattr(args, "use_ema", True)),
        use_culture=bool(getattr(args, "use_culture", True)),
    )

    return ModelBundle(
        name=name,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        args=args,
        model=model,
        diffusion=diffusion,
        device=device,
    )


def _decode_6d_to_positions(
    rot6d: np.ndarray,
    info_data: dict,
    links_len: np.ndarray,
    keypoint_indices: Sequence[int],
) -> np.ndarray:
    safe_info = dict(info_data) if isinstance(info_data, dict) else {}
    if "skeleton_links" not in safe_info:
        safe_info["skeleton_links"] = [
            (7, 8),
            (8, 9),
            (8, 11),
            (11, 12),
            (12, 13),
            (8, 14),
            (14, 15),
            (15, 16),
        ]
    if "fps" not in safe_info:
        safe_info["fps"] = 30.0

    processor = MotionPreprocessor(
        poses=[],
        scene=(0, 0),
        scene_fps=float(safe_info.get("fps", 30.0)),
        skeleton_info=safe_info,
        keypoint_indices=list(keypoint_indices),
        enable_debug_gifs=False,
    )
    processor.create_parent_indices()
    rot_tensor = torch.tensor(rot6d, dtype=torch.float32)
    rot_mats = processor.rotation_6d_to_matrix(rot_tensor).cpu().numpy()
    original = processor.convert_to_original_representation(rot_mats)
    return processor.compute_joint_positions(original, np.asarray(links_len).squeeze())


def _decode_generated_codebooks(
    generated_codebooks: torch.Tensor,
    vqvae_model: torch.nn.Module,
    info_data: dict,
    links_len: np.ndarray,
    keypoint_indices: Sequence[int],
) -> np.ndarray:
    model_core = vqvae_model.module if hasattr(vqvae_model, "module") else vqvae_model
    model_device = next(model_core.parameters()).device
    with torch.no_grad():
        codebook_vectors = generated_codebooks.to(model_device).permute(0, 2, 1)
        zs = model_core.bottleneck.encode([codebook_vectors])
        decoded = model_core.decode(zs)
    decoded = decoded.view(decoded.shape[0], decoded.shape[1], decoded.shape[2] // 6, 6).detach().cpu().numpy()
    return _decode_6d_to_positions(decoded[0], info_data, links_len, keypoint_indices)


def _prepare_conditioning_windows(
    resampled_motion: np.ndarray,
    clip_audio: np.ndarray,
    clip_start_abs_sec: float,
    text_data: Sequence[str],
    audio_model,
    audio_processor,
    text_model,
    tokenizer,
    vqvae_model,
    vqvae_config,
    device: torch.device,
) -> List[Dict[str, np.ndarray]]:
    duration_frames_motion = int(round(WINDOW_DURATION_SEC * TARGET_FPS))
    prefix_frames = int(round(PREFIX_SECONDS * TARGET_FPS))
    generation_frames = duration_frames_motion - prefix_frames

    duration_spectrogram = pr_audio.calc_spectrogram_length_from_motion_length(
        duration_frames_motion, TARGET_FPS, TARGET_SR, 512
    )
    duration_wav2vec = pr_audio.calc_wav2vec2_output_frames(WINDOW_DURATION_SEC, TARGET_SR, 320)
    downsample_wav2vec = 5
    duration_wav2vec = int(np.round(duration_wav2vec / downsample_wav2vec))

    audio_mels = pr_audio.extract_mel_log(clip_audio, TARGET_SR)
    audio_onsets = pr_audio.extract_onsets(clip_audio, TARGET_SR)

    windows: List[Dict[str, np.ndarray]] = []
    num_frames = int(resampled_motion.shape[0])
    real_motion_start_frame = 0
    iteration = 0

    while real_motion_start_frame < num_frames:
        if iteration == 0:
            window_start = real_motion_start_frame
            window_end = min(real_motion_start_frame + duration_frames_motion, num_frames)
        else:
            window_start = real_motion_start_frame - prefix_frames
            window_end = min(real_motion_start_frame + generation_frames, num_frames)

        motion_window = resampled_motion[window_start:window_end]
        if motion_window.shape[0] < duration_frames_motion:
            pad_len = duration_frames_motion - motion_window.shape[0]
            pad = np.zeros((pad_len, motion_window.shape[1], motion_window.shape[2]), dtype=motion_window.dtype)
            motion_window = np.concatenate([motion_window, pad], axis=0)

        abs_start = clip_start_abs_sec + (window_start / TARGET_FPS)
        abs_end = abs_start + WINDOW_DURATION_SEC
        local_audio_start = window_start / TARGET_FPS
        local_audio_end = local_audio_start + WINDOW_DURATION_SEC

        audio_feat = extract_audio_features(
            clip_audio,
            audio_model,
            audio_processor,
            TARGET_SR,
            local_audio_start,
            local_audio_end,
            duration_spectrogram,
            duration_wav2vec,
            audio_mels,
            audio_onsets,
            downsample_wav2vec=downsample_wav2vec,
        )
        text_feat, _ = extract_text_features(text_data, text_model, tokenizer, abs_start, abs_end)
        motion_cb = extract_gesture_features(motion_window, vqvae_model, vqvae_config, duration_frames_motion)

        windows.append(
            {
                "motion_cb": motion_cb.astype(np.float32),
                "text_feat": text_feat.astype(np.float32),
                "audio_mel": audio_feat["mel"].astype(np.float32),
                "audio_onset": audio_feat["onset"].astype(np.float32),
                "audio_wav2vec": audio_feat["wav2vec"].astype(np.float32),
                "is_first": np.array([1 if iteration == 0 else 0], dtype=np.int64),
            }
        )

        real_motion_start_frame += duration_frames_motion if iteration == 0 else generation_frames
        iteration += 1

    return windows


def _run_generation_for_model(bundle: ModelBundle, windows: Sequence[Dict[str, np.ndarray]]) -> torch.Tensor:
    all_gen_codebooks: List[torch.Tensor] = []
    previous_generated = None

    for idx, win in enumerate(windows):
        motion_features = (
            torch.tensor(win["motion_cb"], dtype=torch.float32, device=bundle.device)
            .unsqueeze(0)
            .permute(0, 2, 1)
        )
        text_features = torch.tensor(win["text_feat"], dtype=torch.float32, device=bundle.device).unsqueeze(0)
        audio_mels = (
            torch.tensor(win["audio_mel"], dtype=torch.float32, device=bundle.device)
            .unsqueeze(0)
            .permute(0, 2, 1)
        )
        audio_onsets = torch.tensor(win["audio_onset"], dtype=torch.float32, device=bundle.device).unsqueeze(0)
        audio_wav2vec = torch.tensor(win["audio_wav2vec"], dtype=torch.float32, device=bundle.device).unsqueeze(0)

        if idx == 0:
            in_motion = motion_features
        else:
            x_prefix = previous_generated[:, -PREFIX_CODEBOOKS:]
            in_motion = torch.zeros_like(motion_features)
            in_motion[:, :PREFIX_CODEBOOKS] = x_prefix

        batch = [in_motion, text_features, audio_mels, audio_onsets, audio_wav2vec]
        with torch.no_grad():
            new_motion, _ = bundle.diffusion.p_sample_loop(bundle.model, batch, clip_denoised=False)
        previous_generated = new_motion.detach()

        if idx == 0:
            all_gen_codebooks.append(new_motion.detach().cpu())
        else:
            all_gen_codebooks.append(new_motion[:, PREFIX_CODEBOOKS:].detach().cpu())

    return torch.cat(all_gen_codebooks, dim=1)


def _to_clip_audio(
    audio_full: np.ndarray,
    clip_start_abs_sec: float,
    duration_sec: float,
    sr: int = TARGET_SR,
) -> np.ndarray:
    start = max(0, int(round(clip_start_abs_sec * sr)))
    length = int(round(duration_sec * sr))
    end = min(len(audio_full), start + length)
    clip = audio_full[start:end]
    if len(clip) < length:
        clip = np.pad(clip, (0, length - len(clip)), mode="constant")
    return clip.astype(np.float32)


def _load_text_for_generation(candidate: ClipCandidate) -> List[str]:
    with candidate.native_subs_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    return lines


def _load_overlay_subtitles(
    candidate: ClipCandidate,
    clip_start_abs_sec: float,
    clip_end_abs_sec: float,
    use_machine_translation: bool,
    translation_cache: Dict[Tuple[str, str], str],
) -> Tuple[List[Tuple[float, float, str]], str, str]:
    lang_code = LANGUAGE_NAME_TO_CODE[candidate.language_name]
    if candidate.english_subs_path and candidate.english_subs_path.exists():
        rows = _parse_subtitles(candidate.english_subs_path)
        return _clip_subtitles(rows, clip_start_abs_sec, clip_end_abs_sec), "english_subtitles", "en"

    rows = _parse_subtitles(candidate.native_subs_path)
    clipped = _clip_subtitles(rows, clip_start_abs_sec, clip_end_abs_sec)

    if not use_machine_translation:
        return clipped, "native_subtitles_no_translation", lang_code

    translated: List[Tuple[float, float, str]] = []
    source = "machine_translation"
    for s, e, txt in clipped:
        cache_key = (lang_code, txt)
        if cache_key in translation_cache:
            new_txt = translation_cache[cache_key]
        else:
            translated_txt, src = translate_to_english(txt, lang_code, enabled=True)
            if src not in ("machine_translation", "identity_en"):
                source = src
            new_txt = translated_txt
            translation_cache[cache_key] = new_txt
        translated.append((s, e, new_txt))
    return translated, source, "en"


def _build_vqvae_config(
    repo_root: Path,
    vqvae_config_path: Path,
    device: torch.device,
    reference_links_path: Optional[Path],
    reference_info_path: Optional[Path],
) -> EasyDict:
    with vqvae_config_path.open("r", encoding="utf-8") as f:
        cfg = EasyDict(yaml.safe_load(f))
    cfg.checkpoint_path = str(_resolve_local_path(str(cfg.checkpoint_path), repo_root))
    if reference_links_path is not None:
        cfg.speaker_link_len = str(reference_links_path)
    if reference_info_path is not None:
        cfg.skeleton_info_path = str(reference_info_path)
    if device.type == "cuda":
        gpu_idx = str(device.index if device.index is not None else 0)
    else:
        gpu_idx = "0"
    cfg.gpu = gpu_idx
    cfg.no_cuda = [gpu_idx]
    return cfg


def _load_vqvae_model(vqvae_config: EasyDict, device: torch.device) -> torch.nn.Module:
    model = VQVAE(vqvae_config.VQVAE, 9 * 6)
    use_dataparallel = device.type == "cuda" and len(getattr(vqvae_config, "no_cuda", [])) > 1
    if use_dataparallel:
        device_ids = [int(x) for x in getattr(vqvae_config, "no_cuda", ["0"])]
        model = nn.DataParallel(model, device_ids=device_ids)
    model = model.to(device)

    checkpoint = torch.load(str(vqvae_config.checkpoint_path), map_location=device)
    state_dict = checkpoint.get("model_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported VQ-VAE checkpoint format: {vqvae_config.checkpoint_path}")

    model_is_dp = isinstance(model, nn.DataParallel)
    state_has_module_prefix = any(str(k).startswith("module.") for k in state_dict.keys())
    if model_is_dp and not state_has_module_prefix:
        state_dict = {f"module.{k}": v for k, v in state_dict.items()}
    elif (not model_is_dp) and state_has_module_prefix:
        state_dict = {(k[7:] if str(k).startswith("module.") else k): v for k, v in state_dict.items()}

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            print(f"[VQ-VAE] Missing keys: {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            print(f"[VQ-VAE] Unexpected keys: {incompatible.unexpected_keys}")

    model.eval()
    return model


def _prepare_sequence_assets(
    sequence_id: str,
    selected: SelectedClip,
    dataset_root: Path,
    trial_dir: Path,
    bundles: Dict[str, ModelBundle],
    vqvae_model,
    vqvae_config,
    audio_model,
    audio_processor,
    text_model,
    tokenizer,
    clip_duration_sec: float,
    regenerate_existing: bool,
    use_machine_translation: bool,
    translation_cache: Dict[Tuple[str, str], str],
) -> Dict[str, object]:
    cand = selected.candidate
    base_name = f"{sequence_id}_{cand.culture_key}_{cand.speaker}"
    audio_clip_path = trial_dir / f"{base_name}_audio.wav"
    video_paths = {cond: trial_dir / f"{base_name}_{cond}.mp4" for cond in CONDITIONS}

    if (
        not regenerate_existing
        and audio_clip_path.exists()
        and all(p.exists() for p in video_paths.values())
    ):
        return {
            "sequence_id": sequence_id,
            "culture": cand.culture_key,
            "culture_display": CULTURE_DISPLAY_NAME[cand.culture_key],
            "speaker": cand.speaker,
            "language": cand.language_name,
            "clip_start_abs_sec": selected.clip_start_abs_sec,
            "clip_end_abs_sec": selected.clip_end_abs_sec,
            "source_motion_path": str(cand.motion_path),
            "videos": {k: str(video_paths[k]) for k in CONDITIONS},
            "audio_path": str(audio_clip_path),
            "subtitle_source": "preexisting_outputs",
        }

    motion_payload = _safe_load_pickle(cand.motion_path)
    info_raw = _safe_load_pickle(cand.info_path)
    info_data = info_raw.get("meta_info", info_raw) if isinstance(info_raw, dict) else info_raw
    links_len = (
        np.asarray(_safe_load_pickle(cand.links_len_path)).squeeze()
        if cand.links_len_path and cand.links_len_path.exists()
        else np.ones((len(KEYPOINT_INDICES),), dtype=np.float32)
    )

    motion_6d = np.asarray(motion_payload["data"], dtype=np.float32)
    clip_offset_sec_from_motion_start = selected.clip_start_abs_sec - (cand.real_start / cand.scene_fps)
    clip_pose_start = int(round(clip_offset_sec_from_motion_start * cand.pose_fps))
    clip_pose_len = int(round(clip_duration_sec * cand.pose_fps))
    clip_pose_start = max(0, min(clip_pose_start, motion_6d.shape[0] - 1))
    clip_pose_end = min(motion_6d.shape[0], clip_pose_start + clip_pose_len)
    clip_motion = motion_6d[clip_pose_start:clip_pose_end]
    if clip_motion.shape[0] == 0:
        raise ValueError(f"Empty clip motion for {cand.motion_path}")

    resampled_clip_motion = resample_motion(clip_motion, cand.pose_fps, TARGET_FPS)
    target_frames = int(round(clip_duration_sec * TARGET_FPS))
    real_positions = _fit_pose_length(
        _decode_6d_to_positions(resampled_clip_motion, info_data, links_len, KEYPOINT_INDICES),
        target_frames,
    )

    playlist_path = dataset_root / cand.culture_folder
    audio_full = pr_audio.load_audio(str(playlist_path), cand.speaker, target_sr=TARGET_SR)
    clip_audio = _to_clip_audio(audio_full, selected.clip_start_abs_sec, clip_duration_sec, sr=TARGET_SR)
    audio_clip_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(audio_clip_path), clip_audio, TARGET_SR)

    text_data = _load_text_for_generation(cand)
    subtitle_rows, subtitle_source, subtitle_language = _load_overlay_subtitles(
        candidate=cand,
        clip_start_abs_sec=selected.clip_start_abs_sec,
        clip_end_abs_sec=selected.clip_end_abs_sec,
        use_machine_translation=use_machine_translation,
        translation_cache=translation_cache,
    )

    windows = _prepare_conditioning_windows(
        resampled_motion=resampled_clip_motion,
        clip_audio=clip_audio,
        clip_start_abs_sec=selected.clip_start_abs_sec,
        text_data=text_data,
        audio_model=audio_model,
        audio_processor=audio_processor,
        text_model=text_model,
        tokenizer=tokenizer,
        vqvae_model=vqvae_model,
        vqvae_config=vqvae_config,
        device=next(vqvae_model.parameters()).device,
    )

    parent_indices = _create_parent_indices(info_data, KEYPOINT_INDICES)
    _render_single_skeleton_video(
        poses=real_positions,
        audio_path=audio_clip_path,
        subtitles=subtitle_rows,
        parent_indices=parent_indices,
        out_path=video_paths["real"],
        language=subtitle_language,
        fps=TARGET_FPS,
    )

    for model_name in ["no_culture", "fishr", "adversarial"]:
        gen_codebooks = _run_generation_for_model(bundles[model_name], windows)
        gen_positions = _fit_pose_length(
            _decode_generated_codebooks(
                generated_codebooks=gen_codebooks,
                vqvae_model=vqvae_model,
                info_data=info_data,
                links_len=links_len,
                keypoint_indices=KEYPOINT_INDICES,
            ),
            target_frames,
        )
        _render_single_skeleton_video(
            poses=gen_positions,
            audio_path=audio_clip_path,
            subtitles=subtitle_rows,
            parent_indices=parent_indices,
            out_path=video_paths[model_name],
            language=subtitle_language,
            fps=TARGET_FPS,
        )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "sequence_id": sequence_id,
        "culture": cand.culture_key,
        "culture_display": CULTURE_DISPLAY_NAME[cand.culture_key],
        "speaker": cand.speaker,
        "language": cand.language_name,
        "clip_start_abs_sec": selected.clip_start_abs_sec,
        "clip_end_abs_sec": selected.clip_end_abs_sec,
        "source_motion_path": str(cand.motion_path),
        "videos": {k: str(video_paths[k]) for k in CONDITIONS},
        "audio_path": str(audio_clip_path),
        "subtitle_source": subtitle_source,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare local assets for the culture gesture user study.")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=DEFAULT_DATASET_ROOT,
        help="Root folder containing culture subfolders and full_dataset.",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for generated study assets.")
    parser.add_argument("--clip-duration-sec", type=float, default=30.0)
    parser.add_argument("--intro-per-culture", type=int, default=2)
    parser.add_argument("--trials-per-culture", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--no-culture-run-dir", type=str, default=str(REPO_ROOT / "mdm_runs/no_culture_500k"))
    parser.add_argument("--fishr-run-dir", type=str, default=str(REPO_ROOT / "mdm_runs/fishr_culclI_500k"))
    parser.add_argument("--adversarial-run-dir", type=str, default=str(REPO_ROOT / "mdm_runs/adv_culclI_500k_noheadloss"))
    parser.add_argument("--no-culture-checkpoint", type=str, default=None)
    parser.add_argument("--fishr-checkpoint", type=str, default=None)
    parser.add_argument("--adversarial-checkpoint", type=str, default=None)
    parser.add_argument("--regenerate-existing", action="store_true")
    parser.add_argument(
        "--translate-subtitles-to-english",
        action="store_true",
        help="Use machine translation when *_subtitles_en.txt is missing.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    repo_root = _resolve_repo_root()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    paths = build_study_paths(Path(args.output_dir))

    min_duration_sec = args.clip_duration_sec
    print(f"[1/6] Scanning dataset for >= {min_duration_sec:.1f}s valid scenes...")
    candidates_by_culture = _scan_candidates(dataset_root, min_duration_sec=min_duration_sec)
    for culture_key in sorted(candidates_by_culture.keys()):
        print(f"  - {culture_key}: {len(candidates_by_culture[culture_key])} candidates")
        if len(candidates_by_culture[culture_key]) < args.intro_per_culture + args.trials_per_culture:
            raise RuntimeError(
                f"Not enough candidates for {culture_key}. Need at least "
                f"{args.intro_per_culture + args.trials_per_culture}, found {len(candidates_by_culture[culture_key])}."
            )

    print("[2/6] Selecting intro and trial clips...")
    intro_selected: List[SelectedClip] = []
    trial_selected: List[SelectedClip] = []
    excluded_motion_paths: set = set()

    for culture_key in sorted(candidates_by_culture.keys()):
        culture_candidates = candidates_by_culture[culture_key]
        intro_candidates = _pick_diverse(
            culture_candidates,
            n=args.intro_per_culture,
            rng=rng,
            excluded_motion_paths=excluded_motion_paths,
        )
        for c in intro_candidates:
            excluded_motion_paths.add(str(c.motion_path))
            intro_selected.append(_sample_clip_offset(c, args.clip_duration_sec, rng))

        trial_candidates = _pick_diverse(
            culture_candidates,
            n=args.trials_per_culture,
            rng=rng,
            excluded_motion_paths=excluded_motion_paths,
        )
        for c in trial_candidates:
            excluded_motion_paths.add(str(c.motion_path))
            trial_selected.append(_sample_clip_offset(c, args.clip_duration_sec, rng))

    print(f"  - intro clips: {len(intro_selected)}")
    print(f"  - trial sequences: {len(trial_selected)}")

    print("[3/6] Generating intro videos...")
    intro_entries: List[Dict[str, object]] = []
    for idx, selected in enumerate(intro_selected):
        cand = selected.candidate
        intro_path = paths.intro_dir / f"intro_{idx:03d}_{cand.culture_key}.mp4"
        if not intro_path.exists() or args.regenerate_existing:
            _extract_intro_video(
                source_video=cand.video_path,
                out_path=intro_path,
                start_sec=selected.clip_start_abs_sec,
                duration_sec=args.clip_duration_sec,
            )
        intro_entries.append(
            {
                "intro_id": f"intro_{idx:03d}",
                "culture": cand.culture_key,
                "culture_display": CULTURE_DISPLAY_NAME[cand.culture_key],
                "speaker": cand.speaker,
                "video_path": str(intro_path),
                "clip_start_abs_sec": selected.clip_start_abs_sec,
                "clip_end_abs_sec": selected.clip_end_abs_sec,
                "source_video_path": str(cand.video_path),
            }
        )

    print("[4/6] Resolving checkpoints and loading models...")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    no_culture_run_dir = _resolve_local_path(args.no_culture_run_dir, repo_root)
    fishr_run_dir = _resolve_local_path(args.fishr_run_dir, repo_root)
    adversarial_run_dir = _resolve_local_path(args.adversarial_run_dir, repo_root)

    no_culture_ckpt = _resolve_best_checkpoint(
        no_culture_run_dir,
        _resolve_local_path(args.no_culture_checkpoint, repo_root) if args.no_culture_checkpoint else None,
    )
    fishr_ckpt = _resolve_best_checkpoint(
        fishr_run_dir,
        _resolve_local_path(args.fishr_checkpoint, repo_root) if args.fishr_checkpoint else None,
    )
    adversarial_ckpt = _resolve_best_checkpoint(
        adversarial_run_dir,
        _resolve_local_path(args.adversarial_checkpoint, repo_root) if args.adversarial_checkpoint else None,
    )

    print(f"  - no_culture checkpoint: {no_culture_ckpt}")
    print(f"  - fishr checkpoint: {fishr_ckpt}")
    print(f"  - adversarial checkpoint: {adversarial_ckpt}")

    bundles = {
        "no_culture": _load_mdm_bundle(
            name="no_culture",
            run_dir=no_culture_run_dir,
            checkpoint_path=no_culture_ckpt,
            repo_root=repo_root,
            device=device,
        ),
        "fishr": _load_mdm_bundle(
            name="fishr",
            run_dir=fishr_run_dir,
            checkpoint_path=fishr_ckpt,
            repo_root=repo_root,
            device=device,
        ),
        "adversarial": _load_mdm_bundle(
            name="adversarial",
            run_dir=adversarial_run_dir,
            checkpoint_path=adversarial_ckpt,
            repo_root=repo_root,
            device=device,
        ),
    }

    reference_candidate = trial_selected[0].candidate
    vqvae_config_path = _resolve_local_path(bundles["no_culture"].args.vqvae_config_path, repo_root)
    vqvae_config = _build_vqvae_config(
        repo_root=repo_root,
        vqvae_config_path=vqvae_config_path,
        device=device,
        reference_links_path=reference_candidate.links_len_path,
        reference_info_path=reference_candidate.info_path,
    )
    vqvae_model = _load_vqvae_model(vqvae_config, device=device)

    # Keep feature extractors on CPU to reduce GPU memory pressure.
    audio_model, audio_processor = _load_audio_model_for_features()
    text_model, tokenizer = _load_text_model_for_features()

    print("[5/6] Generating trial videos (real + 3 model conditions)...")
    translation_cache: Dict[Tuple[str, str], str] = {}
    trial_entries: List[Dict[str, object]] = []
    for idx, selected in enumerate(trial_selected):
        sequence_id = f"seq_{idx:03d}"
        cand = selected.candidate
        print(
            f"  [{idx + 1}/{len(trial_selected)}] {sequence_id} "
            f"{cand.culture_key}/{cand.speaker} ({cand.motion_path.name})"
        )
        seq_entry = _prepare_sequence_assets(
            sequence_id=sequence_id,
            selected=selected,
            dataset_root=dataset_root,
            trial_dir=paths.trial_dir,
            bundles=bundles,
            vqvae_model=vqvae_model,
            vqvae_config=vqvae_config,
            audio_model=audio_model,
            audio_processor=audio_processor,
            text_model=text_model,
            tokenizer=tokenizer,
            clip_duration_sec=args.clip_duration_sec,
            regenerate_existing=args.regenerate_existing,
            use_machine_translation=args.translate_subtitles_to_english,
            translation_cache=translation_cache,
        )
        trial_entries.append(seq_entry)

    print("[6/6] Writing study manifest...")
    manifest = {
        "version": 1,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "study_name": paths.output_dir.name,
        "output_root": "..",
        "clip_duration_sec": float(args.clip_duration_sec),
        "intro_per_culture": int(args.intro_per_culture),
        "trials_per_culture": int(args.trials_per_culture),
        "seed": int(args.seed),
        "cultures": [
            {"key": ck, "display_name": CULTURE_DISPLAY_NAME[ck], "folder": CULTURE_KEY_TO_FOLDER[ck]}
            for ck in sorted(CULTURE_KEY_TO_FOLDER.keys())
        ],
        "conditions": CONDITIONS,
        "questions": QUESTION_DEFS,
        "models": {
            "no_culture": {
                "run_dir": rel_to(repo_root, no_culture_run_dir),
                "checkpoint_path": rel_to(repo_root, no_culture_ckpt),
            },
            "fishr": {
                "run_dir": rel_to(repo_root, fishr_run_dir),
                "checkpoint_path": rel_to(repo_root, fishr_ckpt),
            },
            "adversarial": {
                "run_dir": rel_to(repo_root, adversarial_run_dir),
                "checkpoint_path": rel_to(repo_root, adversarial_ckpt),
            },
        },
        "intro_videos": [
            {
                **entry,
                "video_path": rel_to(paths.output_dir, Path(entry["video_path"])),
                "source_video_path": rel_to(dataset_root, Path(entry["source_video_path"])),
                "video_relpath": rel_to(paths.output_dir, Path(entry["video_path"])),
            }
            for entry in intro_entries
        ],
        "trial_sequences": [
            {
                **entry,
                "source_motion_path": rel_to(dataset_root, Path(entry["source_motion_path"])),
                "audio_path": rel_to(paths.output_dir, Path(entry["audio_path"])),
                "videos": {
                    k: rel_to(paths.output_dir, Path(v)) for k, v in entry["videos"].items()
                },
                "audio_relpath": rel_to(paths.output_dir, Path(entry["audio_path"])),
                "videos_relpath": {
                    k: rel_to(paths.output_dir, Path(v)) for k, v in entry["videos"].items()
                },
            }
            for entry in trial_entries
        ],
    }

    manifest_path = paths.metadata_dir / "study_manifest.json"
    dump_json(manifest_path, manifest)
    print(f"Study manifest saved to: {manifest_path}")
    print(f"Local website assets root: {paths.output_dir}")
    print("Preparation completed.")


if __name__ == "__main__":
    main()
