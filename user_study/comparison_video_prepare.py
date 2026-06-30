#!/usr/bin/env python3
import argparse
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
from moviepy.editor import AudioFileClip, CompositeVideoClip, VideoClip
from PIL import Image, ImageDraw

from user_study import prepare as usp
from user_study.common import (
    CULTURE_DISPLAY_NAME,
    CULTURE_KEY_TO_FOLDER,
    LANGUAGE_NAME_TO_CODE,
    dump_json,
    rel_to,
)


COMPARISON_ORDER = ["real", "no_culture", "adversarial", "fishr"]
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = os.environ.get("SICAGE_DATASET_ROOT", str(REPO_ROOT / "data_root"))
COMPARISON_LABELS = {
    "real": "Real",
    "no_culture": "No Culture",
    "adversarial": "Adversarial",
    "fishr": "Fishr",
}
COMPARISON_LABEL_STYLE = {
    "real": {
        "fill": (28, 92, 155, 230),
        "accent": (170, 216, 255, 255),
        "text": (250, 252, 255, 255),
    },
    "no_culture": {
        "fill": (116, 76, 32, 230),
        "accent": (248, 211, 146, 255),
        "text": (255, 249, 240, 255),
    },
    "adversarial": {
        "fill": (138, 42, 58, 230),
        "accent": (255, 180, 191, 255),
        "text": (255, 246, 248, 255),
    },
    "fishr": {
        "fill": (29, 112, 83, 230),
        "accent": (173, 241, 217, 255),
        "text": (245, 255, 251, 255),
    },
}


def _rounded_box(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int], fill: Tuple[int, int, int, int]) -> None:
    if hasattr(draw, "rounded_rectangle"):
        draw.rounded_rectangle(box, radius=14, fill=fill)
    else:
        draw.rectangle(box, fill=fill)


def _text_bbox(draw: ImageDraw.ImageDraw, text: str, font) -> Tuple[int, int]:
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        return draw.textsize(text, font=font)


def _build_label_overlay(video_size: Tuple[int, int]) -> Image.Image:
    width, height = video_size
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = usp._safe_pil_font("en", 34)

    quad_w = width / 2.0
    quad_h = height / 2.0
    positions = {
        "real": (0.0, 0.0),
        "no_culture": (quad_w, 0.0),
        "adversarial": (0.0, quad_h),
        "fishr": (quad_w, quad_h),
    }

    pad_x = 18
    pad_y = 14
    text_pad_x = 24
    text_pad_y = 12
    accent_w = 10

    for condition in COMPARISON_ORDER:
        qx, qy = positions[condition]
        label = COMPARISON_LABELS[condition].upper()
        style = COMPARISON_LABEL_STYLE[condition]
        tw, th = _text_bbox(draw, label, font)
        x0 = int(qx + pad_x)
        y0 = int(qy + pad_y)
        x1 = x0 + tw + 2 * text_pad_x + accent_w
        y1 = y0 + th + 2 * text_pad_y
        _rounded_box(draw, (x0, y0, x1, y1), fill=style["fill"])
        _rounded_box(draw, (x0, y0, x0 + accent_w, y1), fill=style["accent"])
        tx = x0 + text_pad_x + accent_w
        ty = y0 + text_pad_y - 2
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx != 0 or dy != 0:
                    draw.text((tx + dx, ty + dy), label, font=font, fill=(10, 18, 24, 180))
        draw.text((tx, ty), label, font=font, fill=style["text"])

    return overlay


def _render_comparison_video(
    poses_by_condition: Dict[str, np.ndarray],
    audio_path: Path,
    subtitles: Sequence[Tuple[float, float, str]],
    parent_indices: Sequence[int],
    out_path: Path,
    language: str,
    fps: int = usp.TARGET_FPS,
    line_width: float = 7.5,
    camera_elev: int = 10,
    camera_azim: int = 45,
) -> None:
    import matplotlib.pyplot as plt
    from moviepy.video.io.bindings import mplfig_to_npimage

    fig = plt.figure(figsize=(14.0, 8.8), facecolor="white")
    grid = fig.add_gridspec(
        2,
        2,
        left=0.015,
        right=0.985,
        bottom=0.02,
        top=0.985,
        wspace=0.03,
        hspace=0.04,
    )

    axes = {}
    line_objs = {}
    for idx, condition in enumerate(COMPARISON_ORDER):
        row = idx // 2
        col = idx % 2
        ax = fig.add_subplot(grid[row, col], projection="3d")
        ax.set_axis_off()
        ax.view_init(elev=camera_elev, azim=camera_azim)
        ax.set_box_aspect([1, 1, 1])
        axes[condition] = ax

    all_poses = np.concatenate([poses_by_condition[name] for name in COMPARISON_ORDER], axis=0)
    all_xyz = all_poses.reshape(-1, 3)
    mins, maxs = all_xyz.min(axis=0), all_xyz.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max((maxs - mins).max() / 2.0, 1e-3)
    zoom_radius = max(radius * 0.62, 1e-3)

    palette = plt.get_cmap("tab10").colors
    for condition, ax in axes.items():
        ax.set_xlim(center[0] - zoom_radius, center[0] + zoom_radius)
        ax.set_ylim(center[1] - zoom_radius, center[1] + zoom_radius)
        ax.set_zlim(center[2] - zoom_radius, center[2] + zoom_radius)
        line_objs[condition] = [
            ax.plot([], [], [], lw=line_width, color=palette[i % len(palette)], solid_capstyle="round")[0]
            for i in range(len(parent_indices))
        ]

    total_frames = max(poses_by_condition[name].shape[0] for name in COMPARISON_ORDER)
    label_overlay = None

    def make_frame(t: float):
        nonlocal label_overlay
        idx = min(int(t * fps), total_frames - 1)
        for condition in COMPARISON_ORDER:
            p = poses_by_condition[condition][idx]
            for joint_idx, parent in enumerate(parent_indices):
                if parent < 0:
                    continue
                line = line_objs[condition][joint_idx]
                line.set_data([p[joint_idx, 0], p[parent, 0]], [p[joint_idx, 1], p[parent, 1]])
                line.set_3d_properties([p[joint_idx, 2], p[parent, 2]])

        frame = mplfig_to_npimage(fig)
        if label_overlay is None:
            label_overlay = _build_label_overlay((frame.shape[1], frame.shape[0]))
        frame_rgba = Image.fromarray(frame).convert("RGBA")
        frame_rgba.alpha_composite(label_overlay)
        return np.asarray(frame_rgba.convert("RGB"))

    audio_clip = AudioFileClip(str(audio_path))
    duration = audio_clip.duration
    core = VideoClip(make_frame, duration=duration).set_fps(fps).set_audio(audio_clip)

    subtitle_clips = []
    for s, e, top_line, bottom_line in usp._build_youtube_caption_states(subtitles):
        s_clamped = max(0.0, s)
        e_clamped = min(duration, e)
        if e_clamped - s_clamped < 0.08:
            continue
        clip = usp._make_caption_window_clip(
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


def _load_native_subtitles(
    candidate: usp.ClipCandidate,
    clip_start_abs_sec: float,
    clip_end_abs_sec: float,
) -> Tuple[List[Tuple[float, float, str]], str]:
    lang_code = LANGUAGE_NAME_TO_CODE[candidate.language_name]
    rows = usp._parse_subtitles(candidate.native_subs_path)
    clipped = usp._clip_subtitles(rows, clip_start_abs_sec, clip_end_abs_sec)
    return clipped, lang_code


def _prepare_comparison_assets(
    sequence_id: str,
    selected: usp.SelectedClip,
    dataset_root: Path,
    comparison_dir: Path,
    audio_dir: Path,
    bundles: Dict[str, usp.ModelBundle],
    vqvae_model,
    vqvae_config,
    audio_model,
    audio_processor,
    text_model,
    tokenizer,
    clip_duration_sec: float,
    regenerate_existing: bool,
) -> Dict[str, object]:
    cand = selected.candidate
    base_name = f"{sequence_id}_{cand.culture_key}_{cand.speaker}"
    video_path = comparison_dir / cand.culture_key / f"{base_name}_comparison.mp4"
    audio_path = audio_dir / f"{base_name}.wav"

    if not regenerate_existing and video_path.exists() and audio_path.exists():
        return {
            "sequence_id": sequence_id,
            "culture": cand.culture_key,
            "culture_display": CULTURE_DISPLAY_NAME[cand.culture_key],
            "speaker": cand.speaker,
            "language": cand.language_name,
            "clip_start_abs_sec": selected.clip_start_abs_sec,
            "clip_end_abs_sec": selected.clip_end_abs_sec,
            "source_motion_path": str(cand.motion_path),
            "video_path": str(video_path),
            "audio_path": str(audio_path),
            "subtitle_source": "native",
            "models": {name: COMPARISON_LABELS[name] for name in COMPARISON_ORDER},
        }

    motion_payload = usp._safe_load_pickle(cand.motion_path)
    info_raw = usp._safe_load_pickle(cand.info_path)
    info_data = info_raw.get("meta_info", info_raw) if isinstance(info_raw, dict) else info_raw
    links_len = (
        np.asarray(usp._safe_load_pickle(cand.links_len_path)).squeeze()
        if cand.links_len_path and cand.links_len_path.exists()
        else np.ones((len(usp.KEYPOINT_INDICES),), dtype=np.float32)
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

    resampled_clip_motion = usp.resample_motion(clip_motion, cand.pose_fps, usp.TARGET_FPS)
    target_frames = int(round(clip_duration_sec * usp.TARGET_FPS))
    real_positions = usp._fit_pose_length(
        usp._decode_6d_to_positions(resampled_clip_motion, info_data, links_len, usp.KEYPOINT_INDICES),
        target_frames,
    )

    playlist_path = dataset_root / cand.culture_folder
    audio_full = usp.pr_audio.load_audio(str(playlist_path), cand.speaker, target_sr=usp.TARGET_SR)
    clip_audio = usp._to_clip_audio(audio_full, selected.clip_start_abs_sec, clip_duration_sec, sr=usp.TARGET_SR)
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(audio_path), clip_audio, usp.TARGET_SR)

    text_data = usp._load_text_for_generation(cand)
    subtitle_rows, subtitle_language = _load_native_subtitles(
        candidate=cand,
        clip_start_abs_sec=selected.clip_start_abs_sec,
        clip_end_abs_sec=selected.clip_end_abs_sec,
    )

    windows = usp._prepare_conditioning_windows(
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

    poses_by_condition: Dict[str, np.ndarray] = {
        "real": real_positions,
    }
    for model_name in ["no_culture", "adversarial", "fishr"]:
        gen_codebooks = usp._run_generation_for_model(bundles[model_name], windows)
        poses_by_condition[model_name] = usp._fit_pose_length(
            usp._decode_generated_codebooks(
                generated_codebooks=gen_codebooks,
                vqvae_model=vqvae_model,
                info_data=info_data,
                links_len=links_len,
                keypoint_indices=usp.KEYPOINT_INDICES,
            ),
            target_frames,
        )

    parent_indices = usp._create_parent_indices(info_data, usp.KEYPOINT_INDICES)
    _render_comparison_video(
        poses_by_condition=poses_by_condition,
        audio_path=audio_path,
        subtitles=subtitle_rows,
        parent_indices=parent_indices,
        out_path=video_path,
        language=subtitle_language,
        fps=usp.TARGET_FPS,
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
        "video_path": str(video_path),
        "audio_path": str(audio_path),
        "subtitle_source": "native",
        "models": {name: COMPARISON_LABELS[name] for name in COMPARISON_ORDER},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create per-culture comparison videos for real, no-culture 001, adversarial 001, and fishr 001."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=DEFAULT_DATASET_ROOT,
        help="Root folder containing culture subfolders and full_dataset.",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for videos and metadata.")
    parser.add_argument("--clip-duration-sec", type=float, default=20.0)
    parser.add_argument("--min-sequence-duration-sec", type=float, default=20.0)
    parser.add_argument("--sequences-per-culture", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--no-culture-run-dir", type=str, default=str(REPO_ROOT / "mdm_runs/no_culture_500k_001"))
    parser.add_argument("--fishr-run-dir", type=str, default=str(REPO_ROOT / "mdm_runs/fishr_culclI_500k_001"))
    parser.add_argument("--adversarial-run-dir", type=str, default=str(REPO_ROOT / "mdm_runs/adv_culclI_500k_001"))
    parser.add_argument("--no-culture-checkpoint", type=str, default=None)
    parser.add_argument("--fishr-checkpoint", type=str, default=None)
    parser.add_argument("--adversarial-checkpoint", type=str, default=None)
    parser.add_argument("--regenerate-existing", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    repo_root = usp._resolve_repo_root()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    comparison_dir = output_dir / "comparison_videos"
    audio_dir = output_dir / "audio"
    metadata_dir = output_dir / "metadata"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    min_duration_sec = max(float(args.min_sequence_duration_sec), float(args.clip_duration_sec))
    print(f"[1/5] Scanning dataset for >= {min_duration_sec:.1f}s valid scenes...")
    candidates_by_culture = usp._scan_candidates(dataset_root, min_duration_sec=min_duration_sec)
    for culture_key in sorted(candidates_by_culture.keys()):
        count = len(candidates_by_culture[culture_key])
        print(f"  - {culture_key}: {count} candidates")
        if count < args.sequences_per_culture:
            raise RuntimeError(
                f"Not enough candidates for {culture_key}. Need at least "
                f"{args.sequences_per_culture}, found {count}."
            )

    print("[2/5] Selecting clips...")
    selected_by_culture: Dict[str, List[usp.SelectedClip]] = {}
    for culture_key in sorted(CULTURE_KEY_TO_FOLDER.keys()):
        culture_candidates = candidates_by_culture[culture_key]
        selected = usp._pick_diverse(
            culture_candidates,
            n=args.sequences_per_culture,
            rng=rng,
        )
        selected_by_culture[culture_key] = [
            usp._sample_clip_offset(candidate, args.clip_duration_sec, rng) for candidate in selected
        ]
        print(f"  - {culture_key}: selected {len(selected_by_culture[culture_key])} clips")

    print("[3/5] Resolving checkpoints and loading models...")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    no_culture_run_dir = usp._resolve_local_path(args.no_culture_run_dir, repo_root)
    fishr_run_dir = usp._resolve_local_path(args.fishr_run_dir, repo_root)
    adversarial_run_dir = usp._resolve_local_path(args.adversarial_run_dir, repo_root)

    no_culture_ckpt = usp._resolve_best_checkpoint(
        no_culture_run_dir,
        usp._resolve_local_path(args.no_culture_checkpoint, repo_root) if args.no_culture_checkpoint else None,
    )
    fishr_ckpt = usp._resolve_best_checkpoint(
        fishr_run_dir,
        usp._resolve_local_path(args.fishr_checkpoint, repo_root) if args.fishr_checkpoint else None,
    )
    adversarial_ckpt = usp._resolve_best_checkpoint(
        adversarial_run_dir,
        usp._resolve_local_path(args.adversarial_checkpoint, repo_root) if args.adversarial_checkpoint else None,
    )

    print(f"  - no_culture checkpoint: {no_culture_ckpt}")
    print(f"  - fishr checkpoint: {fishr_ckpt}")
    print(f"  - adversarial checkpoint: {adversarial_ckpt}")

    bundles = {
        "no_culture": usp._load_mdm_bundle(
            name="no_culture",
            run_dir=no_culture_run_dir,
            checkpoint_path=no_culture_ckpt,
            repo_root=repo_root,
            device=device,
        ),
        "fishr": usp._load_mdm_bundle(
            name="fishr",
            run_dir=fishr_run_dir,
            checkpoint_path=fishr_ckpt,
            repo_root=repo_root,
            device=device,
        ),
        "adversarial": usp._load_mdm_bundle(
            name="adversarial",
            run_dir=adversarial_run_dir,
            checkpoint_path=adversarial_ckpt,
            repo_root=repo_root,
            device=device,
        ),
    }

    reference_selected = next(iter(next(iter(selected_by_culture.values()))))
    reference_candidate = reference_selected.candidate
    vqvae_config_path = usp._resolve_local_path(bundles["no_culture"].args.vqvae_config_path, repo_root)
    vqvae_config = usp._build_vqvae_config(
        repo_root=repo_root,
        vqvae_config_path=vqvae_config_path,
        device=device,
        reference_links_path=reference_candidate.links_len_path,
        reference_info_path=reference_candidate.info_path,
    )
    vqvae_model = usp._load_vqvae_model(vqvae_config, device=device)

    audio_model, audio_processor = usp._load_audio_model_for_features()
    text_model, tokenizer = usp._load_text_model_for_features()

    print("[4/5] Generating comparison videos...")
    sequence_entries: List[Dict[str, object]] = []
    for culture_key in sorted(selected_by_culture.keys()):
        for idx, selected in enumerate(selected_by_culture[culture_key]):
            sequence_id = f"{culture_key}_{idx:02d}"
            cand = selected.candidate
            print(
                f"  - {sequence_id} {cand.culture_key}/{cand.speaker} "
                f"({cand.motion_path.name})"
            )
            entry = _prepare_comparison_assets(
                sequence_id=sequence_id,
                selected=selected,
                dataset_root=dataset_root,
                comparison_dir=comparison_dir,
                audio_dir=audio_dir,
                bundles=bundles,
                vqvae_model=vqvae_model,
                vqvae_config=vqvae_config,
                audio_model=audio_model,
                audio_processor=audio_processor,
                text_model=text_model,
                tokenizer=tokenizer,
                clip_duration_sec=args.clip_duration_sec,
                regenerate_existing=args.regenerate_existing,
            )
            sequence_entries.append(entry)

    print("[5/5] Writing metadata...")
    manifest = {
        "version": 1,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "output_root": "..",
        "clip_duration_sec": float(args.clip_duration_sec),
        "min_sequence_duration_sec": float(args.min_sequence_duration_sec),
        "sequences_per_culture": int(args.sequences_per_culture),
        "seed": int(args.seed),
        "cultures": [
            {"key": ck, "display_name": CULTURE_DISPLAY_NAME[ck], "folder": CULTURE_KEY_TO_FOLDER[ck]}
            for ck in sorted(CULTURE_KEY_TO_FOLDER.keys())
        ],
        "models": {
            "no_culture": {
                "display_name": COMPARISON_LABELS["no_culture"],
                "run_dir": rel_to(repo_root, no_culture_run_dir),
                "checkpoint_path": rel_to(repo_root, no_culture_ckpt),
            },
            "adversarial": {
                "display_name": COMPARISON_LABELS["adversarial"],
                "run_dir": rel_to(repo_root, adversarial_run_dir),
                "checkpoint_path": rel_to(repo_root, adversarial_ckpt),
            },
            "fishr": {
                "display_name": COMPARISON_LABELS["fishr"],
                "run_dir": rel_to(repo_root, fishr_run_dir),
                "checkpoint_path": rel_to(repo_root, fishr_ckpt),
            },
        },
        "sequences": [
            {
                **entry,
                "source_motion_path": rel_to(dataset_root, Path(entry["source_motion_path"])),
                "video_path": rel_to(output_dir, Path(entry["video_path"])),
                "audio_path": rel_to(output_dir, Path(entry["audio_path"])),
                "video_relpath": rel_to(output_dir, Path(entry["video_path"])),
                "audio_relpath": rel_to(output_dir, Path(entry["audio_path"])),
            }
            for entry in sequence_entries
        ],
    }
    manifest_path = metadata_dir / "comparison_manifest.json"
    dump_json(manifest_path, manifest)
    print(f"Comparison manifest saved to: {manifest_path}")
    print(f"Comparison videos saved under: {comparison_dir}")
    print("Preparation completed.")


if __name__ == "__main__":
    main()
