import numpy as np
from vq_vae.vqvae import VQVAE
import torch.nn as nn
import torch
import json, os, pickle, argparse, yaml
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from easydict import EasyDict
import textwrap
import librosa
from matplotlib.animation import FuncAnimation
from matplotlib import cm               # NEW
import matplotlib.colors as mcolors     # NEW
import matplotlib.pyplot as plt
from moviepy.editor import VideoClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip, TextClip
from moviepy.video.VideoClip import ImageClip
from moviepy.video.io.bindings import mplfig_to_npimage
from deep_translator import GoogleTranslator
from matplotlib.font_manager import FontProperties
from gtts import gTTS
from bark import generate_audio, preload_models

_BARK_PRELOADED = False


def _ensure_bark_models_loaded():
    global _BARK_PRELOADED
    if not _BARK_PRELOADED:
        preload_models()
        _BARK_PRELOADED = True

from PIL import Image, ImageDraw, ImageFont
import soundfile as sf
from itertools import cycle
import re
try:
    import whisper
except Exception:
    whisper = None

from mdm_generator.diffusion.utils.fixseed import fixseed
from mdm_generator.hierarchical_mdm import Hierarchical_MDM
from dataset import prepare_data
from mdm_generator.diffusion.utils.model_util import create_gaussian_diffusion
from mdm_generator.diffusion.utils import dist_util
from mdm_generator.diffusion.resample import create_named_schedule_sampler
from visualize_vqvae_data import sample_generation_from_codebooks
from test_hierachical_mdm import load_and_sync_parameters
from dataset import (load_text_model, load_audio_model, resample_motion,
                     extract_text_features, extract_audio_features, extract_gesture_features)
from TED4CL import process_text as pr_text
from TED4CL import process_audio as pr_audio
from TED4CL import process_poses as pr_motion

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LANGUAGE_ORDER  = ["ja", "hi", "it", "tr"]
LANG_DISPLAY    = {"ja": "Japanese", "hi": "Hindi", "it": "Italian", "tr": "Turkish"}
MODEL_ORDER = ["no_culture", "fishr", "adversarial"]
MODEL_DISPLAY = {
    "no_culture": "No Culture",
    "fishr": "Fishr",
    "adversarial": "Adversarial",
}

LANG_VOICES = {"ja": "v2/ja_speaker_4", "hi": "v2/hi_speaker_5", "it": "v2/it_speaker_4", "tr": "v2/tr_speaker_4", "en":"v2/en_speaker_9"}
VIDEO_FFMPEG_PRESET = "medium"
SAFE_CODEC = "mpeg4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
REPO_ROOT = Path(__file__).resolve().parent
LOCAL_FONT_DIR = REPO_ROOT / "fonts"
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


def _is_valid_font_file(path: Optional[str]) -> bool:
    if not path or not os.path.isfile(path):
        return False
    try:
        with open(path, "rb") as f:
            head = f.read(8)
        # TrueType / OpenType / TTC signatures
        if head.startswith(b"\x00\x01\x00\x00") or head.startswith(b"OTTO") or head.startswith(b"ttcf"):
            return True
        # Common failure in this workspace: HTML downloaded as .ttf/.otf
        lowered = head.lower()
        if lowered.startswith(b"<!do") or lowered.startswith(b"<html"):
            return False
        return False
    except Exception:
        return False


def _pick_valid_font_path(language: str) -> Optional[str]:
    candidates = []
    candidates.extend(FONT_CANDIDATES.get(language, []))
    default_font = FONT_PATHS.get(language)
    if default_font:
        candidates.append(default_font)
    candidates.extend(FONT_CANDIDATES.get("en", []))
    for candidate in candidates:
        if _is_valid_font_file(candidate):
            return candidate
    return None


def _safe_font_properties(language: str, size: int) -> FontProperties:
    font_path = _pick_valid_font_path(language)
    if font_path:
        try:
            return FontProperties(fname=font_path, size=size)
        except Exception:
            pass
    return FontProperties(size=size)


def _safe_pil_font(language: str, size: int):
    font_path = _pick_valid_font_path(language)
    if font_path:
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            pass
    return ImageFont.load_default()

def load_vqvae_model(args, model_path):
    with torch.no_grad():
        model = VQVAE(args.VQVAE, 9 * 6)  # n_joints * n_channels
        if torch.cuda.is_available():
            no_cuda_ids = getattr(args, "no_cuda", [str(getattr(args, "gpu", "0"))])
            model = nn.DataParallel(model, device_ids=[eval(i) for i in no_cuda_ids])
        model = model.to(device)

        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint.get("model_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if not isinstance(state_dict, dict):
            raise ValueError(f"Unsupported VQ-VAE checkpoint format at {model_path}")

        def _has_module_prefix(sd):
            return any(str(k).startswith("module.") for k in sd.keys())

        model_is_dataparallel = isinstance(model, nn.DataParallel)
        state_has_module_prefix = _has_module_prefix(state_dict)
        if model_is_dataparallel and not state_has_module_prefix:
            state_dict = {f"module.{k}": v for k, v in state_dict.items()}
        elif (not model_is_dataparallel) and state_has_module_prefix:
            state_dict = {
                (k[7:] if str(k).startswith("module.") else k): v
                for k, v in state_dict.items()
            }

        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            incompatible = model.load_state_dict(state_dict, strict=False)
            if incompatible.missing_keys:
                print(f"[VQ-VAE] Missing keys while loading checkpoint: {incompatible.missing_keys}")
            if incompatible.unexpected_keys:
                print(f"[VQ-VAE] Unexpected keys while loading checkpoint: {incompatible.unexpected_keys}")
        model.eval()

    return model


def detect_subtitle_overlaps(subtitles):
    """
    Detect overlapping subtitles and assign vertical positions to avoid conflicts.
    Returns a list of (start, end, text, vertical_position) tuples.
    """
    if not subtitles:
        return []

    # Sort subtitles by start time
    sorted_subs = sorted(subtitles, key=lambda x: x[0])

    # Track active subtitle "lanes" - each lane has an end time
    active_lanes = []
    positioned_subs = []

    for start, end, text in sorted_subs:
        # Find the first available lane (one that has ended before this subtitle starts)
        lane_idx = None
        for i, lane_end_time in enumerate(active_lanes):
            if lane_end_time <= start:  # This lane is free
                lane_idx = i
                break

        # If no lane is available, create a new one
        if lane_idx is None:
            lane_idx = len(active_lanes)
            active_lanes.append(end)
        else:
            # Update the end time of the reused lane
            active_lanes[lane_idx] = end

        positioned_subs.append((start, end, text, lane_idx))

        # Clean up expired lanes to keep the list manageable
        active_lanes = [lane_end for lane_end in active_lanes if lane_end > start]

    return positioned_subs


def translate_and_tts(sentence: str, work_dir: str, tts_speed: float = 1.0):
    """Translate **sentence** to JA / HI / IT / TR and synthesise speech for all
    5 languages (incl. English).  Returns a dict:
        {
            'en': {'text': "Hello…", 'audio_path': "work/en.wav"},
            'ja': {…}, …
        }
    """
    os.makedirs(work_dir, exist_ok=True)
    out = {}
    all_subs = {}
    model_whisper = None
    options = dict(beam_size=5, word_timestamps=True)

    if whisper is not None and hasattr(whisper, "load_model"):
        try:
            model_whisper = whisper.load_model("small")  # or tiny/base
        except Exception as e:
            print(f"[ASR] Whisper load failed, using fallback timings. Error: {e}")
    else:
        print("[ASR] OpenAI Whisper API not available in this environment; using fallback timings.")

    def _fallback_word_timestamps(text: str, duration_s: float, lang: str) -> List[Dict[str, float]]:
        txt = str(text).strip()
        words = [w for w in txt.split() if w]

        # Japanese usually has no whitespace, so split into short chunks to spread across the timeline.
        if lang == "ja" and len(words) <= 1:
            cleaned = re.sub(r"[、。！？!?,.]", " ", txt)
            segments = [seg for seg in cleaned.split() if seg]
            chunked = []
            for seg in segments:
                if len(seg) <= 2:
                    chunked.append(seg)
                else:
                    chunked.extend([seg[i:i + 2] for i in range(0, len(seg), 2)])
            words = chunked[:36]

        if not words:
            stripped = "".join(ch for ch in txt if not ch.isspace())
            words = list(stripped)[:24] if stripped else ["..."]
        n = max(1, len(words))
        dur = max(float(duration_s), 1e-3)
        step = dur / n
        return [
            {"start": i * step, "end": min((i + 1) * step, dur), "word": words[i]}
            for i in range(n)
        ]

    for lang in ["en", *LANGUAGE_ORDER]:
        # ----------------- translation -----------------
        translated = sentence if lang == "en" else GoogleTranslator(source='auto', target=lang).translate(sentence)
        print(translated)
        # ----------------- TTS synthesis ---------------
        wav_path = os.path.join(work_dir, f"{lang}.wav")
        #gTTS(text=translated, lang=lang, slow=False).save(wav_path)
        _ensure_bark_models_loaded()
        gen_audio = generate_audio(translated, history_prompt=LANG_VOICES[lang])
        sf.write(wav_path, gen_audio, samplerate=24000)
        duration_s = len(gen_audio) / 24000.0 if len(gen_audio) else 0.0

        word_timestamps = []
        subs = {"segments": []}
        if model_whisper is not None:
            try:
                subs = model_whisper.transcribe(os.path.join(work_dir, f"{lang}.wav"), **options)
                for segment in subs.get("segments", []):
                    word_timestamps.extend(segment.get("words", []))
            except Exception as e:
                print(f"[ASR] Whisper transcription failed for {lang}, using fallback timings. Error: {e}")
                word_timestamps = _fallback_word_timestamps(translated, duration_s, lang)
                subs = {"segments": [{"words": word_timestamps}]}
        else:
            word_timestamps = _fallback_word_timestamps(translated, duration_s, lang)
            subs = {"segments": [{"words": word_timestamps}]}

        new_subs = [
            f"{word_data['start']} - {word_data['end']}: {word_data['word']}"
            for word_data in word_timestamps
        ]
        all_subs[lang] = subs
        out[lang] = {
            "text": translated,
            "audio_path": wav_path,
            "subs": new_subs,
            "word_timestamps": word_timestamps,
        }
    return out


DEFAULT_REUSED_TRANSLATIONS = {
    "en": "This example helps explain the idea of cultural styles",
    "ja": "\u3053\u306e\u4f8b\u306f\u6587\u5316\u7684\u30b9\u30bf\u30a4\u30eb\u306e\u8003\u3048\u65b9\u3092\u8aac\u660e\u3059\u308b\u306e\u306b\u5f79\u7acb\u3061\u307e\u3059",
    "hi": "\u092f\u0939 \u0909\u0926\u093e\u0939\u0930\u0923 \u0938\u093e\u0902\u0938\u094d\u0915\u0943\u0924\u093f\u0915 \u0936\u0948\u0932\u093f\u092f\u094b\u0902 \u0915\u0947 \u0935\u093f\u091a\u093e\u0930 \u0915\u094b \u0938\u092e\u091d\u093e\u0928\u0947 \u092e\u0947\u0902 \u092e\u0926\u0926 \u0915\u0930\u0924\u093e \u0939\u0948",
    "it": "Questo esempio aiuta a spiegare l'idea degli stili culturali",
    "tr": "Bu \u00f6rnek k\u00fclt\u00fcrel tarzlar fikrini a\u00e7\u0131klamaya yard\u0131mc\u0131 olur",
}


def _fallback_word_timestamps_for_text(text: str, duration_s: float, lang: str) -> List[Dict[str, float]]:
    txt = str(text).strip()
    words = [w for w in txt.split() if w]
    if lang == "ja" and len(words) <= 1:
        cleaned = re.sub(r"[、。！？!?,.]", " ", txt)
        segments = [seg for seg in cleaned.split() if seg]
        chunked = []
        for seg in segments:
            if len(seg) <= 2:
                chunked.append(seg)
            else:
                chunked.extend([seg[i:i + 2] for i in range(0, len(seg), 2)])
        words = chunked[:36]
    if not words:
        stripped = "".join(ch for ch in txt if not ch.isspace())
        words = list(stripped)[:24] if stripped else ["..."]
    n = max(1, len(words))
    dur = max(float(duration_s), 1e-3)
    step = dur / n
    return [
        {"start": i * step, "end": min((i + 1) * step, dur), "word": words[i]}
        for i in range(n)
    ]


def _load_reused_audio_io_data(audio_dir: str, languages: List[str], sentence: str) -> Dict[str, dict]:
    io_data = {}
    for lang in ["en", *languages]:
        audio_path = os.path.join(audio_dir, f"{lang}.wav")
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Missing reused audio file: {audio_path}")
        audio, sr = sf.read(audio_path, dtype="float32")
        duration_s = len(audio) / float(sr) if len(audio) else 0.0
        text = sentence if lang == "en" else DEFAULT_REUSED_TRANSLATIONS.get(lang, sentence)
        word_timestamps = _fallback_word_timestamps_for_text(text, duration_s, lang)
        io_data[lang] = {
            "text": text,
            "audio_path": audio_path,
            "subs": [
                f"{word_data['start']} - {word_data['end']}: {word_data['word']}"
                for word_data in word_timestamps
            ],
            "word_timestamps": word_timestamps,
        }
    return io_data


def _infer_speaker_link_len_path(motion_file_path: str) -> Optional[str]:
    motion_path = Path(motion_file_path)
    if not motion_path.name.startswith("motion_"):
        return None
    candidate = motion_path.with_name(motion_path.name.replace("motion_", "person_segments_", 1))
    return str(candidate) if candidate.is_file() else None


# --------------------------------------------------------------------------------
# 2) PREFIX ENCODER (still pose → VQVAE codebooks) --------------------------------
# --------------------------------------------------------------------------------

def encode_motion_prefix(x_pre_path: str, vqvae_model, vqvae_config, fps: int = 15,
                         prefix_seconds: float = 5.0): #as it is 1 sec + noise, we can put 5 seconds and the enxt 4 seconds are replaced with noise
    """Encode a *still* pose (J,3) OR short sequence (T,J,3) to VQ‑VAE codebooks.
    Returns a tensor shaped (1, n_codebooks, 25, 512) suitable as prefix.
    """
    import random

    with open(x_pre_path, "rb") as file:
        motion = pickle.load(file)
        motion = motion["data"]

    x_pre_np = random.choice(motion) #pick a random pose to start. The pose is already in 6d representation
    T = int(prefix_seconds * fps)
    x_pre_np = np.tile(x_pre_np[None, :, :], (T, 1, 1)) # (75,9,3)
    n_pre_codebooks = extract_gesture_features(x_pre_np, vqvae_model, vqvae_config,T) # (512,25)
    #n_pre = n_pre_codebooks.unsqueeze(0).to(device).permute(0, 2, 1)  #(1,25,512)

    return n_pre_codebooks


def generate_motion_sequence(x_prefix_codebooks: torch.Tensor,
                             text_data: list,
                             audio_data: dict,
                             total_poses: int,
                             mdm_model, diffusion, vqvae_model, vqvae_config,
                             duration_spectrogram:int,
                             duration_wav2vec:int,
                             downsample_wav2vec:int,
                             prefix_codebooks: int = 5,
                             prefix_frames: int = 15,
                             generation_frames: int = 60,
                             target_fps: int = 15,
                             target_sr: int = 16000,
                             duration: int = 5):
    """Generate a full motion sequence conditioned on prefix + audio + text."""
    all_subs = []

    # Initialize for continuous generation
    previous_generated_codebooks = None  # Will store generated codebooks for next prefix
    real_motion_start_frame = 0  # Track where we are in real motion
    padding_shape = None
    duration_frames_motion = duration * target_fps
    real_start = 0
    all_real_codebooks = []
    all_gen_codebooks = []


    iteration = 0
    motion_features = x_prefix_codebooks  # initially, there are 75 poses that are all the same
    while real_motion_start_frame < total_poses:
        print(f"\n--- Iteration {iteration + 1} ---")

        # Determine the window for this iteration
        if iteration == 0:
            window_start = real_motion_start_frame
            window_end = min(real_motion_start_frame + duration_frames_motion, total_poses)
            motion_features = torch.tensor(motion_features).unsqueeze(0).to(device).permute(0, 2, 1) #initially, is 1,25,512
            motion_shape = motion_features.shape
            prefix_from_real = True
        else:
            window_start = real_motion_start_frame - prefix_frames
            window_end = min(real_motion_start_frame + generation_frames, total_poses)
            prefix_from_real = False

        #start_time = real_start + (window_start / target_fps)
        #end_time = start_time + duration
        audio_start_sec = (window_start / target_fps)
        end_audio_sec = audio_start_sec + duration
        if iteration == 0:
            end_audio_sec -= 1

        audio_features = extract_audio_features(audio_data['raw'], audio_model, audio_processor,
                                                target_sr, audio_start_sec, end_audio_sec,
                                                duration_spectrogram,
                                                duration_wav2vec,
                                                audio_data['mels'], audio_data['onsets'],
                                                downsample_wav2vec)
        text_features, sentence = extract_text_features(text_data, text_model, tokenizer,
                                                        audio_start_sec, end_audio_sec)

        # Convert to tensors
        text_features = torch.tensor(text_features).to(device).unsqueeze(0)
        audio_mels = torch.tensor(audio_features["mel"]).to(device).unsqueeze(0).permute(0, 2, 1)
        audio_onsets = torch.tensor(audio_features["onset"]).to(device).unsqueeze(0)
        audio_wav2vec = torch.tensor(audio_features["wav2vec"]).to(device).unsqueeze(0)


        all_subs.append((audio_start_sec, end_audio_sec, sentence))

        # Prepare input for generation
        if iteration == 0:
            x_prefix = motion_features[:, :prefix_codebooks]
        else:
            x_prefix = previous_generated_codebooks[:, -prefix_codebooks:]

        #motion_features_modified = motion_features.clone()

        if iteration > 0:
            motion_features = torch.zeros(motion_shape).to(device)
            motion_features[:, :prefix_codebooks] = x_prefix

        batch = (motion_features, text_features, audio_mels, audio_onsets, audio_wav2vec)
        in_data = list(batch)

        new_motion, _ = diffusion.p_sample_loop(mdm_model, in_data, clip_denoised=False)

        previous_generated_codebooks = new_motion.clone()

        # Accumulate codebooks for one-shot decode
        all_real_codebooks.append(motion_features[:, prefix_codebooks:])
        all_gen_codebooks.append(new_motion[:, prefix_codebooks:])

        # Update real motion position
        real_motion_start_frame +=  generation_frames
        iteration += 1

    # After generation: one-shot decode full real & generated streams
    real_cb_seq = torch.cat(all_real_codebooks, dim=1)
    gen_cb_seq  = torch.cat(all_gen_codebooks,  dim=1)
    all_true_poses, all_generated_poses = sample_generation_from_codebooks(
        vqvae_config,
        generated_codebooks=gen_cb_seq,
        real_codebooks=real_cb_seq,
        save_path_real='',
        save_path_fake='',
        plot_poses=False
    )

    return all_generated_poses[0]

def make_quadrant_pose_video(poses_by_lang: dict, english_audio_path: str, subtitles,
                             parent_indices, save_path: str,
                             fps: int = 15, line_width: int = 6,
                             figsize=(10, 10), camera_elev: int = 10, camera_azim: int = 45):
    """Render 2×2 grid video: JA, HI, IT, TR quadrants. English audio & subs."""
    # ------ gather global bounds --------------------------------------------
    any_pose = next(iter(poses_by_lang.values()))
    T = any_pose.shape[0]
    duration = T / fps

    # ------ Matplotlib figure & axes ----------------------------------------
    fig = plt.figure(figsize=figsize, facecolor="white")
    axes, line_objs = {}, {}
    cmap10 = plt.get_cmap("tab10")

    for idx, lang in enumerate(LANGUAGE_ORDER):
        ax = fig.add_subplot(2, 2, idx + 1, projection="3d")
        ax.set_title(LANG_DISPLAY[lang], fontsize=14)
        ax.set_axis_off()
        ax.view_init(elev=camera_elev, azim=camera_azim)
        axes[lang] = ax
        # create per‑bone line objects
        line_objs[lang] = [
            ax.plot([], [], [], lw=line_width, color=cmap10(idx))[0]
            for _ in parent_indices
        ]

    # equalise scales ---------------------------------------------------------
    all_poses = np.concatenate(list(poses_by_lang.values()), axis=0)
    mins, maxs = all_poses.reshape(-1, 3).min(0), all_poses.reshape(-1, 3).max(0)
    centre = (mins + maxs) / 2
    radius = (maxs - mins).max() / 2
    for ax in axes.values():
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlim(centre[0] - radius, centre[0] + radius)
        ax.set_ylim(centre[1] - radius, centre[1] + radius)
        ax.set_zlim(centre[2] - radius, centre[2] + radius)

    # frame generator ---------------------------------------------------------
    def make_frame(t):
        i = min(int(t * fps), T - 1)
        for lang in LANGUAGE_ORDER:
            P = poses_by_lang[lang][i] - centre
            for j, p in enumerate(parent_indices):
                if p < 0:
                    continue
                line = line_objs[lang][j]
                line.set_data([P[j, 0], P[p, 0]], [P[j, 1], P[p, 1]])
                line.set_3d_properties([P[j, 2], P[p, 2]])
        return mplfig_to_npimage(fig)

    # MoviePy assembly --------------------------------------------------------
    core = VideoClip(make_frame, duration=duration).set_fps(fps)
    core = core.set_audio(AudioFileClip(english_audio_path).subclip(0, duration))

    # English subtitles -------------------------------------------------------
    subtitle_clips = [
        make_subtitle_clip(txt, s, e, core.size, vertical_position=0, language="en")
        for s, e, txt in subtitles
    ]

    final = CompositeVideoClip([core, *subtitle_clips]).set_duration(duration)
    final.write_videofile(save_path,
                          codec="libx264",
                          audio_codec="aac",
                          preset=VIDEO_FFMPEG_PRESET,
                          threads=os.cpu_count(),
                          fps=fps,
                          logger="bar")
    plt.close(fig)

def run_multilang_pipeline(sentence: str, x_pre_path: str,
                           audio_model, audio_processor,
                           text_model, tokenizer,
                           mdm_model, diffusion,
                           vqvae_model, vqvae_config,
                           parent_indices,
                           target_sr: int = 16000,
                           target_fps: int = 15,
                           duration: float = 5.0,
                           work_dir: str = "./ml_tmp",
                           io_data: Optional[Dict[str, dict]] = None,
                           x_prefix_codebooks: Optional[np.ndarray] = None,
                           languages: Optional[List[str]] = None):
    """Runs the *whole* multilingual pipeline, returning `poses_by_lang` AND the
    path to the English audio, plus subtitles list."""

    # ---------------- translations & speech -----------------------
    io = io_data if io_data is not None else translate_and_tts(sentence, work_dir)

    # ---------------- encode still prefix -------------------------
    if x_prefix_codebooks is None:
        x_prefix_codebooks = encode_motion_prefix(x_pre_path, vqvae_model, vqvae_config, fps=target_fps)
    duration_frames_motion = int(duration * target_fps)
    duration_sample_spectrogram = pr_audio.calc_spectrogram_length_from_motion_length(duration_frames_motion,
                                                                                      target_fps, target_sr, 512)
    duration_sample_wav2vec = pr_audio.calc_wav2vec2_output_frames(duration, target_sr, 320)
    downsample_wav2vec = 5
    duration_sample_wav2vec = int(np.round(duration_sample_wav2vec / downsample_wav2vec))


    # ---------------- loop over each language ---------------------
    poses_by_lang = {}
    selected_languages = languages or LANGUAGE_ORDER
    for lang in selected_languages:
        # ---- AUDIO ------------------------------------------------
        wav_np, sr = sf.read(io[lang]["audio_path"], dtype="float32")
        if sr != target_sr:
            if wav_np.ndim == 2:
                # transpose to (channels, samples)
                wav_np = wav_np.T
                wav_np = librosa.resample(wav_np, orig_sr=sr, target_sr=target_sr)
                # back to (samples, channels)
                wav_np = wav_np.T
            else:
                wav_np = librosa.resample(wav_np, orig_sr=sr, target_sr=target_sr)
        audio_scene = wav_np
        audio_data_mel = pr_audio.extract_mel_log(audio_scene, target_sr)
        # Check audio-motion alignment
        audio_data_onsets = pr_audio.extract_onsets(audio_scene, target_sr)
        audio_feats = {}
        audio_feats['mels'] = audio_data_mel
        audio_feats['onsets'] = audio_data_onsets
        audio_feats['raw'] = audio_scene
        total_poses = round(len(wav_np)/target_sr * target_fps)

        # ---- MOTION ----------------------------------------------
        pose_seq = generate_motion_sequence(
            x_prefix_codebooks, io[lang]['subs'], audio_feats, total_poses,
            mdm_model, diffusion, vqvae_model, vqvae_config, duration_spectrogram=duration_sample_spectrogram,
        duration_wav2vec = duration_sample_wav2vec, downsample_wav2vec = downsample_wav2vec)
        poses_by_lang[lang] = pose_seq


    # ---------------- subtitles (English only) --------------------
    subtitles = [(0.0, duration, sentence)]

    return poses_by_lang, io #, #subtitles

def _scale_pose_sequence(
    seq: np.ndarray,
    keypoint_indices: List[int],
    *,
    height_scale: float = 1.0,
    shoulder_scale: float = 2.0,
) -> np.ndarray:
    poses = seq.copy()
    centroid = poses.reshape(-1, 3).mean(0)
    poses = (poses - centroid) * height_scale + centroid

    if abs(shoulder_scale - 1.0) > 1e-6:
        try:
            root = poses[:, keypoint_indices.index(8)]
            for side in [(11, 12, 13), (14, 15, 16)]:
                s, e, w = [keypoint_indices.index(x) for x in side]
                for f in range(len(poses)):
                    vec = poses[f, s] - root[f]
                    new_s = root[f] + vec * shoulder_scale
                    delta = new_s - poses[f, s]
                    poses[f, [s, e, w]] += delta
        except ValueError:
            pass
    return poses


def _seconds_and_last_frame_times(duration_sec: float) -> List[float]:
    duration_sec = max(0.0, float(duration_sec))
    if duration_sec <= 0.0:
        return [0.0]

    whole_seconds = [float(i) for i in range(1, int(np.floor(duration_sec)) + 1)]
    if not whole_seconds:
        return [duration_sec]
    if abs(whole_seconds[-1] - duration_sec) > 1e-3:
        whole_seconds.append(duration_sec)
    return whole_seconds


def _parse_word_timestamps(words_or_subs: List) -> List[Dict[str, float]]:
    parsed = []
    if not words_or_subs:
        return parsed

    if isinstance(words_or_subs[0], dict):
        for item in words_or_subs:
            if {"start", "end", "word"}.issubset(item.keys()):
                parsed.append(
                    {
                        "start": float(item["start"]),
                        "end": float(item["end"]),
                        "word": str(item["word"]).strip(),
                    }
                )
        return sorted(parsed, key=lambda x: x["start"])

    pattern = re.compile(r"^\s*([\d.]+)\s*-\s*([\d.]+)\s*:\s*(.+)$")
    for item in words_or_subs:
        m = pattern.match(str(item))
        if not m:
            continue
        parsed.append(
            {
                "start": float(m.group(1)),
                "end": float(m.group(2)),
                "word": m.group(3).strip(),
            }
        )
    return sorted(parsed, key=lambda x: x["start"])


def _compute_pose_center_radius(seqs: List[np.ndarray]) -> Tuple[np.ndarray, float]:
    cloud = np.concatenate([seq.reshape(-1, 3) for seq in seqs], axis=0)
    mins = cloud.min(axis=0)
    maxs = cloud.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max((maxs - mins).max() / 2.0, 1e-3)
    return center, radius


def _draw_snapshot(
    ax,
    poses: np.ndarray,
    parent_indices: List[int],
    frame_idx: int,
    center: np.ndarray,
    radius: float,
    *,
    trail_frames: int,
):
    ax.set_axis_off()
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=10, azim=45)
    zoom_radius = radius * 0.68
    ax.set_xlim(-zoom_radius, zoom_radius)
    ax.set_ylim(-zoom_radius, zoom_radius)
    ax.set_zlim(-zoom_radius, zoom_radius)

    start_idx = max(0, frame_idx - trail_frames)
    denom = max(1, frame_idx - start_idx)
    for k in range(start_idx, frame_idx + 1):
        alpha = 0.12 + 0.88 * ((k - start_idx) / denom)
        lw = 1.5 if k < frame_idx else 2.8
        p = poses[k] - center
        for j, parent in enumerate(parent_indices):
            if parent < 0:
                continue
            ax.plot(
                [p[j, 0], p[parent, 0]],
                [p[j, 1], p[parent, 1]],
                [p[j, 2], p[parent, 2]],
                color=mcolors.to_rgba("#1f77b4", alpha=alpha),
                lw=lw,
                solid_capstyle="round",
            )


def make_culture_comparison_figure(
    culture_lang: str,
    poses_by_model: Dict[str, np.ndarray],
    audio_path: str,
    word_timestamps: List,
    parent_indices: List[int],
    keypoint_indices: List[int],
    out_path: str,
    *,
    fps: int = 15,
    target_sr: int = 16000,
):
    waveform, sr = sf.read(audio_path, dtype="float32")
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if sr != target_sr:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    duration = len(waveform) / float(sr) if len(waveform) > 0 else 0.0
    frame_times = _seconds_and_last_frame_times(duration)
    words = _parse_word_timestamps(word_timestamps)

    model_names = [m for m in MODEL_ORDER if m in poses_by_model]
    if not model_names:
        raise ValueError("No model outputs available to render comparison figure.")

    scaled_poses = {
        model_name: _scale_pose_sequence(
            poses_by_model[model_name],
            keypoint_indices,
            height_scale=1.0,
            shoulder_scale=2.0,
        )
        for model_name in model_names
    }
    center, radius = _compute_pose_center_radius(list(scaled_poses.values()))

    n_rows = len(model_names) + 2
    n_cols = max(1, len(frame_times))
    fig_w = max(11, 2.25 * n_cols)
    fig_h = 2.55 * len(model_names) + 1.30
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    gs = fig.add_gridspec(
        nrows=n_rows,
        ncols=n_cols,
        height_ratios=[1.0] * len(model_names) + [0.12, 0.11],
        hspace=0.03,
        wspace=0.01,
    )

    for row_idx, model_name in enumerate(model_names):
        poses = scaled_poses[model_name]
        previous_t = 0.0
        for col_idx, t in enumerate(frame_times):
            ax = fig.add_subplot(gs[row_idx, col_idx], projection="3d")
            frame_idx = min(max(int(round(t * fps)) - 1, 0), len(poses) - 1)
            segment_seconds = max(1.0 / float(fps), float(t) - previous_t)
            trail_frames = max(1, int(round(segment_seconds * fps)))
            _draw_snapshot(
                ax,
                poses,
                parent_indices,
                frame_idx,
                center,
                radius,
                trail_frames=trail_frames,
            )
            previous_t = float(t)
            if row_idx == 0:
                ax.set_title(f"{t:.1f}s", fontsize=10, pad=2)
            if col_idx == 0:
                ax.text2D(
                    -0.09,
                    0.5,
                    MODEL_DISPLAY.get(model_name, model_name),
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                    rotation=90,
                    rotation_mode="anchor",
                    clip_on=False,
                )

    time_axis = np.arange(len(waveform), dtype=np.float32) / float(sr) if len(waveform) else np.array([0.0])
    wave_ax = fig.add_subplot(gs[len(model_names), :])
    step = max(1, int(len(waveform) / 6000)) if len(waveform) else 1
    wave_ax.plot(time_axis[::step], waveform[::step], color="#2f4f4f", lw=0.8)
    wave_ax.set_ylabel("Wave", fontsize=10)
    wave_ax.set_yticks([])
    wave_ax.grid(axis="x", linestyle="--", alpha=0.25)
    wave_ax.yaxis.set_label_coords(-0.018, 0.86)

    words_ax = fig.add_subplot(gs[len(model_names) + 1, :], sharex=wave_ax)
    words_ax.set_ylim(0.0, 1.0)
    words_ax.set_yticks([])
    words_ax.set_ylabel("Words", fontsize=10)
    words_ax.set_xlabel("Time s", fontsize=10)
    words_ax.spines["right"].set_visible(False)
    words_ax.spines["top"].set_visible(False)
    words_ax.spines["left"].set_visible(False)
    words_ax.yaxis.set_label_coords(-0.018, 0.14)

    words_font = _safe_font_properties(culture_lang, 8)

    for t in frame_times:
        wave_ax.axvline(t, color="#a0a0a0", lw=0.8, alpha=0.75)
        words_ax.axvline(t, color="#d0d0d0", lw=0.8, alpha=0.7)

    for idx, word in enumerate(words):
        start = float(word["start"])
        end = float(word["end"])
        mid = 0.5 * (start + end)
        lane_y = 0.52 if idx % 2 == 0 else 0.20
        words_ax.plot([start, end], [lane_y - 0.08, lane_y - 0.08], color="#808080", lw=1.2)
        words_ax.text(
            mid,
            lane_y,
            word["word"],
            fontsize=8,
            ha="center",
            va="center",
            fontproperties=words_font,
        )

    max_x = max(frame_times[-1] if frame_times else 0.0, duration)
    wave_ax.set_xlim(0.0, max(0.1, max_x))

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_bark_audios_to_output(io_data: Dict[str, dict], output_dir: str) -> Dict[str, dict]:
    audios_dir = os.path.join(output_dir, "bark_audios")
    os.makedirs(audios_dir, exist_ok=True)
    for lang, payload in io_data.items():
        src = payload.get("audio_path")
        if not src or not os.path.exists(src):
            continue
        dst = os.path.join(audios_dir, f"{lang}.wav")
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.copy2(src, dst)
        payload["audio_path"] = dst
    return io_data

def make_subtitle_clip(txt, start, end, video_size, vertical_position=0, language='en', fontsize=24):
    """
    Renders `txt` onto a transparent image of size `video_size`,
    returns a MoviePy ImageClip with the correct timing.
    Uses language-specific TTF fonts (falls back if missing).

    vertical_position: 0 = bottom, 1 = second from bottom, etc.
    """
    w, h = video_size
    img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # --- load font ---------------------------------------------------
    font = _safe_pil_font(language, fontsize)

    # --- wrap text & measure ----------------------------------------
    # wrap at ~ w / (fontsize*0.6) characters per line
    max_chars = max(20, int(w / (fontsize * 0.6)))
    lines = textwrap.wrap(txt, width=max_chars)
    # get line heights
    line_heights = []
    line_widths = []
    for line in lines:
        try:
            tw, th = draw.textsize(line, font=font)
        except AttributeError:
            bb = draw.textbbox((0, 0), line, font=font)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
        line_widths.append(tw)
        line_heights.append(th)
    total_h = sum(line_heights) + (len(lines) - 1) * 4  # 4px between lines
    max_w = max(line_widths)

    # --- compute box & text origins ---------------------------------
    padding = 8
    box_w = max_w + 2 * padding
    box_h = total_h + 2 * padding
    x0 = (w - box_w) // 2

    # Calculate vertical position based on vertical_position parameter
    base_y = h - box_h - 10  # Default bottom position
    vertical_offset = vertical_position * (box_h + 15)  # 15px gap between subtitle rows
    y0 = base_y - vertical_offset

    # Ensure subtitle doesn't go off-screen at the top
    y0 = max(10, y0)

    x_text = x0 + padding
    y_text = y0 + padding

    # --- draw semi-opaque background -------------------------------
    # Use slightly different background colors for different positions to distinguish them
    bg_alpha = 200 - (vertical_position * 20)  # Slightly more transparent for higher positions
    bg_alpha = max(120, bg_alpha)  # Minimum transparency

    draw.rectangle(
        [x0, y0, x0 + box_w, y0 + box_h],
        fill=(0, 0, 0, bg_alpha)
    )

    # --- draw each line with outline + fill ------------------------
    for i, line in enumerate(lines):
        ly = y_text + sum(line_heights[:i]) + i * 4
        # outline
        for dx in (-1, 1):
            for dy in (-1, 1):
                draw.text((x_text + dx, ly + dy), line, font=font, fill=(0, 0, 0, 255))
        # fill
        draw.text((x_text, ly), line, font=font, fill=(255, 255, 255, 255))

    arr = np.array(img)
    return (
        ImageClip(arr)
        .set_start(start)
        .set_end(end)
        .set_pos(('center', 'bottom'))
    )


def create_parent_indices(skeleton_info, keypoint_indices):
    '''
    :return: a list of parent indices showing the parent of each keypoint
    '''
    skeleton_links = skeleton_info['skeleton_links']

    # Create a mapping from child to parent
    child_to_parent = {child: parent for parent, child in skeleton_links}

    # Initialize parent_indices for the selected keypoint_indices
    parent_indices = []
    for kp in keypoint_indices:
        parent = child_to_parent.get(kp, -1)  # Get parent of the keypoint; default to -1 if not found
        if parent in keypoint_indices:
            # Map to the index within keypoint_indices
            parent_idx = keypoint_indices.index(parent)
        else:
            # Parent is not in keypoint_indices; set to -1 (hip)
            parent_idx = -1
        parent_indices.append(parent_idx)

        parent_indices = parent_indices
    return parent_indices


def arrange_subtitles_youtube_style(subtitles, max_line_width=60):
    """
    Arrange subtitles in YouTube style with 2 lines maximum.

    Logic:
    1. Start with bottom line only
    2. When overlap occurs, move existing subtitle left and add new one right
    3. When bottom line is full, start using top line
    4. When both lines are full, bump old subtitles up and add new ones at bottom

    Returns: List of (start, end, text, line_position, horizontal_position) tuples
    - line_position: 0 = bottom, 1 = top
    - horizontal_position: 0 = left, 1 = center, 2 = right
    """
    if not subtitles:
        return []

    # Sort subtitles by start time
    sorted_subs = sorted(subtitles, key=lambda x: x[0])

    # Track active subtitles on each line
    # Each entry: (end_time, text, horizontal_position)
    bottom_line = []  # line_position = 0
    top_line = []  # line_position = 1

    positioned_subs = []

    for start, end, text in sorted_subs:
        # Clean up expired subtitles
        current_time = start
        bottom_line = [(e, t, h) for e, t, h in bottom_line if e > current_time]
        top_line = [(e, t, h) for e, t, h in top_line if e > current_time]

        # Determine text length for positioning
        text_length = len(text)

        # Strategy 1: Try to place on bottom line
        if len(bottom_line) == 0:
            # Bottom line is empty - place in center
            bottom_line.append((end, text, 1))  # center position
            positioned_subs.append((start, end, text, 0, 1))

        elif len(bottom_line) == 1:
            # One subtitle on bottom line - move it left, add new one right
            # Move existing subtitle to left
            existing_end, existing_text, _ = bottom_line[0]
            # Update positioned_subs to move existing subtitle left
            for i, (s, e, t, line, _) in enumerate(positioned_subs):
                if t == existing_text and e == existing_end and line == 0:
                    positioned_subs[i] = (s, e, t, line, 0)  # move to left
                    break

            # Update bottom_line
            bottom_line[0] = (existing_end, existing_text, 0)  # left position

            # Add new subtitle to right
            bottom_line.append((end, text, 2))  # right position
            positioned_subs.append((start, end, text, 0, 2))

        elif len(bottom_line) >= 2:
            # Bottom line is full - try top line
            if len(top_line) == 0:
                # Top line is empty - place in center
                top_line.append((end, text, 1))  # center position
                positioned_subs.append((start, end, text, 1, 1))

            elif len(top_line) == 1:
                # One subtitle on top line - move it left, add new one right
                existing_end, existing_text, _ = top_line[0]
                # Update positioned_subs to move existing subtitle left
                for i, (s, e, t, line, _) in enumerate(positioned_subs):
                    if t == existing_text and e == existing_end and line == 1:
                        positioned_subs[i] = (s, e, t, line, 0)  # move to left
                        break

                # Update top_line
                top_line[0] = (existing_end, existing_text, 0)  # left position

                # Add new subtitle to right
                top_line.append((end, text, 2))  # right position
                positioned_subs.append((start, end, text, 1, 2))

            else:
                # Both lines are full - bump strategy
                # Remove oldest subtitles and shift remaining ones
                # This is a simplified version - in real YouTube, it's more complex

                # Clear old subtitles that are about to end anyway
                bottom_line = [(e, t, h) for e, t, h in bottom_line if e > current_time + 1.0]
                top_line = [(e, t, h) for e, t, h in top_line if e > current_time + 1.0]

                # If still full, force place on bottom line (will overlap but better than nothing)
                if len(bottom_line) >= 2:
                    # Replace the subtitle that ends earliest
                    earliest_idx = 0
                    earliest_end = bottom_line[0][0]
                    for i, (e, t, h) in enumerate(bottom_line):
                        if e < earliest_end:
                            earliest_end = e
                            earliest_idx = i

                    bottom_line[earliest_idx] = (end, text, 1)  # center position
                    positioned_subs.append((start, end, text, 0, 1))
                else:
                    # There's space now
                    bottom_line.append((end, text, 1))  # center position
                    positioned_subs.append((start, end, text, 0, 1))

    return positioned_subs


def make_language_pose_video(
    poses: np.ndarray,                 # (T, J, 3)
    audio_path: str,                   # path to the .wav
    subtitles: list,                   # list of dicts {start,end,word}
    parent_indices: list,         # [parent_index for each joint]
    keypoint_indices: list,       # your J indices → global skeleton
    out_path: str,
    *,
    fps: int = 15,
    line_width: int = 8,
    figsize=(7, 7),
    camera_elev: int = 10,
    camera_azim: int = 45,
    height_scale: float = 1.0,
    shoulder_scale: float = 2.0,
    target_height: float = 1.0,
    max_words_per_row: int = 6,
    font_size: int = 20,
    language: str = "en",
    frame_times: list = None,  # <--- add this
    frame_out_dir: str = None
):
    """
    Renders a single‐language motion video:
     – 3D skeleton in one pane
     – audio from `audio_path`
     – two‐row subtitle window, shifting words in a FIFO queue
    """

    def _apply_scaling(seq):
        # identical to your comparison function’s helper
        # 1) height
        poses = seq.copy()
        centroid = poses.reshape(-1, 3).mean(0)
        poses = (poses - centroid) * height_scale + centroid
        # 2) shoulders

        if abs(shoulder_scale - 1) > 1e-6:
            try:
                root = poses[:, keypoint_indices.index(8)]
                for side in [(11,12,13), (14,15,16)]:
                    s,e,w = [keypoint_indices.index(x) for x in side]
                    for f in range(len(poses)):
                        vec = poses[f, s] - root[f]
                        new_s = root[f] + vec * shoulder_scale
                        delta = new_s - poses[f, s]
                        poses[f, [s,e,w]] += delta
            except ValueError:
                pass

        # 3) normalize to target_height
        #z0 = poses[0,:,2]; cur_h = z0.max() - z0.min()
        #if cur_h > 1e-6:
        #    poses = poses * (target_height/cur_h)
        return poses


    fontsize = 20
    font_prop = _safe_font_properties(language, font_size)
    poses = _apply_scaling(poses)

    # set up Matplotlib figure & 3D axis
    fig = plt.figure(figsize=figsize, facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_axis_off()
    ax.view_init(elev=camera_elev, azim=camera_azim)

    # equalize the cube
    all_xyz = poses.reshape(-1,3)
    mins, maxs = all_xyz.min(0), all_xyz.max(0)
    ctr = (mins+maxs)/2
    rad = (maxs-mins).max()/2
    pad = rad * 0.5  # pad only half as much (or set to a small value)
    size = 0.2 * height_scale  # increase if you want more room or less cropping
    for lim, getter in [("x", ax.set_xlim), ("y", ax.set_ylim), ("z", ax.set_zlim)]:
        getter(-size, size)

    # prepare line objects
    palette = cycle(plt.get_cmap("tab10").colors)
    lines = [
        ax.plot([],[],[], lw=line_width, color=next(palette), solid_capstyle="round")[0]
        for _ in parent_indices
    ]

    # Pre‑sort subtitles by start time
    if subtitles and isinstance(subtitles[0], str):
        parsed = []
        for s in subtitles:
            # split "0.12 - 0.35: Hello" → ["0.12 - 0.35", "Hello"]
            times, word = s.split(":", 1)
            start_str, end_str = times.split("-", 1)
            parsed.append({
                "start": float(start_str.strip()),
                "end":   float(end_str.strip()),
                "word":  word.strip(),
            })
        subtitles = parsed

    subtitles = sorted(subtitles, key=lambda d: d["start"])

    def make_frame(t):
        # 1) update skeleton at frame i
        i = min(int(t*fps), poses.shape[0]-1)
        P = poses[i] - ctr
        for j,p in enumerate(parent_indices):
            if p < 0: continue
            lines[j].set_data([P[j,0],P[p,0]], [P[j,1],P[p,1]])
            lines[j].set_3d_properties([P[j,2],P[p,2]])

        # 2) figure out which words have appeared so far
        words = [d["word"] for d in subtitles if d["start"] <= t]
        # 3) split into two rolling rows
        row2 = words[-max_words_per_row:]
        row1 = words[-2*max_words_per_row:-max_words_per_row]

        # 4) clear old texts, then draw
        for txt in list(fig.texts):
            fig.texts.remove(txt)
        fig.text(0.5, 0.08, " ".join(row1),
                 ha="center", va="center", fontsize=font_size, fontproperties=font_prop)
        fig.text(0.5, 0.04, " ".join(row2),
                 ha="center", va="center", fontsize=font_size, fontproperties=font_prop)

        return mplfig_to_npimage(fig)

    # build MoviePy clip
    audio_clip = AudioFileClip(audio_path)
    duration = audio_clip.duration
    video_clip = VideoClip(make_frame, duration=duration).set_fps(fps)
    video_clip = video_clip.set_audio(audio_clip)

    duration_video = video_clip.duration
    # how many half‑second steps fit (including t=0)
    n_steps = int(duration_video / 0.5) + 1
    # build frame_times: [0.0, 0.5, 1.0, ..., last ≤ duration]
    frame_times = [min(i * 0.5, duration_video) for i in range(n_steps)]

    if frame_times is None:
        # Default: first, middle, last
        frame_times = [0, audio_clip.duration / 2, max(audio_clip.duration - 1 / fps, 0)]
    if frame_out_dir is not None:
        os.makedirs(frame_out_dir, exist_ok=True)
        for idx, t in enumerate(frame_times):
            frame_path = os.path.join(frame_out_dir, f"{language}_frame_{idx}_{t:.2f}s.png")
            video_clip.save_frame(frame_path, t)

    # write file
    video_clip.write_videofile(out_path,
                               codec="libx264",
                               audio_codec="aac",
                               preset="medium",
                               threads=os.cpu_count(),
                               fps=fps)
    plt.close(fig)


def make_subtitle_clip_youtube_style(txt, start, end, video_size, line_position=0, horizontal_position=1, language='en',
                                     fontsize=24):
    """
    Create a YouTube-style subtitle clip.

    line_position: 0 = bottom, 1 = top
    horizontal_position: 0 = left, 1 = center, 2 = right
    """
    w, h = video_size
    img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Load font

    font = _safe_pil_font(language, fontsize)

    # Wrap text for subtitle width
    max_chars = max(20, int(w / (fontsize * 0.6)))
    lines = textwrap.wrap(txt, width=max_chars)

    # Get line dimensions
    line_heights = []
    line_widths = []
    for line in lines:
        try:
            tw, th = draw.textsize(line, font=font)
        except AttributeError:
            bb = draw.textbbox((0, 0), line, font=font)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
        line_widths.append(tw)
        line_heights.append(th)

    total_h = sum(line_heights) + (len(lines) - 1) * 4  # 4px between lines
    max_w = max(line_widths)

    # Calculate positioning
    padding = 8
    box_w = max_w + 2 * padding
    box_h = total_h + 2 * padding

    # Horizontal positioning
    if horizontal_position == 0:  # left
        x0 = w // 6  # Left side
    elif horizontal_position == 2:  # right
        x0 = w - box_w - w // 6  # Right side
    else:  # center (default)
        x0 = (w - box_w) // 2  # Center

    # Vertical positioning
    bottom_margin = 60
    line_spacing = box_h + 20  # Space between two subtitle lines

    if line_position == 0:  # bottom line
        y0 = h - box_h - bottom_margin
    else:  # top line
        y0 = h - box_h - bottom_margin - line_spacing

    # Ensure subtitle doesn't go off-screen
    x0 = max(10, min(x0, w - box_w - 10))
    y0 = max(10, y0)

    x_text = x0 + padding
    y_text = y0 + padding

    # Draw background with slight transparency variation
    bg_alpha = 220 if line_position == 0 else 200  # Bottom line slightly more opaque
    draw.rectangle(
        [x0, y0, x0 + box_w, y0 + box_h],
        fill=(0, 0, 0, bg_alpha)
    )

    # Draw text with outline
    for i, line in enumerate(lines):
        ly = y_text + sum(line_heights[:i]) + i * 4
        # Black outline
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx != 0 or dy != 0:
                    draw.text((x_text + dx, ly + dy), line, font=font, fill=(0, 0, 0, 255))
        # White text
        draw.text((x_text, ly), line, font=font, fill=(255, 255, 255, 255))

    arr = np.array(img)
    return ImageClip(arr).set_start(start).set_end(end).set_pos(('left', 'top'))

def make_pose_comparison_video(
        real_poses,
        gen_poses,
        audio_path,
        subtitles,
        parent_indices,
        out_path,
        *,
        keypoint_indices,
        fps=15,
        line_width = 7,
        figsize=(10, 5),
        camera_elev=10,
        camera_azim=45,
        height_scale=1,
        shoulder_scale=1.0,
        target_height=2.0,
        color_real = "#d62728",
        color_gen="#d62728",
        language = "en"
):
    """
    Creates a side-by-side video (left = real, right = generated) with
    optional height- and shoulder-scaling, thicker links and custom colours.

    Parameters
    ----------
    keypoint_indices : list[int]
        Indices that map your subset (e.g. [7,8,9,14,15,16,11,12,13]) to
        the skeleton used in `parent_indices`. Needed for shoulder scaling.
    Other parameters unchanged from the original.
    """
    # ------------------------------------------------------ helpers
    def _apply_scaling(seq, height_scale=None, shoulder_scale=1.0):
        """In-place height & shoulder scaling; returns scaled copy."""
        poses = seq.copy()                           # (T, J, 3)

        # -- 1) global height scaling about hip or centroid ------------
        if height_scale is not None and abs(height_scale - 1.0) > 1e-6:
            if 0 in keypoint_indices:                # hip present?
                hip_idx = keypoint_indices.index(0)
                hip = poses[:, hip_idx:hip_idx+1]    # (T,1,3)
                poses = (poses - hip) * height_scale + hip
            else:
                centroid = poses.reshape(-1, 3).mean(0)
                poses = (poses - centroid) * height_scale + centroid

        # -- 2) shoulder stretch ---------------------------------------
        if shoulder_scale is not None and abs(shoulder_scale - 1.0) > 1e-6:
            try:
                spine_up = poses[:, keypoint_indices.index(8)]
                ls, rs = map(keypoint_indices.index, [11, 14])   # shoulders
                le, re = map(keypoint_indices.index, [12, 15])   # elbows
                lw, rw = map(keypoint_indices.index, [13, 16])   # wrists
            except ValueError:
                # required joints not present – skip stretch
                return poses

            for f in range(len(poses)):
                root = spine_up[f]
                for side, (s, e, w) in enumerate([(ls, le, lw), (rs, re, rw)]):
                    vec = poses[f, s] - root
                    new_s = root + vec * shoulder_scale
                    delta = new_s - poses[f, s]
                    poses[f, [s, e, w]] += delta

        # -- 3) fit overall hip-to-head height to target ---------------
        z0 = poses[0, :, 2]
        cur_h = np.nanmax(z0) - np.nanmin(z0)
        if cur_h > 1e-6:
            poses *= target_height / cur_h

        return poses

    real_poses = _apply_scaling(real_poses, height_scale, shoulder_scale)
    gen_poses = _apply_scaling(gen_poses, height_scale, shoulder_scale)

    # ------------------------------------------------------ figure/axes
    fig = plt.figure(figsize=figsize, facecolor="white")
    ax_real = fig.add_subplot(121, projection="3d")
    ax_gen  = fig.add_subplot(122, projection="3d")

    for ax, ttl in [(ax_real, "Real Motion"), (ax_gen, "Generated Motion")]:
        ax.set_title(ttl, fontsize=16)
        ax.set_axis_off()
        ax.grid(False)
        ax.view_init(elev=camera_elev, azim=camera_azim)

    # equalise axis ranges across both views
    all_poses = np.concatenate([real_poses, gen_poses], axis=0)
    mins, maxs = all_poses.reshape(-1, 3).min(0), all_poses.reshape(-1, 3).max(0)
    centre = (mins + maxs) / 2
    radius = (maxs - mins).max() / 2
    rng = radius /height_scale
    #rng = (maxs - mins).max() * 0.3

    real_poses -= centre
    gen_poses -= centre

    for ax in (ax_real, ax_gen):
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlim(centre[0] - rng, centre[0] + rng)
        ax.set_ylim(centre[1] - rng, centre[1] + rng)
        ax.set_zlim(centre[2] - rng, centre[2] + rng)
        #ax.set_xlim(-rng, +radius)
        #ax.set_ylim(-rng, +radius)
        #ax.set_zlim(-rng, +radius)

    palette_real = cycle(plt.get_cmap("tab10").colors)
    palette_gen = cycle(plt.get_cmap("tab20").colors)
    lines_real = [ax_real.plot([], [], [], lw=line_width,
                               solid_capstyle="round",
                               color=next(palette_real))[0]
                  for _ in parent_indices]

    lines_gen = [ax_gen.plot([], [], [], lw=line_width,
                               solid_capstyle="round",
                               color=next(palette_gen))[0]
                  for _ in parent_indices]

    # ------------------------------------------------------ frame fn
    T = real_poses.shape[0]
    #duration = T / fps


    def make_frame(t):
        i = min(int(t * fps), T - 1)
        Rp, Gp = real_poses[i], gen_poses[i]
        for j, p in enumerate(parent_indices):
            if p < 0:
                continue
            # real
            lines_real[j].set_data([Rp[j, 0], Rp[p, 0]], [Rp[j, 1], Rp[p, 1]])
            lines_real[j].set_3d_properties([Rp[j, 2], Rp[p, 2]])
            # generated
            lines_gen[j].set_data([Gp[j, 0], Gp[p, 0]], [Gp[j, 1], Gp[p, 1]])
            lines_gen[j].set_3d_properties([Gp[j, 2], Gp[p, 2]])
        return mplfig_to_npimage(fig)

    # ------------------------------------------------------ moviepy
    from moviepy.editor import VideoClip, CompositeVideoClip, AudioFileClip
    audio_clip = AudioFileClip(audio_path)
    audio_dur = audio_clip.duration
    core = VideoClip(make_frame, duration=audio_dur).set_fps(fps)
    core = core.set_audio(audio_clip)

    # Detect overlapping subtitles and assign positions
    positioned_subtitles = detect_subtitle_overlaps(subtitles)
    # Create subtitle clips with positioning
    subtitle_clips = []

    #print(f"Processing {len(positioned_subtitles)} positioned subtitle segments...")
    overlap_count = 0

    for i, (s, e, txt, vertical_pos) in enumerate(positioned_subtitles):
        if vertical_pos > 0:
            overlap_count += 1

        #print(f"Subtitle {i + 1}: {s:.2f}s - {e:.2f}s (pos: {vertical_pos}): '{txt[:50]}...'")

        # Skip subtitles that are completely outside the audio duration
        if e <= 0 or s >= audio_dur:
            #print(f"  -> Skipped (outside audio duration 0-{audio_dur:.2f}s)")
            continue

        # Clamp subtitle times to audio duration
        s_clamped = max(0, s)
        e_clamped = min(audio_dur, e)

        # Skip if duration becomes too short after clamping
        if e_clamped - s_clamped < 0.1:
            #print(f"  -> Skipped (duration too short after clamping)")
            continue

        #print(f"  -> Using: {s_clamped:.2f}s - {e_clamped:.2f}s at position {vertical_pos}")
        subtitle_clips.append(
            make_subtitle_clip(
                txt, s_clamped, e_clamped, core.size,
                vertical_position=vertical_pos, language=language
            )
        )
    #core = VideoClip(make_frame, duration=duration).set_fps(fps)

    # add audio if present

    final = CompositeVideoClip([core] + subtitle_clips).set_duration(audio_dur)

    # make subtitle clips
    #subs = [make_subtitle_clip(txt, s, e, core.size)
    #        for s, e, txt in subtitles]

    #final = CompositeVideoClip([core, *subs_clamped]).set_duration(audio_dur)

    # ------------------------------------------------------ write
    final.write_videofile(
        out_path,
        codec="libx264",
        audio_codec="aac",
        preset=VIDEO_FFMPEG_PRESET,
        threads=os.cpu_count(),
        fps=fps,
        logger="bar",
    )
    plt.close(fig)


def load_sub_file(path):
    """
    Yields (start_sec, end_sec, text) tuples.
    Expected line format:  START - DUR: subtitle
    """
    pat = re.compile(r"^\s*([\d.]+)\s*-\s*([\d.]+)\s*:\s*(.+)$")
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            m = pat.match(line)
            if not m:
                continue
            t0  = float(m.group(1))
            dur = float(m.group(2))
            txt = m.group(3).strip()
            yield t0, t0 + dur, txt


def test_and_plot_pose(motion_file_path, links_len_file_path, info_file_path, mdm_model, vqvae_model,
                       vqvae_config, diffusion, save_path, text_path = None):
    keypoint_indices = [7, 8, 9, 14, 15, 16, 11, 12, 13]
    playlist_path = "/" + os.path.join(*motion_file_path.split("/")[:-3])
    video_folder = motion_file_path.split("/")[-3]
    language = motion_file_path.split("/")[-4].split("_")[-2]
    target_sr = 16000
    duration = 5.0
    target_fps = 15
    duration_frames_motion = int(duration * target_fps)  # 75 frames

    # Define generation parameters
    prefix_seconds = 1.0  # 1 second prefix
    prefix_frames = int(prefix_seconds * target_fps)  # 15 frames
    prefix_codebooks = 5  # 5 codebooks for 1 second
    generation_seconds = 4.0  # Generate 4 seconds of new motion
    generation_frames = int(generation_seconds * target_fps)  # 60 frames
    generation_codebooks = 20  # 20 codebooks for 4 seconds

    # Accumulate codebooks for one-shot decoding
    all_real_codebooks = []
    all_gen_codebooks = []

    with open(motion_file_path, "rb") as file:
        motion = pickle.load(file)

    with open(links_len_file_path, "rb") as file:
        links_len = pickle.load(file)

    with open(info_file_path, "rb") as file:
        info_data = pickle.load(file)
        info_data = info_data['meta_info']
        real_fps = info_data['fps']  # FPS of the video

    audio_model, audio_processor = load_audio_model()
    text_model, tokenizer = load_text_model()
    text_data, lang_code = pr_text.load_subs(playlist_path, video_folder, language)
    #load_sub_file(text_path)
    audio_data = pr_audio.load_audio(playlist_path, video_folder, target_sr=16000)
    motion_data = motion['data']
    real_start = motion['real_start']  # in frames (at real_fps speed)
    real_end = motion['real_end']

    duration_sample_spectrogram = pr_audio.calc_spectrogram_length_from_motion_length(duration_frames_motion,
                                                                                      target_fps, target_sr, 512)
    duration_sample_wav2vec = pr_audio.calc_wav2vec2_output_frames(duration, target_sr, 320)
    downsample_wav2vec = 5
    duration_sample_wav2vec = int(np.round(duration_sample_wav2vec / downsample_wav2vec))

    downsample_factor = max(1, round(real_fps / target_fps))
    original_pose_fps = real_fps / downsample_factor

    resampled_motion = resample_motion(motion_data, original_pose_fps, target_fps)
    num_resampled_frames = resampled_motion.shape[0]

    audio_start = int(real_start / real_fps * target_sr)
    audio_end = int(real_end / real_fps * target_sr)
    audio_end = min(audio_end, len(audio_data))
    audio_scene = audio_data[audio_start:audio_end]
    audio_data_mel = pr_audio.extract_mel_log(audio_scene, target_sr)
    clip_t0 = real_start / real_fps
    clip_t1 = real_end / real_fps

    subs_from_file = [
        (s - clip_t0, e - clip_t0, txt)  # re-anchor
        for s, e, txt in load_sub_file(text_path)
        if e > clip_t0 and s < clip_t1  # keep only overlapping lines
    ]
    # Check audio-motion alignment
    all_expected_audio_len = pr_audio.calc_spectrogram_length_from_motion_length(
        len(resampled_motion), target_fps, target_sr, 512)

    if abs(all_expected_audio_len - audio_data_mel.shape[1]) > 16:
        print("MISMATCH BETWEEN AUDIO AND MOTION: SKIPPED",
              abs(all_expected_audio_len - audio_data_mel.shape[1]), video_folder, motion_file_path)
        return
    else:
        audio_data_onsets = pr_audio.extract_onsets(audio_scene, target_sr)

    all_subs = []

    # Initialize for continuous generation
    previous_generated_codebooks = None  # Will store generated codebooks for next prefix
    real_motion_start_frame = 0  # Track where we are in real motion
    padding_shape = None

    iteration = 0
    while real_motion_start_frame < num_resampled_frames:
        print(f"\n--- Iteration {iteration + 1} ---")

        # Determine the window for this iteration
        if iteration == 0:
            window_start = real_motion_start_frame
            window_end = min(real_motion_start_frame + duration_frames_motion, num_resampled_frames)
            prefix_from_real = True
        else:
            window_start = real_motion_start_frame - prefix_frames
            window_end = min(real_motion_start_frame + generation_frames, num_resampled_frames)
            prefix_from_real = False

        motion_window = resampled_motion[window_start:window_end]

        if motion_window.shape[0] < duration_frames_motion:
            padding_shape = (duration_frames_motion - motion_window.shape[0],) + motion_window.shape[1:]
            motion_window = np.concatenate([motion_window, np.zeros(padding_shape)], axis=0)

        start_time = (real_start / real_fps) + (window_start / target_fps)
        end_time = start_time + duration
        audio_start_sec = (window_start / target_fps)
        end_audio_sec = audio_start_sec + duration

        audio_features = extract_audio_features(audio_scene, audio_model, audio_processor,
                                                target_sr, audio_start_sec, end_audio_sec,
                                                duration_sample_spectrogram,
                                                duration_sample_wav2vec,
                                                audio_data_mel, audio_data_onsets,
                                                downsample_wav2vec)
        text_features, sentence = extract_text_features(text_data, text_model, tokenizer,
                                                        start_time, end_time)
        motion_features = extract_gesture_features(motion_window, vqvae_model, vqvae_config,
                                                   duration_frames_motion)

        # Convert to tensors
        text_features = torch.tensor(text_features).to(device).unsqueeze(0)
        audio_mels = torch.tensor(audio_features["mel"]).to(device).unsqueeze(0).permute(0, 2, 1)
        audio_onsets = torch.tensor(audio_features["onset"]).to(device).unsqueeze(0)
        audio_wav2vec = torch.tensor(audio_features["wav2vec"]).to(device).unsqueeze(0)
        motion_features = torch.tensor(motion_features).unsqueeze(0).to(device).permute(0, 2, 1)

        all_subs.append((start_time, end_time, sentence))

        # Prepare input for generation
        if iteration == 0:
            x_prefix = motion_features[:, :prefix_codebooks]
        else:
            x_prefix = previous_generated_codebooks[:, -prefix_codebooks:]

        motion_features_modified = motion_features.clone()
        if iteration > 0:
            motion_features_modified[:, :prefix_codebooks] = x_prefix

        batch = (motion_features_modified, text_features, audio_mels, audio_onsets, audio_wav2vec)
        in_data = list(batch)

        new_motion, _ = diffusion.p_sample_loop(mdm_model, in_data, clip_denoised=False)

        previous_generated_codebooks = new_motion.clone()

        # Accumulate codebooks for one-shot decode
        if iteration == 0:
            all_real_codebooks.append(motion_features)
            all_gen_codebooks.append(new_motion)
        else:
            all_real_codebooks.append(motion_features[:, prefix_codebooks:])
            all_gen_codebooks.append(new_motion[:, prefix_codebooks:])

        # Update real motion position
        real_motion_start_frame += (duration_frames_motion if iteration == 0 else generation_frames)
        iteration += 1

    # After generation: one-shot decode full real & generated streams
    real_cb_seq = torch.cat(all_real_codebooks, dim=1)
    gen_cb_seq  = torch.cat(all_gen_codebooks,  dim=1)
    all_true_poses, all_generated_poses = sample_generation_from_codebooks(
        vqvae_config,
        generated_codebooks=gen_cb_seq,
        real_codebooks=real_cb_seq,
        save_path_real='',
        save_path_fake='',
        plot_poses=False
    )

    real_seq = all_true_poses[0]
    gen_seq  = all_generated_poses[0]

    # Final video
    tmp_wav_path = os.path.join("/" + os.path.join(*save_path.split("/")[:-1]), "tmp_audio.wav")
    sf.write(tmp_wav_path, audio_scene, target_sr)

    all_subs.sort(key=lambda x: x[0])
    t0 = all_subs[0][0]
    #subs_aligned = [(s - t0, e - t0, txt) for s, e, txt in all_subs]

    parent_indices = create_parent_indices(info_data, keypoint_indices)
    make_pose_comparison_video(
        real_seq, gen_seq, tmp_wav_path,
        subs_from_file, parent_indices, save_path,
        fps=target_fps,
        keypoint_indices=keypoint_indices,
        height_scale=1.6, shoulder_scale=3,
        line_width=6.0, color_real="#4e79a7", color_gen="#e15759", language = lang_code
    )


def _load_args_from_json(args_path: str) -> argparse.Namespace:
    if not os.path.exists(args_path):
        raise FileNotFoundError(f"args.json not found at: {args_path}")
    with open(args_path, "r") as fr:
        return argparse.Namespace(**json.load(fr))


def _build_mdm_model(args, culture_config, n_train_speakers: int, model_name: str):
    use_culture = model_name != "no_culture"
    use_adversarial = model_name == "adversarial"
    print(
        f"[Model] {model_name}: "
        f"use_culture={use_culture}, use_adversarial={use_adversarial}"
    )
    return Hierarchical_MDM(
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
        use_culture=use_culture,
        use_adversarial=use_adversarial,
        use_native_attention=getattr(args, "use_native_attention", False),
    )


def _load_mdm_and_diffusion(
    args,
    culture_config,
    n_train_speakers: int,
    model_name: str,
    checkpoint_path: str,
):
    mdm_model = _build_mdm_model(args, culture_config, n_train_speakers, model_name)
    diffusion_model = create_gaussian_diffusion(args)
    mdm_model.to(dist_util.dev())
    mdm_model.eval()
    load_and_sync_parameters(
        checkpoint_path,
        device=device,
        mdm_model=mdm_model,
        use_ema=args.use_ema,
        use_culture=(model_name != "no_culture"),
    )
    return mdm_model, diffusion_model


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate multilingual model-comparison timelines. "
            "Each culture gets a figure with rows=model variants and columns=1s snapshots + last frame."
        )
    )
    parser.add_argument("--args-path", required=True, help="Path to hierarchical MDM args.json")
    parser.add_argument("--motion-file-path", required=True, help="Path to prefix motion .pkl")
    parser.add_argument("--info-file-path", required=True, help="Path to skeleton info .pkl")
    parser.add_argument("--output-dir", required=True, help="Output directory for figures and temporary files")
    parser.add_argument("--no-culture-checkpoint", required=True, help="Checkpoint path for no-culture model")
    parser.add_argument("--fishr-checkpoint", required=True, help="Checkpoint path for Fishr model")
    parser.add_argument("--adversarial-checkpoint", required=True, help="Checkpoint path for adversarial model")
    parser.add_argument(
        "--sentence",
        default="Nice to see you all! Look how I move using different cultural styles",
        help="Input sentence to translate and synthesize.",
    )
    parser.add_argument("--work-dir", default=None, help="Optional working dir for generated multilingual audio")
    parser.add_argument(
        "--reuse-audio-dir",
        default=None,
        help="Reuse existing <lang>.wav files from this directory instead of translating and synthesizing audio.",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        choices=LANGUAGE_ORDER,
        default=LANGUAGE_ORDER,
        help="Languages to regenerate.",
    )
    parser.add_argument(
        "--skip-motion-videos",
        action="store_true",
        help="Only write timeline figures; skip per-language motion videos.",
    )
    parser.add_argument("--target-sr", type=int, default=16000)
    parser.add_argument("--target-fps", type=int, default=15)
    parser.add_argument("--duration-window", type=float, default=5.0, help="Generation window length (seconds).")
    parser.add_argument("--seed", type=int, default=10)
    return parser


if __name__ == "__main__":
    cli_args = _build_cli_parser().parse_args()
    os.makedirs(cli_args.output_dir, exist_ok=True)
    fixseed(cli_args.seed)
    selected_languages = list(cli_args.languages)

    args = _load_args_from_json(cli_args.args_path)

    with open(args.culture_config_path) as f:
        culture_config = EasyDict(yaml.safe_load(f))

    dataset_path = args.lmdb_path
    metadata_path = args.metadata_path
    dataset_info_path = encodings_path = normalization_path = args.info_path
    batch_size = args.batch_size
    sep_people = "_sep_people"
    motion_only = False

    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
        sample_keys = metadata["sample_keys"]
        culture_speakers = metadata["culture_speakers"]

    splits_data_path = os.path.join(dataset_info_path, "whole_dataset_splits_subject_independent.pkl")
    _, _, _, _, _, n_train_speakers = prepare_data(
        sample_keys,
        culture_speakers,
        motion_only,
        batch_size,
        splits_data_path,
        dataset_path,
        encodings_path,
        normalization_path,
        sep_people,
    )

    with open(args.vqvae_config_path) as f:
        vqvae_config = EasyDict(yaml.safe_load(f))
    vqvae_model = load_vqvae_model(vqvae_config, vqvae_config.checkpoint_path)

    audio_model, audio_processor = load_audio_model()
    text_model, tokenizer = load_text_model()

    keypoint_indices = [7, 8, 9, 14, 15, 16, 11, 12, 13]
    with open(cli_args.info_file_path, "rb") as file:
        info_data = pickle.load(file)
        info_data = info_data["meta_info"]
    parent_indices = create_parent_indices(info_data, keypoint_indices)

    work_dir = cli_args.work_dir or os.path.join(cli_args.output_dir, "multilang_audio")
    if cli_args.reuse_audio_dir:
        io_data = _load_reused_audio_io_data(cli_args.reuse_audio_dir, selected_languages, cli_args.sentence)
    else:
        io_data = translate_and_tts(cli_args.sentence, work_dir)
        io_data = _save_bark_audios_to_output(io_data, cli_args.output_dir)
    x_prefix_codebooks = encode_motion_prefix(
        cli_args.motion_file_path,
        vqvae_model,
        vqvae_config,
        fps=cli_args.target_fps,
    )

    checkpoints = {
        "no_culture": cli_args.no_culture_checkpoint,
        "fishr": cli_args.fishr_checkpoint,
        "adversarial": cli_args.adversarial_checkpoint,
    }
    poses_by_language_and_model = {lang: {} for lang in selected_languages}

    for model_name in MODEL_ORDER:
        mdm_model, diffusion_model = _load_mdm_and_diffusion(
            args,
            culture_config,
            n_train_speakers,
            model_name,
            checkpoints[model_name],
        )

        poses_by_lang, _ = run_multilang_pipeline(
            cli_args.sentence,
            cli_args.motion_file_path,
            audio_model,
            audio_processor,
            text_model,
            tokenizer,
            mdm_model,
            diffusion_model,
            vqvae_model,
            vqvae_config,
            parent_indices,
            target_sr=cli_args.target_sr,
            target_fps=cli_args.target_fps,
            duration=cli_args.duration_window,
            work_dir=work_dir,
            io_data=io_data,
            x_prefix_codebooks=x_prefix_codebooks,
            languages=selected_languages,
        )
        for lang in selected_languages:
            poses_by_language_and_model[lang][model_name] = poses_by_lang[lang]

        del mdm_model, diffusion_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    figures_dir = os.path.join(cli_args.output_dir, "culture_model_timeline_figures")
    os.makedirs(figures_dir, exist_ok=True)
    for lang in selected_languages:
        out_path = os.path.join(figures_dir, f"{lang}_comparison.png")
        make_culture_comparison_figure(
            lang,
            poses_by_language_and_model[lang],
            io_data[lang]["audio_path"],
            io_data[lang].get("word_timestamps", io_data[lang].get("subs", [])),
            parent_indices,
            keypoint_indices,
            out_path,
            fps=cli_args.target_fps,
            target_sr=cli_args.target_sr,
        )
        print(f"[Saved] {out_path}")

    if not cli_args.skip_motion_videos:
        videos_dir = os.path.join(cli_args.output_dir, "culture_model_motion_videos")
        os.makedirs(videos_dir, exist_ok=True)
        for model_name in MODEL_ORDER:
            model_video_dir = os.path.join(videos_dir, model_name)
            os.makedirs(model_video_dir, exist_ok=True)
            for lang in selected_languages:
                video_out_path = os.path.join(model_video_dir, f"{lang}_{model_name}_motion.mp4")
                make_language_pose_video(
                    poses=poses_by_language_and_model[lang][model_name],
                    audio_path=io_data[lang]["audio_path"],
                    subtitles=io_data[lang].get("word_timestamps", io_data[lang].get("subs", [])),
                    parent_indices=parent_indices,
                    keypoint_indices=keypoint_indices,
                    out_path=video_out_path,
                    fps=cli_args.target_fps,
                    language=lang,
                )
                print(f"[Saved] {video_out_path}")

    print(f"\nDone. Figures written to: {figures_dir}\n")
