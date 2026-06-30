import os
import pickle
import random
import sys
import argparse
import logging
from typing import Any, Dict, List, Optional, Tuple

import yaml
from pprint import pprint
from easydict import EasyDict
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader,Sampler

from TED4CL import process_text as pr_text
from TED4CL import process_audio as pr_audio
from transformers import AutoModel, AutoTokenizer
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.interpolate import interp1d
from vq_vae.vqvae import VQVAE
from vq_vae.configs.parse_args import parse_args
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lmdb

try:
    from deep_translator import GoogleTranslator
except Exception:
    GoogleTranslator = None

#from huggingface_hub import login

# CULTURES: {'indian': 0, 'italian': 1, 'japanese': 2, 'turkish': 3}


# Global variables to store loaded models within each worker
global_audio_model = None
global_audio_processor = None
global_text_model = None
global_text_tokenizer = None
global_motion_model = None
global_motion_model_config = None
global_total_processed_samples = 0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False

COMMIT_EVERY = 2000

LANGUAGE_NAME_TO_CODE = {
    "english": "en",
    "italian": "it",
    "japanese": "ja",
    "turkish": "tr",
    "hindi": "hi",
}


def normalize_language_code(language: Optional[str]) -> str:
    if not language:
        return "unknown"
    lang = str(language).strip().lower().replace("_", "-")
    if lang in LANGUAGE_NAME_TO_CODE:
        return LANGUAGE_NAME_TO_CODE[lang]
    if "-" in lang:
        return lang.split("-")[0]
    return lang


def translate_to_english(text: str, source_lang: Optional[str], enabled: bool = False) -> Tuple[str, str]:
    """Translate text to English when requested; fallback to identity on failures."""
    text = text if text is not None else ""
    source_lang = normalize_language_code(source_lang)

    if text.strip() == "":
        return text, "empty_text"
    if source_lang == "en":
        return text, "identity_en"
    if not enabled:
        return text, "translation_disabled"
    if GoogleTranslator is None:
        return text, "translator_unavailable"

    src = source_lang if source_lang not in ("", "unknown") else "auto"
    try:
        translated = GoogleTranslator(source=src, target="en").translate(text)
        if translated and translated.strip():
            return translated, "machine_translation"
        return text, "machine_translation_empty"
    except Exception:
        return text, "machine_translation_failed"

# Estimates the dimension of a sample in bytes
def estimate_sample_size(sample):
    """
    Serializes a sample and returns its byte size.
    """
    serialized_sample = pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL)
    return len(serialized_sample)

# Calculates the available space in LMDB dataset
def calculate_available_space(env):
    """
    Calculates the available space in the LMDB environment.

    Args:
        env (lmdb.Environment): The LMDB environment.

    Returns:
        int: Available space in bytes.
    """
    info = env.info()
    map_size = info['map_size']
    psize = env.stat()['psize']          # LMDB page size
    last_pgno = info['last_pgno']
    used_space = last_pgno * psize
    return map_size - used_space

# To increase the dimension of LMDB dynamically without losing data
def increase_map_size(env, additional_size_gb=10):

    current_map_size = int(env.info()['map_size'])
    additional_size_bytes = int(additional_size_gb * 1024 ** 3)
    new_map_size = current_map_size + additional_size_bytes

    psize = env.stat()['psize']
    if new_map_size % psize != 0:
        new_map_size = ((new_map_size // psize) + 1) * psize

    env.set_mapsize(int(new_map_size))
    print(f"Map size increased to {new_map_size / (1024 ** 3):.2f} GB")

# Get LMDB used memory
def get_used_size(env):
    info = env.info()
    page_size = info['page_size']
    last_pgno = info['last_pgno']
    used_size = last_pgno * page_size
    return used_size


# Create a LMDB env
def create_lmdb_env(path, initial_size_gb=100):
    """
    Create and return an LMDB environment with a preallocated size.

    Args:
        path (str): Path to the LMDB database.
        initial_size_gb (int): Initial size of the database in GB.

    Returns:
        lmdb.Environment: The LMDB environment.
    """
    map_size = initial_size_gb * 1024 * 1024 * 1024  # Convert GB to bytes
    env = lmdb.open(path, map_size=map_size)
    return env

# Save the keys to a file
def save_sample_keys(keys, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(keys, f)
    print(f"Sample keys saved to {filepath}")

# Load multilingual wav2vec
def load_audio_model():
    # Load and return the pre-trained audio model

    model_name = "voidful/wav2vec2-xlsr-multilingual-56"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    model.to(device)
    model.eval()

    return model, processor

# Load multilingual LaBSE
def load_text_model():
    # Load LaBSE model and tokenizer

    model_name = 'sentence-transformers/LaBSE'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(device)

    return model, tokenizer

# Load pretrained VQVAE
def load_vqvae_model():
    # Load pretrained VQVAE for gesture encodings
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    for k, v in vars(args).items():
        config[k] = v
    #pprint(config)

    config = EasyDict(config)
    mydevice = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    config.stage = "inference"

    model = VQVAE(config.VQVAE, 9 * 6).to(mydevice)  # n_joints * n_chanels
    if torch.cuda.is_available() and hasattr(args, "gpus") and len(args.gpus) > 1:
        model = torch.nn.DataParallel(model, device_ids=args.gpus, output_device=args.gpus[0])
    model = model.to(mydevice)
    checkpoint = torch.load(config.VQVAE_model_path, map_location="cpu")
    model.load_state_dict(checkpoint['model_dict'])
    model.eval() #put in evaluation mode
    #use_bottleneck = config.VQVAE.use_bottleneck

    return model, config

# Load and inits all the models
def initialize_models(load_audio: bool = True, load_text: bool = True, load_motion: bool = True):
    """Initialize global models if they haven't been loaded yet.

    This module is used both for dataset building and for downstream training scripts.
    In many cases (e.g., motion-only datasets) you do not want to download/load heavy
    HuggingFace models. The optional flags allow skipping model loading.
    """
    global global_audio_model, global_audio_processor, global_text_model, global_text_tokenizer
    global global_motion_model, global_motion_model_config

    if load_audio and (global_audio_model is None or global_audio_processor is None):
        print("Loading audio model...")
        global_audio_model, global_audio_processor = load_audio_model()

    if load_text and (global_text_model is None or global_text_tokenizer is None):
        print("Loading text model...")
        global_text_model, global_text_tokenizer = load_text_model()

    if load_motion and (global_motion_model is None or global_motion_model_config is None):
        print("Loading VQVAE motion model...")
        global_motion_model, global_motion_model_config = load_vqvae_model()

# Save and load checkpoints in case pickle dataset is used
def save_checkpoint(checkpoint_path, data):
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(checkpoint_path):
    with open(checkpoint_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Checkpoint loaded from {checkpoint_path}")
    return data


def _unwrap_torch_model(m):
    return m.module if hasattr(m, "module") else m

# Extract gesture vqvae encodings
def extract_gesture_features(poses, model, config, duration):
    vq = _unwrap_torch_model(model)

    # Put poses on the same device as the VQVAE parameters
    model_device = next(vq.parameters()).device
    poses = torch.from_numpy(poses.reshape((poses.shape[0], -1))).to(model_device)

    if poses.shape[0] == duration:
        x = vq.encode_for_net(poses.unsqueeze(0), bottleneck=config.VQVAE.use_bottleneck)
    else:
        n, d = poses.shape
        remainder = duration - (n % duration) if n % duration != 0 else 0
        pad_size = remainder if remainder != 0 else 0
        if pad_size > 0:
            pad = torch.zeros(pad_size, d, dtype=poses.dtype, device=poses.device)
            poses_padded = torch.cat((poses, pad), dim=0)
        else:
            poses_padded = poses

        poses_reshaped = poses_padded.view(-1, duration, d).unsqueeze(0)

        results = []
        for i in range(poses_reshaped.shape[1]):
            chunk = poses_reshaped[:, i, :, :]
            encoded = vq.encode_for_net(chunk, bottleneck=config.VQVAE.use_bottleneck)
            encoded = encoded[0][0]
            results.append(encoded)

        all_results = torch.cat(results, dim=2)
        x = all_results[:, :, :int(n / 4)]

    if config.VQVAE.use_bottleneck and poses.shape[0] == duration:
        x = x[0][0]
    elif poses.shape[0] == duration:
        x = x[0]

    _, n1, n2 = x.shape
    x = x.reshape(n1, n2)

    return x.detach().cpu().numpy().astype(np.float32)


# Extract features for one audio sample
def extract_audio_features(audio_data, model, processor, target_sr, start_time, end_time,
                            duration_spectrogram, duration_wav2vec, mels, onsets, downsample_wav2vec = 5):


    audio_start_mel = pr_audio.calc_spectrogram_length_from_time(start_time, target_sr, 512)
    audio_end_mel = pr_audio.calc_spectrogram_length_from_time(end_time, target_sr, 512)
    audio_start = int(start_time * target_sr)  # audio sample start for extracting wav2vec features
    audio_end = int(end_time * target_sr)
    # Clamp indices to valid range
    mel_start = max(audio_start_mel, 0)
    mel_end = min(audio_end_mel, mels.shape[1])
    mel_sample = mels[:, mel_start:mel_end]
    onset_sample = onsets[mel_start:mel_end]
    wav2vec_sample = pr_audio.extract_wav2vec_embeddings(audio_data[audio_start:audio_end], model, processor, target_sr)

    # Downsample Wav2Vec embeddings before padding to reduce memory usage
    downsampled_wav2vec_sample = downsample_temporal_average(wav2vec_sample.detach().cpu().numpy(), factor=downsample_wav2vec)

    # Pad mel spectrogram
    mel_frames = mel_sample.shape[1]
    if mel_frames < duration_spectrogram:
        padding = np.zeros((mels.shape[0], duration_spectrogram - mel_frames))
        padded_mel_sample = np.concatenate([mel_sample, padding], axis=1)
    else:
        padded_mel_sample = mel_sample[:, :duration_spectrogram]

    # Pad onset sample
    onset_frames = onset_sample.shape[0]
    if onset_frames < duration_spectrogram:
        onset_padding = np.zeros(duration_spectrogram - onset_frames)
        padded_onset_sample = np.concatenate([onset_sample, onset_padding])
    else:
        padded_onset_sample = onset_sample[:duration_spectrogram]

    # Pad Wav2Vec embeddings
    wav2vec_frames = downsampled_wav2vec_sample.shape[0]
    wav2vec_dim = downsampled_wav2vec_sample.shape[1]

    if wav2vec_frames < duration_wav2vec:
        wav2vec_padding = np.zeros((duration_wav2vec - wav2vec_frames, wav2vec_dim))
        padded_wav2vec_sample = np.concatenate([downsampled_wav2vec_sample, wav2vec_padding], axis=0)
    else:
        padded_wav2vec_sample = downsampled_wav2vec_sample[:duration_wav2vec, :]

    return {"mel":padded_mel_sample.astype(np.float32), "onset":padded_onset_sample.astype(np.float32),
            "wav2vec":padded_wav2vec_sample.astype(np.float32)}

# Downsample features that have form of (t, d) across temporal dimension using average pooling.
# t is calculated as int(np.round(num_frames / factor))
def downsample_temporal_average(features, factor=5):
    num_frames, feature_dim = features.shape
    # Calculate the rounded number of downsampled frames
    new_num_frames = int(np.round(num_frames / factor))
    required_total_frames = new_num_frames * factor
    padding_needed = required_total_frames - num_frames

    if padding_needed > 0:
        # Pad by repeating the last frame
        pad_features = np.repeat(features[-1][np.newaxis, :], padding_needed, axis=0)
        features_padded = np.vstack([features, pad_features])
    else:
        features_padded = features[:required_total_frames]

    # Reshape and average
    downsampled = features_padded.reshape(new_num_frames, factor, feature_dim).mean(axis=1)
    return downsampled

# Extracts text embeddings
def extract_text_features(text_data, model, tokenizer, start_time, end_time):
    subs = pr_text.extract_sample_text(start_time, end_time, text_data)
    subs_encoding = pr_text.encode_text_with_labse(subs, tokenizer, model)
    return np.squeeze(subs_encoding).astype(np.float32), subs


def extract_text_features_with_translation(
    text_data,
    english_text_data,
    model,
    tokenizer,
    start_time,
    end_time,
    source_language: Optional[str],
    allow_machine_translation: bool = False,
    translation_cache: Optional[Dict[Tuple[str, str], Tuple[str, str]]] = None,
):
    """Extract original/translated text views and their LaBSE embeddings."""
    original_text = pr_text.extract_sample_text(start_time, end_time, text_data)
    original_features = np.squeeze(
        pr_text.encode_text_with_labse(original_text, tokenizer, model)
    )

    translated_text = original_text
    translation_source = "identity"
    source_language = normalize_language_code(source_language)

    if english_text_data is not None:
        english_text = pr_text.extract_sample_text(start_time, end_time, english_text_data)
        if english_text.strip():
            translated_text = english_text
            translation_source = "english_subtitles"

    if translation_source == "identity":
        cache_key = (source_language, original_text)
        if translation_cache is not None and cache_key in translation_cache:
            translated_text, translation_source = translation_cache[cache_key]
        else:
            translated_text, translation_source = translate_to_english(
                original_text, source_language, enabled=allow_machine_translation
            )
            if translation_cache is not None:
                translation_cache[cache_key] = (translated_text, translation_source)

    if translated_text == original_text:
        translated_features = original_features.copy()
    else:
        translated_features = np.squeeze(
            pr_text.encode_text_with_labse(translated_text, tokenizer, model)
        )

    return {
        "text_features": original_features,
        "text_data": original_text,
        "text_features_translated": translated_features,
        "translated_text": translated_text,
        "translation_source": translation_source,
    }


def parallel_process_playlist(args):
    initialize_models()

    return process_playlist(*args)


# To simplify the sampling process, poses were resampled with a downsampling factor equal to
# int(real_fps / 15). If real_fps is 25, then the downsampling factor is 2 and the sampling fps of
# poses is 12.5 instead of 15. If it is 30 or 60 then we don't have any problem. To make the motion duration
# more consistent and to better align it with audio / text, this function adjusts the sampling rate to 15 fps
def resample_motion(motion, original_fps, target_fps):
    """
    Resample the motion data to a target frame rate.

    :param motion: Numpy array of motion data (num_frames, num_joints, num_channels)
    :param original_fps: Source pose frame rate (can be non-integer, e.g. 12.5)
    :param target_fps: Desired pose frame rate (e.g. 15)
    :return: Motion resampled at target_fps

    We treat each frame as a sample at time t = i / fps (i = 0..N-1).
    This avoids the subtle off-by-one drift you get when using linspace endpoints
    with duration = N / fps.
    """

    if original_fps <= 0 or target_fps <= 0:
        raise ValueError(f"FPS must be positive. original_fps={original_fps}, target_fps={target_fps}")

    motion = np.asarray(motion)
    if motion.ndim != 3:
        raise ValueError(f"Expected motion with shape (T, J, C). Got shape={motion.shape}")

    num_frames, num_joints, num_channels = motion.shape
    if num_frames == 0:
        raise ValueError("Cannot resample empty motion array.")
    if num_frames == 1 or abs(original_fps - target_fps) < 1e-6:
        out = motion.astype(np.float32, copy=True)
        if np.isnan(out).any():
            out = np.nan_to_num(out)
        return out

    # Time axis for source and target samples (seconds).
    t_src = np.arange(num_frames, dtype=np.float32) / float(original_fps)
    duration_last = float(t_src[-1])
    target_frames = int(np.round(duration_last * float(target_fps))) + 1
    target_frames = max(target_frames, 1)

    t_tgt = np.arange(target_frames, dtype=np.float32) / float(target_fps)

    resampled_motion = np.empty((target_frames, num_joints, num_channels), dtype=np.float32)

    # Interpolate each joint/channel independently.
    for j in range(num_joints):
        for c in range(num_channels):
            y = motion[:, j, c].astype(np.float32)
            f = interp1d(t_src, y, kind='linear', bounds_error=False, fill_value="extrapolate")
            resampled_motion[:, j, c] = f(t_tgt).astype(np.float32)

    # Replace NaNs/Infs if any numerical issues appear.
    if not np.isfinite(resampled_motion).all():
        print("Warning: non-finite values detected in resampled motion. Replacing with zeros.")
        resampled_motion = np.nan_to_num(resampled_motion, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    return resampled_motion


# Process a playlist of videos
def _find_pose_pickle_for_fps(video_folder_path: str, video_folder: str) -> Optional[str]:
    """Lookup of the final MMPose pose pickle (used only as a fallback to obtain fps)."""
    candidates = [
        os.path.join(video_folder_path, f"{video_folder}_video_mmpose_data_output_final.pkl"),
        os.path.join(video_folder_path, f"{video_folder}_mmpose_data_output_final.pkl"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    # fall back to any matching file
    for fn in os.listdir(video_folder_path):
        if fn.endswith("_mmpose_data_output_final.pkl") and os.path.isfile(os.path.join(video_folder_path, fn)):
            return os.path.join(video_folder_path, fn)
    return None


def _safe_load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def process_playlist(
    playlist_path,
    culture,
    language,
    duration,
    stride,
    target_sr,
    motion_only,
    lmdb_path,
    data_space_threshold=3 * 1024 ** 3,
    allow_machine_translation: bool = False,
):

    # Use the loaded models
    global global_audio_model, global_audio_processor, global_text_model, \
        global_text_tokenizer, global_motion_model, global_motion_model_config, global_total_processed_samples

    env = lmdb.open(lmdb_path, readonly=False)

    # culture_speakers will represent all the speakers contained in each culture
    culture_speakers = {}
    samples_counter = {'total': 0, 'cultures': {}}
    video_processing_counter = 0

    # Initialize culture entry in samples_counter
    samples_counter['cultures'][culture] = {'total': 0, 'speakers': {}}

    playlist_language = normalize_language_code(language)

    # Desired output pose FPS (the whole dataset timebase)
    poses_fps = 15.0

    # Convert user-specified duration/stride (seconds) to frames on the target timebase.
    # Note: stride_frames_motion will generally not equal exactly stride seconds due to integer frame steps.
    duration_frames_motion = int(round(duration * poses_fps))  # e.g. 5.0s -> 75 frames
    stride_frames_motion = int(stride * poses_fps)             # e.g. 0.5s -> 7 frames (closest at 15 fps)
    stride_frames_motion = max(1, stride_frames_motion)

    try:
        for video_folder in sorted(os.listdir(playlist_path)):
            video_folder_path = os.path.join(playlist_path, video_folder)
            if not os.path.isdir(video_folder_path):
                continue

            print("Processed samples until now", global_total_processed_samples)
            video_processing_counter += 1
            print(f"Processing video {video_folder}, culture {culture}, number {video_processing_counter}")

            # Load modalities that are video-level (audio/subs).
            text_language = playlist_language
            if not motion_only:
                try:
                    text_data, loaded_lang = pr_text.load_subs(playlist_path, video_folder, language)
                    text_language = normalize_language_code(loaded_lang)
                except Exception as e:
                    print(f"[Warn] Failed to load subtitles for {video_folder}: {e}")
                    text_data = None
                    loaded_lang = None

                english_text_data = None
                try:
                    english_text_data, _ = pr_text.load_subs(playlist_path, video_folder, "english")
                except Exception:
                    english_text_data = None

                translation_cache = {}

                try:
                    audio_data = pr_audio.load_audio(playlist_path, video_folder, target_sr)
                except Exception as e:
                    print(f"[Warn] Failed to load audio for {video_folder}: {e}")
                    audio_data = None

                duration_sample_spectrogram = pr_audio.calc_spectrogram_length_from_motion_length(
                    duration_frames_motion, poses_fps, target_sr, 512 #duration of each mel spectrogram sample
                )
                duration_sample_wav2vec = pr_audio.calc_wav2vec2_output_frames(duration, target_sr, 320)  #duration of each wav2vec sample
                downsample_wav2vec = 5
                duration_sample_wav2vec = int(np.round(duration_sample_wav2vec / downsample_wav2vec)) #downsample with average pooling

            motion_data_folder = os.path.join(video_folder_path, 'all_motion_data')
            if not os.path.isdir(motion_data_folder):
                print(f"[Warn] No all_motion_data folder found for {video_folder_path}. Skipping video.")
                continue

            # get video fps from the pose pickle.
            real_fps_fallback = None
            pose_pickle_for_fps = _find_pose_pickle_for_fps(video_folder_path, video_folder)
            if pose_pickle_for_fps is not None:
                try:
                    poses_data_3d = _safe_load_pickle(pose_pickle_for_fps)
                    real_fps_fallback = float(poses_data_3d.get('meta_info', {}).get('fps', 0.0)) or None #video fps
                except Exception:
                    real_fps_fallback = None

            # Process each processed scene file.
            for motion_file in sorted(os.listdir(motion_data_folder)):
                # if the file is "person_segments", i.e. contains links length, or it is not a pickle file, i.e. a valid motion file, continue
                if (not motion_file.endswith('.pkl')) or ("person_segments" in motion_file):
                    continue

                motion_file_path = os.path.join(motion_data_folder, motion_file)

                try:
                    motion_data = _safe_load_pickle(motion_file_path)
                except Exception as e:
                    print(f"[Warn] Could not load {motion_file_path}: {e}")
                    continue

                # Validate required keys
                if not isinstance(motion_data, dict):
                    print(f"[Warn] motion file has unexpected type: {motion_file_path}")
                    continue
                for k in ('real_start', 'real_end', 'data'):
                    if k not in motion_data:
                        print(f"[Warn] Missing key '{k}' in {motion_file_path}. Skipping.")
                        continue

                print(f"Processing scene {motion_file} of video {video_folder} with culture {culture}")

                real_start = int(motion_data['real_start'])  # in frames (at scene_fps)
                real_end = int(motion_data['real_end'])
                motion = np.asarray(motion_data['data'])

                # Scene fps and pose fps are written by process_poses.py. Get it if it was not possible to obtain it from the whole motion output file
                scene_fps = float(motion_data.get('scene_fps', 0.0)) or real_fps_fallback
                if scene_fps is None or scene_fps <= 0:
                    print(f"[Warn] Could not determine scene_fps for {motion_file_path}. Skipping.")
                    continue

                source_pose_fps = float(motion_data.get('pose_fps', 0.0)) #fps of pose sampling.
                if source_pose_fps <= 0:
                    # Fallback for older processed files: approximate the effective pose fps from the scene fps.
                    downsample_factor = max(1, int(round(scene_fps / poses_fps)))
                    source_pose_fps = float(scene_fps / downsample_factor)

                # Sanity checks on motion shape
                if motion.ndim != 3 or motion.shape[0] < 2:
                    print(f"[Warn] Unexpected motion shape {motion.shape} in {motion_file_path}. Skipping.")
                    continue
                if not np.isfinite(motion).all():
                    motion = np.nan_to_num(motion, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

                # Resample motion data to fixed poses_fps (15 fps).
                try:
                    resampled_motion = resample_motion(motion, source_pose_fps, poses_fps)
                except Exception as e:
                    print(f"[Warn] Failed to resample motion for {motion_file_path}: {e}")
                    continue

                num_resampled_frames = int(resampled_motion.shape[0])
                if num_resampled_frames < duration_frames_motion:
                    # Too short for one window at the desired duration.
                    continue

                # Extract audio features for the whole scene (so per-window slicing is cheap).
                if not motion_only:
                    if audio_data is None or text_data is None:
                        print(f"[Warn] Can't build the full dataset sample without audio and text modalities")
                        continue

                    audio_start = int((real_start / scene_fps) * target_sr)
                    audio_end = int((real_end / scene_fps) * target_sr)
                    audio_end = min(audio_end, len(audio_data))
                    audio_start = max(0, min(audio_start, audio_end))
                    audio_scene = audio_data[audio_start:audio_end]

                    # Precompute mel + onset for the entire scene once.
                    try:
                        audio_data_mel = pr_audio.extract_mel_log(audio_scene, target_sr)
                        audio_data_onsets = pr_audio.extract_onsets(audio_scene, target_sr)
                    except Exception as e:
                        print(f"[Warn] Audio feature extraction failed for {motion_file_path}: {e}")
                        continue

                    # Rough alignment check: scene mel length vs expected mel length from motion duration.
                    all_expected_audio_len = pr_audio.calc_spectrogram_length_from_motion_length(
                        len(resampled_motion), poses_fps, target_sr, 512
                    )
                    if abs(all_expected_audio_len - audio_data_mel.shape[1]) > 16:
                        # Big mismatch usually signals a scene-to-pose mapping error; skip this scene.
                        print("MISMATCH BETWEEN AUDIO AND MOTION: SKIPPED",
                              abs(all_expected_audio_len - audio_data_mel.shape[1]), video_folder, motion_file_path)
                        continue

                process_sample = 0

                # Create samples
                for t in range(0, num_resampled_frames - duration_frames_motion + 1, stride_frames_motion):
                    motion_window = resampled_motion[t:t + duration_frames_motion]

                    if motion_window.shape[0] != duration_frames_motion: # we reached the end of the motion file
                        continue

                    # Calculate corresponding times for audio/text extraction (in seconds).
                    start_time = (real_start / scene_fps) + (t / poses_fps)   # absolute time in video
                    end_time = start_time + float(duration)

                    # audio_scene is already scene-local, so use scene-relative time for audio extraction
                    start_audio = (t / poses_fps)
                    end_audio = start_audio + float(duration)

                    # Adjust sample start/end in real frame coordinates (for debugging / alignment)
                    sample_start = int(round(start_time * scene_fps))
                    sample_end = int(round(end_time * scene_fps))

                    if motion_only:
                        sample = {
                            'motion': motion_window.astype(np.float32),
                            'sample_start': sample_start,
                            'sample_end': sample_end,
                            'culture': culture,
                            'speaker': video_folder,
                            'motion_fps': poses_fps,
                            'scene_fps': scene_fps,
                            'language': playlist_language,
                            'text_language': text_language,
                        }
                    else:
                        # Extract all features. wav2vec features are downsampled by a factor of 5 across the temporal
                        # dimension with average pooling to reduce memory usage.
                        try:
                            audio_features = extract_audio_features(
                                audio_scene, global_audio_model, global_audio_processor,
                                target_sr, start_audio, end_audio,
                                duration_sample_spectrogram, duration_sample_wav2vec,
                                audio_data_mel, audio_data_onsets,
                                downsample_wav2vec
                            )
                        except Exception as e:
                            print(f"[Warn] Failed to extract audio features for window in {motion_file_path}: {e}")
                            continue

                        try:
                            text_payload = extract_text_features_with_translation(
                                text_data=text_data,
                                english_text_data=english_text_data,
                                model=global_text_model,
                                tokenizer=global_text_tokenizer,
                                start_time=start_time,
                                end_time=end_time,
                                source_language=text_language,
                                allow_machine_translation=allow_machine_translation,
                                translation_cache=translation_cache,
                            )
                        except Exception as e:
                            print(f"[Warn] Failed to extract text features for window in {motion_file_path}: {e}")
                            continue

                        # VQVAE expects 9 joints x 6 channels. If your processed poses use a different set,
                        # you must adapt both the VQVAE input size and the downstream models.
                        try:
                            motion_features = extract_gesture_features(
                                motion_window, global_motion_model, global_motion_model_config,
                                duration_frames_motion
                            )
                        except Exception as e:
                            print(f"[Warn] Failed to extract motion features for window in {motion_file_path}: {e}")
                            continue

                        sample = {
                            'motion': motion_features,
                            'audio_mels': audio_features["mel"],
                            'audio_onsets': audio_features["onset"],
                            "audio_wav2vec": audio_features["wav2vec"],
                            'text_features': text_payload["text_features"],
                            'text_data': text_payload["text_data"],
                            'text_features_translated': text_payload["text_features_translated"],
                            'translated_text': text_payload["translated_text"],
                            'translation_source': text_payload["translation_source"],
                            'sample_start': sample_start,
                            'sample_end': sample_end,
                            'culture': culture,
                            'speaker': video_folder,
                            'motion_fps': poses_fps,
                            'scene_fps': scene_fps,
                            'language': playlist_language,
                            'text_language': text_language,
                        }

                    process_sample += 1
                    global_total_processed_samples += 1

                    sample_data = pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL)

                    # Key name to recover the sample
                    sample_key = f"{culture}_{video_folder}_{real_start}_{real_end}_{process_sample}_{global_total_processed_samples}"
                    k = sample_key.encode("ascii")
                    inserted = False
                    while True:
                        try:
                            # Use short write transactions to avoid losing large uncommitted
                            # batches when LMDB raises MapFullError.
                            with env.begin(write=True) as txn:
                                inserted = txn.put(k, sample_data, overwrite=False)
                            break
                        except lmdb.MapFullError:
                            increase_map_size(env, additional_size_gb=10)

                    if not inserted:
                        print(f"[Warn] Duplicate sample key encountered; skipping key: {sample_key}")
                        continue

                    # Update counts. Note that If I don't have any sample for a speaker, then
                    # I don't get to this point, i.e., I will not have data for that speaker
                    samples_counter['total'] += 1
                    samples_counter['cultures'][culture]['total'] += 1
                    if video_folder not in samples_counter['cultures'][culture]['speakers']:
                        samples_counter['cultures'][culture]['speakers'][video_folder] = 0
                    samples_counter['cultures'][culture]['speakers'][video_folder] += 1

                    # Build culture_speakers dict
                    if culture not in culture_speakers:
                        culture_speakers[culture] = set()
                    culture_speakers[culture].add(video_folder)

    except Exception as e:
        raise e
    finally:
        env.close()

    return culture_speakers, samples_counter



# Collect samples from playlists
def collect_samples(
    playlists_folder,
    duration,
    stride,
    target_sr,
    metadata_path='',
    dataset_path='',
    motion_only=False,
    initial_size_db=100,
    allow_machine_translation: bool = False,
):

    initialize_models(load_audio=not motion_only, load_text=not motion_only, load_motion=not motion_only)
    total_culture_speakers = {}
    total_samples_generated = {'total': 0, 'cultures_counter': {}}


    final_dataset_path = dataset_path

    metadata_file_path = metadata_path
    if os.path.isdir(metadata_path):
        metadata_file_path = os.path.join(metadata_path, "metadata.pkl")

    # Load existing dataset
    if os.path.exists(final_dataset_path) and os.path.exists(metadata_file_path):
        print("Loading existing dataset and metadata...")
        with open(metadata_file_path, 'rb') as f:
            metadata = pickle.load(f)
            sample_keys = metadata['sample_keys']
            culture_speakers = metadata['culture_speakers']
            samples_counter = metadata['samples_counter']
        print(f"Loaded {len(sample_keys)} sample keys from metadata.")
        return sample_keys, culture_speakers, samples_counter

    # If dataset does not exist, proceed to create it
    print("Dataset not found. Processing playlists and creating LMDB dataset.")


    # TODO adapt to work with lmdb. Current LMDB dataset doesn't use checkpoints
    '''
    # Check for final dataset checkpoint
    if motion_only:
        save_name_playlist = "playlist_motion" # final_dataset_path = os.path.join(checkpoint_folder, "final_dataset_motion.pkl")
    else:
        save_name_playlist = "playlist_data" # final_dataset_path = os.path.join(checkpoint_folder, "final_dataset.pkl")
        
    # Load from checkpoint if available
    if os.path.exists(final_dataset_path):
        print("Loading final dataset from checkpoint...")
        with open(final_dataset_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
            return checkpoint_data['samples'], checkpoint_data['culture_speakers'], checkpoint_data['samples_counter']
    '''
    # Initialize LMDB environment with an appropriate map_size
    initial_size_db = initial_size_db * 1024 ** 3  # Convert GB to bytes
    env = lmdb.open(final_dataset_path, map_size=initial_size_db, readonly=False) #map_async=True, readahead=False, meminit=False) #additional settings

    # Close the environment in the main process
    env.close()

    args_list = []
    for playlist in os.listdir(playlists_folder):
        if not "ted" in playlist: #avoid processing folders that do not contain TED talks. Important: each playlist should contain "ted", the culture and the spoken language.
            continue
        playlist_path = os.path.join(playlists_folder, playlist)
        if os.path.isdir(playlist_path):
            playlist_parts = playlist.split('_')
            culture = playlist_parts[0]
            language = playlist_parts[-2]

            # Prepare arguments for each playlist.
            args = (
                playlist_path,
                culture,
                language,
                duration,
                stride,
                target_sr,
                motion_only,
                final_dataset_path,
                3 * 1024 ** 3,
                allow_machine_translation,
            )
            args_list.append(args)
    for args in args_list:
        culture = args[1]
        try:
            culture_speakers, samples_counter = process_playlist(*args)

            # Aggregate culture speakers
            for culture_key, speakers_set in culture_speakers.items():
                if culture_key not in total_culture_speakers:
                    total_culture_speakers[culture_key] = set()
                total_culture_speakers[culture_key].update(speakers_set)

            # Update samples_generated
            total_samples_generated['total'] += samples_counter['total']
            for culture_key, culture_data in samples_counter['cultures'].items():
                if culture_key not in total_samples_generated['cultures_counter']:
                    total_samples_generated['cultures_counter'][culture_key] = {'total': 0, 'speakers': {}}
                total_samples_generated['cultures_counter'][culture_key]['total'] += culture_data['total']
                for speaker_key, count in culture_data['speakers'].items():
                    if speaker_key not in total_samples_generated['cultures_counter'][culture_key]['speakers']:
                        total_samples_generated['cultures_counter'][culture_key]['speakers'][speaker_key] = 0
                    total_samples_generated['cultures_counter'][culture_key]['speakers'][speaker_key] += count
            print(f"Counter of culture {culture}:", samples_counter)
        except Exception as e:
            raise ValueError(f"Error processing playlist for culture {culture}: {e}")

    # TODO: this part can be completed once a dataset is created. If it fails, it is needed to process the dataset from
    # TODO scratch. Add a loading mechanism if the dataset wasn't completely created with checkpoints
    sample_keys = []
    print("Creating sample keys list...")
    print("Loading data...")
    env = lmdb.open(final_dataset_path, readonly=True)
    print("Data loaded")
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, _ in cursor:
            sample_keys.append(key.decode('ascii'))
    env.close()

    print(f"Collected {len(sample_keys)} sample keys from LMDB.")

    os.makedirs(metadata_path, exist_ok=True)

    #save culture and speaker encodings in metadata and also in separate files

    speaker_encodings_path = os.path.join(metadata_path, 'speaker_encodings.pkl')
    culture_encodings_path = os.path.join(metadata_path, 'culture_encodings.pkl')
    language_encodings_path = os.path.join(metadata_path, 'language_encodings.pkl')

    if os.path.exists(speaker_encodings_path) and os.path.exists(culture_encodings_path):
        with open(speaker_encodings_path, 'rb') as f:
            speaker_encodings = pickle.load(f)
        with open(culture_encodings_path, 'rb') as f:
            culture_encodings = pickle.load(f)
    else:
        speaker_encodings, culture_encodings = create_and_save_culture_speaker_encodings(
            sample_keys, total_culture_speakers, metadata_path
        )
    if os.path.exists(language_encodings_path):
        with open(language_encodings_path, 'rb') as f:
            language_encodings = pickle.load(f)
    else:
        language_encodings = create_and_save_language_encodings(
            sample_keys=sample_keys,
            lmdb_path=final_dataset_path,
            save_path=metadata_path
        )

    # Save sample_keys and culture_speakers to metadata
    metadata_file_path = os.path.join(metadata_path, 'metadata.pkl')
    print("Saving metadata...")
    with open(metadata_file_path, 'wb') as f:
        pickle.dump({
            'sample_keys': sample_keys,
            'culture_speakers': total_culture_speakers,
            'samples_counter': total_samples_generated,
            'culture_encodings': culture_encodings,
            'speaker_encodings': speaker_encodings,
            'language_encodings': language_encodings,
        }, f)
    print(f"Metadata saved to {metadata_path}.")

    print(f"Total samples generated: {total_samples_generated['total']}")


    return sample_keys, total_culture_speakers, total_samples_generated
    #return total_samples, total_culture_speakers, total_samples_generated


def rebuild_metadata_from_lmdb(
    dataset_path: str,
    metadata_path: str,
    include_language_encodings: bool = True,
):
    """Regenerate metadata/encodings from an existing LMDB dataset without rebuilding samples."""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    os.makedirs(metadata_path, exist_ok=True)
    env = lmdb.open(dataset_path, readonly=True, lock=False, readahead=False, meminit=False)

    sample_keys: List[str] = []
    culture_speakers: Dict[str, set] = {}
    samples_counter = {"total": 0, "cultures_counter": {}}

    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            key_str = key.decode("ascii")
            sample_keys.append(key_str)

            # Keys are formatted as:
            # culture_speaker_real_start_real_end_process_sample_global_counter
            # Speaker can contain underscores, so parse from the right.
            parts = key_str.split("_")
            culture = None
            speaker = None
            if len(parts) >= 6:
                try:
                    int(parts[-1]); int(parts[-2]); int(parts[-3]); int(parts[-4])
                    culture = parts[0]
                    speaker = "_".join(parts[1:-4])
                except Exception:
                    culture = None
                    speaker = None

            # Fallback to payload if key parsing is not possible.
            if culture is None or speaker is None or speaker == "":
                try:
                    sample = pickle.loads(value)
                    culture = str(sample.get("culture", "unknown"))
                    speaker = str(sample.get("speaker", "unknown"))
                except Exception:
                    culture = "unknown"
                    speaker = "unknown"

            if culture not in culture_speakers:
                culture_speakers[culture] = set()
            culture_speakers[culture].add(speaker)

            if culture not in samples_counter["cultures_counter"]:
                samples_counter["cultures_counter"][culture] = {"total": 0, "speakers": {}}
            samples_counter["total"] += 1
            samples_counter["cultures_counter"][culture]["total"] += 1
            samples_counter["cultures_counter"][culture]["speakers"][speaker] = (
                samples_counter["cultures_counter"][culture]["speakers"].get(speaker, 0) + 1
            )

    env.close()

    # Build and save speaker/culture encodings deterministically.
    all_speakers = sorted({sp for speakers in culture_speakers.values() for sp in speakers})
    all_cultures = sorted(culture_speakers.keys())
    speaker_encodings = {speaker: idx for idx, speaker in enumerate(all_speakers)}
    culture_encodings = {culture: idx for idx, culture in enumerate(all_cultures)}

    with open(os.path.join(metadata_path, "speaker_encodings.pkl"), "wb") as f:
        pickle.dump(speaker_encodings, f)
    with open(os.path.join(metadata_path, "culture_encodings.pkl"), "wb") as f:
        pickle.dump(culture_encodings, f)

    if include_language_encodings:
        language_encodings = create_and_save_language_encodings(
            sample_keys=sample_keys,
            lmdb_path=dataset_path,
            save_path=metadata_path,
        )
    else:
        language_encodings = {"unknown": 0}
        with open(os.path.join(metadata_path, "language_encodings.pkl"), "wb") as f:
            pickle.dump(language_encodings, f)

    metadata_file_path = os.path.join(metadata_path, "metadata.pkl")
    with open(metadata_file_path, "wb") as f:
        pickle.dump(
            {
                "sample_keys": sample_keys,
                "culture_speakers": culture_speakers,
                "samples_counter": samples_counter,
                "culture_encodings": culture_encodings,
                "speaker_encodings": speaker_encodings,
                "language_encodings": language_encodings,
            },
            f,
        )

    print(f"Metadata rebuilt from LMDB at {metadata_file_path}")
    print(f"Collected {len(sample_keys)} keys across {len(all_cultures)} cultures and {len(all_speakers)} speakers.")
    return sample_keys, culture_speakers, samples_counter


# Prepare data subject independent or subject dependent data loaders for train, validation and test
# if it is present "sep_people" in the last argument string, then subject independent data splits are used
def prepare_data(sample_keys, culture_speakers, motion_only=False,
                batch_size=128, splits_path='', lmdb_path='',
                encodings_path='', normalization_path='', sep_people = '',
                use_translated_text: bool = False, use_language_features: bool = True,
                use_translated_text_eval: Optional[bool] = None,
                use_language_features_eval: Optional[bool] = None):

    if use_translated_text_eval is None:
        use_translated_text_eval = use_translated_text
    if use_language_features_eval is None:
        use_language_features_eval = use_language_features

    # Step 1: Create and save encodings if not already saved
    speaker_encodings_path = os.path.join(encodings_path, 'speaker_encodings.pkl')
    culture_encodings_path = os.path.join(encodings_path, 'culture_encodings.pkl')
    language_encodings_path = os.path.join(encodings_path, 'language_encodings.pkl')

    if os.path.exists(speaker_encodings_path) and os.path.exists(culture_encodings_path):
        with open(speaker_encodings_path , 'rb') as f:
            speaker_encodings = pickle.load(f)
        with open(culture_encodings_path , 'rb') as f:
            culture_encodings = pickle.load(f)
        print("Loaded existing speaker and culture encodings.")
    else:
        speaker_encodings, culture_encodings = create_and_save_culture_speaker_encodings(sample_keys, culture_speakers, encodings_path)

    if os.path.exists(language_encodings_path):
        with open(language_encodings_path, 'rb') as f:
            language_encodings = pickle.load(f)
        print("Loaded existing language encodings.")
    else:
        language_encodings = create_and_save_language_encodings(sample_keys, lmdb_path, encodings_path)

    # Step 2: Create dataset splits
    if '_sep_people' in sep_people:
        train_loader, val_loader, test_loader, n_train_speakers, n_val_speakers, n_test_speakers =\
            create_subject_independent_split(
            sample_keys, culture_speakers, motion_only=motion_only,
            batch_size=batch_size, save_path=splits_path, lmdb_path=lmdb_path,
            speaker_encodings=speaker_encodings, culture_encodings=culture_encodings,
            language_encodings=language_encodings, use_translated_text=use_translated_text,
            use_language_features=use_language_features,
            save_speakers=True
        )
    else:
        train_loader, val_loader, test_loader = create_subject_dependent_split(
            sample_keys, motion_only=motion_only,
            batch_size=batch_size, save_path=splits_path, lmdb_path=lmdb_path,
            speaker_encodings=speaker_encodings, culture_encodings=culture_encodings,
            language_encodings=language_encodings, use_translated_text=use_translated_text,
            use_language_features=use_language_features
        )
        n_train_speakers = 0

    # Step 3: If needed, compute normalization on training set. In the actual version we don't normalize data, we
    # use layer norm as normalization
    '''
    normalization_file = 'features_normalization_speaker_independent.pkl' \
        if "_sep_people" in sep_people else 'features_normalization_speaker_dependent.pkl'
    final_norm_path = os.path.join(normalization_path, normalization_file)
    if not os.path.exists(final_norm_path):
        normalization = compute_normalization(train_loader)
        save_normalization(normalization, final_norm_path)
    else:
        normalization = load_normalization(final_norm_path)
    '''
    # Step 4: Re-create datasets with normalization
    train_dataset = GestureDataset(
        lmdb_path=lmdb_path, keys=train_loader.dataset.keys, motion_only=motion_only,
        speaker_encodings=speaker_encodings, culture_encodings=culture_encodings,
        language_encodings=language_encodings, use_translated_text=use_translated_text,
        use_language_features=use_language_features,
        #normalize_params=normalization
    )
    val_dataset = GestureDataset(
        lmdb_path=lmdb_path, keys=val_loader.dataset.keys, motion_only=motion_only,
        speaker_encodings=speaker_encodings, culture_encodings=culture_encodings,
        language_encodings=language_encodings, use_translated_text=use_translated_text_eval,
        use_language_features=use_language_features_eval,
        #normalize_params=normalization
    )
    test_dataset = GestureDataset(
        lmdb_path=lmdb_path, keys=test_loader.dataset.keys, motion_only=motion_only,
        speaker_encodings=speaker_encodings, culture_encodings=culture_encodings,
        language_encodings=language_encodings, use_translated_text=use_translated_text_eval,
        use_language_features=use_language_features_eval,
        #normalize_params=normalization
    )

    # Step 5: Create new DataLoaders with normalized datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, prefetch_factor=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=8, prefetch_factor=4, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,num_workers=8, prefetch_factor=4, pin_memory=False)


    return train_loader, val_loader, test_loader, speaker_encodings, culture_encodings, n_train_speakers


# Encode speakers and cultures with integer numbers. Save the encoding to use them later
def create_and_save_culture_speaker_encodings(sample_keys, culture_speakers, save_path):
    """
    Encodes speakers and cultures and saves the mappings to disk.

    Args:
        sample_keys (list): List of strings where each string is in the format 'culture_speaker'.
        save_path (str): Directory path to save the encoding files.

    Returns:
        tuple: Dictionaries for speaker and culture encodings.
    """
    speaker_set = set()
    culture_set = set()

    # Extract unique speakers and cultures
    for key in sample_keys:
        parts = key.split('_')
        culture = parts[0]
        if culture in culture_speakers:
            possible_speakers = culture_speakers[culture]
            for speaker in possible_speakers:
                if f"_{speaker}_" in key:  # Check if the speaker matches the current key
                    speaker_set.add(speaker)
                    break
            else:
                raise ValueError(f"Could not match speaker for key: {key}")
        else:
            raise ValueError(f"Culture '{culture}' not found in culture_speakers.")
        culture_set.add(culture)

    # Create sorted lists of unique speakers and cultures
    speakers = sorted(speaker_set)
    cultures = sorted(culture_set)

    # Create one-hot encodings
    speaker_encodings= {speaker: idx for idx, speaker in enumerate(speakers)}
    culture_encodings = {culture: idx for idx, culture in enumerate(cultures)}

    # Save the encodings to disk
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, 'speaker_encodings.pkl'), 'wb') as f:
        pickle.dump(speaker_encodings, f)

    with open(os.path.join(save_path, 'culture_encodings.pkl'), 'wb') as f:
        pickle.dump(culture_encodings, f)

    print("Speaker and culture encodings saved.")
    return speaker_encodings, culture_encodings


def create_and_save_language_encodings(sample_keys, lmdb_path, save_path):
    """Encodes languages from LMDB samples and saves mapping to disk."""
    language_set = set()
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    try:
        with env.begin(write=False) as txn:
            for key in sample_keys:
                sample_raw = txn.get(key.encode("ascii"))
                if sample_raw is None:
                    continue
                sample = pickle.loads(sample_raw)
                language = normalize_language_code(
                    sample.get("language", sample.get("text_language", "unknown"))
                )
                language_set.add(language)
    finally:
        env.close()

    if not language_set:
        language_set.add("unknown")
    if "unknown" not in language_set:
        language_set.add("unknown")

    languages = sorted(language_set)
    language_encodings = {language: idx for idx, language in enumerate(languages)}

    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "language_encodings.pkl"), "wb") as f:
        pickle.dump(language_encodings, f)

    print("Language encodings saved.")
    return language_encodings


# To compute normalization values of data. In the current version it is not used
def compute_normalization(train_loader):
    # Initialize accumulators
    sum_gesture = 0
    sum_gesture_sq = 0

    sum_audio_mels = 0
    sum_audio_mels_sq = 0

    #Min Max normalization since we have only positive values
    global_onsets_min = float('inf')
    global_onsets_max = -float('inf')

    sum_audio_wav2vec = 0
    sum_audio_wav2vec_sq = 0

    sum_text_features = 0
    sum_text_features_sq = 0

    num_wav2vec_frames = 0
    num_gesture_frames = 0
    num_audio_mels_frames = 0
    num_audio_onsets_frames = 0
    num_text_frames = 0  # Assuming one per sample

    # Initialize lists to collect culture and speaker labels
    #all_cultures = []
    #all_speakers = []
    #num_samples = 0

    for i,batch in enumerate(train_loader):
        print(f"iteration {i} / {len(train_loader)}")
        motion, text_features, audio_mels, audio_onsets, audio_wav2vec, metadata = batch
        batch_size = motion.size(0)
        #num_samples += batch_size

        # Gesture Codebooks (batch_size, frames_gesture, embedding_dim_gesture)
        sum_gesture += motion.sum(dim=[0, 1])  # Sum over batch and frames (to have one normalization value for each embedding component)
        sum_gesture_sq += (motion ** 2).sum(dim=[0, 1])
        num_gesture_frames += batch_size * motion.size(1)

        # Audio Mels (batch_size, frames_audio_mels, n_mels)
        sum_audio_mels += audio_mels.sum(dim=[0, 1])
        sum_audio_mels_sq += (audio_mels ** 2).sum(dim=[0, 1])
        num_audio_mels_frames += batch_size * audio_mels.size(1)

        # Audio Onsets (batch_size, frames_audio_onsets)
        batch_onsets_min = audio_onsets.min().item()
        batch_onsets_max = audio_onsets.max().item()
        global_onsets_min = min(global_onsets_min, batch_onsets_min)
        global_onsets_max = max(global_onsets_max, batch_onsets_max)
        num_audio_onsets_frames += batch_size * audio_onsets.size(1)

        # Audio Wav2Vec (batch_size, frames_audio_wav2vec, embedding_dim_wav2vec)
        sum_audio_wav2vec += audio_wav2vec.sum(dim=[0, 1])
        sum_audio_wav2vec_sq += (audio_wav2vec ** 2).sum(dim=[0, 1])
        num_wav2vec_frames += batch_size * audio_wav2vec.size(1)

        # Text Encodings (batch_size, embedding_dim_text)
        sum_text_features += text_features.sum(dim=0)
        sum_text_features_sq += (text_features ** 2).sum(dim=0)
        num_text_frames += batch_size


    # Compute means and stds for gesture codebooks
    gesture_mean = sum_gesture / num_gesture_frames
    gesture_var = (sum_gesture_sq / num_gesture_frames) - (gesture_mean ** 2)
    gesture_var = torch.clamp(gesture_var, min=0.0)
    gesture_std = torch.sqrt(gesture_var)

    # Compute means and stds for audio mels
    audio_mels_mean = sum_audio_mels / num_audio_mels_frames
    audio_mels_var = (sum_audio_mels_sq / num_audio_mels_frames) - (audio_mels_mean ** 2)
    audio_mels_var = torch.clamp(audio_mels_var, min=0.0) #Prevent negative variance
    audio_mels_std = torch.sqrt(audio_mels_var)

    # Compute min and max for audio onsets
    audio_onsets_min = global_onsets_min
    audio_onsets_max = global_onsets_max

    # Compute means and stds for audio wav2vec
    audio_wav2vec_mean = sum_audio_wav2vec / num_wav2vec_frames
    audio_wav2vec_var = (sum_audio_wav2vec_sq / num_wav2vec_frames) - (audio_wav2vec_mean ** 2)
    audio_wav2vec_var = torch.clamp(audio_wav2vec_var, min=0.0)
    audio_wav2vec_std = torch.sqrt(audio_wav2vec_var)

    # Compute means and stds for text encodings
    text_features_mean = sum_text_features / num_text_frames
    text_features_var = (sum_text_features_sq / num_text_frames) - (text_features_mean ** 2)
    text_features_var = torch.clamp(text_features_var, min=0.0)
    text_features_std = torch.sqrt(text_features_var)

    normalization = {
        'gesture_mean': gesture_mean,
        'gesture_std': gesture_std,
        'audio_mels_mean': audio_mels_mean,
        'audio_mels_std': audio_mels_std,
        'audio_onsets_min': audio_onsets_min,
        'audio_onsets_max': audio_onsets_max,
        'audio_wav2vec_mean': audio_wav2vec_mean,
        'audio_wav2vec_std': audio_wav2vec_std,
        'text_features_mean': text_features_mean,
        'text_features_std': text_features_std
    }

    return normalization

# Save and load normalization parameters
def save_normalization(normalization, save_path):
    #os.makedirs(save_path, exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(normalization, f)
    print("Normalization parameters saved.")

def load_normalization(save_path):
    with open(save_path, 'rb') as f:
        normalization = pickle.load(f)
    print("Normalization parameters loaded.")
    return normalization

# Create subject dependent data splits (data of same subjects can be present in all the sets, but each split have different samples)
def create_subject_dependent_split(sample_keys, motion_only=False,
                                   batch_size=128, save_path='',lmdb_path='',
                                   speaker_encodings=None, culture_encodings=None,
                                   language_encodings=None, use_translated_text: bool = False,
                                   use_language_features: bool = True):
    train_keys, val_keys, test_keys = load_dataset_splits(save_path)
    if train_keys is not None and val_keys is not None and test_keys is not None:
        print("Using saved dataset splits.")
    else:
        # Randomly shuffle all sample keys
        shuffled_keys = sample_keys.copy()
        random.shuffle(shuffled_keys)
        num_samples = len(shuffled_keys)
        train_size = int(0.8 * num_samples)
        val_size = int(0.1 * num_samples)
        #test_size = num_samples - train_size - val_size

        train_keys = shuffled_keys[:train_size]
        val_keys = shuffled_keys[train_size:train_size + val_size]
        test_keys = shuffled_keys[train_size + val_size:]

        # Save the newly created dataset splits
        save_dataset_splits(train_keys, val_keys, test_keys, save_path)

    # Create datasets
    train_dataset = GestureDataset(lmdb_path=lmdb_path, keys=train_keys, motion_only=motion_only,
                                   speaker_encodings=speaker_encodings, culture_encodings=culture_encodings,
                                   language_encodings=language_encodings,
                                   use_translated_text=use_translated_text,
                                   use_language_features=use_language_features)
    val_dataset = GestureDataset(lmdb_path=lmdb_path, keys=val_keys, motion_only=motion_only,
                                 speaker_encodings=speaker_encodings, culture_encodings=culture_encodings,
                                 language_encodings=language_encodings,
                                 use_translated_text=use_translated_text,
                                 use_language_features=use_language_features)
    test_dataset = GestureDataset(lmdb_path=lmdb_path, keys=test_keys, motion_only=motion_only,
                                  speaker_encodings=speaker_encodings, culture_encodings=culture_encodings,
                                  language_encodings=language_encodings,
                                  use_translated_text=use_translated_text,
                                  use_language_features=use_language_features)

    print("len training:",len(train_dataset),"len validation:",len(val_dataset),"len test:", len(test_dataset),
          "total len:",len(train_dataset)+len(val_dataset)+len(test_dataset))

    # Create DataLoaders for subject-dependent split
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    return train_loader, val_loader, test_loader

# Creates subject independent data splits (subjects in training can't be present in validation and test)
# Note that here we extract also the ID of training, validation and test speakers that are needed when we use a
# speaker classification head in the model, so we avoid considering test speakers
def create_subject_independent_split(sample_keys, culture_speakers, motion_only=False,
                                     batch_size=128, save_path='', lmdb_path='',
                                     speaker_encodings=None, culture_encodings=None,
                                     language_encodings=None, use_translated_text: bool = False,
                                     use_language_features: bool = True, save_speakers = True):
    # Try loading dataset splits if already available
    if save_speakers:
        train_keys, val_keys, test_keys, n_train_speakers , n_val_speakers, n_test_speakers = load_dataset_splits(save_path, n_speakers=save_speakers)
    else:
        train_keys, val_keys, test_keys = load_dataset_splits(save_path, n_speakers=save_speakers)

    if train_keys is not None and val_keys is not None and test_keys is not None:
        print("Using saved dataset splits.")
    else:
        print("Creating new dataset splits.")
        speaker_keys = {}
        for key in sample_keys:
            # Extract culture from the key
            culture = key.split('_')[0]

            # Extract speaker by matching against culture_speakers dictionary
            if culture in culture_speakers:
                possible_speakers = culture_speakers[culture]
                for speaker in possible_speakers:
                    if f"_{speaker}_" in key:  # Check if the speaker matches the current key
                        if speaker not in speaker_keys:
                            speaker_keys[speaker] = []
                        speaker_keys[speaker].append(key)
                        break
                else:

                    raise ValueError(f"Could not match speaker for key: {key}")
            else:
                raise ValueError(f"Culture '{culture}' not found in culture_speakers.")

        train_keys = []
        val_keys = []
        test_keys = []

        train_speakers_set = set()
        val_speakers_set = set()
        test_speakers_set = set()

        missing_speakers = {"train": set(), "val": set(), "test": set()}
        for culture, speakers in culture_speakers.items(): # culture:{speaker1,speaker2,....,speakerN}
            speakers = list(speakers)
            random.shuffle(speakers)
            num_speakers = len(speakers)
            train_size = max(1, int(0.8 * num_speakers))
            val_size = max(1, int(0.1 * num_speakers))
            test_size = num_speakers - train_size - val_size
            if test_size == 0 and num_speakers >= 3:
                test_size = 1
                val_size -= 1
            train_speakers = speakers[:train_size] #for each culture, we split train, validation and test speakers
            val_speakers = speakers[train_size:train_size + val_size]
            test_speakers = speakers[train_size + val_size:]

            # Collect samples for each set
            for speaker in train_speakers:
                if speaker in speaker_keys:
                    train_speakers_set.add(speaker)
                    train_keys.extend(speaker_keys[speaker]) #put all the keys of a given speaker
                else:
                    missing_speakers["train"].add(speaker)
            for speaker in val_speakers:
                if speaker in speaker_keys:
                    val_speakers_set.add(speaker)
                    val_keys.extend(speaker_keys[speaker])
                else:
                    missing_speakers["val"].add(speaker)
            for speaker in test_speakers:
                if speaker in speaker_keys:
                    test_speakers_set.add(speaker)
                    test_keys.extend(speaker_keys[speaker])
                else:
                    missing_speakers["test"].add(speaker)

        for split_name in ("train", "val", "test"):
            if missing_speakers[split_name]:
                print(
                    f"[Warn] Missing {split_name} speakers with no dataset samples: "
                    f"{sorted(missing_speakers[split_name])}"
                )

        n_train_speakers = len(train_speakers_set)
        n_val_speakers = len(val_speakers_set)
        n_test_speakers = len(test_speakers_set)

        if not save_speakers:
            n_speakers = ()
        else:
            n_speakers = (n_train_speakers,n_val_speakers,n_test_speakers)

        # Save the newly created dataset splits
        save_dataset_splits(train_keys, val_keys, test_keys, save_path, n_speakers = n_speakers)

    train_dataset = GestureDataset(lmdb_path=lmdb_path, keys=train_keys, motion_only=motion_only,
                                   speaker_encodings=speaker_encodings, culture_encodings=culture_encodings,
                                   language_encodings=language_encodings,
                                   use_translated_text=use_translated_text,
                                   use_language_features=use_language_features)
    val_dataset = GestureDataset(lmdb_path=lmdb_path, keys=val_keys, motion_only=motion_only,
                                 speaker_encodings=speaker_encodings, culture_encodings=culture_encodings,
                                 language_encodings=language_encodings,
                                 use_translated_text=use_translated_text,
                                 use_language_features=use_language_features)
    test_dataset = GestureDataset(lmdb_path=lmdb_path, keys=test_keys, motion_only=motion_only,
                                  speaker_encodings=speaker_encodings, culture_encodings=culture_encodings,
                                  language_encodings=language_encodings,
                                  use_translated_text=use_translated_text,
                                  use_language_features=use_language_features)

    # DataLoaders for subject-dependent split
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=8,
                              pin_memory=True,
                              persistent_workers=True,
                              prefetch_factor=4
                              )
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=8,
                            pin_memory=True,
                            persistent_workers=True,
                            prefetch_factor=4
                            )
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=8,
                             pin_memory=True,
                             persistent_workers=True,
                             prefetch_factor=4
                             )

    if save_speakers:
        return train_loader, val_loader, test_loader, n_train_speakers, n_val_speakers , n_test_speakers
    else:
        return train_loader, val_loader, test_loader

# The dataset used to train the gesture generation model, the vqvae gesture encoder, the subject-independent culture classifier.
# if motion_only = True then only raw - motion data is saved. Raw motion data is used for vqvae gesture encoder
class GestureDataset(Dataset):
    def __init__(
        self,
        lmdb_path,
        keys,
        motion_only=False,
        speaker_encodings=None,
        culture_encodings=None,
        language_encodings=None,
        normalize_params=None,
        use_translated_text: bool = False,
        use_language_features: bool = True,
    ):
        self.lmdb_path = lmdb_path
        self.keys = keys
        self.motion_only = motion_only
        self.speaker_encodings = speaker_encodings if speaker_encodings else {} #one hot speakers encoding
        self.culture_encodings = culture_encodings if culture_encodings else {} #one hot culture encoding
        self.language_encodings = language_encodings if language_encodings else {"unknown": 0}
        if "unknown" not in self.language_encodings:
            self.language_encodings["unknown"] = len(self.language_encodings)
        self.normalize_params = normalize_params
        self.use_translated_text = use_translated_text
        self.use_language_features = use_language_features
        self._env = None  # This will be initialized lazily

    def __len__(self):
        return len(self.keys)

    def _initialize_env(self):
        if self._env is None:
            self._env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)

    def _encode_language(self, sample):
        language_raw = normalize_language_code(
            sample.get("language", sample.get("text_language", "unknown"))
        )
        text_language_raw = normalize_language_code(sample.get("text_language", language_raw))
        language_idx = self.language_encodings.get(
            language_raw, self.language_encodings["unknown"]
        )
        return language_raw, text_language_raw, torch.tensor(language_idx, dtype=torch.int64)

    def __getitem__(self, idx):
        self._initialize_env()
        key = self.keys[idx]

        with self._env.begin(write=False) as txn:
            sample = txn.get(key.encode('ascii'))
            if sample is None:
                raise KeyError(f"Key {key} not found in LMDB.")
            sample = pickle.loads(sample)

        if not self.motion_only:
            motion = torch.tensor(sample['motion'],dtype=torch.float32).transpose(0, 1) #Motion is VQVAE encoding here. (25,512)
            if self.normalize_params:
                motion = (motion - self.normalize_params['gesture_mean']) / self.normalize_params['gesture_std']

            audio_mels = torch.tensor(sample['audio_mels'],dtype=torch.float32).transpose(0, 1) #(156,64)
            if self.normalize_params:
                audio_mels = (audio_mels - self.normalize_params['audio_mels_mean']) / self.normalize_params['audio_mels_std']

            audio_onsets = torch.tensor(sample['audio_onsets'],dtype=torch.float32) #(156)
            if self.normalize_params:
                onset_min = self.normalize_params['audio_onsets_min']
                onset_max = self.normalize_params['audio_onsets_max']
                audio_onsets = (audio_onsets - onset_min) / (onset_max - onset_min + 1e-8)

            audio_wav2vec = torch.tensor(sample['audio_wav2vec'],dtype=torch.float32) #(50,1024)
            if self.normalize_params:
                audio_wav2vec = (audio_wav2vec - self.normalize_params['audio_wav2vec_mean']) / self.normalize_params['audio_wav2vec_std']

            # get either originsl text features or translated text features, depending on the flag. If the flag is true but translated features are not present, use original text features
            text_feature_key = 'text_features'
            if self.use_translated_text and 'text_features_translated' in sample:
                text_feature_key = 'text_features_translated'
            text_features = torch.tensor(sample[text_feature_key],dtype=torch.float32) #(768)
            if not self.use_language_features:
                text_features = torch.zeros_like(text_features)
            if self.normalize_params:
                text_features = (text_features - self.normalize_params['text_features_mean']) / self.normalize_params['text_features_std']

            speaker = sample['speaker']
            culture = sample['culture']
            language_raw, text_language_raw, language_encoding = self._encode_language(sample)

            # Speaker and culture encodings
            if speaker not in self.speaker_encodings:
                raise ValueError(f"Speaker {speaker} not in speaker_to_idx mapping.")
            if culture not in self.culture_encodings:
                raise ValueError(f"Culture {culture} not in culture_to_idx mapping.")

            speaker_encoding = torch.tensor(self.speaker_encodings[speaker], dtype=torch.int64)
            culture_encoding = torch.tensor(self.culture_encodings[culture], dtype=torch.int64)
            # Labels or metadata
            labels = {
                'culture_enc': culture_encoding,
                'speaker_enc': speaker_encoding,
                'culture_raw': sample['culture'],
                'speaker_raw': sample['speaker'],
                'text_data': sample.get('translated_text', sample.get('text_data', '')) if self.use_translated_text else sample.get('text_data', ''),
                'text_data_original': sample.get('text_data', ''),
                'translated_text': sample.get('translated_text', sample.get('text_data', '')),
                'translation_source': sample.get('translation_source', 'identity'),
                'language_raw': language_raw,
                'text_language_raw': text_language_raw,
                'language_enc': language_encoding,
                'text_feature_key': text_feature_key,
                'sample_start': sample['sample_start'],
                'sample_end': sample['sample_end'],
                'motion_fps': sample['motion_fps'],
                'scene_fps': sample['scene_fps']
            }
        else:
            # Here motion is (75,54), i.e. 5 seconds 15 fps 6D representation of 9 joints 6 relative rotations
            # No normalization is needed, rotations are already (0, 2pi). Avoid over-normalization
            speaker = sample['speaker']
            culture = sample['culture']
            language_raw, text_language_raw, language_encoding = self._encode_language(sample)

            # Speaker and culture encodings
            if speaker not in self.speaker_encodings:
                raise ValueError(f"Speaker {speaker} not in speaker_to_idx mapping.")
            if culture not in self.culture_encodings:
                raise ValueError(f"Culture {culture} not in culture_to_idx mapping.")

            speaker_encoding = torch.tensor(self.speaker_encodings[speaker], dtype=torch.int64)
            culture_encoding = torch.tensor(self.culture_encodings[culture], dtype=torch.int64)

            labels = {
                'culture_enc': culture_encoding,
                'speaker_enc': speaker_encoding,  # Assuming speaker is
                'language_raw': language_raw,
                'text_language_raw': text_language_raw,
                'language_enc': language_encoding,
                'sample_start': sample['sample_start'],
                'sample_end': sample['sample_end'],
                'motion_fps': sample['motion_fps'],
                'scene_fps': sample['scene_fps']
            }
            motion = sample['motion']
            motion = torch.tensor(motion, dtype=torch.float32).reshape(75, 54)

        if not self.motion_only:
            return motion, text_features, audio_mels, audio_onsets, audio_wav2vec, labels
        else:
            return motion, labels


# Load training, validation and test keys of the two possible data splits, i.e. subject dependent and subject independent
def load_dataset_splits(save_path='',n_speakers = False):
    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            data = pickle.load(f)
            print(f"Loaded dataset splits from {save_path}")
            if n_speakers:
                return data['train_keys'], data['val_keys'], data['test_keys'], \
                    data['n_train_speakers'], data['n_val_speakers'], data['n_test_speakers']
            else:
                return data['train_keys'], data['val_keys'], data['test_keys']
    else:
        if n_speakers:
            return None, None, None, None, None, None
        else:
            return None, None, None

# To save keys of the data splits
def save_dataset_splits(train_keys, val_keys, test_keys, save_path='',n_speakers = ()):
    with open(save_path, 'wb') as f:
        if not n_speakers:
            pickle.dump({
                'train_keys': train_keys,
                'val_keys': val_keys,
                'test_keys': test_keys
            }, f)
        else:
            pickle.dump({
                'train_keys': train_keys,
                'val_keys': val_keys,
                'test_keys': test_keys,
                'n_train_speakers': n_speakers[0],
                'n_val_speakers':n_speakers[1],
                'n_test_speakers': n_speakers[2]
            }, f)
    print(f"Dataset splits saved to {save_path}")

# To load keys of the data splits
def load_samples(dataset_splits_path=''):
    """
    Load all sample keys from dataset splits.

    :param dataset_splits_path: Path to the dataset splits file.
    :return: Tuple of (sample_keys, culture_speakers, samples_counter) or (None, None, None) if file doesn't exist.
    """
    if os.path.exists(dataset_splits_path):
        with open(dataset_splits_path, 'rb') as f:
            dataset = pickle.load(f)
        sample_keys = dataset['train_keys'] + dataset['val_keys'] + dataset['test_keys']
        # Optionally, load culture_speakers and samples_counter if needed
        return sample_keys, None, None
    else:
        return None, None, None

# For pickle datasets (not used)
'''
def save_dataset(dataset_path, train_samples, val_samples, test_samples):
    dataset = {
        'train': train_samples,
        'val': val_samples,
        'test': test_samples
    }
    with open(dataset_path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Dataset saved to {dataset_path}")


def load_dataset(dataset_path):
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    print(f"Dataset loaded from {dataset_path}")
    return dataset['train'], dataset['val'], dataset['test']
'''

# Function used to analyze the dataset, such as video lengths, FPS, number of scenes, number of skeletons, etc.
def dataset_analysis(playlist_folder,metadata_path):
    import cv2

    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
        sample_keys = metadata['sample_keys']
        culture_speakers = metadata['culture_speakers']

    for culture in culture_speakers.keys():
        print(f"culture {culture} has {len(culture_speakers[culture])} speakers")

    playlist_video_lengths = {}
    playlist_video_fps_length = {}
    playlist_nscenes = {}
    poses_fps = 15
    speaker_nposes = {}
    n_skipped = 0
    # Dictionaries to collect per-video data
    culture_video_lengths = {}  # { culture: [list of video lengths in seconds] }
    culture_video_fps = {}      # { culture: [list of FPS values] }
    culture_video_scenes = {}
    culture_video_skeletons = {}
    culture_scene_lengths = {}


    video_processing_counter = 0
    for playlist in os.listdir(playlist_folder):
        playlist_path = os.path.join(playlist_folder, playlist)
        if not os.path.isdir(playlist_path):
            continue

        # Derive culture from folder name and validate it against metadata.
        # Supports both metadata keys like "indian" and "indian_ted_hindi_language".
        playlist_name = playlist.strip().lower()
        playlist_short = playlist_name.split('_')[0]
        if playlist_short in culture_speakers:
            culture = playlist_short
        elif playlist_name in culture_speakers:
            culture = playlist_name
        else:
            continue

        n_skeletons = 0
        n_real_skeletons = 0
        n_scenes = 0
        video_lengths = []
        video_fps = []

        if culture not in culture_video_lengths:
            culture_video_lengths[culture] = []
        if culture not in culture_video_fps:
            culture_video_fps[culture] = []
        if culture not in culture_video_scenes:
            culture_video_scenes[culture] = []
        if culture not in culture_video_skeletons:
            culture_video_skeletons[culture] = []
        if culture not in culture_scene_lengths:
            culture_scene_lengths[culture] = []

        for speaker in culture_speakers[culture]:
            video_folder = speaker
            video_folder_path = os.path.join(playlist_path, video_folder)
            if not os.path.isdir(video_folder_path):
                continue

            print(f"Processing video {video_folder}, culture {culture}, number {video_processing_counter}")
            video_processing_counter += 1
            video_mp4 = video_folder + "_video.mp4"
            path_to_video_mp4 = os.path.join(video_folder_path, video_mp4)

            real_fps = None
            if os.path.exists(path_to_video_mp4):
                # Extract video length using cv2
                cap = cv2.VideoCapture(path_to_video_mp4)
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    if fps > 0:
                        real_fps = fps
                        video_length = frame_count / fps
                        video_lengths.append(video_length)
                        video_fps.append(fps)
                        culture_video_lengths[culture].append(video_length)
                        culture_video_fps[culture].append(fps)
                        playlist_video_fps_length[speaker] = (fps, frame_count)
                cap.release()

            n_skeletons_video = 0
            n_scenes_video = 0
            all_motion_data_dir = os.path.join(video_folder_path, "all_motion_data")
            if not os.path.isdir(all_motion_data_dir):
                continue

            for file in os.listdir(all_motion_data_dir):
                if "motion" not in file:
                    continue

                path_to_file = os.path.join(all_motion_data_dir, file)
                try:
                    with open(path_to_file, "rb") as f:
                        n_scenes += 1
                        n_scenes_video += 1
                        motion_data = pickle.load(f)
                        motion = motion_data['data']
                        n_real_skeletons += len(motion)
                        real_start = motion_data['real_start']
                        real_end = motion_data['real_end']
                        scene_fps = float(motion_data.get('scene_fps', real_fps if real_fps else poses_fps))
                        if scene_fps <= 0:
                            scene_fps = poses_fps
                        scene_duration = (real_end - real_start) / scene_fps
                        culture_scene_lengths[culture].append(scene_duration)

                        downsample_factor = max(1, round(scene_fps / poses_fps))
                        original_pose_fps = scene_fps / downsample_factor

                        # Resample motion data to fixed pose_fps
                        resampled_motion = resample_motion(motion, original_pose_fps, poses_fps)
                        n_skeletons += len(resampled_motion)

                        speaker_nposes[speaker] = (n_real_skeletons, n_skeletons)
                        n_skeletons_video += len(resampled_motion)

                except (pickle.UnpicklingError, KeyError, ValueError, FileNotFoundError, EOFError) as e:
                    # Log the error and skip the problematic file
                    print(f"Error processing file {path_to_file}: {e}")
                    n_skipped += 1
                    continue
            culture_video_scenes[culture].append(n_scenes_video)
            culture_video_skeletons[culture].append(n_skeletons_video)
        all_scene_lenghts = [(culture_name, np.sum(culture_scene_lengths[culture_name])) for culture_name in culture_scene_lengths.keys()]
        tot_scenes_lengths = np.sum([np.sum(culture_scene_lengths[culture_name]) for culture_name in culture_scene_lengths.keys()])
        # Compute statistics for the playlist
        if video_lengths:
            tot_length = np.sum(video_lengths)
            tot_length_minutes = tot_length / 60
            tot_length_hours = tot_length / 3600
            avg_length = np.mean(video_lengths)
            max_length = np.max(video_lengths)
            min_length = np.min(video_lengths)
            std_dev_length = np.std(video_lengths)
            avg_fps = np.mean(video_fps)
            std_fps = np.std(video_fps)
            playlist_video_lengths[playlist] = {
                'tot_length': tot_length,
                'tot_scene_length': tot_scenes_lengths / 3600,
                'all_scene_length': all_scene_lenghts,
                'tot_length_minutes': tot_length_minutes,
                'tot_length_hours': tot_length_hours,
                'min_length': min_length,
                'max_length': max_length,
                'tot_scenes': n_scenes,
                'n_real_skeletons': n_real_skeletons,
                'n_skeletons': n_skeletons,
                'average_length': avg_length,
                'std_deviation': std_dev_length,
                'average_fps': avg_fps,
                'std_fps': std_fps,
                'video_count': len(video_lengths)
            }

    print("Dataset Analysis:",playlist_video_lengths)
    print("Speakers skeletons:", speaker_nposes)
    return culture_video_lengths, culture_video_fps, culture_video_scenes, culture_video_skeletons, culture_scene_lengths

# Visualize top k gestures
def top_k_visualizer(data, k, save_path='', filename=''):
    """
    Expects `data` to be a dictionary mapping codebook vectors
    to a dictionary of cultures and their frequencies.
    Visualizes the top-k codebook frequencies for each culture.
    """
    # Dynamically extract all possible categories (cultures) from the dataset
    all_cultures = set(cat for vec in data.values() for cat in vec)

    # Initialize a dictionary to hold the top k frequencies for each culture
    top_k_data = {culture: [] for culture in all_cultures}

    # Process each culture
    for culture in all_cultures:
        # If the keys in data are strings (i.e. vector identifiers), we iterate accordingly.
        # Here we assume data keys are the codebook vector (as tuple of numbers)
        freqs = [(vector, data[vector].get(culture, 0)) for vector in data]
        # Sort by frequency and keep the top k
        top_freqs = sorted(freqs, key=lambda x: x[1], reverse=True)[:k]
        # Update the data structure for visualization
        top_k_data[culture] = top_freqs

    # Visualization: create one subplot per culture
    fig, axes = plt.subplots(1, len(all_cultures), figsize=(5 * len(all_cultures), 10))

    # In case there is only one culture, make sure axes is iterable.
    if len(all_cultures) == 1:
        axes = [axes]

    for ax, culture in zip(axes, all_cultures):
        names = [str(vec[0]) for vec in top_k_data[culture]]
        values = [vec[1] for vec in top_k_data[culture]]
        ax.bar(names, values)
        ax.set_title(f'Top {k} codebook vectors in {culture}')
        ax.set_xlabel('Codebook vectors')
        ax.set_ylabel('Frequency')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=90)

    plt.tight_layout()
    if save_path and filename:
        plt.savefig(os.path.join(save_path, filename))
    plt.show()


# Collect target_count codebooks for each culture to have a balanced number of samples for each culture
def collect_codebooks_from_batches(train_loader, target_count=100000):
    """
    Iterates over batches from train_loader and collects codebook vectors (from motion)
    for each culture (using the culture name from labels['culture_raw']) until at least
    `target_count` vectors are collected per culture.

    Returns:
        poses_data: dict mapping culture (str) to a list of NumPy arrays (each of shape (25, 512))
                    collected from various samples.
    """
    poses_data = {}  # culture (str) -> list of codebook arrays (each sample, shape: (25, 512))
    culture_counts = {}  # culture (str) -> count of codebook vectors collected so far

    # Iterate over batches from the training loader
    for batch in train_loader:
        motion, text_features, audio_mels, audio_onsets, audio_wav2vec, labels = batch
        batch_size = motion.shape[0]
        # Labels is a dictionary in which each value is an indexable container (e.g., a list)
        for i in range(batch_size):
            # Choose a key for culture. We can use either the integer or the raw culture name.
            # Here we use the raw culture name.
            culture = labels['culture_raw'][i]

            # Initialize data structures when finding the culture for the first time
            if culture not in poses_data:
                poses_data[culture] = []
                culture_counts[culture] = 0

            # If we have reached the target for this culture, skip this sample.
            if culture_counts[culture] >= target_count:
                continue

            # Extract the codebook vectors for the i-th sample.
            # motion[i] has shape (25, 512). (Assuming the tensor is already on CPU, otherwise call .cpu().numpy())
            codebooks = motion[i].cpu().numpy() if hasattr(motion[i], 'cpu') else motion[i]

            # Append the sample’s codebooks to the culture's list.
            poses_data[culture].append(codebooks)

            # Update count – note that each sample contributes 25 codebook vectors.
            culture_counts[culture] += codebooks.shape[0]

        # Optionally, check if all cultures have reached the target and break early.
        if poses_data and all(count >= target_count for count in culture_counts.values()):
            break

    # Optionally, print out the counts per culture.
    for culture, count in culture_counts.items():
        print(f"Collected {count} codebook vectors for culture {culture}")

    return poses_data


# Count gesture codebooks for top-k codebook analysis
def gestures_counter(poses_data, save_path):
    """
    Given poses_data as a dictionary mapping each culture to a list of NumPy arrays (each array
    represents codebook vectors from one sample), this function accumulates the frequency of each
    unique codebook vector per culture and performs visualization.
    """
    # Ensure the output directory exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Dictionary to store counts for each unique codebook vector across cultures.
    all_embedding_instances = {}

    # Mapping each unique codebook vector to a unique index (for labeling)
    vec_quantizedvec_mapping = {}

    # Process each culture and each sample within that culture.
    for culture in poses_data.keys():
        for codebook_array in poses_data[culture]:  # codebook_array shape: (25, 512)
            # Convert the array to a tuple of tuples so it is hashable (each row is a code vector)
            # If you need to count each code vector individually, iterate over rows.
            codebook_tuples = tuple(map(tuple, codebook_array))  # length should be 25
            for vector in codebook_tuples:
                if vector not in all_embedding_instances:
                    all_embedding_instances[vector] = {}
                if culture not in all_embedding_instances[vector]:
                    all_embedding_instances[vector][culture] = 0
                all_embedding_instances[vector][culture] += 1

    # Create a mapping from each unique codebook vector (hashable) to a unique index for labelling.
    for i, vector in enumerate(all_embedding_instances.keys()):
        vec_quantizedvec_mapping[vector] = i

    # Compute frequencies for each culture
    # This dictionary will map culture -> { vector: frequency, ... }
    gesture_frequencies = {culture: {} for culture in poses_data.keys()}
    for vector, cultures in all_embedding_instances.items():
        for culture, count in cultures.items():
            gesture_frequencies[culture][vector] = count

    # Combine frequencies to find the global top-10 codebook vectors
    combined_frequencies = {}
    for culture, gestures in gesture_frequencies.items():
        for vector, count in gestures.items():
            if vector not in combined_frequencies:
                combined_frequencies[vector] = 0
            combined_frequencies[vector] += count

    top_k = 10
    top_k_gestures = sorted(combined_frequencies, key=combined_frequencies.get, reverse=True)[:top_k]

    # Build comparative frequencies across cultures for these top-k codebook vectors.
    # Here, we use the culture names as keys.
    comparative_frequencies = {culture: [] for culture in poses_data.keys()}
    for vector in top_k_gestures:
        for culture in poses_data.keys():
            comparative_frequencies[culture].append(gesture_frequencies[culture].get(vector, 0))

    # Create labels for the x-axis using the unique mapping.
    gesture_labels = [f"v_{vec_quantizedvec_mapping[vector]}" for vector in top_k_gestures]
    print("Top-k gesture labels:", gesture_labels)

    # Plot the comparative frequencies using a Pandas DataFrame.
    df = pd.DataFrame(comparative_frequencies, index=gesture_labels)
    ax = df.plot.bar(figsize=(10, 6), width=0.4)
    plt.xticks(rotation=45)
    plt.ylabel('Frequency')
    plt.title('Comparative Frequencies for Top-10 Codebook Vectors Across Cultures')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "comparative_gestures.png"))
    plt.close()

    # Optionally, also visualize the top-10 gestures for each culture separately.
    for culture in poses_data.keys():
        top_vectors = sorted(gesture_frequencies[culture], key=gesture_frequencies[culture].get, reverse=True)[:10]
        top_counts = [gesture_frequencies[culture][vector] for vector in top_vectors]

        plt.figure(figsize=(10, 6))
        plt.bar([f"v_{vec_quantizedvec_mapping[vector]}" for vector in top_vectors],
                top_counts,
                width=0.4)
        plt.xticks(rotation=45)
        plt.ylabel('Frequency')
        plt.title(f'Top-10 Codebook Vectors for {culture}')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{culture}_top_10_gestures.png"))
        plt.close()

    # Finally, you may call the top_k_visualizer function to get a global overview.
    top_k_visualizer(all_embedding_instances, top_k, save_path, filename="top_10_gestures.png")


def cli_main(argv: Optional[List[str]] = None) -> None:
    """Command-line entry point.

    Typical usage (LMDB dataset build):
        python dataset.py build --playlists-folder /path/to/playlists --motion-only

    If you omit --dataset-path / --metadata-path, sensible defaults are used inside the playlists folder.
    """
    parser = argparse.ArgumentParser(description="Build and manage the gesture dataset (LMDB).")

    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Build an LMDB dataset from processed poses (all_motion_data).")
    p_build.add_argument("--playlists-folder", required=True, help="Folder that contains culture/language playlists.")
    p_build.add_argument("--motion-only", action="store_true", help="Store only motion windows (no audio/text).")

    p_build.add_argument("--duration", type=float, default=5.0, help="Window duration in seconds.")
    p_build.add_argument("--stride", type=float, default=0.5, help="Stride in seconds (will be rounded to frames at 15 fps).")
    p_build.add_argument("--target-sr", type=int, default=16000, help="Target audio sampling rate.")

    p_build.add_argument("--dataset-path", default=None, help="Output LMDB path. Default depends on --motion-only.")
    p_build.add_argument("--metadata-path", default=None, help="Output metadata pickle path.")
    p_build.add_argument("--initial-size-gb", type=float, default=None, help="Initial LMDB size in GB.")
    p_build.add_argument(
        "--allow-machine-translation",
        action="store_true",
        help="When English subtitles are missing, translate extracted text to English before encoding.",
    )

    p_analyze = sub.add_parser("analyze", help="Run dataset analysis utilities.")
    p_analyze.add_argument("--playlists-folder", required=True)
    p_analyze.add_argument("--metadata-path", required=True)

    p_rebuild = sub.add_parser("rebuild-metadata", help="Rebuild metadata from an existing LMDB dataset (no sample rebuild).")
    p_rebuild.add_argument("--dataset-path", required=True, help="Path to existing LMDB dataset.")
    p_rebuild.add_argument("--metadata-path", required=True, help="Folder where metadata.pkl and encodings are written.")
    p_rebuild.add_argument(
        "--skip-language-encodings",
        action="store_true",
        help="Skip scanning LMDB for language encodings (writes {'unknown': 0} instead).",
    )

    args = parser.parse_args(argv)

    # Basic logging setup for CLI use. Library users can configure logging separately.
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    if args.command == "build":
        playlists_folder = args.playlists_folder
        motion_only = bool(args.motion_only)
        duration = float(args.duration)
        stride = float(args.stride)
        target_sr = int(args.target_sr)

        if args.dataset_path is None:
            dataset_path = os.path.join(playlists_folder, 'gesture_dataset' if motion_only else 'whole_dataset')
        else:
            dataset_path = args.dataset_path

        if args.metadata_path is None:
            metadata_path = os.path.join(playlists_folder, 'gesture_dataset_metadata.pkl' if motion_only else 'whole_dataset_metadata.pkl')
        else:
            metadata_path = args.metadata_path

        # Motion-only is much smaller.
        if args.initial_size_gb is None:
            initial_size_gb = 10 if motion_only else 50
        else:
            initial_size_gb = float(args.initial_size_gb)

        collect_samples(
            playlists_folder=playlists_folder,
            duration=duration,
            stride=stride,
            target_sr=target_sr,
            metadata_path=metadata_path,
            dataset_path=dataset_path,
            motion_only=motion_only,
            initial_size_db=initial_size_gb,
            allow_machine_translation=bool(args.allow_machine_translation),
        )

    elif args.command == "analyze":
        dataset_analysis(args.playlists_folder, args.metadata_path)
    elif args.command == "rebuild-metadata":
        rebuild_metadata_from_lmdb(
            dataset_path=args.dataset_path,
            metadata_path=args.metadata_path,
            include_language_encodings=not bool(args.skip_language_encodings),
        )


def main():
    # External scripts that call main() will get the CLI behavior.
    cli_main()


if __name__ == '__main__':
    cli_main()
