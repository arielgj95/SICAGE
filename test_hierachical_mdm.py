import copy
import functools
import time
import torch
import tqdm
import os
import json
import argparse
import yaml
import pickle
import numpy as np
import sys
import importlib
import traceback

from easydict import EasyDict
from collections import OrderedDict
import warnings, math
from typing import Tuple, Optional, Dict, Any, List
from scipy import linalg, stats
import re

#from process_video.MotionBERT.infer_wild_mesh import checkpoint
# Keep JIT enabled by default so paper-style BAS beat tracking can run.
# Users can still override by exporting NUMBA_DISABLE_JIT=1 explicitly.
os.environ.setdefault("NUMBA_DISABLE_JIT", "0")

from mdm_generator.diffusion.utils.fixseed import fixseed
from mdm_generator.diffusion.utils import dist_util
from functools import partial
from mdm_generator.hierarchical_mdm import Hierarchical_MDM
from dataset import prepare_data
from tqdm.auto import tqdm
from mdm_generator.diffusion.utils.model_util import create_gaussian_diffusion as create_hierarchical_gaussian_diffusion

from visualize_vqvae_data import sample_generation_from_codebooks
import visualize_vqvae_data as vq_vis
from mdm_generator.eval import eval_ted4cl
from mdm_generator.eval.eval_ted4cl_new import evaluate_all_metrics, evaluate_all_metrics_bulk
from mdm_generator.diffusion.utils.model_util import load_model
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import torch.nn as nn
import random

from vq_vae.vqvae import VQVAE

BaselineMDM = None
create_baseline_gaussian_diffusion = None
load_baseline_model = None
_baseline_import_error = None
_baseline_import_source = None


def _load_baseline_modules() -> bool:
    """
    Load baseline modules with robust fallbacks.
    This avoids requiring a specific PYTHONPATH layout.
    """
    global BaselineMDM, create_baseline_gaussian_diffusion, load_baseline_model
    global _baseline_import_error, _baseline_import_source

    if (
        BaselineMDM is not None
        and create_baseline_gaussian_diffusion is not None
        and load_baseline_model is not None
    ):
        return True

    errors: List[str] = []

    import_attempts = [
        ("diffustylegesture_and_mdm package import", None, "diffustylegesture_and_mdm.model.mdm", "diffustylegesture_and_mdm.utils.model_util"),
        (
            "diffustylegesture_and_mdm local-path import",
            os.path.join(os.path.abspath(os.path.dirname(__file__)), "diffustylegesture_and_mdm"),
            "model.mdm",
            "utils.model_util",
        ),
    ]

    for source_name, extra_path, mdm_mod_name, util_mod_name in import_attempts:
        path_added = False
        try:
            if extra_path and os.path.isdir(extra_path) and extra_path not in sys.path:
                sys.path.insert(0, extra_path)
                path_added = True

            mdm_module = importlib.import_module(mdm_mod_name)
            util_module = importlib.import_module(util_mod_name)

            BaselineMDM = getattr(mdm_module, "MDM")
            create_baseline_gaussian_diffusion = getattr(util_module, "create_gaussian_diffusion")
            load_baseline_model = getattr(util_module, "load_model_wo_clip")
            _baseline_import_source = source_name
            _baseline_import_error = None
            print(f"[Eval] Baseline modules loaded via: {source_name}")
            return True
        except Exception as exc:
            errors.append(
                f"- {source_name}: {type(exc).__name__}: {exc}\n"
                f"{traceback.format_exc(limit=8)}"
            )
            if path_added:
                try:
                    sys.path.remove(extra_path)
                except ValueError:
                    pass

    _baseline_import_error = "\n".join(errors)
    return False


seed = 10
global_decode_metadata_mode = "uninitialized"
EVAL_WINDOW_SECONDS = 4.0
EVAL_POSE_FPS = 15
EVAL_AUDIO_ONSET_HZ = 31.25
EVAL_MOTION_STEPS_4S = 20
EVAL_POSE_FRAMES_4S = int(EVAL_WINDOW_SECONDS * EVAL_POSE_FPS)
EVAL_ONSET_BINS_4S = int(round(EVAL_WINDOW_SECONDS * EVAL_AUDIO_ONSET_HZ))

# For reproducibility to have the same samples
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)

def _get_vqvae_core(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def _default_skeleton_links_for_keypoints(keypoint_indices: List[int]) -> List[Tuple[int, int]]:
    # H36M-style upper-body tree used in this codebase.
    parent_by_h36m = {
        7: -1,   # spine-centre (root)
        8: 7,    # spine-upper
        9: 8,    # neck
        11: 8,   # left-shoulder
        12: 11,  # left-elbow
        13: 12,  # left-wrist
        14: 8,   # right-shoulder
        15: 14,  # right-elbow
        16: 15,  # right-wrist
    }
    links = []
    for child in keypoint_indices:
        parent = parent_by_h36m.get(int(child), -1)
        if parent in keypoint_indices:
            links.append((int(parent), int(child)))
    if links:
        return links
    # Generic fallback when keypoints differ from the expected subset.
    return [(int(keypoint_indices[i - 1]), int(keypoint_indices[i])) for i in range(1, len(keypoint_indices))]


def _default_segment_lengths_for_keypoints(keypoint_indices: List[int], skeleton_links: List[Tuple[int, int]]) -> np.ndarray:
    child_to_parent = {int(child): int(parent) for parent, child in skeleton_links}
    lengths = np.ones((len(keypoint_indices),), dtype=np.float32)
    for i, h36m_id in enumerate(keypoint_indices):
        if child_to_parent.get(int(h36m_id), -1) not in keypoint_indices:
            lengths[i] = 0.0
    return lengths


def load_vqvae_model(args, model_path):

    global global_vqvae_model, global_info_data, global_links_len, global_keypoint_indices, global_decode_metadata_mode
    keypoint_indices = getattr(args, "keypoint_indices", [7, 8, 9, 14, 15, 16, 11, 12, 13])
    links_len = None
    info_data = None

    links_len_file = getattr(args, "speaker_link_len", None)
    if links_len_file and os.path.exists(links_len_file):
        with open(links_len_file, "rb") as file:
            links_len = pickle.load(file)

    info_file = getattr(args, "skeleton_info_path", None)
    if info_file and os.path.exists(info_file):
        with open(info_file, "rb") as file:
            info_raw = pickle.load(file)
            info_data = info_raw.get("meta_info", info_raw) if isinstance(info_raw, dict) else info_raw

    decode_mode = "exact"
    if info_data is None or "skeleton_links" not in info_data:
        default_links = _default_skeleton_links_for_keypoints(keypoint_indices)
        info_data = {"skeleton_links": default_links}
        decode_mode = "fallback_default_skeleton"
    if links_len is None:
        links_len = _default_segment_lengths_for_keypoints(keypoint_indices, info_data["skeleton_links"])
        decode_mode = (
            "fallback_default_skeleton_and_lengths"
            if decode_mode != "exact"
            else "fallback_default_lengths"
        )

    mydevice = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        model = VQVAE(args.VQVAE, 9 * 6)  # n_joints * n_channels
        if torch.cuda.is_available():
            no_cuda_ids = getattr(args, "no_cuda", [getattr(args, "gpu", "0")])
            model = nn.DataParallel(model, device_ids=[eval(i) for i in no_cuda_ids])
        model = model.to(mydevice)
        checkpoint = torch.load(model_path, map_location=mydevice)
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
            # Backward compatibility fallback for minor key drifts.
            incompatible = model.load_state_dict(state_dict, strict=False)
            if incompatible.missing_keys:
                print(f"[VQ-VAE] Missing keys while loading checkpoint: {incompatible.missing_keys}")
            if incompatible.unexpected_keys:
                print(f"[VQ-VAE] Unexpected keys while loading checkpoint: {incompatible.unexpected_keys}")
        model.eval()

    global_vqvae_model = model
    global_info_data = info_data
    global_links_len = links_len
    global_keypoint_indices = keypoint_indices
    global_decode_metadata_mode = decode_mode

    return model,info_data,links_len,keypoint_indices

class GeneratedDataset(Dataset):
    def __init__(self, *samples):
        """
        Initialize the dataset with the list of generated samples.
        """
        self.samples = samples

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return self.samples[0].size(0)

    def __getitem__(self, index):
        """
        Retrieve a sample by index.
        """
        return tuple(tensor[index] for tensor in self.samples)


def _compute_activation_statistics_pt(x: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Computes mean and sample covariance for features using PyTorch."""
    # (Implementation from the previous answer)
    if x.dim() == 3:
        num_samples, time_dim, feature_dim = x.shape
        x_flat = x.reshape(-1, feature_dim)
    elif x.dim() == 2:
        num_samples, feature_dim = x.shape
        x_flat = x
    else:
        warnings.warn(f"Input tensor has unexpected dimension {x.dim()}. Skipping.")
        return None, None

    if x_flat.shape[0] < 2:
         # warnings.warn(f"Not enough samples ({x_flat.shape[0]}) to compute covariance. Need >= 2.")
         return None, None # Return None gracefully

    mu = torch.mean(x_flat, dim=0)
    diff = x_flat - mu
    cov = (diff.t() @ diff) / (x_flat.shape[0] - 1)
    return mu, cov

def _calculate_frechet_distance_np(mu1, sigma1, mu2, sigma2, eps=1e-6) -> Optional[float]:
    """Numpy implementation of the Frechet Distance using scipy.linalg.sqrtm."""
    # (Implementation from the previous answer, including stability checks)
    mu1 = np.asarray(mu1, dtype=np.float64)
    sigma1 = np.asarray(sigma1, dtype=np.float64)
    mu2 = np.asarray(mu2, dtype=np.float64)
    sigma2 = np.asarray(sigma2, dtype=np.float64)

    if mu1.shape != mu2.shape or sigma1.shape != sigma2.shape: return None

    diff = mu1 - mu2
    diff_sq_norm = np.dot(diff, diff)

    try:
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    except Exception:
         offset = np.eye(sigma1.shape[0]) * eps
         try:
             covmean, _ = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset), disp=False)
         except Exception as e2:
             warnings.warn(f"sqrtm failed even after adding epsilon: {e2}. Returning None.")
             return None

    if not np.isfinite(covmean).all():
        warnings.warn(f"FID calculation produces non-finite elements in sqrtm result. Returning None.")
        return None

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            # warnings.warn(f"sqrtm result has non-negligible imaginary component. Returning None.")
            # return None # Be strict
             pass # Or proceed cautiously if assuming it's minor error
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    trace_component = np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    fd = diff_sq_norm + trace_component
    return float(max(0.0, fd)) # Clamp at 0

def calculate_overall_fgd(real_encodings: torch.Tensor, gen_encodings: torch.Tensor) -> Optional[float]:
    """Calculates FGD between two sets of continuous encodings."""
    if real_encodings is None or gen_encodings is None:
        return None
    if real_encodings.shape[0] < 2 or gen_encodings.shape[0] < 2:
        # warnings.warn("Not enough samples for FGD calculation.")
        return None

    mu_real, cov_real = _compute_activation_statistics_pt(real_encodings)
    mu_gen, cov_gen = _compute_activation_statistics_pt(gen_encodings)

    if mu_real is None or cov_real is None or mu_gen is None or cov_gen is None:
        # warnings.warn("Could not compute statistics for FGD.")
        return None

    # Convert to NumPy on CPU for robust calculation
    mu1_np = mu_real.detach().cpu().numpy()
    cov1_np = cov_real.detach().cpu().numpy()
    mu2_np = mu_gen.detach().cpu().numpy()
    cov2_np = cov_gen.detach().cpu().numpy()

    return _calculate_frechet_distance_np(mu1_np, cov1_np, mu2_np, cov2_np)


def _diffusion_accepts_culture_labels(diffusion) -> bool:
    return str(diffusion.__class__.__module__).startswith("mdm_generator.")


def _build_diffusion_inputs(batch, device, include_culture_labels: bool):
    motion, text_features, audio_mels, audio_onsets, audio_wav2vec, labels = batch
    in_data = [
        motion.to(device),
        text_features.to(device),
        audio_mels.to(device),
        audio_onsets.to(device),
        audio_wav2vec.to(device),
    ]
    if include_culture_labels:
        in_data.append(labels["culture_enc"].to(device))
    return in_data

def generate_during_training(
        val_data,
        test_data,
        device,
        diffusion,
        mdm_model,
        save_path,
        vqvae_model,
        vqvae_config,
        use_alignment: bool = False,
        max_samples: int = 1500,
        decode_3d: bool = False,
):

    global global_decode_metadata_mode
    plot_data = None
    generated_samples = []
    real_samples = []
    source_loader = test_data if test_data is not None else val_data
    if source_loader is None:
        raise ValueError("No source loader provided for sample generation.")

    mdm_model.eval()
    vqvae_core = _get_vqvae_core(vqvae_model)
    can_decode_3d = decode_3d and (global_info_data is not None) and (global_links_len is not None)
    decoded_3d_available = False
    decode_3d_mode = global_decode_metadata_mode if decode_3d else "disabled"
    if decode_3d and not can_decode_3d:
        warnings.warn("3D decoding requested but metadata is unavailable. Using placeholder 3D poses.")
    include_culture_labels = _diffusion_accepts_culture_labels(diffusion)

    collected_samples = 0
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(source_loader, desc="Creating generated subsets...", position=1, leave=True)):
            if max_samples is not None and collected_samples >= max_samples:
                break

            motion, text_features, audio_mels, audio_onsets, audio_wav2vec, labels = batch
            culture_real = labels['culture_enc']
            speaker_real = labels['speaker_enc']

            real_codebooks = motion.permute(0, 2, 1).to(device)
            zs_real = vqvae_core.bottleneck.encode([real_codebooks])
            real_cont_encoding = vqvae_core.bottleneck.decode(zs_real)[0]
            real_cont_encoding = real_cont_encoding.permute(0, 2, 1)[:, 5:, :]

            x_start = motion[:, 5:, :].to(device)
            in_data = _build_diffusion_inputs(batch, device, include_culture_labels)
            motion = in_data[0]

            new_motion, all_outputs = diffusion.p_sample_loop(mdm_model, in_data, clip_denoised=False)
            # Keep evaluation on a strict 4-second segment.
            motion_output = new_motion[:, 5:, :][:, :EVAL_MOTION_STEPS_4S, :]
            culture_output = None
            if isinstance(all_outputs, (tuple, list)) and len(all_outputs) > 1:
                culture_output = all_outputs[1]

            if can_decode_3d:
                try:
                    real_3d_poses, generated_3d_poses = sample_generation_from_codebooks(
                        vqvae_config,
                        new_motion,
                        motion,
                        save_path_real='',
                        save_path_fake='',
                        plot_poses=False
                    )
                    gen_3d = np.array(generated_3d_poses)[:, 15:, :, :]
                    real_3d = np.array(real_3d_poses)[:, 15:, :, :]
                    gen_3d = gen_3d[:, :EVAL_POSE_FRAMES_4S, :, :]
                    real_3d = real_3d[:, :EVAL_POSE_FRAMES_4S, :, :]
                    decoded_3d_available = True
                except Exception as e:
                    warnings.warn(f"3D pose decoding failed ({e}). Falling back to placeholder 3D tensors.")
                    can_decode_3d = False
            if not can_decode_3d:
                pose_len = min(int(motion_output.shape[1]), EVAL_POSE_FRAMES_4S)
                gen_3d = np.zeros((motion_output.shape[0], pose_len, 1, 3), dtype=np.float32)
                real_3d = np.zeros((motion_output.shape[0], pose_len, 1, 3), dtype=np.float32)

            gen_codebooks = new_motion.permute(0, 2, 1)
            zs_gen = vqvae_core.bottleneck.encode([gen_codebooks])
            gen_cont_encoding = vqvae_core.bottleneck.decode(zs_gen)[0]
            gen_cont_encoding = gen_cont_encoding.permute(0, 2, 1)[:, 5:, :][:, :EVAL_MOTION_STEPS_4S, :]
            real_cont_encoding = real_cont_encoding[:, :EVAL_MOTION_STEPS_4S, :]
            x_start = x_start[:, :EVAL_MOTION_STEPS_4S, :]
            onset_4s = audio_onsets[:, 31:][:, :EVAL_ONSET_BINS_4S]

            if use_alignment:
                if all_outputs is None or not isinstance(all_outputs, (tuple, list)) or len(all_outputs) < 5:
                    raise RuntimeError(
                        "Alignment evaluation requires model outputs "
                        "(culture/logits + pooled contexts), but current model did not return them."
                    )
                motion_output_projection_pooled = all_outputs[2]
                high_level_context = all_outputs[4]
                overall_audio_context = all_outputs[-1]
                real_culture_output = mdm_model.culture_classification_layer(x_start)
                batch_generated = (
                    gen_3d,
                    motion_output.detach().clone(),
                    culture_output.detach().clone(),
                    motion_output_projection_pooled.detach().clone(),
                    high_level_context.detach().clone(),
                    overall_audio_context.detach().clone(),
                    gen_cont_encoding.detach().clone(),
                    culture_real.detach().clone(),
                    speaker_real.detach().clone(),
                    onset_4s.detach().clone(),
                    text_features.detach().clone(),
                    audio_mels.detach().clone(),
                    audio_onsets.detach().clone(),
                    audio_wav2vec.detach().clone(),
                )
                new_motion_pooled = mdm_model.output_projection_pooler(x_start)
                real_batch = (
                    real_3d,
                    x_start.detach().clone(),
                    real_culture_output.detach().clone(),
                    new_motion_pooled.detach().clone(),
                    high_level_context.detach().clone(),
                    overall_audio_context.detach().clone(),
                    real_cont_encoding.detach().clone(),
                    culture_real.detach().clone(),
                    speaker_real.detach().clone(),
                    onset_4s.detach().clone(),
                    text_features.detach().clone(),
                    audio_mels.detach().clone(),
                    audio_onsets.detach().clone(),
                    audio_wav2vec.detach().clone(),
                )
            else:
                batch_generated = (
                    gen_3d,
                    motion_output.detach().clone(),
                    gen_cont_encoding.detach().clone(),
                    culture_real.detach().clone(),
                    speaker_real.detach().clone(),
                    onset_4s.detach().clone(),
                    text_features.detach().clone(),
                    audio_mels.detach().clone(),
                    audio_onsets.detach().clone(),
                    audio_wav2vec.detach().clone(),
                )
                real_batch = (
                    real_3d,
                    x_start.detach().clone(),
                    real_cont_encoding.detach().clone(),
                    culture_real.detach().clone(),
                    speaker_real.detach().clone(),
                    onset_4s.detach().clone(),
                    text_features.detach().clone(),
                    audio_mels.detach().clone(),
                    audio_onsets.detach().clone(),
                    audio_wav2vec.detach().clone(),
                )

            if idx == 0:
                plot_data = (new_motion, motion)

            generated_samples.append(batch_generated)
            real_samples.append(real_batch)
            collected_samples += motion_output.shape[0]

    data_loader_generated, data_loader_real = create_data_loaders(
        generated_samples, real_samples, use_alignment=use_alignment
    )
    return data_loader_generated, data_loader_real, plot_data, decoded_3d_available, decode_3d_mode


def create_data_loaders(generated_samples,real_samples,batch_size=64, use_alignment=False):
    # Helper function to stack tensors and create DataLoaders

    # ----------------------------
    # Process Generated Samples
    # ----------------------------
    data_loader_real = None
    data_loader_generated = None
    if generated_samples:
        # Unpack and concatenate each field across all batches

        if use_alignment:
            gen_3d = torch.cat([torch.from_numpy(batch[0]) for batch in generated_samples], dim=0)
            gen_motion_output = torch.cat([batch[1] for batch in generated_samples], dim=0)
            gen_culture_output = torch.cat([batch[2] for batch in generated_samples], dim=0)
            gen_motion_output_projection_pooled = torch.cat([batch[3] for batch in generated_samples], dim=0)
            #gen_low_level_context = torch.cat([batch[3] for batch in generated_samples], dim=0)
            gen_high_level_context = torch.cat([batch[4] for batch in generated_samples], dim=0)
            gen_overall_audio_context = torch.cat([batch[5] for batch in generated_samples], dim=0)
            #gen_labels = torch.cat([batch[6] for batch in generated_samples], dim=0)
            gen_cont_motion = torch.cat([batch[6] for batch in generated_samples], dim=0)
            gen_culture_labels = torch.cat([batch[7] for batch in generated_samples], dim=0)
            gen_speaker_labels = torch.cat([batch[8] for batch in generated_samples], dim=0)
            gen_onsets = torch.cat([batch[9] for batch in generated_samples], dim=0)
            has_mm_fields = len(generated_samples[0]) >= 14
            if has_mm_fields:
                gen_text_features = torch.cat([batch[10] for batch in generated_samples], dim=0)
                gen_audio_mels = torch.cat([batch[11] for batch in generated_samples], dim=0)
                gen_audio_onsets_full = torch.cat([batch[12] for batch in generated_samples], dim=0)
                gen_audio_wav2vec = torch.cat([batch[13] for batch in generated_samples], dim=0)


            # Create a TensorDataset
            generated_tensors = [
                gen_3d,
                gen_motion_output,
                gen_culture_output,
                gen_motion_output_projection_pooled,
                gen_high_level_context,
                gen_overall_audio_context,
                gen_cont_motion,
                gen_culture_labels,
                gen_speaker_labels,
                gen_onsets,
            ]
            if has_mm_fields:
                generated_tensors.extend([
                    gen_text_features,
                    gen_audio_mels,
                    gen_audio_onsets_full,
                    gen_audio_wav2vec,
                ])
            generated_dataset = GeneratedDataset(*generated_tensors)
        else:
            gen_3d = torch.cat([torch.from_numpy(batch[0]) for batch in generated_samples], dim=0)
            gen_motion_output = torch.cat([batch[1] for batch in generated_samples], dim=0)
            gen_cont_motion = torch.cat([batch[2] for batch in generated_samples], dim=0)
            gen_culture_labels = torch.cat([batch[3] for batch in generated_samples], dim=0)
            gen_speaker_labels = torch.cat([batch[4] for batch in generated_samples], dim=0)
            gen_onsets = torch.cat([batch[5] for batch in generated_samples], dim=0)
            has_mm_fields = len(generated_samples[0]) >= 10
            if has_mm_fields:
                gen_text_features = torch.cat([batch[6] for batch in generated_samples], dim=0)
                gen_audio_mels = torch.cat([batch[7] for batch in generated_samples], dim=0)
                gen_audio_onsets_full = torch.cat([batch[8] for batch in generated_samples], dim=0)
                gen_audio_wav2vec = torch.cat([batch[9] for batch in generated_samples], dim=0)

            # Create a TensorDataset
            generated_tensors = [
                gen_3d,
                gen_motion_output,
                gen_cont_motion,
                gen_culture_labels,
                gen_speaker_labels,
                gen_onsets,
            ]
            if has_mm_fields:
                generated_tensors.extend([
                    gen_text_features,
                    gen_audio_mels,
                    gen_audio_onsets_full,
                    gen_audio_wav2vec,
                ])
            generated_dataset = GeneratedDataset(*generated_tensors)

        # Create a DataLoader
        data_loader_generated = DataLoader(
            generated_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False
        )

    # ----------------------------
    # Process Real Samples
    # ----------------------------
    if real_samples:
        if use_alignment:
            # Unpack and concatenate each field across all batches
            real_3d = torch.cat([torch.from_numpy(batch[0]) for batch in real_samples], dim=0)
            real_x_start = torch.cat([batch[1] for batch in real_samples], dim=0)
            real_culture_real = torch.cat([batch[2] for batch in real_samples], dim=0)
            real_motion_pooled = torch.cat([batch[3] for batch in real_samples], dim=0)
            #real_low_level_context = torch.cat([batch[3] for batch in real_samples], dim=0)
            real_high_level_context = torch.cat([batch[4] for batch in real_samples], dim=0)
            real_overall_audio_context = torch.cat([batch[5] for batch in real_samples], dim=0)
            #real_labels = torch.cat([batch[6] for batch in real_samples], dim=0)
            real_cont_motion = torch.cat([batch[6] for batch in real_samples], dim=0)
            real_culture_labels = torch.cat([batch[7] for batch in real_samples], dim=0)
            real_speaker_labels = torch.cat([batch[8] for batch in real_samples], dim=0)
            real_onsets = torch.cat([batch[9] for batch in real_samples], dim=0)
            has_mm_fields = len(real_samples[0]) >= 14
            if has_mm_fields:
                real_text_features = torch.cat([batch[10] for batch in real_samples], dim=0)
                real_audio_mels = torch.cat([batch[11] for batch in real_samples], dim=0)
                real_audio_onsets_full = torch.cat([batch[12] for batch in real_samples], dim=0)
                real_audio_wav2vec = torch.cat([batch[13] for batch in real_samples], dim=0)
            # Handle labels if needed (optional)

            # Create a TensorDataset
            real_tensors = [
                real_3d,
                real_x_start,
                real_culture_real,
                real_motion_pooled,
                real_high_level_context,
                real_overall_audio_context,
                real_cont_motion,
                real_culture_labels,
                real_speaker_labels,
                real_onsets,
            ]
            if has_mm_fields:
                real_tensors.extend([
                    real_text_features,
                    real_audio_mels,
                    real_audio_onsets_full,
                    real_audio_wav2vec,
                ])
            real_dataset = GeneratedDataset(*real_tensors)
        else:
            # Unpack and concatenate each field across all batches
            real_3d = torch.cat([torch.from_numpy(batch[0]) for batch in real_samples], dim=0)
            real_x_start = torch.cat([batch[1] for batch in real_samples], dim=0)
            real_cont_motion = torch.cat([batch[2] for batch in real_samples], dim=0)
            real_culture_labels = torch.cat([batch[3] for batch in real_samples], dim=0)
            real_speaker_labels = torch.cat([batch[4] for batch in real_samples], dim=0)
            real_onsets = torch.cat([batch[5] for batch in real_samples], dim=0)
            has_mm_fields = len(real_samples[0]) >= 10
            if has_mm_fields:
                real_text_features = torch.cat([batch[6] for batch in real_samples], dim=0)
                real_audio_mels = torch.cat([batch[7] for batch in real_samples], dim=0)
                real_audio_onsets_full = torch.cat([batch[8] for batch in real_samples], dim=0)
                real_audio_wav2vec = torch.cat([batch[9] for batch in real_samples], dim=0)

            # Create a TensorDataset
            real_tensors = [
                real_3d,
                real_x_start,
                real_cont_motion,
                real_culture_labels,
                real_speaker_labels,
                real_onsets,
            ]
            if has_mm_fields:
                real_tensors.extend([
                    real_text_features,
                    real_audio_mels,
                    real_audio_onsets_full,
                    real_audio_wav2vec,
                ])
            real_dataset = GeneratedDataset(*real_tensors)

        # Create a DataLoader
        data_loader_real = DataLoader(
            real_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False
        )
        print("Validation Datasets were generated!!")

    return data_loader_generated, data_loader_real



def plot_generated_samples(mdm_model, vq_vae_config,save_dir,plot_data):
    """
    Generates and plots samples during training by comparing real and generated motions.
    """
    # Set the model to evaluation mode
    mdm_model.eval()

    # Ensure the directory for saving plots exists
    save_dir_plots = os.path.join(save_dir, "plots_during_test")
    os.makedirs(save_dir_plots, exist_ok=True)

    # Select the first batch from validation data
    print(len(plot_data),len(plot_data[0]),len(plot_data[0][0]))
    generated_motion = plot_data[0][:10]
    real_motion = plot_data[1][:10]

    # Define save paths
    #step = self.total_step()
    #save_path = os.path.join(save_dir_plots, f"step_{step}")
    save_path_real = os.path.join(save_dir_plots, "real_samples")
    save_path_generated = os.path.join(save_dir_plots, "generated_samples")

    # Create directories for real and generated samples
    os.makedirs(save_path_real, exist_ok=True)
    os.makedirs(save_path_generated, exist_ok=True)

    # Generate and save plots
    pose_plot_generator = partial(sample_generation_from_codebooks)
    pose_plot_generator(vq_vae_config, generated_motion, real_motion, save_path_real, save_path_generated,
                             step=0)



def _load_state_with_fn(model, state_dict, use_culture=True, load_model_fn=None):
    if load_model_fn is None:
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print("MISSING KEYS:", missing_keys)
        print("UNEXPECTED KEYS:", unexpected_keys)
        return
    try:
        load_model_fn(model, state_dict, use_culture)
    except TypeError:
        load_model_fn(model, state_dict)


def load_and_sync_parameters(
        checkpoint_path,
        device='cuda:0',
        mdm_model=None,
        use_ema=True,
        use_culture=True,
        load_model_fn=load_model,
):

    #resume_step = parse_resume_step_from_filename(checkpoint_path)
    print(f"loading model from checkpoint: {checkpoint_path}...")
    # it is just a torch load to lad the parameters
    state_dict = dist_util.load_state_dict(checkpoint_path, map_location=device)

    if 'model_avg' in state_dict:  # so if I used ema
        print('checkpoint contains both model and model_avg')
        state_dict, state_dict_avg = state_dict['model'], state_dict[
            'model_avg']
        if use_ema:
            print('loading EMA weights for evaluation')
            _load_state_with_fn(
                mdm_model,
                state_dict_avg,
                use_culture=use_culture,
                load_model_fn=load_model_fn,
            )
        else:
            print('loading non-EMA weights for evaluation')
            _load_state_with_fn(
                mdm_model,
                state_dict,
                use_culture=use_culture,
                load_model_fn=load_model_fn,
            )
    else:
        _load_state_with_fn(
            mdm_model,
            state_dict,
            use_culture=use_culture,
            load_model_fn=load_model_fn,
        )


def evaluate(val_data_loader_real, val_data_loader_generated, save_dir, plot_data, mdm_model, vq_vae_config):
    start_eval = time.time()
    print('Running evaluation loop...')
    log_file = os.path.join(save_dir, f'eval_model{0}.log')
    diversity_times = 300
    # This means that evaluate_multimodality is not evaluated during training
    mm_num_times = 0  # mm is super slow hence we won't run it during training
    eval_rep_times = 5
    if plot_data != None:
        plot_generated_samples(mdm_model, vq_vae_config,save_dir,plot_data)
    eval_dict = eval_ted4cl.evaluation(val_data_loader_real, val_data_loader_generated, log_file,
                                       replication_times=eval_rep_times ,
                                       diversity_times=diversity_times,
                                       mm_num_times=mm_num_times,
                                       run_mm=False, eval_platform=None)
    print(eval_dict)
    '''
    for k, v in eval_dict.items():
        if k.startswith('R_precision'):
            for i in range(len(v)):
                self.train_platform.report_scalar(name=f'top{i + 1}_' + k, value=v[i],
                                                  iteration=0,
                                                  group_name='Eval')
        else:
            self.train_platform.report_scalar(name=k, value=v, iteration=0,
                                              group_name='Eval')
    '''
    end_eval = time.time()
    print(f'Evaluation time: {round(end_eval-start_eval)/60}min')


def _safe_scalar(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        value = value.item()
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, (float, int)):
        if np.isfinite(value):
            return float(value)
    return None


def _to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _flatten_numeric_metrics(metrics: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    flat = {}
    for key, value in metrics.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_numeric_metrics(value, prefix=full_key))
            continue
        scalar = _safe_scalar(value)
        if scalar is not None:
            flat[full_key] = scalar
    return flat


def _build_subset_loader(source_loader: DataLoader, subset_size: int, run_seed: int) -> DataLoader:
    dataset_len = len(source_loader.dataset)
    target_size = min(int(subset_size), dataset_len)
    if target_size <= 0:
        raise ValueError("Source loader is empty; cannot build evaluation subset.")
    rng = np.random.default_rng(run_seed)
    subset_indices = rng.choice(dataset_len, size=target_size, replace=False).tolist()
    subset_dataset = Subset(source_loader.dataset, subset_indices)
    persistent_workers = bool(getattr(source_loader, "persistent_workers", False) and source_loader.num_workers > 0)
    loader_kwargs = dict(
        batch_size=source_loader.batch_size,
        shuffle=False,
        num_workers=source_loader.num_workers,
        pin_memory=source_loader.pin_memory,
        drop_last=False,
        persistent_workers=persistent_workers,
        collate_fn=source_loader.collate_fn,
        worker_init_fn=worker_init_fn if source_loader.num_workers > 0 else None,
    )
    if source_loader.num_workers > 0 and hasattr(source_loader, "prefetch_factor"):
        loader_kwargs["prefetch_factor"] = source_loader.prefetch_factor
    return DataLoader(
        subset_dataset,
        **loader_kwargs,
    )


def _paired_tests(values_a: List[float], values_b: List[float]) -> Dict[str, Any]:
    arr_a = np.asarray(values_a, dtype=np.float64)
    arr_b = np.asarray(values_b, dtype=np.float64)
    valid = np.isfinite(arr_a) & np.isfinite(arr_b)
    arr_a = arr_a[valid]
    arr_b = arr_b[valid]
    out = {
        "n": int(arr_a.size),
        "mean_a": float(arr_a.mean()) if arr_a.size else None,
        "var_a": float(arr_a.var(ddof=1)) if arr_a.size > 1 else 0.0 if arr_a.size == 1 else None,
        "mean_b": float(arr_b.mean()) if arr_b.size else None,
        "var_b": float(arr_b.var(ddof=1)) if arr_b.size > 1 else 0.0 if arr_b.size == 1 else None,
        "mean_diff_a_minus_b": float((arr_a - arr_b).mean()) if arr_a.size else None,
        "var_diff_a_minus_b": float((arr_a - arr_b).var(ddof=1)) if arr_a.size > 1 else 0.0 if arr_a.size == 1 else None,
        "ttest_rel": None,
        "wilcoxon": None,
    }
    if arr_a.size < 2:
        return out
    try:
        t_stat, t_p = stats.ttest_rel(arr_a, arr_b, nan_policy="omit")
        out["ttest_rel"] = {"statistic": float(t_stat), "pvalue": float(t_p)}
    except Exception:
        out["ttest_rel"] = None
    try:
        w_stat, w_p = stats.wilcoxon(arr_a, arr_b, zero_method="wilcox")
        out["wilcoxon"] = {"statistic": float(w_stat), "pvalue": float(w_p)}
    except Exception:
        out["wilcoxon"] = None
    return out


def _onesample_tests(values: List[float], popmean: float = 0.0) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    out = {
        "n": int(arr.size),
        "mean": float(arr.mean()) if arr.size else None,
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0 if arr.size == 1 else None,
        "var": float(arr.var(ddof=1)) if arr.size > 1 else 0.0 if arr.size == 1 else None,
        "ttest_1samp": None,
        "wilcoxon": None,
    }
    if arr.size < 2:
        return out
    try:
        t_stat, t_p = stats.ttest_1samp(arr, popmean=popmean, nan_policy="omit")
        out["ttest_1samp"] = {"statistic": float(t_stat), "pvalue": float(t_p)}
    except Exception:
        out["ttest_1samp"] = None
    try:
        w_stat, w_p = stats.wilcoxon(arr - popmean, zero_method="wilcox")
        out["wilcoxon"] = {"statistic": float(w_stat), "pvalue": float(w_p)}
    except Exception:
        out["wilcoxon"] = None
    return out


def _aggregate_run_metrics(run_metric_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    flat_runs = [_flatten_numeric_metrics(metrics) for metrics in run_metric_dicts]
    all_keys = sorted({k for run in flat_runs for k in run.keys()})

    metric_summary = {}
    for key in all_keys:
        values = [run[key] for run in flat_runs if key in run and np.isfinite(run[key])]
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            continue
        metric_summary[key] = {
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "var": float(arr.var(ddof=1)) if arr.size > 1 else 0.0,
            "median": float(np.median(arr)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    pair_candidates = set()
    for key in all_keys:
        if key.startswith("real_"):
            candidate = "gen_" + key[len("real_"):]
            if candidate in all_keys:
                pair_candidates.add((key, candidate))
        if "_real_" in key:
            candidate = key.replace("_real_", "_gen_", 1)
            if candidate in all_keys:
                pair_candidates.add((key, candidate))

    paired_tests = {}
    for key_a, key_b in sorted(pair_candidates):
        vals_a, vals_b = [], []
        for run in flat_runs:
            if key_a in run and key_b in run:
                vals_a.append(run[key_a])
                vals_b.append(run[key_b])
        if vals_a and vals_b:
            paired_tests[f"{key_a}__vs__{key_b}"] = _paired_tests(vals_a, vals_b)

    one_sample_targets = [
        "overall_motion_low_alignment_gap_mean",
        "overall_motion_high_alignment_gap_mean",
    ]
    one_sample_tests = {}
    for key in one_sample_targets:
        values = [run[key] for run in flat_runs if key in run]
        if values:
            one_sample_tests[key] = _onesample_tests(values, popmean=0.0)

    return {
        "metric_summary": metric_summary,
        "paired_tests": paired_tests,
        "one_sample_tests": one_sample_tests,
    }


def run_repeated_subset_evaluation(
        source_loader: DataLoader,
        diffusion,
        mdm_model: nn.Module,
        save_dir: str,
        vqvae_model: nn.Module,
        vqvae_config,
        device,
        use_alignment: bool = True,
        num_runs: int = 10,
        samples_per_run: int = 3000,
        num_cultures: int = 4,
        compute_srgr_beat: bool = False,
        run_culture_classification: bool = False,
        culture_classifier_checkpoint_path: Optional[str] = None,
        culture_classifier_cl_type: str = "culclA",
        culture_classifier_mode: str = "adversarial_backbone",
        culture_classifier_d_model: int = 512,
        srgr_delta: float = 0.05,
        beat_align_sigma: float = 3.0,
        decode_3d: bool = False,
        base_seed: int = 10,
) -> Dict[str, Any]:
    os.makedirs(save_dir, exist_ok=True)
    run_outputs = []
    if compute_srgr_beat and not decode_3d:
        warnings.warn(
            "[Eval] compute_srgr_beat=True requires decoded 3D poses. "
            "Enabling decode_3d automatically."
        )
        decode_3d = True

    beat_align_joint_indices = None
    try:
        if global_keypoint_indices is not None and len(global_keypoint_indices) > 0:
            beat_align_joint_indices = list(range(len(global_keypoint_indices)))
            print(
                f"[Eval] BeatAlign joints set from keypoint layout "
                f"({len(global_keypoint_indices)} joints): {beat_align_joint_indices}"
            )
    except Exception:
        beat_align_joint_indices = None

    for run_idx in range(num_runs):
        run_id = run_idx + 1
        run_seed = base_seed + run_idx
        fixseed(run_seed)
        np.random.seed(run_seed)
        random.seed(run_seed)
        torch.manual_seed(run_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(run_seed)

        run_loader = _build_subset_loader(source_loader, samples_per_run, run_seed)
        data_loader_generated, data_loader_real, plot_data, decoded_3d_available, decode_3d_mode = generate_during_training(
            val_data=None,
            test_data=run_loader,
            device=device,
            diffusion=diffusion,
            mdm_model=mdm_model,
            save_path=save_dir,
            vqvae_model=vqvae_model,
            vqvae_config=vqvae_config,
            use_alignment=use_alignment,
            max_samples=samples_per_run,
            decode_3d=decode_3d,
        )
        if data_loader_generated is None or data_loader_real is None:
            raise RuntimeError("Failed to build generated/real evaluation loaders for the current run.")

        run_plot_dir = os.path.join(save_dir, f"run_{run_id:02d}_samples")
        if plot_data is not None:
            try:
                plot_generated_samples(mdm_model, vqvae_config, run_plot_dir, plot_data)
            except Exception as exc:
                warnings.warn(f"Could not plot sample poses for run {run_id}: {exc}")

        run_compute_srgr_beat = bool(compute_srgr_beat and decoded_3d_available)
        if compute_srgr_beat and not run_compute_srgr_beat:
            raise RuntimeError(
                f"Run {run_id}: 3D decoding unavailable. "
                f"SRGR/Beat are enabled and cannot be skipped. "
                f"decode_mode={decode_3d_mode}"
            )
        if run_compute_srgr_beat and str(decode_3d_mode) != "exact":
            raise RuntimeError(
                f"Run {run_id}: SRGR/BAS require exact 3D decode metadata, "
                f"but decode_mode={decode_3d_mode}. "
                "Provide valid --speaker-link-len-path and --skeleton-info-path."
            )

        log_file = os.path.join(save_dir, f'final_evaluation_run_{run_id:02d}.log')
        run_metrics = evaluate_all_metrics_bulk(
            data_loader_real,
            data_loader_generated,
            log_file,
            save_path=save_dir,
            num_cultures=num_cultures,
            compute_alignment=use_alignment,
            compute_srgr_beat=run_compute_srgr_beat,
            beat_align_kinematic_joints=beat_align_joint_indices,
            run_culture_classification=run_culture_classification,
            culture_classifier_checkpoint_path=culture_classifier_checkpoint_path,
            culture_classifier_cl_type=culture_classifier_cl_type,
            culture_classifier_mode=culture_classifier_mode,
            culture_classifier_d_model=culture_classifier_d_model,
            srgr_delta=srgr_delta,
            beat_align_sigma=beat_align_sigma,
            save_raw_data=False,
        )
        run_payload = {
            "run_id": run_id,
            "seed": run_seed,
            "num_subset_samples": min(samples_per_run, len(source_loader.dataset)),
            "decoded_3d_for_metrics": bool(decoded_3d_available),
            "decoded_3d_mode": decode_3d_mode,
            "metrics": run_metrics,
        }
        run_outputs.append(run_payload)
        run_path = os.path.join(save_dir, f'evaluation_run_{run_id:02d}.json')
        with open(run_path, "w") as f:
            json.dump(_to_jsonable(run_payload), f, indent=2)

    run_metric_dicts = [run["metrics"]["averaged_metrics"] for run in run_outputs]
    summary = _aggregate_run_metrics(run_metric_dicts)
    summary_payload = {
        "num_runs": num_runs,
        "samples_per_run": samples_per_run,
        "run_files": [f"evaluation_run_{i + 1:02d}.json" for i in range(num_runs)],
        "summary": summary,
    }
    summary_path = os.path.join(save_dir, "evaluation_multi_run_summary.json")
    with open(summary_path, "w") as f:
        json.dump(_to_jsonable(summary_payload), f, indent=2)

    return {
        "runs": run_outputs,
        "summary": summary_payload,
        "summary_path": summary_path,
    }


def validate_model_fgd(
        args: object, # Model/diffusion args
        model_dir: str, # Directory containing checkpoints
        step_interval: int, # e.g., 50000
        num_samples_to_generate: int, # e.g., 3000
        data_loader: torch.utils.data.DataLoader, # Validation or Test loader
        mdm_model: torch.nn.Module, # Base MDM model instance
        diffusion_model: object, # Diffusion process instance
        vqvae_model: torch.nn.Module, # Loaded VQ-VAE model
        device: str,
        batch_size: int = 64, # Batch size for generation during validation
        use_ema: bool = True,
        use_culture: bool = True,
        load_model_fn=load_model,
) -> Tuple[Optional[str], Optional[float], Dict[str, Optional[float]]]:
    """
    Iteratively loads model checkpoints, generates samples, calculates overall FGD
    on continuous VQ-VAE encodings, and returns the best checkpoint based on FGD.

    Args:
        (See above)

    Returns:
        Tuple containing:
        - Path to the best checkpoint (lowest FGD).
        - The lowest FGD score.
        - Dictionary mapping checkpoint paths to their FGD scores.
    """
    print(f"\n--- Starting FGD Validation Loop ---")
    print(f"Model Directory: {model_dir}")
    print(f"Step Interval: {step_interval}")
    print(f"Samples per Checkpoint: {num_samples_to_generate}")

    checkpoints = []
    try:
        # Find model checkpoints matching the pattern modelXXXXXX.pt
        for f in os.listdir(model_dir):
            if re.match(r"model\d+\.pt", f):
                # Extract step number
                match = re.search(r"model(\d+)\.pt", f)
                if match:
                    step = int(match.group(1))
                    # Check if step is a multiple of the interval and > 0
                    if step > 0 and step % step_interval == 0:
                        checkpoints.append((step, os.path.join(model_dir, f)))
    except FileNotFoundError:
        print(f"Error: Model directory not found: {model_dir}")
        return None, None, {}

    if not checkpoints:
        print(f"No checkpoints found matching interval {step_interval} in {model_dir}")
        # Optional: Check for final checkpoint anyway?
        # final_ckpt_path = os.path.join(model_dir, 'latest.pt') # Or a specific name convention
        # if os.path.exists(final_ckpt_path):
        #    print("Attempting to validate final checkpoint.")
        #    match = re.search(r"model(\d+)\.pt", final_ckpt_path) # Adapt if naming differs
        #    step = int(match.group(1)) if match else 0
        #    checkpoints.append((step, final_ckpt_path))
        # else:
        return None, None, {}

    # Sort checkpoints by step number
    checkpoints.sort(key=lambda x: x[0])

    best_fgd = float('inf')
    best_checkpoint_path = None
    all_fgd_scores = OrderedDict()

    num_batches_needed = math.ceil(num_samples_to_generate / batch_size)

    # Store original state dict to restore later if needed (optional)
    # original_state_dict = copy.deepcopy(mdm_model.state_dict())

    for step, ckpt_path in checkpoints:
        print(f"\nValidating Checkpoint: {os.path.basename(ckpt_path)} (Step: {step})")

        try:
            # Load model weights for this checkpoint
            print("Device",device,type(device))
            load_and_sync_parameters(
                ckpt_path,
                device=device,
                mdm_model=mdm_model,
                use_ema=use_ema,
                use_culture=use_culture,
                load_model_fn=load_model_fn,
            )
            mdm_model.eval()
            vqvae_model.eval() # Ensure VQVAE is also in eval mode
        except Exception as e:
            print(f"  Error loading checkpoint {ckpt_path}: {e}. Skipping.")
            all_fgd_scores[ckpt_path] = None
            continue

        all_gen_cont_encodings = []
        all_real_cont_encodings = []
        samples_collected = 0
        vqvae_core = _get_vqvae_core(vqvae_model)
        include_culture_labels = _diffusion_accepts_culture_labels(diffusion_model)

        # Generate samples
        with torch.no_grad():
            pbar = tqdm(total=num_samples_to_generate, desc=f"  Generating Val Samples (Ckpt {step})")
            batch_count = 0
            for batch in data_loader:
                if samples_collected >= num_samples_to_generate:
                    break

                # Move batch data needed for generation to device
                motion_real, text_features, audio_mels, audio_onsets, audio_wav2vec, labels = batch
                culture_real_labels = labels['culture_enc']
                speaker_real_labels = labels['speaker_enc']

                # Prepare input data for diffusion model
                # Ensure correct data is passed based on how diffusion.p_sample_loop expects it
                # Assuming it needs [motion_prefix, text_features, audio_mels, audio_onsets, audio_wav2vec]
                # Adjust the prefix slicing as needed by your model/diffusion
                motion_prefix = motion_real[:, :5, :].to(device) # Example prefix
                real_motion_target = motion_real[:, 5:, :].to(device) # The part corresponding to generation

                in_data_diffusion = _build_diffusion_inputs(batch, device, include_culture_labels)
                mdm_model = mdm_model.to(device)

                # Generate motion codebooks (output shape may vary, adjust accordingly)
                # Assuming new_motion has shape [B, T', D'] where T' matches real_motion_target's time dim
                new_motion, _ = diffusion_model.p_sample_loop(mdm_model, in_data_diffusion, clip_denoised=False)

                # --- Extract Continuous Encodings ---
                # Generated
                gen_codebooks = new_motion.permute(0, 2, 1).to(device) # B, D', T' -> B, T', D' for VQVAE
                try:
                    zs_gen = vqvae_core.bottleneck.encode([gen_codebooks])
                    gen_cont_encoding = vqvae_core.bottleneck.decode(zs_gen)[0] # B, D', T'
                    gen_cont_encoding = gen_cont_encoding.permute(0, 2, 1) # B, T', D'
                    # Apply slicing as requested: [:, 5:, :] - CHECK if this slicing index '5' makes sense for the generated T' dim
                    # If generation starts *after* the prefix, the time dim T' might already correspond to index 5 onwards.
                    # Let's assume T' corresponds to time 5...end, so we take the whole generated encoding.
                    # If T' still includes the prefix time, then slicing is needed. Revisit this if needed.
                    # Assuming T' corresponds to the target part, so we take all:
                    # gen_cont_encoding_sliced = gen_cont_encoding
                    # If T' has same length as original motion and needs slicing:
                    gen_cont_encoding_sliced = gen_cont_encoding[:, 5:, :] # Re-enable slicing if VQVAE input/output time dim requires it

                except Exception as e:
                     print(f"  Error processing generated motion with VQVAE: {e}")
                     gen_cont_encoding_sliced = None


                # Real
                real_codebooks = motion_real.permute(0, 2, 1).to(device) # B, D, T_full -> B, T_full, D for VQVAE
                try:
                    zs_real = vqvae_core.bottleneck.encode([real_codebooks])
                    real_cont_encoding = vqvae_core.bottleneck.decode(zs_real)[0] # B, D, T_full
                    real_cont_encoding = real_cont_encoding.permute(0, 2, 1) # B, T_full, D
                    # Apply slicing as requested
                    real_cont_encoding_sliced = real_cont_encoding[:, 5:, :]
                except Exception as e:
                    print(f"  Error processing real motion with VQVAE: {e}")
                    real_cont_encoding_sliced = None


                # Store results if valid
                if gen_cont_encoding_sliced is not None and real_cont_encoding_sliced is not None:
                    # Ensure time dimensions match for fair comparison if needed (though FGD handles feature dim)
                    # min_t = min(gen_cont_encoding_sliced.shape[1], real_cont_encoding_sliced.shape[1])
                    # all_gen_cont_encodings.append(gen_cont_encoding_sliced[:, :min_t, :].detach().cpu())
                    # all_real_cont_encodings.append(real_cont_encoding_sliced[:, :min_t, :].detach().cpu())
                    all_gen_cont_encodings.append(gen_cont_encoding_sliced.detach().cpu())
                    all_real_cont_encodings.append(real_cont_encoding_sliced.detach().cpu())

                    samples_in_batch = gen_cont_encoding_sliced.shape[0]
                    samples_collected += samples_in_batch
                    pbar.update(samples_in_batch)

                batch_count += 1
                if batch_count >= num_batches_needed:
                    break
            pbar.close()


        # Check if enough samples were collected
        if samples_collected < num_samples_to_generate * 0.8: # Allow some margin
             warnings.warn(f"  Collected only {samples_collected}/{num_samples_to_generate} samples for Ckpt {step}. FGD might be unreliable.")
        if not all_gen_cont_encodings or not all_real_cont_encodings:
             print("  Failed to collect valid encodings. Skipping FGD calculation.")
             all_fgd_scores[ckpt_path] = None
             continue


        # Concatenate all collected encodings
        try:
            gen_encodings_tensor = torch.cat(all_gen_cont_encodings, dim=0)[:num_samples_to_generate]
            real_encodings_tensor = torch.cat(all_real_cont_encodings, dim=0)[:num_samples_to_generate]
            print(f"  Final Encoding Shapes for FGD: Gen={gen_encodings_tensor.shape}, Real={real_encodings_tensor.shape}")
        except Exception as e:
            print(f"  Error concatenating encodings: {e}. Skipping FGD calculation.")
            all_fgd_scores[ckpt_path] = None
            continue


        # Calculate Overall FGD
        current_fgd = calculate_overall_fgd(real_encodings_tensor, gen_encodings_tensor)
        all_fgd_scores[ckpt_path] = current_fgd
        print(f"  FGD Score: {current_fgd}")

        # Update best score
        if current_fgd is not None and current_fgd < best_fgd:
            best_fgd = current_fgd
            best_checkpoint_path = ckpt_path
            print(f"  *** New Best FGD Score Found ***")

    # Restore original model state if needed (optional)
    # mdm_model.load_state_dict(original_state_dict)

    print("\n--- FGD Validation Complete ---")
    if best_checkpoint_path:
        print(f"Best Checkpoint: {os.path.basename(best_checkpoint_path)}")
        print(f"Lowest FGD Score: {best_fgd:.4f}")
    else:
        print("No valid FGD scores were computed.")

    return best_checkpoint_path, best_fgd, all_fgd_scores



def test_real_data_with_classification_layer(val_data,mdm_model,device, use_alignment = False):

    eval_dict = OrderedDict()
    metrics = OrderedDict({
        'F1_Score': None,
        'Balanced_Accuracy': None,
        'Accuracy': None,
        'ROC_AUC': None
    })

    all_labels = []
    all_preds = []
    all_probs = []
    num_classes = 4 #num_cultures
    mdm_model.eval()

    with torch.no_grad():
        for batch in tqdm(val_data, desc="evaluating data...", position=1,
                          leave=True):  # , desc="Sample generation from validation data"

            # Move batch data to device
            if use_alignment:
                (poses, final_motion, culture_output, motion_pooled, high_level_context,
                low_level_context, cont_motion, culture_labels, speaker_labels, onsets) = batch
            else:
                (poses, final_motion, cont_motion, culture_labels, speaker_labels, onsets) = batch
            culture_real = culture_labels.cpu().numpy()
            all_labels.extend(culture_real)
            motion = final_motion
            culture_logits = mdm_model.culture_classification_layer(motion)
            culture_probs = torch.softmax(culture_logits, dim=-1).cpu().numpy()
            preds = np.argmax(culture_probs, axis=1)
            all_preds.extend(preds)
            all_probs.extend(culture_probs)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    f1 = f1_score(all_labels, all_preds, average='macro')
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    all_labels_binarized = label_binarize(all_labels, classes=range(num_classes))
    roc_auc = roc_auc_score(all_labels_binarized, all_probs, average='macro', multi_class='ovr')
    metrics['F1_Score'] = f1
    metrics['Balanced_Accuracy'] = balanced_acc
    metrics['Accuracy'] = acc
    metrics['ROC_AUC'] = roc_auc

    print(f"===== F1 Score: {f1} ===== \n ===== Balanced Accuracy: {balanced_acc} \n"
          f" ===== Accuracy: {acc} ===== \n ===== ROC AUC: {roc_auc} =====")




def test(
        args_path,
        checkpoint_path=None,
        save_dir='',
        model_family: Optional[str] = None,
        validate=False,
        num_eval_runs: int = 10,
        samples_per_run: int = 3000,
        compute_srgr_beat: bool = False,
        run_culture_classification: bool = False,
        decode_3d_for_eval: bool = False,
        run_validation_sweep: bool = False,
        step_interval: int = 50000,
        validation_samples: int = 10000,
        validation_split: str = "val",
        test_split: str = "test",
        model_dir: Optional[str] = None,
        fishr_model_path: Optional[str] = None,
        adversarial_checkpoint_path: Optional[str] = None,
        adversarial_model_save_path: Optional[str] = None,
        culture_classifier_checkpoint_path: Optional[str] = None,
        culture_classifier_cl_type: str = "culclA",
        culture_classifier_mode: str = "adversarial_backbone",
        culture_classifier_d_model: int = 512,
        srgr_delta: float = 0.05,
        beat_align_sigma: float = 3.0,
        speaker_link_len_path: Optional[str] = None,
        skeleton_info_path: Optional[str] = None,
):
    if os.path.exists(args_path):
        with open(args_path, 'r') as fr:
            args_dict = json.load(fr)
        args = argparse.Namespace(**args_dict)
    else:
        raise FileNotFoundError('args.json was not found in the specified path.')

    culture_config_path = getattr(args, "culture_config_path", "culture_encoder/config.yml")
    with open(culture_config_path) as f:
        culture_config = yaml.safe_load(f)
    culture_config = EasyDict(culture_config)
    if fishr_model_path:
        culture_config.fishr_model_path = fishr_model_path
    if adversarial_checkpoint_path:
        culture_config.adversarial_checkpoint_path = adversarial_checkpoint_path
    if adversarial_model_save_path:
        culture_config.adversarial_model_save_path = adversarial_model_save_path

    dataset_path = getattr(args, "lmdb_path", None) or getattr(args, "dataset_path", None)
    if dataset_path is None:
        raise ValueError("args.json must include either `lmdb_path` or `dataset_path`.")
    metadata_path = args.metadata_path
    dataset_info_path = getattr(args, "info_path", None) or getattr(args, "dataset_info_path", None)
    if dataset_info_path is None:
        raise ValueError("args.json must include either `info_path` or `dataset_info_path`.")
    encodings_path = normalization_path = dataset_info_path
    batch_size = args.batch_size
    fixseed(getattr(args, "seed", 10))
    sep_people = getattr(args, "sep_people", "_sep_people")
    motion_only = False

    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
        sample_keys = metadata['sample_keys']
        culture_speakers = metadata['culture_speakers']

    splits_data_path = getattr(
        args,
        "splits_data_path",
        os.path.join(dataset_info_path, "whole_dataset_splits_subject_independent.pkl"),
    )
    if "_sep_people" not in str(sep_people):
        warnings.warn(
            "[Eval] sep_people does not contain '_sep_people'. "
            "This implies subject-dependent splits, not subject-independent."
        )
    if "subject_independent" not in os.path.basename(str(splits_data_path)):
        warnings.warn(
            f"[Eval] splits_data_path does not look subject-independent: {splits_data_path}"
        )
    train_loader, val_loader, test_loader, speaker_enc, culture_enc, n_train_speakers = prepare_data(
        sample_keys,
        culture_speakers,
        motion_only,
        batch_size,
        splits_data_path,
        dataset_path,
        encodings_path,
        normalization_path,
        sep_people,
        use_translated_text=getattr(args, "use_translated_text", False),
        use_language_features=getattr(args, "use_language_features", True),
        use_translated_text_eval=getattr(args, "use_translated_text_eval", None),
        use_language_features_eval=getattr(args, "use_language_features_eval", None),
    )

    requested_device = getattr(args, "device", 0)
    if isinstance(requested_device, str):
        if requested_device.startswith("cuda:"):
            requested_device = int(requested_device.split(":")[1])
        else:
            try:
                requested_device = int(requested_device)
            except Exception:
                requested_device = 0
    dist_util.setup_dist(requested_device)
    cuda_device = dist_util.dev()

    vqvae_config_path = getattr(args, "vqvae_config_path", "vq_vae/configs/codebook.yml")
    with open(vqvae_config_path) as f:
        vq_vae_config = yaml.safe_load(f)
        vq_vae_config = EasyDict(vq_vae_config)
    if speaker_link_len_path:
        vq_vae_config.speaker_link_len = speaker_link_len_path
    if skeleton_info_path:
        vq_vae_config.skeleton_info_path = skeleton_info_path

    global_vqvae_model, global_info_data, global_links_len, global_keypoint_indices = load_vqvae_model(
        vq_vae_config,
        vq_vae_config.checkpoint_path,
    )
    # Keep decode/plot utilities on the same loaded VQ-VAE + metadata instance.
    vq_vis.global_vqvae_model = global_vqvae_model
    vq_vis.global_info_data = global_info_data
    vq_vis.global_links_len = global_links_len
    vq_vis.global_keypoint_indices = global_keypoint_indices
    print(f"[Eval] 3D decode metadata mode: {global_decode_metadata_mode}")

    use_culture = bool(getattr(args, "use_culture", True))
    use_adversarial = bool(getattr(args, "use_adversarial", False))
    use_one_hot_culture = bool(getattr(args, "use_one_hot_culture", False))
    culture_encoder_type = getattr(args, "culture_encoder_type", "")
    if use_culture and (culture_encoder_type in {"one_hot", "onehot"} or use_one_hot_culture):
        print("Using trainable one-hot culture encoder")
    elif use_culture and use_adversarial:
        print("Using adversarial culture encoder")
    elif use_culture and not use_adversarial:
        print("Using Fishr culture encoder")
    else:
        print("Using no culture encoding")

    effective_model_family = (model_family or getattr(args, "model_family", "hierarchical")).strip().lower()
    if effective_model_family in {"hierarchical", "hierachical"}:
        eval_model = Hierarchical_MDM(
            vqvae_dim=(25, 512),
            audio_onset_dim=156,
            audio_mel_dim=(156, 64),
            audio_wav2vec_dim=(50, 1024),
            text_embedding_dim=768,
            culture_embedding_dim=512,
            latent_dim=args.latent_dim,
            n_train_speakers=n_train_speakers,
            n_cultures=len(culture_enc),
            num_heads=args.heads,
            num_layers=args.layers,
            ff_size=args.ffn_size,
            dropout=0.1,
            activation='gelu',
            culture_embedder_config=culture_config,
            device=str(cuda_device),
            dataset_type="_sep_people",
            motion_prefix_len=args.motion_prefix_len,
            audio_prefix_len=args.audio_prefix_len,
            motion_mask_prob=getattr(args, "motion_mask_prob", 0.0),
            audio_mask_prob=getattr(args, "audio_mask_prob", 0.0),
            use_culture=use_culture,
            use_adversarial=use_adversarial,
            use_one_hot_culture=use_one_hot_culture,
            culture_encoder_type=culture_encoder_type,
            use_adain=getattr(args, "use_adain", True),
            use_alignment=getattr(args, "use_alignment", True),
            use_culture_classification_layer=getattr(args, "use_culture_classification_layer", True),
            use_native_attention=getattr(args, "use_native_attention", False),
        )
        diffusion_model = create_hierarchical_gaussian_diffusion(args)
        use_alignment = True
        selected_load_model_fn = load_model
    elif effective_model_family in {
        "baseline_mdm",
        "mdm",
        "baseline_diffustylegesture_plus",
        "diffustylegesture_plus",
        "diffustylegesture+",
    }:
        if not _load_baseline_modules():
            raise ImportError(
                "Could not import baseline modules from diffustylegesture_and_mdm.\n"
                "Tried package and local-path imports.\n"
                f"Details:\n{_baseline_import_error}"
            )
        cond_mode = "mdm" if effective_model_family in {"baseline_mdm", "mdm"} else "cross_local_attention4_style1"
        style_dim = 512 if use_culture else int(getattr(args, "style_dim", 512))
        eval_model = BaselineMDM(
            modeltype="",
            njoints=getattr(args, "njoints", 2052),
            nfeats=1,
            cond_mode=cond_mode,
            audio_feat=getattr(args, "audio_feat", "wavlm"),
            arch="trans_enc",
            latent_dim=getattr(args, "latent_dim", 512),
            n_seed=getattr(args, "n_seed", 5),
            cond_mask_prob=getattr(args, "cond_mask_prob", 0.1),
            device=str(cuda_device),
            style_dim=style_dim,
            source_audio_dim=getattr(args, "audio_feature_dim", 1857),
            audio_feat_dim_latent=getattr(args, "audio_feat_dim_latent", 96),
            batch_size=getattr(args, "batch_size", 64),
            use_culture=use_culture,
            use_adversarial=use_adversarial,
            n_train_speakers=n_train_speakers,
            num_layers=getattr(args, "layers", 10),
            num_heads=getattr(args, "heads", 8),
            ff_size=getattr(args, "ffn_size", 2048),
            culture_embedder_config=culture_config,
            culture_config_path=culture_config_path,
            fishr_model_path=fishr_model_path or getattr(args, "fishr_model_path", None),
            adversarial_checkpoint_path=adversarial_checkpoint_path or getattr(
                args, "adversarial_checkpoint_path", None
            ),
            adversarial_model_save_path=adversarial_model_save_path or getattr(
                args, "adversarial_model_save_path", None
            ),
            use_alignment_module=getattr(args, "use_alignment_module", False),
            use_culture_guidance_loss=getattr(args, "use_culture_guidance_loss", False),
            n_cultures=len(culture_enc),
        )
        diffusion_model = create_baseline_gaussian_diffusion(args)
        use_alignment = False
        selected_load_model_fn = load_baseline_model
    else:
        raise ValueError(
            f"Unsupported model family '{effective_model_family}'. "
            "Use hierarchical, baseline_mdm, or baseline_diffustylegesture_plus."
        )

    eval_model.to(dist_util.dev())
    os.makedirs(save_dir, exist_ok=True)

    model_dir = model_dir or (os.path.split(checkpoint_path)[0] if checkpoint_path else args.save_dir)
    if validate and test_split == "test":
        test_split = "val"

    selected_checkpoint = checkpoint_path
    best_fgd = None
    all_fgd_scores = None
    if run_validation_sweep:
        val_source_loader = val_loader if validation_split == "val" else test_loader
        selected_checkpoint, best_fgd, all_fgd_scores = validate_model_fgd(
            args=args,
            model_dir=model_dir,
            step_interval=step_interval,
            num_samples_to_generate=validation_samples,
            data_loader=val_source_loader,
            mdm_model=eval_model,
            diffusion_model=diffusion_model,
            vqvae_model=global_vqvae_model,
            device=cuda_device,
            batch_size=args.batch_size,
            use_ema=getattr(args, "use_ema", True),
            use_culture=use_culture,
            load_model_fn=selected_load_model_fn,
        )
        fgd_path = os.path.join(save_dir, "validation_fgd_scores.json")
        with open(fgd_path, "w") as f:
            json.dump(
                _to_jsonable(
                    {
                        "best_checkpoint_path": selected_checkpoint,
                        "best_fgd": best_fgd,
                        "scores": all_fgd_scores,
                    }
                ),
                f,
                indent=2,
            )
        print(f"Saved validation sweep to {fgd_path}")

    if not selected_checkpoint:
        raise ValueError("No checkpoint selected. Provide --checkpoint-path or enable --run-validation-sweep.")

    load_and_sync_parameters(
        selected_checkpoint,
        device=cuda_device,
        mdm_model=eval_model,
        use_ema=getattr(args, "use_ema", True),
        use_culture=use_culture,
        load_model_fn=selected_load_model_fn,
    )
    eval_source_loader = val_loader if test_split == "val" else test_loader
    evaluation_results = run_repeated_subset_evaluation(
        source_loader=eval_source_loader,
        diffusion=diffusion_model,
        mdm_model=eval_model,
        save_dir=save_dir,
        vqvae_model=global_vqvae_model,
        vqvae_config=vq_vae_config,
        device=cuda_device,
        use_alignment=use_alignment,
        num_runs=num_eval_runs,
        samples_per_run=samples_per_run,
        num_cultures=4,
        compute_srgr_beat=compute_srgr_beat,
        run_culture_classification=run_culture_classification,
        culture_classifier_checkpoint_path=culture_classifier_checkpoint_path,
        culture_classifier_cl_type=culture_classifier_cl_type,
        culture_classifier_mode=culture_classifier_mode,
        culture_classifier_d_model=culture_classifier_d_model,
        srgr_delta=srgr_delta,
        beat_align_sigma=beat_align_sigma,
        decode_3d=decode_3d_for_eval,
        base_seed=args.seed,
    )
    evaluation_results["selected_checkpoint_path"] = selected_checkpoint
    evaluation_results["best_validation_fgd"] = best_fgd
    metadata_path = os.path.join(save_dir, "evaluation_run_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(
            _to_jsonable(
                {
                    "args_path": args_path,
                    "model_family": effective_model_family,
                    "use_alignment": use_alignment,
                    "selected_checkpoint_path": selected_checkpoint,
                    "run_validation_sweep": run_validation_sweep,
                    "step_interval": step_interval,
                    "validation_samples": validation_samples,
                    "validation_split": validation_split,
                    "test_split": test_split,
                    "sep_people": sep_people,
                    "splits_data_path": splits_data_path,
                    "num_eval_runs": num_eval_runs,
                    "samples_per_run": samples_per_run,
                    "compute_srgr_beat": compute_srgr_beat,
                    "run_culture_classification": run_culture_classification,
                    "culture_classifier_checkpoint_path": culture_classifier_checkpoint_path,
                    "culture_classifier_cl_type": culture_classifier_cl_type,
                    "culture_classifier_mode": culture_classifier_mode,
                    "culture_classifier_d_model": culture_classifier_d_model,
                    "srgr_delta": srgr_delta,
                    "beat_align_sigma": beat_align_sigma,
                    "speaker_link_len_path": speaker_link_len_path,
                    "skeleton_info_path": skeleton_info_path,
                }
            ),
            f,
            indent=2,
        )
    print(f"Saved evaluation metadata to {metadata_path}")
    print(f"Saved multi-run summary to {evaluation_results['summary_path']}")
    return evaluation_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate and evaluate diffusion checkpoints.")
    parser.add_argument("--args-path", type=str, required=True, help="Path to training args.json.")
    parser.add_argument(
        "--model-family",
        type=str,
        default=None,
        choices=[
            "hierarchical",
            "baseline_mdm",
            "mdm",
            "baseline_diffustylegesture_plus",
            "diffustylegesture_plus",
        ],
        help="Model family used to instantiate the checkpoint architecture.",
    )
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Checkpoint to evaluate directly.")
    parser.add_argument("--model-dir", type=str, default=None, help="Directory containing model*.pt checkpoints.")
    parser.add_argument("--save-dir", type=str, required=True, help="Directory to save evaluation outputs.")
    parser.add_argument("--run-validation-sweep", action="store_true", help="Sweep checkpoints and pick best by FGD.")
    parser.add_argument("--step-interval", type=int, default=50000, help="Checkpoint interval for validation sweep.")
    parser.add_argument("--validation-samples", type=int, default=10000, help="Samples used per checkpoint for FGD.")
    parser.add_argument("--validation-split", type=str, choices=["val", "test"], default="val")
    parser.add_argument("--test-split", type=str, choices=["val", "test"], default="test")
    parser.add_argument("--num-eval-runs", type=int, default=10)
    parser.add_argument("--samples-per-run", type=int, default=3000)
    parser.add_argument("--compute-srgr-beat", action="store_true")
    parser.add_argument("--run-culture-classification", action="store_true")
    parser.add_argument(
        "--culture-classifier-checkpoint-path",
        type=str,
        default=None,
        help="Optional checkpoint for the motion culture classifier used in evaluation.",
    )
    parser.add_argument("--culture-classifier-cl-type", type=str, default="culclA")
    parser.add_argument(
        "--culture-classifier-mode",
        type=str,
        choices=["adversarial_backbone", "fishr_backbone", "motion"],
        default="adversarial_backbone",
    )
    parser.add_argument("--culture-classifier-d-model", type=int, default=512)
    parser.add_argument("--srgr-delta", type=float, default=0.05)
    parser.add_argument("--beat-align-sigma", type=float, default=3.0)
    parser.add_argument("--decode-3d-for-eval", action="store_true")
    parser.add_argument("--speaker-link-len-path", type=str, default=None)
    parser.add_argument("--skeleton-info-path", type=str, default=None)
    parser.add_argument("--fishr-model-path", type=str, default=None)
    parser.add_argument("--adversarial-checkpoint-path", type=str, default=None)
    parser.add_argument("--adversarial-model-save-path", type=str, default=None)
    cli_args = parser.parse_args()

    test(
        args_path=cli_args.args_path,
        checkpoint_path=cli_args.checkpoint_path,
        save_dir=cli_args.save_dir,
        model_family=cli_args.model_family,
        run_validation_sweep=cli_args.run_validation_sweep,
        step_interval=cli_args.step_interval,
        validation_samples=cli_args.validation_samples,
        validation_split=cli_args.validation_split,
        test_split=cli_args.test_split,
        model_dir=cli_args.model_dir,
        num_eval_runs=cli_args.num_eval_runs,
        samples_per_run=cli_args.samples_per_run,
        compute_srgr_beat=cli_args.compute_srgr_beat,
        run_culture_classification=cli_args.run_culture_classification,
        culture_classifier_checkpoint_path=cli_args.culture_classifier_checkpoint_path,
        culture_classifier_cl_type=cli_args.culture_classifier_cl_type,
        culture_classifier_mode=cli_args.culture_classifier_mode,
        culture_classifier_d_model=cli_args.culture_classifier_d_model,
        srgr_delta=cli_args.srgr_delta,
        beat_align_sigma=cli_args.beat_align_sigma,
        decode_3d_for_eval=cli_args.decode_3d_for_eval,
        speaker_link_len_path=cli_args.speaker_link_len_path,
        skeleton_info_path=cli_args.skeleton_info_path,
        fishr_model_path=cli_args.fishr_model_path,
        adversarial_checkpoint_path=cli_args.adversarial_checkpoint_path,
        adversarial_model_save_path=cli_args.adversarial_model_save_path,
    )
