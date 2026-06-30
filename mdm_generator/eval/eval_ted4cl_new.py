import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
from itertools import combinations
from collections import defaultdict,OrderedDict
import numpy as np
import pickle
from tqdm.auto import tqdm
import os
try:
    import librosa
except Exception as _librosa_exc:
    librosa = None
    print(f"[eval_ted4cl_new] Warning: librosa import failed ({_librosa_exc}). Beat-alignment metrics that require librosa will be skipped.")
from sklearn.preprocessing import label_binarize
from scipy import stats, linalg
from scipy.signal import find_peaks, argrelextrema
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple, List, Optional, Union
import warnings
from types import SimpleNamespace
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, roc_auc_score
from culture_encoder.fishr_classifier import Fishr, evaluate_set

try:
    from numba.core.errors import NumbaPendingDeprecationWarning
except Exception:
    NumbaPendingDeprecationWarning = None

if NumbaPendingDeprecationWarning is not None:
    warnings.filterwarnings("ignore", category=NumbaPendingDeprecationWarning)

_NUMBA_DISABLE_JIT_ENV = str(os.environ.get("NUMBA_DISABLE_JIT", "")).strip().lower() in {"1", "true", "yes", "on"}
_LIBROSA_BEAT_TRACK_ENABLED = bool(librosa is not None and not _NUMBA_DISABLE_JIT_ENV)
_LIBROSA_BEAT_TRACK_DISABLE_REASON = (
    "NUMBA_DISABLE_JIT is enabled" if (librosa is not None and _NUMBA_DISABLE_JIT_ENV) else None
)
# Ai Choreographer reports sigma=3 with 60 FPS sequences.
BAS_SIGMA_REFERENCE_FPS = 60.0


################################################################################
# 1) R match scores for high-level and low-level alignments (only for generated motion)
################################################################################
def compute_r_match_scores_for_generated(
        motion_pooled: torch.Tensor,
        low_level_context: torch.Tensor,
        high_level_context: torch.Tensor,
        labels: torch.Tensor,
) -> Dict[str, Any]:
    """
    Computes alignment ('R match' scores) between:
      - motion_pooled (batch_size, 512) and high_level_context (batch_size, 512)
      - final_motion (batch_size, time_dim, 512) and low_level_context (batch_size, time_dim, 512)
    ONLY for generated motion batches.

    Parameters
    ----------
    motion_pooled : torch.Tensor
        Pooled version of final_motion of shape (batch_size, 512)
    low_level_context : torch.Tensor
        Low-level context of shape (batch_size, 512)
    high_level_context : torch.Tensor
        High-level context of shape (batch_size, 512)
    labels : torch.Tensor
        Culture labels of shape (batch_size,)

    Returns
    -------
    Dict[str, Any]
        {
            "r_match_high_level": <float or tensor>,
            "r_match_low_level": <float or tensor>
        }
        The dictionary can contain any statistics you want (averages, per-culture, etc.).
    """
    # Example approach: use cosine similarity as a rough measure of "R match".
    # You could also use dot product, correlation, or any custom alignment metric.

    # (1) High-level alignment (batch_size, 512) vs (batch_size, 512)
    #     shape: (batch_size,)
    r_match_high_level = F.cosine_similarity(motion_pooled, high_level_context, dim=-1)
    r_match_high_level_mean = r_match_high_level.mean().item()

    # (2) Low-level alignment (batch_size, time_dim, 512) vs (batch_size, time_dim, 512)
    #     We'll average across time_dim, then compute a similarity measure.
    #     shape: (batch_size, time_dim)
    r_match_low_level = F.cosine_similarity(motion_pooled, low_level_context, dim=-1)
    # Average across time_dim => shape: (batch_size,)
    #r_match_low_level = cos_sims.mean(dim=-1)
    r_match_low_level_mean = r_match_low_level.mean().item()

    return {
        "r_match_high_level": r_match_high_level_mean,
        "r_match_low_level": r_match_low_level_mean
    }



################################################################################
# 2) L2 distance between generated motion and real motion
################################################################################
def compute_l2_distance(
        real_motion: torch.Tensor,
        generated_motion: torch.Tensor
) -> Dict[str, Any]:
    """
    Computes the L2 distance (e.g., per-frame MSE or per-sequence MSE) between
    real motion and generated motion.

    Parameters
    ----------
    real_motion : torch.Tensor
        Real motion of shape (batch_size, time_dim, 512)
    generated_motion : torch.Tensor
        Generated motion of shape (batch_size, time_dim, 512)

    Returns
    -------
    Dict[str, Any]
        {
            "l2_distance_mean": <float>,
            "l2_distance_per_batch": <torch.Tensor of shape (batch_size,)>
        }
    """
    # Example: mean squared error across time_dim and feature_dim = L2 distance
    # shape: (batch_size, time_dim, 512)
    diff = real_motion - generated_motion

    # Option A: MSE across entire sequence
    per_batch_mse = (diff ** 2).mean(dim=(1, 2))  # (batch_size,)
    overall_mean_mse = per_batch_mse.mean().item()

    return {
        "l2_distance_mean": overall_mean_mse,
        "l2_distance_per_batch": per_batch_mse
    }


def compute_l1_distance(
        real_motion: torch.Tensor,
        generated_motion: torch.Tensor
) -> Dict[str, Any]:
    """
    Computes the L1 distance (Mean Absolute Error) between real motion and generated motion.

    Parameters
    ----------
    real_motion : torch.Tensor
        Real motion of shape (batch_size, time_dim, 512)
    generated_motion : torch.Tensor
        Generated motion of shape (batch_size, time_dim, 512)

    Returns
    -------
    Dict[str, Any]
        {
            "l1_distance_mean": <float>,
            "l1_distance_per_batch": <torch.Tensor of shape (batch_size,)>
        }
    """
    # Ensure the input tensors have the same shape
    if real_motion.shape != generated_motion.shape:
        raise ValueError(f"Shape mismatch: real_motion has shape {real_motion.shape}, "
                         f"but generated_motion has shape {generated_motion.shape}")

    # Compute the absolute difference
    diff = torch.abs(real_motion - generated_motion)

    # Compute L1 distance per batch (Mean Absolute Error across time and feature dimensions)
    per_batch_l1 = diff.mean(dim=(1, 2))  # Shape: (batch_size,)

    # Compute the overall mean L1 distance across all batches
    overall_mean_l1 = per_batch_l1.mean().item()

    return {
        "l1_distance_mean": overall_mean_l1,
        "l1_distance_per_batch": per_batch_l1
    }


def compute_fgd_scores_bootstrap(
    real_motion: torch.Tensor,
    gen_motion: torch.Tensor,
    real_culture_labels: torch.Tensor,
    gen_culture_labels: torch.Tensor,
    real_speaker_labels: torch.Tensor,
    gen_speaker_labels: torch.Tensor,
    num_cultures: int = 4
) -> Dict[str, Any]:
    """
    Computes various Frechet Gesture Distance (FGD) scores with balanced samples via bootstrapping.

    1) Overall FGD between generated and real motion.
    2) FGD per culture (generated vs real).
    3) Real-vs-Real: for each pair of cultures (c1, c2), treat c1 as "generated" and c2 as "real" (only real data).
    4) Generated-vs-Generated: for each pair of cultures (c1, c2), treat c1 as "generated" and c2 as "real" (only generated data).
    5) For real data, within each culture, consider different pairs of speakers; treat one speaker as "generated" and the other as "real". Compute and average.
    6) For generated data, same as (5).

    Parameters
    ----------
    real_motion : torch.Tensor
        Real motion, shape: (batch_size, time_dim, feature_dim=512)
    gen_motion : torch.Tensor
        Generated motion, shape: (batch_size, time_dim, feature_dim=512)
    real_culture_labels : torch.Tensor
        Encoded culture labels for real data, shape: (batch_size,)
    gen_culture_labels : torch.Tensor
        Encoded culture labels for generated data, shape: (batch_size,)
    real_speaker_labels : torch.Tensor
        Encoded speaker labels for real data, shape: (batch_size,)
    gen_speaker_labels : torch.Tensor
        Encoded speaker labels for generated data, shape: (batch_size,)
    num_cultures : int
        Number of different culture labels (default=4).

    Returns
    -------
    Dict[str, Any]
        A dictionary containing various FGD scores.
    """

    ###########################################################################
    #  Helper functions
    ###########################################################################
    def _split_into_two_groups(x: torch.Tensor, labels: torch.Tensor, culture: int, min_samples: int = 4) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """
        Splits the samples of a given culture into two distinct non-overlapping groups.
        Ensures that each group has at least min_samples to compute meaningful FGD.

        Parameters
        ----------
        x : torch.Tensor
            Motion embeddings, shape: (num_samples, time_dim, feature_dim)
        labels : torch.Tensor
            Culture labels corresponding to x, shape: (num_samples,)
        culture : int
            The culture ID to process.
        min_samples : int
            Minimum number of samples required per group.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Two tensors representing the two distinct groups. Returns (None, None) if split is not possible.
        """
        # Filter samples for the specified culture
        culture_idx = (labels == culture)
        culture_samples = x[culture_idx]
        num_culture_samples = culture_samples.shape[0]

        # Define the split ratio (e.g., 50-50 split)
        split_ratio = 0.5
        split_size = int(num_culture_samples * split_ratio)

        # Ensure that each group has at least min_samples
        if num_culture_samples < 2 * min_samples:
            return None, None

        # Shuffle the samples
        shuffled_indices = torch.randperm(num_culture_samples)
        split_indices_1 = shuffled_indices[:split_size]
        split_indices_2 = shuffled_indices[split_size:]

        group1 = culture_samples[split_indices_1]
        group2 = culture_samples[split_indices_2]

        return group1, group2

    def _compute_mean_cov(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Flattens x from (batch_size, time_dim, 512) to (batch_size * time_dim, 512),
        then computes mean and covariance.
        """
        x_flat = x.reshape(-1, x.shape[-1])  # shape: (B*T, 512)
        mu = x_flat.mean(dim=0)  # (512,)
        diff = x_flat - mu
        cov = (diff.t() @ diff) / (x_flat.shape[0] - 1)  # (512, 512)
        return mu, cov

    def _matrix_sqrt(m: torch.Tensor) -> torch.Tensor:
        """
        Computes the matrix square root using eigenvalue decomposition.
        This method is stable for symmetric positive semi-definite matrices.
        """
        # Ensure the matrix is symmetric
        m = (m + m.t()) / 2

        # Eigenvalue decomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(m)  # Eigen decomposition

        # Clamp eigenvalues to avoid numerical issues (ensure non-negativity)
        eigenvalues_clamped = torch.clamp(eigenvalues, min=1e-10)

        # Compute the square root of the matrix
        sqrt_m = eigenvectors @ torch.diag(eigenvalues_clamped.sqrt()) @ eigenvectors.t()
        return sqrt_m

    def _frechet_distance(mu1, cov1, mu2, cov2):
        """
        Computes Frechet distance analogous to FID:
            FGD = ||mu1 - mu2||^2 + trace(cov1 + cov2 - 2 * sqrt(cov1 * cov2))
        """
        diff = mu1 - mu2
        diff_sq = diff.dot(diff)

        cov_sum = cov1 + cov2
        try:
            cov_prod_sqrt = _matrix_sqrt(cov1 @ cov2)
            trace_component = torch.trace(cov_sum - 2.0 * cov_prod_sqrt)
        except Exception:
            # Fallback if sqrt fails
            trace_component = torch.trace(cov_sum)

        return diff_sq + trace_component

    def _compute_fgd_distribution(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience: returns (mu, cov) for a motion batch (B, T, 512).
        If x has fewer than 2 frames total, returns None, None to skip.
        """
        if x.shape[0] < 1:
            return None, None
        # If there's not enough data to compute meaningful covariance (e.g. 1 sample),
        # you may want to skip or fallback. We'll just handle it with shape checks:
        if x.shape[0] * x.shape[1] < 2:
            return None, None
        return _compute_mean_cov(x)

    def _bootstrap_balance(x: torch.Tensor, culture_labels: torch.Tensor, speaker_labels: torch.Tensor, num_cultures: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Balances the dataset by resampling each culture to have the same number of samples.
        Uses bootstrapping (sampling with replacement) if necessary.

        Parameters
        ----------
        x : torch.Tensor
            Motion embeddings, shape: (batch_size, time_dim, feature_dim)
        culture_labels : torch.Tensor
            Encoded culture labels, shape: (batch_size,)
        speaker_labels : torch.Tensor
            Encoded speaker labels, shape: (batch_size,)
        num_cultures : int
            Number of different cultures.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Balanced motion embeddings, corresponding culture labels, and speaker labels.
        """
        # Determine the minimum number of samples across all cultures
        counts = [(culture_labels == c).sum().item() for c in range(num_cultures)]
        # Exclude cultures with zero samples to avoid min on empty list
        counts_non_zero = [count for count in counts if count > 0]
        if not counts_non_zero:
            raise ValueError("No samples found for any culture.")
        min_count = min(counts_non_zero)

        balanced_indices = []

        for c in range(num_cultures):
            c_indices = torch.where(culture_labels == c)[0]
            num_c = len(c_indices)
            if num_c == 0:
                continue  # Skip cultures with no samples

            if num_c >= min_count:
                # Sample without replacement
                selected = c_indices[torch.randperm(num_c)[:min_count]]
            else:
                # Sample with replacement
                selected = c_indices[torch.randint(0, num_c, (min_count,), dtype=torch.long)]
            balanced_indices.append(selected)

        if not balanced_indices:
            raise ValueError("No samples found after balancing.")

        # Concatenate all selected indices and shuffle
        balanced_indices = torch.cat(balanced_indices)
        balanced_indices = balanced_indices[torch.randperm(len(balanced_indices))]

        return x[balanced_indices], culture_labels[balanced_indices], speaker_labels[balanced_indices]

    ###########################################################################
    #  Bootstrap Balancing for Real and Generated Data
    ###########################################################################
    balanced_real_motion, balanced_real_culture_labels, balanced_real_speaker_labels = _bootstrap_balance(
        real_motion, real_culture_labels, real_speaker_labels, num_cultures
    )
    balanced_gen_motion, balanced_gen_culture_labels, balanced_gen_speaker_labels = _bootstrap_balance(
        gen_motion, gen_culture_labels, gen_speaker_labels, num_cultures
    )

    ###########################################################################
    # 1) Overall FGD (real vs. generated)
    ###########################################################################
    mu_r, cov_r = _compute_mean_cov(balanced_real_motion)
    mu_g, cov_g = _compute_mean_cov(balanced_gen_motion)
    fgd_overall = _frechet_distance(mu_r, cov_r, mu_g, cov_g).item()

    ###########################################################################
    # 2) FGD per culture (generated vs real)
    ###########################################################################
    # We'll assume balanced_real_culture_labels and balanced_gen_culture_labels exist
    real_culture_enc = balanced_real_culture_labels
    gen_culture_enc = balanced_gen_culture_labels

    fgd_per_culture = {}
    for c in range(num_cultures):
        real_idx = (real_culture_enc == c)
        gen_idx = (gen_culture_enc == c)

        if real_idx.sum() < 1 or gen_idx.sum() < 1:
            fgd_per_culture[c] = None
            continue

        mu_r_c, cov_r_c = _compute_mean_cov(balanced_real_motion[real_idx])
        mu_g_c, cov_g_c = _compute_mean_cov(balanced_gen_motion[gen_idx])

        # If we can't compute the covariance, skip
        if mu_r_c is None or mu_g_c is None:
            fgd_per_culture[c] = None
            continue

        fgd_per_culture[c] = _frechet_distance(mu_r_c, cov_r_c, mu_g_c, cov_g_c).item()

    ###########################################################################
    # 3) Real vs Real, for each pair of cultures
    #    c1 as "generated" data, c2 as "real" data
    ###########################################################################
    fgd_real_culture_couples = {}
    all_cultures = list(range(num_cultures))
    for (c1, c2) in combinations(all_cultures, 2):
        c1_idx = (real_culture_enc == c1)
        c2_idx = (real_culture_enc == c2)

        # c1 -> "generated", c2 -> "real"
        real_motion_c1 = balanced_real_motion[c1_idx]
        real_motion_c2 = balanced_real_motion[c2_idx]

        mu_c1, cov_c1 = _compute_fgd_distribution(real_motion_c1)
        mu_c2, cov_c2 = _compute_fgd_distribution(real_motion_c2)
        if mu_c1 is None or cov_c1 is None or mu_c2 is None or cov_c2 is None:
            fgd_value = None
        else:
            fgd_value = _frechet_distance(mu_c1, cov_c1, mu_c2, cov_c2).item()

        fgd_real_culture_couples[(c1, c2)] = fgd_value

        # Opposite direction (c2->"generated", c1->"real")
        if mu_c2 is None or cov_c2 is None or mu_c1 is None or cov_c1 is None:
            fgd_value_rev = None
        else:
            fgd_value_rev = _frechet_distance(mu_c2, cov_c2, mu_c1, cov_c1).item()

        # Store it under (c2, c1) to keep track
        fgd_real_culture_couples[(c2, c1)] = fgd_value_rev

    ###########################################################################
    # 4) Generated vs Generated, for each pair of cultures
    #    c1 as "generated", c2 as "real" -- but all from the generated data
    ###########################################################################
    fgd_gen_culture_couples = {}
    for (c1, c2) in combinations(all_cultures, 2):
        c1_idx = (gen_culture_enc == c1)
        c2_idx = (gen_culture_enc == c2)

        gen_motion_c1 = balanced_gen_motion[c1_idx]
        gen_motion_c2 = balanced_gen_motion[c2_idx]

        mu_c1, cov_c1 = _compute_fgd_distribution(gen_motion_c1)
        mu_c2, cov_c2 = _compute_fgd_distribution(gen_motion_c2)
        if mu_c1 is None or cov_c1 is None or mu_c2 is None or cov_c2 is None:
            fgd_value = None
        else:
            fgd_value = _frechet_distance(mu_c1, cov_c1, mu_c2, cov_c2).item()

        fgd_gen_culture_couples[(c1, c2)] = fgd_value

        # Opposite direction (c2->generated, c1->real)
        if mu_c2 is None or cov_c2 is None or mu_c1 is None or cov_c1 is None:
            fgd_value_rev = None
        else:
            fgd_value_rev = _frechet_distance(mu_c2, cov_c2, mu_c1, cov_c1).item()

        # Store it under (c2, c1) to keep track
        fgd_gen_culture_couples[(c2, c1)] = fgd_value_rev

    ###########################################################################
    # 5) Real Data: For each culture, consider different speakers inside that culture
    #    treat one speaker as "generated", the other as "real". Compute for each pair,
    #    then average the results.
    ###########################################################################
    real_speaker_enc = balanced_real_speaker_labels
    fgd_real_speaker_couples = {}

    # We'll group speaker IDs by culture:
    #   culture -> set/list of speaker_enc
    culture_speakers_real = {}
    for c in range(num_cultures):
        culture_speakers_real[c] = torch.unique(real_speaker_enc[real_culture_enc == c]).tolist()

    # For each culture, compute FGD across speaker pairs
    for c in range(num_cultures):
        spk_list = culture_speakers_real[c]
        if len(spk_list) < 2:
            fgd_real_speaker_couples[c] = None
            continue

        results = []
        # Consider all pairs of speakers (with or without repetition in both directions)
        for s1, s2 in combinations(spk_list, 2):
            s1_idx = (real_speaker_enc == s1) & (real_culture_enc == c)
            s2_idx = (real_speaker_enc == s2) & (real_culture_enc == c)

            # s1 -> "generated", s2 -> "real"
            motion_s1 = balanced_real_motion[s1_idx]
            motion_s2 = balanced_real_motion[s2_idx]
            mu_s1, cov_s1 = _compute_fgd_distribution(motion_s1)
            mu_s2, cov_s2 = _compute_fgd_distribution(motion_s2)
            if mu_s1 is not None and cov_s1 is not None and mu_s2 is not None and cov_s2 is not None:
                dist_12 = _frechet_distance(mu_s1, cov_s1, mu_s2, cov_s2).item()
                results.append(dist_12)

            # s2 -> "generated", s1 -> "real"
            if mu_s2 is not None and cov_s2 is not None and mu_s1 is not None and cov_s1 is not None:
                dist_21 = _frechet_distance(mu_s2, cov_s2, mu_s1, cov_s1).item()
                results.append(dist_21)

        if len(results) == 0:
            fgd_real_speaker_couples[c] = None
        else:
            fgd_real_speaker_couples[c] = float(torch.tensor(results).mean())

    ###########################################################################
    # 6) Generated Data: For each culture, consider different speakers
    #    (based on gen_speaker_labels), treat them as "generated" vs "real".
    ###########################################################################
    gen_speaker_enc = balanced_gen_speaker_labels
    fgd_gen_speaker_couples = {}

    culture_speakers_gen = {}
    for c in range(num_cultures):
        culture_speakers_gen[c] = torch.unique(gen_speaker_enc[gen_culture_enc == c]).tolist()

    for c in range(num_cultures):
        spk_list = culture_speakers_gen[c]
        if len(spk_list) < 2:
            fgd_gen_speaker_couples[c] = None
            continue

        results = []
        for s1, s2 in combinations(spk_list, 2):
            s1_idx = (gen_speaker_enc == s1) & (gen_culture_enc == c)
            s2_idx = (gen_speaker_enc == s2) & (gen_culture_enc == c)

            motion_s1 = balanced_gen_motion[s1_idx]
            motion_s2 = balanced_gen_motion[s2_idx]
            mu_s1, cov_s1 = _compute_fgd_distribution(motion_s1)
            mu_s2, cov_s2 = _compute_fgd_distribution(motion_s2)
            if mu_s1 is not None and cov_s1 is not None and mu_s2 is not None and cov_s2 is not None:
                dist_12 = _frechet_distance(mu_s1, cov_s1, mu_s2, cov_s2).item()
                results.append(dist_12)

                dist_21 = _frechet_distance(mu_s2, cov_s2, mu_s1, cov_s1).item()
                results.append(dist_21)

        if len(results) == 0:
            fgd_gen_speaker_couples[c] = None
        else:
            fgd_gen_speaker_couples[c] = float(torch.tensor(results).mean())

    ###########################################################################
    # 7) Intra-Culture FGD for Real Samples
    ###########################################################################
    fgd_real_intra_culture = {}
    for c in range(num_cultures):
        group1, group2 = _split_into_two_groups(balanced_real_motion, real_culture_enc, c)
        if group1 is None or group2 is None:
            fgd_real_intra_culture[c] = None
            continue

        mu1, cov1 = _compute_fgd_distribution(group1)
        mu2, cov2 = _compute_fgd_distribution(group2)

        if mu1 is None or cov1 is None or mu2 is None or cov2 is None:
            fgd_real_intra_culture[c] = None
            continue

        fgd_real_intra_culture[c] = _frechet_distance(mu1, cov1, mu2, cov2).item()

    ###########################################################################
    # 8) Intra-Culture FGD for Generated Samples
    ###########################################################################
    fgd_gen_intra_culture = {}
    for c in range(num_cultures):
        group1, group2 = _split_into_two_groups(balanced_gen_motion, gen_culture_enc, c)
        if group1 is None or group2 is None:
            fgd_gen_intra_culture[c] = None
            continue

        mu1, cov1 = _compute_fgd_distribution(group1)
        mu2, cov2 = _compute_fgd_distribution(group2)

        if mu1 is None or cov1 is None or mu2 is None or cov2 is None:
            fgd_gen_intra_culture[c] = None
            continue

        fgd_gen_intra_culture[c] = _frechet_distance(mu1, cov1, mu2, cov2).item()

    ###########################################################################
    # Prepare final results
    ###########################################################################
    results = {
        "fgd_overall": fgd_overall,
        "fgd_per_culture": fgd_per_culture,

        "fgd_real_culture_couples": fgd_real_culture_couples,
        "fgd_gen_culture_couples": fgd_gen_culture_couples,

        "fgd_real_speaker_couples": fgd_real_speaker_couples,
        "fgd_gen_speaker_couples": fgd_gen_speaker_couples,

        "fgd_real_intra_culture": fgd_real_intra_culture,
        "fgd_gen_intra_culture": fgd_gen_intra_culture
    }

    return results


'''
################################################################################
# 3) FGD (Frechet Gesture Distance) scores
#    (a) Overall FGD between real and generated
#    (b) FGD for each culture label (real vs generated per culture)
#    (c) FGD for each pair of culture labels, for both real and generated
#    + a statistical test for significance of differences
################################################################################
def compute_fgd_scores(
    real_motion: torch.Tensor,
    gen_motion: torch.Tensor,
    real_culture_labels: torch.Tensor,
    gen_culture_labels: torch.Tensor,
    real_speaker_labels: torch.Tensor ,
    gen_speaker_labels: torch.Tensor,
    num_cultures: int = 4
):
    """
    Computes various Frechet Gesture Distance (FGD) scores:

    1) Overall FGD between generated and real motion.
    2) FGD per culture (generated vs real).
    3) Real-vs-Real: for each pair of cultures (c1, c2), treat c1 as "generated" and c2 as "real" (only real data).
    4) Generated-vs-Generated: for each pair of cultures (c1, c2), treat c1 as "generated" and c2 as "real" (only generated data).
    5) For real data, within each culture, consider different pairs of speakers; treat one speaker as "generated" and the other as "real". Compute and average.
    6) For generated data, same as (5).

    Parameters
    ----------
    real_motion : torch.Tensor
        Real motion, shape: (batch_size, time_dim, feature_dim=512)
    gen_motion : torch.Tensor
        Generated motion, shape: (batch_size, time_dim, feature_dim=512)
    real_labels : dict
        Dictionary of labels for real data. Example structure:
            real_labels = {
                'culture_enc': torch.Tensor of shape (batch_size,),
                'speaker_enc': torch.Tensor of shape (batch_size,),
                'culture_raw': list or array of str,
                'speaker_raw': list or array of str,
                'text_data': ...,
                ...
            }
    gen_labels : dict
        Dictionary of labels for generated data, with same keys as real_labels.
    num_cultures : int
        Number of different culture labels (default=4).

    Returns
    -------
    Dict[str, Any]
    {
      "fgd_overall": float,
      "fgd_per_culture": { culture_id: float or None },

      "fgd_real_culture_couples": { (c1, c2): float or None },
      "fgd_gen_culture_couples": { (c1, c2): float or None },

      "fgd_real_speaker_couples": { culture_id: (float or None) },
      "fgd_gen_speaker_couples": { culture_id: (float or None) },
    }
    """

    ###########################################################################
    #  Helper functions
    ###########################################################################
    def _split_into_two_groups(x: torch.Tensor, labels: torch.Tensor, culture: int, min_samples: int = 4) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """
        Splits the samples of a given culture into two distinct non-overlapping groups.
        Ensures that each group has at least min_samples to compute meaningful FGD.

        Parameters
        ----------
        x : torch.Tensor
            Motion embeddings, shape: (num_samples, time_dim, feature_dim)
        labels : torch.Tensor
            Culture labels corresponding to x, shape: (num_samples,)
        culture : int
            The culture ID to process.
        min_samples : int
            Minimum number of samples required per group.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Two tensors representing the two distinct groups. Returns (None, None) if split is not possible.
        """
        # Filter samples for the specified culture
        culture_idx = (labels == culture)
        culture_samples = x[culture_idx]
        num_culture_samples = culture_samples.shape[0]

        # Define the split ratio (e.g., 50-50 split)
        split_ratio = 0.5
        split_size = int(num_culture_samples * split_ratio)

        # Ensure that each group has at least min_samples
        if num_culture_samples < 2 * min_samples:
            return None, None

        # Shuffle the samples
        shuffled_indices = torch.randperm(num_culture_samples)
        split_indices_1 = shuffled_indices[:split_size]
        split_indices_2 = shuffled_indices[split_size:]

        group1 = culture_samples[split_indices_1]
        group2 = culture_samples[split_indices_2]

        return group1, group2

    def _compute_mean_cov(x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Flattens x from (batch_size, time_dim, 512) to (batch_size * time_dim, 512),
        then computes mean and covariance.
        """
        x_flat = x.reshape(-1, x.shape[-1])  # shape: (B*T, 512)
        mu = x_flat.mean(dim=0)  # (512,)
        diff = x_flat - mu
        cov = (diff.t() @ diff) / (x_flat.shape[0] - 1)  # (512, 512)
        return mu, cov

    def _matrix_sqrt(m: torch.Tensor) -> torch.Tensor:
        """
        Computes the matrix square root using eigenvalue decomposition.
        This method is stable for symmetric positive semi-definite matrices.
        """
        # Ensure the matrix is symmetric
        m = (m + m.t()) / 2

        # Eigenvalue decomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(m)  # Eigen decomposition

        # Clamp eigenvalues to avoid numerical issues (ensure non-negativity)
        eigenvalues_clamped = torch.clamp(eigenvalues, min=1e-10)

        # Compute the square root of the matrix
        sqrt_m = eigenvectors @ torch.diag(eigenvalues_clamped.sqrt()) @ eigenvectors.t()
        return sqrt_m

    def _frechet_distance(mu1, cov1, mu2, cov2):
        """
        Computes Frechet distance analogous to FID:
            FGD = ||mu1 - mu2||^2 + trace(cov1 + cov2 - 2 * sqrt(cov1 * cov2))
        """
        diff = mu1 - mu2
        diff_sq = diff.dot(diff)

        cov_sum = cov1 + cov2
        try:
            cov_prod_sqrt = _matrix_sqrt(cov1 @ cov2)
            trace_component = torch.trace(cov_sum - 2.0 * cov_prod_sqrt)
        except Exception:
            # Fallback if sqrt fails
            trace_component = torch.trace(cov_sum)

        return diff_sq + trace_component

    def _compute_fgd_distribution(x: torch.Tensor) -> tuple:
        """
        Convenience: returns (mu, cov) for a motion batch (B, T, 512).
        If x has fewer than 2 frames total, returns None, None to skip.
        """
        if x.shape[0] < 1:
            return None, None
        # If there's not enough data to compute meaningful covariance (e.g. 1 sample),
        # you may want to skip or fallback. We'll just handle it with shape checks:
        if x.shape[0] * x.shape[1] < 2:
            return None, None
        return _compute_mean_cov(x)

    ###########################################################################
    # 1) Overall FGD (real vs. generated)
    ###########################################################################
    mu_r, cov_r = _compute_mean_cov(real_motion)
    mu_g, cov_g = _compute_mean_cov(gen_motion)
    fgd_overall = _frechet_distance(mu_r, cov_r, mu_g, cov_g).item()

    ###########################################################################
    # 2) FGD per culture (generated vs real)
    ###########################################################################
    # We'll assume real_labels['culture_enc'] and gen_labels['culture_enc'] exist
    real_culture_enc = real_culture_labels
    gen_culture_enc = gen_culture_labels

    fgd_per_culture = {}
    for c in range(num_cultures):
        real_idx = (real_culture_enc == c)
        gen_idx = (gen_culture_enc == c)

        if real_idx.sum() < 1 or gen_idx.sum() < 1:
            fgd_per_culture[c] = None
            continue

        mu_r_c, cov_r_c = _compute_mean_cov(real_motion[real_idx])
        mu_g_c, cov_g_c = _compute_mean_cov(gen_motion[gen_idx])

        # If we can't compute the covariance, skip
        if mu_r_c is None or mu_g_c is None:
            fgd_per_culture[c] = None
            continue

        fgd_per_culture[c] = _frechet_distance(mu_r_c, cov_r_c, mu_g_c, cov_g_c).item()

    ###########################################################################
    # 3) Real vs Real, for each pair of cultures
    #    c1 as "generated" data, c2 as "real" data
    ###########################################################################
    fgd_real_culture_couples = {}
    all_cultures = list(range(num_cultures))
    for (c1, c2) in combinations(all_cultures, 2):
        c1_idx = (real_culture_enc == c1)
        c2_idx = (real_culture_enc == c2)

        # c1 -> "generated", c2 -> "real"
        real_motion_c1 = real_motion[c1_idx]
        real_motion_c2 = real_motion[c2_idx]

        mu_c1, cov_c1 = _compute_fgd_distribution(real_motion_c1)
        mu_c2, cov_c2 = _compute_fgd_distribution(real_motion_c2)
        if mu_c1 is None or mu_c2 is None:
            fgd_value = None
        else:
            fgd_value = _frechet_distance(mu_c1, cov_c1, mu_c2, cov_c2).item()

        fgd_real_culture_couples[(c1, c2)] = fgd_value

        # Optionally, if you also want the opposite direction (c2->"generated", c1->"real"),
        # you can do that as well. Below is how you'd do it. If you do NOT want it, comment it out.
        mu_c2_rev, cov_c2_rev = _compute_fgd_distribution(real_motion_c2)
        mu_c1_rev, cov_c1_rev = _compute_fgd_distribution(real_motion_c1)
        if mu_c2_rev is None or mu_c1_rev is None:
            fgd_value_rev = None
        else:
            fgd_value_rev = _frechet_distance(mu_c2_rev, cov_c2_rev, mu_c1_rev, cov_c1_rev).item()

        # We'll store it under (c2, c1) to keep track:
        fgd_real_culture_couples[(c2, c1)] = fgd_value_rev

    ###########################################################################
    # 4) Generated vs Generated, for each pair of cultures
    #    c1 as "generated", c2 as "real" -- but all from the generated data
    ###########################################################################
    fgd_gen_culture_couples = {}
    gen_culture_enc = gen_culture_labels
    for (c1, c2) in combinations(all_cultures, 2):
        c1_idx = (gen_culture_enc == c1)
        c2_idx = (gen_culture_enc == c2)

        gen_motion_c1 = gen_motion[c1_idx]
        gen_motion_c2 = gen_motion[c2_idx]

        mu_c1, cov_c1 = _compute_fgd_distribution(gen_motion_c1)
        mu_c2, cov_c2 = _compute_fgd_distribution(gen_motion_c2)
        if mu_c1 is None or mu_c2 is None:
            fgd_value = None
        else:
            fgd_value = _frechet_distance(mu_c1, cov_c1, mu_c2, cov_c2).item()

        fgd_gen_culture_couples[(c1, c2)] = fgd_value

        # Opposite direction (c2->generated, c1->real)
        mu_c2_rev, cov_c2_rev = _compute_fgd_distribution(gen_motion_c2)
        mu_c1_rev, cov_c1_rev = _compute_fgd_distribution(gen_motion_c1)
        if mu_c2_rev is None or mu_c1_rev is None:
            fgd_value_rev = None
        else:
            fgd_value_rev = _frechet_distance(mu_c2_rev, cov_c2_rev, mu_c1_rev, cov_c1_rev).item()

        fgd_gen_culture_couples[(c2, c1)] = fgd_value_rev

    ###########################################################################
    # 5) Real Data: For each culture, consider different speakers inside that culture
    #    treat one speaker as "generated", the other as "real". Compute for each pair,
    #    then average the results.
    ###########################################################################
    real_speaker_enc = real_speaker_labels
    fgd_real_speaker_couples = {}

    # We'll group speaker IDs by culture:
    #   culture -> set/list of speaker_enc
    culture_speakers_real = {}
    for c in range(num_cultures):
        culture_speakers_real[c] = torch.unique(real_speaker_enc[real_culture_enc == c]).tolist()

    # For each culture, compute FGD across speaker pairs
    for c in range(num_cultures):
        spk_list = culture_speakers_real[c]
        if len(spk_list) < 2:
            fgd_real_speaker_couples[c] = None
            continue

        results = []
        # Consider all pairs of speakers (with or without repetition in both directions)
        for s1, s2 in combinations(spk_list, 2):
            s1_idx = (real_speaker_enc == s1) & (real_culture_enc == c)
            s2_idx = (real_speaker_enc == s2) & (real_culture_enc == c)

            # s1 -> "generated", s2 -> "real"
            motion_s1 = real_motion[s1_idx]
            motion_s2 = real_motion[s2_idx]
            mu_s1, cov_s1 = _compute_fgd_distribution(motion_s1)
            mu_s2, cov_s2 = _compute_fgd_distribution(motion_s2)
            if mu_s1 is not None and mu_s2 is not None:
                dist_12 = _frechet_distance(mu_s1, cov_s1, mu_s2, cov_s2).item()
                results.append(dist_12)

            # s2 -> "generated", s1 -> "real"
            if mu_s2 is not None and mu_s1 is not None:
                dist_21 = _frechet_distance(mu_s2, cov_s2, mu_s1, cov_s1).item()
                results.append(dist_21)

        if len(results) == 0:
            fgd_real_speaker_couples[c] = None
        else:
            fgd_real_speaker_couples[c] = float(torch.tensor(results).mean())

    ###########################################################################
    # 6) Generated Data: For each culture, consider different speakers
    #    (based on gen_labels['speaker_enc']), treat them as "generated" vs "real".
    ###########################################################################
    gen_speaker_enc = gen_speaker_labels
    fgd_gen_speaker_couples = {}

    culture_speakers_gen = {}
    for c in range(num_cultures):
        culture_speakers_gen[c] = torch.unique(gen_speaker_enc[gen_culture_enc == c]).tolist()

    for c in range(num_cultures):
        spk_list = culture_speakers_gen[c]
        if len(spk_list) < 2:
            fgd_gen_speaker_couples[c] = None
            continue

        results = []
        for s1, s2 in combinations(spk_list, 2):
            s1_idx = (gen_speaker_enc == s1) & (gen_culture_enc == c)
            s2_idx = (gen_speaker_enc == s2) & (gen_culture_enc == c)

            motion_s1 = gen_motion[s1_idx]
            motion_s2 = gen_motion[s2_idx]
            mu_s1, cov_s1 = _compute_fgd_distribution(motion_s1)
            mu_s2, cov_s2 = _compute_fgd_distribution(motion_s2)
            if mu_s1 is not None and mu_s2 is not None:
                dist_12 = _frechet_distance(mu_s1, cov_s1, mu_s2, cov_s2).item()
                results.append(dist_12)

                dist_21 = _frechet_distance(mu_s2, cov_s2, mu_s1, cov_s1).item()
                results.append(dist_21)

        if len(results) == 0:
            fgd_gen_speaker_couples[c] = None
        else:
            fgd_gen_speaker_couples[c] = float(torch.tensor(results).mean())

    ###########################################################################
    # 7) Intra-Culture FGD for Real Samples
    ###########################################################################
    fgd_real_intra_culture = {}
    for c in range(num_cultures):
        group1, group2 = _split_into_two_groups(real_motion, real_culture_labels, c)
        if group1 is None or group2 is None:
            fgd_real_intra_culture[c] = None
            continue

        mu1, cov1 = _compute_fgd_distribution(group1)
        mu2, cov2 = _compute_fgd_distribution(group2)

        if mu1 is None or mu2 is None:
            fgd_real_intra_culture[c] = None
            continue

        fgd_real_intra_culture[c] = _frechet_distance(mu1, cov1, mu2, cov2)

    ###########################################################################
    # 8) Intra-Culture FGD for Generated Samples
    ###########################################################################
    fgd_gen_intra_culture = {}
    for c in range(num_cultures):
        group1, group2 = _split_into_two_groups(gen_motion, gen_culture_labels, c)
        if group1 is None or group2 is None:
            fgd_gen_intra_culture[c] = None
            continue

        mu1, cov1 = _compute_fgd_distribution(group1)
        mu2, cov2 = _compute_fgd_distribution(group2)

        if mu1 is None or mu2 is None:
            fgd_gen_intra_culture[c] = None
            continue

        fgd_gen_intra_culture[c] = _frechet_distance(mu1, cov1, mu2, cov2)

    ###########################################################################
    # Prepare final results
    ###########################################################################
    results = {
        "fgd_overall": fgd_overall,
        "fgd_per_culture": fgd_per_culture,

        "fgd_real_culture_couples": fgd_real_culture_couples,
        "fgd_gen_culture_couples": fgd_gen_culture_couples,

        "fgd_real_speaker_couples": fgd_real_speaker_couples,
        "fgd_gen_speaker_couples": fgd_gen_speaker_couples,

        "fgd_real_intra_culture": fgd_real_intra_culture,
        "fgd_gen_intra_culture": fgd_gen_intra_culture
    }

    return results
'''


# Helper function to compute activation statistics (mean and covariance) using PyTorch
def _compute_activation_statistics_pt(x: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Computes mean and sample covariance for features.

    Parameters
    ----------
    x : torch.Tensor
        Input features, shape: (batch_size, time_dim, feature_dim) or (num_samples, feature_dim)

    Returns
    -------
    Optional[Tuple[torch.Tensor, torch.Tensor]]
        (mean, covariance) tensors, or (None, None) if computation is not possible.
        Mean shape: (feature_dim,)
        Covariance shape: (feature_dim, feature_dim)
    """
    if x.dim() == 3:
        # Flatten (batch_size, time_dim, feature_dim) to (batch_size * time_dim, feature_dim)
        num_samples, time_dim, feature_dim = x.shape
        x_flat = x.reshape(-1, feature_dim)
    elif x.dim() == 2:
        # Assume input is already (num_samples, feature_dim)
        num_samples, feature_dim = x.shape
        x_flat = x
        time_dim = 1 # To make the check below work
    else:
        warnings.warn(f"Input tensor has unexpected dimension {x.dim()}, expected 2 or 3. Skipping.")
        return None, None

    # Check if enough data points exist to compute covariance (at least 2 feature vectors)
    if x_flat.shape[0] < 2:
         warnings.warn(f"Not enough samples ({x_flat.shape[0]}) to compute covariance. Need at least 2. Skipping.")
         return None, None

    mu = torch.mean(x_flat, dim=0)
    # Use torch.cov for sample covariance (divides by N-1 by default)
    # Note: torch.cov expects features in rows, samples in columns, so transpose needed if N > D
    # If samples < features, behavior might differ. Let's stick to manual calculation for clarity with (N, D) input.
    diff = x_flat - mu
    cov = (diff.t() @ diff) / (x_flat.shape[0] - 1)

    # Add a small epsilon for numerical stability if covariance matrix is nearly singular
    # This helps prevent issues in the downstream sqrtm calculation
    # Check if diagonal has very small or zero values
    # min_diag = torch.min(torch.diag(cov))
    # if min_diag < 1e-6:
       # eps_val = 1e-6
       # cov += torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype) * eps_val
       # warnings.warn(f"Added epsilon {eps_val} to covariance diagonal for numerical stability.")

    # Alternative stability check (condition number) - might be too slow
    # cond = torch.linalg.cond(cov)
    # if cond > 1/torch.finfo(cov.dtype).eps: # Check if condition number is very large
    #    warnings.warn("Covariance matrix is ill-conditioned.")
    #    eps_val = 1e-6
    #    cov += torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype) * eps_val

    return mu, cov


# Frechet Distance calculation using SciPy for robustness
def _calculate_frechet_distance_np(mu1, sigma1, mu2, sigma2, eps=1e-6) -> Optional[float]:
    """Numpy implementation of the Frechet Distance using scipy.linalg.sqrtm.

    Stable version adapted from FID computation code.

    Params:
    -- mu1, mu2 : Numpy array mean vectors.
    -- sigma1, sigma2: Numpy array covariance matrices.
    -- eps: Epsilon for numerical stability if matrix product is singular.

    Returns:
    -- float or None : The Frechet Distance, or None if calculation fails.
    """
    # Ensure inputs are NumPy arrays
    mu1 = np.asarray(mu1, dtype=np.float64)
    sigma1 = np.asarray(sigma1, dtype=np.float64)
    mu2 = np.asarray(mu2, dtype=np.float64)
    sigma2 = np.asarray(sigma2, dtype=np.float64)

    if mu1.shape != mu2.shape:
        warnings.warn(f"Mean vectors have different lengths: {mu1.shape} vs {mu2.shape}")
        return None
    if sigma1.shape != sigma2.shape:
        warnings.warn(f"Covariance matrices have different dimensions: {sigma1.shape} vs {sigma2.shape}")
        return None

    diff = mu1 - mu2

    # Calculate squared norm of mean difference
    diff_sq_norm = np.dot(diff, diff)

    # Calculate sqrt of covariance matrix product
    try:
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False) # Try computing sqrtm
    except Exception as e:
         warnings.warn(f"sqrtm calculation failed initially: {e}. Adding epsilon {eps}.")
         # Add epsilon and retry if sqrtm fails
         offset = np.eye(sigma1.shape[0]) * eps
         try:
             covmean, _ = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset), disp=False)
         except Exception as e2:
             warnings.warn(f"sqrtm failed even after adding epsilon: {e2}. Returning None.")
             return None # Return None if it fails even with epsilon

    # Check for numerical errors leading to non-finite results in covmean
    if not np.isfinite(covmean).all():
        warnings.warn(f"FID calculation produces non-finite elements in sqrtm result even after adding epsilon {eps}. Returning None.")
        # Add epsilon again just in case (though previous try/except should handle this)
        # offset = np.eye(sigma1.shape[0]) * eps
        # try:
        #     covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        #     if not np.isfinite(covmean).all():
        #          warnings.warn(f"FID calculation still produces non-finite elements after adding epsilon {eps}. Returning None.")
        #          return None
        # except Exception as e3:
        #      warnings.warn(f"sqrtm retry failed: {e3}. Returning None.")
        #      return None
        return None # If still not finite, fail gracefully

    # Handle potential complex numbers resulting from numerical errors
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            warnings.warn(f"sqrtm result has non-negligible imaginary component ({m}). Handling cautiously.")
            # Depending on policy, either return None or take real part if imaginary part is truly small
            # For safety, let's return None if imaginary part is significant
            # return None
        covmean = covmean.real # Take real part (assuming imaginary part is negligible)

    # Calculate the trace component: trace(sigma1 + sigma2 - 2 * covmean)
    tr_covmean = np.trace(covmean)
    trace_component = np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    # Final Frechet distance
    fd = diff_sq_norm + trace_component

    # Handle potential negative results due to numerical instability
    if fd < 0:
        warnings.warn(f"Calculated Frechet distance is negative ({fd}). Clamping to 0.")
        return 0.0

    return float(fd)


# Main function
def compute_fgd_scores(
    real_motion: torch.Tensor,
    gen_motion: torch.Tensor,
    real_culture_labels: torch.Tensor,
    gen_culture_labels: torch.Tensor,
    real_speaker_labels: torch.Tensor,
    gen_speaker_labels: torch.Tensor,
    num_cultures: int = 4,
    min_samples_for_fgd: int = 5, # Minimum samples needed per group for stable FGD
    compute_speaker_couples: bool = False,
) -> Dict[str, Any]:
    """
    Computes various Frechet Gesture Distance (FGD) scores using robust calculation.

    Calculations include:
    1) Overall FGD (real vs. generated).
    2) FGD per culture (real vs. generated).
    3) Real-vs-Real FGD between pairs of cultures.
    4) Generated-vs-Generated FGD between pairs of cultures.
    5) Average FGD between pairs of speakers within each culture (Real data, optional).
    6) Average FGD between pairs of speakers within each culture (Generated data, optional).
    7) Intra-culture FGD by splitting samples within each culture (Real data).
    8) Intra-culture FGD by splitting samples within each culture (Generated data).

    Parameters
    ----------
    real_motion : torch.Tensor
        Real motion features, shape: (batch_size, time_dim, feature_dim)
    gen_motion : torch.Tensor
        Generated motion features, shape: (batch_size, time_dim, feature_dim)
    real_culture_labels : torch.Tensor
        Culture labels for real motion, shape: (batch_size,)
    gen_culture_labels : torch.Tensor
        Culture labels for generated motion, shape: (batch_size,)
    real_speaker_labels : torch.Tensor
        Speaker labels for real motion, shape: (batch_size,)
    gen_speaker_labels : torch.Tensor
        Speaker labels for generated motion, shape: (batch_size,)
    num_cultures : int
        Number of distinct culture labels (default=4).
    min_samples_for_fgd : int
         Minimum number of samples (e.g., sequences) required per group being compared
         to compute a meaningful FGD score (default=5). Affects intra-culture splits
         and checks before computing any FGD.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing all computed FGD scores. Values are float or None if
        computation was not possible (e.g., insufficient data, numerical instability).
        Keys: "fgd_overall", "fgd_per_culture", "fgd_real_culture_couples",
              "fgd_gen_culture_couples", "fgd_real_speaker_couples",
              "fgd_gen_speaker_couples", "fgd_real_intra_culture", "fgd_gen_intra_culture"
    """

    results = {} # Initialize results dictionary

    # Ensure inputs are on CPU for potential NumPy conversion
    # It's often better to keep computations on GPU if possible, but scipy requires CPU numpy arrays.
    # We will convert just before calling the numpy-based FD function.
    # device = real_motion.device # Keep track of original device if needed later

    ###########################################################################
    #  Helper function to get distribution parameters (memoized)
    ###########################################################################
    # Memoization cache to avoid recomputing stats for the same data subset
    dist_cache = {}

    def get_distribution(data_key: str, data_tensor: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if data_tensor.shape[0] < min_samples_for_fgd:
             warnings.warn(f"Subset '{data_key}' has only {data_tensor.shape[0]} samples, less than minimum {min_samples_for_fgd}. Skipping FGD calculation for this subset.")
             return None # Not enough samples for reliable mean/cov
        if data_key not in dist_cache:
            stats = _compute_activation_statistics_pt(data_tensor)
            if stats is None or stats[0] is None:
                 dist_cache[data_key] = None # Cache failure
            else:
                 dist_cache[data_key] = stats # Cache success (mu, cov)
        return dist_cache[data_key]

    # Helper function to compute FGD between two data tensors using cache
    def compute_fgd_between(key1: str, data1: torch.Tensor, key2: str, data2: torch.Tensor) -> Optional[float]:
        dist1 = get_distribution(key1, data1)
        dist2 = get_distribution(key2, data2)

        if dist1 is None or dist2 is None:
            return None # Cannot compute FGD if stats are missing

        mu1, cov1 = dist1
        mu2, cov2 = dist2

        # Convert to NumPy on CPU right before calculation
        mu1_np = mu1.detach().cpu().numpy()
        cov1_np = cov1.detach().cpu().numpy()
        mu2_np = mu2.detach().cpu().numpy()
        cov2_np = cov2.detach().cpu().numpy()

        return _calculate_frechet_distance_np(mu1_np, cov1_np, mu2_np, cov2_np)

    ###########################################################################
    # 1) Overall FGD (real vs. generated)
    ###########################################################################
    print("Calculating Overall FGD...")
    results["fgd_overall"] = compute_fgd_between("real_all", real_motion, "gen_all", gen_motion)
    print(f"Overall FGD: {results['fgd_overall']}")

    ###########################################################################
    # 2) FGD per culture (generated vs real)
    ###########################################################################
    print("Calculating Per-Culture FGD (Real vs Gen)...")
    fgd_per_culture = {}
    for c in range(num_cultures):
        real_idx = (real_culture_labels == c)
        gen_idx = (gen_culture_labels == c)

        real_subset = real_motion[real_idx]
        gen_subset = gen_motion[gen_idx]

        fgd_val = compute_fgd_between(f"real_c{c}", real_subset, f"gen_c{c}", gen_subset)
        fgd_per_culture[c] = fgd_val
        print(f"  Culture {c}: {fgd_val}")
    results["fgd_per_culture"] = fgd_per_culture

    ###########################################################################
    # 3) Real vs Real, for each pair of cultures
    ###########################################################################
    print("Calculating Inter-Culture FGD (Real vs Real)...")
    fgd_real_culture_couples = {}
    all_cultures = list(range(num_cultures))
    for (c1, c2) in combinations(all_cultures, 2):
        c1_idx = (real_culture_labels == c1)
        c2_idx = (real_culture_labels == c2)

        real_motion_c1 = real_motion[c1_idx]
        real_motion_c2 = real_motion[c2_idx]

        # Calculate distance once, store symmetrically
        fgd_value = compute_fgd_between(f"real_c{c1}", real_motion_c1, f"real_c{c2}", real_motion_c2)
        print(f"  Real Culture Pair ({c1}, {c2}): {fgd_value}")

        fgd_real_culture_couples[(c1, c2)] = fgd_value
        fgd_real_culture_couples[(c2, c1)] = fgd_value # Store symmetrically

    results["fgd_real_culture_couples"] = fgd_real_culture_couples

    ###########################################################################
    # 4) Generated vs Generated, for each pair of cultures
    ###########################################################################
    print("Calculating Inter-Culture FGD (Gen vs Gen)...")
    fgd_gen_culture_couples = {}
    for (c1, c2) in combinations(all_cultures, 2):
        c1_idx = (gen_culture_labels == c1)
        c2_idx = (gen_culture_labels == c2)

        gen_motion_c1 = gen_motion[c1_idx]
        gen_motion_c2 = gen_motion[c2_idx]

        # Calculate distance once, store symmetrically
        fgd_value = compute_fgd_between(f"gen_c{c1}", gen_motion_c1, f"gen_c{c2}", gen_motion_c2)
        print(f"  Gen Culture Pair ({c1}, {c2}): {fgd_value}")

        fgd_gen_culture_couples[(c1, c2)] = fgd_value
        fgd_gen_culture_couples[(c2, c1)] = fgd_value # Store symmetrically

    results["fgd_gen_culture_couples"] = fgd_gen_culture_couples

    ###########################################################################
    # 5) / 6) Inter-speaker FGDs within each culture (optional, expensive)
    ###########################################################################
    if compute_speaker_couples:
        print("Calculating Intra-Culture Inter-Speaker FGD (Real)...")
        fgd_real_speaker_couples = {}
        culture_speakers_real = {}
        for c in range(num_cultures):
            culture_idx = (real_culture_labels == c)
            if culture_idx.sum() > 0:
                culture_speakers_real[c] = torch.unique(real_speaker_labels[culture_idx]).tolist()
            else:
                culture_speakers_real[c] = []

        for c in range(num_cultures):
            spk_list = culture_speakers_real[c]
            if len(spk_list) < 2:
                print(f"  Real Culture {c}: Not enough speakers ({len(spk_list)})")
                fgd_real_speaker_couples[c] = None
                continue

            culture_fgd_results = []
            culture_idx = (real_culture_labels == c)

            for s1, s2 in combinations(spk_list, 2):
                s1_idx = (real_speaker_labels == s1) & culture_idx
                s2_idx = (real_speaker_labels == s2) & culture_idx

                motion_s1 = real_motion[s1_idx]
                motion_s2 = real_motion[s2_idx]

                fgd_val = compute_fgd_between(f"real_c{c}_s{s1}", motion_s1, f"real_c{c}_s{s2}", motion_s2)
                if fgd_val is not None:
                    culture_fgd_results.append(fgd_val)

            if not culture_fgd_results:
                print(f"  Real Culture {c}: No valid speaker pair FGDs computed.")
                fgd_real_speaker_couples[c] = None
            else:
                avg_fgd = np.mean(culture_fgd_results)
                fgd_real_speaker_couples[c] = float(avg_fgd)
                print(f"  Real Culture {c}: Avg Speaker Pair FGD = {avg_fgd:.4f} (from {len(culture_fgd_results)} pairs)")

        print("Calculating Intra-Culture Inter-Speaker FGD (Gen)...")
        fgd_gen_speaker_couples = {}
        culture_speakers_gen = {}
        for c in range(num_cultures):
            culture_idx = (gen_culture_labels == c)
            if culture_idx.sum() > 0:
                culture_speakers_gen[c] = torch.unique(gen_speaker_labels[culture_idx]).tolist()
            else:
                culture_speakers_gen[c] = []

        for c in range(num_cultures):
            spk_list = culture_speakers_gen[c]
            if len(spk_list) < 2:
                print(f"  Gen Culture {c}: Not enough speakers ({len(spk_list)})")
                fgd_gen_speaker_couples[c] = None
                continue

            culture_fgd_results = []
            culture_idx = (gen_culture_labels == c)

            for s1, s2 in combinations(spk_list, 2):
                s1_idx = (gen_speaker_labels == s1) & culture_idx
                s2_idx = (gen_speaker_labels == s2) & culture_idx

                motion_s1 = gen_motion[s1_idx]
                motion_s2 = gen_motion[s2_idx]

                fgd_val = compute_fgd_between(f"gen_c{c}_s{s1}", motion_s1, f"gen_c{c}_s{s2}", motion_s2)
                if fgd_val is not None:
                    culture_fgd_results.append(fgd_val)

            if not culture_fgd_results:
                print(f"  Gen Culture {c}: No valid speaker pair FGDs computed.")
                fgd_gen_speaker_couples[c] = None
            else:
                avg_fgd = np.mean(culture_fgd_results)
                fgd_gen_speaker_couples[c] = float(avg_fgd)
                print(f"  Gen Culture {c}: Avg Speaker Pair FGD = {avg_fgd:.4f} (from {len(culture_fgd_results)} pairs)")
    else:
        print("Skipping inter-speaker FGD within culture (compute_speaker_couples=False).")
        fgd_real_speaker_couples = {c: None for c in range(num_cultures)}
        fgd_gen_speaker_couples = {c: None for c in range(num_cultures)}

    results["fgd_real_speaker_couples"] = fgd_real_speaker_couples
    results["fgd_gen_speaker_couples"] = fgd_gen_speaker_couples


    ###########################################################################
    # Helper function for Intra-Culture split (moved here for clarity)
    ###########################################################################
    def _split_into_two_groups(x: torch.Tensor, labels: torch.Tensor, culture: int, min_samples_per_group: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """ Splits samples of a culture into two random non-overlapping groups. """
        culture_idx = (labels == culture)
        culture_samples = x[culture_idx]
        num_culture_samples = culture_samples.shape[0]

        # Need at least 2 * min_samples_per_group overall to potentially form two valid groups
        if num_culture_samples < 2 * min_samples_per_group:
            #print(f"Culture {culture}: Only {num_culture_samples} samples, need at least {2 * min_samples_per_group} for intra-culture split.")
            return None, None

        split_size = num_culture_samples // 2  # Integer division for roughly 50/50 split

        # Ensure both potential groups meet the minimum size requirement
        if split_size < min_samples_per_group or (num_culture_samples - split_size) < min_samples_per_group:
             #print(f"Culture {culture}: Cannot split {num_culture_samples} samples into two groups of at least {min_samples_per_group} each.")
             return None, None

        shuffled_indices = torch.randperm(num_culture_samples, device=x.device) # Keep indices on same device
        indices1 = shuffled_indices[:split_size]
        indices2 = shuffled_indices[split_size:]

        group1 = culture_samples[indices1]
        group2 = culture_samples[indices2]

        return group1, group2

    ###########################################################################
    # 7) Intra-Culture FGD for Real Samples
    ###########################################################################
    print("Calculating Intra-Culture FGD (Real Data Split)...")
    fgd_real_intra_culture = {}
    for c in range(num_cultures):
        group1, group2 = _split_into_two_groups(real_motion, real_culture_labels, c, min_samples_for_fgd)

        if group1 is None or group2 is None:
            fgd_real_intra_culture[c] = None
            print(f"  Real Culture {c}: Skipped (split failed or insufficient samples)")
            continue

        fgd_val = compute_fgd_between(f"real_c{c}_split1", group1, f"real_c{c}_split2", group2)
        fgd_real_intra_culture[c] = fgd_val
        print(f"  Real Culture {c}: {fgd_val}")

    results["fgd_real_intra_culture"] = fgd_real_intra_culture

    ###########################################################################
    # 8) Intra-Culture FGD for Generated Samples
    ###########################################################################
    print("Calculating Intra-Culture FGD (Generated Data Split)...")
    fgd_gen_intra_culture = {}
    for c in range(num_cultures):
        group1, group2 = _split_into_two_groups(gen_motion, gen_culture_labels, c, min_samples_for_fgd)

        if group1 is None or group2 is None:
            fgd_gen_intra_culture[c] = None
            print(f"  Gen Culture {c}: Skipped (split failed or insufficient samples)")
            continue

        fgd_val = compute_fgd_between(f"gen_c{c}_split1", group1, f"gen_c{c}_split2", group2)
        fgd_gen_intra_culture[c] = fgd_val
        print(f"  Gen Culture {c}: {fgd_val}")

    results["fgd_gen_intra_culture"] = fgd_gen_intra_culture

    # Clear cache if desired after run
    dist_cache.clear()

    print("FGD Calculation Complete.")
    return results

def compute_diversity(
        real_motion: torch.Tensor,
        generated_motion: torch.Tensor,
        real_labels: torch.Tensor,
        gen_labels: torch.Tensor,
        diversity_times: int
) -> Dict[str, Any]:
    """
    Computes diversity (multimodality) measures for real and generated motions.

    Specifically, it calculates:
        1. Overall diversity of real data.
        2. Overall diversity of generated data.
        3. Diversity for each culture separately in real data.
        4. Diversity for each culture separately in generated data.

    Parameters
    ----------
    real_motion : torch.Tensor
        Real motion embeddings, shape (batch_size, time_dim, 512).
    generated_motion : torch.Tensor
        Generated motion embeddings, shape (batch_size, time_dim, 512).
    real_labels : torch.Tensor
        Culture labels for each sample, shape (batch_size,).
    gen_labels : torch.Tensor
        Culture labels for each sample, shape (batch_size,).
    diversity_times : int
        Number of random pairs to sample for diversity calculation.

    Returns
    -------
    Dict[str, Any]
        {
            "real_diversity": <float>,
            "generated_diversity": <float>,
            "real_diversity_by_culture": { culture_id: <float> or None, ... },
            "generated_diversity_by_culture": { culture_id: <float> or None, ... }
        }
    """

    def calculate_diversity(activation: torch.Tensor, diversity_times: int, replace: bool = False) -> float:
        """
        Calculates the average Euclidean distance between randomly sampled pairs of activations.

        Parameters
        ----------
        activation : torch.Tensor
            Flattened motion embeddings, shape (num_samples, dim).
        diversity_times : int
            Number of random pairs to sample.
        replace : bool, optional
            Whether to sample with replacement (for bootstrapping), by default False.

        Returns
        -------
        float
            Average pairwise Euclidean distance.
        """
        assert activation.dim() == 2, "Activation tensor must be 2-dimensional (num_samples, dim)."
        num_samples = activation.size(0)
        assert num_samples >= 2, (
            f"Number of samples ({num_samples}) must be at least 2 to compute diversity."
        )
        if not replace:
            assert num_samples > diversity_times, (
                f"Number of samples ({num_samples}) must be greater than diversity_times ({diversity_times}) when replace=False."
            )

        # Randomly sample indices
        first_indices = np.random.choice(num_samples, diversity_times, replace=replace)
        second_indices = np.random.choice(num_samples, diversity_times, replace=replace)

        # Compute differences and Euclidean distances
        diffs = activation[first_indices] - activation[second_indices]  # Shape: (diversity_times, dim)
        dists = torch.norm(diffs, dim=1).cpu().numpy()  # Shape: (diversity_times,)

        return dists.mean()

    # Flatten motion tensors: (batch_size, time_dim, 512) -> (batch_size, time_dim * 512)
    real_motion_flat = real_motion.view(real_motion.size(0), -1)
    generated_motion_flat = generated_motion.view(generated_motion.size(0), -1)

    # Initialize result dictionary
    results = OrderedDict()

    # 1. Overall diversity of real data
    if real_motion_flat.size(0) > diversity_times:
        real_overall_diversity = calculate_diversity(real_motion_flat, diversity_times)
    else:
        real_overall_diversity = None  # Not enough samples
    results["real_diversity"] = real_overall_diversity

    # 2. Overall diversity of generated data
    if generated_motion_flat.size(0) > diversity_times:
        generated_overall_diversity = calculate_diversity(generated_motion_flat, diversity_times)
    else:
        generated_overall_diversity = None  # Not enough samples
    results["gen_diversity"] = generated_overall_diversity

    # 3. Diversity per culture in real data
    unique_cultures_real = real_labels.unique()
    real_diversity_by_culture = {}

    desired_sample_size = diversity_times // 4  # Assuming 4 cultures

    for c in unique_cultures_real:
        c = c.item()
        mask = (real_labels == c)
        subset = real_motion_flat[mask]
        num_subset = subset.size(0)

        if num_subset >= desired_sample_size:
            replace = False
            effective_diversity_times = diversity_times // 4
        elif num_subset >= 2:
            replace = True
            # To allow diversity_times//4 pairs, we need at least diversity_times//4 samples when replace=True
            # However, with replacement, we can always sample the required number of pairs
            effective_diversity_times = diversity_times // 4
        else:
            # Not enough samples to compute diversity
            real_diversity_by_culture[c] = None
            continue

        try:
            diversity = calculate_diversity(subset, effective_diversity_times, replace=replace)
        except AssertionError as e:
            print(f"Warning: {e} for real culture {c}. Setting diversity to None.")
            diversity = None

        real_diversity_by_culture[c] = diversity
    results["real_diversity_by_culture"] = real_diversity_by_culture

    # 4. Diversity per culture in generated data
    unique_cultures_gen = gen_labels.unique()
    generated_diversity_by_culture = {}

    for c in unique_cultures_gen:
        c = c.item()
        mask = (gen_labels == c)
        subset = generated_motion_flat[mask]
        num_subset = subset.size(0)

        if num_subset >= desired_sample_size:
            replace = False
            effective_diversity_times = diversity_times // 4
        elif num_subset >= 2:
            replace = True
            effective_diversity_times = diversity_times // 4
        else:
            # Not enough samples to compute diversity
            generated_diversity_by_culture[c] = None
            continue

        try:
            diversity = calculate_diversity(subset, effective_diversity_times, replace=replace)
        except AssertionError as e:
            print(f"Warning: {e} for generated culture {c}. Setting diversity to None.")
            diversity = None

        generated_diversity_by_culture[c] = diversity
    results["gen_diversity_by_culture"] = generated_diversity_by_culture

    return results

def compute_multimodality_random(
    real_motion: torch.Tensor,
    generated_motion: torch.Tensor,
    real_labels: torch.Tensor,
    num_random_pairs: int = 10000
) -> Dict[str, Any]:
    """
    Compute multimodality measures (average random pairwise distance) using a
    random-sampling approach similar to the old snippet.

    1) Overall multimodality of real data
    2) Overall multimodality of generated data
    3) Multimodality per culture (real data)
    4) Multimodality per culture (generated data)

    Parameters
    ----------
    real_motion : torch.Tensor
        Shape (batch_size, time_dim, feature_dim=512).
        All the real gesture embeddings or features.
    generated_motion : torch.Tensor
        Shape (batch_size, time_dim, feature_dim=512).
        All the generated gesture embeddings or features.
        (We assume the same batch_size or a corresponding distribution,
         but that may vary in practice.)
    real_labels : torch.Tensor
        1D tensor of shape (batch_size,).
        Culture labels for each motion in real_motion *and*
        for the corresponding sample in generated_motion if you
        reused the same labels.
        (If you have separate labels for generated data, adapt accordingly.)
    num_random_pairs : int
        How many random pairs to sample (max).
        If the batch is smaller, we’ll reduce the number of pairs accordingly.

    Returns
    -------
    Dict[str, Any]
        {
          "real_multimodality": <float>,
          "generated_multimodality": <float>,
          "real_multimodality_by_culture": { culture_id: <float>, ... },
          "generated_multimodality_by_culture": { culture_id: <float>, ... }
        }
    """

    def _random_pairwise_distance(x: torch.Tensor, n_pairs: int) -> float:
        """
        Flatten each motion from (time_dim, feature_dim) => 1D,
        then sample random pairs among the batch dimension to compute
        average Euclidean distance (L2 norm).

        x shape: (B, T, 512).
        We flatten to (B, T*512). Then pick random pairs from [0..B-1].
        """
        B = x.shape[0]
        # If not enough samples or only 1 sample, distance is zero (or ill-defined).
        if B < 2:
            return 0.0

        # Flatten time + feature_dim => single vector
        x_flat = x.reshape(B, -1)  # shape: (B, T*512)

        # We can't sample more pairs than the total number of possible pairs.
        # Max distinct pairs = B*(B-1)/2. But here we do random selection, so we set an upper bound.
        n_pairs = min(n_pairs, B)  # Typically, you might do min(n_pairs, B*(B-1)//2), or similar.

        # Randomly choose indices for pairs
        idx1 = np.random.choice(B, n_pairs, replace=False)
        idx2 = np.random.choice(B, n_pairs, replace=False)

        # Distances between chosen pairs
        diffs = x_flat[idx1] - x_flat[idx2]
        dists = torch.norm(diffs, dim=1)  # L2 norm for each pair => shape (n_pairs,)

        return dists.mean().item()

    # 1) Overall real multimodality
    real_multimodality = _random_pairwise_distance(real_motion, num_random_pairs)

    # 2) Overall generated multimodality
    generated_multimodality = _random_pairwise_distance(generated_motion, num_random_pairs)

    # 3) Multimodality per culture (real data)
    unique_cultures = real_labels.unique()
    real_multimodality_by_culture = {}
    for c in unique_cultures:
        idx_c = (real_labels == c)
        # Subset the real motions for culture c
        real_motions_c = real_motion[idx_c]
        mm_c = _random_pairwise_distance(real_motions_c, num_random_pairs)
        real_multimodality_by_culture[int(c.item())] = mm_c

    # 4) Multimodality per culture (generated data)
    #    If we have the same label structure for generated data,
    #    re-using real_labels might be a placeholder. Adjust if needed.
    generated_multimodality_by_culture = {}
    for c in unique_cultures:
        idx_c = (real_labels == c)  # or gen_labels == c if you have a separate label tensor
        gen_motions_c = generated_motion[idx_c]
        mm_c = _random_pairwise_distance(gen_motions_c, num_random_pairs)
        generated_multimodality_by_culture[int(c.item())] = mm_c

    return {
        "real_multimodality": real_multimodality,
        "generated_multimodality": generated_multimodality,
        "real_multimodality_by_culture": real_multimodality_by_culture,
        "generated_multimodality_by_culture": generated_multimodality_by_culture
    }

################################################################################
# 4) Multimodality of generated gestures and real gestures
################################################################################
def compute_multimodality(
        real_motion: torch.Tensor,
        generated_motion: torch.Tensor,
        real_labels: torch.Tensor
) -> Dict[str, Any]:
    """
    Computes a measure of multimodality for real and generated gestures.
    'Multimodality' can be assessed in many ways, e.g., measuring
    the diversity of samples across the batch or within each label.

    Example approach:
      - For each label, compute the average pairwise distance among gestures
        in that label. If the average distance is large, it suggests higher
        multimodality.

    Parameters
    ----------
    real_motion : torch.Tensor
        Shape (batch_size, time_dim, 512)
    generated_motion : torch.Tensor
        Shape (batch_size, time_dim, 512)
    real_labels : torch.Tensor
        Culture labels (batch_size,)

    Returns
    -------
    Dict[str, Any]
        {
            "real_multimodality": <float>,
            "generated_multimodality": <float>,
            "details_per_label": {
               label0: {"real": <float>, "generated": <float>}, ...
            }
        }
    """

    def avg_pairwise_distance(x: torch.Tensor) -> float:
        """
        A simple function to compute the average pairwise distance
        for a batch of motions x of shape (N, time_dim, 512).
        We'll flatten time_dim for simplicity: (N, time_dim*512).
        """
        if x.shape[0] < 2:
            return 0.0

        x_flat = x.reshape(x.shape[0], -1)  # shape: (N, time_dim*512)
        # Compute pairwise distances (N, N)
        # For efficiency, you could use pdist or other approaches.
        dist_matrix = torch.cdist(x_flat, x_flat, p=2)  # (N, N)
        # Return average of the upper triangle (excluding diagonal)
        n = x.shape[0]
        triu_indices = torch.triu_indices(n, n, offset=1)
        distances = dist_matrix[triu_indices[0], triu_indices[1]]
        return distances.mean().item()

    # Overall real multimodality
    real_multimodality = avg_pairwise_distance(real_motion)
    # Overall generated multimodality
    generated_multimodality = avg_pairwise_distance(generated_motion)

    # Per-label details
    unique_labels = real_labels.unique()
    details_per_label = {}
    for lbl in unique_labels:
        real_idx = (real_labels == lbl)
        # If your generated dataset has a corresponding label tensor, you’d use that.
        # Otherwise, we assume real_labels also describe which generated belongs to which label.
        gen_idx = (real_labels == lbl)

        real_m = real_motion[real_idx]
        gen_m = generated_motion[gen_idx]

        details_per_label[int(lbl.item())] = {
            "real": avg_pairwise_distance(real_m),
            "generated": avg_pairwise_distance(gen_m)
        }

    return {
        "real_multimodality": real_multimodality,
        "generated_multimodality": generated_multimodality,
        "details_per_label": details_per_label
    }

def calculate_velocity(poses: np.ndarray) -> np.ndarray:
    """Calculates frame-to-frame velocity for all joints."""
    if poses.shape[0] < 2:
        return np.zeros((0, poses.shape[1], poses.shape[2]), dtype=poses.dtype)
    # Ensure correct axis for diff if shape is (n_frames, n_joints, 3)
    return np.diff(poses, axis=0)


def _to_numpy_1d(x: Union[np.ndarray, torch.Tensor, List[float]]) -> np.ndarray:
    """Convert input to a contiguous 1D numpy array."""
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    if arr.ndim != 1:
        arr = np.reshape(arr, (-1,))
    return np.ascontiguousarray(arr)


def _to_pose_numpy_4d(motion: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Convert motion into (B, T, J, 3):
      - accepts (B, T, J, 3)
      - accepts (B, T, F) with F % 3 == 0
    """
    if torch.is_tensor(motion):
        motion = motion.detach().cpu().numpy()
    motion_np = np.asarray(motion)
    if motion_np.ndim == 4:
        if motion_np.shape[-1] != 3:
            raise ValueError(f"Expected last dim=3 for 4D motion, got shape={motion_np.shape}")
        return np.ascontiguousarray(motion_np)
    if motion_np.ndim == 3:
        bsz, tdim, feat = motion_np.shape
        if feat % 3 != 0:
            raise ValueError(f"Feature dim must be divisible by 3, got shape={motion_np.shape}")
        return np.ascontiguousarray(motion_np.reshape(bsz, tdim, feat // 3, 3))
    raise ValueError(f"Unsupported motion shape {motion_np.shape}; expected 3D or 4D tensor.")




def find_signal_peaks(signal: np.ndarray,
                      timestamps: np.ndarray,
                      height: Optional[float] = None,
                      distance: Optional[int] = None) -> np.ndarray:
    """Finds peaks in a 1D signal and returns their timestamps."""
    signal = _to_numpy_1d(signal)
    timestamps = _to_numpy_1d(timestamps)
    if len(signal) == 0 or len(timestamps) != len(signal):
        warnings.warn(f"Signal length ({len(signal)}) != Timestamps length ({len(timestamps)}). Returning empty peaks.")
        return np.array([], dtype=np.float64)
    if len(signal) < 2:  # Need at least 2 points for diff
        return np.array([], dtype=np.float64)

    # Ensure timestamps has length matching signal AFTER potential diff (for velocity)
    if len(timestamps) == len(signal) + 1:
        # Common case if timestamps included t=0 but velocity starts at t=1
        timestamps = timestamps[1:]
        if len(timestamps) != len(signal):  # Check again after adjustment
            warnings.warn(
                f"Adjusted Timestamps length ({len(timestamps)}) still != Signal length ({len(signal)}). Returning empty peaks.")
            return np.array([], dtype=np.float64)

    if distance is None:
        mean_diff = np.mean(np.diff(timestamps))
        if mean_diff > 0 and not np.isnan(mean_diff):
            # Calculate distance in samples based on 100ms desired separation
            distance = max(1, int(0.1 / mean_diff))
        else:
            distance = 1  # Default distance if timestamps are weird
        # print(f"Auto distance: {distance} samples") # Debugging

    try:
        peaks_indices, _ = find_peaks(signal, height=height, distance=distance)
        #print("HERE 2", peaks_indices)
        # Ensure indices are within bounds for timestamps
        valid_indices = peaks_indices[peaks_indices < len(timestamps)]
        return timestamps[valid_indices]
    except Exception as e:
        warnings.warn(f"Error during find_peaks: {e}")
        return np.array([], dtype=np.float64)


def compute_beat_align_scores(  # Renamed from compute_beat_align_score for clarity
        poses: np.ndarray,  # Shape (n_frames, n_joints, 3)
        pose_timestamps: np.ndarray,  # Shape (n_frames,)
        onset_strength: np.ndarray,  # Shape (n_onset_samples,)
        onset_timestamps: np.ndarray,  # Shape (n_onset_samples,)
        kinematic_joint_indices: List[int],  # Indices of joints to track (e.g., wrists)
        sigma: float,  # Paper sigma in 60-FPS frame units (Ai Choreographer uses 3)
        velocity_peak_height_threshold: Optional[float] = None,  # kept for API compatibility
        onset_peak_height_threshold: Optional[float] = None  # kept for API compatibility
) -> Optional[float]:  # Return Optional[float] to handle errors
    """
    Paper-style Beat Alignment Score (BAS).

    - Motion beats: local minima of kinematic velocity.
    - Audio beats: librosa beat tracking on onset envelope.
    - Sigma is interpreted in 60-FPS frame units and converted to seconds internally.
    """
    poses = np.asarray(poses)
    pose_timestamps = _to_numpy_1d(pose_timestamps)
    onset_strength = _to_numpy_1d(onset_strength)
    onset_timestamps = _to_numpy_1d(onset_timestamps)

    # --- Input Checks ---
    if poses.ndim != 3 or poses.shape[0] < 2 or poses.shape[2] != 3:
        warnings.warn(f"Invalid poses shape: {poses.shape}. Expected (n_frames>=2, n_joints, 3).")
        return None
    if pose_timestamps.ndim != 1 or pose_timestamps.shape[0] != poses.shape[0]:
        warnings.warn(f"Invalid pose_timestamps shape: {pose_timestamps.shape}. Expected ({poses.shape[0]},).")
        return None
    if onset_strength.ndim != 1 or len(onset_strength) == 0:
        warnings.warn(f"Invalid onset_strength shape: {onset_strength.shape}. Expected (n_onset_samples>0,).")
        return None
    if onset_timestamps.ndim != 1 or onset_timestamps.shape[0] != onset_strength.shape[0]:
        warnings.warn(
            f"Invalid onset_timestamps shape: {onset_timestamps.shape}. Expected ({onset_strength.shape[0]},).")
        return None
    if sigma <= 0:
        warnings.warn("sigma must be positive.")
        return None
    if not kinematic_joint_indices:
        warnings.warn("No kinematic joint indices provided for beat alignment.")
        return None
    max_joint_index = max(kinematic_joint_indices)
    if max_joint_index >= poses.shape[1]:
        warnings.warn(f"Max kinematic joint index {max_joint_index} out of bounds for {poses.shape[1]} joints.")
        return None

    # Convert paper sigma (in 60-FPS frame units) to seconds.
    sigma_seconds = float(sigma) / BAS_SIGMA_REFERENCE_FPS
    if sigma_seconds <= 0:
        warnings.warn("sigma_seconds must be positive.")
        return None

    # --- Find Gesture (Kinematic) Beats (G) ---
    velocity = calculate_velocity(poses)  # Shape (n_frames-1, n_joints, 3)
    if velocity.shape[0] == 0:
        return 0.0
    velocity_timestamps = pose_timestamps[1:]  # Timestamps for velocity frames

    try:
        selected_velocity = velocity[:, kinematic_joint_indices, :]
        # Calculate magnitude per joint, then average across selected joints
        velocity_magnitude_per_joint = np.linalg.norm(selected_velocity,
                                                      axis=-1)  # Shape (n_frames-1, n_kinematic_joints)
        mean_velocity_magnitude = np.mean(velocity_magnitude_per_joint, axis=1)  # Shape (n_frames-1,)
    except Exception as e:
        warnings.warn(f"Error calculating velocity magnitude: {e}")
        return None

    # Add check: velocity_timestamps length must match mean_velocity_magnitude length
    if len(velocity_timestamps) != len(mean_velocity_magnitude):
        warnings.warn(
            f"Velocity timestamps length {len(velocity_timestamps)} != velocity magnitude length {len(mean_velocity_magnitude)}.")
        return None

    # BAS (Ai Choreographer): motion beats are local minima of kinetic velocity.
    if mean_velocity_magnitude.shape[0] < 3:
        return 0.0
    gesture_beat_indices = argrelextrema(mean_velocity_magnitude, np.less)[0]
    if gesture_beat_indices.size == 0:
        return 0.0
    gesture_beat_indices = gesture_beat_indices[
        (gesture_beat_indices >= 0) & (gesture_beat_indices < velocity_timestamps.shape[0])
    ]
    gesture_beat_times = velocity_timestamps[gesture_beat_indices]

    # --- Find Audio Beats (A) ---
    # Strict paper path: beat tracking from onset envelope.
    global _LIBROSA_BEAT_TRACK_ENABLED, _LIBROSA_BEAT_TRACK_DISABLE_REASON
    if not _LIBROSA_BEAT_TRACK_ENABLED or onset_timestamps.shape[0] < 2:
        if _LIBROSA_BEAT_TRACK_DISABLE_REASON:
            warnings.warn(
                f"Beat tracking disabled for BAS ({_LIBROSA_BEAT_TRACK_DISABLE_REASON}). "
                "Paper BAS requires librosa beat tracking."
            )
        return None

    try:
        onset_dt = float(np.mean(np.diff(onset_timestamps)))
        if not np.isfinite(onset_dt) or onset_dt <= 0:
            warnings.warn("Invalid onset timestamps spacing while computing BAS.")
            return None
        audio_sr = float(1.0 / onset_dt)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if NumbaPendingDeprecationWarning is not None:
                warnings.filterwarnings("ignore", category=NumbaPendingDeprecationWarning)
            _, beat_frames = librosa.beat.beat_track(
                onset_envelope=onset_strength.astype(np.float32),
                sr=audio_sr,
                hop_length=1,
            )
        beat_frames = np.asarray(beat_frames, dtype=np.int64)
        if beat_frames.size == 0:
            return 0.0
        beat_frames = beat_frames[(beat_frames >= 0) & (beat_frames < onset_timestamps.shape[0])]
        audio_beat_times = onset_timestamps[beat_frames]
    except Exception as exc:
        _LIBROSA_BEAT_TRACK_ENABLED = False
        _LIBROSA_BEAT_TRACK_DISABLE_REASON = f"{type(exc).__name__}: {exc}"
        warnings.warn(f"Beat tracking failed for BAS: {_LIBROSA_BEAT_TRACK_DISABLE_REASON}")
        return None

    # --- Calculate Gaussian Beat Alignment ---
    num_gesture_beats = len(gesture_beat_times)  # |G|
    if num_gesture_beats == 0:
        return 0.0

    if len(audio_beat_times) == 0:
        return 0.0

    total_score = 0.0
    two_sigma_squared = 2 * (sigma_seconds ** 2)

    # Calculate alignment score for each gesture beat
    try:
        diff_matrix = gesture_beat_times[:, np.newaxis] - audio_beat_times[np.newaxis, :]
        sq_dist_matrix = diff_matrix ** 2
        min_sq_dist_per_gesture = np.min(sq_dist_matrix, axis=1)
        scores_per_gesture = np.exp(-min_sq_dist_per_gesture / two_sigma_squared)
        total_score = float(np.sum(scores_per_gesture))

    except Exception as e:
        warnings.warn(f"Error during beat alignment calculation: {e}")
        return None  # Indicate failure

    # Average the scores over all gesture beats
    final_beat_align_score = float(total_score / float(num_gesture_beats))
    return float(final_beat_align_score)


def compute_srgr_score(
        generated_poses: np.ndarray,  # Shape (n_frames, n_joints, 3) -> \hat{p}
        ground_truth_poses: np.ndarray,  # Shape (n_frames, n_joints, 3) -> p
        delta: float,  # Euclidean distance threshold
        lambda_scale: float = 1.0  # Scaling factor lambda
) -> Optional[float]:  # Return Optional[float] to handle errors
    """
    Calculates the Geometric Reconstruction Accuracy (SRGR/DSRGR).
    Returns float score or None if calculation fails.
    """
    if generated_poses.shape != ground_truth_poses.shape:
        warnings.warn(f"SRGR Error: Gen shape {generated_poses.shape} != GT shape {ground_truth_poses.shape}")
        return None
    if generated_poses.ndim != 3 or generated_poses.shape[2] != 3:
        warnings.warn(f"SRGR Error: Invalid poses shape {generated_poses.shape}. Expected (n_frames, n_joints, 3).")
        return None

    n_frames, n_joints, n_coords = generated_poses.shape
    if n_frames == 0 or n_joints == 0:
        return 0.0  # Score is 0 if no data

    try:
        # BEAT SRGR: indicator over Euclidean distance threshold per frame and joint.
        distances = np.linalg.norm(ground_truth_poses - generated_poses, axis=-1)

        # Apply the indicator function
        indicator = (distances < float(delta)).astype(float)

        # Sum over all frames and joints
        total_correct_points = np.sum(indicator)

        # Normalize and scale
        normalized_score = total_correct_points / (n_frames * n_joints)
        dsrgr_score = float(lambda_scale) * normalized_score
        return float(dsrgr_score)

    except Exception as e:
        warnings.warn(f"Error calculating SRGR score: {e}")
        return None


# --- End of assumed helper functions ---


def compute_beat_alignment_analysis(
        real_motion: torch.Tensor,  # Shape: (batch, time, n_joints, n_coordinates)
        gen_motion: torch.Tensor,  # Shape: (batch, time, n_joints, n_coordinates)
        real_culture_labels: torch.Tensor,  # Shape: (batch,)
        gen_culture_labels: torch.Tensor,  # Shape: (batch,)
        pose_timestamps: np.ndarray,  # Shape: (batch, time) or (time,) - Timestamps for motion frames
        # --- Audio data (assuming one set per motion sample in batch) ---
        real_audio_onset_strength: Union[np.ndarray, List[np.ndarray]],
        real_audio_onset_timestamps: Union[np.ndarray, List[np.ndarray]],
        gen_audio_onset_strength: Union[np.ndarray, List[np.ndarray]],
        gen_audio_onset_timestamps: Union[np.ndarray, List[np.ndarray]],
        # --- Metric Parameters ---
        num_cultures: int = 4,
        beat_align_sigma: float = 3.0,
        beat_align_kinematic_joints: List[int] = [0, 1, 2],  # Example: use first 3 joints
        velocity_peak_height_threshold: Optional[float] = None,
        onset_peak_height_threshold: Optional[float] = None,
        # --- Other parameters ---
        min_samples_per_culture: int = 3  # Min samples needed per culture
) -> Dict[str, Any]:
    """
    Computes Beat Alignment scores overall and per culture for real and generated data.

    Parameters are similar to compute_combined_scores, focusing on Beat Alignment inputs.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing computed scores:
        - "beat_align_real_overall": float or None
        - "beat_align_gen_overall": float or None
        - "beat_align_real_per_culture": Dict[int, float or None]
        - "beat_align_gen_per_culture": Dict[int, float or None]
    """
    results = {}
    real_motion_np = _to_pose_numpy_4d(real_motion)
    gen_motion_np = _to_pose_numpy_4d(gen_motion)
    batch_size, time_dim, n_joints, n_coordinates = real_motion_np.shape

    if gen_motion_np.shape != real_motion_np.shape:
        # Warn but proceed, BeatAlign doesn't strictly require matching shapes unless comparing directly
        warnings.warn(
            f"Real motion shape {real_motion_np.shape} differs from generated motion shape {gen_motion_np.shape}."
        )
    if real_culture_labels.shape[0] != batch_size or gen_culture_labels.shape[0] != batch_size:
        raise ValueError("Label tensors must match batch size.")

    real_culture_labels_np = _to_numpy_1d(real_culture_labels).astype(np.int64)
    gen_culture_labels_np = _to_numpy_1d(gen_culture_labels).astype(np.int64)

    pose_timestamps_np_input = pose_timestamps.detach().cpu().numpy() if torch.is_tensor(pose_timestamps) else np.asarray(pose_timestamps)
    if pose_timestamps_np_input.ndim == 1 and len(pose_timestamps_np_input) == time_dim:
        pose_timestamps_np = np.tile(pose_timestamps_np_input, (batch_size, 1))
    elif pose_timestamps_np_input.shape == (batch_size, time_dim):
        pose_timestamps_np = pose_timestamps_np_input
    else:
        raise ValueError(
            f"pose_timestamps shape {pose_timestamps_np_input.shape} invalid for batch size {batch_size} and time dim {time_dim}."
        )

    # --- Helper to calculate average score over a batch/subset ---
    def _calculate_average_beat_align(poses_batch, timestamps_batch, onset_strength_batch, onset_timestamps_batch,
                                      **kwargs):
        scores = []
        num_samples = poses_batch.shape[0]
        #print("IFOAHFO",poses_batch.shape, np.array(onset_strength_batch).shape, np.array(onset_timestamps_batch).shape, np.array(timestamps_batch).shape)
        if num_samples == 0:
            return None

        for i in range(num_samples):
            # Handle list vs array for audio data
            current_onset_strength = onset_strength_batch[i] if isinstance(onset_strength_batch, list) else \
            onset_strength_batch[i]
            current_onset_timestamps = onset_timestamps_batch[i] if isinstance(onset_timestamps_batch, list) else \
            onset_timestamps_batch[i]
            current_onset_strength = _to_numpy_1d(current_onset_strength)
            current_onset_timestamps = _to_numpy_1d(current_onset_timestamps)

            # Ensure timestamps_batch[i] is 1D
            current_timestamps = _to_numpy_1d(timestamps_batch[i])
            if current_timestamps.ndim != 1:
                warnings.warn(f"Sample {i} has non-1D timestamps: shape {current_timestamps.shape}. Skipping.")
                continue  # Skip this sample

            score = compute_beat_align_scores(
                poses=poses_batch[i],
                pose_timestamps=current_timestamps,
                onset_strength=current_onset_strength,
                onset_timestamps=current_onset_timestamps,
                **kwargs
            )
            #print("HERE 3", score)
            if score is not None:
                scores.append(score)
            # else: # Optional: print warning if score is None
            #    warnings.warn(f"Beat align calculation failed for sample {i}")

        return float(np.mean(scores)) if scores else None

    # --- Overall Calculation ---
    print("Calculating Overall Beat Alignment Scores...")
    common_kwargs = {
        "kinematic_joint_indices": beat_align_kinematic_joints,
        "sigma": beat_align_sigma,
        "velocity_peak_height_threshold": velocity_peak_height_threshold,
        "onset_peak_height_threshold": onset_peak_height_threshold
    }
    results["beat_align_real_overall"] = _calculate_average_beat_align(
        real_motion_np, pose_timestamps_np, real_audio_onset_strength, real_audio_onset_timestamps, **common_kwargs
    )
    print(f"Overall Beat Alignment (Real): {results['beat_align_real_overall']}")

    results["beat_align_gen_overall"] = _calculate_average_beat_align(
        gen_motion_np, pose_timestamps_np, gen_audio_onset_strength, gen_audio_onset_timestamps, **common_kwargs
    )
    print(f"Overall Beat Alignment (Generated): {results['beat_align_gen_overall']}")

    # --- Per-Culture Calculation ---
    print("\nCalculating Per-Culture Beat Alignment Scores...")
    beat_align_real_per_culture = {}
    beat_align_gen_per_culture = {}

    for c in range(num_cultures):
        print(f"--- Culture {c} ---")
        real_culture_idx_np = np.where(real_culture_labels_np == c)[0]
        gen_culture_idx_np = np.where(gen_culture_labels_np == c)[0]

        # Calculate for Real data
        if len(real_culture_idx_np) >= min_samples_per_culture:
            real_motion_subset = real_motion_np[real_culture_idx_np]
            timestamps_subset = pose_timestamps_np[real_culture_idx_np]
            if isinstance(real_audio_onset_strength, list):
                audio_strength_subset = [real_audio_onset_strength[i] for i in real_culture_idx_np]
                audio_timestamps_subset = [real_audio_onset_timestamps[i] for i in real_culture_idx_np]
            else:
                audio_strength_subset = real_audio_onset_strength[real_culture_idx_np]
                audio_timestamps_subset = real_audio_onset_timestamps[real_culture_idx_np]

            score = _calculate_average_beat_align(
                real_motion_subset, timestamps_subset, audio_strength_subset, audio_timestamps_subset, **common_kwargs
            )
            beat_align_real_per_culture[c] = score
            print(f"  Real Samples: {len(real_culture_idx_np)}, Score: {score}")
        else:
            print(f"  Real Samples: {len(real_culture_idx_np)} < min {min_samples_per_culture}. Skipping.")
            beat_align_real_per_culture[c] = None

        # Calculate for Generated data
        if len(gen_culture_idx_np) >= min_samples_per_culture:
            gen_motion_subset = gen_motion_np[gen_culture_idx_np]
            timestamps_subset = pose_timestamps_np[gen_culture_idx_np]  # Assume same timestamps apply
            if isinstance(gen_audio_onset_strength, list):
                audio_strength_subset = [gen_audio_onset_strength[i] for i in gen_culture_idx_np]
                audio_timestamps_subset = [gen_audio_onset_timestamps[i] for i in gen_culture_idx_np]
            else:
                audio_strength_subset = gen_audio_onset_strength[gen_culture_idx_np]
                audio_timestamps_subset = gen_audio_onset_timestamps[gen_culture_idx_np]

            score = _calculate_average_beat_align(
                gen_motion_subset, timestamps_subset, audio_strength_subset, audio_timestamps_subset, **common_kwargs
            )
            beat_align_gen_per_culture[c] = score
            print(f"  Gen Samples: {len(gen_culture_idx_np)}, Score: {score}")
        else:
            print(f"  Gen Samples: {len(gen_culture_idx_np)} < min {min_samples_per_culture}. Skipping.")
            beat_align_gen_per_culture[c] = None

    results["beat_align_real_per_culture"] = beat_align_real_per_culture
    results["beat_align_gen_per_culture"] = beat_align_gen_per_culture

    print("\nBeat Alignment Analysis Complete.")
    return results


def compute_srgr_analysis(
        real_motion: torch.Tensor,  # Shape: (batch, time, n_joints, n_coordinates) - Ground Truth
        gen_motion: torch.Tensor,  # Shape: (batch, time, n_joints, n_coordinates) - Generated, MUST correspond to real_motion
        real_culture_labels: torch.Tensor,  # Shape: (batch,)
        gen_culture_labels: torch.Tensor,  # Shape: (batch,) - MUST correspond to real_culture_labels
        # --- Metric Parameters ---
        num_cultures: int = 4,
        srgr_delta: float = 0.05,  # PCK Euclidean distance threshold
        srgr_lambda: float = 1.0,  # Scaling factor
        # --- Other parameters ---
        min_samples_per_culture: int = 3  # Min samples needed per culture
) -> Dict[str, Any]:
    """
    Computes SRGR (Geometric Reconstruction Accuracy) scores overall and per culture.

    Crucially assumes that `real_motion` and `gen_motion` samples are paired correctly
    and that `real_culture_labels` and `gen_culture_labels` are identical.

    Parameters
    ----------
    real_motion : torch.Tensor
        Real motion features (Ground Truth), shape: (batch, time, features).
    gen_motion : torch.Tensor
        Generated motion features, shape: (batch, time, features). Must correspond to real_motion.
    real_culture_labels : torch.Tensor
        Culture labels for real motion, shape: (batch,).
    gen_culture_labels : torch.Tensor
        Culture labels for generated motion, shape: (batch,). Must align with real_culture_labels.
    num_cultures : int
        Number of distinct culture labels.
    srgr_delta : float
        Euclidean distance threshold for SRGR.
    srgr_lambda : float
        Scaling factor for SRGR.
    min_samples_per_culture : int
        Minimum samples required within a culture to compute per-culture metrics.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing computed scores:
        - "srgr_accuracy_overall": float or None
        - "srgr_accuracy_per_culture": Dict[int, float or None]
    """
    results = {}
    real_motion_np = _to_pose_numpy_4d(real_motion)
    gen_motion_np = _to_pose_numpy_4d(gen_motion)
    batch_size, time_dim, n_joints, n_coordinates = real_motion_np.shape

    # --- Input Validation and Preparation ---
    if gen_motion_np.shape != real_motion_np.shape:
        raise ValueError(
            f"SRGR requires real and generated motion tensors to have the same shape. Got {real_motion_np.shape} and {gen_motion_np.shape}"
        )
    if real_culture_labels.shape[0] != batch_size or gen_culture_labels.shape[0] != batch_size:
        raise ValueError("Label tensors must match batch size.")
    if not torch.equal(real_culture_labels, gen_culture_labels):
        warnings.warn(
            "SRGR Warning: Real and generated culture labels differ. Calculation assumes alignment by index, which might be incorrect.")

    # Use real labels for indexing, assuming alignment
    real_culture_labels_np = _to_numpy_1d(real_culture_labels).astype(np.int64)

    # --- Helper to calculate average SRGR over a batch/subset ---
    def _calculate_average_srgr(real_poses_batch, gen_poses_batch, delta, lambda_scale):
        scores = []
        num_samples = real_poses_batch.shape[0]
        if num_samples == 0:
            return None

        for i in range(num_samples):
            score = compute_srgr_score(
                generated_poses=gen_poses_batch[i],
                ground_truth_poses=real_poses_batch[i],
                delta=delta,
                lambda_scale=lambda_scale
            )
            if score is not None:
                scores.append(score)
            # else: # Optional: print warning if score is None
            #     warnings.warn(f"SRGR calculation failed for sample {i}")

        return float(np.mean(scores)) if scores else None

    # --- Overall Calculation ---
    print("Calculating Overall SRGR Score...")
    results["srgr_accuracy_overall"] = _calculate_average_srgr(
        real_motion_np, gen_motion_np, srgr_delta, srgr_lambda
    )
    print(f"Overall SRGR Accuracy: {results['srgr_accuracy_overall']}")

    # --- Per-Culture Calculation ---
    print("\nCalculating Per-Culture SRGR Scores...")
    srgr_accuracy_per_culture = {}

    for c in range(num_cultures):
        print(f"--- Culture {c} ---")
        # Use real_culture_labels for indexing, assuming alignment
        culture_idx_np = np.where(real_culture_labels_np == c)[0]

        num_samples_in_culture = len(culture_idx_np)
        print(f"  Samples found: {num_samples_in_culture}")

        if num_samples_in_culture < min_samples_per_culture:
            print(f"  Skipping culture {c} (samples {num_samples_in_culture} < min {min_samples_per_culture})")
            srgr_accuracy_per_culture[c] = None
            continue

        # Filter both real and generated using the same indices to maintain pairing
        real_motion_subset = real_motion_np[culture_idx_np]
        gen_motion_subset = gen_motion_np[culture_idx_np]

        score = _calculate_average_srgr(
            real_motion_subset, gen_motion_subset, srgr_delta, srgr_lambda
        )
        srgr_accuracy_per_culture[c] = score
        print(f"  SRGR Score: {score}")

    results["srgr_accuracy_per_culture"] = srgr_accuracy_per_culture

    print("\nSRGR Analysis Complete.")
    return results


def compute_alignment_scores(
    motion_enc: torch.Tensor,
    lowlevel_enc: torch.Tensor,
    highlevel_enc: torch.Tensor,
    real_labels: torch.Tensor,
    measure: str = "cosine",
    seed: int = 1234
) -> Dict[str, Any]:
    """
    Compute alignment between three sets of encodings (motion, low-level, high-level)
    according to the described steps.

    Steps:
        1) Overall alignment between motion and low-level
        2) Overall alignment between motion and high-level
        3) Alignment between motion and low-level for each culture
        4) Alignment between motion and high-level for each culture
        5) Randomly shuffle low-level encodings across the batch; overall alignment
        6) Randomly shuffle high-level encodings across the batch; overall alignment
        7) Randomly shuffle low-level encodings across the batch, for each culture
        8) Randomly shuffle high-level encodings across the batch, for each culture

    Parameters
    ----------
    motion_enc : torch.Tensor
        Shape (batch_size, 512). Motion encodings.
    lowlevel_enc : torch.Tensor
        Shape (batch_size, 512). Low-level encodings.
    highlevel_enc : torch.Tensor
        Shape (batch_size, 512). High-level encodings.
    real_labels : torch.Tensor
        Shape (batch_size,). Culture labels for each sample.
    measure : str, optional
        Alignment measure to use. Supported: "cosine" for cosine similarity.
        (You could extend to "dot", "l2" distance, etc.)
    seed : int, optional
        Random seed for reproducible shuffles.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing alignment measures for steps 1-8.
    """

    # -------------------------------------------------------------------------
    #  Helper: measure_alignment
    # -------------------------------------------------------------------------
    def measure_alignment(x: torch.Tensor, y: torch.Tensor, measure: str = "cosine") -> float:
        """
        Measures alignment between x and y along the batch dimension (dim=0).
        x, y: shape (B, 512)
        measure: "cosine" => average cosine similarity across the batch
        """
        if measure == "cosine":
            # Cosine similarity for each pair (elementwise), then average
            sim = F.cosine_similarity(x, y, dim=1)  # shape: (B,)
            return sim.mean().item()
        else:
            raise ValueError(f"Unsupported measure: {measure}")

    # -------------------------------------------------------------------------
    #  Helper: random_shuffle
    # -------------------------------------------------------------------------
    def random_shuffle(x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        """
        Returns a shuffled version of x along the batch dimension (dim=0),
        preserving shape. x: shape (B, 512).
        """
        B = x.shape[0]
        perm = torch.randperm(B, generator=generator)
        return x[perm]

    # -------------------------------------------------------------------------
    # 1) Overall alignment: motion vs. low-level
    # -------------------------------------------------------------------------
    overall_motion_low = measure_alignment(motion_enc, lowlevel_enc, measure=measure)

    # -------------------------------------------------------------------------
    # 2) Overall alignment: motion vs. high-level
    # -------------------------------------------------------------------------
    overall_motion_high = measure_alignment(motion_enc, highlevel_enc, measure=measure)

    # -------------------------------------------------------------------------
    # 3) Alignment: motion vs. low-level, per culture
    # 4) Alignment: motion vs. high-level, per culture
    # -------------------------------------------------------------------------
    unique_cultures = real_labels.unique()
    motion_low_by_culture = {}
    motion_high_by_culture = {}

    for c in unique_cultures:
        idx = (real_labels == c)
        if idx.sum() < 1:
            # No samples for this culture
            motion_low_by_culture[int(c.item())] = None
            motion_high_by_culture[int(c.item())] = None
            continue

        # motion vs low-level
        motion_low_by_culture[int(c.item())] = measure_alignment(
            motion_enc[idx], lowlevel_enc[idx], measure=measure
        )
        # motion vs high-level
        motion_high_by_culture[int(c.item())] = measure_alignment(
            motion_enc[idx], highlevel_enc[idx], measure=measure
        )

    # -------------------------------------------------------------------------
    # 5) Randomly shuffle low-level across the batch; overall alignment
    # 6) Randomly shuffle high-level across the batch; overall alignment
    # -------------------------------------------------------------------------
    # We'll fix a generator for reproducibility:
    gen = torch.Generator()
    gen.manual_seed(seed)

    lowlevel_shuffled = random_shuffle(lowlevel_enc, gen)
    highlevel_shuffled = random_shuffle(highlevel_enc, gen)

    overall_motion_low_shuf = measure_alignment(motion_enc, lowlevel_shuffled, measure=measure)
    overall_motion_high_shuf = measure_alignment(motion_enc, highlevel_shuffled, measure=measure)

    # -------------------------------------------------------------------------
    # 7) Randomly shuffle low-level for each culture
    # 8) Randomly shuffle high-level for each culture
    # -------------------------------------------------------------------------
    motion_low_shuf_by_culture = {}
    motion_high_shuf_by_culture = {}

    # For each culture, shuffle the subset of lowlevel or highlevel for that culture
    # or shuffle them across the entire batch?
    # **Interpretation**: We'll shuffle encodings only among the same-culture subset
    # so the "random" is restricted within each culture.
    # (If you meant shuffle across the entire batch but compute alignment by culture,
    # that is a different approach. We'll assume a per-culture shuffle.)
    for c in unique_cultures:
        idx = (real_labels == c)
        n_c = idx.sum().item()
        if n_c < 2:
            # If there's only one sample or none, alignment is ill-defined
            motion_low_shuf_by_culture[int(c.item())] = None
            motion_high_shuf_by_culture[int(c.item())] = None
            continue

        # Shuffle just within the c-subset
        subset_low = lowlevel_enc[idx]
        subset_high = highlevel_enc[idx]
        subset_motion = motion_enc[idx]

        # Shuffle each subset separately
        perm_c = torch.randperm(n_c, generator=gen)
        subset_low_shuf = subset_low[perm_c]
        perm_c2 = torch.randperm(n_c, generator=gen)
        subset_high_shuf = subset_high[perm_c2]

        # measure alignment
        motion_low_shuf_by_culture[int(c.item())] = measure_alignment(
            subset_motion, subset_low_shuf, measure=measure
        )
        motion_high_shuf_by_culture[int(c.item())] = measure_alignment(
            subset_motion, subset_high_shuf, measure=measure
        )

    # Compile results
    results = {
        # 1 & 2
        "overall_motion_low_alignment": overall_motion_low,
        "overall_motion_high_alignment": overall_motion_high,

        # 3 & 4
        "motion_low_by_culture": motion_low_by_culture,
        "motion_high_by_culture": motion_high_by_culture,

        # 5 & 6
        "overall_motion_low_shuffled_alignment": overall_motion_low_shuf,
        "overall_motion_high_shuffled_alignment": overall_motion_high_shuf,

        # 7 & 8
        "motion_low_shuffled_by_culture": motion_low_shuf_by_culture,
        "motion_high_shuffled_by_culture": motion_high_shuf_by_culture
    }

    return results

################################################################################
# 5) Variance between gestures of the same culture, for both real and generated,
#    and variance between each culture combination, for both real and generated.
################################################################################
def compute_variance(
        real_motion: torch.Tensor,
        generated_motion: torch.Tensor,
        real_labels: torch.Tensor,
        num_cultures: int = 4
) -> Dict[str, Any]:
    """
    Computes:
      (a) variance within each culture for real and generated gestures
      (b) variance for each pair of cultures (combining data from both cultures)
          for real and generated gestures

    Parameters
    ----------
    real_motion : torch.Tensor
        (batch_size, time_dim, 512)
    generated_motion : torch.Tensor
        (batch_size, time_dim, 512)
    real_labels : torch.Tensor
        (batch_size,)
    num_cultures : int
        Number of unique culture labels

    Returns
    -------
    Dict[str, Any]
        {
          "variance_per_culture": {
              0: {"real": <float>, "generated": <float>},
              1: {...},
              ...
          },
          "variance_culture_pairs": {
              (0,1): {"real": <float>, "generated": <float>},
              (0,2): {...},
              ...
          }
        }
    """

    # We'll define a helper to compute variance in a flattened space
    def compute_flattened_variance(x: torch.Tensor) -> float:
        # Flatten: (batch_size, time_dim, 512) => (batch_size * time_dim, 512)
        if x.shape[0] == 0:
            return 0.0
        x_flat = x.reshape(-1, x.shape[-1])  # shape: (B*T, 512)
        return x_flat.var(dim=0).mean().item()  # mean variance across feature dimension

    # (a) Variance per culture
    variance_per_culture = {}
    for culture_id in range(num_cultures):
        real_idx = (real_labels == culture_id)
        gen_idx = (real_labels == culture_id)

        real_var = compute_flattened_variance(real_motion[real_idx])
        gen_var = compute_flattened_variance(generated_motion[gen_idx])

        variance_per_culture[culture_id] = {
            "real": real_var,
            "generated": gen_var
        }

    # (b) Variance for each pair of cultures
    from itertools import combinations
    culture_pairs = list(combinations(range(num_cultures), 2))
    variance_culture_pairs = {}

    for (c1, c2) in culture_pairs:
        real_idx_c1 = (real_labels == c1)
        real_idx_c2 = (real_labels == c2)
        real_motion_pair = torch.cat([real_motion[real_idx_c1], real_motion[real_idx_c2]], dim=0)

        gen_idx_c1 = (real_labels == c1)
        gen_idx_c2 = (real_labels == c2)
        gen_motion_pair = torch.cat([generated_motion[gen_idx_c1], generated_motion[gen_idx_c2]], dim=0)

        real_var_pair = compute_flattened_variance(real_motion_pair)
        gen_var_pair = compute_flattened_variance(gen_motion_pair)

        variance_culture_pairs[(c1, c2)] = {
            "real": real_var_pair,
            "generated": gen_var_pair
        }

    return {
        "variance_per_culture": variance_per_culture,
        "variance_culture_pairs": variance_culture_pairs
    }


def evaluate_culture_classification(
    generated_loader,
    real_loader,
    num_classes=None,
    classifier_checkpoint_path: Optional[str] = None,
    classifier_cl_type: str = "culclA",
    classifier_mode: str = "adversarial_backbone",
    classifier_d_model: int = 512,
    motion_eval_steps: int = 20,
):
    """
    Evaluates culture classification performance by computing F1 Score, Balanced Accuracy,
    Accuracy, and ROC AUC.

    Args:
        val_loader (torch.utils.data.DataLoader): Validation data loader.
        num_classes (int, optional): Number of classes in the classification task.
                                     If None, inferred from data.

    Returns:
        dict: A dictionary containing the computed metrics.
    """
    eval_dict = OrderedDict()
    metrics = OrderedDict({
        'F1_Score': None,
        'Balanced_Accuracy': None,
        'Accuracy': None,
        'ROC_AUC': None
    })

    def _compute_cls_metrics(labels, preds, probs, n_classes):
        labels = np.asarray(labels)
        preds = np.asarray(preds)
        probs = np.asarray(probs)
        if n_classes is None:
            if probs.ndim == 2 and probs.shape[1] > 1:
                resolved_n_classes = int(probs.shape[1])
            else:
                max_label = max(np.max(labels), np.max(preds)) if labels.size and preds.size else 1
                resolved_n_classes = int(max_label) + 1
        else:
            resolved_n_classes = int(n_classes)
        is_binary = probs.ndim == 1 or (probs.ndim == 2 and probs.shape[1] == 1)
        if is_binary:
            f1 = f1_score(labels, preds, average='binary')
            per_class_values = f1_score(labels, preds, average=None, labels=[0, 1], zero_division=0)
            per_class_f1 = {str(class_idx): float(score) for class_idx, score in enumerate(per_class_values.tolist())}
        else:
            f1 = f1_score(labels, preds, average='macro')
            per_class_values = f1_score(
                labels,
                preds,
                average=None,
                labels=list(range(resolved_n_classes)),
                zero_division=0,
            )
            per_class_f1 = {
                str(class_idx): float(score)
                for class_idx, score in enumerate(per_class_values.tolist())
            }
        balanced_acc = balanced_accuracy_score(labels, preds)
        acc = accuracy_score(labels, preds)
        roc_auc = None
        try:
            if is_binary:
                scores = probs if probs.ndim == 1 else probs[:, 0]
                roc_auc = roc_auc_score(labels, scores)
            else:
                labels_binarized = label_binarize(labels, classes=range(resolved_n_classes))
                roc_auc = roc_auc_score(labels_binarized, probs, average='macro', multi_class='ovr')
        except ValueError as e:
            print(f"ROC AUC computation failed: {e}")
        return {
            "f1": f1,
            "per_class_f1": per_class_f1,
            "balanced_accuracy": balanced_acc,
            "accuracy": acc,
            "roc_auc": roc_auc,
        }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    culture_embedder = None
    uses_motion_features = False
    def _normalize_cl_type(cl_type: str) -> str:
        raw = str(cl_type or "").strip()
        if not raw:
            return "culclA"
        if raw.lower().startswith("culcl") and len(raw) >= 6:
            return "culcl" + raw[5:].upper()
        return raw

    classifier_cl_type = _normalize_cl_type(classifier_cl_type)
    classifier_mode = str(classifier_mode or "adversarial_backbone").strip().lower()
    valid_modes = {"adversarial_backbone", "fishr_backbone", "motion"}
    if classifier_mode not in valid_modes:
        warnings.warn(
            f"[Culture Classification] Unknown classifier_mode='{classifier_mode}'. "
            "Falling back to 'adversarial_backbone'."
        )
        classifier_mode = "adversarial_backbone"

    def _cl_type_requires_multimodal(cl_type: str) -> bool:
        cl = str(cl_type or "").strip().lower()
        needs_text = cl in {"culclb", "culclf", "culclh", "culcli"}
        needs_audio = cl in {"culclc", "culcld", "culcle", "culclg", "culclh", "culcli", "culclk", "culcll"}
        return bool(needs_text or needs_audio)

    def _mode_requires_multimodal(mode: str, cl_type: str) -> bool:
        if mode == "fishr_backbone":
            return True
        if mode == "adversarial_backbone":
            return _cl_type_requires_multimodal(cl_type)
        return False

    def _cl_type_signature(cl_type: str) -> Dict[str, bool]:
        cl = str(cl_type or "").strip().lower()
        return {
            "poses": cl in {"culcla", "culclb", "culclc", "culcld", "culcle", "culclh", "culclj"},
            "text": cl in {"culclb", "culclf", "culclh", "culcli"},
            "audio": cl in {"culclc", "culcle", "culclg", "culclh", "culcli", "culcll"},
            "wav2vec": cl in {"culcld", "culcle", "culclg", "culclh", "culcli", "culclk"},
        }

    def _infer_checkpoint_mode(state_dict: Dict[str, Any]) -> Optional[str]:
        keys = [str(k) for k in state_dict.keys()]
        has_motion_pool = any("motion_pool" in k for k in keys)
        has_adv_backbone = any(
            ("attention_poolers" in k)
            or ("audio_encoders" in k)
            or ("sentence_mapper" in k)
            or ("core_model" in k)
            for k in keys
        )
        if has_adv_backbone and not has_motion_pool:
            return "adversarial_backbone"
        if has_motion_pool and not has_adv_backbone:
            return "motion"
        return None

    def _state_has_key_fragment(state_dict: Dict[str, Any], fragment: str) -> bool:
        return any(fragment in str(k) for k in state_dict.keys())

    def _state_get_tensor_by_suffix(state_dict: Dict[str, Any], suffix: str):
        for key, value in state_dict.items():
            if str(key).endswith(suffix):
                return value
        return None

    def _checkpoint_has_motion_branch(state_dict: Dict[str, Any]) -> bool:
        keys = [str(k) for k in state_dict.keys()]
        return any(("motion_proj" in k) or ("motion_pool" in k) for k in keys)

    def _checkpoint_signature(state_dict: Dict[str, Any]) -> Dict[str, bool]:
        return {
            "poses": _state_has_key_fragment(state_dict, "attention_poolers.poses"),
            "text": _state_has_key_fragment(state_dict, "sentence_mapper"),
            "audio": _state_has_key_fragment(state_dict, "attention_poolers.audio"),
            "wav2vec": _state_has_key_fragment(state_dict, "attention_poolers.wav2vec"),
        }

    def _infer_checkpoint_cl_type(state_dict: Dict[str, Any]) -> Optional[str]:
        sig = _checkpoint_signature(state_dict)
        signature_to_cl_type = {
            (False, False, False, True): "culclK",
            (False, False, True, False): "culclL",
            (False, False, True, True): "culclG",
            (False, True, False, False): "culclF",
            (False, True, True, True): "culclI",
            (True, False, True, False): "culclC",
            (True, False, False, True): "culclD",
            (True, False, True, True): "culclE",
            (True, True, False, False): "culclB",
            (True, True, True, True): "culclH",
        }
        key = (sig["poses"], sig["text"], sig["audio"], sig["wav2vec"])
        if key == (True, False, False, False):
            pose_proj_w = _state_get_tensor_by_suffix(state_dict, "pose_encoder.input_projection_layer.weight")
            if torch.is_tensor(pose_proj_w) and pose_proj_w.ndim == 2 and int(pose_proj_w.shape[1]) == 54:
                return "culclJ"
            return "culclA"
        return signature_to_cl_type.get(key)

    def _has_module_prefix(sd):
        return any(str(k).startswith("module.") for k in sd.keys())

    def _align_module_prefix(sd, model_obj):
        model_is_dp = isinstance(model_obj, nn.DataParallel)
        state_has_dp = _has_module_prefix(sd)
        if model_is_dp and not state_has_dp:
            return {f"module.{k}": v for k, v in sd.items()}
        if (not model_is_dp) and state_has_dp:
            return {(k[7:] if str(k).startswith("module.") else k): v for k, v in sd.items()}
        return sd

    def _adapt_positional_embeddings(sd, model_obj):
        model_state = model_obj.state_dict()
        adapted = dict(sd)
        for key, value in list(adapted.items()):
            if key not in model_state:
                continue
            target = model_state[key]
            if value.shape == target.shape:
                continue
            if str(key).endswith("pos_encoder.pe") and value.ndim == target.ndim:
                fixed = target.clone()
                slices = tuple(slice(0, min(a, b)) for a, b in zip(fixed.shape, value.shape))
                fixed[slices] = value[slices].to(fixed.dtype)
                adapted[key] = fixed
        return adapted

    def _extract_eval_fields(batch):
        """
        Extracts fields from GeneratedDataset tuples.
        Supported formats:
          - no-alignment:  (poses, motion, cont_motion, culture, speaker, onsets)
          - alignment:     (poses, motion, culture_logits, pooled, high_ctx, audio_ctx, cont_motion, culture, speaker, onsets)
        """
        if len(batch) >= 14:
            # alignment + multimodal extras:
            # (poses, motion, culture_logits, pooled, high_ctx, audio_ctx, cont_motion, culture, speaker, onsets_eval,
            #  text, mel, onsets_full, wav2vec)
            return {
                "final_motion": batch[1],
                "culture_labels": batch[7],
                "audio_onsets": batch[12],
                "text_features": batch[10],
                "audio_mels": batch[11],
                "audio_wav2vec": batch[13],
                "culture_logits": batch[2],
            }
        if len(batch) >= 10:
            # no-alignment + multimodal extras:
            # (poses, motion, cont_motion, culture, speaker, onsets_eval, text, mel, onsets_full, wav2vec)
            maybe_labels = batch[3]
            if torch.is_tensor(maybe_labels) and maybe_labels.ndim == 1:
                return {
                    "final_motion": batch[1],
                    "culture_labels": batch[3],
                    "audio_onsets": batch[8],
                    "text_features": batch[6],
                    "audio_mels": batch[7],
                    "audio_wav2vec": batch[9],
                    "culture_logits": None,
                }
            # legacy alignment without multimodal extras:
            # (poses, motion, culture_logits, pooled, high_ctx, audio_ctx, cont_motion, culture, speaker, onsets_eval)
            return {
                "final_motion": batch[1],
                "culture_labels": batch[7],
                "audio_onsets": batch[9],
                "text_features": None,
                "audio_mels": None,
                "audio_wav2vec": None,
                "culture_logits": batch[2],
            }
        if len(batch) >= 6:
            return {
                "final_motion": batch[1],
                "culture_labels": batch[3],
                "audio_onsets": batch[5],
                "text_features": None,
                "audio_mels": None,
                "audio_wav2vec": None,
                "culture_logits": None,
            }
        raise ValueError(f"Unsupported evaluation batch format with length={len(batch)}")

    classifier_checkpoint_path = (
        str(classifier_checkpoint_path).strip()
        if classifier_checkpoint_path is not None
        else None
    )
    requested_external_classifier = bool(classifier_checkpoint_path)
    if requested_external_classifier and not os.path.isfile(classifier_checkpoint_path):
        raise FileNotFoundError(
            f"[Culture Classification] External classifier checkpoint not found: {classifier_checkpoint_path}"
        )
    use_external_classifier = requested_external_classifier
    external_state_dict = None

    # Check multimodal availability in evaluation tuples.
    has_multimodal_features = False
    try:
        first_batch = next(iter(generated_loader))
        first_fields = _extract_eval_fields(first_batch)
        has_multimodal_features = all(
            first_fields.get(k) is not None for k in ("text_features", "audio_mels", "audio_onsets", "audio_wav2vec")
        )
    except Exception:
        has_multimodal_features = False

    if use_external_classifier:
        try:
            external_state_dict = torch.load(classifier_checkpoint_path, map_location=device)
            if isinstance(external_state_dict, dict) and "model_dict" in external_state_dict:
                external_state_dict = external_state_dict["model_dict"]

            inferred_mode = _infer_checkpoint_mode(external_state_dict)
            checkpoint_mode = classifier_mode
            if inferred_mode is not None and inferred_mode != classifier_mode:
                warnings.warn(
                    "[Culture Classification] Requested mode "
                    f"'{classifier_mode}' does not match checkpoint structure (inferred '{inferred_mode}'). "
                    f"Loading with inferred checkpoint architecture '{inferred_mode}'. "
                    "This changes the loader architecture, not the checkpoint training objective."
                )
                checkpoint_mode = inferred_mode

            inferred_cl_type = _infer_checkpoint_cl_type(external_state_dict)
            if inferred_cl_type is not None and inferred_cl_type != classifier_cl_type:
                warnings.warn(
                    "[Culture Classification] Requested cl_type "
                    f"'{classifier_cl_type}' does not match checkpoint structure "
                    f"(inferred '{inferred_cl_type}'). Using inferred cl_type '{inferred_cl_type}'."
                )
                classifier_cl_type = inferred_cl_type

            checkpoint_signature = _checkpoint_signature(external_state_dict)
            if checkpoint_mode == "motion" and not _checkpoint_has_motion_branch(external_state_dict):
                pose_only_signature = {"poses": True, "text": False, "audio": False, "wav2vec": False}
                if checkpoint_signature == pose_only_signature:
                    fallback_cl_type = inferred_cl_type or "culclA"
                    warnings.warn(
                        "[Culture Classification] Requested mode='motion' but checkpoint has "
                        "pose-only Fishr backbone (no `motion_proj`/`motion_pool`). "
                        "Falling back to mode='adversarial_backbone' with "
                        f"cl_type='{fallback_cl_type}'."
                    )
                    checkpoint_mode = "adversarial_backbone"
                    classifier_cl_type = fallback_cl_type
                else:
                    raise RuntimeError(
                        "Checkpoint is incompatible with classifier_mode='motion': "
                        "missing motion-specific weights (`motion_proj` / `motion_pool`) "
                        "and not a pose-only Fishr backbone."
                    )

            requested_signature = _cl_type_signature(classifier_cl_type)
            if checkpoint_mode == "adversarial_backbone" and requested_signature != checkpoint_signature:
                raise RuntimeError(
                    "External classifier cl_type is incompatible with checkpoint structure. "
                    f"requested={classifier_cl_type} signature={requested_signature} "
                    f"checkpoint_signature={checkpoint_signature}"
                )
            uses_motion_features = bool(checkpoint_mode == "motion" or requested_signature.get("poses", False))
            if not uses_motion_features:
                raise RuntimeError(
                    "[Culture Classification] The selected classifier configuration does not consume motion inputs. "
                    "This evaluation requires motion-based cultural classification of generated gestures. "
                    "Use `--culture-classifier-mode motion` with a motion-trained Fishr checkpoint "
                    "or a cl_type that includes poses."
                )

            requires_multimodal = _mode_requires_multimodal(checkpoint_mode, classifier_cl_type)
            if requires_multimodal and not has_multimodal_features:
                raise RuntimeError(
                    "External classifier requires text/mel/onset/wav2vec for this mode/cl_type, "
                    "but evaluation batches do not provide them."
                )

            if checkpoint_mode == "motion":
                culture_embedder = Fishr(
                    proj_dim=int(classifier_d_model),
                    num_classes=4,
                    num_domains=0,
                    is_nonlinear=False,
                    use_motion=True,
                ).to(device)
            elif checkpoint_mode == "adversarial_backbone":
                backbone_cfg = SimpleNamespace(
                    dropout=0.1,
                    layer_neurons=int(classifier_d_model),
                    levels=1,
                    embed_dim=int(classifier_d_model),
                )
                culture_embedder = Fishr(
                    proj_dim=int(classifier_d_model),
                    num_classes=4,
                    num_domains=0,
                    is_nonlinear=False,
                    use_motion=False,
                    cl_type=classifier_cl_type,
                    d_model=int(classifier_d_model),
                    pose_enc_type="transformer",
                    audio_enc_type=None,
                    raw_poses=False,
                    backbone_config=backbone_cfg,
                    use_adversarial_backbone=True,
                ).to(device)
            else:  # fishr_backbone
                culture_embedder = Fishr(
                    proj_dim=int(classifier_d_model),
                    num_classes=4,
                    num_domains=0,
                    is_nonlinear=False,
                    use_motion=False,
                    cl_type=classifier_cl_type,
                    d_model=int(classifier_d_model),
                    use_adversarial_backbone=False,
                ).to(device)
            state_dict = external_state_dict
            state_dict = _align_module_prefix(state_dict, culture_embedder)
            state_dict = _adapt_positional_embeddings(state_dict, culture_embedder)

            model_state_keys = set(culture_embedder.state_dict().keys())
            ckpt_keys = set(state_dict.keys())
            matched = len(model_state_keys.intersection(ckpt_keys))
            coverage_model = matched / max(1, len(model_state_keys))
            coverage_ckpt = matched / max(1, len(ckpt_keys))
            print(
                "[Culture Classification] Checkpoint compatibility: "
                f"matched={matched} model_cov={coverage_model:.3f} ckpt_cov={coverage_ckpt:.3f}"
            )
            if coverage_model < 0.70:
                raise RuntimeError(
                    "External classifier checkpoint is incompatible with the selected evaluation mode. "
                    f"coverage_model={coverage_model:.3f}"
                )

            incompatible = culture_embedder.load_state_dict(state_dict, strict=False)
            if incompatible.missing_keys:
                print(f"[Culture Classification] Missing keys: {incompatible.missing_keys}")
            if incompatible.unexpected_keys:
                print(f"[Culture Classification] Unexpected keys: {incompatible.unexpected_keys}")
            culture_embedder.eval()
            print(
                f"[Culture Classification] Using external classifier "
                f"(requested={classifier_mode}, architecture={checkpoint_mode}, cl_type={classifier_cl_type}) "
                f"from: {classifier_checkpoint_path}"
            )
        except Exception as exc:
            raise RuntimeError(
                f"[Culture Classification] Failed to load external classifier from "
                f"'{classifier_checkpoint_path}': {exc}"
            ) from exc
    else:
        print("[Culture Classification] No external classifier checkpoint provided. Using batch logits.")

    all_labels = []
    all_labels_real = []

    all_preds_emb = []
    all_probs_emb = []
    all_preds_emb_real = []
    all_probs_emb_real = []
    warned_missing_logits = False
    checkpoint_mode_runtime = locals().get("checkpoint_mode", classifier_mode)
    requires_multimodal_runtime = _mode_requires_multimodal(checkpoint_mode_runtime, classifier_cl_type)

    with (torch.no_grad()):
        for batch_gen, batch_real in zip(generated_loader, real_loader):
            gen_fields = _extract_eval_fields(batch_gen)
            real_fields = _extract_eval_fields(batch_real)
            gen_final_motion = gen_fields["final_motion"]
            real_final_motion = real_fields["final_motion"]
            if torch.is_tensor(gen_final_motion) and gen_final_motion.ndim >= 3 and int(gen_final_motion.shape[1]) > int(motion_eval_steps):
                gen_final_motion = gen_final_motion[:, :int(motion_eval_steps), :]
            if torch.is_tensor(real_final_motion) and real_final_motion.ndim >= 3 and int(real_final_motion.shape[1]) > int(motion_eval_steps):
                real_final_motion = real_final_motion[:, :int(motion_eval_steps), :]
            gen_culture_labels = gen_fields["culture_labels"]
            real_culture_labels = real_fields["culture_labels"]
            gen_culture_output = gen_fields["culture_logits"]
            real_culture_output = real_fields["culture_logits"]

            if use_external_classifier:
                gen_motion = gen_final_motion.to(device)
                real_motion = real_final_motion.to(device)
                if checkpoint_mode_runtime == "motion":
                    cl_inputs_gen = (gen_motion,)
                    cl_inputs_real = (real_motion,)
                else:
                    if requires_multimodal_runtime and any(
                        gen_fields.get(k) is None or real_fields.get(k) is None
                        for k in ("text_features", "audio_mels", "audio_onsets", "audio_wav2vec")
                    ):
                        raise RuntimeError(
                            "Missing multimodal features in eval batches for "
                            f"classifier_mode='{classifier_mode}' and cl_type='{classifier_cl_type}'."
                        )
                    gen_bsz = gen_motion.shape[0]
                    real_bsz = real_motion.shape[0]
                    gen_dtype = gen_motion.dtype
                    real_dtype = real_motion.dtype

                    gen_text = gen_fields["text_features"]
                    if gen_text is None:
                        gen_text = torch.zeros((gen_bsz, 768), device=device, dtype=gen_dtype)
                    else:
                        gen_text = gen_text.to(device)
                    gen_mel = gen_fields["audio_mels"]
                    if gen_mel is None:
                        gen_mel = torch.zeros((gen_bsz, 156, 64), device=device, dtype=gen_dtype)
                    else:
                        gen_mel = gen_mel.to(device)
                    gen_onset = gen_fields["audio_onsets"]
                    if gen_onset is None:
                        gen_onset = torch.zeros((gen_bsz, 156), device=device, dtype=gen_dtype)
                    else:
                        gen_onset = gen_onset.to(device)
                    gen_w2v = gen_fields["audio_wav2vec"]
                    if gen_w2v is None:
                        gen_w2v = torch.zeros((gen_bsz, 50, 1024), device=device, dtype=gen_dtype)
                    else:
                        gen_w2v = gen_w2v.to(device)

                    real_text = real_fields["text_features"]
                    if real_text is None:
                        real_text = torch.zeros((real_bsz, 768), device=device, dtype=real_dtype)
                    else:
                        real_text = real_text.to(device)
                    real_mel = real_fields["audio_mels"]
                    if real_mel is None:
                        real_mel = torch.zeros((real_bsz, 156, 64), device=device, dtype=real_dtype)
                    else:
                        real_mel = real_mel.to(device)
                    real_onset = real_fields["audio_onsets"]
                    if real_onset is None:
                        real_onset = torch.zeros((real_bsz, 156), device=device, dtype=real_dtype)
                    else:
                        real_onset = real_onset.to(device)
                    real_w2v = real_fields["audio_wav2vec"]
                    if real_w2v is None:
                        real_w2v = torch.zeros((real_bsz, 50, 1024), device=device, dtype=real_dtype)
                    else:
                        real_w2v = real_w2v.to(device)

                    cl_inputs_gen = (gen_motion, gen_text, gen_mel, gen_onset, gen_w2v)
                    cl_inputs_real = (real_motion, real_text, real_mel, real_onset, real_w2v)
                culture_emb_outputs = culture_embedder.predict(cl_inputs_gen)
                culture_emb_outputs_real = culture_embedder.predict(cl_inputs_real)
            else:
                if gen_culture_output is None or real_culture_output is None:
                    if not warned_missing_logits:
                        print("[Culture Classification] Missing batch logits; skipping classification.")
                        warned_missing_logits = True
                    continue
                culture_emb_outputs = gen_culture_output.to(device)
                culture_emb_outputs_real = real_culture_output.to(device)

            np_probs_outputs_cultures = F.softmax(culture_emb_outputs, dim=1).data.cpu().numpy()
            np_probs_outputs_cultures_real = F.softmax(culture_emb_outputs_real, dim=1).data.cpu().numpy()
            np_pred_outputs_cultures = np.argmax(np_probs_outputs_cultures, axis=1)
            np_pred_outputs_cultures_real = np.argmax(np_probs_outputs_cultures_real, axis=1)
            all_preds_emb.extend(np_pred_outputs_cultures)
            all_probs_emb.extend(np_probs_outputs_cultures)
            all_preds_emb_real.extend(np_pred_outputs_cultures_real)
            all_probs_emb_real.extend(np_probs_outputs_cultures_real)

            culture_real = gen_culture_labels.cpu().numpy()
            culture_real_2 = real_culture_labels.cpu().numpy()
            all_labels.extend(culture_real)
            all_labels_real.extend(culture_real_2)

    if len(all_labels) == 0:
        print("[Culture Classification] No valid batches available for classification.")
        empty_cls_metrics = {
            "f1": None,
            "per_class_f1": {},
            "balanced_accuracy": None,
            "accuracy": None,
            "roc_auc": None,
        }
        eval_dict["generated"] = empty_cls_metrics.copy()
        eval_dict["real"] = empty_cls_metrics.copy()
        eval_dict["source"] = "external_classifier" if use_external_classifier else "batch_logits"
        return eval_dict

    generated_metrics = _compute_cls_metrics(all_labels, all_preds_emb, all_probs_emb, num_classes)
    real_metrics = _compute_cls_metrics(all_labels_real, all_preds_emb_real, all_probs_emb_real, num_classes)

    print(
        "CULTURE EVAL (generated): "
        f"F1={generated_metrics['f1']} BalancedAcc={generated_metrics['balanced_accuracy']} "
        f"Acc={generated_metrics['accuracy']} ROC_AUC={generated_metrics['roc_auc']}"
    )
    print(
        "CULTURE EVAL (real): "
        f"F1={real_metrics['f1']} BalancedAcc={real_metrics['balanced_accuracy']} "
        f"Acc={real_metrics['accuracy']} ROC_AUC={real_metrics['roc_auc']}"
    )

    eval_dict["generated"] = generated_metrics
    eval_dict["real"] = real_metrics
    eval_dict["source"] = "external_classifier" if use_external_classifier else "batch_logits"
    eval_dict["classifier_mode"] = classifier_mode if use_external_classifier else None
    eval_dict["classifier_checkpoint_architecture"] = checkpoint_mode_runtime if use_external_classifier else None
    eval_dict["classifier_cl_type"] = classifier_cl_type if use_external_classifier else None
    eval_dict["uses_motion_features"] = bool(uses_motion_features) if use_external_classifier else None
    return eval_dict

################################################################################
# 6) (Repeated bullet) Multimodality or any additional advanced measure
#    If needed, you can replicate or extend the above approach. Shown here as a stub.
################################################################################
def compute_multimodality_extended(
        real_motion: torch.Tensor,
        generated_motion: torch.Tensor,
        real_labels: torch.Tensor
) -> Dict[str, Any]:
    """
    An extended or alternative multimodality measure, or an extra advanced metric
    if you want to differentiate from the basic 'compute_multimodality' function.

    For demonstration, this function can simply wrap the existing `compute_multimodality`
    or define a more advanced measure.
    """
    # Example: Just re-use compute_multimodality here, or define your own approach
    base_multimodality = compute_multimodality(real_motion, generated_motion, real_labels)

    # Possibly incorporate more advanced statistics, like measuring spread in PCA space, etc.
    # For now, we will just pass it through.

    # You could add additional fields to show how you'd "extend" it
    base_multimodality["extended_metric"] = 0.0  # placeholder for a new measure
    return base_multimodality


def compute_beat_align_score(real_onset_strength: torch.Tensor,
                             generated_3d_poses: torch.Tensor,
                             sigma: float = 0.5) -> float:
    """
    Computes the Beat Alignment Score by extracting audio beats from the onset strength
    using librosa, and motion beats as the local minima of the kinetic velocity.

    Parameters
    ----------
    real_onset_strength : torch.Tensor
        Tensor of onset strengths with shape (n_samples, 156).
    generated_3d_poses : torch.Tensor
        Tensor of generated 3D poses with shape (n_samples, 75, 9, 3).
    sigma : float, optional
        Standard deviation parameter for the Gaussian weighting (default: 0.5).

    Returns
    -------
    float
        The average beat alignment score.
    """
    if librosa is None:
        return float("nan")
    # Convert tensors to numpy arrays for processing with librosa and scipy.
    real_onset_strength_np = real_onset_strength.cpu().numpy()
    gen_3d_poses_np = generated_3d_poses.cpu().numpy()
    n_samples = real_onset_strength_np.shape[0]
    beat_scores = []

    # Define sampling rates based on data length and duration (5 seconds)
    audio_sr = 156 / 5.0  # ~31.2 Hz for audio onset strength
    motion_sr = 75 / 5.0  # 15 Hz for motion frames

    for i in range(n_samples):
        # ----- Audio Beat Extraction -----
        # Get onset envelope for sample i
        onset_env = real_onset_strength_np[i]
        # Use librosa to detect beats from the onset envelope.
        # Note: librosa.onset.onset_detect returns indices relative to the onset envelope.
        audio_beats_indices = librosa.onset.onset_detect(onset_envelope=onset_env, sr=audio_sr, backtrack=False)
        # Convert indices to time (seconds)
        audio_beats_times = audio_beats_indices / audio_sr

        # ----- Motion Beat Extraction -----
        # Get generated 3D poses for sample i (shape: (75, 9, 3))
        poses = gen_3d_poses_np[i]
        # Compute kinetic velocity: differences between consecutive frames.
        # For each joint, compute the Euclidean norm of the difference between consecutive frames.
        vel = np.linalg.norm(np.diff(poses, axis=0), axis=-1)  # shape: (74, 9)
        # Average over joints to obtain a single velocity time-series per frame.
        vel_mean = vel.mean(axis=1)  # shape: (74,)

        # Find local minima in the velocity (i.e., candidate motion beats).
        # If too few frames, return an empty array.
        if len(vel_mean) < 3:
            motion_beats_indices = np.array([])
        else:
            motion_beats_indices = argrelextrema(vel_mean, np.less)[0]
        # Convert motion beat frame indices to time (seconds)
        motion_beats_times = motion_beats_indices / motion_sr

        # ----- Compute Beat Alignment for the sample -----
        if len(motion_beats_times) == 0 or len(audio_beats_times) == 0:
            sample_score = 0.0
        else:
            score_sum = 0.0
            for t_motion in motion_beats_times:
                # Find the distance to all detected audio beats
                diffs = np.abs(audio_beats_times - t_motion)
                # Get the minimum difference
                min_diff = diffs.min() if diffs.size > 0 else np.inf
                # Apply the Gaussian weighting
                score_sum += np.exp(- (min_diff ** 2) / (2 * sigma ** 2))
            sample_score = score_sum / len(motion_beats_times)
        beat_scores.append(sample_score)

    # Average the score over all samples
    return float(np.mean(beat_scores))



def geodesic_distance(R, R_hat):
    # R, R_hat: (..., 3, 3) rotation matrices
    # Compute trace along the last two dimensions
    # Clamp the value inside the valid range for arccos to avoid numerical issues
    trace_val = torch.clamp((R.transpose(-2, -1) @ R_hat).diagonal(dim1=-2, dim2=-1).sum(-1), -1.0, 3.0)
    return torch.acos((trace_val - 1) / 2)

def compute_srgr(relative_R_real, relative_R_gen, delta=0.05, lambda_factor=1.0):
    """
    relative_R_real, relative_R_gen: tensors of shape (P, T, J, 3, 3)
    """
    P, T, J, _, _ = relative_R_real.shape
    # Compute geodesic distance for each sample, frame, and joint.
    dists = geodesic_distance(relative_R_real, relative_R_gen)  # shape (P, T, J)
    avg_error_per_sample = dists.mean(dim=(1, 2))  # shape (P,)
    indicators = (avg_error_per_sample < delta).float()
    srgr = lambda_factor * indicators.mean().item()
    return srgr

def compute_pck(relative_R_real, relative_R_gen, delta=0.05):
    """
    relative_R_real, relative_R_gen: tensors of shape (P, T, J, 3, 3)
    """
    dists = geodesic_distance(relative_R_real, relative_R_gen)  # shape (P, T, J)
    indicators = (dists < delta).float()
    pck = indicators.mean().item()
    return pck

def evaluate_all_metrics(real_loader, gen_loader, log_file, save_path = '', num_cultures: int = 4):
    from tqdm.auto import tqdm
    """
    Iterates over batches from the real and generated data loaders, computing all metrics,
    accumulating results over batches, and prints the average results.

    Parameters
    ----------
    real_loader : DataLoader
        Data loader for real motion batches.
    gen_loader : DataLoader
        Data loader for generated motion batches.
    num_cultures : int
        Number of unique culture labels (default: 4).
    """

    # Accumulators for metrics. Each entry is a list that we average later.
    r_match_high_level_list = []
    r_match_low_level_list = []
    l2_distance_mean_list = []
    l1_distance_mean_list = []

    # For FGD, we accumulate the overall scalar, and dictionaries for per-culture and culture pairs
    fgd_overall_list = []
    fgd_per_culture_list = []  # list of dicts: {culture: value}
    fgd_real_culture_couples_list = []
    fgd_gen_culture_couples_list = []
    fgd_real_speaker_couples_list = []
    fgd_gen_speaker_couples_list = []
    fgd_real_intra_culture_list = []
    fgd_gen_intra_culture_list = []

    # For diversity
    real_diversity_list = []
    gen_diversity_list = []
    real_diversity_by_culture_list = []
    gen_diversity_by_culture_list = []

    overall_motion_low_alignment_list = []
    overall_motion_high_alignment_list = []
    motion_low_by_culture_list = []
    motion_high_by_culture_list = []
    overall_motion_low_shuffled_alignment_list = []
    overall_motion_high_shuffled_alignment_list = []
    motion_low_shuffled_by_culture_list = []
    motion_high_shuffled_by_culture_list = []

    # Accumulator for per-culture FGD scores across batches
    fgd_per_culture_scores = {c: [] for c in range(num_cultures)}


    # For variance
    variance_per_culture_list = []  # list of dicts: {culture: {"real":..., "generated":...}}
    variance_culture_pairs_list = []  # list of dicts: {(c1, c2): {"real":..., "generated":...}}

    # Counters for batches
    batch_count = 0

    # Loop over both loaders. We assume the same number of batches
    for (real_batch, gen_batch) in tqdm(zip(real_loader, gen_loader)):
        #if batch_count == 2:
        #    break
        # For both batches, unpack:
        # For real data loader: final_motion is real motion.

        (real_3d_poses, real_final_motion, real_culture_output, real_motion_pooled, real_high_level_context,
         real_low_level_context, real_cont_motion, real_culture_labels, real_speaker_labels, *_) = real_batch

        # For generated data loader: final_motion is generated motion.
        (generated_3d_poses, gen_final_motion, gen_culture_output, gen_motion_pooled, gen_high_level_context,
         gen_low_level_context, gen_cont_motion, gen_culture_labels, gen_speaker_labels, *_) = gen_batch

        # For simplicity, assume that the labels (and culture outputs if needed) are the same across the two
        # Otherwise, adjust accordingly.
        # -----------------------------
        # 1. R match scores (only for generated motion)
        print("Evaluating R match...")
        r_match_res = compute_r_match_scores_for_generated(
            motion_pooled=gen_motion_pooled,
            low_level_context=gen_low_level_context,
            high_level_context=gen_high_level_context,
            labels=gen_culture_labels
        )
        r_match_high_level_list.append(r_match_res["r_match_high_level"])
        r_match_low_level_list.append(r_match_res["r_match_low_level"])

        # -----------------------------
        # 2. L2 distance between real and generated motion
        print("Evaluating L2 distance...")
        l2_res = compute_l2_distance(
            real_motion=real_final_motion,
            generated_motion=gen_final_motion
        )
        l2_distance_mean_list.append(l2_res["l2_distance_mean"])

        # 2. L1 distance between real and generated motion
        print("Evaluating L1 distance...")
        l1_res = compute_l1_distance(
            real_motion=real_final_motion,
            generated_motion=gen_final_motion
        )
        l1_distance_mean_list.append(l1_res["l1_distance_mean"])

        # -----------------------------
        # 3. FGD scores
        print("Evaluating FGD...")

        fgd_res = compute_fgd_scores(
            real_motion=real_cont_motion,
            gen_motion=gen_cont_motion,
            real_culture_labels=real_culture_labels,
            gen_culture_labels=gen_culture_labels,
            real_speaker_labels=real_speaker_labels,
            gen_speaker_labels=gen_speaker_labels,
            num_cultures=num_cultures
        )

        fgd_overall_list.append(fgd_res["fgd_overall"])
        fgd_per_culture_list.append(fgd_res["fgd_per_culture"])
        fgd_real_culture_couples_list.append(fgd_res["fgd_real_culture_couples"])
        fgd_gen_culture_couples_list.append(fgd_res["fgd_gen_culture_couples"])
        fgd_real_speaker_couples_list.append(fgd_res["fgd_real_speaker_couples"])
        fgd_gen_speaker_couples_list.append(fgd_res["fgd_gen_speaker_couples"])
        fgd_real_intra_culture_list.append(fgd_res["fgd_real_intra_culture"])
        fgd_gen_intra_culture_list.append(fgd_res["fgd_gen_intra_culture"])

        # Accumulate per-culture FGD scores for statistical testing
        for c, fgd in fgd_res["fgd_per_culture"].items():
            if fgd is not None:
                fgd_per_culture_scores[c].append(fgd)



        # -----------------------------
        # 4. Diversity
        print("Evaluating Diversity...")
        diversity_res = compute_diversity(
            real_motion=real_cont_motion,
            generated_motion=gen_cont_motion,
            real_labels=real_culture_labels,
            gen_labels=gen_culture_labels,
            diversity_times=50
        )
        real_diversity_list.append(diversity_res["real_diversity"])
        gen_diversity_list.append(diversity_res["gen_diversity"])
        real_diversity_by_culture_list.append(diversity_res["real_diversity_by_culture"])
        gen_diversity_by_culture_list.append(diversity_res["gen_diversity_by_culture"])

        print("Evaluating Alignment...")
        alignments = compute_alignment_scores(motion_enc = gen_motion_pooled,
                                              lowlevel_enc= gen_low_level_context,
                                              highlevel_enc= gen_high_level_context,
                                              real_labels= gen_culture_labels)

        overall_motion_low_alignment_list.append(alignments["overall_motion_low_alignment"])
        overall_motion_high_alignment_list.append(alignments["overall_motion_high_alignment"])
        motion_low_by_culture_list.append(alignments["motion_low_by_culture"])
        motion_high_by_culture_list.append(alignments["motion_high_by_culture"])
        overall_motion_low_shuffled_alignment_list.append(alignments["overall_motion_low_shuffled_alignment"])
        overall_motion_high_shuffled_alignment_list.append(alignments["overall_motion_high_shuffled_alignment"])
        motion_low_shuffled_by_culture_list.append(alignments["motion_low_shuffled_by_culture"])
        motion_high_shuffled_by_culture_list.append(alignments["motion_high_shuffled_by_culture"])




        # -----------------------------
        # 5. Variance
        '''
        print("Evaluating Variance...")
        variance_res = compute_variance(
            real_motion=real_final_motion,
            generated_motion=gen_final_motion,
            real_labels=real_labels,
            num_cultures=num_cultures
        )
        variance_per_culture_list.append(variance_res["variance_per_culture"])
        variance_culture_pairs_list.append(variance_res["variance_culture_pairs"])
        '''

        batch_count += 1

    classification_results = evaluate_culture_classification(gen_loader, real_loader, num_classes=None)

    # --- AVERAGING AND PRINTING RESULTS ---
    def average_list(lst):
        return sum(lst) / len(lst) if lst else None

    averaged_results = OrderedDict()

    # 1. R Match Scores
    averaged_results["r_match_high_level_mean"] = average_list(r_match_high_level_list)
    averaged_results["r_match_low_level_mean"] = average_list(r_match_low_level_list)

    # 2. L2 Distance
    averaged_results["l2_distance_mean"] = average_list(l2_distance_mean_list)

    # 2.1. L1 Distance
    averaged_results["l1_distance_mean"] = average_list(l1_distance_mean_list)

    # 3. FGD Scores
    averaged_results["fgd_overall_mean"] = average_list(fgd_overall_list)

    # FGD per culture
    fgd_per_culture_avg = {}
    for c in range(num_cultures):
        vals = [d[c] for d in fgd_per_culture_list if d[c] is not None]
        fgd_per_culture_avg[c] = average_list(vals)
    averaged_results["fgd_per_culture_mean"] = fgd_per_culture_avg


    # FGD real culture couples
    fgd_real_culture_couples_avg = {}
    for key in fgd_real_culture_couples_list[0].keys():
        vals = [d[key] for d in fgd_real_culture_couples_list if d[key] is not None]
        fgd_real_culture_couples_avg[key] = average_list(vals)
    averaged_results["fgd_real_culture_couples_mean"] = fgd_real_culture_couples_avg

    # FGD generated culture couples
    fgd_gen_culture_couples_avg = {}
    for key in fgd_gen_culture_couples_list[0].keys():
        vals = [d[key] for d in fgd_gen_culture_couples_list if d[key] is not None]
        fgd_gen_culture_couples_avg[key] = average_list(vals)
    averaged_results["fgd_gen_culture_couples_mean"] = fgd_gen_culture_couples_avg

    # FGD real speaker couples
    fgd_real_speaker_couples_avg = {}
    for c in range(num_cultures):
        vals = [d[c] for d in fgd_real_speaker_couples_list if d[c] is not None]
        fgd_real_speaker_couples_avg[c] = average_list(vals)
    averaged_results["fgd_real_speaker_couples_mean"] = fgd_real_speaker_couples_avg

    # FGD generated speaker couples
    fgd_gen_speaker_couples_avg = {}
    for c in range(num_cultures):
        vals = [d[c] for d in fgd_gen_speaker_couples_list if d[c] is not None]
        fgd_gen_speaker_couples_avg[c] = average_list(vals)
    averaged_results["fgd_gen_speaker_couples_mean"] = fgd_gen_speaker_couples_avg

    # 4. Diversity
    averaged_results["real_diversity_mean"] = average_list(real_diversity_list)
    averaged_results["generated_diversity_mean"] = average_list(gen_diversity_list)

    # Diversity by culture
    real_diversity_by_culture_avg = {}
    for c in range(num_cultures):
        vals = [d[c] for d in real_diversity_by_culture_list if d[c] is not None]
        real_diversity_by_culture_avg[c] = average_list(vals)
    averaged_results["real_diversity_by_culture_mean"] = real_diversity_by_culture_avg

    generated_diversity_by_culture_avg = {}
    for c in range(num_cultures):
        vals = [d[c] for d in gen_diversity_by_culture_list if d[c] is not None]
        generated_diversity_by_culture_avg[c] = average_list(vals)
    averaged_results["generated_diversity_by_culture_mean"] = generated_diversity_by_culture_avg

    # 7. Intra-Culture FGD for Real Samples
    fgd_real_intra_culture_avg = {}
    for c in range(num_cultures):
        vals = [d[c] for d in fgd_real_intra_culture_list if d[c] is not None]
        fgd_real_intra_culture_avg[c] = average_list(vals)
    averaged_results["fgd_real_intra_culture_mean"] = fgd_real_intra_culture_avg

    # 8. Intra-Culture FGD for Generated Samples
    fgd_gen_intra_culture_avg = {}
    for c in range(num_cultures):
        vals = [d[c] for d in fgd_gen_intra_culture_list if d[c] is not None]
        fgd_gen_intra_culture_avg[c] = average_list(vals)
    averaged_results["fgd_gen_intra_culture_mean"] = fgd_gen_intra_culture_avg

    # 5. Alignment Scores
    averaged_results["overall_motion_low_alignment_mean"] = average_list(overall_motion_low_alignment_list)
    averaged_results["overall_motion_high_alignment_mean"] = average_list(overall_motion_high_alignment_list)

    # Alignment by culture
    motion_low_by_culture_avg = {}
    for c in range(num_cultures):
        vals = [d[c] for d in motion_low_by_culture_list if d[c] is not None]
        motion_low_by_culture_avg[c] = average_list(vals)
    averaged_results["motion_low_by_culture_mean"] = motion_low_by_culture_avg

    motion_high_by_culture_avg = {}
    for c in range(num_cultures):
        vals = [d[c] for d in motion_high_by_culture_list if d[c] is not None]
        motion_high_by_culture_avg[c] = average_list(vals)
    averaged_results["motion_high_by_culture_mean"] = motion_high_by_culture_avg

    # Shuffled Alignment
    averaged_results["overall_motion_low_shuffled_alignment_mean"] = average_list(
        overall_motion_low_shuffled_alignment_list)
    averaged_results["overall_motion_high_shuffled_alignment_mean"] = average_list(
        overall_motion_high_shuffled_alignment_list)

    # Shuffled Alignment by culture
    motion_low_shuffled_by_culture_avg = {}
    for c in range(num_cultures):
        vals = [d[c] for d in motion_low_shuffled_by_culture_list if d[c] is not None]
        motion_low_shuffled_by_culture_avg[c] = average_list(vals)
    averaged_results["motion_low_shuffled_by_culture_mean"] = motion_low_shuffled_by_culture_avg

    motion_high_shuffled_by_culture_avg = {}
    for c in range(num_cultures):
        vals = [d[c] for d in motion_high_shuffled_by_culture_list if d[c] is not None]
        motion_high_shuffled_by_culture_avg[c] = average_list(vals)
    averaged_results["motion_high_shuffled_by_culture_mean"] = motion_high_shuffled_by_culture_avg

    # 6. Classification Results
    print("Evaluating classification results...")
    classification_results = evaluate_culture_classification(gen_loader, real_loader, num_classes=num_cultures)
    averaged_results.update(classification_results)

    # --- Perform Statistical Significance Tests ---
    print("Performing statistical significance tests...")
    significance_results = OrderedDict()

    # Helper function for paired t-test
    def paired_t_test(real, gen, metric_name):
        if len(real) != len(gen):
            print(f"Cannot perform paired t-test for {metric_name}: unequal lengths.")
            return None, None
        t_stat, p_val = stats.ttest_rel(real, gen)
        return t_stat, p_val

    # 1. R Match Scores
    t_stat, p_val = paired_t_test(r_match_high_level_list, r_match_low_level_list, "R Match High Level vs Low Level")
    significance_results["r_match_high_vs_low_pval"] = p_val

    # 2. L2 Distance
    # Typically, L2 distance is between real and generated, but to perform a t-test, you'd need multiple measurements.
    # Assuming l2_distance_mean_list contains per-batch L2 distances between real and generated.
    # However, t-test might not be meaningful here as it's not paired measurements of the same entities.
    # Instead, consider independent samples t-test.
    if len(l2_distance_mean_list) >= 2:
        t_stat_l2, p_val_l2 = stats.ttest_1samp(l2_distance_mean_list, 0)
        significance_results["l2_distance_pval"] = p_val_l2
    else:
        significance_results["l2_distance_pval"] = None

    # 3. FGD Overall
    if len(fgd_overall_list) >= 2:
        t_stat_fgd, p_val_fgd = stats.ttest_1samp(fgd_overall_list, 0)
        significance_results["fgd_overall_pval"] = p_val_fgd
    else:
        significance_results["fgd_overall_pval"] = None

    # 4. Diversity
    # Compare real_diversity_list vs gen_diversity_list
    if len(real_diversity_list) == len(gen_diversity_list) and len(real_diversity_list) >= 2:
        t_stat_div, p_val_div = stats.ttest_rel(real_diversity_list, gen_diversity_list)
        significance_results["diversity_pval"] = p_val_div
    else:
        significance_results["diversity_pval"] = None

    # 5. Alignment Scores
    if len(overall_motion_low_alignment_list) >= 2:
        t_stat_align_low, p_val_align_low = stats.ttest_1samp(overall_motion_low_alignment_list, 0)
        significance_results["alignment_low_pval"] = p_val_align_low
    else:
        significance_results["alignment_low_pval"] = None

    if len(overall_motion_high_alignment_list) >= 2:
        t_stat_align_high, p_val_align_high = stats.ttest_1samp(overall_motion_high_alignment_list, 0)
        significance_results["alignment_high_pval"] = p_val_align_high
    else:
        significance_results["alignment_high_pval"] = None

    # --- Statistical Significance Tests for Per-Culture FGD Scores ---
    print("Performing pairwise statistical tests for per-culture FGD scores...")
    for (c1, c2) in combinations(range(num_cultures), 2):
        scores_c1 = fgd_per_culture_scores[c1]
        scores_c2 = fgd_per_culture_scores[c2]
        if len(scores_c1) >= 2 and len(scores_c2) >= 2:
            # Perform independent two-sample t-test (Welch's t-test)
            t_stat_c, p_val_c = stats.ttest_ind(scores_c1, scores_c2, equal_var=False)
            significance_results[f"fgd_culture_{c1}_vs_{c2}_pval"] = p_val_c
        else:
            significance_results[f"fgd_culture_{c1}_vs_{c2}_pval"] = None  # Not enough data


    # --- Statistical Significance Tests for Intra-Culture FGDs ---
    print("Performing statistical significance tests for intra-culture FGDs...")
    for c in range(num_cultures):
        real_fgd_scores = [d[c].cpu().numpy() for d in fgd_real_intra_culture_list if d[c] is not None]
        gen_fgd_scores = [d[c].cpu().numpy() for d in fgd_gen_intra_culture_list if d[c] is not None]

        # Ensure equal number of samples for paired test
        min_len = min(len(real_fgd_scores), len(gen_fgd_scores))
        if min_len >= 2:
            # Truncate to the same length
            real_fgd_scores = real_fgd_scores[:min_len]
            gen_fgd_scores = gen_fgd_scores[:min_len]

            # Perform paired t-test
            t_stat, p_val = stats.ttest_rel(real_fgd_scores, gen_fgd_scores)
            significance_results[f"fgd_intra_culture_{c}_real_vs_gen_pval"] = p_val
        else:
            significance_results[f"fgd_intra_culture_{c}_real_vs_gen_pval"] = None  # Not enough data


    print("Performing statistical significance tests for real culture pairwise FGD vs intra-culture FGD...")
    for (c1, c2) in combinations(range(num_cultures), 2):
        # Extract FGD scores for the culture pair across all batches
        fgd_pair_scores = []
        fgd_intra_c1_scores = []
        fgd_intra_c2_scores = []

        for batch_idx in range(len(fgd_real_culture_couples_list)):
            pair_key = (c1, c2)
            fgd_pair = fgd_real_culture_couples_list[batch_idx].get(pair_key, None)
            fgd_intra_c1 = fgd_real_intra_culture_list[batch_idx].get(c1, None)
            fgd_intra_c2 = fgd_real_intra_culture_list[batch_idx].get(c2, None)

            if fgd_pair is not None and fgd_intra_c1 is not None and fgd_intra_c2 is not None:
                fgd_pair_scores.append(fgd_pair)
                fgd_intra_c1_scores.append(fgd_intra_c1.cpu().numpy())
                fgd_intra_c2_scores.append(fgd_intra_c2.cpu().numpy())

        # Compare culture pair FGD vs intra-culture c1
        if len(fgd_pair_scores) >= 2 and len(fgd_intra_c1_scores) >= 2:
            t_stat_pair_vs_c1, p_val_pair_vs_c1 = stats.ttest_rel(fgd_pair_scores, fgd_intra_c1_scores)
            significance_results[f"fgd_real_culture_{c1}_vs_{c2}_pair_vs_intra_culture_{c1}_pval"] = p_val_pair_vs_c1
        else:
            significance_results[
                f"fgd_real_culture_{c1}_vs_{c2}_pair_vs_intra_culture_{c1}_pval"] = None  # Not enough data

        # Compare culture pair FGD vs intra-culture c2
        if len(fgd_pair_scores) >= 2 and len(fgd_intra_c2_scores) >= 2:
            t_stat_pair_vs_c2, p_val_pair_vs_c2 = stats.ttest_rel(fgd_pair_scores, fgd_intra_c2_scores)
            significance_results[f"fgd_real_culture_{c1}_vs_{c2}_pair_vs_intra_culture_{c2}_pval"] = p_val_pair_vs_c2
        else:
            significance_results[
                f"fgd_real_culture_{c1}_vs_{c2}_pair_vs_intra_culture_{c2}_pval"] = None  # Not enough data


    # --- Statistical Significance Tests for Pairwise Culture Diversity Scores ---
    print("Performing statistical significance tests for pairwise culture diversity scores...")
    for (c1, c2) in combinations(range(num_cultures), 2):
        # Extract diversity scores for each culture across all batches
        diversity_c1_scores = [d[c1] for d in real_diversity_by_culture_list if d[c1] is not None]
        diversity_c2_scores = [d[c2] for d in real_diversity_by_culture_list if d[c2] is not None]

        if len(diversity_c1_scores) >= 2 and len(diversity_c2_scores) >= 2:
            # Perform independent two-sample t-test (Welch's t-test)
            t_stat_div, p_val_div = stats.ttest_ind(diversity_c1_scores, diversity_c2_scores, equal_var=False)
            significance_results[f"diversity_culture_{c1}_vs_{c2}_pval"] = p_val_div
        else:
            significance_results[f"diversity_culture_{c1}_vs_{c2}_pval"] = None  # Not enough data

    # --- Prepare Final Results ---
    final_results = {
        "averaged_metrics": averaged_results,
        "statistical_significance": significance_results
    }



    # Add more statistical tests as needed for other metrics

    # --- Prepare Final Results ---
    final_results = {
        "averaged_metrics": averaged_results,
        "statistical_significance": significance_results
    }

    # --- Print Results ---
    print("\n===== Averaged Metrics =====")
    for key, value in averaged_results.items():
        print(f"{key}: {value}")

    print("\n===== Statistical Significance =====")
    for key, value in significance_results.items():
        print(f"{key}: {value}")

    # --- Log Results ---
    print(f"\nLogging results to {log_file}...")
    with open(log_file, 'w') as f:
        f.write("===== Averaged Metrics =====\n")
        for key, value in averaged_results.items():
            f.write(f"{key}: {value}\n")

        f.write("\n===== Statistical Significance =====\n")
        for key, value in significance_results.items():
            f.write(f"{key}: {value}\n")

    print("Evaluation complete.")

    all_lists = {
        "overall_motion_low_alignment_list": overall_motion_low_alignment_list,
        "overall_motion_high_alignment_list": overall_motion_high_alignment_list,
        "motion_low_by_culture_list": motion_low_by_culture_list,
        "motion_high_by_culture_list": motion_high_by_culture_list,
        "overall_motion_low_shuffled_alignment_list": overall_motion_low_shuffled_alignment_list,
        "overall_motion_high_shuffled_alignment_list": overall_motion_high_shuffled_alignment_list,
        "motion_low_shuffled_by_culture_list": motion_low_shuffled_by_culture_list,
        "motion_high_shuffled_by_culture_list": motion_high_shuffled_by_culture_list,
        "real_diversity_list": real_diversity_list,
        "gen_diversity_list": gen_diversity_list,
        "real_diversity_by_culture_list": real_diversity_by_culture_list,
        "gen_diversity_by_culture_list": gen_diversity_by_culture_list,
        "fgd_overall_list": fgd_overall_list,
        "fgd_per_culture_list": fgd_per_culture_list,
        "fgd_real_culture_couples_list": fgd_real_culture_couples_list,
        "fgd_gen_culture_couples_list": fgd_gen_culture_couples_list,
        "fgd_real_speaker_couples_list": fgd_real_speaker_couples_list,
        "fgd_gen_speaker_couples_list": fgd_gen_speaker_couples_list,
        "fgd_real_intra_culture_list": fgd_real_intra_culture_list,
        "fgd_gen_intra_culture_list": fgd_gen_intra_culture_list,
        "l2_distance_mean_list": l2_distance_mean_list,
        "r_match_high_level_list": r_match_high_level_list,
        "r_match_low_level_list": r_match_low_level_list
    }

    # Save the dictionary to a pickle file
    pickle_file_path = "evaluation_results.pkl"
    pickle_save_path = os.path.join(save_path, pickle_file_path)
    with open(pickle_save_path, "wb") as pickle_file:
        pickle.dump(all_lists, pickle_file)

    return final_results














def evaluate_all_metrics_bulk(
    real_loader: DataLoader,
    gen_loader: DataLoader,
    log_file: str,
    save_path: str = '',
    num_cultures: int = 4,
    diversity_times: int = 300,
    # --- SRGR Parameters ---
    srgr_delta: float = 0.05,
    srgr_lambda: float = 1.0,
    # --- Beat Alignment Parameters ---
    beat_align_sigma: float = 3.0,
    beat_align_kinematic_joints: Optional[List[int]] = None, # Define default joint list if desired
    velocity_peak_height_threshold: Optional[float] = None,
    onset_peak_height_threshold: Optional[float] = None,
    # --- Other metric params (can be added as needed) ---
    min_samples_for_metrics: int = 3, # Min samples per culture for per-culture metrics
    compute_alignment: bool = False,   # Wether we want to comptue alignment scores
    compute_srgr_beat: bool = False,
    run_culture_classification: bool = False,
    culture_classifier_checkpoint_path: Optional[str] = None,
    culture_classifier_cl_type: str = "culclA",
    culture_classifier_mode: str = "adversarial_backbone",
    culture_classifier_d_model: int = 512,
    save_raw_data: bool = False,

    ):
    """
    Collects all batches from the real and generated data loaders, computes all metrics
    (including SRGR and Beat Alignment) on the aggregated data, and prints/logs the results.

    Parameters
    ----------
    real_loader : DataLoader
        Data loader for real motion batches. MUST yield tuples containing:
        (3d_poses, final_motion, culture_output, motion_pooled, high_level_context,
         low_level_context, cont_motion, culture_labels, speaker_labels,
         pose_timestamps, audio_onset_strength, audio_onset_timestamps)
    gen_loader : DataLoader
        Data loader for generated motion batches. MUST yield tuples containing:
        (3d_poses, final_motion, culture_output, motion_pooled, high_level_context,
         low_level_context, cont_motion, culture_labels, speaker_labels,
         audio_onset_strength, audio_onset_timestamps)
         *Note: Assumes pose_timestamps from real_loader apply to generated data.*
    log_file : str
        Path to the log file where results will be saved.
    save_path : str, optional
        Directory path where the pickle file will be saved (default is '').
    num_cultures : int, optional
        Number of unique culture labels (default: 4).
    srgr_delta : float, optional
        Squared distance threshold for SRGR.
    srgr_lambda : float, optional
        Scaling factor for SRGR.
    beat_align_sigma : float, optional
        BAS Gaussian kernel width in 60-FPS frame units (paper default: 3).
    beat_align_kinematic_joints : List[int], optional
        Indices of joints for Beat Alignment kinematic beats. Defaults to using all joints if None.
    velocity_peak_height_threshold : float, optional
        Kept for compatibility; ignored in paper BAS mode.
    onset_peak_height_threshold : float, optional
        Kept for compatibility; ignored in paper BAS mode.
    min_samples_for_metrics : int, optional
        Minimum samples required per culture for reliable per-culture metric calculation.
    """
    print(
        f"[Eval] Metric params: srgr_delta={srgr_delta}, srgr_lambda={srgr_lambda}, "
        f"beat_align_sigma={beat_align_sigma} frames@60fps"
    )

    if beat_align_kinematic_joints is None:
        # Attempt to determine number of joints later if caller did not specify.
        pass


    # Initialize lists to collect all data
    # Real data
    real_3d_poses_all = []
    real_final_motion_all = []
    real_culture_output_all = []
    real_motion_pooled_all = []
    real_high_level_context_all = []
    real_low_level_context_all = []
    real_cont_motion_all = []
    real_culture_labels_all = []
    real_speaker_labels_all = []
    # Data needed specifically for Beat Align / SRGR
    real_pose_timestamps_all = [] # Expect NumPy arrays or list of arrays
    real_audio_onset_s_all = []   # Expect NumPy arrays or list of arrays
    real_audio_onset_t_all = []   # Expect NumPy arrays or list of arrays

    # Generated data
    gen_3d_poses_all = []
    gen_final_motion_all = []
    gen_culture_output_all = []
    gen_motion_pooled_all = []
    gen_high_level_context_all = []
    gen_low_level_context_all = []
    gen_cont_motion_all = []
    gen_culture_labels_all = []
    gen_speaker_labels_all = []
    # Data needed specifically for Beat Align / SRGR
    gen_audio_onset_s_all = []    # Expect NumPy arrays or list of arrays
    gen_audio_onset_t_all = []    # Expect NumPy arrays or list of arrays

    # Ensure both loaders have the same number of batches
    if len(real_loader) != len(gen_loader):
        warnings.warn(f"Real loader length ({len(real_loader)}) != Generated loader length ({len(gen_loader)}). Using minimum length.")
        # Or raise ValueError("Real and generated loaders have different number of batches.")

    num_batches = min(len(real_loader), len(gen_loader))

    print("Collecting all data from loaders...")
    # Determine number of joints from first batch for default BeatAlign joints
    first_real_batch = next(iter(real_loader))
    if beat_align_kinematic_joints is None:
        try:
            # Assuming first element is 3d_poses: (batch, time, features) or (batch, time, joints, 3)
            first_poses = first_real_batch[0]
            if first_poses.ndim == 4 and first_poses.shape[-1] == 3:
                num_joints = first_poses.shape[2]
                beat_align_kinematic_joints = list(range(num_joints))
                print(f"Auto-determined {num_joints} joints. Using all for Beat Alignment.")
            elif first_poses.ndim == 3:
                features = first_poses.shape[-1]
                if features % 3 == 0:
                     num_joints = features // 3
                     beat_align_kinematic_joints = list(range(num_joints))
                     print(f"Auto-determined {num_joints} joints (from features). Using all for Beat Alignment.")
                else:
                     raise ValueError("Cannot determine joints automatically.")
            else:
                raise ValueError("Cannot determine joints automatically.")
        except Exception as e:
            print(f"Error auto-determining joints for Beat Alignment: {e}. Please specify `beat_align_kinematic_joints`.")
            # Or set a fixed default: beat_align_kinematic_joints = [0, 1, 2]
            # return None # Exit if joints are essential and undetermined

    # Reset loaders if needed or iterate directly
    for real_batch, gen_batch in tqdm(zip(real_loader, gen_loader), total=num_batches):
        try:
            # --- Unpack real batch ---
            # **ADJUST THIS UNPACKING based on your DataLoader's output structure**
            if compute_alignment == True:

                (real_3d_poses, real_final_motion, real_culture_output, real_motion_pooled, real_high_level_context,
                 real_low_level_context, real_cont_motion, real_culture_labels, real_speaker_labels, real_audio_onset_s, *_) = real_batch
            else:
                (real_3d_poses, real_final_motion, real_cont_motion, real_culture_labels, real_speaker_labels,
                 real_audio_onset_s, *_) = real_batch

            pose_fps = 15.0
            pose_len = int(real_3d_poses.shape[1])
            onset_len = int(real_audio_onset_s.shape[1])
            # Keep audio and motion on the same temporal window to avoid biased BAS.
            duration_sec = float(pose_len) / pose_fps if pose_len > 0 else 0.0

            real_pose_timestamps = np.linspace(
                0.0, duration_sec, num=pose_len, endpoint=False, dtype=np.float32
            )
            real_pose_timestamps = np.tile(real_pose_timestamps, (real_3d_poses.shape[0], 1))

            if onset_len > 0 and duration_sec > 0:
                real_audio_onset_t = np.linspace(
                    0.0, duration_sec, num=onset_len, endpoint=False, dtype=np.float32
                )
            else:
                real_audio_onset_t = np.zeros((onset_len,), dtype=np.float32)
            real_audio_onset_t = np.tile(real_audio_onset_t, (real_audio_onset_s.shape[0], 1))
            # Convert to torch.Tensor
            real_pose_timestamps = torch.from_numpy(real_pose_timestamps).float()
            real_audio_onset_t = torch.from_numpy(real_audio_onset_t).float()


            real_3d_poses_all.append(real_3d_poses)
            real_final_motion_all.append(real_final_motion)
            if compute_alignment == True:
                real_culture_output_all.append(real_culture_output)
                real_motion_pooled_all.append(real_motion_pooled)
                real_high_level_context_all.append(real_high_level_context)
                real_low_level_context_all.append(real_low_level_context)
            real_cont_motion_all.append(real_cont_motion)
            real_culture_labels_all.append(real_culture_labels)
            real_speaker_labels_all.append(real_speaker_labels)
            # Append new data
            real_pose_timestamps_all.append(real_pose_timestamps)
            real_audio_onset_s_all.append(real_audio_onset_s)
            real_audio_onset_t_all.append(real_audio_onset_t)

            # --- Unpack generated batch ---
             # **ADJUST THIS UNPACKING based on your DataLoader's output structure**
            if compute_alignment == True:
                (generated_3d_poses, gen_final_motion, gen_culture_output, gen_motion_pooled, gen_high_level_context,
                 gen_low_level_context, gen_cont_motion, gen_culture_labels, gen_speaker_labels, gen_audio_onset_s, *_) = gen_batch
            else:
                (generated_3d_poses, gen_final_motion, gen_cont_motion, gen_culture_labels, gen_speaker_labels,
                 gen_audio_onset_s, *_) = gen_batch # No timestamps assumed needed here

            gen_audio_onset_t = real_audio_onset_t

            gen_3d_poses_all.append(generated_3d_poses)
            gen_final_motion_all.append(gen_final_motion)
            if compute_alignment == True:
                gen_culture_output_all.append(gen_culture_output)
                gen_motion_pooled_all.append(gen_motion_pooled)
                gen_high_level_context_all.append(gen_high_level_context)
                gen_low_level_context_all.append(gen_low_level_context)
            gen_cont_motion_all.append(gen_cont_motion)
            gen_culture_labels_all.append(gen_culture_labels)
            gen_speaker_labels_all.append(gen_speaker_labels)
            # Append new data
            gen_audio_onset_s_all.append(gen_audio_onset_s)
            gen_audio_onset_t_all.append(gen_audio_onset_t)

        except ValueError as e:
            print(f"\nError unpacking batch: {e}. Check DataLoader output structure.")
            print(f"Expected Real: 12 elements (..., timestamps, onset_s, onset_t)")
            print(f"Expected Gen: 11 elements (..., onset_s, onset_t)")
            print(f"Got Real len: {len(real_batch)}, Got Gen len: {len(gen_batch)}")
            # Decide how to handle: skip batch, raise error, etc.
            # raise e # Re-raise to stop execution
            continue # Skip this batch

    # --- Concatenate all batches ---
    print("Concatenating all batches...")
    try:
        # Standard Tensors
        real_3d_poses_all = torch.cat(real_3d_poses_all, dim=0)
        real_final_motion_all = torch.cat(real_final_motion_all, dim=0)
        if compute_alignment == True:
            real_culture_output_all = torch.cat(real_culture_output_all, dim=0)
            real_motion_pooled_all = torch.cat(real_motion_pooled_all, dim=0)
            real_high_level_context_all = torch.cat(real_high_level_context_all, dim=0)
            real_low_level_context_all = torch.cat(real_low_level_context_all, dim=0)
        real_cont_motion_all = torch.cat(real_cont_motion_all, dim=0)
        real_culture_labels_all = torch.cat(real_culture_labels_all, dim=0)
        real_speaker_labels_all = torch.cat(real_speaker_labels_all, dim=0)
        real_audio_onset_s_all = torch.cat(real_audio_onset_s_all, dim=0)
        real_audio_onset_t_all = torch.cat(real_audio_onset_t_all, dim=0)
        real_pose_timestamps_all = torch.cat(real_pose_timestamps_all, dim=0)

        gen_3d_poses_all = torch.cat(gen_3d_poses_all, dim=0)
        gen_final_motion_all = torch.cat(gen_final_motion_all, dim=0)
        if compute_alignment == True:
            gen_culture_output_all = torch.cat(gen_culture_output_all, dim=0)
            gen_motion_pooled_all = torch.cat(gen_motion_pooled_all, dim=0)
            gen_high_level_context_all = torch.cat(gen_high_level_context_all, dim=0)
            gen_low_level_context_all = torch.cat(gen_low_level_context_all, dim=0)
        gen_cont_motion_all = torch.cat(gen_cont_motion_all, dim=0)
        gen_culture_labels_all = torch.cat(gen_culture_labels_all, dim=0)
        gen_speaker_labels_all = torch.cat(gen_speaker_labels_all, dim=0)
        gen_audio_onset_s_all = torch.cat(gen_audio_onset_s_all, dim=0)
        gen_audio_onset_t_all = torch.cat(gen_audio_onset_t_all, dim=0)

        # Handle potentially non-tensor data (timestamps, audio onsets)
        # Timestamps: Expect (B, T) NumPy arrays -> vstack
        if isinstance(real_pose_timestamps_all[0], np.ndarray):
            real_pose_timestamps_all = np.vstack(real_pose_timestamps_all)
        # Add more sophisticated handling if timestamps are lists or other types

        # Audio Onsets: Expect lists of NumPy arrays -> flatten the list of lists
        if isinstance(real_audio_onset_s_all[0], list) and isinstance(real_audio_onset_s_all[0][0], np.ndarray):
             real_audio_onset_s_all = [item for sublist in real_audio_onset_s_all for item in sublist]
             real_audio_onset_t_all = [item for sublist in real_audio_onset_t_all for item in sublist]
             gen_audio_onset_s_all = [item for sublist in gen_audio_onset_s_all for item in sublist]
             gen_audio_onset_t_all = [item for sublist in gen_audio_onset_t_all for item in sublist]
        elif isinstance(real_audio_onset_s_all[0], np.ndarray): # If batches are NumPy arrays (B, N_onset)
             real_audio_onset_s_all = np.concatenate(real_audio_onset_s_all, axis=0)
             real_audio_onset_t_all = np.concatenate(real_audio_onset_t_all, axis=0)
             gen_audio_onset_s_all = np.concatenate(gen_audio_onset_s_all, axis=0)
             gen_audio_onset_t_all = np.concatenate(gen_audio_onset_t_all, axis=0)
        # Add handling for other potential structures (e.g., list of tensors)

    except Exception as e:
        print(f"Error during concatenation: {e}")
        print("Check the data types and structures yielded by the DataLoaders.")
        raise e

    # Enforce the 4-second evaluation window across motion/BAS/SRGR inputs.
    eval_pose_frames_4s = 60
    eval_motion_steps_4s = 20
    eval_onset_bins_4s = 125

    def _crop_time_dim_1(x, target_len: int, name: str):
        if x is None:
            return x
        if torch.is_tensor(x):
            if x.ndim < 2:
                return x
            cur = int(x.shape[1])
            if cur > target_len:
                print(f"[Eval] Cropping {name} from {cur} to {target_len} (4s window).")
                return x[:, :target_len, ...]
            if cur < target_len:
                warnings.warn(f"[Eval] {name} has {cur} (<{target_len}) time steps.")
            return x
        arr = np.asarray(x)
        if arr.ndim < 2:
            return x
        cur = int(arr.shape[1])
        if cur > target_len:
            print(f"[Eval] Cropping {name} from {cur} to {target_len} (4s window).")
            return arr[:, :target_len, ...]
        if cur < target_len:
            warnings.warn(f"[Eval] {name} has {cur} (<{target_len}) time steps.")
        return x

    real_3d_poses_all = _crop_time_dim_1(real_3d_poses_all, eval_pose_frames_4s, "real_3d_poses_all")
    gen_3d_poses_all = _crop_time_dim_1(gen_3d_poses_all, eval_pose_frames_4s, "gen_3d_poses_all")
    real_pose_timestamps_all = _crop_time_dim_1(real_pose_timestamps_all, eval_pose_frames_4s, "real_pose_timestamps_all")

    real_final_motion_all = _crop_time_dim_1(real_final_motion_all, eval_motion_steps_4s, "real_final_motion_all")
    gen_final_motion_all = _crop_time_dim_1(gen_final_motion_all, eval_motion_steps_4s, "gen_final_motion_all")
    real_cont_motion_all = _crop_time_dim_1(real_cont_motion_all, eval_motion_steps_4s, "real_cont_motion_all")
    gen_cont_motion_all = _crop_time_dim_1(gen_cont_motion_all, eval_motion_steps_4s, "gen_cont_motion_all")

    real_audio_onset_s_all = _crop_time_dim_1(real_audio_onset_s_all, eval_onset_bins_4s, "real_audio_onset_s_all")
    real_audio_onset_t_all = _crop_time_dim_1(real_audio_onset_t_all, eval_onset_bins_4s, "real_audio_onset_t_all")
    gen_audio_onset_s_all = _crop_time_dim_1(gen_audio_onset_s_all, eval_onset_bins_4s, "gen_audio_onset_s_all")
    gen_audio_onset_t_all = _crop_time_dim_1(gen_audio_onset_t_all, eval_onset_bins_4s, "gen_audio_onset_t_all")

    try:
        print(
            "[Eval] 4s window lengths: "
            f"final_motion={int(real_final_motion_all.shape[1])}, "
            f"poses3d={int(real_3d_poses_all.shape[1])}, "
            f"onsets={int(real_audio_onset_s_all.shape[1])}"
        )
    except Exception:
        pass

    # --- Compute Metrics ---
    '''
    print("Computing R match scores...")
    # Ensure compute_r_match_scores_for_generated is defined and imported
    try:
         r_match_res = compute_r_match_scores_for_generated(
             motion_pooled=gen_motion_pooled_all,
             low_level_context=gen_low_level_context_all,
             high_level_context=gen_high_level_context_all,
             labels=gen_culture_labels_all
         )
    except NameError:
         print("Warning: `compute_r_match_scores_for_generated` not found. Skipping.")
         r_match_res = {"r_match_high_level": None, "r_match_low_level": None}
    except Exception as e:
        print(f"Error computing R match scores: {e}")
        r_match_res = {"r_match_high_level": None, "r_match_low_level": None}
    '''
    print("Computing L2 distance...")
    try:
         l2_res = compute_l2_distance(
             real_motion=real_final_motion_all, # Using final_motion for L2/L1
             generated_motion=gen_final_motion_all
         )
    except NameError:
        print("Warning: `compute_l2_distance` not found. Skipping.")
        l2_res = {"l2_distance_mean": None}
    except Exception as e:
        print(f"Error computing L2 distance: {e}")
        l2_res = {"l2_distance_mean": None}

    print("Computing L1 distance...")
    try:
         l1_res = compute_l1_distance(
             real_motion=real_final_motion_all, # Using final_motion for L2/L1
             generated_motion=gen_final_motion_all
         )
    except NameError:
         print("Warning: `compute_l1_distance` not found. Skipping.")
         l1_res = {"l1_distance_mean": None}
    except Exception as e:
        print(f"Error computing L1 distance: {e}")
        l1_res = {"l1_distance_mean": None}


    print("Computing FGD scores...")
    try:
         fgd_res = compute_fgd_scores(
             real_motion=real_cont_motion_all, # Using cont_motion for FGD
             gen_motion=gen_cont_motion_all,
             real_culture_labels=real_culture_labels_all,
             gen_culture_labels=gen_culture_labels_all,
             real_speaker_labels=real_speaker_labels_all,
             gen_speaker_labels=gen_speaker_labels_all,
             num_cultures=num_cultures
         )
    except NameError:
        print("Warning: `compute_fgd_scores` not found. Skipping.")
        fgd_res = { # Provide default structure
            "fgd_overall": None, "fgd_per_culture": {c: None for c in range(num_cultures)},
            "fgd_real_culture_couples": {}, "fgd_gen_culture_couples": {},
            "fgd_real_speaker_couples": {c: None for c in range(num_cultures)},
            "fgd_gen_speaker_couples": {c: None for c in range(num_cultures)},
            "fgd_real_intra_culture": {c: None for c in range(num_cultures)},
            "fgd_gen_intra_culture": {c: None for c in range(num_cultures)}
        }
    except Exception as e:
         print(f"Error computing FGD scores: {e}")
         # Handle potential errors, maybe set fgd_res to None or default values
         fgd_res = { # Provide default structure
            "fgd_overall": None, "fgd_per_culture": {c: None for c in range(num_cultures)},
            "fgd_real_culture_couples": {}, "fgd_gen_culture_couples": {},
            "fgd_real_speaker_couples": {c: None for c in range(num_cultures)},
            "fgd_gen_speaker_couples": {c: None for c in range(num_cultures)},
            "fgd_real_intra_culture": {c: None for c in range(num_cultures)},
            "fgd_gen_intra_culture": {c: None for c in range(num_cultures)}
        }


    print("Computing Diversity...")
    try:
         max_diversity_times = max(2, min(int(diversity_times), int(gen_cont_motion_all.shape[0]) - 1))
         diversity_res = compute_diversity(
             real_motion=real_cont_motion_all, # Using cont_motion for Diversity
             generated_motion=gen_cont_motion_all,
             real_labels=real_culture_labels_all,
             gen_labels=gen_culture_labels_all,
             diversity_times=max_diversity_times
         )
    except NameError:
         print("Warning: `compute_diversity` not found. Skipping.")
         diversity_res = {"real_diversity": None, "gen_diversity": None,
                          "real_diversity_by_culture": {c: None for c in range(num_cultures)},
                          "gen_diversity_by_culture": {c: None for c in range(num_cultures)}}
    except Exception as e:
        print(f"Error computing diversity: {e}")
        diversity_res = {"real_diversity": None, "gen_diversity": None,
                         "real_diversity_by_culture": {c: None for c in range(num_cultures)},
                         "gen_diversity_by_culture": {c: None for c in range(num_cultures)}}

    if compute_alignment == True:
        print("Computing Alignment Scores...")
        try:
             alignments = compute_alignment_scores(
                 motion_enc=gen_motion_pooled_all, # Using pooled for alignment
                 lowlevel_enc=gen_low_level_context_all,
                 highlevel_enc=gen_high_level_context_all,
                 real_labels=gen_culture_labels_all
             )
        except NameError:
             print("Warning: `compute_alignment_scores` not found. Skipping.")
             alignments = { # Provide default structure
                 "overall_motion_low_alignment": None, "overall_motion_high_alignment": None,
                 "motion_low_by_culture": {c: None for c in range(num_cultures)},
                 "motion_high_by_culture": {c: None for c in range(num_cultures)},
                 "overall_motion_low_shuffled_alignment": None, "overall_motion_high_shuffled_alignment": None,
                 "motion_low_shuffled_by_culture": {c: None for c in range(num_cultures)},
                 "motion_high_shuffled_by_culture": {c: None for c in range(num_cultures)}
             }
        except Exception as e:
            print(f"Error computing alignment scores: {e}")
            alignments = { # Provide default structure
                 "overall_motion_low_alignment": None, "overall_motion_high_alignment": None,
                 "motion_low_by_culture": {c: None for c in range(num_cultures)},
                 "motion_high_by_culture": {c: None for c in range(num_cultures)},
                 "overall_motion_low_shuffled_alignment": None, "overall_motion_high_shuffled_alignment": None,
                 "motion_low_shuffled_by_culture": {c: None for c in range(num_cultures)},
                 "motion_high_shuffled_by_culture": {c: None for c in range(num_cultures)}
             }

    classification_diffusion_head = {}
    classification_external_fishr = {}
    if run_culture_classification:
        external_ckpt_path = (
            str(culture_classifier_checkpoint_path).strip()
            if culture_classifier_checkpoint_path is not None
            else ""
        )
        if external_ckpt_path:
            if not os.path.isfile(external_ckpt_path):
                raise FileNotFoundError(
                    f"External Fishr checkpoint not found: {external_ckpt_path}"
                )
            classification_diffusion_head = {
                "generated": {"f1": None, "balanced_accuracy": None, "accuracy": None, "roc_auc": None},
                "real": {"f1": None, "balanced_accuracy": None, "accuracy": None, "roc_auc": None},
                "source": None,
                "status": "skipped_external_classifier_requested",
            }
            print("Evaluating Culture Classification (external Fishr checkpoint)...")
            try:
                classification_external_fishr = evaluate_culture_classification(
                    gen_loader,
                    real_loader,
                    num_classes=num_cultures,
                    classifier_checkpoint_path=external_ckpt_path,
                    classifier_cl_type=culture_classifier_cl_type,
                    classifier_mode=culture_classifier_mode,
                    classifier_d_model=culture_classifier_d_model,
                )
            except Exception as e:
                raise RuntimeError(
                    f"External Fishr culture classification failed: {e}"
                ) from e
        else:
            print("Evaluating Culture Classification (diffusion head logits)...")
            try:
                classification_diffusion_head = evaluate_culture_classification(
                    gen_loader,
                    real_loader,
                    num_classes=num_cultures,
                    classifier_checkpoint_path=None,
                    classifier_cl_type=culture_classifier_cl_type,
                    classifier_mode=culture_classifier_mode,
                    classifier_d_model=culture_classifier_d_model,
                )
            except NameError:
                print("Warning: `evaluate_culture_classification` not found. Skipping diffusion-head classification.")
                classification_diffusion_head = {
                    "generated": {"f1": None, "balanced_accuracy": None, "accuracy": None, "roc_auc": None},
                    "real": {"f1": None, "balanced_accuracy": None, "accuracy": None, "roc_auc": None},
                    "source": None,
                }
            except Exception as e:
                print(f"Error evaluating diffusion-head classification: {e}")
                classification_diffusion_head = {
                    "generated": {"f1": None, "balanced_accuracy": None, "accuracy": None, "roc_auc": None},
                    "real": {"f1": None, "balanced_accuracy": None, "accuracy": None, "roc_auc": None},
                    "source": None,
                    "error": str(e),
                }
            classification_external_fishr = {
                "generated": {"f1": None, "balanced_accuracy": None, "accuracy": None, "roc_auc": None},
                "real": {"f1": None, "balanced_accuracy": None, "accuracy": None, "roc_auc": None},
                "source": None,
                "status": "external_checkpoint_not_provided",
            }

    # --- NEW: Compute SRGR ---
    srgr_res = {"srgr_accuracy_overall": None, "srgr_accuracy_per_culture": {c: None for c in range(num_cultures)}}
    if compute_srgr_beat:
        print("Computing SRGR scores...")
        try:
            srgr_res = compute_srgr_analysis(
                real_motion=real_3d_poses_all, # Assuming 3d poses are GT for SRGR
                gen_motion=gen_3d_poses_all,   # Assuming 3d poses are generated for SRGR
                real_culture_labels=real_culture_labels_all, # Must align with gen labels
                gen_culture_labels=gen_culture_labels_all,
                num_cultures=num_cultures,
                srgr_delta=srgr_delta,
                srgr_lambda=srgr_lambda,
                min_samples_per_culture=min_samples_for_metrics
            )
        except NameError:
            print("Warning: `compute_srgr_analysis` not found. Skipping.")
        except Exception as e:
            print(f"Error computing SRGR analysis: {e}")


    # --- NEW: Compute Beat Alignment ---
    beat_align_res = {
        "beat_align_real_overall": None, "beat_align_gen_overall": None,
        "beat_align_real_per_culture": {c: None for c in range(num_cultures)},
        "beat_align_gen_per_culture": {c: None for c in range(num_cultures)}
    }
    if compute_srgr_beat:
        print("Computing Beat Alignment scores...")
        if beat_align_kinematic_joints is None:
             print("Skipping Beat Alignment: `beat_align_kinematic_joints` could not be determined or specified.")
        else:
            beat_align_res = compute_beat_alignment_analysis(
                real_motion=real_3d_poses_all, # Assuming 3d poses are input for Beat Align
                gen_motion=gen_3d_poses_all,
                real_culture_labels=real_culture_labels_all,
                gen_culture_labels=gen_culture_labels_all,
                pose_timestamps=real_pose_timestamps_all, # Use collected timestamps
                real_audio_onset_strength=real_audio_onset_s_all, # Use collected audio data
                real_audio_onset_timestamps=real_audio_onset_t_all,
                gen_audio_onset_strength=gen_audio_onset_s_all,
                gen_audio_onset_timestamps=gen_audio_onset_t_all,
                num_cultures=num_cultures,
                beat_align_sigma=beat_align_sigma,
                beat_align_kinematic_joints=beat_align_kinematic_joints,
                velocity_peak_height_threshold=velocity_peak_height_threshold,
                onset_peak_height_threshold=onset_peak_height_threshold,
                min_samples_per_culture=min_samples_for_metrics
            )



    # --- Aggregate Metrics ---
    averaged_results = OrderedDict()

    # 1. R Match Scores
    #averaged_results["r_match_high_level_mean"] = r_match_res.get("r_match_high_level")
    #averaged_results["r_match_low_level_mean"] = r_match_res.get("r_match_low_level")

    # 2. L2/L1 Distances
    averaged_results["l2_distance_mean"] = l2_res.get("l2_distance_mean")
    averaged_results["l1_distance_mean"] = l1_res.get("l1_distance_mean")

    # 3. FGD Scores
    averaged_results["fgd_overall_mean"] = fgd_res.get("fgd_overall")
    # Safely access nested dicts
    fgd_pc = fgd_res.get("fgd_per_culture", {})
    averaged_results["fgd_per_culture_mean"] = {c: fgd_pc.get(c) for c in range(num_cultures)}
    # ... (similarly safe access for other FGD sub-metrics if needed, or omit if not averaged)
    fgd_rsc = fgd_res.get("fgd_real_speaker_couples", {})
    averaged_results["fgd_real_speaker_couples_mean"] = {c: fgd_rsc.get(c) for c in range(num_cultures)}
    fgd_gsc = fgd_res.get("fgd_gen_speaker_couples", {})
    averaged_results["fgd_gen_speaker_couples_mean"] = {c: fgd_gsc.get(c) for c in range(num_cultures)}
    fgd_ric = fgd_res.get("fgd_real_intra_culture", {})
    averaged_results["fgd_real_intra_culture_mean"] = {c: fgd_ric.get(c) for c in range(num_cultures)}
    fgd_gic = fgd_res.get("fgd_gen_intra_culture", {})
    averaged_results["fgd_gen_intra_culture_mean"] = {c: fgd_gic.get(c) for c in range(num_cultures)}
    # Note: Culture couples are usually presented as is, not averaged. Add if desired.
    # averaged_results["fgd_real_culture_couples"] = fgd_res.get("fgd_real_culture_couples", {})
    # averaged_results["fgd_gen_culture_couples"] = fgd_res.get("fgd_gen_culture_couples", {})


    # 4. Diversity
    averaged_results["real_diversity_mean"] = diversity_res.get("real_diversity")
    averaged_results["gen_diversity_mean"] = diversity_res.get("gen_diversity")
    # Safely access nested dicts
    div_rpc = diversity_res.get("real_diversity_by_culture", {})
    averaged_results["real_diversity_by_culture_mean"] = {c: div_rpc.get(c) for c in range(num_cultures)}
    div_gpc = diversity_res.get("gen_diversity_by_culture", {})
    averaged_results["gen_diversity_by_culture_mean"] = {c: div_gpc.get(c) for c in range(num_cultures)}

    if compute_alignment == True:
        # 5. Alignment Scores
        averaged_results["overall_motion_low_alignment_mean"] = alignments.get("overall_motion_low_alignment")
        averaged_results["overall_motion_high_alignment_mean"] = alignments.get("overall_motion_high_alignment")
        # Safely access nested dicts
        align_mlc = alignments.get("motion_low_by_culture", {})
        averaged_results["motion_low_by_culture_mean"] = {c: align_mlc.get(c) for c in range(num_cultures)}
        align_mhc = alignments.get("motion_high_by_culture", {})
        averaged_results["motion_high_by_culture_mean"] = {c: align_mhc.get(c) for c in range(num_cultures)}
        # (add shuffled alignments similarly if needed)
        averaged_results["overall_motion_low_shuffled_alignment_mean"] = alignments.get("overall_motion_low_shuffled_alignment")
        averaged_results["overall_motion_high_shuffled_alignment_mean"] = alignments.get("overall_motion_high_shuffled_alignment")
        low_gap = None
        high_gap = None
        if averaged_results["overall_motion_low_alignment_mean"] is not None and averaged_results["overall_motion_low_shuffled_alignment_mean"] is not None:
            low_gap = averaged_results["overall_motion_low_alignment_mean"] - averaged_results["overall_motion_low_shuffled_alignment_mean"]
        if averaged_results["overall_motion_high_alignment_mean"] is not None and averaged_results["overall_motion_high_shuffled_alignment_mean"] is not None:
            high_gap = averaged_results["overall_motion_high_alignment_mean"] - averaged_results["overall_motion_high_shuffled_alignment_mean"]
        averaged_results["overall_motion_low_alignment_gap_mean"] = low_gap
        averaged_results["overall_motion_high_alignment_gap_mean"] = high_gap
        align_mlsc = alignments.get("motion_low_shuffled_by_culture", {})
        averaged_results["motion_low_shuffled_by_culture_mean"] = {c: align_mlsc.get(c) for c in range(num_cultures)}
        align_mhsc = alignments.get("motion_high_shuffled_by_culture", {})
        averaged_results["motion_high_shuffled_by_culture_mean"] = {c: align_mhsc.get(c) for c in range(num_cultures)}


    # 6. Classification Results (explicitly separated)
    averaged_results["culture_classification_diffusion_head"] = classification_diffusion_head or {}
    averaged_results["culture_classification_external_fishr"] = classification_external_fishr or {}
    # Backward-compatible flat keys expected by some downstream scripts.
    if classification_external_fishr and classification_external_fishr.get("source") == "external_classifier":
        averaged_results["generated"] = classification_external_fishr.get("generated")
        averaged_results["real"] = classification_external_fishr.get("real")
        averaged_results["source"] = classification_external_fishr.get("source")
    else:
        averaged_results["generated"] = (classification_diffusion_head or {}).get("generated")
        averaged_results["real"] = (classification_diffusion_head or {}).get("real")
        averaged_results["source"] = (classification_diffusion_head or {}).get("source")

    # 7. SRGR Results (NEW)
    averaged_results["srgr_accuracy_overall_mean"] = srgr_res.get("srgr_accuracy_overall")
    # Safely access nested dicts
    srgr_pc = srgr_res.get("srgr_accuracy_per_culture", {})
    averaged_results["srgr_accuracy_per_culture_mean"] = {c: srgr_pc.get(c) for c in range(num_cultures)}

    # 8. Beat Alignment Results (NEW)
    averaged_results["beat_align_real_overall_mean"] = beat_align_res.get("beat_align_real_overall")
    averaged_results["beat_align_gen_overall_mean"] = beat_align_res.get("beat_align_gen_overall")
    # Safely access nested dicts
    ba_rpc = beat_align_res.get("beat_align_real_per_culture", {})
    averaged_results["beat_align_real_per_culture_mean"] = {c: ba_rpc.get(c) for c in range(num_cultures)}
    ba_gpc = beat_align_res.get("beat_align_gen_per_culture", {})
    averaged_results["beat_align_gen_per_culture_mean"] = {c: ba_gpc.get(c) for c in range(num_cultures)}
    # BAS aliases (Ai Choreographer terminology).
    averaged_results["bas_real_overall_mean"] = averaged_results["beat_align_real_overall_mean"]
    averaged_results["bas_gen_overall_mean"] = averaged_results["beat_align_gen_overall_mean"]
    averaged_results["bas_real_per_culture_mean"] = averaged_results["beat_align_real_per_culture_mean"]
    averaged_results["bas_gen_per_culture_mean"] = averaged_results["beat_align_gen_per_culture_mean"]

    # --- Prepare Final Results ---
    # Store raw results as well if needed
    all_metric_results_raw = {
         #"r_match": r_match_res,
         "l2": l2_res,
         "l1": l1_res,
         "fgd": fgd_res,
         "diversity": diversity_res,
         #"alignment": alignments,
         "classification_diffusion_head": classification_diffusion_head,
         "classification_external_fishr": classification_external_fishr,
         "srgr": srgr_res,
         "beat_align": beat_align_res
    }
    final_results = {
        "averaged_metrics": averaged_results,
        # "raw_metrics": all_metric_results_raw # Optional: include raw results
    }

    # --- Print Results ---
    print("\n===== Averaged Metrics =====")
    for key, value in averaged_results.items():
         # Pretty print for nested dicts
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  Culture {sub_key}: {sub_value}")
        else:
             print(f"{key}: {value}")

    # --- Log Results ---
    print(f"\nLogging results to {log_file}...")
    try:
        with open(log_file, 'w') as f:
            f.write("===== Averaged Metrics =====\n")
            for key, value in averaged_results.items():
                 if isinstance(value, dict):
                     f.write(f"{key}:\n")
                     for sub_key, sub_value in value.items():
                         f.write(f"  Culture {sub_key}: {sub_value}\n")
                 else:
                     f.write(f"{key}: {value}\n")
    except IOError as e:
        print(f"Error writing log file {log_file}: {e}")


    # --- Save All Metrics to Pickle ---
    # Consolidate all collected data (optional)
    all_collected_data = {
         "real_3d_poses_all": real_3d_poses_all,
         "real_final_motion_all": real_final_motion_all,
         #"real_culture_output_all": real_culture_output_all,
         #"real_motion_pooled_all": real_motion_pooled_all,
         #"real_high_level_context_all": real_high_level_context_all,
         #"real_low_level_context_all": real_low_level_context_all,
         "real_cont_motion_all": real_cont_motion_all,
         "real_culture_labels_all": real_culture_labels_all,
         "real_speaker_labels_all": real_speaker_labels_all,
         "real_pose_timestamps_all": real_pose_timestamps_all,
         "real_audio_onset_s_all": real_audio_onset_s_all,
         "real_audio_onset_t_all": real_audio_onset_t_all,

         "gen_3d_poses_all": gen_3d_poses_all,
         "gen_final_motion_all": gen_final_motion_all,
         #"gen_culture_output_all": gen_culture_output_all,
         #"gen_motion_pooled_all": gen_motion_pooled_all,
         #"gen_high_level_context_all": gen_high_level_context_all,
         #"gen_low_level_context_all": gen_low_level_context_all,
         "gen_cont_motion_all": gen_cont_motion_all,
         "gen_culture_labels_all": gen_culture_labels_all,
         "gen_speaker_labels_all": gen_speaker_labels_all,
         "gen_audio_onset_s_all": gen_audio_onset_s_all,
         "gen_audio_onset_t_all": gen_audio_onset_t_all,

         "metrics": final_results # Store averaged and potentially raw results
    }

    if save_raw_data and save_path: # Only save if explicitly requested
        pickle_file_path = "evaluation_results_bulk.pkl"
        pickle_save_path = os.path.join(save_path, pickle_file_path)
        print(f"Saving all data and metrics to {pickle_save_path}...")
        try:
            os.makedirs(save_path, exist_ok=True) # Ensure directory exists
            with open(pickle_save_path, "wb") as pickle_file:
                pickle.dump(all_collected_data, pickle_file)
        except Exception as e:
            print(f"Error saving pickle file to {pickle_save_path}: {e}")
    else:
        print("Skipping raw sample pickle save.")

    print("Bulk evaluation complete.")
    return final_results

# === Example of usage ===
# Assuming you already have real_loader and gen_loader defined:
#
# results = evaluate_all_metrics(real_loader, gen_loader)
#
# Now, `results` is a dictionary containing the averaged metrics and the function
# also prints the results to the standard output.
