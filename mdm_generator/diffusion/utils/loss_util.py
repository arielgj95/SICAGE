from mdm_generator.diffusion.nn import mean_flat, sum_flat
import torch
from torch.nn import functional as F
import numpy as np
import torch.nn as nn

def angle_l2(angle1, angle2):
    a = angle1 - angle2
    a = (a + (torch.pi/2)) % torch.pi - (torch.pi/2)
    return a ** 2

def diff_l2(a, b):
    return (a - b) ** 2

def masked_l2(a, b, mask, loss_fn=diff_l2, epsilon=1e-8, entries_norm=True):
    # assuming a.shape == b.shape == bs, J, Jdim, seqlen
    # assuming mask.shape == bs, 1, 1, seqlen
    loss = loss_fn(a, b)
    loss = sum_flat(loss * mask.float())  # gives \sigma_euclidean over unmasked elements
    n_entries = a.shape[1]
    if len(a.shape) > 3:
        n_entries *= a.shape[2]
    non_zero_elements = sum_flat(mask)
    if entries_norm:
        # In cases the mask is per frame, and not specifying the number of entries per frame, this normalization is needed,
        # Otherwise set it to False
        non_zero_elements *= n_entries
    # print('mask', mask.shape)
    # print('non_zero_elements', non_zero_elements)
    # print('loss', loss)
    mse_loss_val = loss / (non_zero_elements + epsilon)  # Add epsilon to avoid division by zero
    # print('mse_loss_val', mse_loss_val)
    return mse_loss_val

class MMDLoss(nn.Module):
    def __init__(self, kernel='rbf', sigma=1.0):
        """
        Args:
            kernel (str): Type of kernel to use ('rbf' supported).
            sigma (float): Bandwidth parameter for the RBF kernel.
        """
        super(MMDLoss, self).__init__()
        self.kernel = kernel
        self.sigma = sigma

    def forward(self, x, y):
        """
        Compute the MMD loss between two batches of samples.

        Args:
            x (Tensor): Generated motion samples [batch_size * seq_len, dim].
            y (Tensor): Real motion samples [batch_size * seq_len, dim].

        Returns:
            Tensor: Scalar MMD loss.
        """
        if self.kernel == 'rbf':
            return self.compute_rbf_mmd(x, y)
        else:
            raise NotImplementedError('Only RBF kernel is supported for MMDLoss.')

    def compute_rbf_mmd(self, x, y):
        """
        Compute the RBF (Gaussian) MMD between two sets of samples.

        Args:
            x (Tensor): Generated samples [N, D].
            y (Tensor): Real samples [M, D].

        Returns:
            Tensor: Scalar MMD loss.
        """
        xx, yy, zz = self._rbf_kernel(x, y)
        mmd = xx.mean() + yy.mean() - 2 * zz.mean()
        return mmd

    def _rbf_kernel(self, x, y):
        """
        Compute the RBF kernel between samples in x and y.

        Args:
            x (Tensor): [N, D]
            y (Tensor): [M, D]

        Returns:
            Tuple[Tensor, Tensor, Tensor]: K(x,x), K(y,y), K(x,y)
        """
        gamma = 1.0 / (2 * self.sigma ** 2)

        # Compute pairwise squared Euclidean distances
        x_sq = torch.sum(x ** 2, dim=1).unsqueeze(1)  # [N, 1]
        y_sq = torch.sum(y ** 2, dim=1).unsqueeze(1)  # [M, 1]

        dist_xx = x_sq + x_sq.t() - 2 * torch.matmul(x, x.t())  # [N, N]
        dist_yy = y_sq + y_sq.t() - 2 * torch.matmul(y, y.t())  # [M, M]
        dist_xy = x_sq + y_sq.t() - 2 * torch.matmul(x, y.t())  # [N, M]

        # Compute RBF kernels
        K_xx = torch.exp(-gamma * dist_xx)
        K_yy = torch.exp(-gamma * dist_yy)
        K_xy = torch.exp(-gamma * dist_xy)

        return K_xx, K_yy, K_xy


def contrastive_loss_fn(anchor, positive, temp):
    # Normalize embeddings.
    anchor_norm = F.normalize(anchor, p=2, dim=-1)
    positive_norm = F.normalize(positive, p=2, dim=-1)
    # Compute similarity matrix.
    sim_matrix = torch.matmul(anchor_norm, positive_norm.T)  # [batch_size, batch_size]
    logits = sim_matrix / temp
    labels = torch.arange(anchor.shape[0]).to(anchor.device)
    return F.cross_entropy(logits, labels)


def _variance_regularization(x: torch.Tensor, target_std: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    """VICReg-style variance floor to discourage embedding collapse."""
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if x.shape[0] < 2:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    std = torch.sqrt(x.var(dim=0, unbiased=False) + eps)
    return torch.relu(target_std - std).mean()


def _to_2d_embedding(x: torch.Tensor) -> torch.Tensor:
    """Convert [B, D] or [B, T, D] tensors to [B, D] embeddings."""
    if x.dim() == 2:
        return x
    if x.dim() == 3:
        return x.mean(dim=1)
    raise ValueError(f"Expected 2D or 3D tensor for embedding, got shape {tuple(x.shape)}")


def gram_alignment_loss(
    motion_embed: torch.Tensor,
    low_level_embed: torch.Tensor,
    high_level_embed: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    GRAM-style alignment loss.

    For each sample, create a 3x3 Gram matrix from normalized motion/low/high vectors.
    Minimizing the simplex volume (sqrt(det(G + eps*I))) pulls modalities together.
    """
    motion_embed = F.normalize(_to_2d_embedding(motion_embed), p=2, dim=-1)
    low_level_embed = F.normalize(_to_2d_embedding(low_level_embed), p=2, dim=-1)
    high_level_embed = F.normalize(_to_2d_embedding(high_level_embed), p=2, dim=-1)

    stacked = torch.stack([motion_embed, low_level_embed, high_level_embed], dim=1)  # [B, 3, D]
    gram = torch.bmm(stacked, stacked.transpose(1, 2))  # [B, 3, 3]
    eye = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype).unsqueeze(0)
    gram = gram + eps * eye

    det = torch.linalg.det(gram)
    volume = torch.sqrt(torch.clamp(det, min=0.0) + eps)
    return volume.mean()


# TODO this version assumes motion_output_projection = motion_output and latent_dim = vqvae_dim. See hiearchical_MDM comments for more info
def compute_losses(
        motion_output = None,
        culture_output = None,
        motion_output_projection_pooled = None,
        real_motion = None,
        culture_labels = None,
        low_level_context = None,
        high_level_context = None,
        motion_mask = None,
        audio_mask = None,
        use_diversity_loss = None,
        use_contrastive_loss = None,
        temperature = 0.7,
        use_gram_loss: bool = False,
        gram_eps: float = 1e-6,
        use_alignment_loss: bool = True,
        use_culture_loss: bool = True,
):
    # TODO NOTE THAT AUDIO HAS BEEN CHANGED WITH COLLAPSED VERSION (512)
    """
    Compute four losses:
    1) Reconstruction loss for the non-masked motion frames.
    2) Cosine similarity loss between non-masked audio features and output motion features.
    3) Cosine alignment between high-level context and output motion features. No masks are applied here
    4) Crossentropy loss for culture classification.

    Args:
        motion_output: Predicted motion [batch_size, seq_len-motion_prefix_len, vqvae_dim]
        culture_output: Predicted culture logits [batch_size, n_cultures]
        motion_output_projection_pooled: Motion pooled representation [batch_size, latent_dim] for high-level alignment
        real_motion: Ground truth motion [batch_size, seq_len-motion_prefix_len, vqvae_dim]
        culture_labels: Ground truth culture labels [batch_size]
        low_level_context: Audio pooled representation [batch_size, seq_len-motion_predix_len, latent_dim] for low-level alignment
        high_level_context: High-level context embedding [batch_size, latent_dim]
        motion_mask: Boolean mask for motion [batch_size, seq_len-motion_prefix_len, latent_dim], True=masked
        audio_mask: Boolean mask for audio [batch_size, seq_len-motion_prefix_len, latent_dim], True=masked
        use_mmd_loss: Boolean for computing mmd loss on unmasked motion frames to ensure similar distributions and diversity

    Returns:
        A dictionary with the four computed losses.
    """

    # 1) Reconstruction loss on unmasked frames
    # motion_mask is [batch_size, seq_len, latent_dim]. We use a frame-level mask, so let's reduce it over dim. This means
    # That if elements of that frame are masked, the frame is masked, if they are not, the frame is not masked.
    # For reconstruction, consider non-masked elements for loss computation. A frame is masked if any feature is masked,
    # so this works also for feature level masking.
    frame_mask_motion = motion_mask.any(dim=2)  # [batch_size, seq_len]
    # We'll compute MSE only on these non-masked frames.
    # Mask out non-masked frames
    unmasked_pred = motion_output[~frame_mask_motion]  # predicted values where masked, [num_unmasked_frames among [batch, seq_len-motion_prefix_len], vqvae_dim]
    unmasked_real = real_motion[~frame_mask_motion]  # real values where masked [num_unmasked_frames among [batch, seq_len-motion_prefix_len], vqvae_dim]
    #print("SHAPE unmasked_pred", unmasked_pred.shape, "SHAPE unmasked_real", unmasked_real.shape,"FRAMES MASK MOTION", frame_mask_motion.shape)
    #print("PRED",unmasked_pred[0])
    #print("REAL",unmasked_real[0])
    #print("FRAMES MASK",frame_mask_motion)


    if unmasked_pred.numel() > 0:
        #recon_loss_unmasked = F.mse_loss(unmasked_pred, unmasked_real)
        recon_loss_unmasked = F.smooth_l1_loss(unmasked_pred, unmasked_real, reduction='none')
        total_loss = recon_loss_unmasked.sum()
        recon_loss_unmasked = total_loss / unmasked_pred.numel()
    else:
        # If no frames are masked (unlikely), just set to zero or handle gracefully
        recon_loss_unmasked = torch.tensor(0.0, device=motion_output.device)

    # 2) Reconstruction loss on masked frames
    masked_pred = motion_output[frame_mask_motion]  # [num_masked_frames among [batch, seq_len-motion_prefix_len], vqvae_dim]
    masked_real = real_motion[frame_mask_motion]    # [num_masked_frames among [batch, seq_len-motion_prefix_len], vqvae_dim]

    if masked_pred.numel() > 0:
        #recon_loss_masked  = F.mse_loss(masked_pred, masked_real)
        recon_loss_masked = F.smooth_l1_loss(masked_pred, masked_real, reduction='none')
        total_loss = recon_loss_masked.sum()
        recon_loss_masked = total_loss / masked_pred.numel()
    else:
        recon_loss_masked  = torch.tensor(0.0, device=motion_output.device)

    # 3) Cosine similarity loss between masked audio features and motion features
    # audio_context_pooled (low_level_context) and motion_output_projection are both [batch_size, 25, latent_dim]
    # audio_mask: similarly reduce to a frame-level mask
    frame_mask_audio = audio_mask.any(dim=2)  # [batch_size, seq_len]

    # Ensure that motion_output and low_level_context have the same seq_len (25)
    # The model should have ensured that already. We'll assume seq_len=25 for both.
    # We'll only consider the masked portion. Let's intersect masked frames for both audio and motion.
    # Typically, you'd want to align masked frames of motion and audio. If the noise portion is known (e.g. after frame 5),
    # you can focus on that. For simplicity, let's just use motion_mask to select frames from both.
    # In practice, you'd decide which mask to use or use both. Let's assume we want alignment on motion-masked frames.
    # TODO masked_mot_proj is not right! I remove masked audio frames but there can be also masked motion frames! I should remove both masked frames. At the moment, it works without masks
    # TODO NOTE THAT NOW AUDIO IS THE COLLAPSED VERSION OF AUDIO

    '''
    masked_mot_proj = motion_output[~frame_mask_audio]  # [num_unmasked_frames among [batch, seq_len-motion_prefix_len], vqvae_dim]
    masked_aud_proj = low_level_context[~frame_mask_audio]  # [num_unmasked_frames among [batch, seq_len-motion_prefix_len], vqvae_dim]

    if masked_mot_proj.numel() > 0 and masked_aud_proj.numel() > 0:
        # Compute cosine similarity
        # cos_sim = F.cosine_similarity(x1, x2, dim=-1)
        cos_sim_ll_loss = F.cosine_similarity(masked_mot_proj, masked_aud_proj, dim=-1)
        # We want them to be similar, so loss = 1 - mean(cos_sim)
        low_level_alignment_loss = 1.0 - cos_sim_ll_loss.mean()
    else:
        low_level_alignment_loss = torch.tensor(0.0, device=motion_output.device)
    '''
    if not use_alignment_loss:
        low_level_alignment_loss = torch.tensor(0.0, device=motion_output.device)
        high_level_alignment_loss = torch.tensor(0.0, device=motion_output.device)
        gram_loss = torch.tensor(0.0, device=motion_output.device)
    elif use_gram_loss:
        low_level_alignment_loss = torch.tensor(0.0, device=motion_output.device)
        high_level_alignment_loss = torch.tensor(0.0, device=motion_output.device)
        gram_loss = gram_alignment_loss(
            motion_output_projection_pooled,
            low_level_context,
            high_level_context,
            eps=gram_eps,
        )
    else:
        cos_sim_ll_loss = F.cosine_similarity(motion_output_projection_pooled, low_level_context, dim=-1)
        low_level_alignment_loss = 1.0 - cos_sim_ll_loss.mean()
        # 4) Cosine alignment between high-level context and motion projected output (pooled)
        # motion_output_projection_pooled: [batch_size, latent_dim]
        # high_level_context: [batch_size, latent_dim]
        # Loss = 1 - mean(cosine_similarity)
        cos_sim_hl_loss = F.cosine_similarity(motion_output_projection_pooled, high_level_context, dim=-1)
        high_level_alignment_loss = 1.0 - cos_sim_hl_loss.mean()
        gram_loss = torch.tensor(0.0, device=motion_output.device)

    # 5) Crossentropy loss for culture classification
    if use_culture_loss:
        culture_loss = F.cross_entropy(culture_output, culture_labels)
    else:
        culture_loss = torch.tensor(0.0, device=motion_output.device)


    # 6) diversity loss for motion diversity
    if use_diversity_loss and unmasked_real is not None:
        # Compute means and variances for real and predicted motion
        real_mean, real_var = torch.mean(unmasked_real, dim=1), torch.var(unmasked_real, dim=1)
        pred_mean, pred_var = torch.mean(unmasked_pred, dim=1), torch.var(unmasked_pred, dim=1)

        # Ensure numerical stability
        epsilon = 1e-6
        real_var = real_var + epsilon
        pred_var = pred_var + epsilon

        # Compute KL divergence between the distributions
        kl_div = torch.log(pred_var / real_var) + \
                 (real_var + (real_mean - pred_mean) ** 2) / (2 * pred_var) - 0.5

        diversity_loss = kl_div.mean()  # Mean over batch
    else:
        diversity_loss = torch.tensor(0.0, device=motion_output.device)


    if not use_alignment_loss:
        contrastive_loss = torch.tensor(0.0, device=motion_output.device)
    elif use_gram_loss:
        # Explicitly disable contrastive loss in GRAM mode.
        contrastive_loss = torch.tensor(0.0, device=motion_output.device)
    elif use_contrastive_loss:
        # Contrast motion with audio (low-level).
        contrastive_loss_audio = contrastive_loss_fn(motion_output_projection_pooled, low_level_context, temperature)
        # Contrast motion with high-level context.
        contrastive_loss_high = contrastive_loss_fn(motion_output_projection_pooled, high_level_context, temperature)
        # Average the two contrastive losses.
        contrastive_loss = (contrastive_loss_audio + contrastive_loss_high) / 2.0
    else:
        contrastive_loss = torch.tensor(0.0, device=motion_output.device)

    # 7) Collapse regularization on latent spaces used by alignment losses.
    if use_alignment_loss:
        collapse_reg_loss = (
            _variance_regularization(motion_output_projection_pooled) +
            _variance_regularization(low_level_context) +
            _variance_regularization(high_level_context)
        ) / 3.0
    else:
        collapse_reg_loss = torch.tensor(0.0, device=motion_output.device)

    losses = {
        'reconstruction_loss_masked': recon_loss_masked,
        'reconstruction_loss': recon_loss_unmasked,
        'low_level_alignment_loss': low_level_alignment_loss,
        'high_level_alignment_loss': high_level_alignment_loss,
        'gram_loss': gram_loss,
        'culture_loss': culture_loss,
        'diversity_loss': diversity_loss,
        'contrastive_loss': contrastive_loss,
        'collapse_reg_loss': collapse_reg_loss
    }

    return losses

def masked_goal_l2(pred_goal, ref_goal, cond, all_goal_joint_names):
    all_goal_joint_names_w_traj = np.append(all_goal_joint_names, 'traj')
    target_joint_idx = [[np.where(all_goal_joint_names_w_traj == j)[0][0] for j in sample_joints] for sample_joints in cond['target_joint_names']]
    loc_mask = torch.zeros_like(pred_goal[:,:-1], dtype=torch.bool)
    for sample_idx in range(loc_mask.shape[0]):
        loc_mask[sample_idx, target_joint_idx[sample_idx]] = True
    loc_mask[:, -1, 1] = False  # vertical joint of 'traj' is always masked out
    loc_loss = masked_l2(pred_goal[:,:-1], ref_goal[:,:-1], loc_mask, entries_norm=False)

    heading_loss = masked_l2(pred_goal[:,-1:, :1], ref_goal[:,-1:, :1], cond['is_heading'].unsqueeze(1).unsqueeze(1), loss_fn=angle_l2, entries_norm=False)

    loss =  loc_loss + heading_loss
    return loss
