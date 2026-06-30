import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import random
import numpy as np
from collections import defaultdict, OrderedDict
import torch.autograd as autograd
from torch.autograd import Variable
from tqdm.auto import tqdm
import math
import logging
import glob
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score, accuracy_score, confusion_matrix
import logging

# Configure logging
logging.basicConfig(
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)



try:
    from backpack import backpack, extend
    from backpack.extensions import BatchGrad
except Exception:
    backpack = None
    extend = None
    BatchGrad = None


#from torch.func import grad, vmap, functional_call
import os
import pickle
from types import SimpleNamespace

from culture_encoder.adversarial_classifier import (
    AttentionPooling as AdvAttentionPooling,
    LinearEncoder as AdvLinearEncoder,
    TransformerEncoder as AdvTransformerEncoder,
    calculate_input_dim as adv_calculate_input_dim,
    map_inputs as adv_map_inputs,
)

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def _normalize_encoder_type(value):
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip()
        if normalized.lower() in {"", "none", "null"}:
            return None
        return normalized
    return value


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, lambda_):
        ctx.lambda_ = lambda_
        return input_tensor.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x, lambda_=1.0):
    return GradientReversalFunction.apply(x, lambda_)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout_layer = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        '''
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term)[:, :-1]
        '''

        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size,seq_length, d_model)
        seq_length = x.size(1)
        x = x + self.pe[:, :seq_length]
        #x = x + self.pe[:seq_length, :]
        return self.dropout_layer(x)

class TransformerEncoder(nn.Module):
    def __init__(self, d_model=512, output_dim=1024, nhead=8, input_dim = 512,
                 num_layers=2, dim_feedforward=2048, dropout=0.1, max_seq_length=5000):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.input_projection_layer = nn.Linear(input_dim, d_model)
        self.input_dropout = nn.Dropout(dropout)  # Dropout after input projection
        self.layer_norm = nn.LayerNorm(d_model)
        self.attention_pool = AttentionPooling(d_model)  # Attention-based pooling
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_length, dropout = dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, )
        self.gelu= nn.GELU()

        # Add a projection layer to map to the desired output dimension
        self.output_projection_layer = nn.Linear(d_model, output_dim)
        self.output_dropout = nn.Dropout(dropout)  # Dropout after output projection

    def forward(self, data):
        # poses: (batch_size, seq_length, feature_dim)
        data = self.input_projection_layer(data)
        data = self.input_dropout(data)
        data = self.pos_encoder(data)
        data = data.permute(1, 0, 2)  # (seq_length, batch_size, d_model)
        transformer_output = self.transformer_encoder(data)  # (seq_length, batch_size, d_model)
        transformer_output = self.layer_norm(transformer_output)
        transformer_output = transformer_output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        #pooled_output = self.attention_pool(transformer_output)  # (batch_size, d_model)
        #transformer_output = transformer_output.mean(dim=0)  # (batch_size, d_model)
        projected_output = self.output_projection_layer(transformer_output)  # Map to (batch_size, output_dim)
        projected_output = self.gelu(projected_output)
        projected_output = self.output_dropout(projected_output)
        return projected_output


def load_last_checkpoint(model, save_dir):
    """
    Look for checkpoint files in save_dir with the pattern 'model_epoch_{epoch}.pt'
    and load the checkpoint with the highest epoch number into the model.
    Returns the next epoch number to resume training from.
    """
    checkpoint_paths = glob.glob(os.path.join(save_dir, "model_epoch_*.pt"))

    if checkpoint_paths:
        def extract_epoch(path):
            base = os.path.basename(path)
            try:
                epoch_num = int(base.split('_')[-1].split('.')[0])
            except ValueError:
                epoch_num = -1
            return epoch_num

        last_checkpoint = max(checkpoint_paths, key=extract_epoch)
        last_epoch = extract_epoch(last_checkpoint)
        print(f"Loading last checkpoint from {last_checkpoint} (epoch {last_epoch})")
        model.load_state_dict(torch.load(last_checkpoint, map_location=model.device))
        # Return the next epoch number (i.e., last_epoch + 1) for resuming training.
        return last_epoch + 1
    else:
        print("No checkpoint found. Training from scratch.")
        return 0

class MultiModalFeaturizer(nn.Module):
    """
    Expects a dictionary input with keys:
      - 'wav2vec': tensor of shape (B, 50, 1024)
      - 'mel': tensor of shape (B, 156, 64)
      - 'onset': tensor of shape (B, 156)
      - 'sentence': tensor of shape (B, 768)
    It projects each modality to a common dimension (proj_dim) and applies attention pooling
    to the temporally varying modalities, then concatenates the outputs.
    """
    def __init__(self, proj_dim, wav2vec_dim=1024, mel_dim=64, onset_dim=1, sentence_dim=768):
        super(MultiModalFeaturizer, self).__init__()
        self.proj_dim = proj_dim
        self.activation = nn.GELU()

        # For wav2vec (temporal modality)
        self.wav2vec_proj = nn.Sequential(nn.Linear(wav2vec_dim, proj_dim),
                                                 nn.LayerNorm(self.proj_dim),
                                                 self.activation,
                                                 nn.Dropout(0.1))


        self.wav2vec_pool = AttentionPooling(proj_dim)

        # For mel-log and onsets: first concatenate along channel dimension.
        self.mel_onset_proj = nn.Sequential(nn.Linear(mel_dim + onset_dim, proj_dim),
                      nn.LayerNorm(self.proj_dim),
                      self.activation,
                      nn.Dropout(0.1))
        self.mel_onset_pool = AttentionPooling(proj_dim)

        # For sentence embeddings (static)
        self.sentence_proj = nn.Sequential(nn.Linear(sentence_dim, proj_dim),
                      nn.LayerNorm(self.proj_dim),
                      self.activation,
                      nn.Dropout(0.1))

    def forward(self, features):
        # Process wav2vec: (B, 50, 1024)
        wav2vec = features["wav2vec"]
        wav2vec_proj = self.wav2vec_proj(wav2vec)  # (B, 50, proj_dim)
        wav2vec_feat = self.wav2vec_pool(wav2vec_proj)  # (B, proj_dim)

        # Process mel and onset: mel (B, 156, 64), onset (B, 156)
        mel = features["mel"]
        onset = features["onset"]
        if onset.dim() == 2: # if onset is (B, 156). If it is already (B, 156, 1) it is not necessary to unsqueeze
            onset = onset.unsqueeze(-1)
        mel_onset = torch.cat([mel, onset], dim=-1)  # (B, 156, 65)
        mel_onset_proj = self.mel_onset_proj(mel_onset)  # (B, 156, proj_dim)
        mel_onset_feat = self.mel_onset_pool(mel_onset_proj)  # (B, proj_dim)

        # Process sentence embeddings: (B, 768)
        sentence = features["sentence"]
        sentence_feat = self.sentence_proj(sentence)  # (B, proj_dim)

        # Concatenate all features: (B, 3*proj_dim)
        combined = torch.cat([wav2vec_feat, mel_onset_feat, sentence_feat], dim=1)
        return combined


class MotionFeaturizer(nn.Module):

    def __init__(self, proj_dim, motion_dim = 512):
        super(MotionFeaturizer, self).__init__()
        self.proj_dim = proj_dim
        self.activation = nn.GELU()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # For wav2vec (temporal modality)
        self.pose_encoder = TransformerEncoder(d_model=motion_dim, input_dim=motion_dim, output_dim=motion_dim, dropout=.1,
                                               nhead=8, num_layers=2, max_seq_length=20, dim_feedforward=motion_dim).to(self.device)

        self.motion_proj = nn.Sequential(nn.Linear(motion_dim, proj_dim),
                                                 nn.LayerNorm(self.proj_dim),
                                                 self.activation,
                                                 nn.Dropout(0.1))


        self.motion_pool = AttentionPooling(proj_dim)


    def forward(self, motion):

        #motion_proj = self.motion_proj(motion)
        motion_proj = self.pose_encoder(motion)
        motion_feat = self.motion_pool(motion_proj)

        return motion_feat #512





class AdversarialAlignedFeaturizer(nn.Module):
    """
    Fishr featurizer aligned with Culture_Classifier.encode() so projections and
    embedding dimensionality match adversarial training.
    """

    def __init__(
        self,
        config,
        device,
        cl_type: str = "culclI",
        d_model: int = 512,
        pose_enc_type="transformer",
        audio_enc_type=None,
        raw_poses: bool = False,
    ):
        super().__init__()
        self.device = device
        self.cl_type = cl_type
        self.d_model = int(d_model)
        self.dropout = float(getattr(config, "dropout", 0.1))
        self.layer_neurons = int(getattr(config, "layer_neurons", self.d_model))
        self.n_levels = int(getattr(config, "levels", 1))
        self.embed_dim = int(getattr(config, "embed_dim", self.layer_neurons))
        self.pose_enc_type = _normalize_encoder_type(pose_enc_type)
        self.audio_enc_type = _normalize_encoder_type(audio_enc_type)

        self.pose_encoder = None
        if self.pose_enc_type == "transformer":
            pose_input_dim = 54 if raw_poses else 512
            max_seq_length = 75 if raw_poses else 25
            self.pose_encoder = AdvTransformerEncoder(
                d_model=self.d_model,
                input_dim=pose_input_dim,
                output_dim=self.d_model,
                dropout=self.dropout,
                nhead=4,
                num_layers=2,
                max_seq_length=max_seq_length,
                dim_feedforward=self.d_model,
            )
        elif self.pose_enc_type == "linear":
            pose_input_dim = 54 if raw_poses else 512
            self.pose_encoder = AdvLinearEncoder(
                pose_input_dim,
                d_model=self.d_model,
                output_dim=self.d_model,
                dropout=self.dropout,
            )

        self.sentence_mapper = None
        if cl_type in ["culclB", "culclF", "culclH", "culclI"]:
            self.sentence_mapper = AdvLinearEncoder(
                768, d_model=self.d_model, output_dim=self.d_model, dropout=self.dropout
            )

        self.audio_encoders = nn.ModuleDict()
        if cl_type in ["culclC", "culclD", "culclE", "culclG", "culclH", "culclI", "culclK", "culclL"]:
            self.audio_encoders["mels"] = AdvLinearEncoder(
                64, d_model=self.d_model, output_dim=self.d_model // 2, dropout=self.dropout
            )
            self.audio_encoders["onsets"] = AdvLinearEncoder(
                1, d_model=self.d_model, output_dim=self.d_model // 2, dropout=self.dropout
            )
            self.audio_encoders["wav2vec"] = AdvLinearEncoder(
                1024, d_model=self.d_model, output_dim=self.d_model, dropout=self.dropout
            )

        if self.audio_enc_type == "transformer":
            self.audio_encoders["trans"] = AdvTransformerEncoder(
                d_model=self.d_model,
                input_dim=self.d_model,
                output_dim=self.d_model,
                dropout=self.dropout,
                nhead=4,
                num_layers=2,
                max_seq_length=156,
            )

        self.attention_poolers = nn.ModuleDict()
        if cl_type in ["culclA", "culclB", "culclC", "culclD", "culclE", "culclH", "culclJ"]:
            self.attention_poolers["poses"] = AdvAttentionPooling(d_model=self.d_model)
        if cl_type in ["culclC", "culclE", "culclG", "culclH", "culclI", "culclL"]:
            self.attention_poolers["mels"] = AdvAttentionPooling(d_model=self.d_model // 2)
            self.attention_poolers["onsets"] = AdvAttentionPooling(d_model=self.d_model // 2)
            self.attention_poolers["audio"] = AdvAttentionPooling(d_model=self.d_model)
        if cl_type in ["culclD", "culclE", "culclG", "culclH", "culclI", "culclK"]:
            self.attention_poolers["wav2vec"] = AdvAttentionPooling(d_model=self.d_model)

        input_dim = adv_calculate_input_dim(cl_type, self.d_model)
        fc_layers = []
        for i in range(self.n_levels):
            in_features = input_dim if i == 0 else self.layer_neurons
            fc_layers.extend(
                [
                    nn.Linear(in_features, self.layer_neurons),
                    nn.LayerNorm(self.layer_neurons),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                ]
            )
        self.fc_layers = nn.Sequential(*fc_layers)
        self.embedding_head = nn.Sequential(
            nn.Linear(self.layer_neurons, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.output_dim = self.embed_dim

    def forward(self, inputs):
        h = adv_map_inputs(
            self.cl_type,
            inputs,
            self.device,
            self.pose_encoder,
            self.pose_enc_type,
            self.audio_encoders,
            self.audio_enc_type,
            self.sentence_mapper,
            self.attention_poolers,
        )
        h = self.fc_layers(h)
        z = self.embedding_head(h)
        return F.normalize(z, p=2, dim=1)


class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (batch_size, seq_length, d_model)
        scores = self.attention(x)  # (batch_size, seq_length, 1)
        weights = F.softmax(scores, dim=1)  # (batch_size, seq_length, 1)
        pooled = torch.sum(x * weights, dim=1)  # (batch_size, d_model)
        return pooled




class MovingAverage:
    def __init__(self, ema, oneminusema_correction=True):
        self.ema = ema
        self.named_parameters = {}
        self._updates = 0
        self._oneminusema_correction = oneminusema_correction

    def update(self, dict_data):
        ema_dict_data = {}
        for name, data in dict_data.items():
            data = data.view(1, -1)
            if self._updates == 0:
                previous_data = torch.zeros_like(data)
            else:
                previous_data = self.named_parameters[name]

            ema_data = self.ema * previous_data + (1 - self.ema) * data
            if self._oneminusema_correction:
                ema_dict_data[name] = ema_data / (1 - self.ema)
            else:
                ema_dict_data[name] = ema_data
            self.named_parameters[name] = ema_data.clone().detach()

        self._updates += 1
        return ema_dict_data





def l2_between_dicts(dict_1, dict_2):
    assert len(dict_1) == len(dict_2)
    dict_1_values = [dict_1[key] for key in sorted(dict_1.keys())]
    dict_2_values = [dict_2[key] for key in sorted(dict_1.keys())]
    return (
        torch.cat(tuple([t.view(-1) for t in dict_1_values])) -
        torch.cat(tuple([t.view(-1) for t in dict_2_values]))
    ).pow(2).mean()





def Classifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.GELU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.GELU(),
            torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features)




class Algorithm(nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """

    def __init__(self, proj_dim, num_classes, num_domains, is_nonlinear, use_motion = False):
        super(Algorithm, self).__init__()
        self.proj_dim = proj_dim
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.is_nonlinear = is_nonlinear
        self.use_motion = use_motion

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError


def predict(self, x):
    # x is expected to be a tuple of modalities.
    if self.use_motion:
        motion = x[0] if isinstance(x, (list, tuple)) else x
        motion = motion.to(self.device)
        if motion.dim() == 3:
            start = self.motion_slice_start
            end = None if self.motion_slice_len is None else (start + self.motion_slice_len)
            if start is not None and start > 0:
                motion = motion[:, start:end, :]
        features = motion
    else:
        motion, text_features, audio_mels, audio_onsets, audio_wav2vec, _ = x
        features = {
            "sentence": text_features.to(self.device),
            "mel": audio_mels.to(self.device),
            "onset": audio_onsets.to(self.device),
            "wav2vec": audio_wav2vec.to(self.device),
        }
    all_z = self.featurizer(features)
    return self.classifier(all_z)


def remap_ids(ids: torch.Tensor, id_to_index: dict, id_map_tensor: torch.Tensor = None,
              unknown_value: int = -1) -> torch.Tensor:
    ids_long = ids.to(torch.long)
    if id_map_tensor is not None:
        return id_map_tensor[ids_long]
    mapped = [id_to_index.get(int(i), unknown_value) for i in ids_long.cpu().tolist()]
    return torch.tensor(mapped, dtype=torch.long)


class SupConLoss(nn.Module):
    """Supervised contrastive loss with optional domain constraint."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z: torch.Tensor, y: torch.Tensor, domain=None,
                require_diff_domain: bool = True) -> torch.Tensor:
        if z.dim() != 2:
            raise ValueError(f"SupConLoss expects z with shape (B, D), got {tuple(z.shape)}")
        y = y.view(-1)
        bsz = z.size(0)
        if bsz <= 1:
            return z.new_tensor(0.0)

        logits = (z @ z.t()) / self.temperature
        diag = torch.eye(bsz, device=z.device, dtype=torch.bool)
        logits = logits.masked_fill(diag, float("-inf"))

        same_class = (y.unsqueeze(1) == y.unsqueeze(0)) & (~diag)
        if domain is not None:
            domain = domain.view(-1)
            if require_diff_domain:
                pos_mask = same_class & (domain.unsqueeze(1) != domain.unsqueeze(0))
            else:
                pos_mask = same_class
        else:
            pos_mask = same_class

        log_den = torch.logsumexp(logits, dim=1)
        pos_logits = logits.masked_fill(~pos_mask, float("-inf"))
        log_num = torch.logsumexp(pos_logits, dim=1)

        valid = torch.isfinite(log_num)
        if valid.sum() == 0:
            return z.new_tensor(0.0)
        return -(log_num[valid] - log_den[valid]).mean()


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


def generate_contrastive_pairs(embeddings, targets, domains, max_pairs: int = 2048):
    """Generate balanced positive/negative pairs for contrastive loss."""
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.cat(list(embeddings), dim=0)

    device = embeddings.device
    targets = targets.view(-1).to(device)
    domains = domains.view(-1).to(device)

    bsz, dim = embeddings.shape
    rng = torch.randperm(bsz, device=device)

    pos_pairs = []
    neg_pairs = []

    class_to_idx = {}
    for c in torch.unique(targets).tolist():
        class_to_idx[int(c)] = (targets == c).nonzero(as_tuple=False).view(-1)

    for i in range(bsz):
        ci = int(targets[i].item())
        idx_pool = class_to_idx[ci]
        if idx_pool.numel() > 1:
            diff_domain = idx_pool[domains[idx_pool] != domains[i]]
            if diff_domain.numel() > 0:
                j = diff_domain[torch.randint(0, diff_domain.numel(), (1,), device=device)].item()
                pos_pairs.append((i, j))

        cj = int(targets[rng[i]].item())
        if cj != ci:
            j = int(rng[i].item())
            neg_pairs.append((i, j))

    n_pairs = min(len(pos_pairs), len(neg_pairs), max_pairs // 2)
    if n_pairs == 0:
        return embeddings.new_zeros((0, 2, dim)), embeddings.new_zeros((0,), dtype=torch.long)

    pos_pairs = pos_pairs[:n_pairs]
    neg_pairs = neg_pairs[:n_pairs]
    pair_idx = pos_pairs + neg_pairs
    labels = torch.tensor([0] * n_pairs + [1] * n_pairs, device=device, dtype=torch.long)

    idx1 = torch.tensor([p[0] for p in pair_idx], device=device, dtype=torch.long)
    idx2 = torch.tensor([p[1] for p in pair_idx], device=device, dtype=torch.long)
    pairs = torch.stack([embeddings[idx1], embeddings[idx2]], dim=1)
    return pairs, labels

def create_domain_loaders(gesture_dataset, batch_size, save_path, min_samples: int = 30,
                          num_workers: int = 0, pin_memory: bool = False):
    """
    Groups the dataset indices by subject (domain) and returns a dictionary
    mapping subject to its DataLoader.
    """
    from collections import defaultdict
    subject_to_indices = defaultdict(list)
    subject_to_domain_indices_file = os.path.join(save_path, "subject_to_domain_indices.pkl")

    if not os.path.exists(subject_to_domain_indices_file):
        for idx in range(len(gesture_dataset)):
            #print("IDX",idx)
            sample = gesture_dataset[idx]
            # When not motion_only, sample returns:
            # (motion, text_features, audio_mels, audio_onsets, audio_wav2vec, labels)
            labels = sample[-1]
            subject = labels['speaker_enc'].item()
            subject_to_indices[subject].append(idx)

        with open(subject_to_domain_indices_file, "wb") as f:
            pickle.dump(subject_to_indices, f)
    else:
        with open(subject_to_domain_indices_file, "rb") as f:
            subject_to_indices = pickle.load(f)


    domain_loaders = {}
    skipped_subjects = 0
    #print("TOT ITEMS",len(subject_to_indices.items()))
    for subject, indices in subject_to_indices.items():
        if len(indices) < min_samples:
            skipped_subjects += 1
            continue  # Skip low-sample domains
        #print("SUBJECT",subject,"INDICES",len(indices))
        #print("SUBJECT",subject)
        domain_loaders[subject] = DataLoader(
            Subset(gesture_dataset, indices),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    print(f"Skipped {skipped_subjects} subjects with fewer than 30 samples.")
    return domain_loaders



def evaluate_set(model, data_loader, device, n_levels, type="train",
                 culture_weights=None):  # evaluation of the val set at the end of each epoch

    model = model.eval()

    loss_entropy = nn.CrossEntropyLoss() if culture_weights == None else nn.CrossEntropyLoss(weight=culture_weights)
    metrics = {'accuracy': 0, 'balanced_accuracy': 0, "weighted_f1": 0, "roc_auc": 0, "cross_entropy": 1e2}

    use_language_metrics = bool(getattr(model, "use_language_adversary", False))
    if use_language_metrics:
        metrics.update({
            "lang_accuracy": 0,
            "lang_balanced_accuracy": 0,
            "lang_weighted_f1": 0,
            "lang_cross_entropy": torch.tensor(0.0),
        })

    all_outputs_cultures = []
    all_outputs_cultures_torch = []
    all_probs_outputs_cultures = []
    all_labels_cultures = []
    all_labels_cultures_torch = []
    all_outputs_languages = []
    all_outputs_languages_torch = []
    all_labels_languages = []
    all_labels_languages_torch = []

    with torch.no_grad():  # No need to track gradients during validation
        for iter_idx, data in enumerate(data_loader, 0):
            inputs = data[:-1]
            inputs_in_device = []
            cultures = data[-1]['culture_enc']
            cultures = cultures.to(device)
            languages = data[-1].get('language_enc', None)
            if languages is not None:
                languages = languages.to(device)
            if getattr(model, "use_motion", False):
                # In motion mode, only the first tensor is used by model.predict.
                inputs_in_device.append(inputs[0].to(device))
            else:
                for input in inputs:
                    inputs_in_device.append(input.to(device))
            culture_outputs, embeddings = model.predict(inputs_in_device, return_embeddings=True)  # F.softmax(model(inputs))

            np_probs_outputs_cultures = F.softmax(culture_outputs, dim=1).data.cpu().numpy()  # it contains the probs of each class
            np_pred_outputs_cultures = np.argmax(np_probs_outputs_cultures, axis=1)  # it contains the indexes of the predictions
            np_cultures = cultures.data.cpu().numpy()

            all_outputs_cultures_torch.append(culture_outputs)
            all_labels_cultures_torch.append(cultures)
            all_labels_cultures.extend(np_cultures)
            all_outputs_cultures.extend(np_pred_outputs_cultures)
            all_probs_outputs_cultures.extend(np_probs_outputs_cultures)

            if use_language_metrics and languages is not None and languages.numel() > 0:
                language_to_index = getattr(model, "language_to_index", None)
                language_map_tensor = getattr(model, "language_map_tensor", None)
                if language_to_index is not None:
                    mapped_languages = remap_ids(languages, language_to_index, language_map_tensor, unknown_value=-1).to(device)
                    valid_lang = mapped_languages >= 0
                else:
                    mapped_languages = languages
                    valid_lang = torch.ones_like(mapped_languages, dtype=torch.bool)

                if valid_lang.any():
                    language_outputs = model.language_classifier(embeddings[valid_lang])
                    np_probs_outputs_languages = F.softmax(language_outputs, dim=1).data.cpu().numpy()
                    np_pred_outputs_languages = np.argmax(np_probs_outputs_languages, axis=1)
                    np_labels_languages = mapped_languages[valid_lang].data.cpu().numpy()
                    all_outputs_languages_torch.append(language_outputs)
                    all_labels_languages_torch.append(mapped_languages[valid_lang])
                    all_labels_languages.extend(np_labels_languages)
                    all_outputs_languages.extend(np_pred_outputs_languages)

    all_labels_cultures_torch = torch.cat(all_labels_cultures_torch, dim=0).long()
    all_outputs_cultures_torch = torch.cat(all_outputs_cultures_torch, dim=0).float()
    model.train()
    for metric in metrics.keys():
        if metric.startswith("lang_"):
            continue
        if metric == 'roc_auc':
            np_labels_cultures_bin = label_binarize(all_labels_cultures, classes=[i for i in range(n_levels)])
            metrics[metric] = roc_auc_score(np_labels_cultures_bin, all_probs_outputs_cultures, average='weighted', multi_class='ovr')
        elif metric == 'accuracy':
            metrics[metric] = accuracy_score(all_labels_cultures, all_outputs_cultures)
        elif metric == 'balanced_accuracy':
            metrics[metric] = balanced_accuracy_score(all_labels_cultures, all_outputs_cultures)
        elif metric == "weighted_f1":
            metrics[metric] = f1_score(all_labels_cultures, all_outputs_cultures, average='weighted')
        elif metric == "cross_entropy":
            metrics[metric] = loss_entropy(all_outputs_cultures_torch, all_labels_cultures_torch)

    if use_language_metrics and len(all_labels_languages) > 0:
        language_loss_entropy = nn.CrossEntropyLoss()
        all_labels_languages_torch = torch.cat(all_labels_languages_torch, dim=0).long()
        all_outputs_languages_torch = torch.cat(all_outputs_languages_torch, dim=0).float()
        metrics["lang_accuracy"] = accuracy_score(all_labels_languages, all_outputs_languages)
        metrics["lang_balanced_accuracy"] = balanced_accuracy_score(all_labels_languages, all_outputs_languages)
        metrics["lang_weighted_f1"] = f1_score(all_labels_languages, all_outputs_languages, average='weighted')
        metrics["lang_cross_entropy"] = language_loss_entropy(all_outputs_languages_torch, all_labels_languages_torch)

    if type != 'train':
        cm = confusion_matrix(all_labels_cultures, all_outputs_cultures)
    else:
        cm = None  # not useful in training, it slows the training process

    return metrics, cm



class Fishr(Algorithm):
    "Invariant Gradients variances for Out-of-distribution Generalization"

    def __init__(self, proj_dim, num_classes, num_domains, is_nonlinear, use_motion = False,
                 use_language_adversary: bool = False, n_languages: int = 0,
                 lang_grl_lambda: float = 1.0, lang_loss_weight: float = 0.0,
                 supcon_weight: float = 0.0, supcon_temperature: float = 0.07,
                 supcon_require_diff_domain: bool = True,
                 contrastive_weight: float = 0.0,
                 mixup_weight: float = 0.0, mixup_alpha: float = 0.0,
                 fishr_penalty_weight: float = 1000.0,
                 fishr_warmup_steps: int = 500,
                 fishr_penalty_ramp_steps: int = 100,
                 grad_clip_norm: float = 1.0,
                 cl_type: str = "culclI",
                 d_model: int = 512,
                 pose_enc_type=None,
                 audio_enc_type=None,
                 raw_poses: bool = False,
                 backbone_config=None,
                 use_adversarial_backbone: bool = True,
                 optimizer_lr: float = 1e-4,
                 optimizer_betas=(0.9, 0.999),
                 optimizer_weight_decay: float = 1e-4):
        #assert backpack is not None, "Install backpack with: 'pip install backpack-for-pytorch==1.3.0'"
        super(Fishr, self).__init__(proj_dim, num_classes, num_domains, is_nonlinear, use_motion)
        self.num_domains = num_domains
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_language_adversary = use_language_adversary and (n_languages > 0)
        self.n_languages = n_languages
        self.lang_grl_lambda = lang_grl_lambda
        self.lang_loss_weight = lang_loss_weight
        self.supcon_weight = float(supcon_weight)
        self.supcon_temperature = float(supcon_temperature)
        self.supcon_require_diff_domain = bool(supcon_require_diff_domain)
        self.contrastive_weight = float(contrastive_weight)
        self.mixup_weight = float(mixup_weight)
        self.mixup_alpha = float(mixup_alpha)
        self.fishr_penalty_weight = float(fishr_penalty_weight)
        self.fishr_warmup_steps = int(fishr_warmup_steps)
        self.fishr_penalty_ramp_steps = max(1, int(fishr_penalty_ramp_steps))
        self.grad_clip_norm = float(grad_clip_norm)
        self.cl_type = cl_type
        self.d_model = int(d_model)
        self.pose_enc_type = _normalize_encoder_type(pose_enc_type)
        self.audio_enc_type = _normalize_encoder_type(audio_enc_type)
        self.raw_poses = bool(raw_poses)
        self.use_adversarial_backbone = bool(use_adversarial_backbone and not use_motion)

        if backbone_config is None:
            backbone_config = SimpleNamespace(dropout=0.1, layer_neurons=self.d_model, levels=1)
        self.backbone_config = backbone_config

        optimizer_betas = tuple(float(x) for x in optimizer_betas)
        if len(optimizer_betas) != 2:
            optimizer_betas = (0.9, 0.999)
        self.optimizer_lr = float(optimizer_lr)
        self.optimizer_betas = optimizer_betas
        self.optimizer_weight_decay = float(optimizer_weight_decay)

        if use_motion:
            self.featurizer = MotionFeaturizer(proj_dim)
            feature_dim = proj_dim
        elif self.use_adversarial_backbone:
            self.featurizer = AdversarialAlignedFeaturizer(
                config=self.backbone_config,
                device=self.device,
                cl_type=self.cl_type,
                d_model=self.d_model,
                pose_enc_type=self.pose_enc_type,
                audio_enc_type=self.audio_enc_type,
                raw_poses=self.raw_poses,
            )
            feature_dim = self.featurizer.output_dim
        else:
            self.featurizer = MultiModalFeaturizer(proj_dim)
            feature_dim = 3 * proj_dim

        base_classifier = Classifier(feature_dim, num_classes, is_nonlinear)
        self.classifier = extend(base_classifier) if extend is not None else base_classifier
        self.network = nn.Sequential(self.featurizer, self.classifier)

        if self.use_language_adversary:
            self.language_classifier = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(feature_dim, self.n_languages),
            )

        self.register_buffer("update_count", torch.tensor([0]))
        base_ce = nn.CrossEntropyLoss(reduction='none')
        self.bce_extended = extend(base_ce) if extend is not None else base_ce
        self.supcon_loss_fn = SupConLoss(temperature=self.supcon_temperature).to(self.device)
        self.contrastive_loss_fn = ContrastiveLoss(margin=1.0).to(self.device)
        #self.update_count = 0
        #self.bce_extended = nn.CrossEntropyLoss(reduction='none')
        #self.bce_extended = extend(nn.CrossEntropyLoss(reduction='none'))
        # self.ema_per_domain = [
        #     MovingAverage(ema=0.95, oneminusema_correction=True)
        #     for _ in range(self.num_domains)
        # ]
        self.ema_per_domain = {}
        self._backpack_batchgrad_enabled = backpack is not None
        self._warned_missing_grad_batch = False
        self._init_optimizer()

    def _current_penalty_weight(self, update_step: int) -> float:
        if update_step < self.fishr_warmup_steps:
            return 0.0
        ramp_progress = (update_step - self.fishr_warmup_steps + 1) / float(self.fishr_penalty_ramp_steps)
        ramp_progress = max(0.0, min(1.0, ramp_progress))
        return self.fishr_penalty_weight * ramp_progress

    def _init_optimizer(self):
        params = list(self.featurizer.parameters()) + list(self.classifier.parameters())
        if self.use_language_adversary:
            params += list(self.language_classifier.parameters())
        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            weight_decay=self.optimizer_weight_decay,
        )

    def feature_extract(self, features):
        cultural_embedding = self.featurizer(features)
        return cultural_embedding


    def update(self, minibatches, domain_ids, unlabeled=False):
        # minibatches: list of batches, one per sampled domain.
        # domain_ids: corresponding list of subject identifiers.
        """
             Expects minibatches as a list of tuples (x, y) per domain.
             Here, each x is itself a tuple:
               (wav2vec, mel_log, onsets, sent)
             """
        '''
        assert len(minibatches) == self.num_domains
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        len_minibatches = [x.shape[0] for x, y in minibatches]

        all_z = self.featurizer(all_x)
        
        '''
        #print("HERE5")
        #print("DIM",len(minibatches), minibatches[0][1].shape[0])
        assert len(minibatches) == len(domain_ids) == self.num_domains
        #all_motions = torch.cat([x[0] for x in minibatches])
        # Prepare the dictionary for the featurizer
        if self.use_motion:
            features = torch.cat([batch[0][:,5:,:] for batch in minibatches], dim=0).to(self.device) #Take only 20 codebooks for 4 second of generated motion
        elif self.use_adversarial_backbone:
            features = [
                torch.cat([batch[0] for batch in minibatches], dim=0).to(self.device),
                torch.cat([batch[1] for batch in minibatches], dim=0).to(self.device),
                torch.cat([batch[2] for batch in minibatches], dim=0).to(self.device),
                torch.cat([batch[3] for batch in minibatches], dim=0).to(self.device),
                torch.cat([batch[4] for batch in minibatches], dim=0).to(self.device),
            ]
        else:
            features = {
                "sentence": torch.cat([batch[1] for batch in minibatches], dim=0).to(self.device),
                "mel": torch.cat([batch[2] for batch in minibatches], dim=0).to(self.device),
                "onset": torch.cat([batch[3] for batch in minibatches], dim=0).to(self.device),
                "wav2vec": torch.cat([batch[4] for batch in minibatches], dim=0).to(self.device)
            }
        #print("HERE6")
        all_y = torch.cat([batch[-1]['culture_enc'] for batch in minibatches], dim=0).to(self.device)
        len_minibatches = [batch[0].shape[0] for batch in minibatches]


        all_domains = torch.cat([batch[-1]['speaker_enc'] for batch in minibatches], dim=0).to(self.device)

        all_z = self.featurizer(features)
        all_z_norm = F.normalize(all_z, p=2, dim=1)
        all_logits = self.classifier(all_z)


        all_nll = F.cross_entropy(all_logits, all_y)
        #print("HERE10")

        language_loss = torch.tensor(0.0, device=self.device)
        if self.use_language_adversary:
            all_languages = torch.cat([batch[-1]['language_enc'] for batch in minibatches], dim=0).to(self.device)
            language_to_index = getattr(self, "language_to_index", None)
            language_map_tensor = getattr(self, "language_map_tensor", None)
            if language_to_index is not None:
                all_languages = remap_ids(
                    all_languages,
                    language_to_index,
                    language_map_tensor,
                    unknown_value=-1,
                ).to(self.device)
                valid_lang = all_languages >= 0
                if valid_lang.any():
                    z_language = grad_reverse(all_z[valid_lang], lambda_=self.lang_grl_lambda)
                    lang_logits = self.language_classifier(z_language)
                    language_loss = F.cross_entropy(lang_logits, all_languages[valid_lang])
            else:
                z_language = grad_reverse(all_z, lambda_=self.lang_grl_lambda)
                lang_logits = self.language_classifier(z_language)
                language_loss = F.cross_entropy(lang_logits, all_languages)

        supcon_loss = torch.tensor(0.0, device=self.device)
        if self.supcon_weight > 0.0:
            supcon_loss = self.supcon_loss_fn(
                all_z_norm,
                all_y,
                domain=all_domains,
                require_diff_domain=self.supcon_require_diff_domain,
            )

        contrastive_loss = torch.tensor(0.0, device=self.device)
        if self.contrastive_weight > 0.0:
            pairs, pair_labels = generate_contrastive_pairs(all_z_norm, all_y, all_domains)
            if pairs.numel() > 0:
                emb1 = pairs[:, 0, :]
                emb2 = pairs[:, 1, :]
                labels = pair_labels.to(dtype=emb1.dtype)
                distances = F.pairwise_distance(emb1, emb2)
                margin = self.contrastive_loss_fn.margin
                contrastive_loss = (
                    (1.0 - labels) * distances.pow(2)
                    + labels * torch.clamp(margin - distances, min=0.0).pow(2)
                ).mean()

        mixup_loss = torch.tensor(0.0, device=self.device)
        if self.mixup_weight > 0.0 and self.mixup_alpha > 0.0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            z = all_z_norm
            perm_idx = torch.arange(z.shape[0], device=z.device)
            for c in torch.unique(all_y):
                idx = (all_y == c).nonzero(as_tuple=False).view(-1)
                if idx.numel() <= 1:
                    continue
                shuf = idx[torch.randperm(idx.numel(), device=z.device)]
                perm_idx[idx] = shuf
            z_perm = z[perm_idx]
            z_mix = F.normalize(lam * z + (1.0 - lam) * z_perm, p=2, dim=1)
            mix_logits = self.classifier(z_mix)
            mixup_loss = F.cross_entropy(mix_logits, all_y)

        update_step = int(self.update_count.item())
        penalty_weight = self._current_penalty_weight(update_step)
        if update_step == self.fishr_warmup_steps:
            # Reset Adam as in IRM or V-REx, because it may not like the sharp jump in
            # gradient magnitudes that happens at this step.
            self._init_optimizer()
        # Skip expensive Fishr gradient-variance computation during warmup.
        penalty = self.compute_fishr_penalty(all_logits, all_y, len_minibatches, domain_ids) \
            if penalty_weight > 0.0 else all_nll.new_zeros(())
        self.update_count += 1

        objective = (
            all_nll
            + penalty_weight * penalty
            + self.lang_loss_weight * language_loss
            + self.supcon_weight * supcon_loss
            + self.contrastive_weight * contrastive_loss
            + self.mixup_weight * mixup_loss
        )
        penalty_finite = bool(torch.isfinite(penalty).all().item())
        if not torch.isfinite(objective):
            logging.warning(
                "Non-finite objective at step %d. Skipping update (nll=%.4f, penalty=%.4f, penalty_weight=%.2f).",
                update_step,
                float(all_nll.detach().item()),
                float(penalty.detach().item()) if penalty_finite else float("nan"),
                float(penalty_weight),
            )
            self.optimizer.zero_grad(set_to_none=True)
            return {
                'loss': float("nan"),
                'nll': all_nll.item(),
                'penalty': penalty.item() if penalty_finite else float("nan"),
                'language_loss': language_loss.item() if self.use_language_adversary else 0.0,
                'supcon_loss': supcon_loss.item() if self.supcon_weight > 0.0 else 0.0,
                'contrastive_loss': contrastive_loss.item() if self.contrastive_weight > 0.0 else 0.0,
                'mixup_loss': mixup_loss.item() if self.mixup_weight > 0.0 else 0.0,
            }

        self.optimizer.zero_grad(set_to_none=True)
        objective.backward()
        if self.grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.grad_clip_norm)
        self.optimizer.step()
        #print("HERE12")

        return {
            'loss': objective.item(),
            'nll': all_nll.item(),
            'penalty': penalty.item(),
            'language_loss': language_loss.item() if self.use_language_adversary else 0.0,
            'supcon_loss': supcon_loss.item() if self.supcon_weight > 0.0 else 0.0,
            'contrastive_loss': contrastive_loss.item() if self.contrastive_weight > 0.0 else 0.0,
            'mixup_loss': mixup_loss.item() if self.mixup_weight > 0.0 else 0.0,
        }

    def compute_fishr_penalty(self, all_logits, all_y, len_minibatches, domain_ids):
        dict_grads = self._get_grads(all_logits, all_y)
        #print("HERE13")
        grads_var_per_domain = self._get_grads_var_per_domain(dict_grads, len_minibatches, domain_ids)
        #print("HERE14")
        return self._compute_distance_grads_var(grads_var_per_domain)

    def _get_grads_autograd_fallback(self, logits, y):
        params = list(self.classifier.named_parameters())
        batch_size = logits.size(0)
        dict_grads = OrderedDict(
            (name, torch.zeros(batch_size, p.numel(), device=p.device, dtype=p.dtype))
            for name, p in params
        )
        for i in range(batch_size):
            loss_i = self.bce_extended(logits[i].unsqueeze(0), y[i].unsqueeze(0)).sum()
            grads_i = torch.autograd.grad(
                loss_i,
                [p for _, p in params],
                create_graph=True,
                retain_graph=True,
                allow_unused=False,
            )
            for (name, _), g in zip(params, grads_i):
                dict_grads[name][i] = g.reshape(-1)
        return dict_grads

    def _get_grads(self, logits, y):
        # Compute per-sample gradients using a loop.
        '''
        batch_size = logits.size(0)
        individual_grads = OrderedDict()
        # Initialize container for each parameter.
        for name, param in self.classifier.named_parameters():
            individual_grads[name] = torch.zeros(batch_size, param.numel(), device=param.device)
            print("HERE16")

        for i in range(batch_size):
            self.optimizer.zero_grad()
            loss_i = self.bce_extended(logits[i].unsqueeze(0), y[i].unsqueeze(0))
            grads_i = torch.autograd.grad(loss_i, list(self.classifier.parameters()), create_graph=True, retain_graph=True)
            for (name, param), grad_val in zip(self.classifier.named_parameters(), grads_i):
                individual_grads[name][i] = grad_val.reshape(-1)
        return individual_grads
        '''
        self.optimizer.zero_grad(set_to_none=True)
        dict_grads = None
        if self._backpack_batchgrad_enabled:
            loss = self.bce_extended(logits, y).sum()
            with backpack(BatchGrad()):
                loss.backward(retain_graph=True, create_graph=True)

            dict_grads = OrderedDict()
            missing_grad_batch = []
            for name, weights in self.classifier.named_parameters():
                grad_batch = getattr(weights, "grad_batch", None)
                if grad_batch is None:
                    missing_grad_batch.append(name)
                    continue
                dict_grads[name] = grad_batch.clone().view(grad_batch.size(0), -1)

            if missing_grad_batch:
                self._backpack_batchgrad_enabled = False
                if not self._warned_missing_grad_batch:
                    logging.warning(
                        "BackPACK grad_batch missing for parameters %s. Falling back to autograd loop for per-sample grads.",
                        missing_grad_batch,
                    )
                    self._warned_missing_grad_batch = True
                dict_grads = None

        if dict_grads is None:
            dict_grads = self._get_grads_autograd_fallback(logits, y)

        # Break the reference cycle warned by torch when using create_graph=True.
        for p in self.classifier.parameters():
            p.grad = None
            if hasattr(p, "grad_batch"):
                try:
                    delattr(p, "grad_batch")
                except Exception:
                    pass
        return dict_grads


    def _get_grads_var_per_domain(self, dict_grads, len_minibatches, domain_ids):
        # Compute per-domain gradient variances.
        grads_var_per_domain = {}
        start_idx = 0
        for i, bsize in enumerate(len_minibatches):
            domain = domain_ids[i]
            end_idx = start_idx + bsize
            # For each parameter, compute variance of gradients in this domain.
            domain_grad_var = {}
            for name, _grads in dict_grads.items():
                domain_grads = _grads[start_idx:end_idx]
                env_mean = domain_grads.mean(dim=0, keepdim=True)
                env_grads_centered = domain_grads - env_mean
                domain_grad_var[name] = (env_grads_centered).pow(2).mean(dim=0)
            start_idx = end_idx

            # Update the EMA for this domain.
            if domain not in self.ema_per_domain:
                self.ema_per_domain[domain] = MovingAverage(ema=0.95, oneminusema_correction=True)
            domain_grad_var = self.ema_per_domain[domain].update(domain_grad_var)
            grads_var_per_domain[domain] = domain_grad_var

        return grads_var_per_domain

    '''
    def _get_grads_var_per_domain(self, dict_grads, len_minibatches, domain_ids):
        # grads var per domain
        #print("HERE18")
        grads_var_per_domain = [{} for _ in range(self.num_domains)]
        #print("HEREEX",len(dict_grads.items()))
        for name, _grads in dict_grads.items():
            #print("HERE19",len_minibatches)
            all_idx = 0
            for domain_id, bsize in enumerate(len_minibatches):
                print(domain_id,bsize)
                env_grads = _grads[all_idx:all_idx + bsize]
                all_idx += bsize
                env_mean = env_grads.mean(dim=0, keepdim=True)
                env_grads_centered = env_grads - env_mean
                grads_var_per_domain[domain_id][name] = (env_grads_centered).pow(2).mean(dim=0)

        # moving average
        for domain_id in range(self.num_domains):
            grads_var_per_domain[domain_id] = self.ema_per_domain[domain_id].update(
                grads_var_per_domain[domain_id]
            )

        return grads_var_per_domain
    '''

    def _compute_distance_grads_var(self, grads_var_per_domain):
        '''
        # compute gradient variances averaged across domains
        grads_var = OrderedDict(
            [
                (
                    name,
                    torch.stack(
                        [
                            grads_var_per_domain[domain_id][name]
                            for domain_id in range(self.num_domains)
                        ],
                        dim=0
                    ).mean(dim=0)
                )
                for name in grads_var_per_domain[0].keys()
            ]
        )

        penalty = 0
        for domain_id in range(self.num_domains):
            penalty += l2_between_dicts(grads_var_per_domain[domain_id], grads_var)
        return penalty / self.num_domains
        '''
        # Compute the average gradient variance across domains.
        all_domains = list(grads_var_per_domain.keys())
        grads_var_avg = OrderedDict()
        for name in grads_var_per_domain[all_domains[0]].keys():
            # Stack and average the EMA-updated variances for each parameter.
            grads_var_avg[name] = torch.stack([
                grads_var_per_domain[d][name] for d in all_domains
            ], dim=0).mean(dim=0)
        penalty = 0
        for d in all_domains:
            penalty += l2_between_dicts(grads_var_per_domain[d], grads_var_avg)
        return penalty / len(all_domains)

    #def predict(self, x):
    #    return self.network(x)
    def predict(self, x, return_embeddings: bool = False):
        #print("HERE19")
        # x is expected to be a tuple of modalities.

        # Prepare dictionary input for the featurizer.
        if self.use_motion:
            motion = x[0] if isinstance(x, (list, tuple)) else x
            motion = motion.to(self.device)
            if motion.dim() == 3:
                # If a prefixed 25-frame sequence is provided, keep the 20-frame target window.
                # If motion is already a 20-frame generated window, do not slice again.
                if motion.shape[1] > 20:
                    motion = motion[:, 5:, :]
            features = motion
        elif self.use_adversarial_backbone:
            if not isinstance(x, (list, tuple)) or len(x) < 5:
                raise ValueError("Adversarial-aligned Fishr expects inputs as (motion, text, mel, onset, wav2vec, ...).")
            features = [
                x[0].to(self.device),
                x[1].to(self.device),
                x[2].to(self.device),
                x[3].to(self.device),
                x[4].to(self.device),
            ]
        else:
            if len(x) == 6:
                _, text_features, audio_mels, audio_onsets, audio_wav2vec, _ = x
            else:
                _, text_features, audio_mels, audio_onsets, audio_wav2vec = x
            features = {
                    "sentence": text_features.to(self.device),
                    "mel": audio_mels.to(self.device),
                    "onset": audio_onsets.to(self.device),
                    "wav2vec": audio_wav2vec.to(self.device),
                }
        all_z = self.featurizer(features)
        logits = self.classifier(all_z)
        if return_embeddings:
            return logits, all_z
        return logits




def train_fishr_model(model, domain_loaders, num_epochs, val_loader, save_dir = '', k_domain=128):
    """
    Custom training loop that draws one batch per sampled domain per iteration.
    Uses restartable iterators per domain to avoid itertools.cycle() caching batches.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_path = os.path.join(save_dir, "model_best.pt")
    best_val_loss = (1e+2, 0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_epoch = load_last_checkpoint(model, save_dir)
    print("Start Epoch", start_epoch)

    all_subjects = list(domain_loaders.keys())
    k_domain = min(k_domain, len(all_subjects))
    model.num_domains = k_domain
    domain_iters = {subj: iter(loader) for subj, loader in domain_loaders.items()}
    num_batches = max(len(loader) for loader in domain_loaders.values())

    if model.use_language_adversary:
        unique_train_languages = set()
        for loader in domain_loaders.values():
            for batch in loader:
                labels = batch[-1]
                if isinstance(labels, dict) and 'language_enc' in labels:
                    unique_train_languages.update(labels['language_enc'].cpu().tolist())
        unique_train_languages = sorted(list(unique_train_languages))
        model.language_to_index = {lang_id: idx for idx, lang_id in enumerate(unique_train_languages)}
        if len(model.language_to_index) == 0:
            model.use_language_adversary = False
            logging.warning("Language adversary requested but no language labels found in training data.")
        else:
            max_language_id = int(max(model.language_to_index.keys()))
            max_allowed_lang = 10000
            if max_language_id <= max_allowed_lang:
                model.language_map_tensor = torch.full((max_language_id + 1,), -1, dtype=torch.long, device=device)
                for lid, mid in model.language_to_index.items():
                    model.language_map_tensor[int(lid)] = int(mid)
            else:
                model.language_map_tensor = None

    for epoch in tqdm(range(start_epoch, num_epochs)):
        for batch_idx in  tqdm(range(num_batches)):
            #if batch_idx >= 1:
            #    break
            sampled_subjects = random.sample(all_subjects, k=k_domain)
            #minibatches = [next(it) for it in domain_iters.values()]
            minibatches = []
            for subj in sampled_subjects:
                try:
                    minibatches.append(next(domain_iters[subj]))
                except StopIteration:
                    # Restart iterator for this domain to keep domain sampling infinite without cycle() cache growth.
                    domain_iters[subj] = iter(domain_loaders[subj])
                    minibatches.append(next(domain_iters[subj]))
            #metrics = model.update(minibatches)
            metrics = model.update(minibatches, domain_ids=sampled_subjects)
            logging.info(f"Epoch {epoch:02d} | Batch {batch_idx + 1:03d} | "
                         f"Loss: {metrics['loss']:.4f}, NLL: {metrics['nll']:.4f}, "
                         f"Penalty: {metrics['penalty']:.4f}, Lang: {metrics.get('language_loss', 0.0):.4f}, "
                         f"SupCon: {metrics.get('supcon_loss', 0.0):.4f}, "
                         f"Contrastive: {metrics.get('contrastive_loss', 0.0):.4f}, "
                         f"Mixup: {metrics.get('mixup_loss', 0.0):.4f}")

        validation, _ = evaluate_set(model, val_loader, device, n_levels=model.num_classes, type="val")
        val_ce = validation['cross_entropy'].item() if torch.is_tensor(validation['cross_entropy']) else float(validation['cross_entropy'])
        if val_ce < best_val_loss[0]:
            best_val_loss = (val_ce, epoch)
            torch.save(model.state_dict(), file_path)
            print(f"Best model saved at epoch {epoch} with loss {best_val_loss[0]:.4f} and F1 score {validation['weighted_f1']:.4f}")
        else:
            print(f"Model not improved at epoch {epoch}. Current best loss: {best_val_loss[0]:.4f} at epoch {best_val_loss[1]}; "
                  f"Current loss:  {val_ce:.4f}; Current F1 score: {validation['weighted_f1']:.4f}")

        # --- Save every 1 epoch ---
        #if epoch % 3 == 0:
        checkpoint_path = os.path.join(save_dir, f"model_epoch_{epoch}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved at epoch {epoch}")
