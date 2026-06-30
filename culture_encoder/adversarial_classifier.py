import gc

from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score, accuracy_score, confusion_matrix, \
    make_scorer, get_scorer
from sklearn.preprocessing import label_binarize
from scipy.stats import randint
import lmdb
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from torch import optim
import numpy as np
import sys
from tqdm.auto import tqdm
import time
from pathlib import Path
import logging
import os
import pickle
from datetime import datetime
import itertools
from itertools import product
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    class SummaryWriter:  # Fallback no-op writer for environments without working tensorboard/distutils
        def __init__(self, *args, **kwargs):
            logging.warning("TensorBoard SummaryWriter unavailable. Continuing without TensorBoard logging.")

        def add_scalar(self, *args, **kwargs):
            return None

        def close(self):
            return None

logging.getLogger().setLevel(logging.INFO)
torch.set_printoptions(threshold=10_000)


# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


class Culture_Classifier(nn.Module):

    def __init__(self, config, device, dropout_prob=0.2, cl_type="culclA", d_model=1024,
                 speaker_penalization=False, grl_flag=False, n_cultures=4, n_speakers=0,
                 language_penalization=False, n_languages=0,
                 grl_lambda=1, grl_flag_culture=False, pose_enc = 'transformer', audio_enc = 'transformer',
                 raw_poses = True):

        super(Culture_Classifier, self).__init__()

        self.config_FC = config
        self.grl_lambda = grl_lambda
        self.cl_type = cl_type
        self.device = device
        self.d_model = d_model
        self.dropout = config.dropout
        #self.n_levels = config.FC.levels # Fully connected layers
        self.n_cultures = n_cultures
        self.n_speakers = n_speakers
        self.n_languages = n_languages
        self.grl_flag = grl_flag
        self.grl_flag_culture = grl_flag_culture
        self.speaker_penalization = speaker_penalization
        self.language_penalization = language_penalization
        self.fc_layers = nn.ModuleList()
        self.pose_enc_type = pose_enc
        self.audio_encoders = {}
        self.audio_enc_type = audio_enc
        self.sentence_mapper = None
        self.input_dim_cl = calculate_input_dim(cl_type,d_model)
        self.layer_neurons = config.layer_neurons
        self.embed_dim = getattr(config, 'embed_dim', self.layer_neurons)


        if self.pose_enc_type == 'transformer':
            print("Using transformer pose encoder")
            # Instantiate the PoseEncoder
            if raw_poses:
                self.pose_encoder = TransformerEncoder(d_model=d_model, input_dim=54, output_dim = d_model, dropout= self.dropout,
                                                    nhead=4, num_layers=2,max_seq_length = 75, dim_feedforward=d_model).to(self.device)
            else:
                self.pose_encoder = TransformerEncoder(d_model=d_model, input_dim=512, output_dim=d_model,dropout= self.dropout,
                                                   nhead=4, num_layers=2, max_seq_length=25, dim_feedforward=d_model).to(self.device)
        elif self.pose_enc_type == 'linear':
            if raw_poses:
                #self.pose_encoder = nn.Sequential(nn.Linear(54, 1024),nn.GELU(),nn.Dropout(self.dropout)).to(self.device)
                self.pose_encoder = LinearEncoder(54, d_model=d_model, output_dim=d_model).to(self.device)
            else:
                #self.pose_encoder = nn.Sequential(nn.Linear(512, 1024),nn.GELU(),nn.Dropout(self.dropout)).to(self.device)
                self.pose_encoder = LinearEncoder(512, d_model=d_model, output_dim=d_model).to(self.device)
        else: #None
            self.pose_encoder = None

        if cl_type in ["culclB", "culclF", "culclH", "culclI"]:
            self.sentence_mapper = LinearEncoder(768, d_model=d_model, output_dim=d_model).to(self.device)  # 768 -> 512

        if cl_type in ["culclC", "culclD", "culclE", "culclG", "culclH", "culclI", "culclK", "culclL"]:
            self.audio_encoders = nn.ModuleDict()
            self.audio_encoders['mels'] = LinearEncoder(64, d_model=d_model, output_dim=d_model//2).to(self.device)  # 156,64 -> 156,256
            self.audio_encoders['onsets'] = LinearEncoder(1, d_model=d_model, output_dim=d_model//2).to(self.device)  # 156,1 -> 156,256
            self.audio_encoders['wav2vec'] =  LinearEncoder(1024, d_model=d_model, output_dim=d_model).to(self.device)

        if self.audio_enc_type == 'transformer':
            print("Using transformer audio encoder")
            self.audio_encoders['trans'] = TransformerEncoder(d_model=d_model, input_dim=d_model, output_dim=d_model,
                                                   dropout=self.dropout,nhead=4, num_layers=2, max_seq_length=156).to(self.device)

        # Define Attention Pooling for each temporal feature
        self.attention_poolers = nn.ModuleDict()
        if cl_type in ["culclA", "culclB", "culclC", "culclD", "culclE", "culclH", "culclJ"]:
            self.attention_poolers['poses'] = AttentionPooling(d_model=d_model).to(self.device)

        if cl_type in ["culclC", "culclE", "culclG", "culclH", "culclI","culclL"]:
            # Replace 1024 and 128 with actual d_model values for mels and onsets
            self.attention_poolers['mels'] = AttentionPooling(d_model=d_model//2).to(self.device)
            self.attention_poolers['onsets'] = AttentionPooling(d_model=d_model//2).to(self.device)  # Example d_model; adjust accordingly
            self.attention_poolers['audio'] = AttentionPooling(d_model=d_model).to(self.device) #just for transformer architecture

        if cl_type in ["culclD", "culclE", "culclG", "culclH", "culclI", "culclK"]:
            self.attention_poolers['wav2vec'] = AttentionPooling(d_model=d_model).to(self.device)

        self.n_levels = config.levels
        self.input_dim_cl = self.input_dim_cl


        for i in range(self.n_levels):
            in_features = self.input_dim_cl if i == 0 else self.layer_neurons
            #print("input_dim",self.input_dim_cl)
            self.fc_layers.append(nn.Linear(in_features, self.layer_neurons))
            #self.fc_layers.append(nn.BatchNorm1d(self.layer_neurons))
            # nn.BatchNorm1d(self.layer_neurons))
            self.fc_layers.append(nn.LayerNorm(self.layer_neurons)) #nn.BatchNorm1d(self.layer_neurons))  # Adding BatchNorm here
            self.fc_layers.append(nn.GELU())
            self.fc_layers.append(nn.Dropout(p=dropout_prob))  # add dropout for regularization
        # self.fc_layer = nn.ModuleList([nn.Linear(input_dim_fcA if i == 0 else self.layer_neurons,
        #                                         self.layer_neurons) for i in range(self.n_levels)]

        # Initialize weights for fully connected layers
        # self._initialize_weights()

        self.embedding_head = nn.Sequential(
            nn.Linear(self.layer_neurons, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_prob),
        )

        # Speaker ID adversarial classifier
        if speaker_penalization == True:
            self.speaker_classifier = nn.Sequential(
                nn.Linear(self.embed_dim, self.layer_neurons),  # Example dimension reduction
                nn.LayerNorm(self.layer_neurons),
                #nn.BatchNorm1d(self.layer_neurons),  # Adding BatchNorm here
                nn.GELU(),
                nn.Dropout(p=dropout_prob),
                nn.Linear(self.layer_neurons, self.n_speakers)  # Number of unique speakers
            )

        if language_penalization == True:
            self.language_classifier = nn.Sequential(
                nn.Linear(self.embed_dim, self.layer_neurons),
                nn.LayerNorm(self.layer_neurons),
                nn.GELU(),
                nn.Dropout(p=dropout_prob),
                nn.Linear(self.layer_neurons, self.n_languages)
            )
        # self.output_layer = nn.Linear(self.layer_neurons, self.n_cultures)
        self.culture_classifier = nn.Sequential(
            nn.Linear(self.embed_dim, self.layer_neurons),  # Example dimension reduction
            #nn.BatchNorm1d(self.layer_neurons),  # Adding BatchNorm here
            nn.LayerNorm(self.layer_neurons),
            nn.GELU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(self.layer_neurons, self.n_cultures)  # Number of unique speakers
        )

        # Convert culture_classifier to individual layers for easier access
        self.culture_classifier_layers = nn.ModuleList([
            self.culture_classifier[0],  # First Linear layer
            self.culture_classifier[1],  # BatchNorm -> Gelu
            self.culture_classifier[2],  # GELU -> Second Linear layer
            self.culture_classifier[3],  # Dropout
            self.culture_classifier[4]   # Second Linear layer
        ])

        #self._initialize_weights()
        #self._initialize_weights_classifier(self.speaker_classifier)
        #self._initialize_weights_classifier(self.culture_classifier)

    def _initialize_weights(self):
        for m in self.fc_layers:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / np.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)

    def _initialize_weights_classifier(self, classifier):
        for m in classifier:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / np.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)

    def encode(self, x):
        """Return a culture embedding z for a batch.

        z is L2-normalized so cosine similarity is meaningful and stable for contrastive training
        and for conditioning a generator.
        """
        h = map_inputs(
            self.cl_type, x, self.device,
            self.pose_encoder, self.pose_enc_type,
            self.audio_encoders, self.audio_enc_type,
            getattr(self, "sentence_mapper", None), self.attention_poolers
        )
        for layer in self.fc_layers:
            h = layer(h)
        z = self.embedding_head(h)
        z = F.normalize(z, p=2, dim=1)
        return z

    def forward(self, x, return_embeddings=False):
        z = self.encode(x)
        if return_embeddings:
            return z

        culture_output = self.culture_classifier(z)

        speaker_output = None
        language_output = None

        if self.speaker_penalization:
            z_speaker = grad_reverse(z, lambda_=self.grl_lambda) if self.grl_flag else z
            speaker_output = self.speaker_classifier(z_speaker)

        if self.language_penalization:
            z_language = grad_reverse(z, lambda_=self.grl_lambda) if self.grl_flag else z
            language_output = self.language_classifier(z_language)

        if self.speaker_penalization and self.language_penalization:
            return culture_output, speaker_output, language_output
        if self.speaker_penalization:
            return culture_output, speaker_output
        if self.language_penalization:
            return culture_output, language_output

        return culture_output


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, lambda_):
        ctx.lambda_ = lambda_
        return input.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x, lambda_=1.0):
    return GradientReversalFunction.apply(x, lambda_)

def random_masking(x, mask_ratio=0.1):
    mask = torch.rand(x.shape, device=x.device) < mask_ratio
    x = x * (~mask).float()
    return x

def adjust_lambda(epoch, initial_lambda, max_lambda, max_epoch):
    return min(max_lambda, initial_lambda + epoch * (max_lambda - initial_lambda) / max_epoch)


def adjust_lambda_exponential(epoch, gamma=1, max_epoch=100):
    p = epoch / max_epoch
    return 2 / (1 + np.exp(-gamma * p)) - 1


def adjust_speaker_weight(epoch, initial_weight, max_weight, max_epoch):
    return min(max_weight, initial_weight - epoch * (initial_weight - max_weight) / max_epoch)



def adjust_grl_lambda(epoch, speaker_balanced_acc, grl_lambda, threshold=0.5, max_lambda=5.0):
    if speaker_balanced_acc < threshold:
        grl_lambda = min(grl_lambda + 0.1, max_lambda)
    else:
        grl_lambda = max(grl_lambda - 0.1, 0.0)
    return grl_lambda


def schedule_grl_lambda(epoch, initial_lambda=0.1, max_lambda=1.0, growth_rate=0.01):
    return min(initial_lambda + epoch * growth_rate, max_lambda)


class LinearEncoder(nn.Module):
    def __init__(self, input_dim, output_dim=1024, dropout=0.1, d_model=1024):
        super(LinearEncoder, self).__init__()
        self.linear = nn.Linear(input_dim, d_model)
        self.layer_norm_in = nn.LayerNorm(d_model)
        self.layer_norm_out = nn.LayerNorm(output_dim)
        #self.layer_norm = nn.LayerNorm(d_model)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(d_model, output_dim)
        self.output_dropout_proj = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_length, feature_dim)

        Returns:
            projected_output: Tensor of shape (batch_size, seq_length, output_dim)
        """

        x = self.linear(x)  # (batch_size, seq_length, d_model)
        x = self.layer_norm_in(x)
        #x = self.layer_norm(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.output_projection(x)  # (batch_size, seq_length, output_dim)
        x = self.layer_norm_out(x)
        #x = self.layer_norm(x)
        x = self.gelu(x)
        x = self.output_dropout_proj(x)
        return x  # (batch_size, seq_length, output_dim)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout_layer = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size,seq_length, d_model)
        seq_length = x.size(1)
        x = x + self.pe[:, :seq_length]
        #x = x + self.pe[:seq_length, :]
        return self.dropout_layer(x)

# This just returns one vector. How it works?
# Given an input, e.g. (25,512), where 25 is the temporal dimension,
# it outputs, through learnable parameters, 25 values that sum up to 1 thanks to softmax
# the parameters are learned so that, each vector, shows how much important is during the learning process.
# In this way, by multiplying these (25) numbers, telling how important is each vector to reach the task,
# with input (25, 512), I am weighting each vector with a value between 0 and 1, then sum. The more important
# is a vector the more weight will have in this new representation. If for example component 24 has attention score
# 0,9999 and all the others values close to 0, then the new representation it contains mostly only vector number 24
# This can be considered a single attention head pooling
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
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True,
                                                   dim_feedforward=dim_feedforward, dropout=dropout)
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
        #data = data.permute(1, 0, 2)  # (seq_length, batch_size, d_model)
        transformer_output = self.transformer_encoder(data)  # (batch_size, seq_length, d_model)
        transformer_output = self.layer_norm(transformer_output)
        #transformer_output = transformer_output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        #pooled_output = self.attention_pool(transformer_output)  # (batch_size, d_model)
        #transformer_output = transformer_output.mean(dim=0)  # (batch_size, d_model)
        projected_output = self.output_projection_layer(transformer_output)  # Map to (batch_size, output_dim)
        projected_output = self.gelu(projected_output)
        projected_output = self.output_dropout(projected_output)
        return projected_output



def map_inputs(cl_type, inputs, device, pose_encoder=None,
               pose_encoder_type = None, audio_encoders = None,
               audio_encoder_type = None, sentence_mapper = None,attention_poolers = None) :

    features_list = []
    if cl_type in ["culclA", "culclB", "culclC", "culclD", "culclE", "culclH"]:
        poses = inputs[0].to(device)  # Shape: (batch_size, seq_length, feature_dim)
        #poses = random_masking(poses, 0.1) #apply masking on poses features
        if pose_encoder is not None:
            if pose_encoder_type == 'transformer':
                transformer_output = pose_encoder(poses)  # (batch_size, d_model)
                poses = attention_poolers['poses'](transformer_output)  # d_model features
                features_list.append(poses)
            elif pose_encoder_type == 'linear':
                linear_output = pose_encoder(poses) #(batch_size, 25,  d_model)
                #poses = linear_output.view(linear_output.shape[0], -1) # 25 * d_model
                poses = attention_poolers['poses'](linear_output) #d_model features
                features_list.append(poses)
        else:
            #poses = poses.view(poses.shape[0],-1)
            poses = attention_poolers['poses'](poses) #d_model features
            features_list.append(poses)

    if cl_type in ["culclB", "culclF", "culclH", "culclI"]:
        sent_emb = inputs[1].to(device)  # (batch_size, embedding_dim)
        sent_emb = sentence_mapper(sent_emb) #map to d_model space
        features_list.append(sent_emb)

    if cl_type in ["culclC", "culclE", "culclG", "culclH", "culclI","culclL"]:
        mels = inputs[2].to(device)
        onsets = inputs[3].to(device)
        if onsets.dim() == 2:
            onsets = onsets.unsqueeze(-1)
        elif onsets.dim() == 3 and onsets.shape[-1] == 1:
            pass
        else:
            raise ValueError(f"Unexpected onsets shape {tuple(onsets.shape)} in map_inputs; expected [B,T] or [B,T,1].")
        mapped_onsets = audio_encoders['onsets'](onsets)  #156,1 -> 156,d_model//2
        mapped_mels = audio_encoders['mels'](mels)  #156,64 -> 156,d_model//2
        features = (mapped_mels, mapped_onsets)
        combined_features = torch.cat(features, dim=-1) #156,d_model
        audio = attention_poolers['audio'](combined_features)

        if audio_encoder_type == 'transformer':
            transformer_output = audio_encoders['trans'](combined_features)  #transformer encoder
            audio = attention_poolers['audio'](transformer_output)  #d_model

        features_list.append(audio)


    if cl_type in ["culclD", "culclE", "culclG", "culclH", "culclI","culclK"]:
        #wav2vec = inputs[4].view(inputs[4].shape[0], -1).to(device)
        wav2vec = inputs[4].to(device)
        wav2vec = audio_encoders['wav2vec'](wav2vec) #map from 1024 to d_model
        wav2vec = attention_poolers['wav2vec'](wav2vec) #pool information
        features_list.append(wav2vec)

    if cl_type == "culclJ":
        #raw_poses = inputs[0].view(inputs[0].shape[0], -1).to(device)
        raw_poses = inputs[0].to(device)
        #raw_poses = random_masking(raw_poses, 0.1)
        if pose_encoder is not None:
            if pose_encoder_type == 'transformer':
                transformer_output = pose_encoder(raw_poses)  # (batch_size, d_model)
                poses = attention_poolers['poses'](transformer_output)
                features_list.append(poses)
            elif pose_encoder_type == 'linear':
                linear_output = pose_encoder(raw_poses) #(batch_size, 75,  d_model)
                poses = attention_poolers['poses'](linear_output)
                #poses = linear_output.view(linear_output.shape[0], -1) # 75 * d_model
                features_list.append(poses)
        else:
            #raw_poses = raw_poses.view(raw_poses.shape[0],-1)
            raw_poses = attention_poolers['poses'](raw_poses)
            features_list.append(raw_poses)

    x = torch.cat(features_list, dim=1)

    return x

def mask_frames(x, mask_ratio=0.15):
    """
    x: Tensor of shape (batch_size, seq_len, feature_dim)
    mask_ratio: Fraction of frames to mask
    """
    batch_size, seq_len, feature_dim = x.shape
    mask = torch.rand(batch_size, seq_len, device=x.device) < mask_ratio  # Create mask
    x[mask] = 0.0  # Set masked frames to zero
    return x, mask


# Calculate input_dim_cl based on the concatenated features
def calculate_input_dim(cl_type, d_model):
    # Add dimensions of other features based on cl_type
    input_dim = 0
    if cl_type in ["culclA", "culclB", "culclC", "culclD", "culclE", "culclH"]:
        '''
        if pose_enc_type == 'transformer':
            input_dim += 1024  # Dimension of poses embedding from transformer (if transformer is not used, then it is 25 * 512)
        elif pose_enc_type == 'linear':
            input_dim += 25 * 1024
        else:
            input_dim += 25 * 512
        '''
        input_dim += d_model #poses

    if cl_type in ["culclC","culclE","culclG","culclH","culclI","culclL"]:
        input_dim += d_model #+ d_model//4 #64 * 156 + 156  # Dimension of mellog + onsets
    if cl_type in ["culclD","culclE","culclG","culclH","culclI","culclK"]:
        input_dim += d_model #50 * 1024  # Dimension of wav2vec
    if cl_type in ["culclB", "culclF", "culclH", "culclI"]:
        input_dim += d_model #768  # Dimension of sent_emb
    if cl_type in ["culclJ"]:
        input_dim += d_model  # raw poses
    # Continue adding dimensions for other features
    return input_dim


class SupConLoss(nn.Module):
    """Supervised contrastive loss with optional domain constraint.

    Implements SupCon (Khosla et al., 2020) and allows restricting positives to:
      same class AND (optionally) different domain.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z: torch.Tensor, y: torch.Tensor, domain=None,
                require_diff_domain: bool = True) -> torch.Tensor:
        if z.dim() != 2:
            raise ValueError(f"SupConLoss expects z with shape (B, D), got {tuple(z.shape)}")
        y = y.view(-1)
        B = z.size(0)
        if B <= 1:
            return z.new_tensor(0.0)

        logits = (z @ z.t()) / self.temperature
        diag = torch.eye(B, device=z.device, dtype=torch.bool)
        logits = logits.masked_fill(diag, float('-inf'))

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
        pos_logits = logits.masked_fill(~pos_mask, float('-inf'))
        log_num = torch.logsumexp(pos_logits, dim=1)

        valid = torch.isfinite(log_num)
        if valid.sum() == 0:
            return z.new_tensor(0.0)

        return -(log_num[valid] - log_den[valid]).mean()

def remap_speakers(speakers: torch.Tensor, speaker_to_index: dict, speaker_map_tensor: torch.Tensor = None) -> torch.Tensor:
    """Map arbitrary speaker ids to contiguous [0..n_train_speakers-1] ids.

    This avoids out-of-range indices for the speaker adversarial head when speaker ids are not contiguous
    in the original dataset.
    """
    if speaker_map_tensor is not None:
        speakers_long = speakers.to(torch.long)
        mapped = speaker_map_tensor[speakers_long]
        if (mapped < 0).any():
            # Fallback to dict mapping for any unseen ids
            mapped_list = [speaker_to_index[int(s)] for s in speakers_long.cpu().tolist()]
            return torch.tensor(mapped_list, dtype=torch.long)
        return mapped
    else:
        mapped_list = [speaker_to_index[int(s)] for s in speakers.to(torch.long).cpu().tolist()]
        return torch.tensor(mapped_list, dtype=torch.long)


def remap_ids(ids: torch.Tensor, id_to_index: dict, id_map_tensor: torch.Tensor = None,
              unknown_value: int = -1) -> torch.Tensor:
    """Remap arbitrary ids to contiguous ids used by a classifier head."""
    ids_long = ids.to(torch.long)
    if id_map_tensor is not None:
        mapped = id_map_tensor[ids_long]
        return mapped

    mapped_list = [id_to_index.get(int(i), unknown_value) for i in ids_long.cpu().tolist()]
    return torch.tensor(mapped_list, dtype=torch.long)


def unpack_outputs(outputs, speaker_penalization: bool = False, language_penalization: bool = False):
    """Normalize variable model outputs into (culture, speaker, language)."""
    if speaker_penalization and language_penalization:
        culture_outputs, speaker_outputs, language_outputs = outputs
    elif speaker_penalization:
        culture_outputs, speaker_outputs = outputs
        language_outputs = None
    elif language_penalization:
        culture_outputs, language_outputs = outputs
        speaker_outputs = None
    else:
        culture_outputs = outputs
        speaker_outputs = None
        language_outputs = None
    return culture_outputs, speaker_outputs, language_outputs

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


def generate_contrastive_pairs(embeddings, targets, speakers, max_pairs: int = 2048):
    """Generate a balanced set of (positive, negative) embedding pairs.

    Returns:
        pairs: (N, 2, D)
        labels: (N,) where 0=positive (same culture, different speaker), 1=negative (different culture)
    """
    if not isinstance(embeddings, torch.Tensor):
        # If user passes a list/tuple, concatenate.
        embeddings = torch.cat(list(embeddings), dim=0)

    device = embeddings.device
    targets = targets.view(-1).to(device)
    speakers = speakers.view(-1).to(device)

    B, D = embeddings.shape
    rng = torch.randperm(B, device=device)

    pos_pairs = []
    neg_pairs = []

    # Build index lists per culture for fast sampling.
    culture_to_idx = {}
    for c in torch.unique(targets).tolist():
        culture_to_idx[int(c)] = (targets == c).nonzero(as_tuple=False).view(-1)

    for i in range(B):
        ci = int(targets[i].item())

        # Positive: same culture, different speaker if possible.
        idx_pool = culture_to_idx[ci]
        if idx_pool.numel() > 1:
            # Filter different speaker
            diff_sp = idx_pool[speakers[idx_pool] != speakers[i]]
            if diff_sp.numel() > 0:
                j = diff_sp[torch.randint(0, diff_sp.numel(), (1,), device=device)].item()
                pos_pairs.append((i, j))

        # Negative: different culture
        cj = int(targets[rng[i]].item())
        if cj != ci:
            j = int(rng[i].item())
            neg_pairs.append((i, j))

    # Truncate / balance
    n = min(len(pos_pairs), len(neg_pairs), max_pairs // 2)
    if n == 0:
        return embeddings.new_zeros((0, 2, D)), embeddings.new_zeros((0,), dtype=torch.long)

    pos_pairs = pos_pairs[:n]
    neg_pairs = neg_pairs[:n]

    pairs_idx = pos_pairs + neg_pairs
    labels = torch.tensor([0] * n + [1] * n, device=device, dtype=torch.long)

    idx1 = torch.tensor([p[0] for p in pairs_idx], device=device, dtype=torch.long)
    idx2 = torch.tensor([p[1] for p in pairs_idx], device=device, dtype=torch.long)
    pairs = torch.stack([embeddings[idx1], embeddings[idx2]], dim=1)  # (2n, 2, D)
    return pairs, labels

def evaluate_set(model, data_loader, device, n_levels, type="train", speaker_penalization=False,
                 language_penalization=False, culture_weights=None, speaker_weights=None,
                 language_weights=None, subj_independent_data = False):  # evaluation of the val set at the end of each epoch
    # start = time.time()
    model = model.eval()
    model.grl_flag_culture = False
    model.grl_flag = False

    #Note I can'tclassify subjects in subject independent data splits because the model has never seen them during training.
    loss_entropy = nn.CrossEntropyLoss() if culture_weights == None else nn.CrossEntropyLoss(weight=culture_weights)
    metrics = {'accuracy': 0, 'balanced_accuracy': 0, "weighted_f1": 0, "roc_auc": 0, "cross_entropy": 1e2}
    if speaker_penalization and not subj_independent_data:
        loss_entropy_speaker = nn.CrossEntropyLoss() if speaker_weights == None else nn.CrossEntropyLoss(weight=speaker_weights)
        metrics_speakers = {'accuracy': 0, 'balanced_accuracy': 0, "weighted_f1": 0, "roc_auc": 0, "cross_entropy": 1e2}
        all_outputs_speakers = []
        all_outputs_speakers_torch = []
        all_probs_outputs_speakers = []
        all_labels_speakers = []
        all_labels_speakers_torch = []

    if language_penalization:
        loss_entropy_language = nn.CrossEntropyLoss() if language_weights is None else nn.CrossEntropyLoss(weight=language_weights)
        all_outputs_languages = []
        all_outputs_languages_torch = []
        all_labels_languages = []
        all_labels_languages_torch = []
        metrics.update({
            'lang_accuracy': 0,
            'lang_balanced_accuracy': 0,
            'lang_weighted_f1': 0,
            'lang_cross_entropy': torch.tensor(0.0),
        })

    all_outputs_cultures = []
    all_outputs_cultures_torch = []
    all_probs_outputs_cultures = []
    all_labels_cultures = []
    all_labels_cultures_torch = []

    with torch.no_grad():  # No need to track gradients during validation
        for iter_idx, data in enumerate(data_loader, 0):
            inputs = data[:-1]
            inputs_in_device = []
            cultures = data[-1]['culture_enc']
            cultures = cultures.to(device)
            languages = data[-1].get('language_enc', None)
            if languages is not None:
                languages = languages.to(device)
            for input in inputs:
                inputs_in_device.append(input.to(device))

            speakers = None
            if speaker_penalization and not subj_independent_data:
                speakers = data[-1]['speaker_enc'].to(device)
                all_labels_speakers_torch.append(speakers)

            outputs = model(inputs_in_device)
            culture_outputs, speaker_outputs, language_outputs = unpack_outputs(
                outputs,
                speaker_penalization=speaker_penalization,
                language_penalization=language_penalization,
            )

            if speaker_penalization and not subj_independent_data and speaker_outputs is not None and speakers is not None:
                np_probs_outputs_speakers = F.softmax(speaker_outputs, dim=1).data.cpu().numpy()
                np_pred_outputs_speakers = np.argmax(np_probs_outputs_speakers, axis=1)
                np_labels_speakers = speakers.data.cpu().numpy()
                all_outputs_speakers_torch.append(speaker_outputs)
                all_labels_speakers.extend(np_labels_speakers)
                all_outputs_speakers.extend(np_pred_outputs_speakers)
                all_probs_outputs_speakers.extend(np_probs_outputs_speakers)

            if language_penalization and language_outputs is not None and languages is not None:
                np_probs_outputs_languages = F.softmax(language_outputs, dim=1).data.cpu().numpy()
                np_pred_outputs_languages = np.argmax(np_probs_outputs_languages, axis=1)
                np_labels_languages = languages.data.cpu().numpy()

                all_outputs_languages_torch.append(language_outputs)
                all_labels_languages_torch.append(languages)
                all_labels_languages.extend(np_labels_languages)
                all_outputs_languages.extend(np_pred_outputs_languages)

            np_probs_outputs_cultures = F.softmax(culture_outputs, dim=1).data.cpu().numpy()  # it contains the probs of each class
            np_pred_outputs_cultures = np.argmax(np_probs_outputs_cultures, axis=1)  # it contains the indexes of the predictions
            np_cultures = cultures.data.cpu().numpy()

            all_outputs_cultures_torch.append(culture_outputs)
            all_labels_cultures_torch.append(cultures)
            all_labels_cultures.extend(np_cultures)
            all_outputs_cultures.extend(np_pred_outputs_cultures)
            all_probs_outputs_cultures.extend(np_probs_outputs_cultures)

    # for metric_m, metric in zip(metrics_mean.keys(),metrics.keys()): # compute the mean of all the metrics
    # metrics_mean[metric_m] = np.mean(metrics_mean[metric_m])
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

    if language_penalization and len(all_labels_languages) > 0:
        all_labels_languages_torch = torch.cat(all_labels_languages_torch, dim=0).long()
        all_outputs_languages_torch = torch.cat(all_outputs_languages_torch, dim=0).float()
        metrics['lang_accuracy'] = accuracy_score(all_labels_languages, all_outputs_languages)
        metrics['lang_balanced_accuracy'] = balanced_accuracy_score(all_labels_languages, all_outputs_languages)
        metrics['lang_weighted_f1'] = f1_score(all_labels_languages, all_outputs_languages, average='weighted')
        metrics['lang_cross_entropy'] = loss_entropy_language(all_outputs_languages_torch, all_labels_languages_torch)

    if speaker_penalization and not subj_independent_data:
        all_labels_speakers_torch = torch.cat(all_labels_speakers_torch, dim=0).long()
        all_outputs_speakers_torch = torch.cat(all_outputs_speakers_torch, dim=0).float()
        for metric in metrics_speakers.keys():
            if metric == 'roc_auc':
                continue
            elif metric == 'accuracy':
                metrics_speakers[metric] = accuracy_score(all_labels_speakers, all_outputs_speakers)
            elif metric == 'balanced_accuracy':
                metrics_speakers[metric] = balanced_accuracy_score(all_labels_speakers, all_outputs_speakers)
            elif metric == "weighted_f1":
                metrics_speakers[metric] = f1_score(all_labels_speakers, all_outputs_speakers, average='weighted')
            elif metric == "cross_entropy":
                metrics_speakers[metric] = loss_entropy_speaker(all_outputs_speakers_torch, all_labels_speakers_torch)
            model.train()
        return metrics, metrics_speakers

    if type != 'train':
        cm = confusion_matrix(all_labels_cultures, all_outputs_cultures)
        # model.train()
    else:
        cm = None  # not useful in training, it slows the training process

    return metrics, cm


def train(config, train_loader, val_loader, device,
            cl_type, culture_weights=None, speaker_weights=None,
            n_cultures=4, n_speakers=0, n_languages=0, d_model=1024, sep_people="",
            raw_poses = False, pose_enc_type = None, audio_enc_type = None):

    # if n_speakers != 0 and class_weights !=None: #the speaker adversary is currently implemented for weighted loss
    sp = True if "adv_train" in sep_people or "multitask" in sep_people else False
    lp = bool(getattr(config, "language_penalization", False) or ("lang_adv" in sep_people))
    separated_people = True if "sep_people" in sep_people else False
    logging.info('len of train loader:{}, len of test loader:{}'.format(len(train_loader), len(val_loader)))

    # in the newer version input_dim_cl is computed inside the class

    # As total speakers are more than training speakers, and as the speaker ID is an incremental number,
    # by randomly picking some speakers it may happen that their ID is higher than the total length of speakers in training
    # which leads to an error as this ID is bigger than the maximum allowed by the speaker classification head. I should
    # create a new mapping...

    train_speaker_to_index_path = os.path.join(config.data_path_info, "training_speakers_id_map.pkl")
    speaker_to_index = None
    if os.path.exists(train_speaker_to_index_path):
        logging.info(f'Loading training speakers ID mapping data...')
        with open(train_speaker_to_index_path,"rb") as f:
            speaker_to_index = pickle.load(f)
        if n_speakers and len(speaker_to_index) != int(n_speakers):
            logging.warning(
                "Existing training speaker mapping size (%d) does not match current train split (%d). Rebuilding mapping.",
                len(speaker_to_index), int(n_speakers)
            )
            speaker_to_index = None
    if speaker_to_index is None:
        logging.info(f'Creating a new mapping for training speakers.')
        unique_train_speakers = set()
        for batch in train_loader:
            speakers = batch[-1]['speaker_enc'].cpu().numpy()
            unique_train_speakers.update(speakers)
        unique_train_speakers = sorted(list(unique_train_speakers))
        speaker_to_index = {speaker_id: idx for idx, speaker_id in enumerate(unique_train_speakers)}
        n_train_speakers = len(speaker_to_index)
        logging.info(f'Created new speaker mapping with {n_train_speakers} speakers.')
        with open(train_speaker_to_index_path, 'wb') as f:
            pickle.dump(speaker_to_index, f)

    # Build a fast speaker-id remapping tensor when speaker ids are reasonably small.
    speaker_map_tensor = None
    try:
        max_speaker_id = int(max(speaker_to_index.keys()))
        max_allowed = int(getattr(config, "speaker_map_max_id", 200000))
        if max_speaker_id <= max_allowed:
            speaker_map_tensor = torch.full((max_speaker_id + 1,), -1, dtype=torch.long)
            for sid, mid in speaker_to_index.items():
                speaker_map_tensor[int(sid)] = int(mid)
    except Exception:
        speaker_map_tensor = None

    language_to_index = None
    language_map_tensor = None
    if lp:
        train_language_to_index_path = os.path.join(config.data_path_info, "training_languages_id_map.pkl")
        if os.path.exists(train_language_to_index_path):
            logging.info('Loading training languages ID mapping data...')
            with open(train_language_to_index_path, "rb") as f:
                language_to_index = pickle.load(f)
        else:
            logging.info('Creating a new mapping for training languages.')
            unique_train_languages = set()
            for batch in train_loader:
                if 'language_enc' in batch[-1]:
                    langs = batch[-1]['language_enc'].cpu().numpy()
                    unique_train_languages.update(langs)
            unique_train_languages = sorted(list(unique_train_languages))
            language_to_index = {language_id: idx for idx, language_id in enumerate(unique_train_languages)}
            with open(train_language_to_index_path, "wb") as f:
                pickle.dump(language_to_index, f)

        if language_to_index is None or len(language_to_index) == 0:
            lp = False
            logging.warning("Language adversarial head requested but no language labels were found. Disabling it.")
        else:
            n_languages = len(language_to_index)
            try:
                max_language_id = int(max(language_to_index.keys()))
                max_allowed_lang = int(getattr(config, "language_map_max_id", 10000))
                if max_language_id <= max_allowed_lang:
                    language_map_tensor = torch.full((max_language_id + 1,), -1, dtype=torch.long)
                    for lid, mid in language_to_index.items():
                        language_map_tensor[int(lid)] = int(mid)
            except Exception:
                language_map_tensor = None

    model = Culture_Classifier(config, device, dropout_prob=config.dropout,
                               cl_type=cl_type, d_model=d_model, speaker_penalization=sp,
                               n_cultures=n_cultures, n_speakers=n_speakers, raw_poses=raw_poses,
                               language_penalization=lp, n_languages=n_languages,
                               pose_enc=pose_enc_type, audio_enc = audio_enc_type)  # n_joints * n_chanels


   # print("PARAMETERS:",config, "\n", device, config.dropout, cl_type, d_model, sp, n_cultures, n_speakers, raw_poses, pose_enc_type, audio_enc_type)

    #######model = nn.DataParallel(model)  # , device_ids=[eval(i) for i in config.no_cuda])
    model = model.to(device)
    #print(next(model.parameters()).device)
    contrastive_loss_fn = ContrastiveLoss(margin=1.0)
    culture_loss_entropy = nn.CrossEntropyLoss()
    speaker_loss_entropy = nn.CrossEntropyLoss()
    language_loss_entropy = nn.CrossEntropyLoss()
    #bal_loss_culture_entropy = nn.CrossEntropyLoss(weight=culture_weights)  # (weight=class_weights)
    bal_loss_culture_entropy = nn.CrossEntropyLoss(weight=culture_weights) if culture_weights is not None else culture_loss_entropy
    bal_loss_speakers_entropy = nn.CrossEntropyLoss(weight=speaker_weights) if speaker_weights is not None else speaker_loss_entropy
    model_save_path = os.path.join(config.model_save_path, cl_type)

    # best_val_loss = (1e+2, 0)  # value, epoch
    best_val_loss = (1e+2, 0)
    # loss =  1e+2
    bal_loss = 1e+2

    updates = 0
    total = len(train_loader)
    tb_path = cl_type + '_' + str(datetime.now().strftime('%Y%m%d_%H%M%S'))
    if culture_weights == None:
        info_loss = "no_weighted_loss"
    else:
        info_loss = "weighted_loss"

    if not os.path.exists(os.path.join(model_save_path, info_loss, f"train_{cl_type}")):
        os.makedirs(os.path.join(model_save_path, info_loss, f"train_{cl_type}"), exist_ok=True)

    tb_writer = SummaryWriter(
        log_dir=f"{model_save_path}/{info_loss}/tensorboard_runs{sep_people}/{tb_path}")

    # Initialize optimizer
    # optimizer = optim.SGD(model.module.parameters(), lr=config.FC.lr*100, momentum=0.8, weight_decay=1e-4, nesterov=True)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, betas=config.betas, weight_decay=1e-4)

    #scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.FC.milestones, gamma=config.FC.gamma)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)

    # --- Culture-specific DG regularizers (optional, set weights to 0.0 to disable) ---
    supcon_weight = float(getattr(config, "supcon_weight", 0.0))
    supcon_temperature = float(getattr(config, "supcon_temperature", 0.07))
    supcon_require_diff_domain = bool(getattr(config, "supcon_require_diff_domain", True))
    mixup_weight = float(getattr(config, "mixup_weight", 0.0))
    mixup_alpha = float(getattr(config, "mixup_alpha", 0.0))
    contrastive_weight = float(getattr(config, "contrastive_weight", 0.0))
    language_loss_weight = float(getattr(config, "language_loss_weight", 1.0))
    language_schedule_gamma = float(getattr(config, "language_schedule_gamma", 5.0))

    supcon_loss_fn = SupConLoss(temperature=supcon_temperature).to(device)



    best_metrics = {'accuracy': 0, 'balanced_accuracy': 0, "weighted_f1": 0, "roc_auc": 0, "cross_entropy": 1e2}
    # train_metrics = best_metrics
    speakers_train_metrics = best_metrics
    iterations_without_improvements = 0
    max_iterations_without_improvements = config.epochs


    use_contrastive_loss = bool(("contrastive" in sep_people) or (contrastive_weight > 0.0))
    sample_idx = 0
    contrastive_loss = torch.tensor(0.0, device=device)

    for epoch in range(1, config.epochs + 1):
        model.grl_flag_culture = False
        if "multitask" in sep_people or "mixed_data" in sep_people:
            ####model.module.grl_flag = False
            model.grl_flag = False  #no adversarial training
        else:
            model.grl_flag = True
            model.grl_lambda = 1

        speaker_weight = adjust_lambda_exponential(epoch, gamma = 5, max_epoch=config.epochs)  #             #adjust_speaker_weight(epoch, initial_speaker_weight, max_speaker_weight, config.FC.epochs)
        language_weight = language_loss_weight * adjust_lambda_exponential(
            epoch, gamma=language_schedule_gamma, max_epoch=config.epochs
        )
        # print("Best Balanced Accuracy",best_metrics["balanced_accuracy"])
        logging.info(f'Epoch: {epoch}')
        final_total_loss = 0
        i = 0
        # train model
        model = model.train()
        start = datetime.now()
        for batch in tqdm(train_loader):
            inputs = batch[:-1]  # target_vec, _, _ = batch
            cultures = batch[-1]['culture_enc']
            speakers = batch[-1]['speaker_enc']
            languages = batch[-1].get('language_enc', None)

            speakers = remap_speakers(speakers, speaker_to_index, speaker_map_tensor)
            if lp:
                if languages is None:
                    raise ValueError("language_penalization=True but batch does not contain 'language_enc'.")
                languages = remap_ids(languages, language_to_index, language_map_tensor, unknown_value=-1)
                if (languages < 0).any():
                    raise ValueError("Found unknown language IDs while training language adversarial head.")

            # optimizer.zero_grad()
            inputs_in_device = []
            for input in inputs:
                inputs_in_device.append(input.to(device, non_blocking=True))####.to(device))
            cultures = cultures.to(device)
            speakers = speakers.to(device)
            if lp:
                languages = languages.to(device)


            # Optional embedding-based regularizers computed on the current batch
            batch_embeddings = None
            extra_losses = {}
            if (supcon_weight > 0.0) or (mixup_weight > 0.0) or (use_contrastive_loss and contrastive_weight > 0.0):
                batch_embeddings = model(inputs_in_device, return_embeddings=True)

            if supcon_weight > 0.0:
                supcon_loss = supcon_loss_fn(batch_embeddings, cultures, domain=speakers,
                                             require_diff_domain=supcon_require_diff_domain)
                extra_losses["supcon_loss"] = supcon_loss

            if mixup_weight > 0.0 and mixup_alpha > 0.0:
                # Culture-preserving mixup in embedding space (mix only within the same culture).
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                z = batch_embeddings
                # Default to identity for classes represented by <=1 sample in the batch.
                perm_idx = torch.arange(z.shape[0], dtype=torch.long, device=z.device)

                for c in torch.unique(cultures):
                    idx = (cultures == c).nonzero(as_tuple=False).view(-1)
                    if idx.numel() <= 1:
                        continue
                    shuf = idx[torch.randperm(idx.numel(), device=z.device)]
                    perm_idx[idx] = shuf

                z_perm = z[perm_idx]
                z_mix = lam * z + (1.0 - lam) * z_perm
                z_mix = F.normalize(z_mix, p=2, dim=1)
                mix_logits = model.culture_classifier(z_mix)
                mixup_loss = culture_loss_entropy(mix_logits, cultures)
                extra_losses["mixup_loss"] = mixup_loss


            if use_contrastive_loss and contrastive_weight > 0.0:
                embeddings = batch_embeddings if batch_embeddings is not None else model(inputs_in_device, return_embeddings=True)
                pairs, labels = generate_contrastive_pairs(embeddings, cultures, speakers)
                contrastive_loss = torch.tensor(0.0, device=device)
                if len(pairs) > 0:
                    for (emb1, emb2), label in zip(pairs, labels):
                        label_tensor = label.unsqueeze(0).float()
                        contrastive_loss += contrastive_loss_fn(emb1.unsqueeze(0), emb2.unsqueeze(0), label_tensor)
                    contrastive_loss /= len(pairs)
                extra_losses['contrastive_loss'] = contrastive_loss

            culture_ce = culture_loss_entropy if culture_weights is None else bal_loss_culture_entropy
            speaker_ce = speaker_loss_entropy if speaker_weights is None else bal_loss_speakers_entropy

            outputs = model(inputs_in_device)
            culture_outputs, speaker_outputs, language_outputs = unpack_outputs(
                outputs,
                speaker_penalization=sp,
                language_penalization=lp,
            )

            culture_loss = culture_ce(culture_outputs, cultures)
            total_loss = culture_loss
            stats = {'updates': updates, 'tot_loss': 0.0, 'culture_loss': culture_loss.item()}

            if sp:
                speaker_loss = speaker_ce(speaker_outputs, speakers)
                total_loss = total_loss + speaker_weight * speaker_loss
                stats['speaker_loss'] = speaker_loss.item()

            if lp:
                language_loss = language_loss_entropy(language_outputs, languages)
                total_loss = total_loss + language_weight * language_loss
                stats['language_loss'] = language_loss.item()

            if 'supcon_loss' in extra_losses:
                total_loss = total_loss + supcon_weight * extra_losses['supcon_loss']
            if 'mixup_loss' in extra_losses:
                total_loss = total_loss + mixup_weight * extra_losses['mixup_loss']
            if use_contrastive_loss and contrastive_weight > 0.0 and 'contrastive_loss' in extra_losses:
                total_loss = total_loss + contrastive_weight * extra_losses['contrastive_loss']
                stats['contrastive_loss'] = extra_losses['contrastive_loss'].item()

            stats['tot_loss'] = total_loss.item()


            tb_writer.add_scalar('Total Loss', total_loss, updates)
            tb_writer.add_scalar('Culture Loss', culture_loss, updates)
            if 'supcon_loss' in extra_losses:
                tb_writer.add_scalar('SupCon Loss', extra_losses['supcon_loss'], updates)
            if 'mixup_loss' in extra_losses:
                tb_writer.add_scalar('Mixup Loss', extra_losses['mixup_loss'], updates)
            if use_contrastive_loss and 'contrastive_loss' in extra_losses:
                tb_writer.add_scalar('Contrastive Loss', extra_losses['contrastive_loss'], updates)
            if sp:
                tb_writer.add_scalar('Speaker Loss', speaker_loss, updates)
            if lp:
                tb_writer.add_scalar('Language Loss', language_loss, updates)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            final_total_loss += total_loss
            stats_str = ' '.join(f'{key}[{val:.4f}]' for key, val in stats.items())
            i += 1
            remaining = str((datetime.now() - start) / i * (total - i))
            remaining = remaining.split('.')[0]
            sample_idx += 1
            if sample_idx % 100 == 0:
                logging.info(f'> epoch [{epoch}] updates[{i}] {stats_str} eta[{remaining}]')

            updates += 1
            #torch.cuda.empty_cache()
            #gc.collect()

        ### Validate the model and save info
        # model.eval() #already done in evaluate set
        val_metrics, val_aux = evaluate_set(
            model,
            val_loader,
            device,
            n_levels=n_cultures,
            type="train",
            speaker_penalization=sp,
            language_penalization=lp,
            subj_independent_data=separated_people,
        )
        print("VAL:", val_metrics)

        if sp and not separated_people and isinstance(val_aux, dict):
            speakers_train_metrics = val_aux
            tb_writer.add_scalar('balanced accuracy speakers on training', speakers_train_metrics['balanced_accuracy'],
                                 epoch)
            tb_writer.add_scalar('cross entropy speaker training', speakers_train_metrics["cross_entropy"], epoch)

        if use_contrastive_loss and contrastive_weight > 0.0:
            tb_writer.add_scalar('contrastive loss', contrastive_loss, epoch)
        tb_writer.add_scalar('balanced accuracy on validation', val_metrics['balanced_accuracy'], epoch)
        #tb_writer.add_scalar('balanced accuracy culture on training', train_metrics['balanced_accuracy'], epoch)
        tb_writer.add_scalar('weighted F1 score on validation', val_metrics["weighted_f1"], epoch)
        tb_writer.add_scalar('cross-entropy on validation', val_metrics["cross_entropy"], epoch)
        #tb_writer.add_scalar('cross entropy culture training', train_metrics["cross_entropy"], epoch)
        tb_writer.add_scalar('weighted ROC AUC score on validation', val_metrics['roc_auc'], epoch)
        if lp:
            tb_writer.add_scalar('language accuracy on validation', val_metrics.get('lang_accuracy', 0), epoch)
            tb_writer.add_scalar('language balanced accuracy on validation', val_metrics.get('lang_balanced_accuracy', 0), epoch)
            lang_ce = val_metrics.get('lang_cross_entropy', torch.tensor(0.0))
            tb_writer.add_scalar('language cross-entropy on validation', lang_ce, epoch)
        tb_writer.add_scalar('Epoch training loss', final_total_loss, epoch)

        scheduler.step()

        is_best = val_metrics['cross_entropy'].item() < best_val_loss[0]

        if is_best:
            iterations_without_improvements = 0
            best_metrics = val_metrics
            logging.info(f'*** BEST METRICS FOUND AT EPOCH {epoch}: {best_metrics}')
            best_val_loss = (val_metrics['cross_entropy'].item(), epoch)
            # best_val_loss = (diff,epoch)
        else:
            iterations_without_improvements += 1
            logging.info(f'*** BEST METRICS SO FAR AT EPOCH {best_val_loss[1]}:{best_metrics}')
            if iterations_without_improvements > max_iterations_without_improvements:
                logging.info(f' NO IMPROVEMENTS FOR {max_iterations_without_improvements} steps. STOPPING TRAINING...')
                break

        if is_best or (epoch % config.save_per_epochs == 0):
            if is_best:
                save_name = '{}/{}/{}/{}_checkpoint_best{}.bin'.format(model_save_path, info_loss,
                                                                          f"train_{cl_type}",
                                                                          cl_type, sep_people)
            else:
                save_name = '{}/{}/{}/{}_checkpoint_{:03d}{}.bin'.format(model_save_path,info_loss,
                                                                         f"train_{cl_type}",
                                                                         cl_type, epoch, sep_people)
            torch.save({
                'config': config, "epoch": epoch, 'model_dict': model.state_dict()
            }, save_name)


    tb_writer.close()
    # print best losses
    logging.info('--------- Final best loss values ---------')
    logging.info('best validation cross-entropy: {:.4f} at EPOCH {}'.format(best_val_loss[0], best_val_loss[1]))
