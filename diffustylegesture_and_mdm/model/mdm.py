import pdb

import numpy as np
import torch
import os
import glob
import torch.nn as nn
import torch.nn.functional as F
import yaml
from easydict import EasyDict

try:
    from diffustylegesture_and_mdm.model.local_attention.rotary import (
        SinusoidalEmbeddings,
        apply_rotary_pos_emb,
    )
    from diffustylegesture_and_mdm.model.local_attention.local_attention import LocalAttention
except Exception:
    from model.local_attention.rotary import SinusoidalEmbeddings, apply_rotary_pos_emb
    from model.local_attention.local_attention import LocalAttention

try:
    from culture_encoder.adversarial_classifier import Culture_Classifier
    from culture_encoder.fishr_classifier import Fishr
except Exception:
    from data.adversarial_classifier.culture_classifier import Culture_Classifier
    from data.culture_classifier import Fishr


def _resolve_repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _normalize_path(path_value):
    if not path_value:
        return None
    path_value = os.path.expanduser(str(path_value))
    if not os.path.isabs(path_value):
        path_value = os.path.join(_resolve_repo_root(), path_value)
    return os.path.abspath(path_value)


def _first_existing_file(candidates):
    tried = []
    for candidate in candidates:
        normalized = _normalize_path(candidate)
        if not normalized:
            continue
        tried.append(normalized)
        if os.path.isfile(normalized):
            return normalized, tried
    return None, tried

class MDM(nn.Module):
    # args.cond_mode = 'cross_local_attention4_style1'
    # args.njoints = 2052
    # args.audio_feat = wavlm
    # args.latent_dim = 512
    # args.motion_dim = 512
    # args.n_seed = 5
    # args.cond_mask_prob = 0.1
    # args.style_dim = 1536
    # args.audio_feature_dim = 1792
    # args.audio_feat_dim_latent = 96
    # device_name = 'cuda:0'
    # modeltype=''
    # nfeats = 1
    # device = cuda:0


    def __init__(self, modeltype, njoints, nfeats,
                 latent_dim=512, ff_size=2048, num_layers=10, num_heads=8, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d', dataset='amass', clip_dim=512,
                 arch='trans_enc', emb_trans_dec=False, audio_feat='', n_seed=1, cond_mode='', device='cpu', 
                 style_dim=-1, source_audio_dim=-1, audio_feat_dim_latent=-1, motion_dim = 512,
                 wav2vec_dim = 1024, mel_dim = 64, n_poses = 25, text_dim = 768, use_adversarial = False, **kargs):
        super().__init__()

        #self.legacy = legacy
        #self.modeltype = modeltype
        self.njoints = njoints
        #self.nfeats = nfeats
        #self.data_rep = data_rep
        #self.dataset = dataset

        self.latent_dim = latent_dim
        self.motion_dim = motion_dim

        self.ff_size = ff_size #1024
        self.num_layers = num_layers #10
        self.num_heads = num_heads #8
        self.dropout = dropout #.1
        self.n_train_speakers = kargs.get('n_train_speakers', 1)
        self.use_culture = kargs.get('use_culture', False)
        #print("HERE",self.use_culture, cond_mode, num_layers, num_heads)
        self.n_poses = n_poses
        self.n_seed = n_seed
        self.wav2vec_dim = wav2vec_dim
        self.mel_dim = mel_dim
        self.device = device
        self.batch_size = kargs.get('batch_size', 64)
        self.use_adversarial = use_adversarial
        self.use_alignment_module = bool(kargs.get("use_alignment_module", False))
        self.use_culture_guidance_loss = bool(kargs.get("use_culture_guidance_loss", False))
        self.n_cultures = int(kargs.get("n_cultures", 4))
        self.culture_embedder_config = kargs.get("culture_embedder_config", None)
        self.culture_embedder = None
        self.culture_embedder_weights_path = None
        self.fishr_uses_predict = False
        self.culture_embedding_dim = int(
            kargs.get(
                "culture_embedding_dim",
                style_dim if isinstance(style_dim, int) and style_dim > 0 else 512,
            )
        )
        if self.use_culture:
            if self.use_adversarial:
                self._init_adversarial_culture_embedder(kargs)
            else:
                self._init_fishr_culture_embedder(kargs)

        #self.ablation = ablation
        self.activation = activation #gelu
        #self.clip_dim = clip_dim

        self.input_feats = 512 #self.njoints * self.nfeats #25

        self.normalize_output = kargs.get('normalize_encoder_output', False) # False

        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.) # .1
        self.arch = arch #trans_enc
        self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0 #0
        
        self.source_audio_dim = source_audio_dim #1857 Note that contains also text, that's why now is called feature encoder
        self.audio_feat = audio_feat #wavlm

        #print('USE WAVLM')
        self.audio_feat_dim = audio_feat_dim_latent   #96     # Linear 1024 -> 64
        self.featureEncoder = FeatureEncoder(self.source_audio_dim, self.audio_feat_dim) #just a linear layer. ok

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        #self.emb_trans_dec = emb_trans_dec

        self.cond_mode = cond_mode #cross_local_attention4_style1
        #self.num_head = 8

        self.audio_pooler = WindowedAttentionPooling(input_dim=self.mel_dim+1, output_dim=self.mel_dim+1,
                                                     window_size=11, stride=6, target_length=self.n_poses-self.n_seed)
        self.wav2vec_pooler = WindowedAttentionPooling(input_dim=self.wav2vec_dim, output_dim=self.wav2vec_dim,
                                                       window_size=2, stride=2, target_length=self.n_poses-self.n_seed)
        if self.use_alignment_module:
            def _alignment_activation():
                return nn.GELU() if activation == "gelu" else nn.LeakyReLU()

            self.alignment_onset_mel_embedding = nn.Sequential(
                nn.Linear(1 + self.mel_dim, self.latent_dim // 2),
                nn.LayerNorm(self.latent_dim // 2),
                _alignment_activation(),
                nn.Dropout(self.dropout),
            )
            self.alignment_wav2vec_embedding = nn.Sequential(
                nn.Linear(self.wav2vec_dim, self.latent_dim // 2),
                nn.LayerNorm(self.latent_dim // 2),
                _alignment_activation(),
                nn.Dropout(self.dropout),
            )
            self.alignment_audio_pooler_1 = WindowedAttentionPooling(
                input_dim=self.latent_dim // 2,
                output_dim=self.latent_dim // 2,
                window_size=9,
                stride=3,
                target_length=self.n_poses * 2,
            )
            self.alignment_audio_pooler_2 = WindowedAttentionPooling(
                input_dim=self.latent_dim,
                output_dim=self.latent_dim,
                window_size=2,
                stride=2,
                target_length=self.n_poses,
            )
            self.alignment_motion_pooler = AttentionPooling(self.latent_dim)
            self.alignment_low_level_context_collapsed = AttentionPooling(self.latent_dim)
            high_level_input_dim = text_dim + (self.culture_embedding_dim if self.use_culture else 0)
            self.alignment_high_level_proj = nn.Sequential(
                nn.Linear(high_level_input_dim, self.latent_dim),
                nn.LayerNorm(self.latent_dim),
                _alignment_activation(),
                nn.Dropout(self.dropout),
            )
        if self.use_culture_guidance_loss:
            guidance_activation = nn.GELU() if activation == "gelu" else nn.LeakyReLU()
            self.culture_classification_layer = nn.Sequential(
                nn.Linear(self.motion_dim, self.latent_dim),
                AttentionPooling(self.latent_dim),
                nn.BatchNorm1d(self.latent_dim),
                guidance_activation,
                nn.Dropout(self.dropout),
                nn.Linear(self.latent_dim, self.n_cultures),
            )
        else:
            self.culture_classification_layer = None
        #self.audio_pooler_1 = WindowedAttentionPooling(input_dim=self.mel_dim+1, output_dim=self.mel_dim+1,
        #                                               window_size=9, stride=3, target_length=self.n_poses-self.n_seed)
        #self.audio_pooler_2 = WindowedAttentionPooling(input_dim=self.latent_dim, output_dim=self.latent_dim,
        #                                               window_size=2, stride=2, target_length=self.vqvae_emb_dim[0])


        #if 'style2' not in self.cond_mode: # YES!
        #    self.input_process = InputProcess(self.data_rep, self.input_feats + self.audio_feat_dim + self.gru_emb_dim, self.latent_dim)

        if self.arch == 'trans_enc': #YES!
            print("TRANS_ENC init")
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation)

            self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                         num_layers=self.num_layers)
        else:
            raise ValueError('Please choose correct architecture [trans_enc]')

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        self.n_seed = n_seed # 5
        if 'style1' in self.cond_mode: # YES!
            print('EMBED STYLE BEGIN TOKEN')
            if 'cross_local_attention3' in self.cond_mode: #no
                self.style_dim = 64
                self.embed_style = nn.Linear(style_dim, self.style_dim)
                self.embed_text = nn.Linear(self.njoints * n_seed, self.latent_dim - self.style_dim)
            elif 'cross_local_attention4' in self.cond_mode: # YES!
                self.style_dim = self.latent_dim #style dim, i.e. culture, becomes 384 from 1536
                style_input_dim = self.culture_embedding_dim if self.use_culture else (
                    style_dim if isinstance(style_dim, int) and style_dim > 0 else self.culture_embedding_dim
                )
                self.embed_style = nn.Linear(style_input_dim, self.style_dim) # From culture embedding to latent dim
                #self.embed_text = nn.Linear(self.njoints, self.audio_feat_dim) #TODO not clear why. Check
                self.embed_text = nn.Linear(self.motion_dim, self.audio_feat_dim)
            elif 'cross_local_attention5' in self.cond_mode: #no
                self.style_dim = self.latent_dim
                self.embed_style = nn.Linear(style_dim, self.style_dim)
                self.embed_text = nn.Linear(self.njoints, self.audio_feat_dim)
                self.embed_text_last = nn.Linear(self.njoints, self.audio_feat_dim)

        elif 'style2' in self.cond_mode: # no
            print('EMBED STYLE ALL FRAMES')
            self.style_dim = 64
            self.embed_style = nn.Linear(style_dim, self.style_dim)
            self.input_process = InputProcess(self.data_rep, self.input_feats + self.audio_feat_dim + self.gru_emb_dim + self.style_dim,
                                              self.latent_dim)
            if self.n_seed != 0:
                self.embed_text = nn.Linear(self.njoints * n_seed, self.latent_dim)
        #elif self.n_seed != 0: # no because previous condtion, "style1" is respected
        #    self.embed_text = nn.Linear(self.njoints * n_seed, self.latent_dim)


        if 'mdm' in self.cond_mode:
            self.input_process = nn.Linear(self.motion_dim, self.latent_dim)
            if self.use_culture:
                self.embed_style = nn.Linear(self.culture_embedding_dim + text_dim, self.latent_dim)
            else:
                self.embed_style = nn.Linear(text_dim, self.latent_dim)

        # TODO change the output process
        # data_rep = rot6d but I am using vqvae. self.input_feats from VQVAE, self.latent_dim = 384, self.njoints is not important as well as n_feats
        #self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints,
        #                                    self.nfeats)
        self.output_process = OutputProcess(self.latent_dim, motion_dim, n_poses)

        if 'cross_local_attention' in self.cond_mode: #YES
            self.rel_pos = SinusoidalEmbeddings(self.latent_dim // self.num_heads)
            #self.input_process = InputProcess(self.data_rep, self.input_feats + self.gru_emb_dim, self.latent_dim) #TODO CHANGE THE INPUT PROCESS
            self.input_process = nn.Linear(self.motion_dim, self.latent_dim)
            self.cross_local_attention = LocalAttention(
                dim=48,  # dimension of each head (you need to pass this in for relative positional encoding)
                window_size=5,  # window size. 512 is optimal, but 256 or 128 yields good enough results. changed from 15 too 5
                causal=True,  # auto-regressive or not
                look_backward=1,  # each window looks at the window before
                look_forward=0,     # for non-auto-regressive case, will default to 1, so each window looks at the window before and after it
                dropout=0.1,  # post-attention dropout
                exact_windowsize=False
                # if this is set to true, in the causal setting, each query will see at maximum the number of keys equal to the window size
            )
            self.input_process2 = nn.Linear(self.latent_dim * 2 + self.audio_feat_dim, self.latent_dim)


    def _build_culture_config(self, kwargs):
        cfg = self.culture_embedder_config
        if cfg is None:
            cfg_candidates = [
                kwargs.get("culture_config_path", None),
                "culture_encoder/config.yml",
                "diffustylegesture_and_mdm/data/adversarial_classifier/config.yml",
            ]
            cfg = None
            for candidate in cfg_candidates:
                resolved = _normalize_path(candidate)
                if resolved and os.path.isfile(resolved):
                    with open(resolved, "r") as f:
                        cfg = EasyDict(yaml.safe_load(f))
                    break
            if cfg is None:
                cfg = EasyDict({})
        elif isinstance(cfg, dict):
            cfg = EasyDict(cfg)
        cfg.setdefault("dropout", 0.1)
        cfg.setdefault("layer_neurons", 512)
        cfg.setdefault("levels", 1)
        cfg.setdefault("embed_dim", cfg.get("layer_neurons", 512))
        cfg.setdefault("pose_enc_type", "transformer")
        cfg.setdefault("audio_enc_type", None)
        return cfg

    def _init_fishr_culture_embedder(self, kwargs):
        cfg = self._build_culture_config(kwargs)
        fishr_d_model = int(getattr(cfg, "layer_neurons", 512) or 512)
        self.culture_embedding_dim = int(getattr(cfg, "embed_dim", fishr_d_model) or fishr_d_model)
        try:
            self.culture_embedder = Fishr(
                proj_dim=fishr_d_model,
                num_classes=4,
                num_domains=self.n_train_speakers,
                is_nonlinear=False,
                use_motion=False,
                cl_type="culclI",
                d_model=fishr_d_model,
                pose_enc_type=getattr(cfg, "pose_enc_type", "transformer"),
                audio_enc_type=getattr(cfg, "audio_enc_type", None),
                raw_poses=False,
                backbone_config=cfg,
                use_adversarial_backbone=True,
            ).to(self.device)
        except TypeError:
            self.culture_embedder = Fishr(
                proj_dim=fishr_d_model,
                num_classes=4,
                num_domains=self.n_train_speakers,
                is_nonlinear=False,
                use_motion=False,
            ).to(self.device)

        fishr_candidates = [
            kwargs.get("fishr_model_path", None),
            getattr(cfg, "fishr_model_path", None),
            getattr(cfg, "fishr_save_dir", None),
            getattr(cfg, "model_save_path", None),
            "culture_encoder/fishr_train/full_data_culclI/model_best.pt",
            "culture_encoder/fishr_train/full_data_culclI",
            "culture_encoder/fishr_train/full_data",
            "diffustylegesture_and_mdm/data/fishr_motion/model_best.pt",
        ]
        fishr_files = []
        for candidate in fishr_candidates:
            normalized = _normalize_path(candidate)
            if not normalized:
                continue
            if os.path.isfile(normalized):
                fishr_files.append(normalized)
            else:
                fishr_files.append(os.path.join(normalized, "model_best.pt"))

        self.culture_embedder_weights_path, tried_paths = _first_existing_file(fishr_files)
        if self.culture_embedder_weights_path is None:
            raise FileNotFoundError(
                "Could not locate Fishr checkpoint for baseline model. Tried:\n  - "
                + "\n  - ".join(tried_paths)
            )
        state = torch.load(self.culture_embedder_weights_path, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.culture_embedder.load_state_dict(state, strict=False)
        for param in self.culture_embedder.parameters():
            param.requires_grad = False
        self.culture_embedder.eval()
        self.fishr_uses_predict = bool(getattr(self.culture_embedder, "use_adversarial_backbone", False))
        print(
            f"[Baseline MDM] Culture encoder: fishr 'culclI' (dim={self.culture_embedding_dim}) "
            f"from {self.culture_embedder_weights_path}"
        )

    def _init_adversarial_culture_embedder(self, kwargs):
        cfg = self._build_culture_config(kwargs)
        culture_embedder_name = "culclI"
        culture_embedder_type = "_sep_people"
        self.culture_embedding_dim = 512
        self.culture_embedder = Culture_Classifier(
            cfg,
            self.device,
            dropout_prob=getattr(cfg, "dropout", 0.1),
            cl_type=culture_embedder_name,
            d_model=self.culture_embedding_dim,
            speaker_penalization=False,
            n_cultures=4,
            n_speakers=self.n_train_speakers,
            pose_enc=getattr(cfg, "pose_enc_type", "transformer"),
            audio_enc=getattr(cfg, "audio_enc_type", None),
            raw_poses=False,
        ).to(self.device)

        adv_explicit = kwargs.get("adversarial_checkpoint_path", None) or getattr(
            cfg, "adversarial_checkpoint_path", None
        )
        adv_root_candidates = [
            kwargs.get("adversarial_model_save_path", None),
            getattr(cfg, "adversarial_model_save_path", None),
            getattr(cfg, "model_save_path", None),
            "culture_encoder/adversarial_train",
            "diffustylegesture_and_mdm/data/adversarial_classifier",
        ]
        adv_filename = f"{culture_embedder_name}_checkpoint_best{culture_embedder_type}.bin"
        adv_file_candidates = [adv_explicit]
        for root_candidate in adv_root_candidates:
            if not root_candidate:
                continue
            train_dir = os.path.join(
                str(root_candidate),
                culture_embedder_name,
                "no_weighted_loss",
                f"train_{culture_embedder_name}",
            )
            normalized_train_dir = _normalize_path(train_dir) or ""
            adv_file_candidates.append(os.path.join(train_dir, adv_filename))
            adv_file_candidates.extend(
                sorted(
                    glob.glob(
                        os.path.join(
                            normalized_train_dir,
                            f"{culture_embedder_name}_checkpoint_best*.bin",
                        )
                    )
                )
            )
        self.culture_embedder_weights_path, tried_paths = _first_existing_file(adv_file_candidates)
        if self.culture_embedder_weights_path is None:
            raise FileNotFoundError(
                "Could not locate adversarial checkpoint for baseline model. Tried:\n  - "
                + "\n  - ".join(tried_paths)
            )
        checkpoint = torch.load(self.culture_embedder_weights_path, map_location=torch.device(self.device))
        model_dict = checkpoint["model_dict"] if isinstance(checkpoint, dict) and "model_dict" in checkpoint else checkpoint
        filtered_state_dict = {k: v for k, v in model_dict.items() if "speaker_classifier" not in k}
        self.culture_embedder.load_state_dict(filtered_state_dict, strict=False)
        for param in self.culture_embedder.parameters():
            param.requires_grad = False
        self.culture_embedder.eval()
        print(
            f"[Baseline MDM] Culture encoder: adversarial 'culclI' (dim={self.culture_embedding_dim}) "
            f"from {self.culture_embedder_weights_path}"
        )

    def _extract_fishr_embedding(self, motion, text_features, audio_mels, audio_onsets, audio_wav2vec):
        with torch.no_grad():
            self.culture_embedder.eval()
            if self.fishr_uses_predict and hasattr(self.culture_embedder, "predict"):
                fishr_onsets = audio_onsets
                if fishr_onsets.dim() == 3 and fishr_onsets.shape[-1] == 1:
                    fishr_onsets = fishr_onsets.squeeze(-1)
                fishr_inputs = (
                    motion.to(self.device),
                    text_features.to(self.device),
                    audio_mels.to(self.device),
                    fishr_onsets.to(self.device),
                    audio_wav2vec.to(self.device),
                )
                prediction = self.culture_embedder.predict(fishr_inputs, return_embeddings=True)
                if isinstance(prediction, tuple):
                    return prediction[1]
                return prediction

            features = {
                "sentence": text_features.to(self.device),
                "mel": audio_mels.to(self.device),
                "onset": audio_onsets.to(self.device),
                "wav2vec": audio_wav2vec.to(self.device),
            }
            return self.culture_embedder.feature_extract(features)

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def forward(self, data, timesteps, motion_seed):

        motion, text_features, audio_mels, audio_onsets, audio_wav2vec = data
        batch_size = motion.shape[0]
        audio_onsets = audio_onsets.unsqueeze(-1)  # OFFICIAL! [batch_size, onset_seq_len, 1]
        text_context = text_features
        culture_emb = None
        #motion_seed = motion[:, :self.n_seed] # OFFICIAL! [batch_size, 5, 512]

        """
        x: [batch_size, max_frames, nfeats], denoted x_t in the paper
        timesteps: [batch_size] (int)
        seed: [batch_size, 5, nfeats]
        """

        bs, nframes, nfeats, = motion.shape   # OFFICIAL! [bs, 25, 512]         # 64, 251, 1, 196
        emb_t = self.embed_timestep(timesteps)  # OFFICIAL! [1, bs, 512]         (1, 2, 256)

        #force_mask = y.get('uncond', False)  # False
        force_mask = False

        #embed_style = self.mask_cond(self.embed_style(y['style']), force_mask=force_mask)       # (bs, 64)
        # No in classical MDM formulation
        if 'cross_local_attention4' in self.cond_mode: # HERE!
            #print(f"Using gesturediffu style with {self.use_culture}")
            if self.use_culture and not self.use_adversarial:
                culture_emb = self._extract_fishr_embedding(
                    motion=motion,
                    text_features=text_features,
                    audio_mels=audio_mels,
                    audio_onsets=audio_onsets,
                    audio_wav2vec=audio_wav2vec,
                )
                embed_style = self.mask_cond(
                    self.embed_style(culture_emb), force_mask=force_mask
                )  # OFFICIAL! (bs, latent_dim)
            elif self.use_culture and self.use_adversarial:
                with torch.no_grad():
                    culture_emb = self.culture_embedder(data, return_embeddings=True)
                    embed_style = self.mask_cond(self.embed_style(culture_emb),
                                                 force_mask=force_mask)  # OFFICIAL! (bs, latent_dim)
            else:
                embed_style = torch.zeros(bs, self.latent_dim).to(self.device)
            #embed_text = self.embed_text(y['seed'].squeeze(2).permute(0, 2, 1)).permute(1, 0, 2)  # (30, bs, 256-64)   (30, bs, 96)
            embed_seed = self.embed_text(motion_seed).permute(1, 0, 2) # OFFICIAL! [5, bs, 96]


            audio_features = torch.cat([audio_mels, audio_onsets], dim=2) # OFFICIAL! [bs, 156, 65]
            #print("audio_features", audio_features.shape)
            audio_features = audio_features[:,31:,:]  # OFFICIAL! [bs, 125, 65], related to only noised motion data (first second is skipped, these are captured at 31.2 fps, so we take from 31 to the end of onsets)
            #print("audio_features2", audio_features.shape)
            audio_features = self.audio_pooler(audio_features) # OFFICIAL! [bs, 20, 65]; aligned with motion frames
            wav2vec_features = audio_wav2vec[:,10:,:] # OFFICIAL! [bs, 40, 1024], related to only noised motion data (from one second features to the end)
            wav2vec_features = self.wav2vec_pooler(wav2vec_features) # OFFICIAL! [bs, 20, 1024]
            text_features = text_features.unsqueeze(1)  # OFFICIAL! [bs, 1, 768]
            text_features = text_features.repeat(1, 20, 1)  # OFFICIAL! [bs, 20, 768]
            all_features = torch.cat([wav2vec_features, audio_features, text_features], dim=2) # OFFICIAL! [bs, 20, 1857]
            enc_features = self.featureEncoder(all_features).permute(1, 0, 2) # OFFICIAL! [20, bs, 96]

            #enc_text = self.featureEncoder(y['audio']).permute(1, 0, 2) # seq_len = 240 - 30 (seed) = 190. enc_text = (210, bs, 96)
            #enc_text = torch.cat((embed_text, enc_text), axis=0) # ( 240, bs, 96)
            enc_features = torch.cat((embed_seed, enc_features), axis=0) # OFFICIAL! [25, bs, 96]
            #x = x.reshape(bs, njoints * nfeats, 1, nframes)  # [2, 135, 1, 240]  (bs, 135, 240)
            # self-attention
            #x_ = self.input_process(x)  # [2, 135, 1, 240] -> [240, 2, 256] (240, bs, 256)
            x_ = self.input_process(motion).permute(1, 0, 2) # OFFICIAL! [25, bs, latent_dim]
            # local-cross-attention
            packed_shape = [torch.Size([bs, self.num_heads])]
            #xseq = torch.cat((x_, enc_text), axis=2)  # [bs, d+joints*feat, 1, #frames], (240, 2, 32) #TODO this is not clear
            #print("HERE",enc_features.shape, x_.shape, packed_shape)
            xseq = torch.cat((x_, enc_features), axis=2)  # OFFICIAL![25, bs, 96 + latent_dim]
            # all frames
            embed_style_2 = (embed_style + emb_t).repeat(nframes, 1, 1)  # OFFICIAL![25, bs, latent_dim]    # (bs, 64) -> (len, bs, 64)

            xseq = torch.cat((embed_style_2, xseq), axis=2)  # OFFICIAL![25, bs, 96 + 2 * latent_dim]
            #xseq = torch.cat((embed_style_2, xseq), axis=2)  # (seq, bs, dim)
            xseq = self.input_process2(xseq)    # OFFICIAL![25, bs, latent_dim]
            xseq = xseq.permute(1, 0, 2)   # OFFICIAL![bs, 25, latent_dim] (bs, len, dim)
            xseq = xseq.view(bs, nframes, self.num_heads, -1) # OFFICIAL![bs, 25, 8, 64] - 64 is obtained by considering latent_dim = 512 / 8
            xseq = xseq.permute(0, 2, 1, 3)  # OFFICIAL![bs, 8, 25, 64]     Need (2, 8, 2048, 64)
            xseq = xseq.reshape(bs * self.num_heads, nframes, -1) # OFFICIAL![bs * 8, 25, 64]
            pos_emb = self.rel_pos(xseq)  #      (89, 32)
            xseq, _ = apply_rotary_pos_emb(xseq, xseq, pos_emb)
            #print("HERE",xseq.shape, xseq.shape)
            local_attn_mask = torch.ones(
                (bs, nframes), dtype=torch.bool, device=xseq.device
            )
            xseq = self.cross_local_attention(
                xseq, xseq, xseq, packed_shape=packed_shape, mask=local_attn_mask
            ) # OFFICIAL![bs * 8, 25, 64]

            #xseq = self.cross_local_attention(xseq, xseq, xseq, packed_shape=packed_shape,
            #                                  mask=y['mask_local'])  # OFFICIAL![bs, 8, 25, 64]  (2, 8, 2048, 64)
            xseq = xseq.permute(0, 2, 1, 3)  # OFFICIAL![bs, 25, 8, 64]  # (bs, len, 8, 64)
            xseq = xseq.reshape(bs, nframes, -1) # OFFICIAL![bs, 25, 512]
            xseq = xseq.permute(1, 0, 2) # OFFICIAL![25, bs, 512]

            xseq = torch.cat((embed_style + emb_t, xseq), axis=0) # OFFICIAL![26, bs, 512]   # [seqlen+1, bs, d]     # [(1, 2, 256), (240, 2, 256)] -> (241, 2, 256)
            xseq = xseq.permute(1, 0, 2) # OFFICIAL![bs, 26, 512]   # (bs, len, dim)
            xseq = xseq.view(bs, nframes + 1, self.num_heads, -1) # OFFICIAL![bs, 26, 8, 64]
            xseq = xseq.permute(0, 2, 1, 3)  # OFFICIAL![bs, 8, 26, 64]  Need (2, 8, 2048, 64)
            xseq = xseq.reshape(bs * self.num_heads, nframes + 1, -1) # OFFICIAL![bs * 8, 26, 64]
            pos_emb = self.rel_pos(xseq)  # (89, 32)
            xseq, _ = apply_rotary_pos_emb(xseq, xseq, pos_emb)
            xseq_rpe = xseq.reshape(bs, self.num_heads, nframes + 1, -1) # OFFICIAL![bs, 8, 26, 64]
            xseq = xseq_rpe.permute(0, 2, 1, 3)  # OFFICIAL![bs, 26, 8, 64] [seqlen+1, bs, d]
            xseq = xseq.view(bs, nframes + 1, -1) # OFFICIAL![bs, 26, 512]
            xseq = xseq.permute(1, 0, 2) # OFFICIAL![26, bs, 512]
            output = self.seqTransEncoder(xseq)[1:].permute(1,0,2) # OFFICIAL![bs, 25, 512]


        elif 'mdm' in self.cond_mode:
            #print(f"Using mdm model with use_culture {self.use_culture}.")
            if self.use_culture and not self.use_adversarial:
                culture_emb = self._extract_fishr_embedding(
                    motion=motion,
                    text_features=text_features,
                    audio_mels=audio_mels,
                    audio_onsets=audio_onsets,
                    audio_wav2vec=audio_wav2vec,
                )
                style_emb = torch.cat((culture_emb,text_features), dim=1)
                embed_style = self.mask_cond(self.embed_style(style_emb),
                                             force_mask=force_mask)  # (bs, latent_dim)

            elif self.use_culture and self.use_adversarial:
                with torch.no_grad():
                    culture_emb = self.culture_embedder(data, return_embeddings=True)
                    style_emb = torch.cat((culture_emb, text_features), dim=1)
                    embed_style = self.mask_cond(self.embed_style(style_emb),
                                                 force_mask=force_mask)  # OFFICIAL! (bs, latent_dim)
            else:
                embed_style = self.mask_cond(self.embed_style(text_features), force_mask=force_mask) # (bs, latent_dim)

            embed_style = embed_style.unsqueeze(0)  # (1, bs, latent_dim)
            emb_t += embed_style # [1, bs, latent_dim]
            x = self.input_process(motion).permute(1, 0, 2)  # [25, bs, latent_dim]

            xseq = torch.cat((emb_t, x), axis=0) # [26, bs, latent_dim]
            xseq = self.sequence_pos_encoder(xseq)  # [26 bs, d]
            output = self.seqTransEncoder(xseq)[1:].permute(1,0,2)  # , src_key_padding_mask=~maskseq)  # [ bs, 25, d]



        elif 'cross_local_attention3' in self.cond_mode: # no
            embed_text = self.embed_text(self.mask_cond(y['seed'].squeeze(2).reshape(bs, -1), force_mask=force_mask))  # (bs, 256-64)
            emb_1 = torch.cat((embed_style, embed_text), dim=1)
            enc_text = self.featureEncoder(y['audio']).permute(1, 0, 2)
            x = x.reshape(bs, njoints * nfeats, 1, nframes)  # [2, 135, 1, 240]
            # self-attention
            x_ = self.input_process(x)  # [2, 135, 1, 240] -> [240, 2, 256]

            # local-cross-attention
            packed_shape = [torch.Size([bs, self.num_head])]
            xseq = torch.cat((x_, enc_text), axis=2)  # [bs, d+joints*feat, 1, #frames], (240, 2, 32)
            # all frames
            embed_style_2 = (emb_1 + emb_t).repeat(nframes, 1, 1)  # (bs, 64) -> (len, bs, 64)
            xseq = torch.cat((embed_style_2, xseq), axis=2)  # (seq, bs, dim)
            xseq = self.input_process2(xseq)
            xseq = xseq.permute(1, 0, 2)  # (bs, len, dim)
            xseq = xseq.view(bs, nframes, self.num_head, -1)
            xseq = xseq.permute(0, 2, 1, 3)  # Need (2, 8, 2048, 64)
            xseq = xseq.reshape(bs * self.num_head, nframes, -1)
            pos_emb = self.rel_pos(xseq)  # (89, 32)
            xseq, _ = apply_rotary_pos_emb(xseq, xseq, pos_emb)
            xseq = self.cross_local_attention(xseq, xseq, xseq, packed_shape=packed_shape,
                                              mask=y['mask_local'])  # (2, 8, 2048, 64)
            xseq = xseq.permute(0, 2, 1, 3)  # (bs, len, 8, 64)
            xseq = xseq.reshape(bs, nframes, -1)
            xseq = xseq.permute(1, 0, 2)

            xseq = torch.cat((emb_1 + emb_t, xseq), axis=0)  # [seqlen+1, bs, d]     # [(1, 2, 256), (240, 2, 256)] -> (241, 2, 256)
            xseq = xseq.permute(1, 0, 2)  # (bs, len, dim)
            xseq = xseq.view(bs, nframes + 1, self.num_head, -1)
            xseq = xseq.permute(0, 2, 1, 3)  # Need (2, 8, 2048, 64)
            xseq = xseq.reshape(bs * self.num_head, nframes + 1, -1)
            pos_emb = self.rel_pos(xseq)  # (89, 32)
            xseq, _ = apply_rotary_pos_emb(xseq, xseq, pos_emb)
            xseq_rpe = xseq.reshape(bs, self.num_head, nframes + 1, -1)
            xseq = xseq_rpe.permute(0, 2, 1, 3)  # [seqlen+1, bs, d]
            xseq = xseq.view(bs, nframes + 1, -1)
            xseq = xseq.permute(1, 0, 2)
            output = self.seqTransEncoder(xseq)[1:]



        elif 'cross_local_attention5' in self.cond_mode:
            embed_text = self.embed_text(y['seed'].squeeze(2).permute(0, 2, 1)).permute(1, 0, 2)  # (30, bs, 256-64)
            enc_text = self.FeatureEncoder(y['audio']).permute(1, 0, 2)
            embed_text_last = self.embed_text_last(y['seed_last'].squeeze(2).permute(0, 2, 1)).permute(1, 0, 2)
            enc_text = torch.cat((embed_text, enc_text, embed_text_last), axis=0)
            x = x.reshape(bs, njoints * nfeats, 1, nframes)  # [2, 135, 1, 240]
            # self-attention
            x_ = self.input_process(x)  # [2, 135, 1, 240] -> [240, 2, 256]
            # local-cross-attention
            packed_shape = [torch.Size([bs, self.num_head])]
            xseq = torch.cat((x_, enc_text), axis=2)  # [bs, d+joints*feat, 1, #frames], (240, 2, 32)
            # all frames
            embed_style_2 = (embed_style + emb_t).repeat(nframes, 1, 1)  # (bs, 64) -> (len, bs, 64)
            xseq = torch.cat((embed_style_2, xseq), axis=2)  # (seq, bs, dim)
            xseq = self.input_process2(xseq)
            xseq = xseq.permute(1, 0, 2)  # (bs, len, dim)
            xseq = xseq.view(bs, nframes, self.num_head, -1)
            xseq = xseq.permute(0, 2, 1, 3)  # Need (2, 8, 2048, 64)
            xseq = xseq.reshape(bs * self.num_head, nframes, -1)
            pos_emb = self.rel_pos(xseq)  # (89, 32)
            xseq, _ = apply_rotary_pos_emb(xseq, xseq, pos_emb)
            xseq = self.cross_local_attention(xseq, xseq, xseq, packed_shape=packed_shape,
                                              mask=y['mask_local'])  # (2, 8, 2048, 64)
            xseq = xseq.permute(0, 2, 1, 3)  # (bs, len, 8, 64)
            xseq = xseq.reshape(bs, nframes, -1)
            xseq = xseq.permute(1, 0, 2)

            xseq = torch.cat((embed_style + emb_t, xseq), axis=0)  # [seqlen+1, bs, d]     # [(1, 2, 256), (240, 2, 256)] -> (241, 2, 256)
            xseq = xseq.permute(1, 0, 2)  # (bs, len, dim)
            xseq = xseq.view(bs, nframes + 1, self.num_head, -1)
            xseq = xseq.permute(0, 2, 1, 3)  # Need (2, 8, 2048, 64)
            xseq = xseq.reshape(bs * self.num_head, nframes + 1, -1)
            pos_emb = self.rel_pos(xseq)  # (89, 32)
            xseq, _ = apply_rotary_pos_emb(xseq, xseq, pos_emb)
            xseq_rpe = xseq.reshape(bs, self.num_head, nframes + 1, -1)
            xseq = xseq_rpe.permute(0, 2, 1, 3)  # [seqlen+1, bs, d]
            xseq = xseq.view(bs, nframes + 1, -1)
            xseq = xseq.permute(1, 0, 2)
            output = self.seqTransEncoder(xseq)[1:]

        alignment_outputs = None
        if self.use_alignment_module:
            alignment_audio_features = torch.cat([audio_mels, audio_onsets], dim=2)
            alignment_mel = self.alignment_onset_mel_embedding(alignment_audio_features)
            alignment_mel = self.alignment_audio_pooler_1(alignment_mel)
            alignment_wav2vec = self.alignment_wav2vec_embedding(audio_wav2vec)
            alignment_audio_context = torch.cat([alignment_mel, alignment_wav2vec], dim=2)
            alignment_audio_context = self.alignment_audio_pooler_2(alignment_audio_context)
            low_level_context = self.alignment_low_level_context_collapsed(alignment_audio_context)
            motion_output_projection_pooled = self.alignment_motion_pooler(output)
            if self.use_culture and culture_emb is not None:
                high_level_input = torch.cat([culture_emb, text_context], dim=1)
            else:
                high_level_input = text_context
            high_level_context = self.alignment_high_level_proj(high_level_input)
            alignment_outputs = (
                motion_output_projection_pooled,
                low_level_context,
                high_level_context,
            )

        output = self.output_process(output)  #OFFICIAL! REMAINS THE SAME  [25, bs, 512]  # [bs, njoints, nfeats, nframes]
        model_outputs = [output]
        if alignment_outputs is not None:
            model_outputs.extend(alignment_outputs)
        if self.culture_classification_layer is not None:
            model_outputs.append(self.culture_classification_layer(output))
        if len(model_outputs) > 1:
            return tuple(model_outputs)
        return output


    @staticmethod
    def apply_rotary(x, sinusoidal_pos):
        sin, cos = sinusoidal_pos
        x1, x2 = x[..., 0::2], x[..., 1::2]
        # 如果是旋转query key的话，下面这个直接cat就行，因为要进行矩阵乘法，最终会在这个维度求和。（只要保持query和key的最后一个dim的每一个位置对应上就可以）
        # torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        # 如果是旋转value的话，下面这个stack后再flatten才可以，因为训练好的模型最后一个dim是两两之间交替的。
        return torch.stack([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).flatten(-2, -1)



class WindowedAttentionPooling(nn.Module):
    def __init__(self, input_dim, output_dim, window_size, stride, target_length):
        super(WindowedAttentionPooling, self).__init__()
        self.window_size = window_size
        self.stride = stride
        self.target_length = target_length
        self.attention = nn.Linear(input_dim, 1)
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        batch_size, seq_len, input_dim = x.size()

        # Calculate number of windows needed
        num_windows = (seq_len - self.window_size) // self.stride + 1

        # Extract windows
        windows = x.unfold(1, self.window_size, self.stride)  # (batch_size, num_windows, input_dim, window_size)
        windows = windows.permute(0, 1, 3, 2) # (batch_size, num_windows, window_size, input_dim)

        # Compute attention scores within each window
        scores = self.attention(windows)  # (batch_size, num_windows, window_size, 1)
        scores = F.softmax(scores, dim=2)  # Softmax over window_size

        # Weighted sum within each window
        weighted = torch.sum(windows * scores, dim=2)  # (batch_size, num_windows, input_dim)

        # Project to output dimension
        out = self.projection(weighted)  # (batch_size, num_windows, output_dim)

        # If num_windows > target_length, truncate or adjust
        if num_windows > self.target_length:
            out = out[:, :self.target_length, :]
        elif num_windows < self.target_length:
            # Optionally pad or interpolate to reach target_length
            padding = self.target_length - num_windows
            pad_tensor = torch.zeros(batch_size, padding, out.size(-1), device=x.device)
            out = torch.cat((out, pad_tensor), dim=1)

        return out  # (batch_size, target_length, output_dim)


class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Linear(d_model, 1)

    def forward(self, x):
        scores = self.attention(x)
        weights = F.softmax(scores, dim=1)
        return torch.sum(x * weights, dim=1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)      # (5000, 128)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)     # (5000, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


# Copied from transformers.models.marian.modeling_marian.MarianSinusoidalPositionalEmbedding with Marian->RoFormer
class RoFormerSinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(
        self, num_positions: int, embedding_dim: int
    ):
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter):
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
                for pos in range(n_pos)
            ]
        )
        out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, seq_len: int, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        positions = torch.arange(
            past_key_values_length,
            past_key_values_length + seq_len,
            dtype=torch.long,
            device=self.weight.device,
        )
        return super().forward(positions)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep #mpt ised
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        if self.data_rep == 'rot_vel':
            self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        #bs, njoints, nfeats, nframes = x.shape
        bs, nframes, nfeats = x.shape
        #x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)
        x = x.permute(1, 0, 2)

        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            x = self.poseEmbedding(x)  # [seqlen, bs, d]
            return x
        elif self.data_rep == 'rot_vel': #NO
            first_pose = x[[0]]  # [1, bs, 150]
            first_pose = self.poseEmbedding(first_pose)  # [1, bs, d]
            vel = x[1:]  # [seqlen-1, bs, 150]
            vel = self.velEmbedding(vel)  # [seqlen-1, bs, d]
            return torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, d]
        else:
            raise ValueError

'''
class OutputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        if self.data_rep == 'rot_vel':
            self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            output = self.poseFinal(output)  # [seqlen, bs, 150]
        elif self.data_rep == 'rot_vel':
            first_pose = output[[0]]  # [1, bs, d]
            first_pose = self.poseFinal(first_pose)  # [1, bs, 150]
            vel = output[1:]  # [seqlen-1, bs, d]
            vel = self.velFinal(vel)  # [seqlen-1, bs, 150]
            output = torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, 150]
        else:
            raise ValueError
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        return output
'''

class OutputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim, nfeats):
        super().__init__()
        self.input_feats = input_feats #512
        self.latent_dim = latent_dim #512
        self.nfeats = nfeats #25
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        output = self.poseFinal(output)  # [seqlen, bs, 512]
        #output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        #output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        return output


class LinearTemporalCrossAttention(nn.Module):

    def __init__(self, seq_len, latent_dim, text_latent_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.text_norm = nn.LayerNorm(text_latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(text_latent_dim, latent_dim)
        self.value = nn.Linear(text_latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = nn.Linear(latent_dim, latent_dim)

    def forward(self, x, xf=None, emb=None):
        """
        x: B, T, D      , [240, 2, 256]
        xf: B, N, L     , [1, 2, 256]
        """
        x = x.permute(1, 0, 2)
        # xf = xf.permute(1, 0, 2)
        B, T, D = x.shape
        # N = xf.shape[1]
        H = self.num_head
        # B, T, D
        query = self.query(self.norm(x))
        # B, N, D
        key = self.key(self.text_norm(x))
        query = F.softmax(query.view(B, T, H, -1), dim=-1)
        key = F.softmax(key.view(B, T, H, -1), dim=1)
        # B, N, H, HD
        value = self.value(self.text_norm(x)).view(B, T, H, -1)
        # B, H, HD, HD
        attention = torch.einsum('bnhd,bnhl->bhdl', key, value)
        y = torch.einsum('bnhd,bhdl->bnhl', query, attention).reshape(B, T, D)
        # y = x + self.proj_out(y, emb)
        return y


class FeatureEncoder(nn.Module):
    def __init__(self, source_dim, audio_feat_dim):
        super().__init__()
        self.audio_feature_map = nn.Linear(source_dim, audio_feat_dim)

    def forward(self, rep):
        rep = self.audio_feature_map(rep)
        return rep


if __name__ == '__main__':
    '''
    cd ./BEAT-main/model
    python mdm.py
    '''
    n_frames = 150
    n_seed = 30
    njoints = 684*3
    audio_feature_dim = 1133 + 301      # audio_f + text_f
    style_dim = 2
    bs = 2
    
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    model = MDM(modeltype='', njoints=njoints, nfeats=1, cond_mode='cross_local_attention5_style1', audio_feat='wavlm',
                arch='trans_enc', latent_dim=512, n_seed=n_seed, cond_mask_prob=0.1, 
                style_dim=style_dim, source_audio_dim=audio_feature_dim).to(device)

    x = torch.randn(bs, njoints, 1, n_frames)
    t = torch.tensor([12, 85])

    model_kwargs_ = {'y': {}}
    model_kwargs_['y']['mask'] = (torch.zeros([1, 1, 1, n_frames]) < 1)     # [..., n_seed:]
    model_kwargs_['y']['audio'] = torch.randn(bs, n_frames - n_seed - n_seed, audio_feature_dim)  # attention5
    # model_kwargs_['y']['audio'] = torch.randn(bs, n_frames - n_seed, audio_feature_dim)       # attention4
    # model_kwargs_['y']['audio'] = torch.randn(bs, n_frames, audio_feature_dim)  # attention3
    model_kwargs_['y']['style'] = torch.randn(bs, style_dim)
    model_kwargs_['y']['mask_local'] = torch.ones(bs, n_frames).bool()
    model_kwargs_['y']['seed'] = x[..., 0:n_seed]       # attention3/4
    model_kwargs_['y']['seed_last'] = x[..., -n_seed:]  # attention5
    model_kwargs_['y']['gesture'] = torch.randn(bs, n_frames, njoints)
    y = model(x, t, model_kwargs_['y'])     # [bs, njoints, nfeats, nframes]
    print(y.shape)
