import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
import glob
from culture_encoder.adversarial_classifier import Culture_Classifier
from culture_encoder.fishr_classifier import Fishr

#normal_repr = torch.Tensor.__repr__
#torch.Tensor.__repr__ = lambda self: f"{self.shape}_{normal_repr(self)}"  # for debug


class OneHotCultureEmbedder(nn.Module):
    """Trainable one-hot culture baseline with the same output width as other culture embedders."""

    def __init__(self, num_cultures, embed_dim, dropout=0.1, activation="gelu"):
        super().__init__()
        self.num_cultures = int(num_cultures)
        self.embed_dim = int(embed_dim)
        activation_layer = nn.GELU() if activation == "gelu" else nn.LeakyReLU()
        self.project = nn.Sequential(
            nn.Linear(self.num_cultures, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            activation_layer,
            nn.Dropout(dropout),
        )

    def forward(self, culture_labels):
        if culture_labels is None:
            raise ValueError(
                "One-hot culture conditioning requires culture labels in the model inputs. "
                "Use labels['culture_enc'] from GestureDataset."
            )
        if culture_labels.dim() > 1 and culture_labels.shape[-1] == self.num_cultures:
            one_hot = culture_labels.float()
        else:
            culture_labels = culture_labels.view(-1).long()
            one_hot = F.one_hot(culture_labels, num_classes=self.num_cultures).float()
        return self.project(one_hot)



class Hierarchical_MDM(nn.Module):
    def __init__(self,
                 vqvae_dim=(25,512),
                 audio_onset_dim=156,
                 audio_mel_dim=(156,64),
                 audio_wav2vec_dim=(50,1024),
                 text_embedding_dim=768,
                 culture_embedding_dim=1024,
                 latent_dim=512,
                 n_train_speakers=0,
                 num_heads=8,
                 num_layers=10,
                 ff_size=2048,
                 n_cultures=4,
                 dropout=0.1,
                 activation='gelu',
                 culture_embedder_config = '',
                 device = 'cuda:0',
                 dataset_type = "_sep_people",
                 motion_prefix_len = 5,
                 audio_prefix_len = 10,
                 pos_enc_max_len = 5000,
                 motion_mask_prob = .1,
                 audio_mask_prob = .1,
                 use_culture = False,
                 use_adversarial = False,
                 use_one_hot_culture: bool = False,
                 culture_encoder_type: str = "",
                 use_adain: bool = True,
                 use_alignment: bool = True,
                 use_culture_classification_layer: bool = True,
                 use_native_attention: bool = False,
        ):
            super(Hierarchical_MDM, self).__init__()

            self.vqvae_emb_dim = vqvae_dim #25,512
            self.audio_onset_dim = audio_onset_dim #156
            self.audio_mel_dim = audio_mel_dim #156,64
            self.wav2vec_dim = audio_wav2vec_dim #50,1024
            self.text_embedding_dim = text_embedding_dim #768
            self.culture_embedding_dim = culture_embedding_dim #1024
            self.latent_dim = latent_dim #latent_dim #512, transformer dim #TODO, if you want to use a custom latent dim, you need to change the output of the model
            self.num_heads = num_heads
            self.num_layers = num_layers
            self.ff_size = ff_size
            self.n_cultures = n_cultures #4
            self.dropout = dropout
            self.activation = activation
            self.device = device
            self.use_culture = use_culture
            self.use_adversarial = use_adversarial
            self.use_one_hot_culture = use_one_hot_culture
            self.use_adain = use_adain
            self.use_alignment = use_alignment
            self.use_culture_classification_layer = use_culture_classification_layer
            self.use_native_attention = use_native_attention

            self.motion_prefix_len = motion_prefix_len #5
            self.audio_prefix_len = audio_prefix_len #10
            self.pos_enc_max_len = pos_enc_max_len
            self.motion_mask_prob = motion_mask_prob
            self.audio_mask_prob = audio_mask_prob
            self.activation = nn.GELU() if activation == 'gelu' else nn.LeakyReLU()

            ### Culture classifier setup
            # culture_classifier_config_path should be provided by config/CLI rather than a local machine path.
            cl_config = culture_embedder_config
            print(cl_config)
            culture_embedder_name = "culclI"
            culture_embedder_type = dataset_type #+ "_adv_train"
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

            def _normalize_path(path_value):
                if not path_value:
                    return None
                path_value = os.path.expanduser(str(path_value))
                if not os.path.isabs(path_value):
                    path_value = os.path.join(repo_root, path_value)
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

            culture_encoder_type = str(culture_encoder_type or "").strip().lower().replace("-", "_")
            if not culture_encoder_type:
                if not use_culture:
                    culture_encoder_type = "none"
                elif use_one_hot_culture:
                    culture_encoder_type = "one_hot"
                elif use_adversarial:
                    culture_encoder_type = "adversarial"
                else:
                    culture_encoder_type = "fishr"
            if culture_encoder_type in {"onehot", "one_hot_culture"}:
                culture_encoder_type = "one_hot"
            if culture_encoder_type not in {"none", "fishr", "adversarial", "one_hot"}:
                raise ValueError(
                    f"Unsupported culture_encoder_type='{culture_encoder_type}'. "
                    "Use one of: none, fishr, adversarial, one_hot."
                )
            if not use_culture:
                culture_encoder_type = "none"
            self.culture_encoder_type = culture_encoder_type
            self.use_culture = culture_encoder_type != "none"
            self.use_adversarial = culture_encoder_type == "adversarial"
            self.use_one_hot_culture = culture_encoder_type == "one_hot"

            if self.culture_encoder_type == "adversarial":
                self.culture_embedding_dim = 512
                self.culture_embedder =  Culture_Classifier(cl_config, device, dropout_prob=cl_config.dropout,
                                                           cl_type=culture_embedder_name ,d_model=culture_embedding_dim, speaker_penalization=False,
                                                           n_cultures=self.n_cultures,n_speakers = n_train_speakers,pose_enc=cl_config.pose_enc_type,
                                                            audio_enc=cl_config.audio_enc_type,raw_poses=False).to(device)
                adv_explicit = getattr(cl_config, "adversarial_checkpoint_path", None)
                adv_root_candidates = [
                    getattr(cl_config, "adversarial_model_save_path", None),
                    getattr(cl_config, "model_save_path", None),
                    "culture_encoder/adversarial_train",
                ]
                adv_file_candidates = [adv_explicit]
                adv_filename = f"{culture_embedder_name}_checkpoint_best{culture_embedder_type}.bin"
                for root_candidate in adv_root_candidates:
                    if not root_candidate:
                        continue
                    train_dir = os.path.join(
                        str(root_candidate),
                        culture_embedder_name,
                        "no_weighted_loss",
                        f"train_{culture_embedder_name}",
                    )
                    # First try exact dataset suffix, then fallback to any best checkpoint.
                    adv_file_candidates.append(os.path.join(train_dir, adv_filename))
                    adv_file_candidates.extend(
                        sorted(glob.glob(os.path.join(_normalize_path(train_dir) or "", f"{culture_embedder_name}_checkpoint_best*.bin")))
                    )

                self.culture_embedder_weights_path, tried_paths = _first_existing_file(adv_file_candidates)
                if self.culture_embedder_weights_path is None:
                    raise FileNotFoundError(
                        "Could not locate adversarial culture checkpoint. Tried:\n  - "
                        + "\n  - ".join(tried_paths)
                    )

                self.culture_embedder_checkpoint = torch.load(
                    self.culture_embedder_weights_path, map_location=torch.device(device)
                )
                filtered_state_dict = {k: v for k, v in self.culture_embedder_checkpoint['model_dict'].items() if 'speaker_classifier' not in k}
                self.culture_embedder.load_state_dict(filtered_state_dict, strict=False)
                for param in self.culture_embedder.parameters():
                    param.requires_grad = False
                self.culture_embedder.eval()
                print(
                    f"[Hierarchical_MDM] Culture encoder: adversarial '{culture_embedder_name}' "
                    f"(dim={self.culture_embedding_dim}) from {self.culture_embedder_weights_path}"
                )

            elif self.culture_encoder_type == "fishr":
                fishr_d_model = int(getattr(cl_config, "layer_neurons", 512))
                self.culture_embedding_dim = int(getattr(cl_config, "embed_dim", fishr_d_model))
                try:
                    self.culture_embedder = Fishr(
                        proj_dim=fishr_d_model,
                        num_classes=self.n_cultures,
                        num_domains=n_train_speakers,
                        is_nonlinear=False,
                        use_motion=False,
                        cl_type="culclI",
                        d_model=fishr_d_model,
                        pose_enc_type=getattr(cl_config, "pose_enc_type", "transformer"),
                        audio_enc_type=getattr(cl_config, "audio_enc_type", None),
                        raw_poses=False,
                        backbone_config=cl_config,
                        use_adversarial_backbone=True,
                    ).to(device)
                except TypeError:
                    # Backward compatibility with older Fishr signatures.
                    self.culture_embedder = Fishr(
                        proj_dim=fishr_d_model,
                        num_classes=self.n_cultures,
                        num_domains=n_train_speakers,
                        is_nonlinear=False,
                        use_motion=False,
                    ).to(device)
                cl_save_path = (
                    getattr(cl_config, "fishr_model_path", None)
                    or getattr(cl_config, "fishr_save_dir", None)
                    or getattr(cl_config, "model_save_path", None)
                    or "culture_encoder/fishr_train/full_data_culclI"
                )
                fishr_candidates = [
                    getattr(cl_config, "fishr_model_path", None),  # explicit file or dir
                    getattr(cl_config, "fishr_save_dir", None),    # dir
                    cl_save_path,                                  # legacy dir
                    "culture_encoder/fishr_train/full_data_culclI",
                    "culture_encoder/fishr_train/full_data",
                    "culture_encoder",
                ]
                fishr_file_candidates = []
                for candidate in fishr_candidates:
                    normalized = _normalize_path(candidate)
                    if not normalized:
                        continue
                    if os.path.isfile(normalized):
                        fishr_file_candidates.append(normalized)
                    else:
                        fishr_file_candidates.append(os.path.join(normalized, "model_best.pt"))

                self.culture_embedder_weights_path, tried_paths = _first_existing_file(fishr_file_candidates)
                if self.culture_embedder_weights_path is None:
                    raise FileNotFoundError(
                        "Could not locate Fishr culture checkpoint. Tried:\n  - "
                        + "\n  - ".join(tried_paths)
                    )

                self.culture_embedder.load_state_dict(
                    torch.load(self.culture_embedder_weights_path, map_location=self.device)
                )
                for param in self.culture_embedder.parameters():
                    param.requires_grad = False
                self.culture_embedder.eval()
                print(
                    f"[Hierarchical_MDM] Culture encoder: fishr 'culclI' "
                    f"(dim={self.culture_embedding_dim}) from {self.culture_embedder_weights_path}"
                )
            elif self.culture_encoder_type == "one_hot":
                self.culture_embedding_dim = int(getattr(cl_config, "embed_dim", culture_embedding_dim) or culture_embedding_dim)
                self.culture_embedder = OneHotCultureEmbedder(
                    num_cultures=self.n_cultures,
                    embed_dim=self.culture_embedding_dim,
                    dropout=self.dropout,
                    activation=activation,
                ).to(device)
                print(
                    f"[Hierarchical_MDM] Culture encoder: trainable one-hot "
                    f"(n_cultures={self.n_cultures}, dim={self.culture_embedding_dim})"
                )
            else:
                print("[Hierarchical_MDM] Culture encoder: disabled")

            # Input processing for pose embeddings
            #TODO removed as the dimension of motion embeddings and latent dim match. If not, add this
            #assert(self.vqvae_emb_dim[1] == self.latent_dim)
            self.input_process = InputProcess('vqvae_codebooks', self.vqvae_emb_dim[1], self.latent_dim, self.dropout) #project vqvae codebooks to transformer space

            # Positional encoding. Now input is (seq_len, batch_size, d_model)
            self.gesture_pos_encoder = PositionalEncoding(self.latent_dim, self.pos_enc_max_len, self.dropout) #to add positional encoding to codebooks
            self.random_gesture_masking = RandomMasking(mask_prob=self.motion_mask_prob, mask_value=0.0) #to mask random gestures

            # Time step embedding
            self.embed_timestep = TimestepEmbedder(self.latent_dim, self.gesture_pos_encoder) #embedd diffusion timestep t

            # Output processing
            #TODO this is needed as the representation before projection is made to be aligned wth audio, text, culture. Eventually the aligned representation is projected to the final motion space
            self.output_process = OutputProcess('vqvae_codebooks', self.latent_dim, self.vqvae_emb_dim[1]) #reconstruct motion from latent space
            self.output_projection = nn.Linear(self.vqvae_emb_dim[1], self.latent_dim) #map vqvae dim to latent dim for comparing alignment with high level and low level features
            self.output_projection_pooler = AttentionPooling(self.latent_dim) #pooling for loss computation with high level features

            # Embedding layers for conditioning inputs
            #TODO check if the new projections are better. Now mels and onsets are concatenated and projected to (156,256),
            # TODO wav2vec was also projected to the same space. Then time dimension is reduced, finally they are concatenated
            self.onset_mel_embedding = nn.Sequential(nn.Linear(1 + audio_mel_dim[1], self.latent_dim//2),
                                                     nn.LayerNorm(self.latent_dim//2),
                                                     self.activation,
                                                    nn.Dropout(self.dropout) )
            self.wav2vec_embedding = nn.Sequential(nn.Linear(self.wav2vec_dim[1], self.latent_dim//2),
                                                   nn.LayerNorm( self.latent_dim // 2),
                                                   self.activation,
                                                   nn.Dropout(self.dropout))
            # 1. Embed wav2vec -> [50,1024] -> [50, 256]
            # 2. Concatenate mel-log and onset -> [156,1], [156,64]  -> [156,65]
            # 3. Embed mel-log + onset -> [156,65] -> [156, 256]
            # 4. Audio_pool_1 on mel-log + onset -> [156,256]  -> [50,256]
            # 5. Concatenate wav2vec and audio_pool_1 -> [50,256] , [50,256] -> [50,512]
            # 6. Audio_pool_2 on wav2vec + audio_pool_1 -> [50,512] -> [25,512]
            self.audio_pooler_1 = WindowedAttentionPooling(input_dim=self.latent_dim//2, output_dim = self.latent_dim//2,
                                                         window_size = 9, stride = 3, target_length = self.vqvae_emb_dim[0]*2)
            self.audio_pooler_2 = WindowedAttentionPooling(input_dim=self.latent_dim, output_dim = self.latent_dim,
                                                           window_size=2, stride=2, target_length=self.vqvae_emb_dim[0])
            #self.wav2vec_proj = nn.Linear(self.latent_dim//2, self.latent_dim//2) #map wav2vec to the same space as mels and onsets
            #self.wav2vec_pooler = WindowedAttentionPooling(input_dim=self.latent_dim//2, output_dim = self.latent_dim//2,
            #                                             window_size = 2, stride = 2, target_length = self.vqvae_emb_dim[0])
            self.random_audio_masking = RandomMasking(mask_prob=self.audio_mask_prob, mask_value=0.0)
            self.low_level_context_collapsed = AttentionPooling(self.latent_dim)
            '''
            self.onset_embedding = nn.Linear(1, self.latent_dim//2) #map onsets to latent dim
            self.mel_embedding = nn.Linear(audio_mel_dim[1], self.latent_dim//2) #map mels to latent dim
            self.wav2vec_embedding = nn.Linear(self.wav2vec_dim[1], self.latent_dim)
            #map mels and onsets length to match wav2vec (from 156 to 50)
            self.audio_pooler = WindowedAttentionPooling(input_dim=self.latent_dim, output_dim = self.latent_dim,
                                                         window_size = 9, stride = 3, target_length = self.wav2vec_dim[0])
    
            #map all audio features from dim 50 to 25 (alternative to avg pooling to keep more information)
            self.low_level_context_attention_pooler = WindowedAttentionPooling(input_dim=self.latent_dim, output_dim = self.latent_dim,
                                                         window_size = 2, stride = 2, target_length = self.vqvae_emb_dim[0])
    
            self.low_level_context_proj = nn.Linear(self.latent_dim * 2, self.latent_dim)
            self.low_level_context_pooler = nn.AvgPool1d(kernel_size=2, stride=2) #to downsample the audio context for loss computing
            self.random_audio_masking = RandomMasking(mask_prob=self.audio_mask_prob, mask_value=0.0)
            self.low_level_context_collapsed = AttentionPooling(self.latent_dim) #pooling that represents the overall low level context
            '''
            self.audio_pos_encoder = PositionalEncoding(self.latent_dim, self.pos_enc_max_len, self.dropout)

            #self.text_embedding = nn.Linear(self.text_embedding_dim, self.latent_dim)
            #Given self.culture_embedding = nn.Linear(self.culture_embedding_dim, self.latent_dim)

            # Optional: Combine high-level context into a single embedding #culture + text + timestep t
            # TODO NOTICE THAT CULTURE HAS BEEN REMOVED
            if self.use_culture:
                print("Culture is used...")
                self.high_level_context_proj = nn.Sequential(nn.Linear(self.text_embedding_dim + self.culture_embedding_dim, self.latent_dim),
                                                                nn.LayerNorm(self.latent_dim),
                                                                self.activation,
                                                                nn.Dropout(self.dropout))
            else:
                self.high_level_context_proj = nn.Sequential(nn.Linear(self.text_embedding_dim, self.latent_dim),
                                                                nn.LayerNorm(self.latent_dim),
                                                                self.activation,
                                                                nn.Dropout(self.dropout))

            if self.use_culture_classification_layer:
                self.culture_classification_layer = nn.Sequential(
                    nn.Linear(self.vqvae_emb_dim[1], self.latent_dim),  # Example dimension reduction
                    AttentionPooling(self.latent_dim),  # Attention pooling
                    nn.BatchNorm1d(self.latent_dim),  # Adding BatchNorm here
                    #####nn.LayerNorm(self.latent_dim),
                    self.activation,            # Adding GELU activation
                    nn.Dropout(p=self.dropout),          # Adding Dropout
                    nn.Linear(self.latent_dim, self.n_cultures)  # Number of unique speakers
                )
            else:
                self.culture_classification_layer = None

            '''
            # Cross-attention layers  #cross-attention between low level and motion features
            self.seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                               nhead=self.num_heads,
                                                               dim_feedforward=self.ff_size,
                                                               dropout=self.dropout,
                                                               activation=self.activation,
                                                               batch_first=True)
            '''
            '''
            self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                            num_layers=self.num_layers)
            '''
            #self.high_level_condition = ConditionalLayerNorm(self.latent_dim,self.latent_dim)
            self.num_layers = num_layers
            self.condition_layers = nn.ModuleList()
            for _ in range(num_layers):
                if self.use_native_attention:
                    self.condition_layers.append(
                        NativeCrossAttentionBlock(
                            latent_dim=self.latent_dim,
                            num_heads=self.num_heads,
                            ff_size=self.ff_size,
                            dropout=self.dropout,
                            activation=activation,
                        )
                    )
                else:
                    self.condition_layers.append(
                        CrossAttentionLayer(
                            self.latent_dim, self.num_heads, self.ff_size, self.dropout, activation
                        )
                    )
                #self.condition_layers.append(self.seqTransDecoderLayer)
                #self.condition_layers.append(GradualAdaIN(self.latent_dim, self.latent_dim))
                if self.use_adain:
                    self.condition_layers.append(AdaIN(self.latent_dim, self.latent_dim))
                #########self.condition_layers.append(HierarchicalCrossAttentionLayer(self.latent_dim, self.num_heads, self.ff_size, self.dropout, activation))
                #self.condition_layers.append(ConditionalLayerNorm(self.latent_dim, self.latent_dim))


    def count_parameters(self):
        """
        Counts the total number of trainable and non-trainable parameters in the model.

        Returns:
            tuple: (trainable_params, non_trainable_params, total_params)
        """
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        total_params = trainable_params + non_trainable_params
        return trainable_params, non_trainable_params, total_params

    def forward(self, data, timesteps):
        """
        x: [batch_size, sequence_length, num_featuers], denoted x_t in the paper
        timesteps: [batch_size] (int)
        y: Dictionary containing conditioning inputs
        """
        if len(data) == 5:
            motion, text_features, audio_mels, audio_onsets, audio_wav2vec = data
            culture_labels = None
        elif len(data) == 6:
            motion, text_features, audio_mels, audio_onsets, audio_wav2vec, culture_labels = data
        else:
            raise ValueError(f"Expected 5 or 6 model inputs, got {len(data)}.")
        modal_data = (motion, text_features, audio_mels, audio_onsets, audio_wav2vec)
        batch_size = motion.shape[0]
        raw_audio_onsets = audio_onsets
        if audio_onsets.dim() == 2:
            audio_onsets = audio_onsets.unsqueeze(-1)
        elif audio_onsets.dim() == 3 and audio_onsets.shape[-1] == 1:
            pass
        else:
            raise ValueError(f"Unexpected audio_onsets shape {tuple(audio_onsets.shape)}; expected [B,T] or [B,T,1].")


        #motion_noised_without_process = motion[:, self.motion_prefix_len:, :] #######FOR DEBUG
        #mse_loss = nn.MSELoss()

        # Process input poses
        motion = self.input_process(motion)  # [batch_size, seq_len, latent_dim]
        #print("motion",motion.shape)
        motion_noised = motion[:, self.motion_prefix_len:, :]  # noised part of motion
        #print("motion_noised_before",motion_noised.shape, motion_noised[0])
        motion_noised, motion_mask = self.random_gesture_masking(motion_noised) #both are [batch_size, seq_len - motion_prefix_len, latent_dim]
        #print("motion_noised:after", motion_noised.shape,motion_noised[0],"motion_mask",motion_mask.shape)
        motion[:, self.motion_prefix_len:, :] = motion_noised #replace noised part of motion with masked noised motion

        # Embed audio features
        wav2vec_emb = self.wav2vec_embedding(audio_wav2vec)  # [batch_size, wav2vec_seq_len = seq_len*2, latent_dim//2] #step 1
        audio_features = torch.cat([audio_mels,audio_onsets],dim=2) #[batch_size, onset_seq_len = mel_sequence_len = 156, mel_dim + 1] #step 2
        audio_embedding = self.onset_mel_embedding(audio_features) #[batch_size, onset_seq_len = mel_sequence_len = 156, latent_dim//2] #step 3
        audio_embedding = self.audio_pooler_1(audio_embedding)  #[batch_size, seq_len*2, latent_dim//2] #step 4
        audio_context = torch.cat([audio_embedding, wav2vec_emb], dim=2)  # [batch_size, seq_len * 2, latent_dim] #step 5
        audio_context = self.audio_pooler_2(audio_context) #[batch_size, seq_len, latent_dim] #step 6

        #wav2vec_emb = self.wav2vec_pooler(wav2vec_emb)  # [batch_size, seq_len, latent_dim//2]

        '''
        onset_emb = self.onset_embedding(audio_onsets)  # [batch_size, onset_seq_len, latent_dim//2] onset_seq_len = mel_seq_len
        mel_emb = self.mel_embedding(audio_mels)        # [batch_size, mel_seq_len, latent_dim//2]
        wav2vec_emb = self.wav2vec_embedding(audio_wav2vec) # [batch_size, mel_seq_len, latent_dim]
        low_level_audio = torch.cat([onset_emb, mel_emb], dim=2)  # [batch_size, audio_seq_len, latent_dim] -> audio_seq_len = mel_seq_len = onset_seq_len = 156
        low_level_audio = self.audio_pooler(low_level_audio) # [batch_size, wav2vec_seq_len, latent_dim]
        audio_context = torch.cat([low_level_audio , wav2vec_emb], dim=2) # [batch_size, wav2vec_seq_len, latent_dim*2]
        audio_context = self.low_level_context_proj(audio_context) #[ batch_size, wav2vec_seq_len, latent_dim]
        ###audio_context = self.audio_pos_encoder(audio_context) #add positional encoding to audio context #TODO, do not add it here. First match motion dimension, then add it
        audio_context = self.low_level_context_attention_pooler(audio_context)  # [batch_size, seq_len, latent_dim] pool everything to match motion sequence len and attend cross-att.
        '''
        #This helps when computing cosine alignment as they have same shape, and when using pos encoding as they have same frame rate
        audio_noised_motion = audio_context[:, self.motion_prefix_len:, :]  #[batch_size, seq_len - motion_prefix_len, latent_dim]
        audio_noised_motion, audio_mask = self.random_audio_masking(audio_noised_motion)  #both are [batch_size, seq_len - motion_prefix_len, latent_dim]
        audio_context[:, self.motion_prefix_len:, :] = audio_noised_motion #masked audio
        overall_audio_context = self.low_level_context_collapsed(audio_context).unsqueeze(1)  # [batch_size, 1, latent_dim] #TO use as low level context token
        #audio_context_pooled = audio_context # TODO deprecate. Before I add different dimensions for audio sequence and motion sequence. the pooled version served to match motion

        #pooled_audio_mask = audio_mask  # TODO deprecate as above

        # Prepare high-level context embeddings
        #text_emb = self.text_embedding(text_features)  # [batch_size, latent_dim]
        # TODO NOTICE CULTURE REMOVAL
        if self.use_culture and self.culture_encoder_type == "fishr":
            #culture_emb = self.culture_embedder(data, return_embeddings=True) # [batch_size, culture_embedding_dim]
            #print("HERE",audio_mels.shape, audio_onsets.shape, audio_wav2vec.shape)
            with torch.no_grad():
                self.culture_embedder.eval()
                if bool(getattr(self.culture_embedder, "use_adversarial_backbone", False)):
                    fishr_onsets = raw_audio_onsets
                    if fishr_onsets.dim() == 3 and fishr_onsets.shape[-1] == 1:
                        fishr_onsets = fishr_onsets.squeeze(-1)
                    fishr_inputs = (
                        data[0].to(self.device),
                        text_features.to(self.device),
                        audio_mels.to(self.device),
                        fishr_onsets.to(self.device),
                        audio_wav2vec.to(self.device),
                    )
                    _, culture_emb = self.culture_embedder.predict(fishr_inputs, return_embeddings=True)
                else:
                    features = {
                        "sentence": text_features.to(self.device),
                        "mel": audio_mels.to(self.device),
                        "onset": raw_audio_onsets.to(self.device),
                        "wav2vec": audio_wav2vec.to(self.device),
                    }
                    culture_emb = self.culture_embedder.feature_extract(features)
                #print("Culture embedding shape", culture_emb.shape)
            high_level_context = torch.cat([text_features, culture_emb],dim=1)  # [batch_size, latent_dim culture + latent_dim_text] #TODO note that timestep embedding now is separate
            high_level_context = self.high_level_context_proj(high_level_context)  # [batch_size, latent_dim]
        elif self.use_culture and self.culture_encoder_type == "adversarial":
            with torch.no_grad():
                self.culture_embedder.eval()
                culture_emb = self.culture_embedder(modal_data, return_embeddings=True)
            high_level_context = torch.cat([text_features, culture_emb],dim=1)  # [batch_size, latent_dim culture + latent_dim_text] #TODO note that timestep embedding now is separate
            high_level_context = self.high_level_context_proj(high_level_context)  # [batch_size, latent_dim]
        elif self.use_culture and self.culture_encoder_type == "one_hot":
            culture_emb = self.culture_embedder(culture_labels.to(self.device) if torch.is_tensor(culture_labels) else culture_labels)
            high_level_context = torch.cat([text_features, culture_emb], dim=1)
            high_level_context = self.high_level_context_proj(high_level_context)
        else:
            high_level_context = text_features
            high_level_context = self.high_level_context_proj(high_level_context)  # [batch_size, latent_dim]
        #culture_emb = self.culture_embedding(culture_emb)  # [batch_size, latent_dim]
        time_emb = self.embed_timestep(timesteps)  # [batch_size, 1, latent_dim]
        #diffusion_emb = time_emb.squeeze(1)  # [batch_size, latent_dim]

        #overall_high_level_context = high_level_context.unsqueeze(1) # [batch_size, 1, latent_dim] #TODO check if it is better repeating across seq_len and compute cross-attention
        #text_emb = text_emb.unsqueeze(1) #Just use it as a input token

        #overall_context = torch.add(high_level_context + overall_audio_context) #represents the overall context of the model

        # Create key_padding_mask (True=mask) for motion and audio for the noised portion. we need [batch_size, seq_len] with True=mask
        total_seq_len = motion.shape[1]
        #Note that the length is total_seq_len + 3 as three context tokens are added to the input -> TODO note that now only time-emb is added to make it simpler
        full_motion_mask = torch.zeros((batch_size, total_seq_len + 1), dtype=torch.bool, device=motion.device)
        motion_frame_mask = motion_mask[..., 0] # [batch_size, 20]
        full_motion_mask[:, self.motion_prefix_len + 1:] = motion_frame_mask

        audio_seq_len = audio_context.shape[1] # -> 25
        full_audio_mask = torch.zeros((batch_size, audio_seq_len), dtype=torch.bool, device=audio_context.device) #[batch_size, 25]
        audio_frame_mask = audio_mask[...,0] # [batch_size, 20]
        full_audio_mask[:,self.motion_prefix_len:] = audio_frame_mask
        # TODO note that now text is directly added to motion and high level context is only represented by culture.
        motion = self.gesture_pos_encoder(motion)  # assign positional encoding to motion
        motion = torch.cat((time_emb, motion), dim=1) #add overall context to make the learning easier as special tokens in transformer -> TODO note that now only time-emb is added to make it simpler
        #print("MOTION BEFORE CROSS ATT",motion[0])
        audio_context = self.audio_pos_encoder(audio_context) # assign positional encoding to audio

        for layer in self.condition_layers:
            #motion = layer(motion, audio_context, overall_high_level_context, motion_mask=full_motion_mask, audio_mask=full_audio_mask, culture_mask=None)
            if isinstance(layer, (CrossAttentionLayer, NativeCrossAttentionBlock)):
                motion = layer(motion, audio_context, x_mask=full_motion_mask, context_mask=full_audio_mask)
            #if isinstance(layer, nn.TransformerDecoderLayer):
            #    motion = layer(tgt=motion, memory=audio_context)#, memory_key_padding_mask=full_audio_mask, tgt_key_padding_mask=full_motion_mask)
            elif isinstance(layer, AdaIN):
                motion = layer(motion, high_level_context) #TODO check if it is better to do high_level_context + diffusion_emb to improve the timestep context recognition

        #motion_audio_emb = self.seqTransDecoder(tgt=motion, memory=audio_context, memory_key_padding_mask=full_audio_mask, tgt_key_padding_mask=full_motion_mask)

        # Transpose x for batch-first operations
        #motion = motion.permute(1, 0, 2)  # [batch_size, seq_len, latent_dim]

        # Cross-attention layers
        #for layer in self.cross_attention_layers:
            # First cross-attention with audio context
        #    motion = layer(motion, audio_context, x_mask=full_motion_mask, context_mask=full_audio_mask)

            # Second cross-attention with high-level context
            #motion = layer(motion, overall_high_level_context, x_mask=full_motion_mask, context_mask=None)

        # Transpose x back to [seq_len, batch_size, latent_dim] -> not necessary. I want the representation [seq_len, batch_size, latent_dim]
        #motion = motion.permute(1, 0, 2)
        motion = motion[:,1:,:] #remove the overall context from the output -> TODO note that now only time-emb is added to make it simpler
        motion_output_projection_pooled = self.output_projection_pooler(motion) # [batch_size, vqvae_dim]  #Representation needed to align high level features

        # Output processing
        # TODO note that motion is no more projected in latent dim for loss computation but it is instead projected to
        # TODO VQVAE dim for consistency. Indeed, this avoids to compute losses on a projected output instead of the real output, ensuring better stability.
        # TODO for better generalizability of the architecture but worse training stability, project motion and compute losses on the projected motion
        motion_output = self.output_process(motion)[:,self.motion_prefix_len:,:]  # [batch_size, seq_len - motion_prefix_len, vqvae_dim]
        #motion_output_projection = self.output_projection(output)  # [batch_size, seq_len, latent_dim]

        '''
        print("INPUT", motion_noised_without_process[0])
        print("INPUT PROCESS",motion_noised[0])
        print("OUTPUT",motion_output[0])
        print("motion_output_projection_pooled", motion_output_projection_pooled[0])
        print("MOTION",motion[0])
        print("AUDIO CONTEXT",audio_context[0])
        print("HIGH LEVEL CONTEXT",overall_high_level_context[0])
        print("MOTION MASK",full_motion_mask)
        print("AUDIO MASK",full_audio_mask)

        #l2_distance = torch.norm(motion_noised_without_process - motion_output, p=2)
        with torch.no_grad():
            distance = mse_loss(motion_noised_without_process, motion_output)
            print("DISTANCE", distance.item())
        '''
        if self.culture_classification_layer is not None:
            culture_output = self.culture_classification_layer(motion_output) #[batch_size, n_cultures] #TODO put motion_output_projection if needed
        else:
            culture_output = torch.zeros(
                motion_output.shape[0],
                self.n_cultures,
                dtype=motion_output.dtype,
                device=motion_output.device,
            )
        # [batch_size, seq_len - motion_prefix_len, vqvae_dim], [batch_size, n_cultures], [batch_size, vqvae_dim], [batch_size, seq_len - motion_prefix_len, latent_dim], [batch_size, latent_dim], [batch_size, seq_len - motion_prefix_len], [batch_size, latent_dim]
        return (motion_output, culture_output, motion_output_projection_pooled,
                audio_context[:, self.motion_prefix_len:, :], high_level_context,
                motion_mask, audio_mask, overall_audio_context.squeeze(1)) #TODO output also motion_output_projection if needed

# This is to reduce mask dimension. For instance, If I have a masked audio vector of dim (50,1024)
# and I want to reduce it to (25,1024) to match motion dimension and allow loss computation, I need to reduce masks accordingly
# By sliding with a window_size = 2 and stride = 2, I can reduce the mask dimension to (25). If there is any masked frame in
# the window, the pooled result is masked
def pool_mask(mask, window_size, stride, target_length):
    """
    Pool a mask tensor in the same way as WindowedAttentionPooling does for features.

    Args:
        mask: Boolean tensor of shape [batch_size, seq_len, dim] (True=masked).
        window_size: Size of each window used by the pooling layer.
        stride: Stride used by the pooling layer.
        target_length: The desired output length after pooling.

    Returns:
        A boolean mask tensor of shape [batch_size, target_length] for the pooled sequence.
    """
    device = mask.device
    batch_size, seq_len, dim = mask.shape

    # Reduce mask along the feature dimension to get a frame-level mask.
    # If any feature among all the features in the frame is masked, the frame is masked.
    frame_mask = mask.any(dim=2)  # [batch_size, seq_len]

    # Compute how many windows we have
    num_windows = (seq_len - window_size) // stride + 1

    # Use unfold to create windows: [batch_size, num_windows, window_size]
    windows = frame_mask.unfold(dimension=1, size=window_size, step=stride)

    # If any frame in a time window is masked, the pooled result is masked
    pooled_mask = windows.any(dim=2)  # [batch_size, num_windows]

    # Adjust to match target_length if necessary
    if num_windows > target_length:
        pooled_mask = pooled_mask[:, :target_length]
    elif num_windows < target_length:
        # Pad with False if we have fewer windows than target_length
        padding = target_length - num_windows
        pad_tensor = torch.zeros(batch_size, padding, dtype=torch.bool, device=device)
        pooled_mask = torch.cat((pooled_mask, pad_tensor), dim=1)

    return pooled_mask  # [batch_size, target_length]

# This is used to random mask frames in the sequence. In the current version is used for low level
# features and input motion. Note that is used only for the 4 second of noise motion and related low level features
# as we assume that prefix is important for reconstruction. Also high level context is not masked as it is used for
# embedding timestep t (which is necessary for motion reconstruction), motion which is foundamental for generating culture-aware
# co-speech gestures, and text input
class RandomMasking(nn.Module):
    def __init__(self, mask_prob=0.1, mask_value=0.0):
        """
        Args:
            mask_prob (float): Probability of masking each frame.
            mask_value (float or Tensor): Value to replace masked frames.
        """
        super(RandomMasking, self).__init__()
        self.mask_prob = mask_prob
        self.mask_value = mask_value
        self.mask_value = mask_value

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape [batch_size, seq_len, dim]

        Returns:
            Tensor: Masked tensor
            Tensor: Mask tensor indicating which frames are masked
        """
        if not self.training:
            # During evaluation, return the input as-is and a mask of all False
            mask = torch.zeros(x.size(), dtype=torch.bool, device=x.device)
            return x, mask
        batch_size, seq_len, dim = x.size()
        mask = torch.rand(batch_size, seq_len, 1, device=x.device) < self.mask_prob  # [seq_len, batch_size, 1]
        mask = mask.expand(-1, -1, dim)  # [batch_size, seq_len, dim]
        x_masked = x.clone()
        x_masked[mask] = self.mask_value
        return x_masked, mask


class HierarchicalCrossAttentionLayer(nn.Module):
    def __init__(self, latent_dim, num_heads, ff_size, dropout=0.1, activation='gelu'):
        """
        Hierarchical Cross-Attention Layer: motion <-> audio, then motion <-> culture.

        Args:
            latent_dim (int): Dimensionality of the latent space.
            num_heads (int): Number of attention heads.
            ff_size (int): Hidden size of the feedforward network.
            dropout (float): Dropout rate.
            activation (str): Activation function for the feedforward layer.
        """
        super(HierarchicalCrossAttentionLayer, self).__init__()

        # Self-attention for motion
        self.self_attn = nn.MultiheadAttention(latent_dim, num_heads, dropout=dropout, batch_first=True)

        # Cross-attention for motion <-> audio
        self.cross_attn_audio = nn.MultiheadAttention(latent_dim, num_heads, dropout=dropout, batch_first=True)

        # Cross-attention for motion <-> culture
        self.cross_attn_context = nn.MultiheadAttention(latent_dim, num_heads, dropout=dropout, batch_first=True)

        # Feed-forward network
        self.feedforward = nn.Sequential(
            nn.Linear(latent_dim, ff_size),
            nn.ReLU() if activation == 'relu' else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, latent_dim),
            nn.Dropout(dropout)
        )

        # Layer normalizations
        self.norm1 = nn.LayerNorm(latent_dim)  # For self-attention
        self.norm2 = nn.LayerNorm(latent_dim)  # For cross-attention (audio)
        self.norm3 = nn.LayerNorm(latent_dim)  # For cross-attention (culture)
        self.norm4 = nn.LayerNorm(latent_dim)  # For feed-forward network

        # Dropouts
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

    def forward(self, motion, audio, culture, motion_mask=None, audio_mask=None, culture_mask=None):
        """
        Forward pass for the hierarchical cross-attention layer.

        Args:
            motion (Tensor): Motion features [batch_size, seq_len_motion, latent_dim].
            audio (Tensor): Audio features [batch_size, seq_len_audio, latent_dim].
            culture (Tensor): Culture features [batch_size, seq_len_culture, latent_dim].
            motion_mask (Tensor, optional): Mask for motion [batch_size, seq_len_motion].
            audio_mask (Tensor, optional): Mask for audio [batch_size, seq_len_audio].
            culture_mask (Tensor, optional): Mask for culture [batch_size, seq_len_culture].

        Returns:
            Tensor: Updated motion features [batch_size, seq_len_motion, latent_dim].
        """
        # Self-attention for motion
        motion_residual = motion
        motion = self.norm1(motion)
        motion_self_attn, _ = self.self_attn(motion, motion, motion, key_padding_mask=motion_mask)
        motion = motion_residual + self.dropout1(motion_self_attn)

        # Cross-attention between motion and audio
        motion_residual = motion
        motion = self.norm2(motion)
        motion_audio_attn, _ = self.cross_attn_audio(motion, audio, audio, key_padding_mask=audio_mask)
        motion = motion_residual + self.dropout2(motion_audio_attn)

        # Cross-attention between motion and culture
        motion_residual = motion
        motion = self.norm3(motion)
        motion_culture_attn, _ = self.cross_attn_context(motion, culture, culture, key_padding_mask=culture_mask)
        motion = motion_residual + self.dropout3(motion_culture_attn)

        # Feed-forward network
        motion_residual = motion
        motion = self.norm4(motion)
        motion_ff = self.feedforward(motion)
        motion = motion_residual + self.dropout4(motion_ff)

        return motion


class NativeCrossAttentionBlock(nn.Module):
    """Native decoder-style block: self-attn on motion, cross-attn to audio, then FFN."""

    def __init__(self, latent_dim, num_heads, ff_size, dropout=0.1, activation='gelu'):
        super().__init__()
        self.motion_audio_decoder = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )

    def forward(self, x, context, x_mask=None, context_mask=None):
        # TransformerDecoderLayer performs:
        # 1) tgt self-attention, 2) tgt->memory cross-attention, 3) FFN.
        x = self.motion_audio_decoder(
            tgt=x,
            memory=context,
            tgt_key_padding_mask=x_mask,
            memory_key_padding_mask=context_mask,
        )
        return x

# This is batch_first version of MultiheadAttention
class CrossAttentionLayer(nn.Module):
    def __init__(self, latent_dim, num_heads, ff_size, dropout=0.1, activation='gelu'):
        super(CrossAttentionLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(latent_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(latent_dim, num_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(latent_dim, ff_size)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ff_size, latent_dim)

        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.norm3 = nn.LayerNorm(latent_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        if activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'relu':
            self.activation = F.relu
        else:
            raise ValueError('Unsupported activation function')

    def forward(self, x, context,  x_mask=None, context_mask=None):
        # x: [batch_size, seq_len, latent_dim]
        # context: [batch_size, context_seq_len, latent_dim]
        # x_mask, context_mask: [batch_size, seq_len] booleans (True=mask)
        # MultiheadAttention in batch_first expects bool masks with True = ignored.

        # Self-attention
        x2, _ = self.self_attn(x, x, x, key_padding_mask=x_mask)#[0]
        x = x + self.dropout1(x2)
        x = self.norm1(x)

        # Cross-attention with context
        x2, _ = self.cross_attn(x, context, context, key_padding_mask=context_mask)#[0]
        x = x + self.dropout2(x2)
        x = self.norm2(x)

        # Feed-forward network
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout3(x2)
        x = self.norm3(x)

        #print("STEP CROSS ATT", x[0])

        return x

# To insert the high level context similar to FiLM
class ConditionalLayerNorm(nn.Module):
    def __init__(self, latent_dim, condition_dim):
        super(ConditionalLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(latent_dim)
        self.gamma = nn.Linear(condition_dim, latent_dim)
        self.beta = nn.Linear(condition_dim, latent_dim)

    def forward(self, x, condition):
        norm_x = self.layer_norm(x)
        gamma = self.gamma(condition).unsqueeze(1)  # [batch, 1, latent_dim]
        beta = self.beta(condition).unsqueeze(1)    # [batch, 1, latent_dim]
        return gamma * norm_x + beta


class AdaIN(nn.Module):
    def __init__(self, latent_dim, style_dim):
        super(AdaIN, self).__init__()
        self.style_scale_transform = nn.Linear(style_dim, latent_dim)
        self.style_shift_transform = nn.Linear(style_dim, latent_dim)
        '''
        nn.init.constant_(self.style_scale_transform.weight, 0)
        nn.init.constant_(self.style_scale_transform.bias, 1)  # Scale starts at 1
        nn.init.constant_(self.style_shift_transform.weight, 0)
        nn.init.constant_(self.style_shift_transform.bias, 0)  # Shift starts at 0
        '''
    def forward(self, x, style_embedding):
        # Calculate per-feature mean and variance
        timestep = x[:,0]
        motion = x[:,1:]
        mean = motion.mean(dim=1, keepdim=True)
        std = motion.std(dim=1, keepdim=True) + 1e-5  # Add epsilon for numerical stability
        normalized_motion = (motion - mean) / std  # Normalize input
        # Compute scale and shift from style embedding
        scale = self.style_scale_transform(style_embedding).unsqueeze(1)
        shift = self.style_shift_transform(style_embedding).unsqueeze(1)
        normalized_motion = scale * normalized_motion + shift
        final_motion = torch.cat([timestep.unsqueeze(1),normalized_motion],dim=1)
        # Apply scale and shift
        return final_motion

class GradualAdaIN(nn.Module):
    def __init__(self, latent_dim, style_dim, initial_alpha=0.0, final_alpha=1, ramp_up_steps=1000):
        super(GradualAdaIN, self).__init__()
        self.style_scale_transform = nn.Linear(style_dim, latent_dim)
        self.style_shift_transform = nn.Linear(style_dim, latent_dim)
        self.register_buffer('alpha', torch.tensor(initial_alpha))
        self.final_alpha = final_alpha
        self.ramp_up_steps = ramp_up_steps
        self.current_step = 0

        # Initialize scale to ones and shift to zeros
        nn.init.constant_(self.style_scale_transform.weight, 0)
        nn.init.constant_(self.style_scale_transform.bias, 1)  # Scale starts at 1
        nn.init.constant_(self.style_shift_transform.weight, 0)
        nn.init.constant_(self.style_shift_transform.bias, 0)  # Shift starts at 0

    def step_alpha(self):
        if self.current_step < self.ramp_up_steps:
            increment = (self.final_alpha - self.alpha.item()) / self.ramp_up_steps
            self.alpha += torch.tensor(increment, device=self.alpha.device)
            self.current_step += 1
        else:
            self.alpha = torch.tensor(self.final_alpha, device=self.alpha.device)

    def forward(self, x, style_embedding):
        # Normalize style embeddings
        #style_embedding = F.normalize(style_embedding, p=2, dim=-1)

        # Calculate per-feature mean and variance
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-5  # Add epsilon for numerical stability
        normalized_x = (x - mean) / std  # Normalize input

        # Compute scale and shift from style embedding
        scale = self.style_scale_transform(style_embedding).unsqueeze(1)  # [batch_size, 1, latent_dim]
        shift = self.style_shift_transform(style_embedding).unsqueeze(1)  # [batch_size, 1, latent_dim]

        # Apply scale and shift
        adain_x = scale * normalized_x + shift  # [batch_size, seq_len, latent_dim]

        # Gated combination with dynamic alpha
        return (1 - self.alpha) * x + self.alpha * adain_x  # [batch_size, seq_len, latent_dim]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout_layer = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1) # (max_len, 1, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size,seq_length, d_model)
        batch_size, seq_length, d_model = x.shape
        x = x.permute(1, 0, 2) # (seq_length, batch_size, d_model)
        x = x + self.pe[:seq_length, :]
        x= x.permute(1, 0, 2)  # back to (batch_size, seq_length, d_model)
        return self.dropout_layer(x)


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
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]) #[ bs, 1, d]


# This is used to downsample a representation. For instance, If I have a vector (156,256) in input and
# I want in output (50,256), given a stride and window size telling the dynamics on which I check the most
# important components for downsampling, I slide this window and the values in this window are passed through a
# a linear layer and a softmax, telling how much the values in this window are important for the reconstruction
# of the vector (50,256). This is useful in all cases where there is an overlap, so for overlapping windows I give
# more importance to the ones having the highest attention score when summing up to get the final vector. If the desired
# output dimension is different from 256, then I also project with another FFN layer to different space
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
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.gelu= nn.GELU()

        # Add a projection layer to map to the desired output dimension
        self.output_projection_layer = nn.Linear(d_model, output_dim)
        self.output_dropout = nn.Dropout(dropout)  # Dropout after output projection

    def forward(self, data):
        # poses: (batch_size, seq_length, feature_dim)
        data = self.input_projection_layer(data)
        data = self.input_dropout(data)
        data = self.pos_encoder(data)
        data = data.permute(1, 0, 2)  # (seq_length, batch_size, feature_dim)
        transformer_output = self.transformer_encoder(data)  # (seq_length, batch_size, d_model)
        transformer_output = self.layer_norm(transformer_output)
        transformer_output = transformer_output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        #pooled_output = self.attention_pool(transformer_output)  # (batch_size, d_model)
        #transformer_output = transformer_output.mean(dim=0)  # (batch_size, d_model)
        projected_output = self.output_projection_layer(transformer_output)  # Map to (batch_size, output_dim)
        projected_output = self.gelu(projected_output)
        projected_output = self.output_dropout(projected_output)
        return projected_output


class InputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, dropout):
        super().__init__()
        self.data_rep = data_rep # 'vqvae_codebooks', at the moment this information is not useful as it is the only representation used
        self.input_feats = input_feats #512
        self.latent_dim = latent_dim #512
        self.layer_norm = nn.LayerNorm(latent_dim)
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, num_features = x.shape #(batch_size, 25, 512)
        #x = x.permute(1, 0, 2).reshape(seq_len, batch_size, num_features)
        x = self.poseEmbedding(x)  # [batch_size, seq_len, latent_dim]
        x = self.gelu(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x

class OutputProcess(nn.Module):
    def __init__(self, data_rep, output_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = output_feats
        self.latent_dim = latent_dim
        self.poseFinal = nn.Linear(self.latent_dim, output_feats)

    def forward(self, output):
        # output = [batch_size, seq_len, latent_dim]
        output = self.poseFinal(output)  # [batch_size, seq_len, vqvae_dim]
        #output = output.permute(1, 0, 2)
        return output

'''
# Example usage:
if __name__ == "__main__":
    # Example input dimensions
    batch_size = 8
    num_joints = 25
    num_features = 512
    sequence_length = 25  # 5 seconds of motion, 5 + 20 embeddings
    latent_dim = 512
    audio_onset_dim = 156
    audio_mel_dim = 64
    text_embedding_dim = 768
    culture_embedding_dim = 512
    num_heads = 8
    num_layers = 4
    ff_size = 1024

    # Create model instance
    model = Hierarchical_MDM(
        pose_embedding_dim=num_features,
        latent_dim=latent_dim,
        num_joints=num_joints,
        num_features=num_features,
        audio_onset_dim=audio_onset_dim,
        audio_mel_dim=audio_mel_dim,
        text_embedding_dim=text_embedding_dim,
        culture_embedding_dim=culture_embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_size=ff_size
    )

    # Create dummy inputs
    x_input = torch.randn(batch_size, num_joints, num_features, sequence_length)
    timesteps = torch.randint(low=0, high=1000, size=(batch_size,))

    conditioning_inputs = {
        'onsets': torch.randn(batch_size, sequence_length, audio_onset_dim),
        'mel': torch.randn(batch_size, sequence_length, audio_mel_dim),
        'text': torch.randn(batch_size, text_embedding_dim),
        'culture': torch.randn(batch_size, culture_embedding_dim)
    }

    # Forward pass
    output = model(x_input, timesteps, y=conditioning_inputs)

    print(f"Output shape: {output.shape}")  # Should be [batch_size, num_joints, num_features, sequence_length]
'''
'''
    Parameters: {'modeltype': '', 'njoints': njoints, 'nfeats': nfeats, 'num_actions': num_actions,
            'translation': True, 'pose_rep': 'rot6d', 'glob': True, 'glob_rot': True,
            'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
            'dropout': 0.1, 'activation': "gelu", 'data_rep': data_rep, 'cond_mode': cond_mode,
            'cond_mask_prob': args.cond_mask_prob, 'action_emb': action_emb, 'arch': args.arch,
            'emb_trans_dec': args.emb_trans_dec, 'clip_version': clip_version, 'dataset': args.dataset,
            'emb_before_mask': args.emb_before_mask, 'text_encoder_type': args.text_encoder_type,
            'pos_embed_max_len': args.pos_embed_max_len, 'mask_frames': args.mask_frames,
            'keyframe_cond_type': args.keyframe_cond_type,
            'pred_len': args.pred_len, 'context_len': args.context_len, 'emb_policy': emb_policy,
            'all_goal_joint_names': all_goal_joint_names, 'multi_target_cond': multi_target_cond, 'multi_encoder_type': multi_encoder_type, 'target_enc_layers': target_enc_layers,
            }
    args.latent_dim = 512, args.cond_mask_prob = .1 = The probability of masking the condition during training
    args.arch = trans_dec, args.emb_before_mask = store_true = If true - for the cond branch - flip between mask and linear blocks
    args.pos_embed_max_len = 5000 = Pose embedding max length, mask_frames = store_True = If true, will fix Rotem's bug and mask invalid frames.
    args.keyframe_cond_typ = '' = prefix conditioning us designed for the RL task. Default is no conditioning
    args.pred_len = 40 = prediction len, args.context_len = 20,  
    args.emb_trans_dec = store_True,  For trans_dec architecture only, if true, will inject condition as a class token
                             (in addition to cross-attention).

'''

'''
class Hierchical_MDM(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d', dataset='amass', clip_dim=512,
                 arch='trans_enc', emb_trans_dec=False, clip_version=None, **kargs):
        super().__init__()

        self.legacy = legacy # False
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_actions = num_actions
        self.data_rep = data_rep
        self.dataset = dataset

        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation
        self.clip_dim = clip_dim #TODO: deprecate
        
        self.keyframe_cond_type = kargs.get('keyframe_cond_type', '')
        self.input_feats = self.njoints * self.nfeats

        self.normalize_output = kargs.get('normalize_encoder_output', False)

        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.emb_before_mask = kargs.get('emb_before_mask', False)
        self.mask_frames = kargs.get('mask_frames', False)
        self.arch = arch
        # self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0
        # self.input_process = InputProcess(self.data_rep, self.input_feats+self.gru_emb_dim, self.latent_dim)
        #projects input poses to latent embedding of dim = 512
        self.input_process = InputProcess(self.data_rep, self.input_feats + (self.keyframe_cond_type != ''), self.latent_dim)

        self.emb_policy = kargs.get('emb_policy', 'add')

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout, max_len=kargs.get('pos_embed_max_len', 5000))
        self.emb_trans_dec = emb_trans_dec

        self.pred_len = kargs.get('pred_len', 0)
        self.context_len = kargs.get('context_len', 0)
        self.total_len = self.pred_len + self.context_len
        self.is_prefix_comp = self.total_len > 0
        self.all_goal_joint_names = kargs.get('all_goal_joint_names', [])
        
        self.multi_target_cond = kargs.get('multi_target_cond', False)
        self.multi_encoder_type = kargs.get('multi_encoder_type', 'multi')
        self.target_enc_layers = kargs.get('target_enc_layers', 1)
        if self.multi_target_cond:
            if self.multi_encoder_type == 'multi':
                self.embed_target_cond = EmbedTargetLocMulti(self.all_goal_joint_names, self.latent_dim)
            elif self.multi_encoder_type == 'single':
               self.embed_target_cond = EmbedTargetLocSingle(self.all_goal_joint_names, self.latent_dim, self.target_enc_layers)       
            elif self.multi_encoder_type == 'split':
               self.embed_target_cond = EmbedTargetLocSplit(self.all_goal_joint_names, self.latent_dim, self.target_enc_layers)     
        
        if self.arch == 'trans_enc':
            print("TRANS_ENC init")
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation)

            self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                         num_layers=self.num_layers)
        elif self.arch == 'trans_dec':
            print("TRANS_DEC init")
            seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=activation)
            self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                         num_layers=self.num_layers)
        elif self.arch == 'gru':
            print("GRU init")
            self.gru = nn.GRU(self.latent_dim, self.latent_dim, num_layers=self.num_layers, batch_first=True)
        else:
            raise ValueError('Please choose correct architecture [trans_enc, trans_dec, gru]')
        
        #positional encoding with PositionalEncoding(self.latent_dim, self.dropout, max_len=kargs.get('pos_embed_max_len', 5000))
        # Embeds timestep with Linear layer
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder) 

        if self.cond_mode != 'no_cond': 
            if 'text' in self.cond_mode:
                # We support CLIP encoder and DistilBERT
                print('EMBED TEXT')
                
                self.text_encoder_type = kargs.get('text_encoder_type', 'clip')
                
                if self.text_encoder_type == "clip":
                    print('Loading CLIP...')
                    self.clip_version = clip_version
                    self.clip_model = self.load_and_freeze_clip(clip_version)
                    self.encode_text = self.clip_encode_text
                elif self.text_encoder_type == 'bert':
                    assert self.arch == 'trans_dec'
                    # assert self.emb_trans_dec == False # passing just the time embed so it's fine
                    print("Loading BERT...")
                    # bert_model_path = 'model/BERT/distilbert-base-uncased'
                    bert_model_path = 'distilbert/distilbert-base-uncased'
                    self.clip_model = load_bert(bert_model_path)
                    self.encode_text = self.bert_encode_text
                    self.clip_dim = 768
                else:
                    raise ValueError('We only support [CLIP, BERT] text encoders') 
                    
                # from text_dim 768, from clip dim to d, where d is the transformer dim. In this way we can compute cross-attention
                # Note that final dimension is Ntokens x d as the input text is Ntokens x clip_dim which is then transformer to Ntokens x d
                self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
                
            if 'action' in self.cond_mode:
                self.embed_action = EmbedAction(self.num_actions, self.latent_dim)
                print('EMBED ACTION')
        #rot6d, n_joints * n_features, 512, n_joints and n_feats depending on dataset
        # adds a fully connected layer at the end and remaps to have the predicted gestures
        self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints,
                                            self.nfeats)

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def mask_cond(self, cond, force_mask=False):
        # cond is text already mapped by linear layer to seq_len * d 
        # cond_mask_prob = .1 
        seq_len, bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond) #put all zeros for the condition 
        elif self.training and self.cond_mask_prob > 0.:
        #Note that the mask is applied to an entire condition element instead of numbers inside (seq_len,d). This means
        #that a (seq_len,d) condition inside the batch can be either 0 or 1 with prob .1. Otherwise is like dropout instead of mask
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(1, bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask) #if mask = 1 then that element is masked
        else:
            return cond

    def clip_encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device
        max_text_len = 20 if self.dataset in ['humanml', 'kit'] else None  # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2 # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
            # print('texts', texts.shape)
            zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
            # print('texts after pad', texts.shape, texts)
        else:
            texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.clip_model.encode_text(texts).float().unsqueeze(0)
    
    def bert_encode_text(self, raw_text):
        # enc_text = self.clip_model(raw_text)
        # enc_text = enc_text.permute(1, 0, 2)
        # return enc_text
        enc_text, mask = self.clip_model(raw_text)  # self.clip_model.get_last_hidden_state(raw_text, return_mask=True)  # mask: False means no token there
        enc_text = enc_text.permute(1, 0, 2)
        mask = ~mask  # mask: True means no token there, we invert since the meaning of mask for transformer is inverted  https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
        return enc_text, mask

    #Note that noise is added before getting here
    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        bs, njoints, nfeats, nframes = x.shape #this is only xt_pred, it doesn't contain the prefix
        time_emb = self.embed_timestep(timesteps)  # [1, bs, d]

        if 'target_cond' in y.keys():  
            # time_emb += self.mask_cond(self.embed_target_cond(y['target_cond'], y['target_joint_names'], y['is_heading'])[None], force_mask=y.get('target_uncond', False))  # For uncond support and CFG
            # add time embedding to target embedding. Note that time is added to target but not to text. In text can concatenated or added. Default is added
            time_emb += self.embed_target_cond(y['target_cond'], y['target_joint_names'], y['is_heading'])[None]  # We don't use CFG for joints!

        # Build input for prefix completion
        if self.is_prefix_comp: #input prefix + pred (x) noised at time t. Now x contains also the prefix
            x = torch.cat([y['prefix'], x], dim=-1)
            y['mask'] = torch.cat([torch.ones([bs, 1, 1, self.context_len], dtype=y['mask'].dtype, device=y['mask'].device), 
                                   y['mask']], dim=-1) #it doesn't mask self.context_len. A series of ones are added to y['mask']

        # process tensor to include keyframes if needed
        # In current setup keyframe_cond_type == '' so I don't enter here
        if self.keyframe_cond_type != '':
            if 'keyframe_mask' not in y.keys():
                print(f'Warning: although [keyframe_cond_type={self.keyframe_cond_type}], [keyframe_mask] was not specified, hence not conditioning on keyframes.')
                y['keyframe_mask'] = torch.zeros_like(y['mask']).to(x.device)
                y['clean_motion'] = torch.zeros_like(x)
            x = (y['clean_motion'] * y['keyframe_mask']) + (x * ~y['keyframe_mask'])  # [batch_size, njoints, nfeats, max_frames]
            keyframes_ch = y['keyframe_mask'].to(x.dtype) *2. -1.
            x = torch.cat([x, keyframes_ch], dim=1)   # [batch_size, njoints+1, nfeats, max_frames]

        force_mask = y.get('text_uncond', False)
        if 'text' in self.cond_mode:
            if 'text_embed' in y.keys():  # caching option
                enc_text = y['text_embed']
            else:
                enc_text = self.encode_text(y['text'])
            if type(enc_text) == tuple:
                enc_text, text_mask = enc_text
                if text_mask.shape[0] == 1 and bs > 1:  # casting mask for the single-prompt-for-all case
                    text_mask = torch.repeat_interleave(text_mask, bs, dim=0)
            if self.emb_before_mask:
                text_emb = self.mask_cond(self.embed_text(enc_text), force_mask=force_mask)
            else:  # default
                # each element in the batch can be either 0 or the element itself according to mask
                text_emb = self.embed_text(self.mask_cond(enc_text, force_mask=force_mask))  # casting mask for the single-prompt-for-all case
            if self.emb_policy == 'add':  #time_emb is added to text and concatened to x_prefix + x_pred (below)
                emb = text_emb + time_emb
            else:
                emb = torch.cat([time_emb, text_emb], dim=0)
                text_mask = torch.cat([torch.zeros_like(text_mask[:, 0:1]), text_mask], dim=1)
        if 'action' in self.cond_mode:
            action_emb = self.embed_action(y['action'])
            emb += self.mask_cond(action_emb, force_mask=force_mask)

        if self.arch == 'gru':
            x_reshaped = x.reshape(bs, njoints*nfeats, 1, nframes)
            emb_gru = emb.repeat(nframes, 1, 1)     #[#frames, bs, d]
            emb_gru = emb_gru.permute(1, 2, 0)      #[bs, d, #frames]
            emb_gru = emb_gru.reshape(bs, self.latent_dim, 1, nframes)  #[bs, d, 1, #frames]
            x = torch.cat((x_reshaped, emb_gru), axis=1)  #[bs, d+joints*feat, 1, #frames]
        #map poses (prefix + noised pred) 
        # process the input and map it Nframes x d, where d is the transformer Dimension 
        # Note that Ntokens may be not the same of Nframes. 
        x = self.input_process(x) 

        # TODO - move to collate
        # Masks input frames x. It works only if y['mask'] has multiple time frames, then I mask some input frames
        frames_mask = None
        is_valid_mask = y['mask'].shape[-1] > 1  # Don't use mask with the generate script
        if self.mask_frames and is_valid_mask:
            frames_mask = torch.logical_not(y['mask'][..., :x.shape[0]].squeeze(1).squeeze(1)).to(device=x.device)
            if self.emb_trans_dec or self.arch == 'trans_enc':
                step_mask = torch.zeros((bs, 1), dtype=torch.bool, device=x.device)
                frames_mask = torch.cat([step_mask, frames_mask], dim=1)

        if self.arch == 'trans_enc':
            # adding the timestep embed
            xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            output = self.seqTransEncoder(xseq, src_key_padding_mask=frames_mask)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        elif self.arch == 'trans_dec':
            if self.emb_trans_dec:
                # add the time embedding to x. after concatenation, the sequence length of the sequence is no more 60 (prefix + noisy x_pred) but 61
                # so the input x contains also the embedded information about timestep t of noise. Does it makes sense to 
                # add the positional encoding to different kinds of data? TODO discover it. PROBABLY NOT!!!!!
                xseq = torch.cat((time_emb, x), axis=0)
            else:
                xseq = x
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]. adds positional encoding. This is not timestep t of diffusion 

            if self.text_encoder_type == 'clip':
                output = self.seqTransDecoder(tgt=xseq, memory=emb, tgt_key_padding_mask=frames_mask)
            elif self.text_encoder_type == 'bert':
                #cross attention between x_prefix + x_pred noisy + encoded diffusion timestep t, and text
                output = self.seqTransDecoder(tgt=xseq, memory=emb, memory_key_padding_mask=text_mask, tgt_key_padding_mask=frames_mask)  # Rotem's bug fix
            else:
                raise ValueError()

            if self.emb_trans_dec:
                output = output[1:] # [seqlen, bs, d]

        elif self.arch == 'gru':
            xseq = x
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen, bs, d]
            output, _ = self.gru(xseq)

        # Extract completed suffix
        if self.is_prefix_comp:
            output = output[self.context_len:] #extract only x_pred (note that output doesn't contain the first element, i.e. timestep t as I did utput = output[1:])
            y['mask'] = y['mask'][..., self.context_len:]
        #now the output contains only x_pred as I removed timestep emebdding and prefix. Map it back to original representation 
        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
        return output



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) 

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)
        


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
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]) #[ bs, 1, d]
        



class InputProcess(nn.Module): 
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        if self.data_rep == 'rot_vel':
            self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
    # x is x_pred, so n_frames = 40
    # input_feats is n_joints * n_feats. So input is bs, input_feats, n_frames. Reshaped is (n_frames, bs , njoints*nfeats)
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)

        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            x = self.poseEmbedding(x)  # [seqlen, bs, d] #seflen = n_frames = 40
            return x
        elif self.data_rep == 'rot_vel':
            first_pose = x[[0]]  # [1, bs, 150]
            first_pose = self.poseEmbedding(first_pose)  # [1, bs, d]
            vel = x[1:]  # [seqlen-1, bs, 150]
            vel = self.velEmbedding(vel)  # [seqlen-1, bs, d]
            return torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, d]
        else:
            raise ValueError


class OutputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats) #project latent embedding (512) to inputs, i.e. n_joints * n_feats
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


class EmbedAction(nn.Module):
    def __init__(self, num_actions, latent_dim):
        super().__init__()
        self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim))

    def forward(self, input):
        idx = input[:, 0].to(torch.long)  # an index array must be long
        output = self.action_embedding[idx]
        return output
    
class EmbedTargetLocSingle(nn.Module):
    def __init__(self, all_goal_joint_names, latent_dim, num_layers=1):
        super().__init__()
        self.extended_goal_joint_names = all_goal_joint_names + ['traj', 'heading']
        self.target_cond_dim = len(self.extended_goal_joint_names) * 4  # 4 => (x,y,z,is_valid)
        self.latent_dim = latent_dim
        _layers = [nn.Linear(self.target_cond_dim, self.latent_dim)]
        for _ in range(num_layers):
            _layers += [nn.SiLU(), nn.Linear(self.latent_dim, self.latent_dim)]
        self.mlp = nn.Sequential(*_layers)

    def forward(self, input, target_joint_names, target_heading):
        # TODO - generate validity from outside the model
        validity = torch.zeros_like(input)[..., :1]
        for sample_idx, sample_joint_names in enumerate(target_joint_names):
            sample_joint_names_w_heading = np.append(sample_joint_names, 'heading') if target_heading[sample_idx] else sample_joint_names
            for j in sample_joint_names_w_heading:
                validity[sample_idx, self.extended_goal_joint_names.index(j)] = 1.

        mlp_input = torch.cat([input, validity], dim=-1).view(input.shape[0], -1)
        return self.mlp(mlp_input)


class EmbedTargetLocSplit(nn.Module):
    def __init__(self, all_goal_joint_names, latent_dim, num_layers=1):
        super().__init__()
        self.extended_goal_joint_names = all_goal_joint_names + ['traj', 'heading']
        self.target_cond_dim = 4
        self.latent_dim = latent_dim
        self.splited_dim = self.latent_dim // len(self.extended_goal_joint_names)
        assert self.latent_dim % len(self.extended_goal_joint_names) == 0
        self.mini_mlps = nn.ModuleList()
        for _ in self.extended_goal_joint_names:
            _layers = [nn.Linear(self.target_cond_dim, self.splited_dim)]
            for _ in range(num_layers):
                _layers += [nn.SiLU(), nn.Linear(self.splited_dim, self.splited_dim)]
            self.mini_mlps.append(nn.Sequential(*_layers))

    def forward(self, input, target_joint_names, target_heading):
        # TODO - generate validity from outside the model
        validity = torch.zeros_like(input)[..., :1]
        for sample_idx, sample_joint_names in enumerate(target_joint_names):
            sample_joint_names_w_heading = np.append(sample_joint_names, 'heading') if target_heading[sample_idx] else sample_joint_names
            for j in sample_joint_names_w_heading:
                validity[sample_idx, self.extended_goal_joint_names.index(j)] = 1.

        mlp_input = torch.cat([input, validity], dim=-1)
        mlp_splits = [self.mini_mlps[i](mlp_input[:, i]) for i in range(mlp_input.shape[1])] 
        return torch.cat(mlp_splits, dim=-1)
  
class EmbedTargetLocMulti(nn.Module):
    def __init__(self, all_goal_joint_names, latent_dim):
        super().__init__()
        
        # todo: use a tensor of weight per joint, and another one for biases, then apply a selection in one go like we to for actions
        self.extended_goal_joint_names = all_goal_joint_names + ['traj', 'heading']
        self.extended_goal_joint_idx = {joint_name: idx for idx, joint_name in enumerate(self.extended_goal_joint_names)}
        self.n_extended_goal_joints = len(self.extended_goal_joint_names)
        self.target_loc_emb = nn.ParameterDict({joint_name: 
            nn.Sequential(
                nn.Linear(3, latent_dim),
                nn.SiLU(),
                nn.Linear(latent_dim, latent_dim)) 
            for joint_name in self.extended_goal_joint_names})  # todo: check if 3 works for heading and traj
            # nn.Linear(3, latent_dim) for joint_name in self.extended_goal_joint_names})  # todo: check if 3 works for heading and traj
        self.target_all_loc_emb = WeightedSum(self.n_extended_goal_joints) # nn.Linear(self.n_extended_goal_joints, latent_dim)
        self.latent_dim = latent_dim

    def forward(self, input, target_joint_names, target_heading):
        output = torch.zeros((input.shape[0], self.latent_dim), dtype=input.dtype, device=input.device)
        
        # Iterate over the batch and apply the appropriate filter for each joint
        for sample_idx, sample_joint_names in enumerate(target_joint_names):
            sample_joint_names_w_heading = np.append(sample_joint_names, 'heading') if target_heading[sample_idx] else sample_joint_names
            output_one_sample = torch.zeros((self.n_extended_goal_joints, self.latent_dim), dtype=input.dtype, device=input.device)
            for joint_name in sample_joint_names_w_heading:
                layer = self.target_loc_emb[joint_name]
                output_one_sample[self.extended_goal_joint_idx[joint_name]] = layer(input[sample_idx, self.extended_goal_joint_idx[joint_name]])  
            output[sample_idx] = self.target_all_loc_emb(output_one_sample)
            # print(torch.where(output_one_sample.sum(axis=1)!=0)[0].cpu().numpy())
               
        return output
'''

'''
audio_noised_motion = audio_context[:, self.audio_prefix_len:, :]  # match noised portion of motion in audio
print("audio_noised_motion",audio_noised_motion.shape)
audio_noised_motion, audio_mask = self.random_audio_masking(audio_noised_motion) #mask random frames in audio. both are [batch_size, wav2vec_seq_len - audio_prefix_len, latent_dim].
print("audio_noised_motion", audio_noised_motion.shape, "audio_mask", audio_mask.shape)
pooled_audio_mask = pool_mask(audio_mask, window_size=2, stride=2, target_length=20) #reduce mask dimension to match motion (from 40 to 20)
print("pooled_audio_mask",pooled_audio_mask.shape)
audio_context[:, self.audio_prefix_len:, :] = audio_noised_motion #replace noised part of audio with masked noised audio
#audio_context = audio_context.permute(0, 2, 1)  # [batch_size, latent_dim, wav2vec_seq_len] -> necessary to compute avg_pool -> use windowed attention pooling instead, so it is no more necessary
#print("audio_context ", audio_context.shape)
#audio_context_pooled = self.low_level_context_pooler(audio_context)  # [batch_size, latent_dim, seq_len]
audio_context_pooled = self.low_level_context_attention_pooler(audio_context) #[batch_size, seq_len, latent_dim]
print("audio_context_pooled ", audio_context_pooled.shape)
#audio_context_pooled = audio_context_pooled.permute(0, 2, 1)  # [batch_size, seq_len, latent_dim] -> not necessary if attention pooler is used
#audio_context = audio_context.permute(0, 2, 1)  # [batch_size, wav2vec_seq_len, latent_dim] -> not necessary if attention pooler is used
#print("audio_context_pooled_2", audio_context_pooled.shape)
overall_audio_context = self.low_level_context_collapsed(audio_context).unsqueeze(1) # [batch_size, 1, latent_dim]
print("overall_audio_context ", overall_audio_context.shape)
'''
