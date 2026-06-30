# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""
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

from easydict import EasyDict

from mdm_generator.diffusion.utils.fixseed import fixseed
from mdm_generator.diffusion.utils.parser_util import train_args
from mdm_generator.diffusion.utils import dist_util
from mdm_generator.hierarchical_mdm import Hierarchical_MDM
from dataset import prepare_data
from mdm_generator.training_loop import TrainLoop
from mdm_generator.diffusion.utils.model_util import create_gaussian_diffusion
from mdm_generator.diffusion.train.train_platforms import (
    ClearmlPlatform,
    TensorboardPlatform,
    NoPlatform,
    WandBPlatform,
)  # required for the eval operation


def main():
    args = train_args()
    model_device = f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu"
    with open(args.culture_config_path) as f:
        culture_config = yaml.safe_load(f)
    culture_config = EasyDict(culture_config)
    if getattr(args, "fishr_model_path", ""):
        culture_config.fishr_model_path = args.fishr_model_path
    if getattr(args, "adversarial_checkpoint_path", ""):
        culture_config.adversarial_checkpoint_path = args.adversarial_checkpoint_path
    if getattr(args, "adversarial_model_save_path", ""):
        culture_config.adversarial_model_save_path = args.adversarial_model_save_path
    dataset_path = args.lmdb_path
    metadata_path = args.metadata_path
    dataset_info_path = encodings_path = normalization_path = args.info_path
    batch_size = args.batch_size
    fixseed(args.seed)
    #TensorboardPlatform creates a tensorboard summary writer an every time report_scalar is called, the summary writes the info
    train_platform_type = eval(args.train_platform_type)  #TensorboardPlatform by default. It converts the name in a class
    train_platform = train_platform_type(args.save_dir) # this need to be inputted
    train_platform.report_args(args, name='Args') # This does not have any effect on Tensorboard
    sep_people = "_sep_people"
    motion_only = False

    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
        sample_keys = metadata['sample_keys']
        culture_speakers = metadata['culture_speakers']

    splits_data_path = os.path.join(dataset_info_path, "whole_dataset_splits_subject_independent.pkl")

    train_loader, val_loader, test_loader, speaker_enc, culture_enc, n_train_speakers = prepare_data(sample_keys,culture_speakers,motion_only,
                                                                                   batch_size,splits_data_path,dataset_path,
                                                                                   encodings_path,normalization_path,
                                                                                   sep_people)
    hierarchical_mdm_model = Hierarchical_MDM(
                 vqvae_dim=(25,512),
                 audio_onset_dim=156,
                 audio_mel_dim=(156,64),
                 audio_wav2vec_dim=(50,1024),
                 text_embedding_dim=768,
                 culture_embedding_dim=512,
                 latent_dim=args.latent_dim,
                 n_train_speakers = n_train_speakers,
                 n_cultures=len(culture_enc),
                 num_heads=args.heads,
                 num_layers=args.layers,
                 ff_size=args.ffn_size,
                 dropout=0.1,
                 activation='gelu',
                 culture_embedder_config = culture_config,
                 device = model_device,
                 dataset_type = "_sep_people",
                 motion_prefix_len=args.motion_prefix_len,
                 audio_prefix_len=args.audio_prefix_len,
                 motion_mask_prob=args.motion_mask_prob,
                 audio_mask_prob=args.audio_mask_prob,
                 use_culture = args.use_culture,
                 use_adversarial = args.use_adversarial,
                 use_one_hot_culture=getattr(args, "use_one_hot_culture", False),
                 culture_encoder_type=getattr(args, "culture_encoder_type", ""),
                 use_adain=getattr(args, "use_adain", True),
                 use_alignment=getattr(args, "use_alignment", True),
                 use_culture_classification_layer=getattr(args, "use_culture_classification_layer", True),
                 use_native_attention=getattr(args, "use_native_attention", False),
    )

    diffusion_model = create_gaussian_diffusion(args) #seems ok
    #TODO give it as input. At the moment default is
    #TODO note that overwrite is True as default. It can create problems when loading data
    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')

    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args_path = os.path.join(args.save_dir, 'args.json') #save args
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    cuda_device = dist_util.setup_dist(args.device) #sets the device

    print("creating data loader...")

    # If HumanML3D dataset is used, hml_type specify type of representation where None refers to the original representation n_frames x 263,
    # global refers to all global positions n_frames x 3,
    # and global_root refers to original altered to have global root position/orientation. Use None for other datasets.
    hierarchical_mdm_model.to(dist_util.dev())
    trainable_params, non_trainable_params, total_params = hierarchical_mdm_model.count_parameters()
    print('Trainable params: %.2fM' % (trainable_params / 1000000.0))
    print('Total params: %.2fM' % (total_params / 1000000.0))

    #print('Total params: %.2fM' % (sum(p.numel() for p in hierarchical_mdm_model.parameters()) / 1000000.0))
    print("Training...")

    TrainLoop(args, train_platform, hierarchical_mdm_model, diffusion_model, train_loader, val_loader).run_loop()

    train_platform.close()

if __name__ == "__main__":
    main()
