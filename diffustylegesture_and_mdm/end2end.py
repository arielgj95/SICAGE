import json
import logging
import os
import pickle
import sys

import torch
import yaml
from easydict import EasyDict

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
for path in [REPO_ROOT, THIS_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from dataset import prepare_data
except Exception:
    from data.dataset import prepare_data

from mydiffusion_beat_twh.configs.parse_args import parse_args

try:
    from diffustylegesture_and_mdm.utils.model_util import create_gaussian_diffusion
    from diffustylegesture_and_mdm.train.training_loop import TrainLoop
    from diffustylegesture_and_mdm.model.mdm import MDM
except Exception:
    from utils.model_util import create_gaussian_diffusion
    from train.training_loop import TrainLoop
    from model.mdm import MDM

logging.getLogger().setLevel(logging.INFO)


def _to_abs(path_value):
    if not path_value:
        return path_value
    path_value = os.path.expanduser(str(path_value))
    if not os.path.isabs(path_value):
        path_value = os.path.join(REPO_ROOT, path_value)
    return os.path.abspath(path_value)


def _load_yaml(path_value):
    resolved = _to_abs(path_value)
    if not resolved or not os.path.isfile(resolved):
        raise FileNotFoundError(f"YAML config not found: {path_value}")
    with open(resolved, "r") as f:
        return EasyDict(yaml.safe_load(f))


def _merge_cli_overrides(config, cli_args):
    merged = dict(config)
    for key, value in vars(cli_args).items():
        if value is not None:
            merged[key] = value
    return EasyDict(merged)


def _select_cond_mode(config):
    if getattr(config, "cond_mode", None):
        return config.cond_mode
    model_name = str(getattr(config, "name", "")).strip()
    if model_name == "DiffuseStyleGesture++":
        return "cross_local_attention5_style1"
    if model_name == "DiffuseStyleGesture+":
        return "cross_local_attention4_style1"
    if model_name == "DiffuseStyleGesture":
        return "cross_local_attention3_style1"
    if model_name == "MDM":
        return "mdm"
    raise ValueError(
        "Unsupported model name. Use one of: MDM, DiffuseStyleGesture, DiffuseStyleGesture+, DiffuseStyleGesture++."
    )


def _resolve_training_paths(config):
    config.dataset_path = _to_abs(getattr(config, "dataset_path", ""))
    config.metadata_path = _to_abs(getattr(config, "metadata_path", ""))
    config.dataset_info_path = _to_abs(getattr(config, "dataset_info_path", ""))
    config.culture_config_path = _to_abs(
        getattr(config, "culture_config_path", "culture_encoder/config.yml")
    )
    config.vqvae_config_path = _to_abs(
        getattr(config, "vqvae_config_path", "vq_vae/configs/codebook.yml")
    )
    config.fishr_model_path = _to_abs(getattr(config, "fishr_model_path", None))
    config.adversarial_checkpoint_path = _to_abs(
        getattr(config, "adversarial_checkpoint_path", None)
    )
    config.adversarial_model_save_path = _to_abs(
        getattr(config, "adversarial_model_save_path", None)
    )

    config.lmdb_path = getattr(config, "lmdb_path", config.dataset_path)
    config.info_path = getattr(config, "info_path", config.dataset_info_path)
    config.motion_prefix_len = int(getattr(config, "motion_prefix_len", 5))
    config.audio_prefix_len = int(getattr(config, "audio_prefix_len", 10))
    config.seed = int(getattr(config, "seed", 10))
    config.device = str(getattr(config, "device", f"cuda:{config.gpu}"))
    config.use_ema = bool(getattr(config, "use_ema", False))
    config.heads = int(getattr(config, "heads", getattr(config, "num_heads", 8)))
    config.ffn_size = int(getattr(config, "ffn_size", getattr(config, "ff_size", 2048)))
    config.layers = int(getattr(config, "layers", 10))

    if not getattr(config, "splits_data_path", None):
        config.splits_data_path = os.path.join(
            config.dataset_info_path,
            "whole_dataset_splits_subject_independent.pkl",
        )
    config.splits_data_path = _to_abs(config.splits_data_path)

    return config


def create_model_and_diffusion(args, device_name, culture_config):
    if args.use_culture:
        if args.use_adversarial:
            style_dim = 512
        else:
            style_dim = int(getattr(culture_config, "embed_dim", 512))
    else:
        default_style_dim = getattr(args, "style_dim", 512)
        style_dim = int(default_style_dim if default_style_dim and default_style_dim > 0 else 512)

    model = MDM(
        modeltype="",
        njoints=args.njoints,
        nfeats=1,
        cond_mode=args.cond_mode,
        audio_feat=args.audio_feat,
        arch="trans_enc",
        latent_dim=args.latent_dim,
        n_seed=args.n_seed,
        cond_mask_prob=args.cond_mask_prob,
        device=device_name,
        style_dim=style_dim,
        source_audio_dim=args.audio_feature_dim,
        audio_feat_dim_latent=args.audio_feat_dim_latent,
        batch_size=args.batch_size,
        use_culture=args.use_culture,
        use_adversarial=args.use_adversarial,
        n_train_speakers=args.n_train_speakers,
        culture_embedder_config=culture_config,
        culture_config_path=args.culture_config_path,
        fishr_model_path=args.fishr_model_path,
        adversarial_checkpoint_path=args.adversarial_checkpoint_path,
        adversarial_model_save_path=args.adversarial_model_save_path,
        use_alignment_module=getattr(args, "use_alignment_module", False),
        use_culture_guidance_loss=getattr(args, "use_culture_guidance_loss", False),
        n_cultures=getattr(args, "n_cultures", 4),
    )
    diffusion = create_gaussian_diffusion(args)
    return model, diffusion


def main(args):
    print("Loading dataset into memory ...")
    motion_only = False
    device = f"cuda:{args.gpu}"
    with open(args.metadata_path, "rb") as f:
        metadata = pickle.load(f)
    sample_keys = metadata["sample_keys"]
    culture_speakers = metadata["culture_speakers"]

    train_loader, val_loader, test_loader, speaker_enc, culture_enc, n_train_speakers = prepare_data(
        sample_keys,
        culture_speakers,
        motion_only,
        args.batch_size,
        args.splits_data_path,
        args.dataset_path,
        args.dataset_info_path,
        args.dataset_info_path,
        args.sep_people,
        use_translated_text=getattr(args, "use_translated_text", False),
        use_language_features=getattr(args, "use_language_features", True),
        use_translated_text_eval=getattr(args, "use_translated_text_eval", None),
        use_language_features_eval=getattr(args, "use_language_features_eval", None),
    )

    args.n_train_speakers = n_train_speakers
    args.n_cultures = len(culture_enc)
    culture_config = _load_yaml(args.culture_config_path)

    model, diffusion = create_model_and_diffusion(args, device, culture_config)
    model.to(device)
    os.makedirs(args.save_dir, exist_ok=True)

    args_dump = dict(args)
    args_dump["n_train_speakers"] = n_train_speakers
    if args.cond_mode == "mdm":
        args_dump["model_family"] = "baseline_mdm"
    elif args.cond_mode == "cross_local_attention4_style1":
        args_dump["model_family"] = "baseline_diffustylegesture_plus"
    else:
        args_dump["model_family"] = "hierarchical"
    args_json_path = os.path.join(args.save_dir, "args.json")
    with open(args_json_path, "w") as f:
        json.dump(args_dump, f, indent=2)
    print(f"Saved args to {args_json_path}")

    TrainLoop(args, model, diffusion, device, data=train_loader).run_loop()


if __name__ == "__main__":
    cli_args = parse_args()
    with open(_to_abs(cli_args.config), "r") as f:
        config_yaml = yaml.safe_load(f)
    config = _merge_cli_overrides(config_yaml, cli_args)
    config.cond_mode = _select_cond_mode(config)
    config.no_cuda = [str(config.gpu)] if getattr(config, "no_cuda", None) is None else config.no_cuda

    default_save_dir = f"./{config.dataset}_mymodel4_512_{config.version}"
    if getattr(config, "suffix", ""):
        default_save_dir = f"{default_save_dir}_{config.suffix}"
    config.save_dir = _to_abs(getattr(config, "save_dir", None) or default_save_dir)
    config = _resolve_training_paths(config)

    print("Configuration", config.name)
    print("Model save path:", config.save_dir, "version:", config.version)
    main(config)
