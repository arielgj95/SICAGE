import argparse
import os
import pickle

import torch
import yaml
from easydict import EasyDict

from dataset import prepare_data
from culture_encoder.adversarial_classifier import (
    Culture_Classifier,
    evaluate_set as evaluate_adversarial,
    train as train_adversarial,
)
from culture_encoder.fishr_classifier import (
    Fishr,
    create_domain_loaders,
    evaluate_set as evaluate_fishr,
    train_fishr_model,
)


def _load_yaml_config(config_path: str) -> EasyDict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    return EasyDict(cfg)


def _load_metadata(metadata_path: str):
    metadata_file = metadata_path
    if os.path.isdir(metadata_path):
        metadata_file = os.path.join(metadata_path, "metadata.pkl")
    with open(metadata_file, "rb") as f:
        return pickle.load(f)


def _resolve_splits_path(dataset_info_path: str, motion_only: bool, subject_independent: bool) -> str:
    if motion_only:
        filename = "gesture_dataset_splits_subject_independent.pkl" if subject_independent \
            else "gesture_dataset_splits_subject_dependent.pkl"
    else:
        filename = "whole_dataset_splits_subject_independent.pkl" if subject_independent \
            else "whole_dataset_splits_subject_dependent.pkl"
    return os.path.join(dataset_info_path, filename)


def _resolve_device(device_str: str) -> torch.device:
    if device_str:
        requested = torch.device(device_str)
        if requested.type == "cuda" and not torch.cuda.is_available():
            print("[Warn] CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")
        return requested
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _normalize_encoder_type(value):
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip()
        if normalized.lower() in {"", "none", "null"}:
            return None
        return normalized
    return value


def _metrics_to_printable(metrics: dict) -> dict:
    printable = {}
    for key, value in metrics.items():
        if torch.is_tensor(value):
            printable[key] = float(value.detach().cpu().item())
        else:
            printable[key] = float(value) if isinstance(value, (int, float)) else value
    return printable


def _selected_eval_loaders(split: str, val_loader, test_loader):
    if split == "val":
        return [("val", val_loader)]
    if split == "both":
        return [("val", val_loader), ("test", test_loader)]
    return [("test", test_loader)]


def _prepare_loaders(args):
    metadata = _load_metadata(args.metadata_path)
    sample_keys = metadata["sample_keys"]
    culture_speakers = metadata["culture_speakers"]

    sep_people_for_split = "_sep_people" if args.subject_independent else ""
    splits_path = args.splits_path or _resolve_splits_path(
        args.dataset_info_path,
        motion_only=args.motion_only,
        subject_independent=args.subject_independent,
    )

    return prepare_data(
        sample_keys=sample_keys,
        culture_speakers=culture_speakers,
        motion_only=args.motion_only,
        batch_size=args.batch_size,
        splits_path=splits_path,
        lmdb_path=args.dataset_path,
        encodings_path=args.dataset_info_path,
        normalization_path=args.dataset_info_path,
        sep_people=sep_people_for_split,
        use_translated_text=args.use_translated_text_train,
        use_language_features=args.use_language_features_train,
        use_translated_text_eval=args.use_translated_text_eval,
        use_language_features_eval=args.use_language_features_eval,
    )


def run_adversarial(args):
    config = _load_yaml_config(args.config)
    device = _resolve_device(args.device)

    train_loader, val_loader, _, _, culture_enc, n_train_speakers = _prepare_loaders(args)
    n_languages = len(getattr(train_loader.dataset, "language_encodings", {}))

    # Override/complete runtime config.
    config.epochs = int(args.epochs)
    config.batch_size = int(args.batch_size)
    config.model_save_path = args.model_save_path
    config.data_path_info = args.dataset_info_path
    config.supcon_weight = float(args.supcon_weight)
    config.supcon_temperature = float(args.supcon_temperature)
    config.supcon_require_diff_domain = bool(args.supcon_require_diff_domain)
    config.contrastive_weight = float(args.contrastive_weight)
    config.mixup_weight = float(args.mixup_weight)
    config.mixup_alpha = float(args.mixup_alpha)
    config.language_penalization = bool(args.language_penalization)
    config.language_loss_weight = float(args.language_loss_weight)
    config.language_schedule_gamma = float(args.language_schedule_gamma)

    if not hasattr(config, "dropout"):
        config.dropout = 0.1
    if not hasattr(config, "save_per_epochs"):
        config.save_per_epochs = 1
    if not hasattr(config, "levels"):
        config.levels = 1
    if not hasattr(config, "layer_neurons"):
        config.layer_neurons = 512
    if not hasattr(config, "lr"):
        config.lr = 1e-4
    if not hasattr(config, "betas"):
        config.betas = [0.9, 0.999]

    d_model = int(args.d_model) if args.d_model is not None else int(config.layer_neurons)
    pose_enc_type = _normalize_encoder_type(
        args.pose_enc_type if args.pose_enc_type is not None else getattr(config, "pose_enc_type", "linear")
    )
    audio_enc_type = _normalize_encoder_type(
        args.audio_enc_type if args.audio_enc_type is not None else getattr(config, "audio_enc_type", None)
    )

    train_adversarial(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        cl_type=args.cl_type,
        culture_weights=None,
        speaker_weights=None,
        n_cultures=len(culture_enc),
        n_speakers=n_train_speakers,
        n_languages=n_languages,
        d_model=d_model,
        sep_people=args.sep_people,
        raw_poses=args.motion_only,
        pose_enc_type=pose_enc_type,
        audio_enc_type=audio_enc_type,
    )


def run_fishr(args):
    config = _load_yaml_config(args.config)
    device = _resolve_device(args.device)
    train_loader, val_loader, _, _, culture_enc, n_train_speakers = _prepare_loaders(args)

    if not hasattr(config, "dropout"):
        config.dropout = 0.1
    if not hasattr(config, "levels"):
        config.levels = 1
    if not hasattr(config, "layer_neurons"):
        config.layer_neurons = 512
    if not hasattr(config, "lr"):
        config.lr = 1e-4
    if not hasattr(config, "betas"):
        config.betas = [0.9, 0.999]
    if args.dropout is not None:
        config.dropout = float(args.dropout)

    d_model = int(args.d_model) if args.d_model is not None else int(config.layer_neurons)
    pose_enc_type = _normalize_encoder_type(
        args.pose_enc_type if args.pose_enc_type is not None else getattr(config, "pose_enc_type", "linear")
    )
    audio_enc_type = _normalize_encoder_type(
        args.audio_enc_type if args.audio_enc_type is not None else getattr(config, "audio_enc_type", None)
    )

    effective_domains = min(int(args.k_domain), max(1, int(n_train_speakers)))
    effective_batch = int(args.batch_size) * effective_domains
    if effective_batch > 2048:
        print(
            f"[Warn] Fishr effective batch is batch_size*k_domain={args.batch_size}*{effective_domains}={effective_batch}. "
            f"This is often too large for a 24 GB GPU; consider lowering --batch-size and/or --k-domain."
        )

    train_dataset = train_loader.dataset
    domain_loaders = create_domain_loaders(
        train_dataset,
        batch_size=args.batch_size,
        save_path=args.dataset_info_path,
        min_samples=args.min_domain_samples,
    )

    n_train_languages = len(getattr(train_dataset, "language_encodings", {}))
    optimizer_weight_decay = (
        float(args.optimizer_weight_decay)
        if args.optimizer_weight_decay is not None
        else float(getattr(config, "weight_decay", 1e-4))
    )

    model = Fishr(
        proj_dim=args.proj_dim,
        num_classes=len(culture_enc),
        num_domains=min(args.k_domain, max(1, n_train_speakers)),
        is_nonlinear=args.is_nonlinear,
        use_motion=args.use_motion,
        use_language_adversary=args.language_penalization,
        n_languages=n_train_languages,
        lang_grl_lambda=args.lang_grl_lambda,
        lang_loss_weight=args.language_loss_weight,
        supcon_weight=args.supcon_weight,
        supcon_temperature=args.supcon_temperature,
        supcon_require_diff_domain=args.supcon_require_diff_domain,
        contrastive_weight=args.contrastive_weight,
        mixup_weight=args.mixup_weight,
        mixup_alpha=args.mixup_alpha,
        fishr_penalty_weight=args.fishr_penalty_weight,
        fishr_warmup_steps=args.fishr_warmup_steps,
        fishr_penalty_ramp_steps=args.fishr_penalty_ramp_steps,
        grad_clip_norm=args.grad_clip_norm,
        cl_type=args.cl_type,
        d_model=d_model,
        pose_enc_type=pose_enc_type,
        audio_enc_type=audio_enc_type,
        raw_poses=args.motion_only,
        backbone_config=config,
        use_adversarial_backbone=not args.legacy_multimodal_featurizer,
        optimizer_lr=float(config.lr),
        optimizer_betas=tuple(config.betas),
        optimizer_weight_decay=optimizer_weight_decay,
    ).to(device)
    model.train()

    train_fishr_model(
        model=model,
        domain_loaders=domain_loaders,
        num_epochs=args.epochs,
        val_loader=val_loader,
        save_dir=args.save_dir,
        k_domain=args.k_domain,
    )


def run_adversarial_test(args):
    config = _load_yaml_config(args.config)
    device = _resolve_device(args.device)

    train_loader, val_loader, test_loader, _, culture_enc, n_train_speakers = _prepare_loaders(args)
    n_languages = len(getattr(train_loader.dataset, "language_encodings", {}))

    if not hasattr(config, "dropout"):
        config.dropout = 0.1
    if not hasattr(config, "levels"):
        config.levels = 1
    if not hasattr(config, "layer_neurons"):
        config.layer_neurons = 512

    d_model = int(args.d_model) if args.d_model is not None else int(config.layer_neurons)
    pose_enc_type = _normalize_encoder_type(
        args.pose_enc_type if args.pose_enc_type is not None else getattr(config, "pose_enc_type", "linear")
    )
    audio_enc_type = _normalize_encoder_type(
        args.audio_enc_type if args.audio_enc_type is not None else getattr(config, "audio_enc_type", None)
    )

    sp = True if ("adv_train" in args.sep_people or "multitask" in args.sep_people) else False
    lp = bool(getattr(config, "language_penalization", False) or ("lang_adv" in args.sep_people))

    model = Culture_Classifier(
        config,
        device,
        dropout_prob=config.dropout,
        cl_type=args.cl_type,
        d_model=d_model,
        speaker_penalization=sp,
        n_cultures=len(culture_enc),
        n_speakers=n_train_speakers,
        language_penalization=lp,
        n_languages=n_languages,
        pose_enc=pose_enc_type,
        audio_enc=audio_enc_type,
        raw_poses=args.motion_only,
    ).to(device)

    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        if not args.model_save_path:
            raise ValueError("--model-save-path is required when --checkpoint-path is not provided.")
        checkpoint_path = os.path.join(
            args.model_save_path,
            args.cl_type,
            "no_weighted_loss",
            f"train_{args.cl_type}",
            f"{args.cl_type}_checkpoint_best{args.sep_people}.bin",
        )

    print(f"Loading adversarial checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    if args.allow_partial_load:
        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            print(f"[Warn] Missing keys: {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            print(f"[Warn] Unexpected keys: {incompatible.unexpected_keys}")
    else:
        model.load_state_dict(state_dict, strict=True)

    for split_name, loader in _selected_eval_loaders(args.split, val_loader, test_loader):
        metrics, cm = evaluate_adversarial(
            model,
            loader,
            device,
            n_levels=len(culture_enc),
            type="test",
            speaker_penalization=sp,
            language_penalization=lp,
            subj_independent_data=args.subject_independent,
        )
        print(f"[Adversarial][{split_name}] metrics: {_metrics_to_printable(metrics)}")
        if cm is not None:
            print(f"[Adversarial][{split_name}] confusion matrix:\n{cm}")


def run_fishr_test(args):
    config = _load_yaml_config(args.config)
    device = _resolve_device(args.device)
    train_loader, val_loader, test_loader, _, culture_enc, n_train_speakers = _prepare_loaders(args)

    if not hasattr(config, "dropout"):
        config.dropout = 0.1
    if not hasattr(config, "levels"):
        config.levels = 1
    if not hasattr(config, "layer_neurons"):
        config.layer_neurons = 512
    if not hasattr(config, "lr"):
        config.lr = 1e-4
    if not hasattr(config, "betas"):
        config.betas = [0.9, 0.999]

    d_model = int(args.d_model) if args.d_model is not None else int(config.layer_neurons)
    pose_enc_type = _normalize_encoder_type(
        args.pose_enc_type if args.pose_enc_type is not None else getattr(config, "pose_enc_type", "linear")
    )
    audio_enc_type = _normalize_encoder_type(
        args.audio_enc_type if args.audio_enc_type is not None else getattr(config, "audio_enc_type", None)
    )
    n_train_languages = len(getattr(train_loader.dataset, "language_encodings", {}))

    model = Fishr(
        proj_dim=args.proj_dim,
        num_classes=len(culture_enc),
        num_domains=min(args.k_domain, max(1, n_train_speakers)),
        is_nonlinear=args.is_nonlinear,
        use_motion=args.use_motion,
        use_language_adversary=args.language_penalization,
        n_languages=n_train_languages,
        lang_grl_lambda=args.lang_grl_lambda,
        lang_loss_weight=args.language_loss_weight,
        supcon_weight=args.supcon_weight,
        supcon_temperature=args.supcon_temperature,
        supcon_require_diff_domain=args.supcon_require_diff_domain,
        contrastive_weight=args.contrastive_weight,
        mixup_weight=args.mixup_weight,
        mixup_alpha=args.mixup_alpha,
        cl_type=args.cl_type,
        d_model=d_model,
        pose_enc_type=pose_enc_type,
        audio_enc_type=audio_enc_type,
        raw_poses=args.motion_only,
        backbone_config=config,
        use_adversarial_backbone=not args.legacy_multimodal_featurizer,
        optimizer_lr=float(config.lr),
        optimizer_betas=tuple(config.betas),
        optimizer_weight_decay=float(getattr(config, "weight_decay", 1e-4)),
    ).to(device)

    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        if not args.save_dir:
            raise ValueError("--save-dir is required when --checkpoint-path is not provided.")
        checkpoint_path = os.path.join(args.save_dir, "model_best.pt")

    print(f"Loading Fishr checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    if isinstance(state_dict, dict) and "model_dict" in state_dict:
        state_dict = state_dict["model_dict"]

    if args.allow_partial_load:
        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            print(f"[Warn] Missing keys: {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            print(f"[Warn] Unexpected keys: {incompatible.unexpected_keys}")
    else:
        model.load_state_dict(state_dict, strict=True)

    for split_name, loader in _selected_eval_loaders(args.split, val_loader, test_loader):
        metrics, cm = evaluate_fishr(
            model,
            loader,
            device,
            n_levels=len(culture_enc),
            type="test",
        )
        print(f"[Fishr][{split_name}] metrics: {_metrics_to_printable(metrics)}")
        if cm is not None:
            print(f"[Fishr][{split_name}] confusion matrix:\n{cm}")


def add_shared_args(sp):
    sp.add_argument("--config", type=str, default="culture_encoder/config.yml")
    sp.add_argument("--metadata-path", type=str, required=True)
    sp.add_argument("--dataset-path", type=str, required=True)
    sp.add_argument("--dataset-info-path", type=str, required=True)
    sp.add_argument("--splits-path", type=str, default=None)
    sp.add_argument("--batch-size", type=int, default=64)
    sp.add_argument("--epochs", type=int, default=30)
    sp.add_argument("--device", type=str, default="")
    sp.add_argument("--motion-only", action="store_true")
    sp.add_argument("--subject-independent", dest="subject_independent", action="store_true", default=True)
    sp.add_argument("--subject-dependent", dest="subject_independent", action="store_false")
    sp.add_argument("--use-translated-text-train", action="store_true")
    sp.add_argument("--use-translated-text-eval", action="store_true")
    sp.add_argument("--use-language-features-train", dest="use_language_features_train", action="store_true", default=True)
    sp.add_argument("--no-language-features-train", dest="use_language_features_train", action="store_false")
    sp.add_argument("--use-language-features-eval", dest="use_language_features_eval", action="store_true", default=True)
    sp.add_argument("--no-language-features-eval", dest="use_language_features_eval", action="store_false")
    sp.add_argument("--language-penalization", action="store_true")
    sp.add_argument("--language-loss-weight", type=float, default=1.0)
    sp.add_argument("--supcon-weight", type=float, default=0.0)
    sp.add_argument("--supcon-temperature", type=float, default=0.07)
    sp.add_argument("--supcon-require-diff-domain", dest="supcon_require_diff_domain", action="store_true", default=True)
    sp.add_argument("--supcon-allow-same-domain", dest="supcon_require_diff_domain", action="store_false")
    sp.add_argument("--contrastive-weight", type=float, default=0.0)
    sp.add_argument("--mixup-weight", type=float, default=0.0)
    sp.add_argument("--mixup-alpha", type=float, default=0.0)


def build_parser():
    parser = argparse.ArgumentParser(description="Train culture classifiers (adversarial or Fishr).")
    sub = parser.add_subparsers(dest="algorithm", required=True)

    p_adv = sub.add_parser("adversarial", help="Train adversarial culture classifier.")
    add_shared_args(p_adv)
    p_adv.add_argument("--model-save-path", type=str, required=True)
    p_adv.add_argument("--cl-type", type=str, default="culclH")
    p_adv.add_argument("--sep-people", type=str, default="_sep_people_adv_train_lang_adv")
    p_adv.add_argument("--d-model", type=int, default=None)
    p_adv.add_argument("--pose-enc-type", type=str, default=None)
    p_adv.add_argument("--audio-enc-type", type=str, default=None)
    p_adv.add_argument("--language-schedule-gamma", type=float, default=5.0)

    p_fishr = sub.add_parser("fishr", help="Train Fishr culture classifier.")
    add_shared_args(p_fishr)
    p_fishr.add_argument("--save-dir", type=str, required=True)
    p_fishr.add_argument("--proj-dim", type=int, default=512)
    p_fishr.add_argument("--cl-type", type=str, default="culclI")
    p_fishr.add_argument("--d-model", type=int, default=None)
    p_fishr.add_argument("--pose-enc-type", type=str, default=None)
    p_fishr.add_argument("--audio-enc-type", type=str, default=None)
    p_fishr.add_argument("--k-domain", type=int, default=64)
    p_fishr.add_argument("--min-domain-samples", type=int, default=30)
    p_fishr.add_argument("--is-nonlinear", action="store_true")
    p_fishr.add_argument("--legacy-multimodal-featurizer", action="store_true")
    p_fishr.add_argument("--use-motion", action="store_true")
    p_fishr.add_argument("--lang-grl-lambda", type=float, default=1.0)
    p_fishr.add_argument("--fishr-penalty-weight", type=float, default=1000.0)
    p_fishr.add_argument("--fishr-warmup-steps", type=int, default=500)
    p_fishr.add_argument("--fishr-penalty-ramp-steps", type=int, default=100)
    p_fishr.add_argument("--grad-clip-norm", type=float, default=1.0)
    p_fishr.add_argument("--dropout", type=float, default=None)
    p_fishr.add_argument("--optimizer-weight-decay", type=float, default=None)

    p_adv_test = sub.add_parser("test-adversarial", help="Evaluate adversarial classifier from checkpoint.")
    add_shared_args(p_adv_test)
    p_adv_test.add_argument("--model-save-path", type=str, default=None)
    p_adv_test.add_argument("--checkpoint-path", type=str, default=None)
    p_adv_test.add_argument("--cl-type", type=str, default="culclI")
    p_adv_test.add_argument("--sep-people", type=str, default="_sep_people_adv_train")
    p_adv_test.add_argument("--d-model", type=int, default=None)
    p_adv_test.add_argument("--pose-enc-type", type=str, default=None)
    p_adv_test.add_argument("--audio-enc-type", type=str, default=None)
    p_adv_test.add_argument("--split", type=str, choices=["val", "test", "both"], default="test")
    p_adv_test.add_argument("--allow-partial-load", action="store_true")

    p_fishr_test = sub.add_parser("test-fishr", help="Evaluate Fishr classifier from checkpoint.")
    add_shared_args(p_fishr_test)
    p_fishr_test.add_argument("--save-dir", type=str, default=None)
    p_fishr_test.add_argument("--checkpoint-path", type=str, default=None)
    p_fishr_test.add_argument("--proj-dim", type=int, default=512)
    p_fishr_test.add_argument("--cl-type", type=str, default="culclI")
    p_fishr_test.add_argument("--d-model", type=int, default=None)
    p_fishr_test.add_argument("--pose-enc-type", type=str, default=None)
    p_fishr_test.add_argument("--audio-enc-type", type=str, default=None)
    p_fishr_test.add_argument("--k-domain", type=int, default=64)
    p_fishr_test.add_argument("--is-nonlinear", action="store_true")
    p_fishr_test.add_argument("--legacy-multimodal-featurizer", action="store_true")
    p_fishr_test.add_argument("--use-motion", action="store_true")
    p_fishr_test.add_argument("--lang-grl-lambda", type=float, default=1.0)
    p_fishr_test.add_argument("--split", type=str, choices=["val", "test", "both"], default="test")
    p_fishr_test.add_argument("--allow-partial-load", action="store_true")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.algorithm == "adversarial":
        run_adversarial(args)
    elif args.algorithm == "fishr":
        run_fishr(args)
    elif args.algorithm == "test-adversarial":
        run_adversarial_test(args)
    elif args.algorithm == "test-fishr":
        run_fishr_test(args)


if __name__ == "__main__":
    main()
