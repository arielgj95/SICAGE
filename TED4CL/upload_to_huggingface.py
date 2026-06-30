#!/usr/bin/env python3
"""Upload the TED4C-L LMDB release folder to a Hugging Face dataset repo."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RECOMMENDED_MAX_FILE_BYTES = 200_000_000_000
HARD_MAX_FILE_BYTES = 500_000_000_000
LICENSE_NAME_PATTERN = re.compile(r"^[a-z0-9-.]+$")


def _format_bytes(size: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(size)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{size} B"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload TED4C-L data.mdb, lock.mdb, metadata, and dataset card to Hugging Face."
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Local folder containing data.mdb, lock.mdb, and metadata/.",
    )
    parser.add_argument(
        "--repo-id",
        default="ariel-95/TED4C-L",
        help="Hugging Face dataset repo id.",
    )
    parser.add_argument(
        "--dataset-card",
        default=str(REPO_ROOT / "huggingface_dataset_card" / "README.md"),
        help="Local HF dataset card to upload as README.md. Use empty string to skip.",
    )
    parser.add_argument(
        "--license-file",
        default=str(REPO_ROOT / "huggingface_dataset_card" / "LICENSE"),
        help="Local composite license notice to upload as LICENSE. Use empty string to skip.",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Make the repo public and ungated before upload, including an existing repo.",
    )
    parser.add_argument(
        "--gated",
        choices=("auto", "manual"),
        default=None,
        help=(
            "Enable access requests on a public repo. Use 'manual' to approve each user, "
            "or 'auto' to grant access after license acknowledgement."
        ),
    )
    parser.add_argument(
        "--commit-message",
        default="Upload TED4C-L dataset release",
        help="Commit message for dataset files.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional HF token. If omitted, uses the logged-in token or HF_TOKEN.",
    )
    parser.add_argument(
        "--token-file",
        default=None,
        help="Optional file containing the HF token. The path must be supplied explicitly.",
    )
    parser.add_argument(
        "--token-index",
        type=int,
        default=None,
        help=(
            "One-based non-empty line to use when --token-file contains multiple tokens. "
            "Omit this option when the file contains exactly one token."
        ),
    )
    parser.add_argument(
        "--ignore-pattern",
        action="append",
        default=[],
        help="Additional upload ignore pattern. Can be passed multiple times.",
    )
    parser.add_argument(
        "--no-high-performance",
        action="store_true",
        help="Do not set HF_XET_HIGH_PERFORMANCE=1 automatically.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate local files and print the planned upload without uploading.",
    )
    parser.add_argument(
        "--squash-history",
        action="store_true",
        help=(
            "After a successful upload, destructively squash the dataset repository history "
            "so superseded dataset objects are no longer available through old revisions."
        ),
    )
    return parser.parse_args()


def _validate_dataset_path(dataset_path: Path) -> None:
    required = [
        dataset_path / "data.mdb",
        dataset_path / "lock.mdb",
        dataset_path / "SANITIZATION.json",
        dataset_path / "metadata" / "metadata.pkl",
        dataset_path / "metadata" / "whole_dataset_splits_subject_independent.pkl",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required dataset files:\n" + "\n".join(missing))

    data_size = (dataset_path / "data.mdb").stat().st_size
    print(f"data.mdb: {_format_bytes(data_size)} ({data_size} bytes)")
    if data_size > HARD_MAX_FILE_BYTES:
        raise ValueError("data.mdb exceeds Hugging Face's 500 GB hard single-file limit.")
    if data_size > RECOMMENDED_MAX_FILE_BYTES:
        print(
            "Warning: data.mdb is above Hugging Face's 200 GB recommended single-file size. "
            "It is below the 500 GB hard limit, so upload is allowed, but download retries may be slower."
        )

    manifest_path = dataset_path / "SANITIZATION.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("readable_subtitles_or_transcripts_in_lmdb") is not False:
        raise ValueError(
            "SANITIZATION.json does not certify that readable subtitles/transcripts are absent."
        )
    removed_fields = set(
        manifest.get("removed_source_fields", manifest.get("removed_fields", []))
    )
    if not {"text_data", "translated_text"}.issubset(removed_fields):
        raise ValueError(
            "SANITIZATION.json does not list both text_data and translated_text as removed fields."
        )


def _validate_dataset_card(card_path: Path) -> None:
    text = card_path.read_text(encoding="utf-8")
    if not text.startswith("---\n") or "\n---\n" not in text[4:]:
        raise ValueError(f"Dataset card has no valid YAML front matter: {card_path}")

    front_matter = text[4:].split("\n---\n", 1)[0]
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("Missing dependency: install with `python -m pip install pyyaml`.") from exc

    metadata = yaml.safe_load(front_matter) or {}
    license_name = metadata.get("license_name")
    if license_name and not LICENSE_NAME_PATTERN.fullmatch(str(license_name)):
        raise ValueError(
            "Hugging Face license_name must match ^[a-z0-9-.]+$: "
            f"{license_name!r}"
        )


def main() -> int:
    args = _parse_args()
    if args.gated and not args.public:
        raise ValueError("--gated requires --public because a private repo is already access-restricted.")

    dataset_path = Path(args.dataset_path).expanduser().resolve()

    if not dataset_path.is_dir():
        raise NotADirectoryError(f"Dataset path does not exist: {dataset_path}")
    _validate_dataset_path(dataset_path)

    files = sorted(path for path in dataset_path.rglob("*") if path.is_file())
    total_size = sum(path.stat().st_size for path in files)
    print(f"Planned upload: {len(files)} files, {_format_bytes(total_size)}")
    for path in files:
        rel = path.relative_to(dataset_path)
        print(f"  {rel} ({_format_bytes(path.stat().st_size)})")

    card_path = Path(args.dataset_card).expanduser() if args.dataset_card else None
    if card_path is not None:
        card_path = card_path.resolve()
        if not card_path.exists():
            raise FileNotFoundError(f"Dataset card does not exist: {card_path}")
        _validate_dataset_card(card_path)
        print(f"Dataset card: {card_path} -> README.md")

    license_path = Path(args.license_file).expanduser() if args.license_file else None
    if license_path is not None:
        license_path = license_path.resolve()
        if not license_path.exists():
            raise FileNotFoundError(f"License notice does not exist: {license_path}")
        print(f"License notice: {license_path} -> LICENSE")

    if args.dry_run:
        return 0

    if not args.no_high_performance:
        os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: install with `python -m pip install -U "
            "\"huggingface_hub[hf_xet]>=0.36.2,<1.0\"`."
        ) from exc

    token = args.token
    if args.token_file:
        token_path = Path(args.token_file).expanduser().resolve()
        tokens = [
            line.strip()
            for line in token_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not tokens:
            raise ValueError(f"HF token file is empty: {token_path}")
        if len(tokens) == 1:
            if args.token_index not in (None, 1):
                raise ValueError("--token-index must be 1 for a single-token file.")
            token = tokens[0]
        else:
            if args.token_index is None:
                raise ValueError(
                    f"HF token file contains {len(tokens)} non-empty lines; pass --token-index "
                    "to select one without exposing it on the command line."
                )
            if not 1 <= args.token_index <= len(tokens):
                raise ValueError(
                    f"--token-index must be between 1 and {len(tokens)} for this token file."
                )
            token = tokens[args.token_index - 1]
    elif args.token_index is not None:
        raise ValueError("--token-index requires --token-file.")

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=not args.public,
        exist_ok=True,
    )
    if args.public:
        api.update_repo_settings(
            repo_id=args.repo_id,
            repo_type="dataset",
            private=False,
            gated=args.gated if args.gated else False,
        )
        access = f"gated ({args.gated} approval)" if args.gated else "ungated"
        print(f"Repository is public and {access}.")

    ignore_patterns = [".cache/huggingface/**", "__pycache__/**", ".DS_Store"]
    ignore_patterns.extend(args.ignore_pattern)
    try:
        api.upload_folder(
            folder_path=str(dataset_path),
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message=args.commit_message,
            ignore_patterns=ignore_patterns,
        )
    except Exception as exc:
        if "Private repository storage limit reached" in str(exc):
            raise RuntimeError(
                "Hugging Face rejected the upload because the repository is private. "
                "Rerun this command with --public to make the repository public and ungated."
            ) from exc
        raise

    # Publish claims about the sanitized release only after its data commit succeeds.
    if card_path is not None:
        api.upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message="Document transcript-free TED4C-L release",
        )

    if license_path is not None:
        api.upload_file(
            path_or_fileobj=str(license_path),
            path_in_repo="LICENSE",
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message="Update TED4C-L composite rights notice",
        )

    if args.squash_history:
        api.super_squash_history(
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message="Publish transcript-free TED4C-L release",
        )
        print("Squashed repository history after the sanitized upload.")

    print(f"Uploaded to https://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
