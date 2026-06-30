import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from dataset import dataset_analysis


def _format_culture_name(culture):
    return str(culture).strip().capitalize()


def _load_metadata(metadata_path):
    with open(metadata_path, "rb") as f:
        return pickle.load(f)


def _load_cultures_from_metadata(metadata):
    cultures = metadata.get("culture_speakers", {}).keys()
    normalized = set()
    for culture in cultures:
        culture_name = str(culture).strip().lower()
        normalized.add(culture_name)
        normalized.add(culture_name.split("_")[0])
    return sorted(normalized)


def _matching_cultures_in_root(root, cultures):
    root = Path(root)
    if not root.is_dir():
        return []
    found = set()
    for child in root.iterdir():
        if not child.is_dir():
            continue
        culture_name = child.name.strip().lower()
        culture_short = culture_name.split("_")[0]
        if culture_name in cultures or culture_short in cultures:
            found.add(culture_short if culture_short in cultures else culture_name)
    return sorted(found)


def _build_candidate_roots(requested_root, metadata_path):
    requested = Path(requested_root).resolve()
    metadata = Path(metadata_path).resolve()

    candidates = [
        requested,
        requested.parent,
        requested.parent.parent,
        metadata.parent,                # .../metadata
        metadata.parent.parent,         # .../full_dataset
        metadata.parent.parent.parent,  # .../dataset_root
    ]

    # Include one-level children of common roots (helps with nested layouts).
    for base in [requested, requested.parent, metadata.parent.parent.parent]:
        if base.is_dir():
            for child in base.iterdir():
                if child.is_dir():
                    candidates.append(child.resolve())

    deduped = []
    seen = set()
    for cand in candidates:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cand)
    return deduped


def _resolve_playlist_root(requested_root, cultures, metadata_path):
    candidates = _build_candidate_roots(requested_root, metadata_path)
    best_root = Path(requested_root).resolve()
    best_matches = []
    best_count = -1
    debug_rows = []

    for idx, candidate in enumerate(candidates):
        matches = _matching_cultures_in_root(candidate, cultures)
        debug_rows.append((idx, str(candidate), matches))
        if len(matches) > best_count:
            best_count = len(matches)
            best_root = candidate
            best_matches = matches

    return best_root, best_matches, debug_rows


def _build_summary_stats(
    culture_video_lengths,
    culture_video_fps,
    culture_video_scenes,
    culture_video_skeletons,
    culture_scene_lengths,
):
    summary = {}
    all_cultures = sorted(culture_video_lengths.keys())
    for culture in all_cultures:
        lengths = np.asarray(culture_video_lengths.get(culture, []), dtype=float)
        fps = np.asarray(culture_video_fps.get(culture, []), dtype=float)
        scenes = np.asarray(culture_video_scenes.get(culture, []), dtype=float)
        skeletons = np.asarray(culture_video_skeletons.get(culture, []), dtype=float)
        scene_lengths = np.asarray(culture_scene_lengths.get(culture, []), dtype=float)

        video_count = int(len(lengths))
        if video_count == 0:
            continue

        tot_length = float(np.sum(lengths))
        display_name = _format_culture_name(culture)
        summary[display_name] = {
            "tot_length": tot_length,
            "tot_length_minutes": tot_length / 60.0,
            "tot_length_hours": tot_length / 3600.0,
            "min_length": float(np.min(lengths)),
            "max_length": float(np.max(lengths)),
            "tot_scenes": int(np.sum(scenes)) if scenes.size else 0,
            "n_skeletons": int(np.sum(skeletons)) if skeletons.size else 0,
            "average_length": float(np.mean(lengths)),
            "std_deviation": float(np.std(lengths)),
            "average_fps": float(np.mean(fps)) if fps.size else 0.0,
            "std_fps": float(np.std(fps)) if fps.size else 0.0,
            "video_count": video_count,
            "tot_scene_len_hours": float(np.sum(scene_lengths) / 3600.0) if scene_lengths.size else 0.0,
        }

    return summary


def _print_summary_stats(summary_stats):
    print("\n=== Extracted Dataset Summary (computed in this script) ===")
    for culture, stats in summary_stats.items():
        print(f"\n[{culture}]")
        print(f"  video_count: {stats['video_count']}")
        print(f"  tot_length_hours: {stats['tot_length_hours']:.2f}")
        print(f"  tot_scene_len_hours: {stats['tot_scene_len_hours']:.2f}")
        print(f"  min_length_s: {stats['min_length']:.2f}")
        print(f"  max_length_s: {stats['max_length']:.2f}")
        print(f"  avg_length_s: {stats['average_length']:.2f} +/- {stats['std_deviation']:.2f}")
        print(f"  avg_fps: {stats['average_fps']:.2f} +/- {stats['std_fps']:.2f}")
        print(f"  total_scenes: {stats['tot_scenes']}")
        print(f"  total_poses: {stats['n_skeletons']}")


def _count_samples_by_culture(sample_keys):
    counts = {}
    for sample_key in sample_keys or []:
        key_str = str(sample_key).strip()
        if not key_str:
            continue
        culture = key_str.split("_", 1)[0].strip().lower()
        if not culture:
            continue
        counts[culture] = counts.get(culture, 0) + 1
    return counts


def _build_all_cultures_summary(summary_stats, sample_keys, culture_speakers):
    if not summary_stats:
        return {}

    included_cultures = {str(culture).strip().lower() for culture in summary_stats.keys()}
    samples_by_culture = _count_samples_by_culture(sample_keys)

    total_video_count = int(sum(int(stats.get("video_count", 0)) for stats in summary_stats.values()))
    total_length = float(sum(float(stats.get("tot_length", 0.0)) for stats in summary_stats.values()))
    total_scenes = int(sum(int(stats.get("tot_scenes", 0)) for stats in summary_stats.values()))
    total_poses = int(sum(int(stats.get("n_skeletons", 0)) for stats in summary_stats.values()))
    total_scene_len_hours = float(sum(float(stats.get("tot_scene_len_hours", 0.0)) for stats in summary_stats.values()))
    total_speakers = int(
        sum(len(set(culture_speakers.get(culture, []))) for culture in included_cultures)
    )
    total_samples = int(sum(samples_by_culture.get(culture, 0) for culture in included_cultures))

    min_lengths = [float(stats["min_length"]) for stats in summary_stats.values() if "min_length" in stats]
    max_lengths = [float(stats["max_length"]) for stats in summary_stats.values() if "max_length" in stats]
    total_fps_weight = float(
        sum(float(stats.get("average_fps", 0.0)) * int(stats.get("video_count", 0)) for stats in summary_stats.values())
    )

    return {
        "tot_length": total_length,
        "tot_length_minutes": total_length / 60.0,
        "tot_length_hours": total_length / 3600.0,
        "min_length": min(min_lengths) if min_lengths else 0.0,
        "max_length": max(max_lengths) if max_lengths else 0.0,
        "tot_scenes": total_scenes,
        "n_skeletons": total_poses,
        "average_length": (total_length / total_video_count) if total_video_count else 0.0,
        "average_fps": (total_fps_weight / total_video_count) if total_video_count else 0.0,
        "video_count": total_video_count,
        "tot_scene_len_hours": total_scene_len_hours,
        "speakers_with_samples": total_speakers,
        "final_dataset_samples": total_samples,
    }


def _save_summary_files(summary_stats, save_path, sample_keys=None, culture_speakers=None):
    summary_path_json = os.path.join(save_path, "summary_stats.json")
    summary_path_csv = os.path.join(save_path, "summary_stats.csv")
    summary_to_save = dict(summary_stats)

    all_cultures_summary = _build_all_cultures_summary(
        summary_stats,
        sample_keys or [],
        culture_speakers or {},
    )
    if all_cultures_summary:
        summary_to_save["All cultures"] = all_cultures_summary

    with open(summary_path_json, "w", encoding="utf-8") as f:
        json.dump(summary_to_save, f, indent=2)

    summary_df = pd.DataFrame.from_dict(summary_to_save, orient="index")
    summary_df.index.name = "culture"
    summary_df.to_csv(summary_path_csv)

    print(f"Saved summary json: {summary_path_json}")
    print(f"Saved summary csv: {summary_path_csv}")


def _save_plot(fig, out_path, show, tight_pad=1.08, pad_inches=0.1):
    if tight_pad is not None:
        fig.tight_layout(pad=tight_pad)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=pad_inches)
    if show:
        plt.show()
    plt.close(fig)


def _plot_overview_pie(summary_stats, save_path, show):
    cultures = list(summary_stats.keys())
    if not cultures:
        return

    total_poses = [summary_stats[c]["n_skeletons"] for c in cultures]
    total_poses_all = float(np.sum(total_poses))
    if total_poses_all <= 0:
        return

    colors = sns.color_palette("Set2", len(cultures))
    fig, ax = plt.subplots(figsize=(10, 6.5), subplot_kw=dict(aspect="equal"))
    wedges, _ = ax.pie(total_poses, colors=colors, startangle=140, wedgeprops={"linewidth": 1, "edgecolor": "white"})

    for idx, wedge in enumerate(wedges):
        angle = (wedge.theta2 - wedge.theta1) / 2.0 + wedge.theta1
        x = 0.62 * np.cos(np.deg2rad(angle))
        y = 0.62 * np.sin(np.deg2rad(angle))
        culture = cultures[idx]
        pct = (total_poses[idx] / total_poses_all) * 100.0
        stats = summary_stats[culture]
        # Culture name in bold/slightly larger text, details below in smaller text.
        ax.text(
            x,
            y + 0.12,
            f"{culture} ({pct:.1f}%)",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )
        details = (
            f"Poses: {stats['n_skeletons']}\n"
            f"Usable: {stats['tot_scene_len_hours']:.2f}h\n"
            f"Raw: {stats['tot_length_hours']:.2f}h\n"
            f"Speakers: {stats['video_count']}\n"
            f"Scenes: {stats['tot_scenes']}"
        )
        ax.text(x, y - 0.05, details, ha="center", va="center", fontsize=8)

    ax.legend(
        wedges,
        cultures,
        loc="center left",
        bbox_to_anchor=(0.9, 0.5),
        frameon=False,
        borderaxespad=0.2,
    )
    fig.subplots_adjust(left=0.03, right=0.82, top=0.97, bottom=0.03)
    _save_plot(
        fig,
        os.path.join(save_path, "3d_pie_chart.png"),
        show,
        tight_pad=None,
        pad_inches=0.02,
    )


def _build_long_df(culture_to_values, value_name, culture_name="Culture"):
    rows = []
    for culture, values in culture_to_values.items():
        for value in values:
            rows.append({culture_name: culture, value_name: value})
    return pd.DataFrame(rows)


def _plot_box(df, x_col, y_col, title, out_name, save_path, show):
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        print(f"[Skip] {out_name}: no valid data.")
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.boxplot(x=x_col, y=y_col, data=df, showfliers=False, ax=ax, palette="Set2")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(x_col, fontweight="bold")
    ax.set_ylabel(y_col, fontweight="bold")
    _save_plot(fig, os.path.join(save_path, out_name), show)


def _plot_strip(df, x_col, y_col, title, out_name, save_path, show):
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        print(f"[Skip] {out_name}: no valid data.")
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.stripplot(x=x_col, y=y_col, data=df, jitter=True, ax=ax, palette="Set2")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(x_col, fontweight="bold")
    ax.set_ylabel(y_col, fontweight="bold")
    _save_plot(fig, os.path.join(save_path, out_name), show)


def _plot_bar_means(summary_stats, field, title, y_label, out_name, save_path, show):
    if not summary_stats:
        print(f"[Skip] {out_name}: no summary stats.")
        return
    cultures = list(summary_stats.keys())
    values = [summary_stats[c][field] for c in cultures]
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(x=cultures, y=values, ax=ax, palette="Set2")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Culture", fontweight="bold")
    ax.set_ylabel(y_label, fontweight="bold")
    _save_plot(fig, os.path.join(save_path, out_name), show)


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Run dataset analysis and generate cultural summary plots.")
    parser.add_argument("--save-path", type=str, required=True, help="Directory where plots and summaries are saved.")
    parser.add_argument("--playlist-folder", type=str, required=True, help="Root folder containing culture/video folders.")
    parser.add_argument("--metadata-path", type=str, required=True, help="Path to metadata.pkl used by dataset_analysis.")
    parser.add_argument("--show", action="store_true", help="Display figures interactively in addition to saving them.")
    return parser


def main():
    args = _build_arg_parser().parse_args()
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    sns.set(style="whitegrid", palette="Set2")

    metadata = _load_metadata(args.metadata_path)
    cultures_from_meta = _load_cultures_from_metadata(metadata)
    sample_keys = metadata.get("sample_keys", [])
    culture_speakers = metadata.get("culture_speakers", {})
    playlist_root, matched_cultures, root_debug = _resolve_playlist_root(
        args.playlist_folder, cultures_from_meta, args.metadata_path
    )

    print("[Info] Playlist root resolution candidates:")
    for _, cand, matches in root_debug:
        if matches:
            print(f"  - {cand} -> matched cultures: {matches}")

    if os.path.abspath(str(playlist_root)) != os.path.abspath(args.playlist_folder):
        print(f"[Info] Auto-detected playlist root: {playlist_root}")

    if not matched_cultures:
        raise RuntimeError(
            "Could not locate playlist root from provided paths.\n"
            f"Given playlist folder: {args.playlist_folder}\n"
            f"Metadata path: {args.metadata_path}\n"
            "No candidate root contained culture directories matching metadata."
        )

    outputs = dataset_analysis(str(playlist_root), args.metadata_path)
    (
        culture_video_lengths,
        culture_video_fps,
        culture_video_scenes,
        culture_video_skeletons,
        culture_scene_lengths,
    ) = outputs

    summary_stats = _build_summary_stats(
        culture_video_lengths,
        culture_video_fps,
        culture_video_scenes,
        culture_video_skeletons,
        culture_scene_lengths,
    )

    if not summary_stats:
        raise RuntimeError(
            "No analyzable culture data was found.\n"
            f"Selected playlist root: {playlist_root}\n"
            f"Matched cultures at root resolution stage: {matched_cultures}\n"
            "This usually means speaker folders listed in metadata are not present under the selected root."
        )

    _print_summary_stats(summary_stats)
    _save_summary_files(
        summary_stats,
        save_path,
        sample_keys=sample_keys,
        culture_speakers=culture_speakers,
    )

    # Main overview figure.
    _plot_overview_pie(summary_stats, save_path, args.show)

    # Distribution plots.
    df_lengths = _build_long_df(culture_video_lengths, "Video Lengths (s)")
    df_fps = _build_long_df(culture_video_fps, "Video FPS")
    df_scenes = _build_long_df(culture_video_scenes, "N. Scenes in Videos")
    df_poses = _build_long_df(culture_video_skeletons, "N. Poses in Videos")
    df_scene_len = _build_long_df(culture_scene_lengths, "Scene Length (s)")

    _plot_box(df_lengths, "Culture", "Video Lengths (s)", "Distribution of Video Lengths per Culture", "boxplot_video_lengths.png", save_path, args.show)
    _plot_strip(df_fps, "Culture", "Video FPS", "Distribution of Video FPS per Culture", "strip_plot_video_fps.png", save_path, args.show)
    _plot_box(df_scenes, "Culture", "N. Scenes in Videos", "Distribution of Number of Scenes per Culture", "boxplot_video_scenes.png", save_path, args.show)
    _plot_box(df_poses, "Culture", "N. Poses in Videos", "Distribution of Number of Poses per Culture", "boxplot_video_poses.png", save_path, args.show)
    _plot_box(df_scene_len, "Culture", "Scene Length (s)", "Distribution of Scene Length per Culture", "boxplot_scene_length.png", save_path, args.show)

    _plot_bar_means(summary_stats, "average_length", "Average Video Length per Culture", "Average Video Length (s)", "bar_plot_avg_length.png", save_path, args.show)
    _plot_bar_means(summary_stats, "average_fps", "Average FPS per Culture", "Average FPS", "bar_plot_avg_fps.png", save_path, args.show)

    print(f"\nAnalysis completed. Outputs saved to: {save_path}")


if __name__ == "__main__":
    main()
