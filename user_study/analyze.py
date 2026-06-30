#!/usr/bin/env python3
import argparse
import itertools
import json
import math
import os
import textwrap
import warnings
from pathlib import Path
from typing import Dict, List, Sequence

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Avoid OpenMP shared-memory issues in restricted environments.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy import stats

from user_study.common import CONDITIONS, CULTURE_DISPLAY_NAME, QUESTION_DEFS

try:
    from PIL import Image, ImageDraw, ImageFont

    HAS_PILLOW = True
except Exception:
    Image = None
    ImageDraw = None
    ImageFont = None
    HAS_PILLOW = False

warnings.filterwarnings(
    "ignore",
    message="Exact p-value calculation does not work if there are zeros.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="Sample size too small for normal approximation.*",
    category=UserWarning,
)


def _load_font(size: int, bold: bool = False):
    if not HAS_PILLOW:
        return None

    candidates = (
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        ]
        if bold
        else [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        ]
    )
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _text_size(draw, text: str, font) -> tuple:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return math.nan


def _condition_label(cond: str) -> str:
    mapping = {
        "no_culture": "No Culture",
        "fishr": "Fishr",
        "adversarial": "Adversarial",
        "real": "Real",
    }
    return mapping.get(cond, cond)


def _question_title_map() -> Dict[str, str]:
    return {q["id"]: q["title"] for q in QUESTION_DEFS}


def _holm_correction(pvals: Dict[str, float]) -> Dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out: Dict[str, float] = {}
    prev = 0.0
    for i, (name, p) in enumerate(items, start=1):
        q = min(1.0, (m - i + 1) * p)
        q = max(q, prev)
        out[name] = q
        prev = q
    return out


def _ordered_trial_payloads(participant_payload: dict) -> List[dict]:
    deduped: Dict[int, dict] = {}
    extras: List[dict] = []
    for trial in participant_payload.get("responses", []):
        try:
            trial_index = int(trial.get("trial_index"))
            deduped[trial_index] = trial
        except Exception:
            extras.append(trial)
    ordered = [deduped[idx] for idx in sorted(deduped.keys())]
    ordered.extend(extras)
    return ordered


def _flatten_participant_trials(participant_payload: dict) -> List[dict]:
    participant = participant_payload.get("participant", {})
    rows: List[dict] = []
    for trial in _ordered_trial_payloads(participant_payload):
        base = {
            "session_id": participant_payload.get("session_id", ""),
            "participant_id": participant.get("participant_id", "anonymous"),
            "age": participant.get("age", np.nan),
            "gender": participant.get("gender", ""),
            "own_culture": participant.get("own_culture", ""),
            "cultures_interacted": ",".join(participant.get("cultures_interacted", [])),
            "trial_index": trial.get("trial_index"),
            "sequence_id": trial.get("sequence_id"),
            "culture": trial.get("culture"),
            "condition": trial.get("condition"),
            "timestamp": trial.get("timestamp"),
        }
        ratings = trial.get("ratings", {})
        for q in QUESTION_DEFS:
            qid = q["id"]
            base[qid] = ratings.get(qid, np.nan)
        rows.append(base)
    return rows


def _descriptive_stats(df_long: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    grouped = df_long.groupby(group_cols)["score"]
    out = grouped.agg(
        n="count",
        mean="mean",
        median="median",
        std="std",
        var="var",
        min="min",
        max="max",
    ).reset_index()
    out["sem"] = out["std"] / np.sqrt(out["n"].clip(lower=1))
    return out


def _paired_ttest_stats(complete: pd.DataFrame, levels: Sequence[str]) -> Dict[str, dict]:
    ttest_stats: Dict[str, dict] = {}
    ttest_pvals: Dict[str, float] = {}

    for a, b in itertools.combinations(levels, 2):
        name = f"{a}__vs__{b}"
        va = complete[a].values
        vb = complete[b].values
        diffs = va - vb
        ttest_stats[name] = {"n": int(len(diffs)), "mean_diff_a_minus_b": _safe_float(np.mean(diffs))}

        try:
            t_stat, t_p = stats.ttest_rel(va, vb, nan_policy="omit")
            ttest_stats[name]["statistic"] = _safe_float(t_stat)
            ttest_stats[name]["pvalue"] = _safe_float(t_p)
            ttest_pvals[name] = _safe_float(t_p)
        except Exception as exc:
            ttest_stats[name]["error"] = str(exc)

    if ttest_pvals:
        qvals = _holm_correction(ttest_pvals)
        for key, qval in qvals.items():
            ttest_stats[key]["qvalue_holm"] = qval

    return {"pairwise_paired_ttest": ttest_stats}


def _run_factor_tests(df_sub: pd.DataFrame, factor_col: str, levels: Sequence[str]) -> dict:
    levels = [level for level in levels if level in df_sub[factor_col].unique()]
    result = {
        "factor": factor_col,
        "levels": {},
        "participant_level": {},
        "test_family": "participant_level_paired_ttest_only",
    }
    if not levels:
        result["error"] = "no_levels_present"
        return result

    by_level = {
        level: df_sub[df_sub[factor_col] == level]["score"].dropna().values for level in levels
    }
    for level in levels:
        vals = by_level[level]
        result["levels"][level] = {
            "n_trials": int(len(vals)),
            "mean": _safe_float(np.mean(vals)) if len(vals) else math.nan,
            "std": _safe_float(np.std(vals, ddof=1)) if len(vals) > 1 else math.nan,
        }

    piv = (
        df_sub.groupby(["participant_id", factor_col])["score"]
        .mean()
        .unstack(factor_col)
        .reindex(columns=levels)
    )
    complete = piv.dropna(axis=0, how="any")
    result["participant_level"]["n_participants_complete"] = int(complete.shape[0])

    if complete.shape[0] >= 2 and len(levels) >= 2:
        result["participant_level"].update(_paired_ttest_stats(complete, levels))

    return result


def _run_significance(df_long: pd.DataFrame) -> Dict[str, dict]:
    results: Dict[str, dict] = {}
    cultures = sorted(df_long["culture"].dropna().unique().tolist())
    questions = [q["id"] for q in QUESTION_DEFS if q["id"] in df_long["question_id"].unique()]

    for qid in questions:
        sub = df_long[df_long["question_id"] == qid].copy()
        question_res = {
            "question_id": qid,
            "condition_tests": _run_factor_tests(sub, "condition", CONDITIONS),
            "culture_tests": _run_factor_tests(sub, "culture", cultures),
            "condition_tests_by_culture": {},
            "culture_tests_by_condition": {},
        }

        for culture in cultures:
            culture_sub = sub[sub["culture"] == culture].copy()
            if culture_sub.empty:
                continue
            question_res["condition_tests_by_culture"][culture] = _run_factor_tests(
                culture_sub,
                "condition",
                CONDITIONS,
            )

        for cond in CONDITIONS:
            cond_sub = sub[sub["condition"] == cond].copy()
            if cond_sub.empty:
                continue
            question_res["culture_tests_by_condition"][cond] = _run_factor_tests(
                cond_sub,
                "culture",
                cultures,
            )

        results[qid] = question_res

    return results


def _plot_grouped_bars(
    stats_df: pd.DataFrame,
    category_col: str,
    series_col: str,
    category_order: Sequence[str],
    series_order: Sequence[str],
    category_labels: Dict[str, str],
    series_labels: Dict[str, str],
    title: str,
    out_path: Path,
    y_max: float = 10.0,
) -> None:
    if not HAS_PILLOW:
        raise RuntimeError(
            "Pillow is required to generate PNG plots. Install it with: pip install Pillow"
        )

    if stats_df.empty:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    category_order = [cat for cat in category_order if cat in stats_df[category_col].unique()]
    series_order = [series for series in series_order if series in stats_df[series_col].unique()]
    if not category_order or not series_order:
        return

    n_series = max(1, len(series_order))
    palette = [
        "#2E86AB",
        "#E07A5F",
        "#3D9970",
        "#C0392B",
        "#7B6D8D",
        "#D4A017",
        "#2A9D8F",
        "#8C564B",
    ]
    title_lines = textwrap.wrap(title, width=62) or [title]
    width = max(920, 160 * len(category_order) + 220)
    height = 660
    margin_left = 92
    margin_right = 42
    margin_top = 34 + 24 * len(title_lines) + 42
    margin_bottom = 170
    plot_left = margin_left
    plot_top = margin_top
    plot_right = width - margin_right
    plot_bottom = height - margin_bottom
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top
    group_width = plot_width / max(1, len(category_order))
    total_bar_width = min(group_width * 0.8, group_width - 12)
    bar_width = total_bar_width / n_series

    color_text = "#222222"
    color_subtext = "#52606d"
    color_grid = "#d9dee4"
    color_axis = "#222222"
    color_error = "#2f2f2f"
    font_title = _load_font(24, bold=True)
    font_label = _load_font(16, bold=False)
    font_axis = _load_font(15, bold=False)
    font_legend = _load_font(14, bold=False)

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    for idx, line in enumerate(title_lines):
        y = 28 + idx * 22
        tw, th = _text_size(draw, line, font_title)
        draw.text(((width - tw) / 2, y - th), line, fill=color_text, font=font_title)

    for tick in range(0, int(y_max) + 1, 2):
        y = plot_bottom - (tick / y_max) * plot_height
        draw.line([(plot_left, y), (plot_right, y)], fill=color_grid, width=1)
        tick_text = str(tick)
        tw, th = _text_size(draw, tick_text, font_axis)
        draw.text((plot_left - 10 - tw, y - th / 2), tick_text, fill=color_subtext, font=font_axis)

    draw.line([(plot_left, plot_top), (plot_left, plot_bottom)], fill=color_axis, width=2)
    draw.line([(plot_left, plot_bottom), (plot_right, plot_bottom)], fill=color_axis, width=2)

    y_label = "Mean Rating"
    y_label_image = Image.new("RGBA", (220, 60), (255, 255, 255, 0))
    y_label_draw = ImageDraw.Draw(y_label_image)
    tw, th = _text_size(y_label_draw, y_label, font_axis)
    y_label_draw.text(((220 - tw) / 2, (60 - th) / 2), y_label, fill=color_text, font=font_axis)
    y_label_rot = y_label_image.rotate(90, expand=True)
    image.paste(
        y_label_rot,
        (10, int((plot_top + plot_bottom - y_label_rot.size[1]) / 2)),
        y_label_rot,
    )

    for cat_idx, category in enumerate(category_order):
        group_left = plot_left + cat_idx * group_width + (group_width - total_bar_width) / 2
        center_x = plot_left + (cat_idx + 0.5) * group_width
        for series_idx, series in enumerate(series_order):
            match = stats_df[
                (stats_df[category_col] == category) & (stats_df[series_col] == series)
            ]
            if match.empty:
                continue
            row = match.iloc[0]
            mean = _safe_float(row["mean"])
            sem = _safe_float(row.get("sem", math.nan))
            if math.isnan(mean):
                continue
            mean = min(max(mean, 0.0), y_max)
            bar_h = (mean / y_max) * plot_height
            x = group_left + series_idx * bar_width
            y = plot_bottom - bar_h
            color = palette[series_idx % len(palette)]
            draw.rectangle(
                [(x, y), (x + bar_width * 0.9, plot_bottom)],
                fill=color,
                outline=color,
            )
            if not math.isnan(sem) and sem > 0:
                low = max(0.0, mean - sem)
                high = min(y_max, mean + sem)
                y_low = plot_bottom - (low / y_max) * plot_height
                y_high = plot_bottom - (high / y_max) * plot_height
                x_mid = x + bar_width * 0.45
                cap = min(8.0, bar_width * 0.28)
                draw.line([(x_mid, y_high), (x_mid, y_low)], fill=color_error, width=2)
                draw.line([(x_mid - cap, y_high), (x_mid + cap, y_high)], fill=color_error, width=2)
                draw.line([(x_mid - cap, y_low), (x_mid + cap, y_low)], fill=color_error, width=2)

        label_lines = textwrap.wrap(
            str(category_labels.get(category, category)),
            width=18,
            break_long_words=False,
            break_on_hyphens=False,
        ) or [str(category)]
        label_lines = label_lines[:3]
        text_y = plot_bottom + 18
        for line_idx, line in enumerate(label_lines):
            tw, th = _text_size(draw, line, font_label)
            draw.text(
                (center_x - tw / 2, text_y + line_idx * 18),
                line,
                fill=color_error,
                font=font_label,
            )

    legend_y = 34 + 24 * len(title_lines)
    legend_labels = [str(series_labels.get(series, series)) for series in series_order]
    legend_font = font_legend
    item_padding = 14
    marker_w = 14
    marker_gap = 8
    gap = 20
    legend_item_widths = []
    for label in legend_labels:
        tw, _ = _text_size(draw, label, legend_font)
        legend_item_widths.append(marker_w + marker_gap + tw + item_padding)

    total_legend_width = sum(legend_item_widths) + gap * max(0, len(legend_item_widths) - 1)
    if total_legend_width > plot_width:
        legend_font = _load_font(12, bold=False)
        legend_item_widths = []
        for label in legend_labels:
            tw, _ = _text_size(draw, label, legend_font)
            legend_item_widths.append(marker_w + marker_gap + tw + item_padding)
        total_legend_width = sum(legend_item_widths) + gap * max(0, len(legend_item_widths) - 1)

    if total_legend_width > plot_width:
        gap = max(6, int((plot_width - sum(legend_item_widths)) / max(1, len(legend_item_widths) - 1)))
        total_legend_width = sum(legend_item_widths) + gap * max(0, len(legend_item_widths) - 1)

    legend_x = plot_left + max(0, int((plot_width - total_legend_width) / 2))
    current_x = legend_x
    for idx, series in enumerate(series_order):
        item_x = current_x
        item_y = legend_y
        color = palette[idx % len(palette)]
        draw.rectangle([(item_x, item_y - 10), (item_x + 14, item_y + 4)], fill=color, outline=color)
        draw.text((item_x + 22, item_y - 11), str(series_labels.get(series, series)), fill=color_error, font=legend_font)
        current_x += legend_item_widths[idx] + gap

    image.save(out_path, format="PNG")


def _generate_plots(df_long: pd.DataFrame, output_dir: Path) -> List[Path]:
    question_titles = _question_title_map()
    cultures = sorted(df_long["culture"].dropna().unique().tolist())

    question_condition = _descriptive_stats(df_long, ["question_id", "condition"])
    question_culture = _descriptive_stats(df_long, ["question_id", "culture"])
    question_condition_culture = _descriptive_stats(df_long, ["question_id", "condition", "culture"])

    paths: List[Path] = []
    plots_dir = output_dir / "plots"

    question_labels = {q["id"]: q["title"] for q in QUESTION_DEFS}
    culture_labels = {culture: CULTURE_DISPLAY_NAME.get(culture, culture.title()) for culture in cultures}
    condition_labels = {cond: _condition_label(cond) for cond in CONDITIONS}

    for q in QUESTION_DEFS:
        qid = q["id"]
        q_stats = question_condition_culture[question_condition_culture["question_id"] == qid].copy()
        out_path = plots_dir / "by_question" / f"{qid}_condition_by_culture.png"
        _plot_grouped_bars(
            stats_df=q_stats,
            category_col="culture",
            series_col="condition",
            category_order=cultures,
            series_order=CONDITIONS,
            category_labels=culture_labels,
            series_labels=condition_labels,
            title=f"{question_titles[qid]} by culture and model",
            out_path=out_path,
        )
        if out_path.exists():
            paths.append(out_path)

    for culture in cultures:
        c_stats = question_condition_culture[question_condition_culture["culture"] == culture].copy()
        out_path = plots_dir / "by_culture" / f"{culture}_question_by_condition.png"
        _plot_grouped_bars(
            stats_df=c_stats,
            category_col="question_id",
            series_col="condition",
            category_order=[q["id"] for q in QUESTION_DEFS],
            series_order=CONDITIONS,
            category_labels=question_labels,
            series_labels=condition_labels,
            title=f"{culture_labels.get(culture, culture)}: scores by question and model",
            out_path=out_path,
        )
        if out_path.exists():
            paths.append(out_path)

    for cond in CONDITIONS:
        cond_stats = question_condition_culture[question_condition_culture["condition"] == cond].copy()
        out_path = plots_dir / "by_condition" / f"{cond}_question_by_culture.png"
        _plot_grouped_bars(
            stats_df=cond_stats,
            category_col="question_id",
            series_col="culture",
            category_order=[q["id"] for q in QUESTION_DEFS],
            series_order=cultures,
            category_labels=question_labels,
            series_labels=culture_labels,
            title=f"{condition_labels.get(cond, cond)}: scores by question and culture",
            out_path=out_path,
        )
        if out_path.exists():
            paths.append(out_path)

    out_path = plots_dir / "overall" / "question_by_condition.png"
    _plot_grouped_bars(
        stats_df=question_condition,
        category_col="question_id",
        series_col="condition",
        category_order=[q["id"] for q in QUESTION_DEFS],
        series_order=CONDITIONS,
        category_labels=question_labels,
        series_labels=condition_labels,
        title="Overall mean ratings by question and model",
        out_path=out_path,
    )
    if out_path.exists():
        paths.append(out_path)

    out_path = plots_dir / "overall" / "question_by_culture.png"
    _plot_grouped_bars(
        stats_df=question_culture,
        category_col="question_id",
        series_col="culture",
        category_order=[q["id"] for q in QUESTION_DEFS],
        series_order=cultures,
        category_labels=question_labels,
        series_labels=culture_labels,
        title="Overall mean ratings by question and culture",
        out_path=out_path,
    )
    if out_path.exists():
        paths.append(out_path)

    return paths


def main():
    parser = argparse.ArgumentParser(description="Analyze gesture user-study ratings.")
    parser.add_argument("--results-dir", type=str, required=True, help="Path to study output results directory.")
    parser.add_argument("--output-dir", type=str, default=None, help="Where to write analysis outputs.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).expanduser().resolve()
    participants_dir = results_dir / "participants"
    if not participants_dir.exists():
        raise FileNotFoundError(f"Participants directory not found: {participants_dir}")

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (results_dir / "analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    participant_files = sorted(participants_dir.glob("*.json"))
    if not participant_files:
        raise RuntimeError(f"No participant files found in {participants_dir}")

    rows: List[dict] = []
    for pf in participant_files:
        payload = _read_json(pf)
        rows.extend(_flatten_participant_trials(payload))

    if not rows:
        raise RuntimeError("Participant files contain no trial responses.")

    df = pd.DataFrame(rows)
    for q in QUESTION_DEFS:
        qid = q["id"]
        df[qid] = pd.to_numeric(df[qid], errors="coerce")

    long_rows = []
    wide_cols = [
        "participant_id",
        "session_id",
        "age",
        "gender",
        "own_culture",
        "cultures_interacted",
        "trial_index",
        "sequence_id",
        "culture",
        "condition",
        "timestamp",
    ]
    for q in QUESTION_DEFS:
        qid = q["id"]
        tmp = df[wide_cols + [qid]].rename(columns={qid: "score"}).copy()
        tmp["question_id"] = qid
        long_rows.append(tmp)

    df_long = pd.concat(long_rows, axis=0, ignore_index=True)
    df_long["score"] = pd.to_numeric(df_long["score"], errors="coerce")
    df_long = df_long.dropna(subset=["score"])

    desc_condition = _descriptive_stats(df_long, ["question_id", "condition"])
    desc_culture = _descriptive_stats(df_long, ["question_id", "culture"])
    desc_condition_culture = _descriptive_stats(df_long, ["question_id", "condition", "culture"])
    desc_question = _descriptive_stats(df_long, ["question_id"])

    desc_question.to_csv(output_dir / "descriptive_by_question.csv", index=False)
    desc_condition.to_csv(output_dir / "descriptive_by_condition.csv", index=False)
    desc_culture.to_csv(output_dir / "descriptive_by_culture.csv", index=False)
    desc_condition_culture.to_csv(output_dir / "descriptive_by_condition_and_culture.csv", index=False)
    df.to_csv(output_dir / "participant_trials_wide.csv", index=False)
    df_long.to_csv(output_dir / "participant_trials_long.csv", index=False)

    significance = _run_significance(df_long)
    plot_paths = _generate_plots(df_long, output_dir)

    summary = {
        "participants_n": int(df["participant_id"].nunique()),
        "trials_n": int(df.shape[0]),
        "ratings_n": int(df_long.shape[0]),
        "conditions": CONDITIONS,
        "cultures": sorted(df_long["culture"].dropna().unique().tolist()),
        "questions": [
            {"id": q["id"], "title": q["title"], "prompt": q["prompt"]} for q in QUESTION_DEFS
        ],
        "plots_enabled": True,
        "generated_plot_count": len(plot_paths),
        "generated_plots": [str(path) for path in plot_paths],
        "significance": significance,
    }
    _write_json(output_dir / "significance_tests.json", summary)

    print(f"Participants: {summary['participants_n']}")
    print(f"Trials: {summary['trials_n']}")
    print(f"Ratings: {summary['ratings_n']}")
    print(f"Saved descriptive stats to: {output_dir / 'descriptive_by_condition.csv'}")
    print(f"Saved culture stats to: {output_dir / 'descriptive_by_culture.csv'}")
    print(f"Saved significance tests to: {output_dir / 'significance_tests.json'}")
    print(f"Generated PNG plots: {len(plot_paths)} under {output_dir / 'plots'}")


if __name__ == "__main__":
    main()
