#!/usr/bin/env python3
"""Compare two multi-run evaluation folders with statistical tests.

This script expects each folder to contain files like:
  - evaluation_run_01.json
  - evaluation_run_02.json
  - ...
produced by test_hierachical_mdm.py.

It computes, for each shared numeric metric:
  - Paired t-test (ttest_rel) on aligned runs
  - Wilcoxon signed-rank test on aligned runs
  - Welch t-test (ttest_ind, equal_var=False) on unpaired runs
  - Mann-Whitney U test on unpaired runs
and applies Benjamini-Hochberg FDR correction per test family.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats


def _safe_scalar(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, (int, float)):
        if math.isfinite(value):
            return float(value)
    return None


def _flatten_numeric_metrics(metrics: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    flat: Dict[str, float] = {}
    for key, value in metrics.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_numeric_metrics(value, prefix=full_key))
            continue
        scalar = _safe_scalar(value)
        if scalar is not None:
            flat[full_key] = scalar
    return flat


def _load_run_files(eval_dir: Path) -> List[Path]:
    run_files = sorted(eval_dir.glob("evaluation_run_*.json"))
    if run_files:
        return run_files

    summary_path = eval_dir / "evaluation_multi_run_summary.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        run_names = payload.get("run_files", [])
        recovered = [eval_dir / str(name) for name in run_names]
        recovered = [p for p in recovered if p.exists()]
        if recovered:
            return sorted(recovered)

    suggestions: List[str] = []
    parent = eval_dir.parent
    if parent.exists() and parent.is_dir():
        for child in sorted(parent.iterdir()):
            if not child.is_dir():
                continue
            has_runs = any(child.glob("evaluation_run_*.json"))
            has_summary = (child / "evaluation_multi_run_summary.json").exists()
            if has_runs or has_summary:
                suggestions.append(str(child))

    msg = f"No evaluation_run_*.json files found in: {eval_dir}"
    if suggestions:
        msg += "\\nNearby candidate evaluation folders:\\n  - " + "\\n  - ".join(suggestions)
    raise FileNotFoundError(msg)


def _load_runs(eval_dir: Path) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for run_file in _load_run_files(eval_dir):
        with run_file.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        metrics_root = payload.get("metrics", {})
        averaged = metrics_root.get("averaged_metrics", {})
        flat = _flatten_numeric_metrics(averaged)

        run_id = payload.get("run_id")
        seed = payload.get("seed")

        runs.append(
            {
                "file": str(run_file),
                "run_id": run_id,
                "seed": seed,
                "flat_metrics": flat,
            }
        )

    return runs


def _build_keyed_runs(
    runs: Sequence[Dict[str, Any]],
    pair_by: str,
) -> Dict[Any, Dict[str, Any]]:
    keyed: Dict[Any, Dict[str, Any]] = {}
    for idx, run in enumerate(runs):
        if pair_by == "seed":
            key = run.get("seed")
        elif pair_by == "run_id":
            key = run.get("run_id")
        elif pair_by == "index":
            key = idx
        else:
            raise ValueError(f"Unsupported pair_by: {pair_by}")

        if key is None:
            # Skip unkeyed runs for paired analyses (still used for unpaired).
            continue

        if key in keyed:
            raise ValueError(
                f"Duplicate pairing key '{key}' detected with pair_by='{pair_by}'."
            )
        keyed[key] = run
    return keyed


def _array_stats(arr: np.ndarray) -> Dict[str, Optional[float]]:
    out = {
        "n": int(arr.size),
        "mean": None,
        "std": None,
        "var": None,
        "median": None,
        "min": None,
        "max": None,
    }
    if arr.size == 0:
        return out
    out["mean"] = float(arr.mean())
    out["median"] = float(np.median(arr))
    out["min"] = float(arr.min())
    out["max"] = float(arr.max())
    if arr.size > 1:
        out["std"] = float(arr.std(ddof=1))
        out["var"] = float(arr.var(ddof=1))
    else:
        out["std"] = 0.0
        out["var"] = 0.0
    return out


def _cohens_d_unpaired(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if a.size < 2 or b.size < 2:
        return None
    va = a.var(ddof=1)
    vb = b.var(ddof=1)
    pooled = (va + vb) / 2.0
    if pooled <= 0:
        return None
    return float((a.mean() - b.mean()) / np.sqrt(pooled))


def _cohens_dz_paired(diff: np.ndarray) -> Optional[float]:
    if diff.size < 2:
        return None
    sd = diff.std(ddof=1)
    if sd <= 0:
        return None
    return float(diff.mean() / sd)


def _run_stat_tests(
    paired_a: np.ndarray,
    paired_b: np.ndarray,
    all_a: np.ndarray,
    all_b: np.ndarray,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "paired": {
            "n": int(paired_a.size),
            "mean_diff_a_minus_b": float((paired_a - paired_b).mean()) if paired_a.size else None,
            "cohens_dz": _cohens_dz_paired(paired_a - paired_b),
            "ttest_rel": None,
            "wilcoxon": None,
        },
        "unpaired": {
            "n_a": int(all_a.size),
            "n_b": int(all_b.size),
            "cohens_d": _cohens_d_unpaired(all_a, all_b),
            "welch_ttest": None,
            "mannwhitney_u": None,
        },
    }

    # Paired parametric
    if paired_a.size >= 2:
        try:
            t_stat, p_val = stats.ttest_rel(paired_a, paired_b, nan_policy="omit")
            if np.isfinite(t_stat) and np.isfinite(p_val):
                result["paired"]["ttest_rel"] = {
                    "statistic": float(t_stat),
                    "pvalue": float(p_val),
                }
        except Exception:
            pass

        # Paired non-parametric
        try:
            w_stat, w_p = stats.wilcoxon(paired_a, paired_b, zero_method="wilcox")
            if np.isfinite(w_stat) and np.isfinite(w_p):
                result["paired"]["wilcoxon"] = {
                    "statistic": float(w_stat),
                    "pvalue": float(w_p),
                }
        except Exception:
            pass

    # Unpaired parametric
    if all_a.size >= 2 and all_b.size >= 2:
        try:
            t_stat_u, p_val_u = stats.ttest_ind(all_a, all_b, equal_var=False, nan_policy="omit")
            if np.isfinite(t_stat_u) and np.isfinite(p_val_u):
                result["unpaired"]["welch_ttest"] = {
                    "statistic": float(t_stat_u),
                    "pvalue": float(p_val_u),
                }
        except Exception:
            pass

    # Unpaired non-parametric
    if all_a.size >= 1 and all_b.size >= 1:
        try:
            u_stat, u_p = stats.mannwhitneyu(all_a, all_b, alternative="two-sided")
            if np.isfinite(u_stat) and np.isfinite(u_p):
                result["unpaired"]["mannwhitney_u"] = {
                    "statistic": float(u_stat),
                    "pvalue": float(u_p),
                }
        except Exception:
            pass

    return result


def _fdr_bh(pvalues_by_metric: Dict[str, float]) -> Dict[str, float]:
    if not pvalues_by_metric:
        return {}
    items = [(metric, p) for metric, p in pvalues_by_metric.items() if p is not None and np.isfinite(p)]
    if not items:
        return {}

    metrics = [m for m, _ in items]
    pvals = np.asarray([p for _, p in items], dtype=np.float64)
    n = pvals.size
    order = np.argsort(pvals)
    ranked = pvals[order]

    qvals = np.empty(n, dtype=np.float64)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        q = ranked[i] * n / rank
        if q > prev:
            q = prev
        prev = q
        qvals[i] = q

    qvals_original = np.empty(n, dtype=np.float64)
    qvals_original[order] = qvals
    return {metrics[i]: float(min(1.0, max(0.0, qvals_original[i]))) for i in range(n)}


def _extract_pvalue(entry: Dict[str, Any], *path: str) -> Optional[float]:
    cur: Any = entry
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    if isinstance(cur, (float, int)) and np.isfinite(cur):
        return float(cur)
    return None


def _collect_model_values(runs: Sequence[Dict[str, Any]], metric: str) -> np.ndarray:
    vals: List[float] = []
    for run in runs:
        v = run["flat_metrics"].get(metric)
        if v is not None and np.isfinite(v):
            vals.append(float(v))
    return np.asarray(vals, dtype=np.float64)


def _collect_paired_values(
    keyed_a: Dict[Any, Dict[str, Any]],
    keyed_b: Dict[Any, Dict[str, Any]],
    metric: str,
) -> Tuple[np.ndarray, np.ndarray, List[Any]]:
    keys = sorted(set(keyed_a.keys()).intersection(keyed_b.keys()), key=lambda x: str(x))
    vals_a: List[float] = []
    vals_b: List[float] = []
    used_keys: List[Any] = []
    for key in keys:
        va = keyed_a[key]["flat_metrics"].get(metric)
        vb = keyed_b[key]["flat_metrics"].get(metric)
        if va is None or vb is None:
            continue
        if not (np.isfinite(va) and np.isfinite(vb)):
            continue
        vals_a.append(float(va))
        vals_b.append(float(vb))
        used_keys.append(key)
    return np.asarray(vals_a, dtype=np.float64), np.asarray(vals_b, dtype=np.float64), used_keys


def compare_evaluations(
    model_a_dir: Path,
    model_b_dir: Path,
    model_a_name: str,
    model_b_name: str,
    pair_by: str,
    alpha: float,
) -> Dict[str, Any]:
    runs_a = _load_runs(model_a_dir)
    runs_b = _load_runs(model_b_dir)

    keyed_a = _build_keyed_runs(runs_a, pair_by=pair_by)
    keyed_b = _build_keyed_runs(runs_b, pair_by=pair_by)
    common_pair_keys = sorted(set(keyed_a.keys()).intersection(keyed_b.keys()), key=lambda x: str(x))

    metrics_a = set()
    for run in runs_a:
        metrics_a.update(run["flat_metrics"].keys())
    metrics_b = set()
    for run in runs_b:
        metrics_b.update(run["flat_metrics"].keys())

    shared_metrics = sorted(metrics_a.intersection(metrics_b))

    per_metric: Dict[str, Any] = {}

    for metric in shared_metrics:
        all_a = _collect_model_values(runs_a, metric)
        all_b = _collect_model_values(runs_b, metric)
        paired_a, paired_b, used_pair_keys = _collect_paired_values(keyed_a, keyed_b, metric)

        tests = _run_stat_tests(
            paired_a=paired_a,
            paired_b=paired_b,
            all_a=all_a,
            all_b=all_b,
        )

        per_metric[metric] = {
            "model_a": _array_stats(all_a),
            "model_b": _array_stats(all_b),
            "paired": {
                "pair_keys_used": used_pair_keys,
                "pair_keys_used_count": int(len(used_pair_keys)),
            },
            "tests": tests,
        }

    # Multiple-testing correction by test family
    p_paired_t = {
        m: _extract_pvalue(per_metric[m], "tests", "paired", "ttest_rel", "pvalue")
        for m in per_metric
    }
    p_wilcoxon = {
        m: _extract_pvalue(per_metric[m], "tests", "paired", "wilcoxon", "pvalue")
        for m in per_metric
    }
    p_welch = {
        m: _extract_pvalue(per_metric[m], "tests", "unpaired", "welch_ttest", "pvalue")
        for m in per_metric
    }
    p_mwu = {
        m: _extract_pvalue(per_metric[m], "tests", "unpaired", "mannwhitney_u", "pvalue")
        for m in per_metric
    }

    q_paired_t = _fdr_bh({m: p for m, p in p_paired_t.items() if p is not None})
    q_wilcoxon = _fdr_bh({m: p for m, p in p_wilcoxon.items() if p is not None})
    q_welch = _fdr_bh({m: p for m, p in p_welch.items() if p is not None})
    q_mwu = _fdr_bh({m: p for m, p in p_mwu.items() if p is not None})

    for metric, payload in per_metric.items():
        payload["tests"]["fdr_bh"] = {
            "paired_ttest_rel_qvalue": q_paired_t.get(metric),
            "paired_wilcoxon_qvalue": q_wilcoxon.get(metric),
            "unpaired_welch_ttest_qvalue": q_welch.get(metric),
            "unpaired_mannwhitney_u_qvalue": q_mwu.get(metric),
        }
        payload["tests"]["significance"] = {
            "alpha": alpha,
            "paired_ttest_rel_significant": (q_paired_t.get(metric) is not None and q_paired_t.get(metric) < alpha),
            "paired_wilcoxon_significant": (q_wilcoxon.get(metric) is not None and q_wilcoxon.get(metric) < alpha),
            "unpaired_welch_ttest_significant": (q_welch.get(metric) is not None and q_welch.get(metric) < alpha),
            "unpaired_mannwhitney_u_significant": (q_mwu.get(metric) is not None and q_mwu.get(metric) < alpha),
        }

    result = {
        "metadata": {
            "model_a_name": model_a_name,
            "model_b_name": model_b_name,
            "model_a_dir": str(model_a_dir),
            "model_b_dir": str(model_b_dir),
            "runs_model_a": len(runs_a),
            "runs_model_b": len(runs_b),
            "pair_by": pair_by,
            "common_pair_keys": common_pair_keys,
            "common_pair_keys_count": len(common_pair_keys),
            "shared_metrics_count": len(shared_metrics),
            "alpha": alpha,
        },
        "metrics": per_metric,
    }
    return result


def _write_csv(output_csv: Path, result: Dict[str, Any]) -> None:
    rows: List[Dict[str, Any]] = []
    model_a_name = result["metadata"]["model_a_name"]
    model_b_name = result["metadata"]["model_b_name"]

    for metric, payload in result["metrics"].items():
        tests = payload["tests"]
        paired = tests.get("paired", {})
        unpaired = tests.get("unpaired", {})
        fdr = tests.get("fdr_bh", {})
        signif = tests.get("significance", {})

        row = {
            "metric": metric,
            f"{model_a_name}_mean": payload["model_a"].get("mean"),
            f"{model_a_name}_std": payload["model_a"].get("std"),
            f"{model_a_name}_n": payload["model_a"].get("n"),
            f"{model_b_name}_mean": payload["model_b"].get("mean"),
            f"{model_b_name}_std": payload["model_b"].get("std"),
            f"{model_b_name}_n": payload["model_b"].get("n"),
            "paired_n": paired.get("n"),
            "mean_diff_a_minus_b": paired.get("mean_diff_a_minus_b"),
            "cohens_dz_paired": paired.get("cohens_dz"),
            "cohens_d_unpaired": unpaired.get("cohens_d"),
            "p_ttest_rel": _extract_pvalue(payload, "tests", "paired", "ttest_rel", "pvalue"),
            "q_ttest_rel_bh": fdr.get("paired_ttest_rel_qvalue"),
            "sig_ttest_rel_bh": signif.get("paired_ttest_rel_significant"),
            "p_wilcoxon": _extract_pvalue(payload, "tests", "paired", "wilcoxon", "pvalue"),
            "q_wilcoxon_bh": fdr.get("paired_wilcoxon_qvalue"),
            "sig_wilcoxon_bh": signif.get("paired_wilcoxon_significant"),
            "p_welch_ttest": _extract_pvalue(payload, "tests", "unpaired", "welch_ttest", "pvalue"),
            "q_welch_ttest_bh": fdr.get("unpaired_welch_ttest_qvalue"),
            "sig_welch_ttest_bh": signif.get("unpaired_welch_ttest_significant"),
            "p_mannwhitney_u": _extract_pvalue(payload, "tests", "unpaired", "mannwhitney_u", "pvalue"),
            "q_mannwhitney_u_bh": fdr.get("unpaired_mannwhitney_u_qvalue"),
            "sig_mannwhitney_u_bh": signif.get("unpaired_mannwhitney_u_significant"),
        }
        rows.append(row)

    rows.sort(key=lambda r: r["metric"])

    if not rows:
        raise ValueError("No rows to write in CSV report.")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute cross-model statistical tests over multi-run evaluation outputs."
    )
    parser.add_argument(
        "--model-a-dir",
        type=str,
        required=True,
        help="Directory containing evaluation_run_*.json for model A.",
    )
    parser.add_argument(
        "--model-b-dir",
        type=str,
        required=True,
        help="Directory containing evaluation_run_*.json for model B.",
    )
    parser.add_argument(
        "--model-a-name",
        type=str,
        default="model_a",
        help="Display name for model A in outputs.",
    )
    parser.add_argument(
        "--model-b-name",
        type=str,
        default="model_b",
        help="Display name for model B in outputs.",
    )
    parser.add_argument(
        "--pair-by",
        type=str,
        choices=["seed", "run_id", "index"],
        default="seed",
        help="How to align runs for paired tests.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level used for BH-corrected significance flags.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        required=True,
        help="Path to save full JSON report.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Path to save compact CSV report. Default: same stem as --output-json.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    model_a_dir = Path(args.model_a_dir).expanduser().resolve()
    model_b_dir = Path(args.model_b_dir).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    output_csv = (
        Path(args.output_csv).expanduser().resolve()
        if args.output_csv
        else output_json.with_suffix(".csv")
    )

    result = compare_evaluations(
        model_a_dir=model_a_dir,
        model_b_dir=model_b_dir,
        model_a_name=args.model_a_name,
        model_b_name=args.model_b_name,
        pair_by=args.pair_by,
        alpha=args.alpha,
    )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    _write_csv(output_csv, result)

    print(f"Saved JSON report: {output_json}")
    print(f"Saved CSV report: {output_csv}")
    print(f"Shared metrics: {result['metadata']['shared_metrics_count']}")
    print(f"Paired runs ({args.pair_by}): {result['metadata']['common_pair_keys_count']}")


if __name__ == "__main__":
    main()
