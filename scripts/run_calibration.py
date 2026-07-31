#!/usr/bin/env python3

"""
Prompt Calibration Pipeline

Selects one scoring prompt per model through a factorial calibration process,
and derives per-institution inverse-RMSE weights for weighted mean aggregation.

Phases:
    split   — compute and display the stratified calibration/test split
    score   — score calibration essays under all 27 prompt × 3 model conditions
    analyse — select one prompt per model on the pooled split, derive per-institution
              RMSE and bootstrap weights under that prompt, update configs

The split is computed deterministically from a seed stored in the institution YAML,
so no database flag is needed, the same seed always produces the same partition.
"""

import os
import sys
import json
import asyncio
import argparse
import math
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.classifier.models import BaseModel, ClaudeModel, GptModel, GeminiModel
from src.classifier.prompt_manager import build_model_prompt
from src.classifier.utils import load_yaml_config
from src.prompts.prompt_renderer import PromptRenderer
from src.utils.scoring_utils import ScoringUtils
from src.utils.db_utils import (
    get_connection,
    ensure_calibration_table,
)
from src.utils.calibration_utils import get_calibration_split


def print_split_summary(
    cal_rows: List[Dict], test_rows: List[Dict], university: str,
) -> None:
    """Print a summary of the calibration/test split."""
    total = len(cal_rows) + len(test_rows)
    print(f"\n{'=' * 60}")
    print(f"  {university.upper()} — Stratified Split")
    print(f"  Total: {total}  |  Calibration: {len(cal_rows)} ({100*len(cal_rows)/total:.0f}%)"
          f"  |  Test: {len(test_rows)} ({100*len(test_rows)/total:.0f}%)")
    print(f"{'=' * 60}")

    # Band breakdown
    cal_bands: Dict[str, int] = {}
    test_bands: Dict[str, int] = {}
    for r in cal_rows:
        b = r["human_grade_band"] or "Unknown"
        cal_bands[b] = cal_bands.get(b, 0) + 1
    for r in test_rows:
        b = r["human_grade_band"] or "Unknown"
        test_bands[b] = test_bands.get(b, 0) + 1

    all_bands = sorted(set(cal_bands) | set(test_bands))
    print(f"  {'Band':<20} {'Cal':>5} {'Test':>5} {'Total':>6}")
    print(f"  {'-'*20} {'-'*5} {'-'*5} {'-'*6}")
    for band in all_bands:
        c = cal_bands.get(band, 0)
        t = test_bands.get(band, 0)
        print(f"  {band:<20} {c:>5} {t:>5} {c+t:>6}")
    print()



# Scoring

async def score_calibration_set(
    university: str,
    task_config: Dict[str, Any],
    cal_rows: List[Dict[str, Any]],
    library_path: Path = None,
) -> None:
    """Score all calibration essays under every model × prompt combination.

    Results are saved to the library's calibration table. Already-scored
    combinations are skipped (resumable).
    """
    renderer = PromptRenderer(library_path)
    table = renderer.config.get("calibration_table", "calibration_scores")
    prompt_ids = sorted(renderer.templates.keys())
    full_prompts = {
        pid: build_model_prompt(renderer, pid, university, task_config)
        for pid in prompt_ids
    }

    models = [
        GeminiModel(os.getenv("GEMINI_MODEL"), name="Gemini"),
        GptModel(os.getenv("GPT_MODEL"), name="GPT"),
        ClaudeModel(os.getenv("CLAUDE_MODEL"), name="Claude"),
    ]

    conn = get_connection()
    ensure_calibration_table(conn, table)

    cal_ids = {r["id"] for r in cal_rows}
    cal_by_id = {r["id"]: r for r in cal_rows}

    # Count total work and already-done
    total_combos = len(cal_ids) * len(models) * len(prompt_ids)
    done = conn.execute(
        "SELECT COUNT(*) FROM {} WHERE essay_id IN ({})".format(
            table, ",".join("?" * len(cal_ids))
        ),
        list(cal_ids),
    ).fetchone()[0]

    remaining = total_combos - done
    print(f"\nCalibration scoring for {university}")
    print(f"  Essays: {len(cal_ids)}  |  Models: {len(models)}  |  Prompts: {len(prompt_ids)}")
    print(f"  Total combinations: {total_combos}  |  Already done: {done}  |  Remaining: {remaining}")

    if remaining == 0:
        print("  All combinations already scored — nothing to do.")
        conn.close()
        return

    # Per-model semaphores — each API has independent rate limits
    model_semaphores = {
        "Gemini": asyncio.Semaphore(15),
        "GPT": asyncio.Semaphore(20),
        "Claude": asyncio.Semaphore(20),
    }
    scored = 0

    async def score_one(essay_id: int, model: BaseModel, prompt_id: str):
        nonlocal scored

        # Check if already scored
        exists = conn.execute(
            f"SELECT 1 FROM {table} WHERE essay_id = ? AND model = ? AND prompt_id = ?",
            (essay_id, model.name, prompt_id),
        ).fetchone()
        if exists:
            return

        async with model_semaphores[model.name]:
            text = cal_by_id[essay_id]["assignment_content"]
            brief = cal_by_id[essay_id].get("assignment_brief")
            if brief:
                text = f"=== ASSIGNMENT BRIEF ===\n{brief}\n\n=== STUDENT SUBMISSION ===\n{text}"
            prompt = full_prompts[prompt_id]

            result = await model.classify_content(
                {"type": "text", "content": text}, prompt
            )

            raw = json.dumps(result) if isinstance(result, dict) else str(result)

            if "error" in result:
                # Don't persist errors — allows retry on restart
                scored += 1
                if scored % 25 == 0 or scored == remaining:
                    print(f"  [{scored}/{remaining}] scored (error: {model.name}/{prompt_id})")
                return

            grade = ScoringUtils.extract_grade(result.get("classification"))
            band = ScoringUtils.get_grade_band(grade) if grade is not None else None

            conn.execute(
                f"INSERT OR REPLACE INTO {table} "
                "(essay_id, model, prompt_id, score, band, raw_response) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (essay_id, model.name, prompt_id, grade, band, raw),
            )
            conn.commit()

            scored += 1
            if scored % 25 == 0 or scored == remaining:
                print(f"  [{scored}/{remaining}] scored")

    # Build task list
    tasks = []
    for essay_id in sorted(cal_ids):
        for model in models:
            for pid in prompt_ids:
                tasks.append(score_one(essay_id, model, pid))

    await asyncio.gather(*tasks)
    conn.close()
    print(f"  Done. Scored {scored} new combinations.")


# Analysis

MODEL_PREFIX_MAP = {
    "gemini": "gemini",
    "gpt": "gpt",
    "claude": "claude",
}


def compute_rmse(truths: List[float], preds: List[float]) -> float:
    """Root mean square error."""
    if not truths:
        return float("inf")
    return math.sqrt(sum((t - p) ** 2 for t, p in zip(truths, preds)) / len(truths))


def load_calibration_scores(
    cal_rows: List[Dict[str, Any]], table: str = "calibration_scores",
) -> Tuple[List[Any], Dict[int, float]]:
    """Load calibration scores for the given essays; returns (rows, human_grades)."""
    conn = get_connection()
    cal_ids = sorted(r["id"] for r in cal_rows)
    human_grades = {r["id"]: r["human_grade"] for r in cal_rows}
    rows = conn.execute(
        "SELECT essay_id, model, prompt_id, score FROM {} "
        "WHERE essay_id IN ({})".format(table, ",".join("?" * len(cal_ids))),
        cal_ids,
    ).fetchall()
    conn.close()
    return rows, human_grades


def build_rmse_matrix(rows, human_grades) -> Dict[str, Dict[str, float]]:
    """RMSE per model × prompt: {model_prefix: {prompt_id: rmse}}."""
    scores: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}
    for row in rows:
        if row["score"] is None:
            continue
        model_key = MODEL_PREFIX_MAP.get(row["model"].lower(), row["model"].lower())
        scores.setdefault(model_key, {}).setdefault(row["prompt_id"], []).append(
            (float(human_grades[row["essay_id"]]), float(row["score"]))
        )

    rmse_matrix: Dict[str, Dict[str, float]] = {}
    for model_key, prompts in scores.items():
        rmse_matrix[model_key] = {}
        for pid, pairs in prompts.items():
            truths, preds = zip(*pairs)
            rmse_matrix[model_key][pid] = compute_rmse(list(truths), list(preds))
    return rmse_matrix


def select_pooled_prompts(
    cal_rows: List[Dict[str, Any]], table: str = "calibration_scores",
) -> Dict[str, str]:
    """Select one prompt per model by lowest RMSE on the pooled calibration split."""
    # Selection is pooled across institutions and the choice deployed unchanged at
    # every site: a deployed system cannot verify a per-site optimum in advance, and
    # per-institution optima are unstable across models. Per-institution optima are
    # reported only as a counterfactual (transfer analysis), never deployed.
    rows, human_grades = load_calibration_scores(cal_rows, table)
    rmse_matrix = build_rmse_matrix(rows, human_grades)
    return {mk: min(pr, key=pr.get) for mk, pr in rmse_matrix.items()}


def analyse_calibration(
    cal_rows: List[Dict[str, Any]], n_bootstrap: int = 1000, seed: int = 42,
    table: str = "calibration_scores", selected_prompts: Dict[str, str] = None,
) -> Dict[str, Any]:
    """Analyze calibration scores: RMSE per model×prompt, per-institution weights.

    Returns a dict with:
        best_prompts: {model_prefix: prompt_id}  (the pooled selection, when given)
        rmse_matrix: {model_prefix: {prompt_id: rmse}}
        calibration_rmse: {model_prefix: rmse}  (selected prompt's RMSE on cal_rows)
        bootstrap_weights: {model_prefix: {mean, std, ci_lower, ci_upper}}
        factorial_analysis: {dimension: {level: mean_rmse}}
    """
    cal_ids = sorted(r["id"] for r in cal_rows)
    rows, human_grades = load_calibration_scores(cal_rows, table)
    rmse_matrix = build_rmse_matrix(rows, human_grades)

    # --- Selected prompt per model ---
    # `selected_prompts` carries the pooled choice. The per-set argmin is a fallback
    # for ad-hoc inspection only; it is not the deployed selection rule.
    best_prompts: Dict[str, str] = {}
    calibration_rmse: Dict[str, float] = {}
    for model_key, prompt_rmses in rmse_matrix.items():
        pid = (selected_prompts or {}).get(model_key) or min(prompt_rmses, key=prompt_rmses.get)
        best_prompts[model_key] = pid
        calibration_rmse[model_key] = prompt_rmses[pid]

    # --- Bootstrap weights ---
    rng = np.random.default_rng(seed)
    n_cal = len(cal_ids)
    model_keys = sorted(calibration_rmse.keys())

    # Build essay_id-keyed lookup of (truth, pred) for each model's best prompt
    best_pair_by_essay: Dict[str, Dict[int, Tuple[float, float]]] = {}
    for mk in model_keys:
        pid = best_prompts[mk]
        lookup = {}
        for row_data in rows:
            if row_data["score"] is None:
                continue
            mk2 = MODEL_PREFIX_MAP.get(row_data["model"].lower(), row_data["model"].lower())
            if mk2 == mk and row_data["prompt_id"] == pid:
                eid = row_data["essay_id"]
                lookup[eid] = (float(human_grades[eid]), float(row_data["score"]))
        best_pair_by_essay[mk] = lookup

    bootstrap_weights_samples: Dict[str, List[float]] = {mk: [] for mk in model_keys}

    for _ in range(n_bootstrap):
        resample_indices = rng.integers(0, n_cal, size=n_cal)
        resample_ids = [cal_ids[i] for i in resample_indices]

        resample_rmses = {}
        for mk in model_keys:
            lookup = best_pair_by_essay[mk]
            truths, preds = [], []
            for eid in resample_ids:
                if eid in lookup:
                    t, p = lookup[eid]
                    truths.append(t)
                    preds.append(p)
            resample_rmses[mk] = compute_rmse(truths, preds) if truths else float("inf")

        # Compute inverse-RMSE weights for this resample
        # Floor RMSE at 0.01 to avoid division by zero when a model scores perfectly
        inv_sum = sum(1.0 / max(r, 0.01) for r in resample_rmses.values() if r < float("inf"))
        if inv_sum > 0:
            for mk in model_keys:
                r = resample_rmses[mk]
                w = (1.0 / max(r, 0.01)) / inv_sum if r < float("inf") else 0.0
                bootstrap_weights_samples[mk].append(w)

    bootstrap_weights: Dict[str, Dict[str, float]] = {}
    for mk in model_keys:
        samples = np.array(bootstrap_weights_samples[mk])
        bootstrap_weights[mk] = {
            "mean": float(np.mean(samples)),
            "std": float(np.std(samples)),
            "ci_lower": float(np.percentile(samples, 2.5)),
            "ci_upper": float(np.percentile(samples, 97.5)),
        }

    # --- Factorial analysis: which dimension matters most ---
    factorial: Dict[str, Dict[str, List[float]]] = {
        "rubric": {},       # A dimension: criterion specificity
        "calibration": {},  # B dimension: calibration intervention
        "strategy": {},     # C dimension: scoring strategy
    }
    dimension_labels = {
        "rubric": {"1": "none", "2": "summary", "3": "full"},
        "calibration": {"1": "none", "2": "distributional", "3": "distributional_metacognitive"},
        "strategy": {"1": "direct", "2": "analytic", "3": "deliberative"},
    }

    for model_key, prompt_rmses in rmse_matrix.items():
        for pid, rmse_val in prompt_rmses.items():
            # Parse A#_B#_C#
            parts = pid.split("_")
            a_level, b_level, c_level = parts[0][1], parts[1][1], parts[2][1]

            a_label = dimension_labels["rubric"][a_level]
            b_label = dimension_labels["calibration"][b_level]
            c_label = dimension_labels["strategy"][c_level]

            factorial["rubric"].setdefault(a_label, []).append(rmse_val)
            factorial["calibration"].setdefault(b_label, []).append(rmse_val)
            factorial["strategy"].setdefault(c_label, []).append(rmse_val)

    factorial_summary = {}
    for dim, levels in factorial.items():
        factorial_summary[dim] = {
            level: round(float(np.mean(vals)), 2)
            for level, vals in sorted(levels.items())
        }

    return {
        "best_prompts": best_prompts,
        "rmse_matrix": {
            mk: {pid: round(v, 2) for pid, v in sorted(prmses.items())}
            for mk, prmses in rmse_matrix.items()
        },
        "calibration_rmse": {mk: round(v, 2) for mk, v in calibration_rmse.items()},
        "bootstrap_weights": bootstrap_weights,
        "factorial_analysis": factorial_summary,
    }


def print_analysis(results: Dict[str, Any], university: str) -> None:
    """Print a human-readable summary of calibration analysis."""
    print(f"\n{'=' * 60}")
    print(f"  CALIBRATION ANALYSIS — {university.upper()}")
    print(f"{'=' * 60}")

    # Selected prompts (pooled choice; RMSE is this institution's)
    print(f"\n  Selected prompt per model (RMSE at this institution):")
    for mk, pid in sorted(results["best_prompts"].items()):
        rmse = results["calibration_rmse"][mk]
        print(f"    {mk:<8} → {pid}  (RMSE = {rmse})")

    # Bootstrap weights
    print(f"\n  Bootstrapped weights (1,000 resamples, 95% CI):")
    for mk, w in sorted(results["bootstrap_weights"].items()):
        print(f"    {mk:<8} → {w['mean']:.4f}  [{w['ci_lower']:.4f}, {w['ci_upper']:.4f}]")

    # Factorial analysis
    print(f"\n  Factorial analysis (mean RMSE across models):")
    for dim, levels in results["factorial_analysis"].items():
        print(f"    {dim}:")
        for level, mean_rmse in levels.items():
            print(f"      {level:<15} {mean_rmse}")

    # Top 5 / bottom 5 prompts overall
    print(f"\n  Top 5 prompts (averaged across models):")
    avg_by_prompt: Dict[str, List[float]] = {}
    for mk, prmses in results["rmse_matrix"].items():
        for pid, rmse in prmses.items():
            avg_by_prompt.setdefault(pid, []).append(rmse)
    avg_sorted = sorted(
        ((pid, float(np.mean(vals))) for pid, vals in avg_by_prompt.items()),
        key=lambda x: x[1],
    )
    for pid, avg_rmse in avg_sorted[:5]:
        print(f"    {pid}: {avg_rmse:.2f}")
    print(f"  Bottom 5 prompts:")
    for pid, avg_rmse in avg_sorted[-5:]:
        print(f"    {pid}: {avg_rmse:.2f}")
    print()


# Config writer

def update_institution_config(
    university: str, results: Dict[str, Any],
) -> None:
    """Write calibration RMSE, best prompts, and default prompt back to institution YAML.

    Uses regex-based replacement to preserve the existing file formatting,
    comments, and inline lists.
    """
    import re

    config_path = PROJECT_ROOT / "src" / "config" / "institutions" / f"{university}.yaml"
    text = config_path.read_text()

    # Determine overall best prompt (lowest average RMSE across models)
    avg_by_prompt: Dict[str, List[float]] = {}
    for _mk, prmses in results["rmse_matrix"].items():
        for pid, rmse_val in prmses.items():
            avg_by_prompt.setdefault(pid, []).append(rmse_val)
    overall_best = min(avg_by_prompt, key=lambda pid: float(np.mean(avg_by_prompt[pid])))

    # 1. Update default_text_prompt_id
    text = re.sub(
        r'(default_text_prompt_id:\s*)"?[\w_]+"?',
        rf'\1"{overall_best}"',
        text,
    )

    # 2. Replace the calibration.rmse block (preserve indentation)
    rmse = results["calibration_rmse"]
    rmse_block = "\n".join(f"    {mk}: {val}" for mk, val in sorted(rmse.items()))
    text = re.sub(
        r'(  rmse:\n)((?:    \w+:.*\n)+)',
        rf'\1{rmse_block}\n',
        text,
    )

    # 3. Add or replace best_prompts block after rmse
    bp = results["best_prompts"]
    bp_block = "  best_prompts:\n" + "\n".join(f"    {mk}: \"{pid}\"" for mk, pid in sorted(bp.items()))
    if "best_prompts:" in text:
        text = re.sub(
            r'  best_prompts:\n((?:    \w+:.*\n)+)',
            bp_block + "\n",
            text,
        )
    else:
        # Insert after the rmse block (before the next top-level section)
        text = re.sub(
            r'(  rmse:\n(?:    \w+:.*\n)+)',
            rf'\1{bp_block}\n',
            text,
        )

    config_path.write_text(text)

    print(f"  Updated {config_path.name}")
    print(f"    RMSE: {rmse}")
    print(f"    Best prompts: {bp}")
    print(f"    Default prompt: {overall_best}")


# CLI

def resolve_library(library: str) -> Path:
    """Resolve a --library argument; None selects src/prompts/prompt_library.json."""
    if library is None:
        return None
    candidate = Path(library)
    if not candidate.exists():
        raise SystemExit(f"Prompt library not found: {candidate}")
    return candidate


def split_for_institution(conn, university: str) -> Tuple[Dict[str, Any], List, List]:
    """Load an institution's config and its seeded calibration/test split."""
    config_path = PROJECT_ROOT / "src" / "config" / "institutions" / f"{university}.yaml"
    config = load_yaml_config(str(config_path))
    cal_cfg = config.get("calibration", {})
    cal_rows, test_rows = get_calibration_split(
        conn, university,
        seed=cal_cfg.get("seed", 42), fraction=cal_cfg.get("fraction", 0.2),
    )
    return config, cal_rows, test_rows


async def cmd_split(args):
    """Show the calibration/test split for each university."""
    conn = get_connection()
    renderer = PromptRenderer()
    universities = [args.university] if args.university else renderer.list_institutions()

    for uni in universities:
        _config, cal_rows, test_rows = split_for_institution(conn, uni)
        if cal_rows or test_rows:
            print_split_summary(cal_rows, test_rows, uni)

    conn.close()


async def cmd_score(args):
    """Score calibration essays under all model × prompt conditions."""
    conn = get_connection()
    library_path = resolve_library(args.library)
    renderer = PromptRenderer(library_path)
    universities = [args.university] if args.university else renderer.list_institutions()

    for uni in universities:
        config, cal_rows, _test = split_for_institution(conn, uni)
        if not cal_rows:
            print(f"No essays found for {uni}")
            continue

        if args.limit and args.limit < len(cal_rows):
            cal_rows = cal_rows[:args.limit]
            print(f"Limiting to {args.limit} calibration essays")

        print_split_summary(cal_rows, [], uni)
        await score_calibration_set(uni, config, cal_rows, library_path=library_path)

    conn.close()


async def cmd_analyse(args):
    """Analyze calibration scores and update institution configs."""
    conn = get_connection()
    library_path = resolve_library(args.library)
    renderer = PromptRenderer(library_path)
    table = renderer.config.get("calibration_table", "calibration_scores")
    all_universities = renderer.list_institutions()
    universities = [args.university] if args.university else all_universities

    # Prompt selection always pools every institution's calibration split, whichever
    # institutions' configs are being written, so --university narrows the write, not
    # the selection.
    pooled_rows = []
    for uni in all_universities:
        _cfg, cal_rows, _test = split_for_institution(conn, uni)
        pooled_rows.extend(cal_rows)
    if not pooled_rows:
        print("No calibration data found.")
        conn.close()
        return

    # Refuse to derive prompts and weights from a partially scored sweep: doing so
    # silently writes a near-arbitrary selection into every institution YAML.
    scored_ids = {r["essay_id"] for r in load_calibration_scores(pooled_rows, table)[0]}
    coverage = len(scored_ids) / len(pooled_rows)
    if coverage < 0.9:
        raise SystemExit(
            f"Table '{table}' covers {len(scored_ids)}/{len(pooled_rows)} calibration "
            f"essays ({coverage:.0%}). Run `calibrate score` to complete the sweep "
            f"before analysing, or pass --library to analyse a different library."
        )

    selected = select_pooled_prompts(pooled_rows, table=table)
    print(f"\n  Pooled prompt selection ({len(pooled_rows)} calibration essays, "
          f"{len(all_universities)} institutions):")
    for mk, pid in sorted(selected.items()):
        print(f"    {mk:<8} → {pid}")

    for uni in universities:
        config, cal_rows, test_rows = split_for_institution(conn, uni)
        seed = config.get("calibration", {}).get("seed", 42)
        if not cal_rows:
            print(f"No calibration data for {uni}")
            continue

        print_split_summary(cal_rows, test_rows, uni)
        results = analyse_calibration(
            cal_rows, n_bootstrap=args.bootstrap, seed=seed, table=table,
            selected_prompts=selected,
        )
        print_analysis(results, uni)

        if not args.dry_run:
            update_institution_config(uni, results)

        # Optionally save full results to JSON
        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            existing = {}
            if out_path.exists():
                with open(out_path) as f:
                    existing = json.load(f)
            existing[uni] = results
            with open(out_path, "w") as f:
                json.dump(existing, f, indent=2)
            print(f"  Saved results to {out_path}")

    conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Prompt calibration pipeline for OPrAISE essay classifier.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # split
    sp = subparsers.add_parser("split", help="Show calibration/test split")
    sp.add_argument("--university", "-u", default=None, help="Single university (default: all)")

    # score
    sp = subparsers.add_parser("score", help="Score calibration set under all conditions")
    sp.add_argument("--university", "-u", default=None, help="Single university (default: all)")
    sp.add_argument("--limit", "-l", type=int, default=None, help="Score at most N calibration essays")
    sp.add_argument("--library", default=None, help="Path to an alternative prompt library (default: src/prompts/prompt_library.json)")

    # analyse
    sp = subparsers.add_parser("analyse", help="Analyse scores and update configs")
    sp.add_argument("--university", "-u", default=None, help="Single university (default: all)")
    sp.add_argument("--bootstrap", "-b", type=int, default=1000, help="Bootstrap resamples (default: 1000)")
    sp.add_argument("--output", "-o", default=None, help="Save full results to JSON file")
    sp.add_argument("--dry-run", action="store_true", help="Print results without updating configs")
    sp.add_argument("--library", default=None, help="Path to an alternative prompt library (default: src/prompts/prompt_library.json)")

    args = parser.parse_args()

    if args.command == "split":
        asyncio.run(cmd_split(args))
    elif args.command == "score":
        asyncio.run(cmd_score(args))
    elif args.command == "analyse":
        asyncio.run(cmd_analyse(args))


if __name__ == "__main__":
    main()
