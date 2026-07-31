#!/usr/bin/env python3

"""
Test-Retest Reliability

Scores a stratified essay sample and stores it as a session; reliability
metrics (ICC(2,1), mean SD, mean range, band stability) are computed across
all stored sessions.
"""

import argparse
import asyncio
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.classifier.models import BaseModel, ClaudeModel, GptModel, GeminiModel
from src.classifier.prompt_manager import build_model_prompt
from src.classifier.utils import load_yaml_config
from src.prompts.prompt_renderer import PromptRenderer
from src.utils.scoring_utils import ScoringUtils
from src.utils.db_utils import get_connection


MODELS = ("gemini", "gpt", "claude")
# Native provider wrappers and the env var holding each one's model ID, so that
# reruns hit the same endpoints as the main scoring pass.
MODEL_CLASSES = {
    "gemini": (GeminiModel, "GEMINI_MODEL"),
    "gpt":    (GptModel,    "GPT_MODEL"),
    "claude": (ClaudeModel, "CLAUDE_MODEL"),
}


# Scoring

def ensure_table(conn) -> None:
    """Create the reliability_runs table if missing. The `timestamp` column
    uniquely identifies a session; one row per (essay, model, session)."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reliability_runs (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            essay_id   INTEGER NOT NULL,
            model      TEXT NOT NULL,
            run_idx    INTEGER NOT NULL DEFAULT 0,
            score      INTEGER,
            band       TEXT,
            temperature REAL,
            model_id   TEXT,
            timestamp  TEXT NOT NULL,
            FOREIGN KEY(essay_id) REFERENCES essays(id)
        )
    """)
    conn.commit()


def stratified_sample(conn, per_band: int, seed: int = 42) -> List[dict]:
    """Sample N essays per (institution × band) stratum."""
    rows = conn.execute("""
        SELECT id, university, human_grade, assignment_brief, assignment_content
        FROM essays
        WHERE LOWER(essay_type) = 'long'
          AND human_grade IS NOT NULL
          AND assignment_content IS NOT NULL
    """).fetchall()

    rng = random.Random(seed)
    buckets: Dict[tuple, list] = {}
    for r in rows:
        key = (r["university"], ScoringUtils.get_grade_band(int(r["human_grade"])))
        buckets.setdefault(key, []).append(r)

    sampled = []
    for key in sorted(buckets.keys()):
        items = buckets[key]
        rng.shuffle(items)
        sampled.extend(items[:per_band])
    return sampled


def build_prompts(university: str) -> Dict[str, str]:
    """Build each model's best calibrated prompt for one university."""
    task_file = PROJECT_ROOT / "src" / "config" / "institutions" / f"{university}.yaml"
    config = load_yaml_config(str(task_file))
    best_prompts = config.get("calibration", {}).get("best_prompts", {})
    default_id = config.get("prompts", {}).get("default_text_prompt_id", "A1_B3_C1")
    renderer = PromptRenderer()
    return {
        m: build_model_prompt(
            renderer, best_prompts.get(m, default_id), university, config,
            include_feedback=False,
        )
        for m in MODELS
    }


def make_models() -> Dict[str, BaseModel]:
    """Initialise one native provider wrapper per model key."""
    models = {}
    for m, (cls, env) in MODEL_CLASSES.items():
        model_id = os.getenv(env)
        if not model_id:
            raise SystemExit(f"Env var {env} not set.")
        models[m] = cls(model_id, name=m)
    return models


async def run_session(per_band: int, concurrency: int, seed: int) -> str:
    """Score a new session; returns the session timestamp."""
    conn = get_connection()
    ensure_table(conn)
    essays = stratified_sample(conn, per_band=per_band, seed=seed)
    print(f"Sampled {len(essays)} essays across institution × band strata")

    prompts_by_uni: Dict[str, Dict[str, str]] = {}
    for uni in {e["university"].lower() for e in essays}:
        prompts_by_uni[uni] = build_prompts(uni)

    models = make_models()
    session_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    semaphore = asyncio.Semaphore(concurrency)

    async def score_one(essay, model_key: str):
        uni = essay["university"].lower()
        prompt = prompts_by_uni[uni][model_key]
        text = essay["assignment_content"]
        brief = essay["assignment_brief"]
        if brief:
            text = f"=== ASSIGNMENT BRIEF ===\n{brief}\n\n=== STUDENT ESSAY ===\n{text}"
        async with semaphore:
            for attempt in range(3):
                resp = await models[model_key].classify_content({"content": text}, prompt)
                if "error" not in resp:
                    break
                await asyncio.sleep(1.5 * (attempt + 1))
            score = ScoringUtils.extract_grade(resp.get("classification"))
            band = ScoringUtils.get_grade_band(score) if score else None
            conn.execute(
                """INSERT INTO reliability_runs
                   (essay_id, model, run_idx, score, band, temperature, model_id, timestamp)
                   VALUES (?, ?, 0, ?, ?, 0.0, ?, ?)""",
                (essay["id"], model_key, score, band,
                 models[model_key].model_id, session_ts),
            )
            conn.commit()

    tasks = [score_one(e, m) for e in essays for m in MODELS]
    print(f"Dispatching {len(tasks)} API calls (concurrency={concurrency})...")
    t0 = time.time()
    done = 0
    for coro in asyncio.as_completed(tasks):
        await coro
        done += 1
        if done % 25 == 0 or done == len(tasks):
            print(f"  [{done}/{len(tasks)}] elapsed {time.time() - t0:.0f}s")

    conn.close()
    return session_ts


# Reliability metrics across all stored sessions

def icc_2_1(matrix: np.ndarray) -> float:
    """ICC(2,1) two-way random, single measures, absolute agreement."""
    m = np.asarray(matrix, dtype=float)
    n, k = m.shape
    if n < 2 or k < 2:
        return float("nan")
    grand = m.mean()
    row_means = m.mean(axis=1)
    col_means = m.mean(axis=0)
    ss_row = k * np.sum((row_means - grand) ** 2)
    ss_col = n * np.sum((col_means - grand) ** 2)
    ss_err = np.sum((m - grand) ** 2) - ss_row - ss_col
    ms_row = ss_row / (n - 1)
    ms_col = ss_col / (k - 1)
    ms_err = ss_err / ((n - 1) * (k - 1))
    denom = ms_row + (k - 1) * ms_err + k * (ms_col - ms_err) / n
    return float((ms_row - ms_err) / denom) if denom else float("nan")


def compute_reliability(conn) -> Dict:
    """Compute reliability metrics per model across all sessions in the DB.

    For each (essay, model), considers the set of scores across all sessions;
    an essay must have >= 2 sessions scored for that model to contribute.
    """
    rows = conn.execute("""
        SELECT r.essay_id, r.model, r.timestamp, r.run_idx, r.score, r.band, e.university
        FROM reliability_runs r
        JOIN essays e ON e.id = r.essay_id
        WHERE r.score IS NOT NULL
    """).fetchall()

    # A "session key" is (timestamp, run_idx) — handles legacy rows that had
    # multiple run_idx values per timestamp, plus new rows where run_idx=0.
    sessions = sorted({(r["timestamp"], r["run_idx"]) for r in rows})

    # Group: per model → per essay → {session_key: (score, band)}
    by_model: Dict[str, Dict[int, Dict[tuple, tuple]]] = {}
    for r in rows:
        key = (r["timestamp"], r["run_idx"])
        by_model.setdefault(r["model"], {}).setdefault(r["essay_id"], {})[key] = (
            r["score"], r["band"]
        )

    summary = {
        "n_sessions": len(sessions),
        "sessions": [f"{ts} (run_idx={i})" for ts, i in sessions],
        "per_model": {},
    }

    for model, essays in by_model.items():
        # Keep only essays scored in >= 2 sessions
        complete = {eid: v for eid, v in essays.items() if len(v) >= 2}
        if len(complete) < 2:
            summary["per_model"][model] = {"n_essays": len(complete), "note": "insufficient data"}
            continue

        # Use intersection of sessions per essay (aligned matrix)
        n_runs = min(len(v) for v in complete.values())
        scores_matrix = []
        bands_per_essay = []
        for session_scores in complete.values():
            sorted_keys = sorted(session_scores.keys())
            scores_matrix.append([session_scores[k][0] for k in sorted_keys[:n_runs]])
            bands_per_essay.append([session_scores[k][1] for k in sorted_keys[:n_runs]])

        mat = np.array(scores_matrix, dtype=float)
        sds = np.std(mat, axis=1, ddof=1)
        ranges = mat.max(axis=1) - mat.min(axis=1)
        band_stable = sum(1 for bs in bands_per_essay if len(set(bs)) == 1)

        summary["per_model"][model] = {
            "n_essays": int(mat.shape[0]),
            "n_runs_used": int(n_runs),
            "icc_2_1": icc_2_1(mat),
            "mean_sd_score": float(sds.mean()),
            "median_sd_score": float(np.median(sds)),
            "mean_range_score": float(ranges.mean()),
            "max_range_score": float(ranges.max()),
            "pct_band_stable": 100 * band_stable / mat.shape[0],
        }

    return summary


def print_summary(summary: Dict) -> None:
    """Print the reliability summary table."""
    print(f"\n{'=' * 69}")
    print(f"  TEST-RETEST RELIABILITY — {summary['n_sessions']} session(s)")
    print(f"{'=' * 69}")
    for i, s in enumerate(summary["sessions"]):
        print(f"  session {i+1}: {s}")
    print(f"\n{'Model':<10}{'n':>5}{'runs':>6}{'ICC(2,1)':>10}"
          f"{'mean SD':>10}{'mean range':>12}{'% band-stable':>16}")
    print(f"{'-' * 69}")
    for model, s in summary["per_model"].items():
        if s.get("n_essays", 0) < 2 or "note" in s:
            print(f"{model:<10}  {s.get('note', 'no data')}")
            continue
        print(
            f"{model:<10}{s['n_essays']:>5}{s['n_runs_used']:>6}"
            f"{s['icc_2_1']:>10.3f}{s['mean_sd_score']:>10.2f}"
            f"{s['mean_range_score']:>12.2f}{s['pct_band_stable']:>15.1f}%"
        )


# Main

def main():
    parser = argparse.ArgumentParser(description="Test-retest reliability.")
    parser.add_argument("--per-band", type=int, default=3,
                        help="Essays per (institution × band) stratum (default 3)")
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-only", action="store_true",
                        help="Skip scoring; compute metrics from existing runs only")
    parser.add_argument("--output", "--out", type=Path, dest="output",
                        default=Path(__file__).parent / "output" / "test_retest.json",
                        help="Path for the results JSON")
    args = parser.parse_args()

    if not args.report_only:
        asyncio.run(run_session(args.per_band, args.concurrency, args.seed))

    conn = get_connection()
    ensure_table(conn)
    summary = compute_reliability(conn)
    conn.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print_summary(summary)
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
