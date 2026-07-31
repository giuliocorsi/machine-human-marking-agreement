#!/usr/bin/env python3

"""
Prompt-Sensitivity Factorial Analysis

Per-cell calibration RMSE over the 3 × 3 × 3 prompt design (A criterion
specificity, B calibration intervention, C scoring strategy), repeated-measures
ANOVA on absolute error, planned contrasts, and heatmap/main-effects figures.
"""

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.prompts.prompt_renderer import PromptRenderer
from src.utils.db_utils import get_connection


MODELS = ("Claude", "Gemini", "GPT")
LEVELS = {"A": ("A1", "A2", "A3"), "B": ("B1", "B2", "B3"), "C": ("C1", "C2", "C3")}
FACTOR_NAMES = {
    "A": "Criterion specificity",
    "B": "Calibration intervention",
    "C": "Scoring strategy",
}
LEVEL_LABELS = {
    "A1": "No rubric", "A2": "Condensed", "A3": "Full rubric",
    "B1": "None", "B2": "Base rates", "B3": "Base rates\n+ debias",
    "C1": "Direct", "C2": "Analytic", "C3": "Band-first",
}
CELLS = [f"{a}_{b}_{c}" for a in LEVELS["A"] for b in LEVELS["B"] for c in LEVELS["C"]]

# The shipped prompt library names the table its sweep writes to, so the analysis
# follows the library rather than carrying its own copy of that mapping.
DEFAULT_TABLE = PromptRenderer().config.get("calibration_table", "calibration_scores")

# Per-model figure colours
MODEL_COLORS = {"Claude": "#2a78d6", "Gemini": "#008300", "GPT": "#e87ba4"}


# Data loading

def load_design(table: str):
    """Return err[model][essay, i, j, k] (signed error) plus essay metadata.
    Complete cases only, so every ANOVA runs on a balanced design."""
    conn = get_connection()
    rows = conn.execute(f"""
        SELECT cs.essay_id, cs.model, cs.prompt_id, cs.score,
               e.human_grade, e.university
        FROM {table} cs JOIN essays e ON e.id = cs.essay_id
        WHERE cs.score IS NOT NULL
    """).fetchall()
    conn.close()
    if not rows:
        sys.exit(f"No rows in {table} — has this library been scored yet?")

    scores: Dict[Tuple[int, str, str], float] = {}
    meta: Dict[int, Dict] = {}
    for r in rows:
        scores[(r["essay_id"], r["model"], r["prompt_id"])] = float(r["score"])
        meta[r["essay_id"]] = {
            "human": float(r["human_grade"]), "university": r["university"]
        }

    complete = [
        eid for eid in sorted(meta)
        if all((eid, m, pid) in scores for m in MODELS for pid in CELLS)
    ]
    dropped = sorted(set(meta) - set(complete))

    err = {}
    for m in MODELS:
        arr = np.empty((len(complete), 3, 3, 3))
        for s, eid in enumerate(complete):
            h = meta[eid]["human"]
            for i, a in enumerate(LEVELS["A"]):
                for j, b in enumerate(LEVELS["B"]):
                    for k, c in enumerate(LEVELS["C"]):
                        arr[s, i, j, k] = scores[(eid, m, f"{a}_{b}_{c}")] - h
        err[m] = arr

    unis = np.array([meta[eid]["university"] for eid in complete])
    return err, unis, complete, dropped


# Repeated-measures ANOVA

def rm_anova(y: np.ndarray, factor_names: List[str]) -> Dict[str, Dict]:
    """Exact RM-ANOVA for a balanced fully-within design.
    y: array (subjects, l1, ..., lm); each effect is tested against its
    effect × subject interaction via an inclusion-exclusion SS decomposition."""
    dims = list(range(y.ndim))          # 0 = subjects
    n_levels = y.shape
    grand = y.mean()

    def margin_mean(keep: Tuple[int, ...]) -> np.ndarray:
        drop = tuple(d for d in dims if d not in keep)
        return y.mean(axis=drop, keepdims=False)

    def effect_ss(keep: Tuple[int, ...]) -> float:
        """SS for the interaction effect on exactly the dims in `keep`."""
        est = np.zeros([n_levels[d] for d in keep]) if keep else np.array(grand)
        for sub in itertools.chain.from_iterable(
            itertools.combinations(keep, r) for r in range(len(keep) + 1)
        ):
            mm = margin_mean(sub)
            # broadcast mm over the kept dims
            shape = [n_levels[d] if d in sub else 1 for d in keep]
            sign = (-1) ** (len(keep) - len(sub))
            est = est + sign * mm.reshape(shape)
        reps = y.size / est.size if est.size else y.size
        return float(reps * np.sum(est ** 2))

    within = dims[1:]
    results = {}
    ss_total = float(np.sum((y - grand) ** 2))
    ss_sum = effect_ss((0,))            # subjects
    for r in range(1, len(within) + 1):
        for combo in itertools.combinations(within, r):
            ss_e = effect_ss(combo)
            ss_err = effect_ss((0,) + combo)
            df_e = int(np.prod([n_levels[d] - 1 for d in combo]))
            df_err = df_e * (n_levels[0] - 1)
            ms_e, ms_err = ss_e / df_e, ss_err / df_err
            f = ms_e / ms_err
            p = float(stats.f.sf(f, df_e, df_err))
            # lower-bound sphericity correction (epsilon = 1/df_e)
            p_lb = float(stats.f.sf(f, 1, n_levels[0] - 1))
            name = " × ".join(factor_names[d - 1] for d in combo)
            results[name] = {
                "df": (df_e, df_err), "F": f, "p": p, "p_lower_bound": p_lb,
                "partial_eta_sq": ss_e / (ss_e + ss_err),
            }
            ss_sum += ss_e + ss_err
    assert abs(ss_sum - ss_total) / ss_total < 1e-9, "SS decomposition failed"
    return results


def holm(pvals: List[float]) -> List[float]:
    """Holm step-down adjusted p-values for one family of tests."""
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return [float(p) for p in adj]


# Descriptives

def cell_rmse(err: np.ndarray) -> np.ndarray:
    """RMSE per cell, shape (3,3,3)."""
    return np.sqrt(np.mean(err ** 2, axis=0))


def cell_id(idx: Tuple[int, int, int]) -> str:
    """Convert a (i, j, k) cell index to its A?_B?_C? identifier."""
    i, j, k = idx
    return f"{LEVELS['A'][i]}_{LEVELS['B'][j]}_{LEVELS['C'][k]}"


def describe_cell(cid: str) -> str:
    """Human-readable description of a cell identifier."""
    a, b, c = cid.split("_")
    return (f"{LEVEL_LABELS[a]} / {LEVEL_LABELS[b]} / "
            f"{LEVEL_LABELS[c]}").replace("\n", " ")


def paired_contrast(diff: np.ndarray) -> Dict:
    """Paired t and Wilcoxon on per-essay differences."""
    t = stats.ttest_rel(diff, np.zeros_like(diff))
    try:
        w = stats.wilcoxon(diff)
        w_p = float(w.pvalue)
    except ValueError:
        w_p = float("nan")
    sd = diff.std(ddof=1)
    return {
        "mean_diff": float(diff.mean()),
        "t": float(t.statistic), "df": len(diff) - 1, "p": float(t.pvalue),
        "wilcoxon_p": w_p,
        "cohen_dz": float(diff.mean() / sd) if sd > 0 else float("nan"),
    }


# Figures

def fig_heatmaps(rmse: Dict[str, np.ndarray], best: Dict[str, Tuple], path: Path) -> None:
    """Per-model cell-RMSE heatmaps with the best cell boxed."""
    vmin = min(m.min() for m in rmse.values())
    vmax = max(m.max() for m in rmse.values())
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 5.6), dpi=300)
    row_labels = [
        f"{LEVEL_LABELS[a]} · {LEVEL_LABELS[b]}".replace("\n", " ")
        for a in LEVELS["A"] for b in LEVELS["B"]
    ]
    col_labels = [LEVEL_LABELS[c] for c in LEVELS["C"]]

    for ax, model in zip(axes, MODELS):
        mat = rmse[model].reshape(9, 3)
        im = ax.imshow(mat, cmap="Blues", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(model, fontsize=11, pad=10)
        ax.set_xticks(range(3), col_labels, fontsize=8.5)
        if model == MODELS[0]:
            ax.set_yticks(range(9), row_labels, fontsize=8.5)
        else:
            ax.set_yticks(range(9), [""] * 9)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        thresh = vmin + 0.6 * (vmax - vmin)
        for r in range(9):
            for c in range(3):
                ax.text(c, r, f"{mat[r, c]:.1f}", ha="center", va="center",
                        fontsize=8, color="white" if mat[r, c] > thresh else "#1a1a1a")
        bi, bj, bk = best[model]
        # white spacers between A blocks; best-cell box drawn above them
        for r in (2.5, 5.5):
            ax.axhline(r, color="white", lw=2)
        ax.add_patch(plt.Rectangle((bk - 0.5, bi * 3 + bj - 0.5), 1, 1,
                                   fill=False, edgecolor="#1a1a1a", lw=2,
                                   zorder=5))

    fig.suptitle("Calibration RMSE across the 3 × 3 × 3 prompt design "
                 "(pooled calibration set; box = best cell)",
                 fontsize=11, y=0.99)
    cbar = fig.colorbar(im, ax=axes, shrink=0.75, pad=0.02)
    cbar.set_label("RMSE (grade points)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_main_effects(err: Dict[str, np.ndarray], path: Path) -> None:
    """Marginal RMSE per factor level, one line per model."""
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), dpi=300)
    for ax, (fkey, axis) in zip(axes, [("A", 1), ("B", 2), ("C", 3)]):
        endpoints = {}
        for model in MODELS:
            e = err[model]
            marginal = [
                float(np.sqrt(np.mean(np.take(e, lv, axis=axis) ** 2)))
                for lv in range(3)
            ]
            ax.plot(range(3), marginal, marker="o", markersize=7, lw=2,
                    color=MODEL_COLORS[model], label=model)
            endpoints[model] = marginal[2]
        # direct labels at right end, dodged apart if endpoints nearly coincide
        min_gap = 0.09 * (max(endpoints.values()) - min(endpoints.values()) + 1e-9)
        min_gap = max(min_gap, 0.18)
        ordered = sorted(endpoints.items(), key=lambda kv: kv[1])
        label_y = {}
        prev = -np.inf
        for model, v in ordered:
            y_pos = max(v, prev + min_gap)
            label_y[model] = y_pos
            prev = y_pos
        for model in MODELS:
            ax.annotate(model, (2, label_y[model]), xytext=(8, 0),
                        textcoords="offset points", fontsize=8.5,
                        color=MODEL_COLORS[model], va="center")
        ax.set_title(FACTOR_NAMES[fkey], fontsize=10)
        ax.set_xticks(range(3), [LEVEL_LABELS[l] for l in LEVELS[fkey]],
                      fontsize=8.5)
        ax.set_xlim(-0.35, 2.85)
        ax.grid(axis="y", color="#e3e3e3", lw=0.7)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=8.5)
    axes[0].set_ylabel("Marginal RMSE (grade points)", fontsize=9)
    ymin = min(ax.get_ylim()[0] for ax in axes)
    ymax = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(ymin, ymax)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# Main

def main():
    parser = argparse.ArgumentParser(description="Prompt-sensitivity factorial analysis.")
    parser.add_argument("--table", default=DEFAULT_TABLE,
                        help=f"Calibration table to analyse (default {DEFAULT_TABLE}, "
                             "from the shipped prompt library)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).parent / "output",
                        help="Directory for results JSON and figures")
    args = parser.parse_args()
    table = args.table
    output_dir = args.output_dir

    err, unis, essay_ids, dropped = load_design(table)
    n = len(essay_ids)
    print(f"Table {table}: {n} complete-case essays "
          f"({dropped and f'dropped {len(dropped)}: {dropped}' or 'none dropped'})")
    for u in sorted(set(unis)):
        print(f"  {u}: {int(np.sum(unis == u))}")

    out: Dict = {"table": table, "n_essays": n,
                 "dropped_essay_ids": dropped}

    # 1. Cell RMSE, best/worst cells
    rmse = {m: cell_rmse(err[m]) for m in MODELS}
    best_idx = {m: np.unravel_index(np.argmin(rmse[m]), (3, 3, 3)) for m in MODELS}
    out["cells"] = {}
    print(f"\n{'-' * 88}")
    print("  Cell RMSE (pooled calibration set)")
    print(f"{'-' * 88}")
    for m in MODELS:
        flat = rmse[m].flatten()
        bi = best_idx[m]
        wi = np.unravel_index(np.argmax(rmse[m]), (3, 3, 3))
        out["cells"][m] = {
            "best_cell": cell_id(bi), "best_rmse": float(rmse[m][bi]),
            "worst_cell": cell_id(wi), "worst_rmse": float(rmse[m][wi]),
            "spread": float(rmse[m][wi] - rmse[m][bi]),
            "matrix": {cid: float(v) for cid, v in zip(CELLS, flat)},
        }
        print(f"  {m:<8} best {cell_id(bi)} = {rmse[m][bi]:.2f}  "
              f"({describe_cell(cell_id(bi))})")
        print(f"  {'':<8} worst {cell_id(wi)} = {rmse[m][wi]:.2f}  "
              f"({describe_cell(cell_id(wi))})   Δ = {rmse[m][wi]-rmse[m][bi]:.2f}")

    # 2. Per-institution best cells (stability of the optimum)
    print(f"\n{'-' * 88}")
    print("  Best cell per institution")
    print(f"{'-' * 88}")
    out["per_institution_best"] = {}
    for m in MODELS:
        out["per_institution_best"][m] = {}
        for u in sorted(set(unis)):
            r = cell_rmse(err[m][unis == u])
            bi = np.unravel_index(np.argmin(r), (3, 3, 3))
            out["per_institution_best"][m][u] = {
                "cell": cell_id(bi), "rmse": float(r[bi]),
                "n": int(np.sum(unis == u)),
            }
            print(f"  {m:<8} {u:<12} {cell_id(bi)}  (RMSE {r[bi]:.2f})")

    # 2b. Cross-institution transfer of the optimum: Spearman correlations
    # between institutions' 27-cell RMSE profiles, and the rank of the
    # pooled-best cell within each institution.
    print(f"\n{'-' * 88}")
    print("  Cross-institution transfer of the optimum")
    print(f"{'-' * 88}")
    uni_names = sorted(set(unis))
    out["profile_correlations"] = {}
    out["pooled_best_rank"] = {}
    all_rho = []
    for m in MODELS:
        profiles = {u: cell_rmse(err[m][unis == u]).flatten() for u in uni_names}
        out["profile_correlations"][m] = {}
        for u1, u2 in itertools.combinations(uni_names, 2):
            rho = float(stats.spearmanr(profiles[u1], profiles[u2]).statistic)
            out["profile_correlations"][m][f"{u1}-{u2}"] = rho
            all_rho.append(rho)
            print(f"  {m:<8} {u1[:4]}-{u2[:4]} ρ = {rho:+.2f}")
        pooled_best_flat = int(np.argmin(rmse[m]))
        out["pooled_best_rank"][m] = {
            u: int(stats.rankdata(profiles[u], method="min")[pooled_best_flat])
            for u in uni_names
        }
        ranks = ", ".join(f"{u[:4]} {r}" for u, r in out["pooled_best_rank"][m].items())
        print(f"  {'':<8} pooled best {cell_id(best_idx[m])} ranks: {ranks} of 27")
    out["profile_correlations"]["summary"] = {
        "min": min(all_rho), "max": max(all_rho),
        "median": float(np.median(all_rho)),
    }
    print(f"  overall: range {min(all_rho):+.2f} to {max(all_rho):+.2f}, "
          f"median {np.median(all_rho):+.2f}")

    # 3. RM-ANOVA on absolute error
    print(f"\n{'-' * 88}")
    print("  Repeated-measures ANOVA on |error| (per model)")
    print(f"{'-' * 88}")
    out["anova_per_model"] = {
        m: rm_anova(np.abs(err[m]), ["A", "B", "C"]) for m in MODELS
    }
    # Holm correction over the 9 model × factor main effects (one family:
    # "which model is sensitive to which property"); interactions uncorrected.
    main_keys = [(m, f) for m in MODELS for f in ("A", "B", "C")]
    adj = holm([out["anova_per_model"][m][f]["p"] for m, f in main_keys])
    for (m, f), p in zip(main_keys, adj):
        out["anova_per_model"][m][f]["p_holm"] = p
    for m in MODELS:
        print(f"  {m}:")
        for name, r in out["anova_per_model"][m].items():
            ph = f"  p_holm = {r['p_holm']:.2e}" if "p_holm" in r else ""
            print(f"    {name:<12} F({r['df'][0]},{r['df'][1]}) = {r['F']:7.2f}  "
                  f"p = {r['p']:.2e}  p_lb = {r['p_lower_bound']:.2e}  "
                  f"η²p = {r['partial_eta_sq']:.3f}{ph}")

    print(f"\n{'-' * 88}")
    print("  Combined ANOVA with Model as a within factor")
    print(f"{'-' * 88}")
    y_all = np.stack([np.abs(err[m]) for m in MODELS], axis=1)  # (n, model, A, B, C)
    res_all = rm_anova(y_all, ["Model", "A", "B", "C"])
    out["anova_combined"] = res_all
    for name, r in res_all.items():
        print(f"  {name:<22} F({r['df'][0]},{r['df'][1]}) = {r['F']:8.2f}  "
              f"p = {r['p']:.2e}  p_lb = {r['p_lower_bound']:.2e}  "
              f"η²p = {r['partial_eta_sq']:.3f}")

    # 4. Planned contrasts
    print(f"\n{'-' * 88}")
    print("  Planned paired contrasts (per essay, averaged over other factors)")
    print(f"{'-' * 88}")
    out["contrasts"] = {}
    for m in MODELS:
        e, a = err[m], np.abs(err[m])
        cons = {
            "A1_vs_A3_abs":  a[:, 0].mean(axis=(1, 2)) - a[:, 2].mean(axis=(1, 2)),
            "B3_vs_B2_abs":  a[:, :, 2].mean(axis=(1, 2)) - a[:, :, 1].mean(axis=(1, 2)),
            "B2_vs_B1_abs":  a[:, :, 1].mean(axis=(1, 2)) - a[:, :, 0].mean(axis=(1, 2)),
            "B3_vs_B2_signed": e[:, :, 2].mean(axis=(1, 2)) - e[:, :, 1].mean(axis=(1, 2)),
            "B2_vs_B1_signed": e[:, :, 1].mean(axis=(1, 2)) - e[:, :, 0].mean(axis=(1, 2)),
        }
        out["contrasts"][m] = {k: paired_contrast(d) for k, d in cons.items()}
    # Holm correction over the 15 planned contrasts as one family
    con_keys = [(m, k) for m in MODELS for k in out["contrasts"][m]]
    adj = holm([out["contrasts"][m][k]["p"] for m, k in con_keys])
    for (m, k), p in zip(con_keys, adj):
        out["contrasts"][m][k]["p_holm"] = p
    for m in MODELS:
        print(f"  {m}:")
        for k, r in out["contrasts"][m].items():
            print(f"    {k:<16} Δ = {r['mean_diff']:+.2f}  t({r['df']}) = "
                  f"{r['t']:+.2f}  p = {r['p']:.2e}  p_holm = {r['p_holm']:.2e}  "
                  f"dz = {r['cohen_dz']:+.2f}")

    # Marginal RMSE by factor level (for the text)
    out["marginal_rmse"] = {}
    for m in MODELS:
        out["marginal_rmse"][m] = {}
        for fkey, axis in (("A", 1), ("B", 2), ("C", 3)):
            out["marginal_rmse"][m][fkey] = {
                LEVELS[fkey][lv]: float(np.sqrt(np.mean(
                    np.take(err[m], lv, axis=axis) ** 2)))
                for lv in range(3)
            }

    # 5. Figures and results JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    heatmap_path = output_dir / f"cell_rmse_heatmap_{table}.png"
    effects_path = output_dir / f"factor_main_effects_{table}.png"
    fig_heatmaps(rmse, best_idx, heatmap_path)
    fig_main_effects(err, effects_path)

    json_path = output_dir / f"factorial_results_{table}.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults written to {json_path}")
    print(f"Figures: {heatmap_path}, {effects_path}")


if __name__ == "__main__":
    main()
