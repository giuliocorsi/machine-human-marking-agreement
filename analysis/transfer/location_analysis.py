#!/usr/bin/env python3

"""
Grade-Location Analysis

Why agreement does not transfer across institutions: human vs predicted
distributions, central-tendency compression, error by grade region, and a
Kitagawa decomposition of the MAE gap vs Cambridge.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.db_utils import get_connection
from src.utils.calibration_utils import get_all_calibration_ids
from src.utils.analysis_utils import INSTITUTIONS, load_complete_case_data


UNIS = tuple(INSTITUTIONS.values())
MODELS = ("gemini", "gpt", "claude")
REGIONS = [(0, 50, "<50"), (50, 60, "50-59"), (60, 65, "60-64"),
           (65, 70, "65-69"), (70, 101, "70+")]
BANDS = [(0, 50, "<50"), (50, 60, "50s"), (60, 70, "60s"), (70, 101, "70+")]

# Institution palette (density-figure style)
UNI_COLORS = {"Cambridge": "#2D8B75", "MMU": "#D4842A", "Nottingham": "#3B4CC0"}
ZONE = (60, 70)          # zone of validity shaded in both panels
ZONE_COLOR = "#ededed"
THIN_N = 10              # bands with n below this get open markers in panel b

# Density-figure style shared by the standalone transfer figures
BG, GRID_C = "#FAFBFD", "#E2E6EE"
TEXT_C, MUTED = "#1A1E2E", "#6B7394"
STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 15, "axes.labelsize": 16,
    "axes.titlesize": 17, "axes.titleweight": "600",
    "axes.labelweight": "500", "axes.labelcolor": TEXT_C,
    "xtick.labelsize": 14, "ytick.labelsize": 14,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "legend.fontsize": 13, "legend.framealpha": 0.0, "legend.edgecolor": "none",
    "figure.dpi": 300, "figure.facecolor": BG, "axes.facecolor": "white",
    "axes.edgecolor": GRID_C, "axes.linewidth": 0.8, "axes.grid": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "text.color": TEXT_C, "savefig.facecolor": BG,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.3,
}


# Descriptives

def band_of(grade: float) -> str:
    """Grade-band label for a human mark."""
    for lo, hi, lab in BANDS:
        if lo <= grade < hi:
            return lab
    return "70+"


def kitagawa(data: Dict, target: str, reference: str = "Cambridge",
             edges=None) -> Dict:
    """Kitagawa two-fold decomposition of the MAE gap (target − reference)
    into band-composition and within-band-rate parts, with midpoint weights."""
    if edges is None:
        cuts = [(lo, hi) for lo, hi, _ in BANDS]
    else:
        cuts = list(zip(edges[:-1], edges[1:]))

    def profile(u):
        h, p = data[u]["human"], data[u]["pred"]
        ae = np.abs(p - h)
        w, m = [], []
        for lo, hi in cuts:
            mask = (h >= lo) & (h < hi)
            w.append(mask.mean())
            m.append(ae[mask].mean() if mask.sum() else np.nan)
        return np.array(w), np.array(m)

    w_t, m_t = profile(target)
    w_r, m_r = profile(reference)
    m_bar = np.nanmean(np.vstack([m_t, m_r]), axis=0)
    w_bar = (w_t + w_r) / 2
    rate_diff = np.where(np.isnan(m_t) | np.isnan(m_r), 0.0, m_t - m_r)

    composition = float(np.nansum((w_t - w_r) * m_bar))
    rate = float(np.nansum(w_bar * rate_diff))
    gap = float(np.nansum(w_t * np.nan_to_num(m_t))
                - np.nansum(w_r * np.nan_to_num(m_r)))
    # one-sided counterfactuals
    comp_at_target_rates = float(np.nansum((w_t - w_r) * np.nan_to_num(m_t)))
    comp_at_ref_rates = float(np.nansum((w_t - w_r) * np.nan_to_num(m_r)))
    return {
        "gap": gap, "composition": composition, "rate": rate,
        "residual_interaction": gap - composition - rate,
        "composition_share": composition / gap if gap else float("nan"),
        "composition_at_target_rates": comp_at_target_rates,
        "composition_at_reference_rates": comp_at_ref_rates,
    }


def band_profile(h: np.ndarray, p: np.ndarray):
    """Band weights and within-band MAEs over BANDS."""
    ae = np.abs(p - h)
    w, m = [], []
    for lo, hi, _ in BANDS:
        mask = (h >= lo) & (h < hi)
        w.append(mask.mean())
        m.append(ae[mask].mean() if mask.sum() else 0.0)
    return np.array(w), np.array(m)


# Figures

def v_fmt(v: float) -> str:
    """Signed one-decimal label with a proper minus sign."""
    return f"{v:+.1f}".replace("-", "−")


def make_figure(data: Dict, attractor: float, kit: Dict, path: Path) -> None:
    """Fig. 2 — a: human-mark histograms; b: signed error by grade band;
    c: Kitagawa decomposition of the MAE gap."""
    fig = plt.figure(figsize=(10.5, 6.4), dpi=300)
    outer = fig.add_gridspec(1, 2, width_ratios=[1, 1.25], wspace=0.24)
    gs_left = outer[0, 0].subgridspec(3, 1, hspace=0.38)
    gs_right = outer[0, 1].subgridspec(3, 1, height_ratios=[2.35, 0.16, 1.15],
                                       hspace=0.42)

    # Panel a — stacked histograms
    bins = np.arange(25, 101, 5)
    hist_axes = []
    for row, u in enumerate(UNIS):
        ax = fig.add_subplot(gs_left[row],
                             sharex=hist_axes[0] if hist_axes else None)
        hist_axes.append(ax)
        h = data[u]["human"]
        ax.axvspan(*ZONE, color=ZONE_COLOR, zorder=0)
        ax.hist(h, bins=bins, color=UNI_COLORS[u], edgecolor="white",
                linewidth=0.6, zorder=2)
        ax.axvline(attractor, color="#1a1a1a", lw=1.1, ls=(0, (4, 2)), zorder=3)
        ax.text(0.02, 0.83, f"{u}  (n = {len(h)})", transform=ax.transAxes,
                fontsize=9, color=UNI_COLORS[u], fontweight="bold")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=8)
        if row < 2:
            plt.setp(ax.get_xticklabels(), visible=False)
    hist_axes[0].set_title(f"Human marks, machine attractor ({attractor:.1f}, dashed)\n"
                           f"and zone of validity ({ZONE[0]}–{ZONE[1]-1}, shaded)",
                           fontsize=9.5, loc="left", pad=8)
    hist_axes[-1].set_xlabel("Human mark", fontsize=9)
    hist_axes[1].set_ylabel("Essays", fontsize=9)
    hist_axes[0].text(-0.1, 1.32, "a", transform=hist_axes[0].transAxes,
                      fontsize=13, fontweight="bold")

    # Panel b — signed error by institution × band
    axb = fig.add_subplot(gs_right[0])
    zone_x = next(i for i, (_, _, lab) in enumerate(BANDS) if lab == "60s")
    axb.axvspan(zone_x - 0.5, zone_x + 0.5, color=ZONE_COLOR, zorder=0)
    axb.axhline(0, color="#1a1a1a", lw=0.9, zorder=1)
    dodge = {"Cambridge": -0.16, "MMU": 0.0, "Nottingham": 0.16}
    endpoints = {}
    for u in UNIS:
        h, p = data[u]["human"], data[u]["pred"]
        e = p - h
        xs, ys, cis, ns = [], [], [], []
        for i, (lo, hi, _) in enumerate(BANDS):
            m = (h >= lo) & (h < hi)
            if not m.sum():
                continue
            vals = e[m]
            ci = (stats.t.ppf(0.975, len(vals) - 1) * vals.std(ddof=1)
                  / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            xs.append(i + dodge[u]); ys.append(vals.mean())
            cis.append(ci); ns.append(int(m.sum()))
        color = UNI_COLORS[u]
        axb.plot(xs, ys, color=color, lw=1.8, zorder=3)
        for x, y, ci, n_band in zip(xs, ys, cis, ns):
            open_marker = n_band < THIN_N
            axb.errorbar(x, y, yerr=ci, color=color, lw=1.4, capsize=2.5,
                         zorder=4, fmt="none")
            axb.plot(x, y, "o", markersize=7, color=color, zorder=5,
                     markerfacecolor="white" if open_marker else color,
                     markeredgewidth=1.6, markeredgecolor=color)
            axb.annotate(f"{n_band}", (x, y + ci), xytext=(0, 4),
                         textcoords="offset points", fontsize=7,
                         color=color, ha="center")
        endpoints[u] = ys[-1]
    # direct labels at right end, dodged apart vertically
    min_gap = 1.6
    prev = -np.inf
    label_y = {}
    for u, v in sorted(endpoints.items(), key=lambda kv: kv[1]):
        y_pos = max(v, prev + min_gap)
        label_y[u] = y_pos
        prev = y_pos
    for u in UNIS:
        axb.annotate(u, (len(BANDS) - 1 + dodge[u], label_y[u]), xytext=(12, 0),
                     textcoords="offset points", fontsize=8.5,
                     color=UNI_COLORS[u], va="center")
    axb.set_xticks(range(len(BANDS)),
                   ["Below 50", "50–59", "60–69", "70 and above"], fontsize=8.5)
    axb.set_xlim(-0.6, len(BANDS) - 0.4 + 0.85)
    axb.set_ylabel("Mean signed error, ensemble − human (grade points)", fontsize=9)
    axb.set_title("Ensemble error by human grade band (95% CI; open = n < 10)",
                  fontsize=9.5, loc="left", pad=8)
    axb.grid(axis="y", color="#e3e3e3", lw=0.7)
    axb.set_axisbelow(True)
    for s in ("top", "right"):
        axb.spines[s].set_visible(False)
    axb.tick_params(labelsize=8.5)
    axb.text(-0.09, 1.075, "b", transform=axb.transAxes,
             fontsize=13, fontweight="bold")

    # Panel c — Kitagawa decomposition of the MAE gap vs Cambridge
    axc = fig.add_subplot(gs_right[2])
    axc.axvline(0, color="#1a1a1a", lw=0.9, zorder=1)
    bar_h = 0.32
    y_pos = {"MMU": (1.0 + bar_h / 2 + 0.03, 1.0 - bar_h / 2 - 0.03),
             "Nottingham": (0.0 + bar_h / 2 + 0.03, 0.0 - bar_h / 2 - 0.03)}
    for target in ("MMU", "Nottingham"):
        k = kit[target]["bands_4"]
        color = UNI_COLORS[target]
        y_comp, y_rate = y_pos[target]
        axc.barh(y_comp, k["composition"], height=bar_h, facecolor="white",
                 edgecolor=color, hatch="////", lw=1.2, zorder=3)
        axc.barh(y_rate, k["rate"], height=bar_h, facecolor=color,
                 edgecolor="white", lw=0.6, zorder=3)
        share = (f" ({k['composition_share']*100:.0f}%)"
                 if abs(k["gap"]) > 0.5 else "")
        rate_share = (f" ({(1 - k['composition_share'])*100:.0f}%)"
                      if abs(k["gap"]) > 0.5 else "")
        for y, v, label in ((y_comp, k["composition"], f"composition {v_fmt(k['composition'])}{share}"),
                            (y_rate, k["rate"], f"within-band accuracy {v_fmt(k['rate'])}{rate_share}")):
            ha = "left" if v >= 0 else "right"
            off = (5, 0) if v >= 0 else (-5, 5)
            axc.annotate(label, (v, y), xytext=off,
                         textcoords="offset points", fontsize=7.5,
                         color="#1a1a1a", va="center", ha=ha)
    axc.set_yticks([1.0, 0.0],
                   [f"MMU − Cambridge\n(gap {kit['MMU']['bands_4']['gap']:.1f})",
                    f"Nottingham − Cambridge\n(gap {kit['Nottingham']['bands_4']['gap']:.1f})"],
                   fontsize=8.5)
    axc.set_ylim(-0.65, 1.65)
    axc.set_xlim(-1.85, 4.35)
    axc.set_xlabel("Contribution to MAE gap vs Cambridge (grade points)", fontsize=9)
    axc.set_title("Decomposition of the accuracy gap: cohort composition "
                  "vs within-band accuracy", fontsize=9.5, loc="left", pad=12)
    axc.grid(axis="x", color="#e3e3e3", lw=0.7)
    axc.set_axisbelow(True)
    for s in ("top", "right"):
        axc.spines[s].set_visible(False)
    axc.tick_params(labelsize=8.5)
    axc.text(-0.09, 1.18, "c", transform=axc.transAxes,
             fontsize=13, fontweight="bold")

    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def zone_and_attractor(ax, attractor: float) -> None:
    """Shade the zone of validity and mark the machine attractor."""
    ax.axvspan(*ZONE, color=GRID_C, alpha=0.45, zorder=0)
    ax.axvline(attractor, color=MUTED, lw=1.6, ls=(0, (5, 3)), zorder=1)


def make_fig_distributions(data: Dict, attractor: float, path: Path) -> None:
    """Human grade densities per institution, with the models' central tendency."""
    from scipy.stats import gaussian_kde

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(11, 4.8))
        grid = np.linspace(25, 100, 400)
        zone_and_attractor(ax, attractor)
        for u in UNIS:
            h = data[u]["human"]
            kde = gaussian_kde(h)
            ax.plot(grid, kde(grid), color=UNI_COLORS[u], lw=2.6, zorder=3,
                    label=f"{u} (n = {len(h)})")
            ax.fill_between(grid, kde(grid), color=UNI_COLORS[u],
                            alpha=0.10, zorder=2)
        ax.set_ylabel("Density")
        ax.set_xlabel("Human grade")
        ax.set_xlim(25, 100)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.12)   # headroom for the label
        ax.set_title("Human grade distributions relative to the models' "
                     "central tendency", loc="left", pad=12)
        ax.legend(loc="upper left", handlelength=1.6)
        ax.annotate(f"models' central tendency ({attractor:.1f})",
                    (attractor, ax.get_ylim()[1] * 0.97), xytext=(10, 0),
                    textcoords="offset points", ha="left", va="top",
                    fontsize=12.5, color=MUTED)
        fig.savefig(path)
        plt.close(fig)


def make_fig_counterfactual(data: Dict, path: Path) -> None:
    """Counterfactual swaps: MMU's MAE with Cambridge's cohort mix vs accuracy."""
    w_m, m_m = band_profile(data["MMU"]["human"], data["MMU"]["pred"])
    w_c, m_c = band_profile(data["Cambridge"]["human"], data["Cambridge"]["pred"])
    camb, mmu = UNI_COLORS["Cambridge"], UNI_COLORS["MMU"]
    rows = [  # label, value, facecolor, hatch
        ("MMU (actual)", float(w_m @ m_m), mmu, None),
        ("MMU with Cambridge's\ncohort mix", float(w_c @ m_m), mmu, "////"),
        ("MMU with Cambridge's\nwithin-band accuracy", float(w_m @ m_c), camb, "////"),
        ("Cambridge (actual)", float(w_c @ m_c), camb, None),
    ]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(11, 4.2))
        ys = np.arange(len(rows))[::-1]
        for y, (label, v, color, hatch) in zip(ys, rows):
            if hatch:
                ax.barh(y, v, height=0.62, facecolor="white", edgecolor=color,
                        hatch=hatch, lw=1.6, zorder=3)
            else:
                ax.barh(y, v, height=0.62, color=color, zorder=3)
            ax.annotate(f"{v:.1f}", (v, y), xytext=(9, 0),
                        textcoords="offset points", va="center",
                        fontsize=15, fontweight="600", color=TEXT_C)
        ax.set_yticks(ys, [r[0] for r in rows])
        ax.set_xlabel("MAE (grade points)")
        ax.set_xlim(0, 9.6)
        ax.set_title("Swapping the cohort does little; swapping within-band "
                     "accuracy closes most of the gap", loc="left", pad=12)
        ax.xaxis.grid(True, color=GRID_C, lw=0.8)
        ax.set_axisbelow(True)
        fig.savefig(path)
        plt.close(fig)


def make_fig_error_by_grade(data: Dict, attractor: float, path: Path) -> None:
    """Mean signed ensemble error by human grade (5-point bins), per institution."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(11, 4.8))
        zone_and_attractor(ax, attractor)
        for u in UNIS:
            h, p = data[u]["human"], data[u]["pred"]
            e = p - h
            xs, ys = [], []
            for lo in range(25, 100, 5):
                m = (h >= lo) & (h < lo + 5)
                if m.sum() >= 5:
                    xs.append(lo + 2.5)
                    ys.append(e[m].mean())
            ax.plot(xs, ys, color=UNI_COLORS[u], lw=2.6, zorder=3,
                    marker="o", markersize=6.5, markeredgecolor="white",
                    markeredgewidth=1.0, label=f"{u} (n = {len(h)})")
        ax.axhline(0, color=TEXT_C, lw=1.0, zorder=1)
        ax.set_ylabel("Mean signed error\n(ensemble − human, grade points)")
        ax.set_xlabel("Human grade")
        ax.set_xlim(25, 100)
        ax.set_title("The shared compression: signed error by human grade",
                     loc="left", pad=12)
        ax.legend(loc="upper right", handlelength=1.6)
        ax.annotate("over-marked", (27, 4.5), fontsize=12.5, color=MUTED)
        ax.annotate("under-marked", (27, -6.5), fontsize=12.5, color=MUTED)
        fig.savefig(path)
        plt.close(fig)


# Main

def main():
    parser = argparse.ArgumentParser(description="Grade-location / transfer analysis.")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).parent / "output",
                        help="Directory for results JSON and figures")
    args = parser.parse_args()
    output_dir = args.output_dir

    conn = get_connection()
    exclude_ids, _ = get_all_calibration_ids(conn)
    raw = load_complete_case_data(conn, exclude_ids=exclude_ids)
    conn.close()

    data = {
        u: {"human": raw[u]["human"], "pred": raw[u]["wmean"],
            **{m: raw[u][m] for m in MODELS}}
        for u in UNIS
    }
    out: Dict = {}

    print(f"{'-' * 88}")
    print("  Distributions (test split, complete cases)")
    print(f"{'-' * 88}")
    for u in UNIS:
        h, p = data[u]["human"], data[u]["pred"]
        d = {
            "n": len(h),
            "human_mean": h.mean(), "human_sd": h.std(ddof=1),
            "pred_mean": p.mean(), "pred_sd": p.std(ddof=1),
            "sd_ratio": p.std(ddof=1) / h.std(ddof=1),
            "pct_first": float(np.mean(h >= 70) * 100),
            "pct_60s": float(np.mean((h >= 60) & (h < 70)) * 100),
            "pct_below_60": float(np.mean(h < 60) * 100),
            "slope_pred_on_human": float(np.polyfit(h, p, 1)[0]),
            "model_slopes": {m: float(np.polyfit(h, data[u][m], 1)[0]) for m in MODELS},
            "corr_error_human": float(np.corrcoef(h, p - h)[0, 1]),
        }
        out[u] = d
        print(f"  {u:<11} n={d['n']:>3} human {d['human_mean']:.1f}±{d['human_sd']:.1f}  "
              f"pred {d['pred_mean']:.1f}±{d['pred_sd']:.1f}  SD ratio {d['sd_ratio']:.2f}  "
              f"slope {d['slope_pred_on_human']:.2f}  r(err,human) {d['corr_error_human']:+.2f}")
        print(f"  {'':<11} composition: {d['pct_below_60']:.0f}% <60, "
              f"{d['pct_60s']:.0f}% 60s, {d['pct_first']:.0f}% ≥70")

    print(f"\n{'-' * 88}")
    print("  Signed error / MAE by human grade region (wmean, pooled)")
    print(f"{'-' * 88}")
    pooled_h = np.concatenate([data[u]["human"] for u in UNIS])
    pooled_e = np.concatenate([data[u]["pred"] - data[u]["human"] for u in UNIS])
    out["pooled_by_region"] = {}
    for lo, hi, lab in REGIONS:
        m = (pooled_h >= lo) & (pooled_h < hi)
        out["pooled_by_region"][lab] = {
            "n": int(m.sum()), "bias": float(pooled_e[m].mean()),
            "mae": float(np.abs(pooled_e[m]).mean()),
        }
        print(f"  {lab:<7} n={m.sum():>3}  bias {pooled_e[m].mean():+5.1f}  "
              f"MAE {np.abs(pooled_e[m]).mean():5.1f}")

    print(f"\n{'-' * 88}")
    print("  Per-institution error by band + Cambridge-mix counterfactual")
    print(f"{'-' * 88}")
    camb_mix = Counter(band_of(g) for g in data["Cambridge"]["human"])
    n_camb = sum(camb_mix.values())
    out["by_band"] = {}
    out["counterfactual_cambridge_mix"] = {}
    for u in UNIS:
        h, p = data[u]["human"], data[u]["pred"]
        ae = np.abs(p - h)
        bands_u = np.array([band_of(g) for g in h])
        out["by_band"][u] = {}
        parts = []
        for _, _, lab in BANDS:
            m = bands_u == lab
            if m.sum():
                out["by_band"][u][lab] = {
                    "n": int(m.sum()), "bias": float((p - h)[m].mean()),
                    "mae": float(ae[m].mean()),
                }
                parts.append(f"{lab} n={m.sum()} bias {(p-h)[m].mean():+.1f} MAE {ae[m].mean():.1f}")
        print(f"  {u:<11} " + " | ".join(parts))
        cf, cover = 0.0, 0.0
        for lab, cnt in camb_mix.items():
            m = bands_u == lab
            if m.sum():
                cf += (cnt / n_camb) * ae[m].mean()
                cover += cnt / n_camb
        out["counterfactual_cambridge_mix"][u] = {
            "actual_mae": float(ae.mean()), "reweighted_mae": float(cf / cover),
            "coverage": float(cover),
        }
        print(f"  {'':<11} MAE actual {ae.mean():.1f} → Cambridge-mix {cf/cover:.1f}")

    print(f"\n{'-' * 88}")
    print("  Kitagawa decomposition of the MAE gap vs Cambridge")
    print(f"{'-' * 88}")
    out["kitagawa"] = {}
    fine_edges = list(range(25, 101, 5))
    for target in ("MMU", "Nottingham"):
        k4 = kitagawa(data, target)
        k5 = kitagawa(data, target, edges=fine_edges)
        out["kitagawa"][target] = {"bands_4": k4, "bins_5pt": k5}
        print(f"  {target} − Cambridge: gap = {k4['gap']:+.2f} MAE points")
        print(f"    4 bands : composition {k4['composition']:+.2f} "
              f"({k4['composition_share']*100:.0f}%)  rate {k4['rate']:+.2f}  "
              f"[one-sided comp: {k4['composition_at_target_rates']:+.2f} at "
              f"{target} rates, {k4['composition_at_reference_rates']:+.2f} at Cambridge rates]")
        print(f"    5-pt bins: composition {k5['composition']:+.2f} "
              f"({k5['composition_share']*100:.0f}%)  rate {k5['rate']:+.2f}")

    attractor = float(np.concatenate([data[u]["pred"] for u in UNIS]).mean())
    out["attractor_pooled_pred_mean"] = attractor
    print(f"\nMachine attractor (pooled wmean prediction mean): {attractor:.1f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "location_analysis.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)

    figure_paths = {
        "zone": output_dir / "fig2_zone_of_validity.png",
        "distributions": output_dir / "fig2_transfer_distributions.png",
        "error_by_grade": output_dir / "fig_compression_error_by_grade.png",
        "counterfactual": output_dir / "fig_transfer_counterfactual.png",
    }
    make_figure(data, attractor, out["kitagawa"], figure_paths["zone"])
    make_fig_distributions(data, attractor, figure_paths["distributions"])
    make_fig_error_by_grade(data, attractor, figure_paths["error_by_grade"])
    make_fig_counterfactual(data, figure_paths["counterfactual"])

    print(f"\nResults written to {json_path}")
    print("Figures: " + "\n         ".join(str(p) for p in figure_paths.values()))


if __name__ == "__main__":
    main()
