#!/usr/bin/env python3

"""
Linguistic-Features Analysis

Fisher-z correlation profiles, Holm-corrected correlation matrix, PCA of the
33 assignment features, and GLMs of marks on the six retained components
"""

import argparse
import sys
from pathlib import Path
from textwrap import fill

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA = PROJECT_ROOT / "paper" / "shared_data" / "linguistic_features.csv"
OUT = PROJECT_ROOT / "paper" / "figures" / "linguistic analysis"

# House style, matching paper/make_figures.py and the Table 1 booktabs look
HUMAN_COLOR = "#C4442A"
ENSEMBLE_COLOR = "#3B4CC0"
TEXT_COLOR = "#1A1E2E"
SUBTLE_TEXT = "#5A617A"
REF_LINE = "#8A90A5"
CONNECTOR = "#C2C7D6"
INK, MUTED, RULE, LIGHT_RULE = "#1a1a1a", "#6b6b6b", "#1a1a1a", "#c8c8c8"
UNI_COLORS = {"CAM": "#1D9471", "MMU": "#C77E22", "NOT": "#3B4CC0"}
UNI_LABELS = {"CAM": "Cambridge", "MMU": "Manchester Met", "NOT": "Nottingham"}
UNIS = ("CAM", "MMU", "NOT")

MM = 1 / 25.4
DOUBLE_COL = 183 * MM


def set_nature_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Helvetica Neue", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "axes.titleweight": "bold",
        "axes.labelcolor": TEXT_COLOR,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "xtick.color": "#B3B9C9",
        "ytick.color": "#B3B9C9",
        "xtick.labelcolor": SUBTLE_TEXT,
        "ytick.labelcolor": SUBTLE_TEXT,
        "legend.fontsize": 6,
        "legend.framealpha": 0.0,
        "legend.edgecolor": "none",
        "legend.labelcolor": SUBTLE_TEXT,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#A6ACBE",
        "axes.linewidth": 0.5,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": TEXT_COLOR,
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# Features by domain (display order everywhere)
DOMAINS = [
    ("Lexical diversity & sophistication", [
        ("TTR", "TTR"),
        ("Guiraud_Index", "Guiraud index"),
        ("Herdans_C", "Herdan's C"),
        ("Maas_a2", "Maas a²"),
        ("Dugast_U", "Dugast U"),
        ("MTLD", "MTLD"),
        ("Lexical_Density", "Lexical density"),
        ("Lexical_Sophistication", "Lexical sophistication"),
    ]),
    ("Readability", [
        ("Flesch_Reading_Ease", "Flesch reading ease"),
        ("SMOG", "SMOG"),
        ("Gunning_Fog", "Gunning fog"),
        ("ARI", "ARI"),
        ("Dale_Chall", "Dale–Chall"),
    ]),
    ("Syntactic & structural", [
        ("Avg_Word_Length", "Avg. word length"),
        ("Mean_Sent_Length", "Mean sentence length"),
        ("Avg_Parse_Depth", "Avg. parse depth"),
        ("Clause_Ratio", "Clause ratio"),
        ("Passive_Ratio", "Passive ratio"),
    ]),
    ("Discourse & cohesion", [
        ("Connective_Count", "Connective count"),
        ("Connective_Density", "Connective density"),
        ("Num_Entities", "Entities"),
        ("Entity_Continuity", "Entity continuity"),
        ("Noun_Ratio", "Noun ratio"),
        ("Verb_Ratio", "Verb ratio"),
        ("Pronouns", "Pronouns"),
        ("Modals", "Modals"),
        ("Contractions", "Contractions"),
        ("Sent_Opening_Diversity", "Sentence-opening diversity"),
        ("Avg_Content_Overlap", "Content overlap"),
    ]),
    ("Writing error", [
        ("grammar_errors", "Grammar errors"),
        ("spelling_errors", "Spelling errors"),
        ("punctuation_issues", "Punctuation errors"),
    ]),
    ("Length", [
        ("Word Count", "Word count"),
    ]),
]
FEATURES = [col for _, feats in DOMAINS for col, _ in feats]
FEATURE_LABEL = {col: lab for _, feats in DOMAINS for col, lab in feats}

DOMAIN_COLORS = {
    "Lexical diversity & sophistication": "#1b9e77",
    "Readability": "#d95f02",
    "Syntactic & structural": "#e7298a",
    "Discourse & cohesion": "#7570b3",
    "Writing error": "#666666",
    "Length": "#a6761d",
}
FEATURE_COLOR = {col: DOMAIN_COLORS[d] for d, feats in DOMAINS for col, _ in feats}

PC_LABELS = [
    ("PC1", "Lexical richness & diversity"),
    ("PC2", "Sentence complexity & readability"),
    ("PC3", "Assignment length & information density"),
    ("PC4", "Lexical density vs cohesion"),
    ("PC5", "Conversational vs formal style"),
    ("PC6", "Informal style"),
]

SCORES = ["human_score", "gpt_score", "gemini_score", "claude_score",
          "weighted_mean_score"]
MARKS = [
    ("human_score", "Human"),
    ("gpt_score", "GPT-5.4"),
    ("claude_score", "Claude Opus 4.6"),
    ("gemini_score", "Gemini 3 Flash"),
    ("weighted_mean_score", "Ensemble"),
]
MODEL_BLOCKS = [
    ("gpt_score", "GPT-5.4"),
    ("gemini_score", "Gemini 3 Flash"),
    ("claude_score", "Claude Opus 4.6"),
]
COVARIATE_BLOCKS = [
    ("human_score", "Human"),
    ("weighted_mean_score", "Ensemble"),
]
DUMMY_TERMS = [
    ("C(university)[T.MMU]", "Manchester Met (vs Cambridge)"),
    ("C(university)[T.NOT]", "Nottingham (vs Cambridge)"),
]


def crop_to_content(fig, ax, y_bottom, pad=0.015):
    """Shrink the axes to the drawn band. set_axis_off() otherwise keeps the
    unused height in the tight bbox as dead space below the table."""
    lo = y_bottom - pad
    ax.set_ylim(lo, 1.0)
    fig.set_figheight(fig.get_figheight() * (1.0 - lo))


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def sig_stars(p):
    if np.isnan(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def load_data():
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    return df[FEATURES + ["university"] + SCORES].dropna().reset_index(drop=True)


def fit_pca_glms(df):
    """Fit the 6-component PCA once and one GLM per mark variable."""
    X = StandardScaler().fit_transform(df[FEATURES])
    pca = PCA(n_components=6).fit(X)
    pcs = pca.transform(X)
    for i in range(6):
        df[f"PC{i + 1}"] = pcs[:, i]
    terms = " + ".join(f"PC{i + 1}" for i in range(6))
    models = {s: ols(f"{s} ~ {terms} + C(university)", data=df).fit()
              for s in SCORES}
    return models, pca


# ---------------------------------------------------- Table 2: GLM summary
def render_glm_table(models, variance):
    human, ens = models["human_score"], models["weighted_mean_score"]

    fig, ax = plt.subplots(figsize=(9.6, 3.4), dpi=300)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    row_h = 0.085

    x_comp = 0.005
    x_var = 0.475
    hum_rights = np.linspace(0.56, 0.74, 3)
    ens_rights = np.linspace(0.82, 1.00, 3)

    y = 0.985
    ax.plot([0, 1], [y, y], color=RULE, lw=1.4, clip_on=False)   # toprule
    y -= row_h * 0.72
    ax.text(np.mean(hum_rights) + 0.02, y, "Human mark", ha="center",
            va="center", fontsize=9, fontweight="bold", color=INK)
    ax.text(np.mean(ens_rights) + 0.02, y, "Ensemble mark", ha="center",
            va="center", fontsize=9, fontweight="bold", color=INK)
    yl = y - row_h * 0.38
    ax.plot([hum_rights[0] - 0.06, hum_rights[-1]], [yl, yl],
            color=LIGHT_RULE, lw=0.6)                             # spanner rules
    ax.plot([ens_rights[0] - 0.06, ens_rights[-1]], [yl, yl],
            color=LIGHT_RULE, lw=0.6)
    y -= row_h * 0.80
    ax.text(x_comp, y, "Component", ha="left", va="center",
            fontsize=9, fontweight="bold", color=INK)
    ax.text(x_var, y, "Var. (%)", ha="right", va="center",
            fontsize=9, fontweight="bold", color=INK)
    for rights in (hum_rights, ens_rights):
        for x, col in zip(rights, ("b", "SE", "P")):
            style = "normal" if col == "SE" else "italic"
            ax.text(x, y, col, ha="right", va="center", fontsize=9,
                    fontweight="bold", fontstyle=style, color=INK)
    y -= row_h * 0.52
    ax.plot([0, 1], [y, y], color=RULE, lw=0.8)                   # midrule

    for (pc, label), var in zip(PC_LABELS, variance):
        y -= row_h * 0.82
        ax.text(x_comp, y, f"{pc}  {label}", ha="left", va="center",
                fontsize=9, color=INK)
        ax.text(x_var, y, f"{var:.1f}", ha="right", va="center",
                fontsize=9, color=INK)
        for m, rights in ((human, hum_rights), (ens, ens_rights)):
            bold = "bold" if m.pvalues[pc] < 0.05 else "normal"
            cells = (f"{m.params[pc]:.2f}".replace("-", "−"),
                     f"{m.bse[pc]:.2f}", fmt_p(m.pvalues[pc]))
            for x, cell in zip(rights, cells):
                ax.text(x, y, cell, ha="right", va="center", fontsize=9,
                        fontweight=bold, color=INK)
    y -= row_h * 0.52
    ax.plot([0, 1], [y, y], color=LIGHT_RULE, lw=0.6)             # stats rule

    y -= row_h * 0.82
    ax.text(x_comp, y, "Model fit", ha="left", va="top",
            fontsize=9, fontstyle="italic", color=MUTED)
    for m, rights in ((human, hum_rights), (ens, ens_rights)):
        # P on a second line: the blocks are 0.26 apart and a single-line
        # "R²; F; P" string overruns into the neighbouring block
        p_f = ("P < 0.001" if m.f_pvalue < 0.001
               else f"P = {m.f_pvalue:.3f}")
        fit = (f"R² = {m.rsquared:.3f};  F(8, {int(m.df_resid)}) "
               f"= {m.fvalue:.2f}\n{p_f}")
        ax.text(rights[-1], y, fit, ha="right", va="top",
                fontsize=8.5, color=INK, linespacing=1.5)
    y -= row_h * 1.85
    ax.plot([0, 1], [y, y], color=RULE, lw=1.4, clip_on=False)    # bottomrule
    crop_to_content(fig, ax, y)

    out = OUT / "table_glm_pc_regressions.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white", pad_inches=0.08)
    plt.close(fig)
    print(f"Saved {out}")

    rows = []
    for name in ("human_score", "weighted_mean_score"):
        m = models[name]
        for pc, label in PC_LABELS:
            rows.append({"outcome": name, "component": pc, "label": label,
                         "beta": m.params[pc], "se": m.bse[pc],
                         "t": m.tvalues[pc], "p": m.pvalues[pc],
                         "r2": m.rsquared, "f": m.fvalue, "f_p": m.f_pvalue,
                         "df_model": int(m.df_model),
                         "df_resid": int(m.df_resid), "n": int(m.nobs)})
    src = OUT / "table_glm_pc_regressions_source.csv"
    pd.DataFrame(rows).to_csv(src, index=False)
    print(f"Saved {src}")


# ------------------------------------------------- Fig. 5: Fisher z profiles
def fisher_z(df):
    rows = []
    for uni in UNIS:
        sub = df[df["university"] == uni]
        for feat in FEATURES:
            for score, series in (("human_score", "Human"),
                                  ("weighted_mean_score", "Ensemble")):
                rho = spearmanr(sub[feat], sub[score]).statistic
                rows.append({"university": uni, "feature": feat,
                             "series": series, "rho": rho,
                             "fisher_z": np.arctanh(np.clip(rho, -0.999999,
                                                            0.999999)),
                             "n": len(sub)})
    return pd.DataFrame(rows)


def render_fisher_figure(fz):
    set_nature_style()

    # y layout: one slot per feature, headers and gaps between domains
    y_pos, headers = {}, []
    y = 0.0
    for domain, feats in DOMAINS:
        headers.append((domain, y))
        y += 1.0
        for col, _ in feats:
            y_pos[col] = y
            y += 1.0
        y += 0.55
    total = y

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 185 * MM),
                             sharey=True, sharex=True)
    lim = float(np.ceil(fz["fisher_z"].abs().max() * 10) / 10) + 0.05

    for ax, uni in zip(axes, UNIS):
        sub = fz[fz["university"] == uni].set_index(["feature", "series"])
        n = int(sub["n"].iloc[0])
        ax.axvline(0, color=REF_LINE, lw=0.6, ls=(0, (2, 2)), zorder=1)
        for col in FEATURES:
            yv = y_pos[col]
            zh = sub.loc[(col, "Human"), "fisher_z"]
            ze = sub.loc[(col, "Ensemble"), "fisher_z"]
            ax.plot([zh, ze], [yv, yv], color=CONNECTOR, lw=0.9, zorder=2)
            ax.plot(zh, yv, "o", ms=3.4, color=HUMAN_COLOR,
                    mec="white", mew=0.5, zorder=3)
            ax.plot(ze, yv, "o", ms=3.4, color=ENSEMBLE_COLOR,
                    mec="white", mew=0.5, zorder=3)
        for domain, yh in headers:
            ax.axhline(yh + 0.5, color="#ECEEF3", lw=0.5, zorder=0)
        ax.set_title(f"{UNI_LABELS[uni]} (n = {n})", fontsize=6.5,
                     color=UNI_COLORS[uni], pad=4)
        ax.set_xlim(-lim, lim)
        ax.set_xlabel("Fisher z (Spearman ρ)", fontsize=6.5)

    ax0 = axes[0]
    ax0.set_ylim(total - 0.3, -0.7)
    ax0.set_yticks([y_pos[c] for c in FEATURES])
    ax0.set_yticklabels([FEATURE_LABEL[c] for c in FEATURES], fontsize=5.8)
    ax0.tick_params(axis="y", length=0)
    for domain, yh in headers:                     # headers end at the axis,
        ax0.text(-0.02, yh, domain,                # aligned with tick labels
                 transform=ax0.get_yaxis_transform(), ha="right", va="center",
                 fontsize=6.2, fontweight="bold", color=TEXT_COLOR)

    fig.legend(handles=[
        plt.Line2D([], [], marker="o", ls="", ms=4, color=HUMAN_COLOR,
                   mec="white", mew=0.5, label="Human mark"),
        plt.Line2D([], [], marker="o", ls="", ms=4, color=ENSEMBLE_COLOR,
                   mec="white", mew=0.5, label="Weighted-mean ensemble"),
    ], loc="upper center", ncol=2, bbox_to_anchor=(0.60, 1.015),
        columnspacing=1.4, handletextpad=0.4)

    fig.subplots_adjust(left=0.22, right=0.99, top=0.955, bottom=0.045,
                        wspace=0.12)
    for ext in ("pdf", "png"):
        out = OUT / f"fig_fisherz_institutions.{ext}"
        fig.savefig(out, dpi=600)
        print(f"Saved {out}")
    plt.close(fig)

    src = OUT / "fig_fisherz_institutions_source.csv"
    fz.to_csv(src, index=False)
    print(f"Saved {src}")


# ------------------------------------- Supp Fig 1: Holm-corrected correlations
def supp_fig_correlation_matrix(df):
    set_nature_style()
    order = [k for k, _ in MARKS] + FEATURES
    labels = [lab for _, lab in MARKS] + [FEATURE_LABEL[c] for c in FEATURES]
    colors = ["#000000"] * len(MARKS) + [FEATURE_COLOR[c] for c in FEATURES]

    num = df[order]
    corr = num.corr(method="spearman")
    n = len(order)

    p_matrix = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            p_matrix[i, j] = spearmanr(num.iloc[:, i], num.iloc[:, j]).pvalue
    iu = np.triu_indices(n, k=1)
    corrected = multipletests(p_matrix[iu], method="holm")[1]
    p_corr = np.full((n, n), np.nan)
    p_corr[iu] = corrected
    p_corr = np.where(np.isnan(p_corr), p_corr.T, p_corr)   # symmetrise

    stars = pd.DataFrame(
        [[sig_stars(p_corr[i, j]) if j < i else "" for j in range(n)]
         for i in range(n)], index=labels, columns=labels)
    corr.index = corr.columns = labels
    mask = np.triu(np.ones((n, n), dtype=bool), k=0)        # lower triangle

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 190 * MM))
    cbar_ax = fig.add_axes([0.90, 0.60, 0.015, 0.30])
    sns.heatmap(
        corr, mask=mask, cmap="RdBu_r", vmin=-1, vmax=1, center=0,
        square=True, linewidths=0.3, linecolor="white",
        annot=stars, fmt="", annot_kws={"fontsize": 3.8, "fontweight": "bold",
                                        "va": "center"},
        cbar_ax=cbar_ax, cbar_kws={"label": "Spearman ρ"},
        ax=ax,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=5.4)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=5.4)
    for lbl, c in zip(ax.get_xticklabels(), colors):
        lbl.set_color(c)
    for lbl, c in zip(ax.get_yticklabels(), colors):
        lbl.set_color(c)
    ax.tick_params(length=0)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=5.5)
    cbar.set_label("Spearman ρ", fontsize=6.5)
    cbar.outline.set_visible(False)

    handles = [plt.Line2D([], [], marker="s", ls="", ms=4.5, color=c, label=d)
               for d, c in DOMAIN_COLORS.items()]
    handles.insert(0, plt.Line2D([], [], marker="s", ls="", ms=4.5,
                                 color="#000000", label="Marks"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.42, 0.99),
              fontsize=6, frameon=False, handletextpad=0.5,
              labelspacing=0.55)

    fig.subplots_adjust(left=0.14, right=0.995, top=0.995, bottom=0.115)
    for ext in ("pdf", "png"):
        out = OUT / f"supp_fig_correlation_matrix.{ext}"
        fig.savefig(out, dpi=600)
        print(f"Saved {out}")
    plt.close(fig)

    corr_raw = num.corr(method="spearman")
    export = [{"var1": order[i], "var2": order[j],
               "rho": corr_raw.iloc[i, j], "p_holm": p_corr[i, j]}
              for i in range(n) for j in range(i + 1, n)]
    src = OUT / "supp_fig_correlation_matrix_source.csv"
    pd.DataFrame(export).to_csv(src, index=False)
    print(f"Saved {src}")

    # Console summary backing the main-text counts
    for score, lab in MARKS:
        hits = sum(1 for row in export
                   if row["var1"] == score and row["var2"] in FEATURES
                   and row["p_holm"] < 0.05)
        print(f"  {lab}: {hits} of {len(FEATURES)} features significant (Holm)")


# ---------------------------------------------- Supp Fig 2: PCA loadings
PC_INTERPRETATIONS = {
    "PC1": "Captures lexical diversity and vocabulary richness. Higher "
           "scores indicate greater lexical variation and richer "
           "vocabulary use.",
    "PC2": "Reflects sentence complexity and readability, characterised by "
           "longer and more structurally complex sentences, higher "
           "readability grade levels, deeper syntactic structures, and "
           "greater use of discourse connectives.",
    "PC3": "Represents assignment length and information density. Higher "
           "scores indicate longer assignments containing more entities, "
           "discourse connectives, and a greater absolute number of "
           "language errors associated with increased assignment length.",
    "PC4": "Contrasts information-dense writing with discourse cohesion. "
           "Higher scores indicate greater lexical density, longer words, "
           "and increased noun use, whereas lower scores reflect greater "
           "use of pronouns and connective devices associated with "
           "cohesive discourse.",
    "PC5": "Higher scores are associated with greater use of verbs, "
           "pronouns, and modal expressions, reflecting a more personal "
           "and action-oriented writing style, whereas lower scores "
           "reflect greater lexical sophistication and a more formal "
           "style.",
    "PC6": "Captures stylistic informality, primarily characterised by the "
           "use of contractions and other features associated with "
           "conversational language. Higher scores indicate a more "
           "informal and speech-like writing style.",
}


def supp_fig_pca_loadings(pca):
    set_nature_style()
    variance = pca.explained_variance_ratio_ * 100
    loadings = pd.DataFrame(pca.components_.T, index=FEATURES,
                            columns=[pc for pc, _ in PC_LABELS])

    fig = plt.figure(figsize=(DOUBLE_COL, 150 * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.55], wspace=0.22)

    # panel a — loadings heatmap, rows grouped by strongest PC then |loading|
    ax1 = fig.add_subplot(gs[0])
    lim = float(np.abs(loadings.values).max())
    annot = pd.DataFrame("", index=loadings.index, columns=loadings.columns)
    for row in loadings.index:
        strongest = loadings.loc[row].abs().idxmax()
        annot.loc[row, strongest] = f"{loadings.loc[row, strongest]:.2f}"
    pcs = list(loadings.columns)
    strongest_pc = loadings.abs().idxmax(axis=1)
    order = sorted(loadings.index,
                   key=lambda c: (pcs.index(strongest_pc[c]),
                                  -abs(loadings.loc[c, strongest_pc[c]])))
    plot = loadings.loc[order]
    annot = annot.loc[order]
    plot.index = [FEATURE_LABEL[c] for c in order]
    annot.index = plot.index
    sns.heatmap(plot, cmap="RdBu_r", vmin=-lim, vmax=lim, center=0,
                annot=annot, fmt="", annot_kws={"fontsize": 4.6},
                linewidths=0.3, linecolor="white",
                cbar_kws={"label": "Loading", "shrink": 0.5, "pad": 0.03},
                ax=ax1)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0, fontsize=5.5)
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontsize=5.2)
    for lbl, c in zip(ax1.get_yticklabels(),
                      [FEATURE_COLOR[f] for f in order]):
        lbl.set_color(c)
    ax1.tick_params(length=0)
    ax1.set_ylabel("")
    cbar = ax1.collections[0].colorbar
    cbar.ax.tick_params(labelsize=5)
    cbar.set_label("Loading", fontsize=6)
    cbar.outline.set_visible(False)
    ax1.text(-0.52, 1.02, "a", transform=ax1.transAxes, fontsize=8,
             fontweight="bold", va="bottom", color=TEXT_COLOR)

    # panel b — interpretation table
    ax2 = fig.add_subplot(gs[1])
    ax2.set_axis_off()
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.text(-0.02, 1.02, "b", transform=ax2.transAxes, fontsize=8,
             fontweight="bold", va="bottom", color=TEXT_COLOR)

    x_pc, x_lab, x_int = 0.02, 0.12, 0.44
    line_h = 0.0225
    y = 0.965
    ax2.plot([0, 1], [y, y], color=RULE, lw=1.2, clip_on=False)
    y -= 0.035
    for x, head in ((x_pc, "PC"), (x_lab, "Label"),
                    (x_int, "Interpretation")):
        ax2.text(x, y, head, ha="left", va="center", fontsize=6.5,
                 fontweight="bold", color=INK)
    y -= 0.022
    ax2.plot([0, 1], [y, y], color=RULE, lw=0.7)

    for pc, label in PC_LABELS:
        wrapped_label = fill(label, width=26)
        wrapped_interp = fill(PC_INTERPRETATIONS[pc], width=58)
        n_lines = max(wrapped_label.count("\n"),
                      wrapped_interp.count("\n")) + 1
        y_top = y - 0.016
        ax2.text(x_pc, y_top, pc, ha="left", va="top", fontsize=6, color=INK)
        ax2.text(x_lab, y_top, wrapped_label, ha="left", va="top",
                 fontsize=6, color=INK, linespacing=1.35)
        ax2.text(x_int, y_top, wrapped_interp, ha="left", va="top",
                 fontsize=6, color=INK, linespacing=1.35)
        y = y_top - line_h * n_lines
    y -= 0.008
    ax2.plot([0, 1], [y, y], color=RULE, lw=1.2, clip_on=False)

    fig.subplots_adjust(left=0.115, right=0.99, top=0.96, bottom=0.05)
    for ext in ("pdf", "png"):
        out = OUT / f"supp_fig_pca_loadings.{ext}"
        fig.savefig(out, dpi=600)
        print(f"Saved {out}")
    plt.close(fig)

    loadings.round(4).to_csv(OUT / "supp_fig_pca_loadings_source.csv")
    print(f"Saved {OUT / 'supp_fig_pca_loadings_source.csv'}")
    print("  variance explained (%):",
          np.round(variance, 1), "sum:", round(variance.sum(), 1))


# ------------------------------------------- Supp Table 8: definitions
FEATURE_DEFINITIONS = [
    ("Lexical diversity &\nsophistication",
     "TTR; Guiraud index; Herdan's C; Maas a²; Dugast U; MTLD; "
     "lexical density; lexical sophistication",
     "Vocabulary range and richness: type–token-based diversity indices, "
     "including length-corrected variants; the proportion of content words "
     "(lexical density); and the proportion of sophisticated, low-frequency "
     "vocabulary (lexical sophistication).",
     "Diversity indices computed with the lexicalrichness library over "
     "spaCy/NLTK-preprocessed text; lexical density from spaCy "
     "part-of-speech tags; sophistication scored against wordfreq "
     "word frequencies."),
    ("Readability",
     "Flesch reading ease; SMOG; Gunning fog; ARI; Dale–Chall",
     "Standard readability formulae estimating the reading level required "
     "to comprehend the text from sentence length, word length or syllable "
     "counts, and familiar-word lists.",
     "Computed with the textstat library."),
    ("Syntactic &\nstructural complexity",
     "Avg. word length; mean sentence length; avg. parse depth; "
     "clause ratio; passive ratio",
     "Word- and sentence-level structural complexity: mean lengths, "
     "syntactic parse-tree depth, the incidence of subordinate clauses, "
     "and the proportion of passive constructions.",
     "Derived from spaCy dependency parses with NLTK preprocessing "
     "(sentence segmentation, parse depth, clause and passive "
     "detection)."),
    ("Discourse & cohesion",
     "Connective count; connective density; entities; entity continuity; "
     "noun ratio; verb ratio; pronouns; modals; contractions; "
     "sentence-opening diversity; content overlap",
     "Discourse organisation and cohesion: discourse-connective use, "
     "entity counts and their continuity across sentences, "
     "part-of-speech distributions, register markers (modals, "
     "contractions), variety of sentence openings, and content-word "
     "overlap between adjacent sentences.",
     "Derived from spaCy/NLTK preprocessing: named-entity recognition, "
     "part-of-speech tagging and connective matching against a "
     "connective inventory."),
    ("Writing error",
     "Grammar errors; spelling errors; punctuation errors",
     "Counts of detected grammar, spelling and punctuation errors in the "
     "assignment.",
     "Grammar, spelling and punctuation detection with symspellpy."),
    ("Length",
     "Word count",
     "Total number of word tokens in the assignment.",
     "Token count of the spaCy/NLTK-tokenised text."),
]


def supp_table_feature_definitions():
    # Text-extent measurements depend on the active font; pin to matplotlib
    # defaults so output is independent of which items rendered before it.
    with mpl.rc_context(rc=mpl.rcParamsDefault):
        _render_feature_definitions()


def _render_feature_definitions():
    col_x = [0.005, 0.16, 0.40, 0.745]
    widths = [19, 29, 42, 31]
    heads = ["Domain", "Features", "Description", "Extraction method"]

    wrapped = [[fill(cell, width=w) if "\n" not in cell else cell
                for cell, w in zip(row, widths)]
               for row in FEATURE_DEFINITIONS]

    fig, ax = plt.subplots(figsize=(10.5, 6.8), dpi=300)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_h_px = ax.get_window_extent(renderer).height

    y = 0.985
    ax.plot([0, 1], [y, y], color=RULE, lw=1.4, clip_on=False)
    y -= 0.042
    for x, head in zip(col_x, heads):
        ax.text(x, y, head, ha="left", va="center", fontsize=9,
                fontweight="bold", color=INK)
    y -= 0.026
    ax.plot([0, 1], [y, y], color=RULE, lw=0.8)

    for i, row in enumerate(wrapped):
        y -= 0.028
        texts = [ax.text(x, y, cell, ha="left", va="top", fontsize=8,
                         color=INK, linespacing=1.4)
                 for x, cell in zip(col_x, row)]
        tallest = max(t.get_window_extent(renderer).height
                      for t in texts) / ax_h_px
        y -= tallest + 0.012
        if i < len(wrapped) - 1:
            ax.plot([0, 1], [y, y], color=LIGHT_RULE, lw=0.5)
    ax.plot([0, 1], [y, y], color=RULE, lw=1.4, clip_on=False)

    out = OUT / "supp_table_feature_definitions.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white", pad_inches=0.08)
    plt.close(fig)
    print(f"Saved {out}")


# --------------------------------------- Supp Table 9: per-model GLMs
def supp_table_per_model_glms(models):
    fig, ax = plt.subplots(figsize=(11.5, 5.6), dpi=300)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    row_h = 0.055
    x_term = 0.005
    block_rights = [np.linspace(0.40 + b * 0.21, 0.52 + b * 0.21, 3)
                    for b in range(3)]

    def cells(m, term):
        bold = "bold" if m.pvalues[term] < 0.05 else "normal"
        return [(f"{m.params[term]:.2f}".replace("-", "−"), bold),
                (f"{m.bse[term]:.2f}", bold),
                (fmt_p(m.pvalues[term]), bold)]

    def header(y, blocks, labels):
        for rights, label in zip(blocks, labels):
            ax.plot([rights[0] - 0.045, rights[-1]],
                    [y - row_h * 0.38, y - row_h * 0.38],
                    color=LIGHT_RULE, lw=0.6)
            ax.text(np.mean(rights) + 0.01, y, label, ha="center",
                    va="center", fontsize=9, fontweight="bold", color=INK)
        yy = y - row_h * 0.85
        for rights in blocks:
            for x, col in zip(rights, ("b", "SE", "P")):
                style = "normal" if col == "SE" else "italic"
                ax.text(x, yy, col, ha="right", va="center", fontsize=9,
                        fontweight="bold", fontstyle=style, color=INK)
        return yy

    # --- section 1: per-model regressions
    y = 0.985
    ax.plot([0, 1], [y, y], color=RULE, lw=1.4, clip_on=False)
    y -= row_h * 0.75
    y = header(y, block_rights, [lab for _, lab in MODEL_BLOCKS])
    y -= row_h * 0.55
    ax.plot([0, 1], [y, y], color=RULE, lw=0.8)

    pc_rows = [(pc, f"{pc}  {label}") for pc, label in PC_LABELS]
    for term, label in pc_rows + DUMMY_TERMS:
        y -= row_h * 0.85
        ax.text(x_term, y, label, ha="left", va="center", fontsize=9,
                color=INK)
        for (score, _), rights in zip(MODEL_BLOCKS, block_rights):
            for x, (text, weight) in zip(rights, cells(models[score], term)):
                ax.text(x, y, text, ha="right", va="center", fontsize=9,
                        fontweight=weight, color=INK)
    y -= row_h * 0.50
    ax.plot([0, 1], [y, y], color=LIGHT_RULE, lw=0.6)
    y -= row_h * 0.85
    ax.text(x_term, y, "Model fit", ha="left", va="center", fontsize=9,
            fontstyle="italic", color=MUTED)
    for (score, _), rights in zip(MODEL_BLOCKS, block_rights):
        m = models[score]
        p_f = ("P < 0.001" if m.f_pvalue < 0.001
               else f"P = {m.f_pvalue:.3f}")
        fit = (f"R² = {m.rsquared:.3f};  F(8, {int(m.df_resid)}) "
               f"= {m.fvalue:.2f}, {p_f}")
        ax.text(rights[-1], y, fit, ha="right", va="center", fontsize=8,
                color=INK)
    y -= row_h * 0.55
    ax.plot([0, 1], [y, y], color=RULE, lw=1.4, clip_on=False)

    # --- section 2: institutional covariates from the Table 2 models
    y -= row_h * 1.15
    ax.text(x_term, y, "Institutional covariates from the human and "
                       "ensemble models (Table 2)",
            ha="left", va="center", fontsize=9, fontstyle="italic",
            color=MUTED)
    y -= row_h * 0.85
    cov_rights = [np.linspace(0.40, 0.52, 3), np.linspace(0.61, 0.73, 3)]
    y = header(y, cov_rights, [lab for _, lab in COVARIATE_BLOCKS])
    y -= row_h * 0.55
    ax.plot([0, 0.73], [y, y], color=RULE, lw=0.8)
    for term, label in DUMMY_TERMS:
        y -= row_h * 0.85
        ax.text(x_term, y, label, ha="left", va="center", fontsize=9,
                color=INK)
        for (score, _), rights in zip(COVARIATE_BLOCKS, cov_rights):
            for x, (text, weight) in zip(rights, cells(models[score], term)):
                ax.text(x, y, text, ha="right", va="center", fontsize=9,
                        fontweight=weight, color=INK)
    y -= row_h * 0.55
    ax.plot([0, 0.73], [y, y], color=RULE, lw=1.4, clip_on=False)
    crop_to_content(fig, ax, y)

    out = OUT / "supp_table_per_model_glms.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white", pad_inches=0.08)
    plt.close(fig)
    print(f"Saved {out}")

    rows = []
    for score, m in models.items():
        for term in [f"PC{i+1}" for i in range(6)] + \
                    [t for t, _ in DUMMY_TERMS] + ["Intercept"]:
            rows.append({"outcome": score, "term": term,
                         "b": m.params[term], "se": m.bse[term],
                         "t": m.tvalues[term], "p": m.pvalues[term],
                         "r2": m.rsquared, "f": m.fvalue, "f_p": m.f_pvalue,
                         "df_resid": int(m.df_resid), "n": int(m.nobs)})
    src = OUT / "supp_table_per_model_glms_source.csv"
    pd.DataFrame(rows).to_csv(src, index=False)
    print(f"Saved {src}")


ITEMS = ("glm_table", "fisher", "correlations", "loadings",
         "definitions", "per_model")


def main():
    ap = argparse.ArgumentParser(description="Linguistic-features analysis.")
    ap.add_argument("--only", nargs="*", choices=ITEMS,
                    help="display items to generate (default: all)")
    args = ap.parse_args()
    selected = set(args.only or ITEMS)

    OUT.mkdir(parents=True, exist_ok=True)
    df = load_data()
    print(f"Rows retained: {len(df)}")
    models, pca = fit_pca_glms(df)

    if "glm_table" in selected:
        render_glm_table(models, pca.explained_variance_ratio_ * 100)
    if "fisher" in selected:
        render_fisher_figure(fisher_z(df))
    if "correlations" in selected:
        supp_fig_correlation_matrix(df)
    if "loadings" in selected:
        supp_fig_pca_loadings(pca)
    if "definitions" in selected:
        supp_table_feature_definitions()
    if "per_model" in selected:
        supp_table_per_model_glms(models)


if __name__ == "__main__":
    main()
