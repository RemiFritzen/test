#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
plot_figure1_umap_bar.py  (revised)
------------------------------------
Create Figure 1 panels:
  A – UMAP coloured by dataset
  B – UMAP coloured by condition (HC / MS)
  C – UMAP coloured by tissue × CXCR3 group
  D – bar + dot chart: % CXCR3+ B cells per patient × group

Changes vs original:
  1.  publication rcParams: font sizes, pdf.fonttype=42 (editable in Illustrator)
  2.  panel letters (A–D) on every individual PDF and on the combined figure
  3.  UMAP axis labels added (UMAP 1 / UMAP 2)
  4.  Panel B – equal alpha=0.6 for all conditions; draw order controlled via
      Categorical category ordering (MS underneath, HC on top)
      → single sc.pl.umap() call, no legend duplication risk
  5.  Panel C – CXCR3- categories listed first in Categorical so their points
      are drawn first (background); CXCR3+ drawn last (foreground / on top)
  6.  Panel D – significance brackets added to the individual-panel PDF
  7.  Panel D – n= per group shown on x-axis tick labels
  8.  Panel D – fixed-seed RNG for jitter (reproducible across runs)
  9.  Panel D – y-axis label explicitly states "mean ± SEM"
  10. DRY: shared helper functions used by both individual panels and combined fig
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
from scipy.stats import mannwhitneyu

from config.config import RESULTS_DIR, FIG_DIRS


# ---------------------------------------------------------------------------
# Global aesthetics – publication-ready defaults
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        12,
    "axes.labelsize":   12,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "xtick.labelsize":  11,
    "ytick.labelsize":  11,
    "legend.fontsize":  10,
    "figure.dpi":       150,
    # Keep text as editable glyphs in PDF/PS (required for Illustrator/Inkscape)
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
})

# One RNG instance, fixed seed → reproducible jitter in every panel
_RNG = np.random.default_rng(42)

# ── Colour palettes ──────────────────────────────────────────────────────────
COND_PALETTE = {
    "HC":  "#1f77b4",   # blue
    "MS":  "#ff7f0e",   # orange
}

PALETTE4 = {
    "PB_CXCR3+":  "#1f77b4",
    "PB_CXCR3-":  "#9ecae1",
    "CSF_CXCR3+": "#d62728",
    "CSF_CXCR3-": "#fcbba1",
}

BAR_COLORS = ["#4C78A8", "#72B7B2", "#E45756", "#F58518"]
ORDER_BAR  = ["HC-PB", "HC-CSF", "MS-PB", "MS-CSF"]

# Distinct colors/markers for patient-level dots, coded by dataset of origin
# -- lets you visually check whether a bar's mean is being driven by
# patients from one study rather than a consistent pattern across both
# (same question the formal dataset-confounding check in fig2.py addresses
# statistically for the composite signature). Built dynamically so it's not
# hardcoded to exactly two datasets.
_DATASET_COLOR_CYCLE = ["#4C78A8", "#E45756", "#54A24B", "#F58518", "#B279A2"]
_DATASET_MARKER_CYCLE = ["o", "^", "s", "D", "v"]


def _build_dataset_palette(datasets: list[str]) -> tuple[dict, dict]:
    """Assign a stable color + marker to each dataset, in sorted order."""
    datasets = sorted(set(datasets))
    colors = {ds: _DATASET_COLOR_CYCLE[i % len(_DATASET_COLOR_CYCLE)] for i, ds in enumerate(datasets)}
    markers = {ds: _DATASET_MARKER_CYCLE[i % len(_DATASET_MARKER_CYCLE)] for i, ds in enumerate(datasets)}
    return colors, markers


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _panel_letter(ax: plt.Axes, letter: str) -> None:
    """Bold panel letter (A, B, C, D) in the upper-left corner of the axis."""
    ax.text(
        -0.12, 1.05, letter,
        transform=ax.transAxes,
        fontsize=16, fontweight="bold",
        va="top", ha="left",
    )


def _umap_axes(ax: plt.Axes) -> None:
    """Minimalist UMAP axis labels (often absent in sc papers, but adds clarity)."""
    ax.set_xlabel("UMAP 1", fontsize=10, labelpad=2)
    ax.set_ylabel("UMAP 2", fontsize=10, labelpad=2)


def _sig_bracket(
    ax: plt.Axes,
    x1: int, x2: int,
    y: float, h: float,
    p: float,
) -> None:
    """Draw a Tukey-style significance bracket between two bar positions."""
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c="k")
    if p < 0.001:
        label = "***"
    elif p < 0.01:
        label = "**"
    elif p < 0.05:
        label = "*"
    else:
        label = f"p = {p:.2f}"
    ax.text(
        (x1 + x2) / 2, y + h * 1.1,
        label, ha="center", va="bottom", fontsize=10,
    )


def _maybe_bracket(
    ax: plt.Axes,
    g1: str, g2: str,
    order: list[str],
    p: float,
    y: float, h: float,
) -> None:
    """Draw a significance bracket only if p is not NaN and both groups are present."""
    if np.isnan(p) or g1 not in order or g2 not in order:
        return
    _sig_bracket(ax, order.index(g1), order.index(g2), y, h, p)


def _draw_bar_panel(
    ax: plt.Axes,
    patient_df: pd.DataFrame,
    bar_df: pd.DataFrame,
    order: list[str],
    pvals: dict[str, float],
    color_dots_by_dataset: bool = True,
    legend_outside: bool = True,
) -> None:
    """
    Core bar + dot logic shared by save_panel_bar() and _save_combined_figure().
    Uses a fresh fixed-seed RNG so jitter is identical in both outputs.

    color_dots_by_dataset: if True (default) and patient_df has a 'dataset'
    column, patient-level dots are colored/marker-coded by dataset of origin
    instead of uniform black. This makes it visually checkable whether a
    bar's mean is being driven by one study rather than reflecting a
    consistent pattern across datasets -- the same question addressed
    statistically for the zinc signature in fig2.py's dataset confounding
    check.

    legend_outside: standalone panel PDFs have room for the dataset legend
    outside the axes (loc='upper left', bbox_to_anchor); the combined 2x2
    figure does not -- pass False there to place it inside the axes instead
    and avoid clipping.
    """
    rng = np.random.default_rng(42)

    ymax = max(
        (bar_df["mean"] + bar_df["sem"]).max() if len(bar_df) else 0,
        patient_df["pct_cxcr3_pos"].max()       if len(patient_df) else 0,
    )

    # Bars
    for i, grp in enumerate(order):
        row = bar_df[bar_df["group4"] == grp]
        if row.empty:
            continue
        ax.bar(
            i, row["mean"].values[0],
            yerr=row["sem"].values[0],
            color=BAR_COLORS[i % len(BAR_COLORS)],
            capsize=5, alpha=0.9, width=0.65,
        )

    # Patient-level dots with fixed-seed jitter, colored by dataset if requested
    use_dataset_color = color_dots_by_dataset and "dataset" in patient_df.columns
    if use_dataset_color:
        ds_colors, ds_markers = _build_dataset_palette(patient_df["dataset"].unique().tolist())

    plotted_datasets = []
    for i, grp in enumerate(order):
        sub = patient_df[patient_df["group4"] == grp]
        if not len(sub):
            continue
        jitter = rng.normal(0, 0.06, size=len(sub))
        if use_dataset_color:
            for ds, ds_sub in sub.groupby("dataset"):
                ds_jitter = rng.normal(0, 0.06, size=len(ds_sub))
                ax.scatter(
                    np.full(len(ds_sub), i) + ds_jitter, ds_sub["pct_cxcr3_pos"].values,
                    color=ds_colors[ds], marker=ds_markers[ds],
                    edgecolors="black", linewidths=0.4,
                    s=30, alpha=0.85, zorder=10,
                    label=ds if ds not in plotted_datasets else None,
                )
                if ds not in plotted_datasets:
                    plotted_datasets.append(ds)
        else:
            ax.scatter(
                np.full(len(sub), i) + jitter, sub["pct_cxcr3_pos"].values,
                color="k", s=25, alpha=0.8, zorder=10,
            )

    if use_dataset_color and plotted_datasets:
        if legend_outside:
            ax.legend(
                title="Dataset", frameon=False, fontsize=8, title_fontsize=8,
                loc="upper left", bbox_to_anchor=(1.01, 1.0),
            )
        else:
            ax.legend(
                title="Dataset", frameon=False, fontsize=7, title_fontsize=7,
                loc="lower right",
            )

    # x-axis tick labels: group name + n=  (FIX: n= added)
    n_labels = []
    for grp in order:
        n = bar_df.loc[bar_df["group4"] == grp, "count"].values
        n_labels.append(f"{grp}\n(n = {int(n[0]) if len(n) else 0})")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(n_labels, rotation=20, ha="right", fontsize=10)

    # Significance brackets  (FIX: now in individual PDF too)
    bh = ymax * 0.05
    y_pb  = ymax * 1.08
    y_csf = y_pb + bh + ymax * 0.08
    _maybe_bracket(ax, "HC-PB",  "MS-PB",  order, pvals.get("PB",  np.nan), y_pb,  bh)
    _maybe_bracket(ax, "HC-CSF", "MS-CSF", order, pvals.get("CSF", np.nan), y_csf, bh)

    ax.set_title("CXCR3⁺ B cells (% of B cells)")
    ax.set_ylabel("% CXCR3⁺ (mean ± SEM)")   # FIX: SEM explicitly labelled
    ax.set_xlabel("")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(-0.6, len(order) - 0.4)


# ---------------------------------------------------------------------------
# Panel savers
# ---------------------------------------------------------------------------

def save_panel_dataset_umap(adata: sc.AnnData, outpath) -> None:
    """Panel A – UMAP coloured by dataset."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sc.pl.umap(
        adata, color="dataset",
        ax=ax, show=False,
        title="UMAP by dataset", frameon=False,
    )
    _umap_axes(ax)
    _panel_letter(ax, "B")
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def save_panel_condition_umap(adata: sc.AnnData, outpath) -> None:
    """
    Panel B – UMAP coloured by condition.

    FIX: a single sc.pl.umap() call with equal alpha=0.6 for all conditions.
    Draw order is controlled by the Categorical category order:
      MS listed first  → drawn first (background layer)
      HC listed second → drawn on top of MS
    This means HC (the smaller group) remains visible without
    artificially fading MS.
    """
    adata = adata.copy()
    # Category order = draw order (scanpy iterates categories in order)
    draw_order = ["MS", "HC"]
    adata.obs["condition"] = pd.Categorical(
        adata.obs["condition"].astype(str),
        categories=draw_order,
        ordered=False,
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    sc.pl.umap(
        adata, color="condition",
        ax=ax, show=False,
        title="UMAP by condition",
        frameon=False,
        alpha=0.6, size=6,
        palette=COND_PALETTE,
    )
    _umap_axes(ax)
    _panel_letter(ax, "C")
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def save_panel_cxcr3_tissue_umap(adata: sc.AnnData, outpath) -> None:
    """
    Panel C – UMAP coloured by tissue × CXCR3 group.

    FIX: CXCR3- categories are listed first in the Categorical so scanpy
    draws them first (background). CXCR3+ categories are listed last so
    they are drawn on top and are not occluded by the majority CXCR3- mass.
    The legend is re-ordered separately to show the biologically natural
    pairing (PB+/-, CSF+/-).
    """
    adata = adata.copy()
    adata.obs["tissue"] = adata.obs["tissue"].replace(
        {"PBMCs": "PB", "PBMC": "PB"}
    ).astype(str)
    adata.obs["cxcr3_tissue_group"] = (
        adata.obs["tissue"] + "_" + adata.obs["cxcr3_group"].astype(str)
    )

    # CXCR3- listed first → drawn underneath; CXCR3+ listed last → on top
    draw_order = ["PB_CXCR3-", "CSF_CXCR3-", "PB_CXCR3+", "CSF_CXCR3+"]
    adata.obs["cxcr3_tissue_group"] = pd.Categorical(
        adata.obs["cxcr3_tissue_group"],
        categories=draw_order,
        ordered=True,
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    sc.pl.umap(
        adata, color="cxcr3_tissue_group",
        ax=ax, show=False,
        title="UMAP by tissue + CXCR3 group",
        frameon=False,
        palette=PALETTE4,
    )

    # Re-order legend to the natural pairing PB+/-, CSF+/-
    legend_order = ["PB_CXCR3+", "PB_CXCR3-", "CSF_CXCR3+", "CSF_CXCR3-"]
    # handles = {t.get_text(): t for t in ax.get_legend().get_texts()}
    leg = ax.get_legend()
    if leg is not None:
        old_handles = leg.legend_handles
        old_labels  = [t.get_text() for t in leg.get_texts()]
        label_to_h  = dict(zip(old_labels, old_handles))
        new_h = [label_to_h[l] for l in legend_order if l in label_to_h]
        ax.legend(
            handles=new_h, labels=[l for l in legend_order if l in label_to_h],
            frameon=False, fontsize=9,
        )

    _umap_axes(ax)
    _panel_letter(ax, "D")
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def save_panel_bar(
    patient_df: pd.DataFrame,
    bar_df: pd.DataFrame,
    order: list[str],
    pvals: dict[str, float],
    outpath,
) -> None:
    """Panel D – bar + dot chart (with significance brackets), dots colored by dataset."""
    fig, ax = plt.subplots(figsize=(7, 5))
    _draw_bar_panel(ax, patient_df, bar_df, order, pvals, legend_outside=True)
    _panel_letter(ax, "E")
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Combined 2×2 figure
# ---------------------------------------------------------------------------

def _save_combined_figure(
    adata: sc.AnnData,
    patient_df: pd.DataFrame,
    bar_df: pd.DataFrame,
    order: list[str],
    pvals: dict[str, float],
    outpath,
) -> None:
    fig = plt.figure(figsize=(16, 12))
    gs  = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.45)

    # ── A ────────────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    sc.pl.umap(adata, color="dataset", ax=ax1, show=False,
               title="UMAP by dataset", frameon=False)
    _umap_axes(ax1)
    _panel_letter(ax1, "B")

    # ── B ────────────────────────────────────────────────────────────────────
    ax2   = fig.add_subplot(gs[0, 1])
    adata_b = adata.copy()
    draw_order_b = ["MS", "HC"]
    adata_b.obs["condition"] = pd.Categorical(
        adata_b.obs["condition"].astype(str),
        categories=draw_order_b, ordered=False,
    )
    sc.pl.umap(adata_b, color="condition", ax=ax2, show=False,
               title="UMAP by condition", frameon=False,
               alpha=0.6, size=6, palette=COND_PALETTE)
    _umap_axes(ax2)
    _panel_letter(ax2, "C")

    # ── C ────────────────────────────────────────────────────────────────────
    ax3   = fig.add_subplot(gs[1, 0])
    adata_c = adata.copy()
    adata_c.obs["tissue"] = adata_c.obs["tissue"].replace(
        {"PBMCs": "PB", "PBMC": "PB"}
    ).astype(str)
    adata_c.obs["cxcr3_tissue_group"] = (
        adata_c.obs["tissue"] + "_" + adata_c.obs["cxcr3_group"].astype(str)
    )
    draw_order_c = ["PB_CXCR3-", "CSF_CXCR3-", "PB_CXCR3+", "CSF_CXCR3+"]
    adata_c.obs["cxcr3_tissue_group"] = pd.Categorical(
        adata_c.obs["cxcr3_tissue_group"],
        categories=draw_order_c, ordered=True,
    )
    sc.pl.umap(adata_c, color="cxcr3_tissue_group", ax=ax3, show=False,
               title="UMAP by tissue + CXCR3 group", frameon=False,
               palette=PALETTE4)
    _umap_axes(ax3)
    _panel_letter(ax3, "D")

    # ── D ────────────────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    _draw_bar_panel(ax4, patient_df, bar_df, order, pvals, legend_outside=False)
    _panel_letter(ax4, "E")

    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined figure → {outpath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def check_bar_dataset_confounding(
    patient_df: pd.DataFrame,
    tissue: str,
    min_patients_per_group: int = 3,
) -> dict:
    """
    Check whether the HC-vs-MS %CXCR3+ comparison (the significance bracket
    in Panel D) is being driven by, or differs meaningfully between, dataset
    of origin -- same rationale as fig2.py's check_dataset_confounding for
    the zinc signature (reviewer concern #4). Two checks:

    1. STRATIFIED: rerun the same HC-vs-MS Mann-Whitney test WITHIN each
       dataset separately.
    2. FORMAL INTERACTION TEST: OLS on pct_cxcr3_pos ~ C(condition) *
       C(dataset) within this tissue; a significant interaction term means
       the HC-vs-MS difference genuinely differs by dataset.
    """
    import statsmodels.formula.api as smf

    tdf = patient_df[
        (patient_df["tissue"] == tissue) & patient_df["condition"].isin(["HC", "MS"])
    ].copy()

    print(f"\n  === Dataset confounding check: {tissue} %CXCR3+ (HC vs MS) ===")

    per_dataset_results = []
    for ds in sorted(tdf["dataset"].dropna().unique()):
        ds_df = tdf[tdf["dataset"] == ds]
        hc = ds_df.loc[ds_df["condition"] == "HC", "pct_cxcr3_pos"]
        ms = ds_df.loc[ds_df["condition"] == "MS", "pct_cxcr3_pos"]
        if len(hc) < min_patients_per_group or len(ms) < min_patients_per_group:
            print(f"    [{ds}] n={len(hc)} HC, {len(ms)} MS -- too few patients to test "
                  f"(< {min_patients_per_group}/group); skipping.")
            per_dataset_results.append({"dataset": ds, "n_HC": len(hc), "n_MS": len(ms),
                                         "median_diff": np.nan, "pvalue": np.nan})
            continue
        stat, p = mannwhitneyu(hc, ms, alternative="two-sided")
        median_diff = ms.median() - hc.median()
        print(f"    [{ds}] n={len(hc)} HC, {len(ms)} MS: median(MS-HC)={median_diff:+.2f} pct-pts, "
              f"Mann-Whitney p={p:.3g}")
        per_dataset_results.append({"dataset": ds, "n_HC": len(hc), "n_MS": len(ms),
                                     "median_diff": median_diff, "pvalue": p})

    per_dataset_df = pd.DataFrame(per_dataset_results)
    testable = per_dataset_df.dropna(subset=["pvalue"])
    directions_agree = testable["median_diff"].apply(np.sign).nunique() <= 1 if len(testable) > 1 else None
    if directions_agree is False:
        print(f"    [WARNING] Direction of the HC-vs-MS %CXCR3+ difference is INCONSISTENT "
              f"across datasets -- investigate before reporting the pooled bracket p-value as "
              f"dataset-independent.")
    elif directions_agree is True:
        print(f"    Direction is consistent across all testable datasets.")

    interaction_p = np.nan
    if tdf["dataset"].nunique() >= 2 and tdf["condition"].nunique() >= 2:
        try:
            ols = smf.ols("pct_cxcr3_pos ~ C(condition) * C(dataset)", data=tdf).fit()
            interaction_terms = [t for t in ols.pvalues.index if ":" in t]
            if interaction_terms:
                interaction_p = ols.pvalues[interaction_terms[0]]
                verdict = "SIGNIFICANT interaction -- effect differs by dataset" if interaction_p < 0.05 \
                    else "no significant interaction -- effect is consistent across datasets"
                print(f"    Interaction test (condition x dataset): p={interaction_p:.3g} ({verdict})")
        except Exception as exc:  # noqa: BLE001
            print(f"    [warn] interaction model failed to fit: {exc}")

    return {"tissue": tissue, "per_dataset": per_dataset_df,
            "directions_agree": directions_agree, "interaction_pvalue": interaction_p}


def main() -> None:
    adata = sc.read_h5ad(RESULTS_DIR / "merged_bcells.h5ad")

    # Ensure unique patient IDs across datasets (028, 266, etc.)
    adata.obs["dataset"] = adata.obs["dataset"].astype(str)
    adata.obs["patient"] = adata.obs["patient"].astype(str)

    # OPTIONAL but recommended if IDs may collide
    adata.obs["patient"] = (
    adata.obs["dataset"] + "_" + adata.obs["patient"]
    )

    # Clean obs dtypes
    adata.obs["tissue"]      = adata.obs["tissue"].replace({"PBMCs": "PB", "PBMC": "PB"}).astype(str)
    adata.obs["condition"]   = adata.obs["condition"].astype(str)
    adata.obs["cxcr3_group"] = adata.obs["cxcr3_group"].astype(str)
    adata.obs["dataset"]     = adata.obs["dataset"].astype(str)
    adata.obs["group4"]      = adata.obs["condition"] + "-" + adata.obs["tissue"]

    order = ORDER_BAR  # ["HC-PB", "HC-CSF", "MS-PB", "MS-CSF"]

    # ── Patient-level proportions ─────────────────────────────────────────
    patient_df = (
        adata.obs
        .groupby(["patient", "condition", "tissue", "dataset"])["cxcr3_group"]
        .apply(lambda x: (x == "CXCR3+").mean() * 100)
        .reset_index(name="pct_cxcr3_pos")
    )
    patient_df["group4"] = patient_df["condition"] + "-" + patient_df["tissue"]
    patient_df = patient_df[patient_df["group4"].isin(order)].copy()
    patient_df["group4"] = pd.Categorical(patient_df["group4"], categories=order, ordered=True)
    patient_df.sort_values("group4", inplace=True)

    # Quick sanity check: dataset composition per group4
    comp = (
        adata.obs
        .groupby(["dataset", "condition", "tissue"])["patient"]
        .nunique()
        .reset_index(name="n_patients")
    )
    print("\nPatients per dataset / condition / tissue:")
    print(comp)


    # Show MS patients from GSE138266 specifically
    ms_266 = (
        adata.obs
        .loc[adata.obs["dataset"] == "GSE138266"]
        .groupby(["patient", "tissue"])["cxcr3_group"]
        .size()
        .reset_index(name="n_cells")
    )
    print("\nGSE138266 MS patients (PB/CSF presence):")
    print(ms_266)

    bar_df = (
        patient_df.groupby("group4", observed=True)["pct_cxcr3_pos"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    bar_df["sem"] = bar_df["std"] / np.sqrt(bar_df["count"])

    # ── Mann-Whitney p-values (HC vs MS within tissue) ────────────────────
    def _mwu(g1: str, g2: str) -> float:
        v1 = patient_df.loc[patient_df["group4"] == g1, "pct_cxcr3_pos"]
        v2 = patient_df.loc[patient_df["group4"] == g2, "pct_cxcr3_pos"]
        if len(v1) and len(v2):
            return mannwhitneyu(v1, v2, alternative="two-sided").pvalue
        return np.nan

    pvals = {
        "PB":  _mwu("HC-PB",  "MS-PB"),
        "CSF": _mwu("HC-CSF", "MS-CSF"),
    }
    for tissue, p in pvals.items():
        msg = f"{p:.4f}" if not np.isnan(p) else "N/A"
        print(f"  Mann-Whitney U  HC vs MS  {tissue}: p = {msg}")

    # Dataset-of-origin confounding check (reviewer concern #4): does the
    # HC-vs-MS %CXCR3+ difference hold within each dataset separately?
    confound_pb = check_bar_dataset_confounding(patient_df, tissue="PB")
    confound_csf = check_bar_dataset_confounding(patient_df, tissue="CSF")
    confound_pb["per_dataset"].to_csv(
        RESULTS_DIR / "Fig1_PB_CXCR3pct_dataset_confounding_check.csv", index=False
    )
    confound_csf["per_dataset"].to_csv(
        RESULTS_DIR / "Fig1_CSF_CXCR3pct_dataset_confounding_check.csv", index=False
    )
    pd.DataFrame([
        {"tissue": "PB", "directions_agree": confound_pb["directions_agree"],
         "interaction_pvalue": confound_pb["interaction_pvalue"]},
        {"tissue": "CSF", "directions_agree": confound_csf["directions_agree"],
         "interaction_pvalue": confound_csf["interaction_pvalue"]},
    ]).to_csv(RESULTS_DIR / "Fig1_dataset_confounding_summary.csv", index=False)

    # ── Individual panel PDFs ─────────────────────────────────────────────
    fig1_dir = FIG_DIRS["fig1"]
    save_panel_dataset_umap(adata,   fig1_dir / "fig1A_umap_dataset.pdf")
    save_panel_condition_umap(adata, fig1_dir / "fig1B_umap_condition.pdf")
    save_panel_cxcr3_tissue_umap(adata, fig1_dir / "fig1C_umap_cxcr3_tissue.pdf")
    save_panel_bar(patient_df, bar_df, order, pvals, fig1_dir / "fig1D_bar_cxcr3_percent.pdf")

    # ── Combined 2×2 figure ───────────────────────────────────────────────
    _save_combined_figure(
        adata, patient_df, bar_df, order, pvals,
        fig1_dir / "figure1_umap_bar.png",
    )

    # ── CSV outputs ───────────────────────────────────────────────────────
    bar_df.to_csv(RESULTS_DIR / "figure1_bar_values.csv",         index=False)
    patient_df.to_csv(RESULTS_DIR / "figure1_bar_patient_values.csv", index=False)

    print("\nDone.")


if __name__ == "__main__":
    main()