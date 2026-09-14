#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
FigS7.py — CXCR3+ vs CXCR3- comparison within CSF, Supplementary Figure 7

PB and CSF are treated in SEPARATE files (FigS6 = PB, FigS7 = CSF) rather
than one combined figure, because this comparison's paired axis is CXCR3
status, not tissue -- a patient's PB CXCR3+ cells only pair meaningfully
with that SAME patient's PB CXCR3- cells (both being subject to whatever
is specific about blood as a compartment); pairing across tissues here
would conflate the CXCR3 effect with a compartment effect. So each tissue
gets its own self-contained paired analysis and figure.

STATISTICS -- patient-level, paired: CXCR3+ and CXCR3- cells within a
given tissue come from the SAME patients (unlike Fig 2's MS vs HC, where
each patient belongs to only one group). Per-cell tests would both
(a) pseudoreplicate -- treating every cell as independent when cells are
correlated within a patient -- and (b) ignore the pairing entirely,
throwing away the fact that a within-patient CXCR3+ vs CXCR3- shift is
what's biologically meaningful here (same rationale as the PB-vs-CSF
paired analysis in FigS4.py/FigS5.py, just with the paired axis being
CXCR3 status instead of tissue).

Both per-gene DE and the composite signature test use the PAIRED
pseudobulk forms from stats_utils.py (paired_pseudobulk_de,
paired_pseudobulk_signature_test), which aggregate cells to one value per
(patient, cxcr3_group), then run a Wilcoxon signed-rank test on each
patient's own CXCR3+ vs CXCR3- difference.

Unlike the PB-vs-CSF comparison, NO subject-ID resolution is needed here:
a patient's PB CXCR3+ cells and PB CXCR3- cells come from the exact same
sample/patient-key (the CXCR3 gate is applied within a single tissue
sample), so the 'patient' column already pairs correctly without any
tissue-token stripping.

Composite zinc/MT signature is sign-corrected (+1 transporter, -1
metallothionein, from compute_gene_signs) as the PRIMARY result here --
unlike the PB-vs-CSF comparison (which defaults to unsigned, no a priori
directional hypothesis for a compartment shift), CXCR3+ vs CXCR3- is
analogous to the MS-vs-HC disease-severity axis (Fig 2): CXCR3+ effector
cells are hypothesized to show the same transporter-up/MT-down pattern
associated with disease, so the same a priori sign scheme applies. Unsigned
is still computed and shown in the supplementary comparison panel.

6-panel figure layout:
  A  Volcano: MS PB B cells – CXCR3+ vs CXCR3-
  B  Barplot: MS Zinc/MT genes (CXCR3+ vs CXCR3-)
  C  Volcano: HC PB B cells – CXCR3+ vs CXCR3-
  D  Barplot: HC Zinc/MT genes (CXCR3+ vs CXCR3-)
  E  Paired slope plot: HC CXCR3+ vs CXCR3- zinc signature (each line = one
     patient's own CXCR3- -> CXCR3+ trajectory, color-coded by direction)
  F  Paired slope plot: MS CXCR3+ vs CXCR3- zinc signature (same as E)
"""

import scipy.sparse as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
from statsmodels.stats.multitest import multipletests

from config.config import RESULTS_DIR, FIG_DIRS, DATASETS_YAML
from data_io import load_cfg
from stats_utils import (
    paired_pseudobulk_de,
    paired_pseudobulk_signature_test,
    compute_gene_signs,
)

TISSUE = "CSF"  # the only thing that differs from FigS6.py (which uses "PB")


# ── Global style ─────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        10,
    "axes.labelsize":   10,
    "axes.titlesize":   11,
    "axes.titleweight": "bold",
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  8,
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
})

SCATTER_COLOR_OTHER    = "#CCCCCC"
SCATTER_COLOR_ZINC_CMP = "#4C78A8"
BAR_COLOR_CMP          = "#4C78A8"
SIG_PADJ_THRESHOLD     = 0.05
MIN_PAIRS              = 3
PANEL_KW               = dict(fontsize=14, fontweight="bold", va="top", ha="left")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _panel_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(-0.12, 1.05, letter, transform=ax.transAxes, **PANEL_KW)


def load_zinc_gene_list() -> list[str]:
    cfg = load_cfg(DATASETS_YAML)
    markers = cfg.get("markers", {})
    artefact = set(markers.get("exclude_artefacts", []))
    all_g = markers.get("zinc_transporters", []) + markers.get("metallothioneins", [])
    return [g for g in all_g if g not in artefact]


def load_zinc_gene_signs() -> dict[str, float]:
    """
    A priori sign for each gene in the zinc/MT panel: +1 for transporters,
    -1 for metallothioneins. See stats_utils.compute_gene_signs for the full
    rationale (signs come from prior biological role, not from any DE result,
    to avoid circularity).
    """
    cfg = load_cfg(DATASETS_YAML)
    markers = cfg.get("markers", {})
    artefact = set(markers.get("exclude_artefacts", []))
    transporters = [g for g in markers.get("zinc_transporters", []) if g not in artefact]
    metallothioneins = [g for g in markers.get("metallothioneins", []) if g not in artefact]
    return compute_gene_signs(transporters, metallothioneins)


def _apply_min_pct_filter_cxcr3(
    sub: sc.AnnData,
    min_pct: float = 0.10,
    always_keep: list[str] | None = None,
) -> sc.AnnData:
    """
    Keep genes expressed in at least one CXCR3 group (+/-) above min_pct.
    This is a per-cell prevalence filter (deciding which genes are worth
    testing at all), not a statistical test itself, so it's unaffected by
    the pseudoreplication fix.

    `always_keep` (the curated zinc/MT panel) is exempted from this filter.
    Without this, a gene can pass the prevalence filter in one condition
    subset (e.g. HC) but fail it in another (e.g. MS), since each
    _run_de_cxcr3pos_vs_neg_paired call filters independently -- silently
    producing a DIFFERENT gene set in each bar-chart panel, which is
    confusing and makes the two panels not directly comparable.
    """
    always_keep = set(always_keep or [])
    X = sub.X if not sp.issparse(sub.X) else sub.X.toarray()
    pos = sub.obs["cxcr3_group"] == "CXCR3+"
    neg = sub.obs["cxcr3_group"] == "CXCR3-"

    pct_pos = (X[pos] > 0).mean(axis=0)
    pct_neg = (X[neg] > 0).mean(axis=0)
    keep = (pct_pos >= min_pct) | (pct_neg >= min_pct)

    forced_idx = [sub.var_names.get_loc(g) for g in always_keep if g in sub.var_names]
    n_forced_back_in = int((~keep[forced_idx]).sum()) if forced_idx else 0
    for idx in forced_idx:
        keep[idx] = True

    print(f"    min_pct={min_pct:.0%}: {keep.sum()}/{len(keep)} genes retained"
          + (f" ({n_forced_back_in} curated zinc/MT gene(s) kept despite low prevalence)"
             if n_forced_back_in else ""))
    return sub[:, keep].copy()


def compute_cell_signature_cxcr3(
    adata: sc.AnnData,
    zinc_genes: list[str],
    gene_signs: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Per-cell zinc/MT signature in {TISSUE} B cells. Columns:
        cell_id, cxcr3_group, condition, patient, zinc_signature

    Saved as supplementary per-cell CSV output only -- the main figure's
    paired slope plot (draw_paired_patient_plot_cxcr3) uses patient-level
    pseudobulk values from patient_signature_test_cxcr3's per_patient_values
    instead, not this per-cell data directly.

    gene_signs: if given, sign-corrects each gene (+1 transporter, -1
    metallothionein) before averaging -- see stats_utils.py. SIGNED is the
    PRIMARY choice for this comparison (unlike PB-vs-CSF): CXCR3+ effector
    cells are hypothesized to show the same transporter-up/MT-down pattern
    associated with disease (Fig 2's MS-vs-HC axis), so the same a priori
    disease-direction sign scheme applies here. Pass None for the unsigned
    (plain mean) variant, computed for the supplementary comparison.
    """
    sub = adata[adata.obs["tissue"] == TISSUE].copy()

    genes = [g for g in zinc_genes if g in sub.var_names]
    X = sub[:, genes].X
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X)

    if gene_signs is not None:
        sign_vec = np.array([gene_signs.get(g, 1.0) for g in genes])
        sig = (X * sign_vec[np.newaxis, :]).mean(axis=1)
    else:
        sig = X.mean(axis=1)

    df = sub.obs[["cxcr3_group", "condition", "patient"]].copy()
    df["cell_id"] = sub.obs_names
    df["zinc_signature"] = sig
    df = df[df["cxcr3_group"].isin(["CXCR3+", "CXCR3-"])].copy()
    return df


def patient_signature_test_cxcr3(
    adata: sc.AnnData,
    zinc_genes: list[str],
    group_label: str,
    gene_signs: dict[str, float] | None = None,
) -> dict:
    """
    Paired, patient-level test of the composite zinc/MT signature for
    CXCR3+ vs CXCR3- {TISSUE} cells within one group (e.g. 'HC' or 'MS').
    n = number of patients with both CXCR3+ and CXCR3- cells present, not
    number of cells.

    gene_signs: see compute_cell_signature_cxcr3 -- a sign dict (signed) is
    the PRIMARY/default choice for this comparison; pass None for the
    unsigned/plain-mean supplementary variant.
    """
    sub = adata[(adata.obs["tissue"] == TISSUE) & (adata.obs["condition"] == group_label)].copy()
    genes = [g for g in zinc_genes if g in sub.var_names]

    return paired_pseudobulk_signature_test(
        sub,
        groupby="cxcr3_group",
        group1="CXCR3-",
        group2="CXCR3+",
        signature_genes=genes,
        patient_col="patient",
        use_raw=False,  # sub.X here is the same matrix compute_cell_signature_cxcr3 uses
        gene_signs=gene_signs,
        min_pairs=MIN_PAIRS,
    )


def check_cxcr3_dataset_confounding(
    adata: sc.AnnData,
    zinc_genes: list[str],
    group_label: str,
    gene_signs: dict[str, float] | None = None,
    min_pairs_per_group: int = 3,
) -> dict:
    """
    Check whether the paired CXCR3+ vs CXCR3- composite signature result is
    being driven by, or differs between, dataset of origin -- same
    rationale as fig2.py's check_dataset_confounding and
    plot_figure1_umap_bar.py's check_bar_dataset_confounding (reviewer
    concern #4), adapted for this paired design.

    Two checks, both operating on each patient's own (CXCR3+ - CXCR3-)
    difference rather than raw values:

    1. STRATIFIED: rerun the same paired Wilcoxon signed-rank test WITHIN
       each dataset separately.
    2. BETWEEN-DATASET COMPARISON OF THE PAIRED DIFFERENCES: an unpaired
       Mann-Whitney U test comparing each patient's (CXCR3+ - CXCR3-)
       difference between datasets. A significant result means the
       MAGNITUDE of the within-patient CXCR3 shift itself differs by
       dataset -- direct evidence of confounding in a paired design.
    """
    from scipy.stats import wilcoxon, mannwhitneyu

    sub = adata[(adata.obs["tissue"] == TISSUE) & (adata.obs["condition"] == group_label)].copy()
    genes = [g for g in zinc_genes if g in sub.var_names]

    X = sub[:, genes].X
    if sp.issparse(X):
        X = X.toarray()
    sign_vec = np.array([(gene_signs.get(g, 1.0) if gene_signs is not None else 1.0) for g in genes])
    signature = (np.asarray(X) * sign_vec[np.newaxis, :]).mean(axis=1)

    df = pd.DataFrame({
        "patient": sub.obs["patient"].values,
        "cxcr3_group": sub.obs["cxcr3_group"].values,
        "dataset": sub.obs["dataset"].values,
        "signature": signature,
    })
    pb_by_patient_group = df.groupby(["patient", "cxcr3_group"])["signature"].mean()
    wide = pb_by_patient_group.unstack("cxcr3_group")
    wide = wide.dropna(subset=[c for c in ["CXCR3-", "CXCR3+"] if c in wide.columns])

    patient_dataset = df.drop_duplicates("patient").set_index("patient")["dataset"]
    wide["dataset"] = patient_dataset.reindex(wide.index)
    wide["diff"] = wide["CXCR3+"] - wide["CXCR3-"]

    label = "signed" if gene_signs is not None else "unsigned"
    print(f"\n  === Dataset confounding check: CXCR3+ vs CXCR3- paired, {TISSUE}, "
          f"{group_label}, {label} signature ===")

    per_dataset_results = []
    for ds in sorted(wide["dataset"].dropna().unique()):
        ds_wide = wide[wide["dataset"] == ds]
        n_pairs = len(ds_wide)
        if n_pairs < min_pairs_per_group:
            print(f"    [{ds}] n={n_pairs} paired patients -- too few to test "
                  f"(< {min_pairs_per_group}); skipping.")
            per_dataset_results.append({"dataset": ds, "n_pairs": n_pairs,
                                         "median_diff": np.nan, "pvalue": np.nan})
            continue
        x, y = ds_wide["CXCR3-"].values, ds_wide["CXCR3+"].values
        if np.all(x == y):
            stat, p = np.nan, 1.0
        else:
            stat, p = wilcoxon(x, y)
        median_diff = ds_wide["diff"].median()
        print(f"    [{ds}] n={n_pairs} paired patients: median(CXCR3+ - CXCR3-)={median_diff:+.3f}, "
              f"Wilcoxon signed-rank p={p:.3g}")
        per_dataset_results.append({"dataset": ds, "n_pairs": n_pairs,
                                     "median_diff": median_diff, "pvalue": p})

    per_dataset_df = pd.DataFrame(per_dataset_results)
    testable = per_dataset_df.dropna(subset=["pvalue"])
    directions_agree = testable["median_diff"].apply(np.sign).nunique() <= 1 if len(testable) > 1 else None
    if directions_agree is False:
        print(f"    [WARNING] Direction of the paired CXCR3+ vs CXCR3- difference is "
              f"INCONSISTENT across datasets -- investigate before reporting the pooled "
              f"paired test as dataset-independent.")
    elif directions_agree is True:
        print(f"    Direction is consistent across all testable datasets.")

    between_ds_p = np.nan
    ds_groups = [g["diff"].values for _, g in wide.groupby("dataset") if len(g) >= min_pairs_per_group]
    if len(ds_groups) == 2:
        stat, between_ds_p = mannwhitneyu(ds_groups[0], ds_groups[1], alternative="two-sided")
        verdict = "SIGNIFICANT -- the magnitude of the paired shift differs by dataset" if between_ds_p < 0.05 \
            else "not significant -- paired shift magnitude is consistent across datasets"
        print(f"    Between-dataset comparison of (CXCR3+ - CXCR3-) differences: "
              f"p={between_ds_p:.3g} ({verdict})")
    elif len(ds_groups) > 2:
        print(f"    [note] >2 datasets with enough pairs -- between-dataset Mann-Whitney only "
              f"handles 2; use per_dataset table above for a >2-dataset comparison.")

    return {
        "group_label": group_label, "signature_label": label,
        "per_dataset": per_dataset_df,
        "directions_agree": directions_agree,
        "between_dataset_pvalue": between_ds_p,
    }


def _run_de_cxcr3_paired(
    adata: sc.AnnData,
    zinc_genes: list[str],
    conditions: list[str],
    label: str,
    min_pct: float = 0.10,
) -> pd.DataFrame | None:
    """
    Paired DE for CXCR3+ vs CXCR3- in {TISSUE} B cells, within the given
    set of raw `condition` values (e.g. ["HC"], or ["MS"]). Returns
    CXCR3+ vs CXCR3-, so positive log2FC means higher in CXCR3+.

    Replaces sc.tl.rank_genes_groups (unpaired, per-cell) with
    paired_pseudobulk_de (paired, per-patient): the same patient
    contributes both CXCR3+ and CXCR3- cells here, so this is a
    within-patient comparison.
    """
    print(f"\nRunning paired DE: CXCR3+ vs CXCR3- {TISSUE}, conditions={conditions} …")

    sub = adata[
        (adata.obs["tissue"] == TISSUE)
        & adata.obs["condition"].isin(conditions)
        & adata.obs["cxcr3_group"].isin(["CXCR3+", "CXCR3-"])
    ].copy()

    if sub.n_obs == 0 or not {"CXCR3+", "CXCR3-"}.issubset(set(sub.obs["cxcr3_group"].unique())):
        print(f"  [skip] {label}: no {TISSUE} CXCR3+/- cells for conditions {conditions}.")
        return None

    n_patients_pos = sub.obs.loc[sub.obs["cxcr3_group"] == "CXCR3+", "patient"].nunique()
    n_patients_neg = sub.obs.loc[sub.obs["cxcr3_group"] == "CXCR3-", "patient"].nunique()
    n_both = len(set(sub.obs.loc[sub.obs["cxcr3_group"] == "CXCR3+", "patient"])
                 & set(sub.obs.loc[sub.obs["cxcr3_group"] == "CXCR3-", "patient"]))
    print(f"    {label}: {n_patients_pos} patients w/ CXCR3+, {n_patients_neg} w/ CXCR3-, "
          f"{n_both} with BOTH ({sub.n_obs} cells total)")
    if n_both < MIN_PAIRS:
        print(f"    [skip] {label}: only {n_both} paired patients (< {MIN_PAIRS}) -- "
              f"not enough for a paired test.")
        return None

    if not sub.var_names.is_unique:
        print(f"    [fix] {label}: duplicate gene symbols in var_names -- making unique")
        sub.var_names_make_unique()

    missing_from_data = [g for g in zinc_genes if g not in sub.var_names]
    if missing_from_data:
        print(f"    [warn] {label}: {len(missing_from_data)} curated zinc/MT gene(s) not in "
              f"the dataset at all: {missing_from_data}")

    sub = _apply_min_pct_filter_cxcr3(sub, min_pct=min_pct, always_keep=zinc_genes)

    df = paired_pseudobulk_de(
        sub,
        groupby="cxcr3_group",
        group1="CXCR3-",
        group2="CXCR3+",
        patient_col="patient",
        min_pairs=MIN_PAIRS,
    )
    if df.empty:
        print(f"    [skip] {label}: paired DE returned no genes with enough pairs.")
        return None

    df = df.rename(columns={"log2fc": "log2FC", "pval": "pvalue"})
    df = df[np.isfinite(df["log2FC"])].copy()
    df["is_zinc"] = df["gene"].isin(zinc_genes)

    # Two FDR corrections, same rationale as fig2.py's _run_de:
    #   'padj'            -- genome-wide (all tested genes), used for the
    #                        volcano plot (exploratory scan).
    #   'padj_zinc_panel' -- BH correction restricted to ONLY the curated
    #                        zinc/MT genes, used for the bar-chart stars.
    zinc_mask = df["is_zinc"].values
    df["padj_zinc_panel"] = np.nan
    if zinc_mask.sum() > 0:
        df.loc[zinc_mask, "padj_zinc_panel"] = multipletests(
            df.loc[zinc_mask, "pvalue"].fillna(1.0), method="fdr_bh"
        )[1]

    return df


def _select_genes_for_bars(df: pd.DataFrame, zinc_genes: list[str]) -> tuple[list[str], list[str]]:
    """
    Bar-chart gene list = the FULL curated zinc/MT panel (every gene that
    was actually tested -- see _apply_min_pct_filter_cxcr3's always_keep),
    sorted by log2FC descending. Any curated gene missing from `df` entirely
    (not in the dataset) is returned separately so the caller can report it.
    """
    tested = df[df["gene"].isin(zinc_genes)].copy()
    tested = tested.sort_values("log2FC", ascending=False)
    genes_present = tested["gene"].tolist()
    genes_missing = [g for g in zinc_genes if g not in set(genes_present)]
    return genes_present, genes_missing


def _draw_volcano_cxcr3(
    ax: plt.Axes,
    df: pd.DataFrame,
    title: str,
    label: str,
) -> None:
    df = df.copy()
    df["-log10padj"] = -np.log10(df["padj"].clip(lower=1e-300))

    other = df[~df["is_zinc"]]
    zinc = df[df["is_zinc"]]
    zinc_sig = zinc[zinc["padj"] < SIG_PADJ_THRESHOLD]
    zinc_ns = zinc[zinc["padj"] >= SIG_PADJ_THRESHOLD]

    ax.scatter(
        other["log2FC"], other["-log10padj"],
        s=5, color=SCATTER_COLOR_OTHER, alpha=0.35,
        rasterized=True, label="Other genes",
    )
    if not zinc_ns.empty:
        ax.scatter(
            zinc_ns["log2FC"], zinc_ns["-log10padj"],
            s=28, color=SCATTER_COLOR_ZINC_CMP, alpha=0.4,
            edgecolors="black", linewidths=0.4,
            label="Zinc/MT (ns)",
        )
    if not zinc_sig.empty:
        ax.scatter(
            zinc_sig["log2FC"], zinc_sig["-log10padj"],
            s=45, color=SCATTER_COLOR_ZINC_CMP, alpha=0.9,
            edgecolors="black", linewidths=0.6,
            label=f"Zinc/MT (padj < {SIG_PADJ_THRESHOLD})",
        )

    ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.axhline(-np.log10(SIG_PADJ_THRESHOLD), color="grey", linestyle=":", linewidth=0.8)

    ax.set_xlabel("log2FC (CXCR3+ vs CXCR3-)")
    ax.set_ylabel("−log10(padj)")
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper left", fontsize=7)
    _panel_letter(ax, label)


def _sig_label(padj_val: float) -> str | None:
    if pd.isna(padj_val):
        return None
    if padj_val < 0.001:
        return "***"
    if padj_val < 0.01:
        return "**"
    if padj_val < SIG_PADJ_THRESHOLD:
        return "*"
    return None


def _draw_bars_cxcr3(
    ax: plt.Axes,
    df: pd.DataFrame,
    genes: list[str],
    title: str,
    label: str,
    color: str = BAR_COLOR_CMP,
    outlier_cap: float | None = None,
) -> None:
    if not genes:
        ax.text(
            0.5, 0.5,
            "No zinc/MT genes\npassing padj threshold",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=9, color="grey",
        )
        ax.set_title(title)
        _panel_letter(ax, label)
        return

    df_b = df[df["gene"].isin(genes)].copy()
    df_b["gene"] = pd.Categorical(df_b["gene"], categories=genes, ordered=True)
    df_b = df_b.sort_values("gene").reset_index(drop=True)

    x = np.arange(len(genes))
    fc_raw = df_b.set_index("gene").loc[genes, "log2FC"].values
    # Stars use padj corrected within the curated zinc/MT panel only -- NOT
    # the genome-wide padj used for the volcano's threshold line.
    padj_v = df_b.set_index("gene").loc[genes, "padj_zinc_panel"].values

    fc_plot = fc_raw.copy()
    capped_mask = np.zeros(len(fc_raw), dtype=bool)
    if outlier_cap is not None:
        capped_mask = np.abs(fc_raw) > outlier_cap
        fc_plot[capped_mask] = np.sign(fc_raw[capped_mask]) * outlier_cap * 0.9

    for i in range(len(genes)):
        hatch = "//" if capped_mask[i] else None
        ax.bar(
            x[i], fc_plot[i], width=0.65,
            color=color, edgecolor="black", linewidth=0.6,
            hatch=hatch,
        )

    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)

    for i in np.where(capped_mask)[0]:
        true_val = fc_raw[i]
        cap_val = fc_plot[i]
        direction = np.sign(cap_val)
        ax.annotate(
            f"{true_val:.1f}",
            xy=(x[i], cap_val),
            xytext=(x[i], cap_val + direction * abs(cap_val) * 0.08),
            ha="center", va="bottom" if direction > 0 else "top",
            fontsize=8, color=color, fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", lw=1, color=color),
            clip_on=False,
        )

    y_range = max(np.abs(fc_plot).max(), 0.5)
    y_offset = y_range * 0.04
    for i in range(len(genes)):
        sig = _sig_label(padj_v[i])
        if sig is None:
            continue
        y_pos = max(fc_plot[i], 0) + y_offset
        ax.text(i, y_pos, sig, ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(genes, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("log2FC (CXCR3+ vs CXCR3-)")
    ax.set_xlabel("")
    ax.set_title(title, fontsize=10)
    _panel_letter(ax, label)


def draw_paired_patient_plot_cxcr3(ax, sig_test_result, group_label, letter, signed: bool = True):
    """
    Paired slope plot of PATIENT-LEVEL pseudobulk composite signature
    values: each patient's own CXCR3- and CXCR3+ pseudobulk values
    connected by a line, so the within-patient shift the paired Wilcoxon
    signed-rank test actually operates on is directly visible.

    This is deliberately NOT two independent-looking distributions side by
    side -- CXCR3+ and CXCR3- cells here come from the SAME patients, and
    the paired test only cares about each patient's own CXCR3- -> CXCR3+
    difference, not the two groups' marginal distributions in isolation.

    `signed` defaults to True here (unlike the PB-vs-CSF paired plot),
    since the sign-corrected composite is the PRIMARY result for this
    comparison -- see compute_cell_signature_cxcr3's docstring.
    """
    per_patient = sig_test_result.get("per_patient_values")
    p = sig_test_result.get("wilcoxon_p", np.nan)
    n_pairs = sig_test_result.get("n_pairs", "?")

    if (per_patient is None or per_patient.empty
            or "CXCR3-" not in per_patient.columns or "CXCR3+" not in per_patient.columns):
        ax.text(0.5, 0.5, "No paired per-patient data available", ha="center", va="center",
                transform=ax.transAxes)
        _panel_letter(ax, letter)
        return

    neg_vals = per_patient["CXCR3-"].values
    pos_vals = per_patient["CXCR3+"].values

    ax.boxplot(
        [neg_vals, pos_vals], positions=[1, 2], widths=0.5,
        showfliers=False, patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        boxprops=dict(facecolor="#D9D9D9", alpha=0.5, edgecolor="black"),
        whiskerprops=dict(color="black"), capprops=dict(color="black"),
        zorder=1,
    )

    rng = np.random.default_rng(42)
    jitter = rng.normal(0, 0.04, size=len(neg_vals))
    for i in range(len(neg_vals)):
        direction_color = "#4C78A8" if pos_vals[i] >= neg_vals[i] else "#E45756"
        ax.plot(
            [1 + jitter[i], 2 + jitter[i]], [neg_vals[i], pos_vals[i]],
            color=direction_color, alpha=0.5, linewidth=1.2, zorder=5,
        )
        ax.scatter(
            [1 + jitter[i], 2 + jitter[i]], [neg_vals[i], pos_vals[i]],
            color=direction_color, s=24, alpha=0.8, edgecolors="black", linewidths=0.3, zorder=10,
        )

    if np.isfinite(p):
        print(
            f"[{group_label}] PAIRED PATIENT-LEVEL plot: n={n_pairs} patient pairs "
            f"(each line = one patient's CXCR3- -> CXCR3+ trajectory), Wilcoxon signed-rank p={p:.3e}"
        )
    else:
        print(f"[{group_label}] n={n_pairs} pairs -- too few to test")

    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"CXCR3-\n(n={n_pairs})", f"CXCR3+\n(n={n_pairs})"])
    score_label = "signed" if signed else "unsigned (plain mean)"
    ax.set_ylabel(f"Patient-level mean zinc/MT signature ({score_label})")
    title_p = f"{p:.2e}" if np.isfinite(p) else "n/a (too few paired patients)"
    ax.set_title(
        f"{group_label} {TISSUE} – CXCR3+ vs CXCR3- ({score_label})\n"
        f"Paired Wilcoxon signed-rank p = {title_p} (n={n_pairs} patient pairs)",
        fontsize=9,
    )
    ax.spines[["top", "right"]].set_visible(False)
    _panel_letter(ax, letter)


def _save_single_volcano_cxcr3(df, title, label, out_png, out_pdf):
    fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))
    _draw_volcano_cxcr3(ax, df, title=title, label=label)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def make_figS7_cxcr3_CSF() -> None:
    merged_path = RESULTS_DIR / "merged_bcells.h5ad"
    if not merged_path.exists():
        raise FileNotFoundError(f"merged_bcells.h5ad not found at {merged_path}")

    adata = sc.read_h5ad(merged_path)
    if not adata.obs_names.is_unique:
        print("[fix] Making obs_names unique")
        adata.obs_names_make_unique()

    for col in ("condition", "cxcr3_group", "tissue", "patient"):
        if col not in adata.obs.columns:
            raise KeyError(f"Column '{col}' missing in adata.obs.")

    out_dir = FIG_DIRS.get("supp_cxcr3_paired_csf", RESULTS_DIR / "supp" / "cxcr3_paired_csf")
    out_dir.mkdir(parents=True, exist_ok=True)

    zinc_genes = load_zinc_gene_list()
    gene_signs = load_zinc_gene_signs()
    print(f"Loaded {len(zinc_genes)} curated zinc/MT genes: {zinc_genes}")
    print(f"Gene signs for composite signature: {gene_signs}")

    # MS paired DE
    df_de_ms = _run_de_cxcr3_paired(
        adata, zinc_genes=zinc_genes, conditions=["MS"], label="MS", min_pct=0.10,
    )
    bar_genes_ms: list[str] = []
    if df_de_ms is not None:
        print(f"  {df_de_ms.shape[0]} genes in CXCR3+ vs CXCR3- DE table (MS)")
        bar_genes_ms, ms_missing = _select_genes_for_bars(df_de_ms, zinc_genes=zinc_genes)
        print(f"MS CXCR3+ vs CXCR3- bar genes ({len(bar_genes_ms)}/{len(zinc_genes)} of curated panel):", bar_genes_ms)
        if ms_missing:
            print(f"  [note] MS: {len(ms_missing)} curated gene(s) not in dataset: {ms_missing}")

    # HC paired DE
    df_de_hc = _run_de_cxcr3_paired(
        adata, zinc_genes=zinc_genes, conditions=["HC"], label="HC", min_pct=0.10,
    )
    bar_genes_hc: list[str] = []
    if df_de_hc is not None:
        print(f"  {df_de_hc.shape[0]} genes in CXCR3+ vs CXCR3- DE table (HC)")
        bar_genes_hc, hc_missing = _select_genes_for_bars(df_de_hc, zinc_genes=zinc_genes)
        print(f"HC CXCR3+ vs CXCR3- bar genes ({len(bar_genes_hc)}/{len(zinc_genes)} of curated panel):", bar_genes_hc)
        if hc_missing:
            print(f"  [note] HC: {len(hc_missing)} curated gene(s) not in dataset: {hc_missing}")

    if df_de_ms is not None and df_de_hc is not None:
        set_diff = set(bar_genes_ms).symmetric_difference(set(bar_genes_hc))
        if set_diff:
            print(f"  [WARNING] HC and MS bar-chart gene sets differ: {sorted(set_diff)} -- panels are not directly comparable.")

    # Compute signatures (SIGNED is primary for this figure -- see
    # compute_cell_signature_cxcr3's docstring for why, unlike PB-vs-CSF)
    df_sig = compute_cell_signature_cxcr3(
        adata, zinc_genes=zinc_genes, gene_signs=gene_signs
    )

    if "HC" in df_sig["condition"].unique():
        df_sig[df_sig["condition"] == "HC"].to_csv(
            out_dir / f"FigS7_CXCR3_{TISSUE}_HC_zinc_signature_per_cell.csv",
            index=False
        )
    if "MS" in df_sig["condition"].unique():
        df_sig[df_sig["condition"] == "MS"].to_csv(
            out_dir / f"FigS7_CXCR3_{TISSUE}_MS_zinc_signature_per_cell.csv",
            index=False
        )

    sig_test_hc = patient_signature_test_cxcr3(
        adata, zinc_genes=zinc_genes, gene_signs=gene_signs, group_label="HC"
    )
    sig_test_ms = patient_signature_test_cxcr3(
        adata, zinc_genes=zinc_genes, gene_signs=gene_signs, group_label="MS"
    )

    for name, res in [("HC", sig_test_hc), ("MS", sig_test_ms)]:
        pd.DataFrame([{k: v for k, v in res.items() if k != "per_patient_values"}]).to_csv(
            out_dir / f"FigS7_CXCR3_{TISSUE}_{name}_zinc_signature_patient_level_test.csv",
            index=False,
        )

    # Unsigned comparison (supplementary sanity check, not plotted in the
    # main 6-panel figure but saved for transparency)
    sig_test_hc_unsigned = patient_signature_test_cxcr3(
        adata, zinc_genes=zinc_genes, gene_signs=None, group_label="HC"
    )
    sig_test_ms_unsigned = patient_signature_test_cxcr3(
        adata, zinc_genes=zinc_genes, gene_signs=None, group_label="MS"
    )
    print(
        f"\nSigned vs unsigned signature p-values (sanity check):\n"
        f"  HC: signed p={sig_test_hc.get('wilcoxon_p', float('nan')):.3e}   "
        f"unsigned p={sig_test_hc_unsigned.get('wilcoxon_p', float('nan')):.3e}\n"
        f"  MS: signed p={sig_test_ms.get('wilcoxon_p', float('nan')):.3e}   "
        f"unsigned p={sig_test_ms_unsigned.get('wilcoxon_p', float('nan')):.3e}"
    )

    # Dataset-of-origin confounding check
    confound_hc = check_cxcr3_dataset_confounding(
        adata, zinc_genes=zinc_genes, group_label="HC", gene_signs=gene_signs
    )
    confound_ms = check_cxcr3_dataset_confounding(
        adata, zinc_genes=zinc_genes, group_label="MS", gene_signs=gene_signs
    )

    # ===== CREATE THE 6-PANEL FIGURE =====
    fig, axes = plt.subplots(3, 2, figsize=(12, 14))

    if df_de_ms is not None:
        _draw_volcano_cxcr3(
            axes[0, 0], df_de_ms,
            title=f"MS {TISSUE} B cells: CXCR3+ vs CXCR3-",
            label="A"
        )
    else:
        axes[0, 0].text(0.5, 0.5, "No MS data", ha="center", va="center", transform=axes[0, 0].transAxes)

    if df_de_ms is not None:
        _draw_bars_cxcr3(
            axes[0, 1], df_de_ms, bar_genes_ms,
            title="MS Zinc/MT genes: CXCR3+ vs CXCR3-",
            label="B",
            outlier_cap=3.0
        )
    else:
        axes[0, 1].text(0.5, 0.5, "No MS data", ha="center", va="center", transform=axes[0, 1].transAxes)

    if df_de_hc is not None:
        _draw_volcano_cxcr3(
            axes[1, 0], df_de_hc,
            title=f"HC {TISSUE} B cells: CXCR3+ vs CXCR3-",
            label="C"
        )
    else:
        axes[1, 0].text(0.5, 0.5, "No HC data", ha="center", va="center", transform=axes[1, 0].transAxes)

    if df_de_hc is not None:
        _draw_bars_cxcr3(
            axes[1, 1], df_de_hc, bar_genes_hc,
            title="HC Zinc/MT genes: CXCR3+ vs CXCR3-",
            label="D",
            outlier_cap=3.0
        )
    else:
        axes[1, 1].text(0.5, 0.5, "No HC data", ha="center", va="center", transform=axes[1, 1].transAxes)

    draw_paired_patient_plot_cxcr3(
        axes[2, 0], sig_test_hc,
        group_label="HC",
        letter="E",
        signed=True
    )

    draw_paired_patient_plot_cxcr3(
        axes[2, 1], sig_test_ms,
        group_label="MS",
        letter="F",
        signed=True
    )

    fig.tight_layout()

    stem = f"FigS7_CXCR3_{TISSUE}_6panel"
    out_png = out_dir / f"{stem}.png"
    out_pdf = out_dir / f"{stem}.pdf"
    print(f"\n✓ Figure saved to {out_png} and {out_pdf}")

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    if df_de_ms is not None:
        df_de_ms.to_csv(out_dir / f"{stem}_MS_full_DE.csv", index=False)
        pd.DataFrame({"gene": bar_genes_ms}).to_csv(
            out_dir / f"{stem}_MS_zinc_bar_genes.csv", index=False
        )

    if df_de_hc is not None:
        df_de_hc.to_csv(out_dir / f"{stem}_HC_full_DE.csv", index=False)
        pd.DataFrame({"gene": bar_genes_hc}).to_csv(
            out_dir / f"{stem}_HC_zinc_bar_genes.csv", index=False
        )


if __name__ == "__main__":
    make_figS7_cxcr3_CSF()
