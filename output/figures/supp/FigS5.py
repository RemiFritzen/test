#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
figS5.py — PB vs CSF comparison, Supplementary Figure 5

Covers BOTH CXCR3+ and CXCR3- B cell populations, produced as two separate
6-panel figures by two entry functions:
  make_figS5_cxcr3pos_pb_vs_csf() -> FigS5_CXCR3pos_PB_vs_CSF_6panel.{png,pdf}
  make_figS5_pb_vs_csf()          -> FigS5_CXCR3neg_PB_vs_CSF_6panel.{png,pdf}
Both are run from __main__.

CXCR3+ content was moved here from fig3.py: those per-gene/composite-
signature results are non-significant for CXCR3+ cells (0/14 genes at
either correction level in either group; composite signature null in both
HC and MS), so fig3.py's main figure now shows only the competitive
gene-set rank test instead. The CXCR3+-specific pipeline functions
(_run_de_pb_vs_csf_paired, compute_cell_signature_pb_vs_csf,
patient_signature_test_pb_vs_csf, check_pb_vs_csf_dataset_confounding,
_draw_volcano_pb_vs_csf, _draw_bars_pb_vs_csf, draw_paired_patient_plot)
are DUPLICATED in this file (with a `_pos` suffix) rather than imported
from fig3.py -- this file defines its own CXCR3- versions of the same
functions locally.

Each 6-panel figure layout:
  A  Volcano: MS B cells – CSF vs PB
  B  Barplot: MS zinc/MT genes (CSF vs PB)
  C  Volcano: HC B cells – CSF vs PB
  D  Barplot: HC zinc/MT genes (CSF vs PB)
  E  Paired slope plot: HC PB vs CSF zinc signature (each line = one
     patient's own PB->CSF trajectory, color-coded by direction -- NOT a
     violin/independent-distribution plot, since PB and CSF cells here come
     from the same patients and the paired test only cares about each
     patient's own within-patient difference)
  F  Paired slope plot: MS PB vs CSF zinc signature (same as E)

STATISTICS -- patient-level, paired (same rationale for both populations):
PB and CSF cells come from the SAME patients (unlike Fig 2's MS vs HC,
where each patient belongs to only one group). Per-cell tests would both
(a) pseudoreplicate -- treating every cell as independent when cells are
correlated within a patient -- and (b) ignore the pairing entirely,
throwing away the fact that a within-patient PB-vs-CSF shift is what's
biologically meaningful here.

Both per-gene DE and the composite signature test use the PAIRED pseudobulk
forms from stats_utils.py (paired_pseudobulk_de, paired_pseudobulk_signature_test),
which aggregate cells to one value per (patient, tissue), then run a
Wilcoxon signed-rank test on each patient's own PB-vs-CSF difference. This
requires a tissue-independent subject ID -- see resolve_pb_csf_subject_ids()
and the diagnostic printout in each entry function, which verifies pairing
is actually possible with this dataset's patient-key convention before
trusting any result built on it.

The composite zinc/MT signature sign-corrects each gene (+1 transporter,
-1 metallothionein, from compute_gene_signs) before averaging, instead of a
plain mean -- see fig2.py / stats_utils.py for the full rationale (plain
averaging lets up- and down-regulated genes cancel toward zero). UNSIGNED
is used as the primary/default here (no a priori directional hypothesis for
a PB-vs-CSF compartment shift, unlike MS-vs-HC disease state).
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
    resolve_pb_csf_subject_ids,
)

# (CXCR3+ pipeline functions defined below, duplicated from fig3.py's
# CXCR3+ versions with a _pos suffix -- see module docstring for why.)


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


def _apply_min_pct_filter_tissue(
    sub: sc.AnnData,
    min_pct: float = 0.10,
    always_keep: list[str] | None = None,
) -> sc.AnnData:
    """
    Keep genes expressed in at least one tissue above min_pct. This is a
    per-cell prevalence filter (deciding which genes are worth testing at
    all), not a statistical test itself, so it's unaffected by the
    pseudoreplication fix.

    `always_keep` (the curated zinc/MT panel) is exempted from this filter.
    Without this, a gene can pass the prevalence filter in one condition
    subset (e.g. HC) but fail it in another (e.g. MS), since each
    _run_de_pb_vs_csf_paired call filters independently -- silently
    producing a DIFFERENT gene set in each bar-chart panel, which is
    confusing and makes the two panels not directly comparable.
    """
    always_keep = set(always_keep or [])
    X = sub.X if not sp.issparse(sub.X) else sub.X.toarray()
    pb = sub.obs["tissue"] == "PB"
    csf = sub.obs["tissue"] == "CSF"

    pct_pb = (X[pb] > 0).mean(axis=0)
    pct_csf = (X[csf] > 0).mean(axis=0)
    keep = (pct_pb >= min_pct) | (pct_csf >= min_pct)

    forced_idx = [sub.var_names.get_loc(g) for g in always_keep if g in sub.var_names]
    n_forced_back_in = int((~keep[forced_idx]).sum()) if forced_idx else 0
    for idx in forced_idx:
        keep[idx] = True

    print(f"    min_pct={min_pct:.0%}: {keep.sum()}/{len(keep)} genes retained"
          + (f" ({n_forced_back_in} curated zinc/MT gene(s) kept despite low prevalence)"
             if n_forced_back_in else ""))
    return sub[:, keep].copy()


def compute_cell_signature_pb_vs_csf(
    adata: sc.AnnData,
    zinc_genes: list[str],
    gene_signs: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Per-cell zinc/MT signature in CXCR3- B cells. Columns:
        cell_id, tissue, condition, group_label, subject_id, zinc_signature

    Saved as supplementary per-cell CSV output only -- the main figure's
    paired slope plot (draw_paired_patient_plot) uses patient-level
    pseudobulk values from patient_signature_test_pb_vs_csf's
    per_patient_values instead, not this per-cell data directly.

    gene_signs: if given, sign-corrects each gene (+1 transporter, -1
    metallothionein) before averaging -- see stats_utils.py. If None
    (the default here, unlike fig2.py), computes a plain unsigned mean
    instead. Unsigned is the default/primary choice for THIS comparison
    (PB vs CSF) because there's no clean a priori directional hypothesis
    for a compartment shift the way there is for MS vs HC disease state --
    your own manuscript text (Fig 3B) describes a mixed-direction
    compartmental pattern even within the transporter category (e.g.
    SLC39A9/SLC30A9 lower in CSF while others don't move much), so applying
    a disease-direction sign scheme here would encode an assumption that
    isn't well supported for this specific comparison. The signed version is
    still computed and plotted in the supplementary figure for comparison.

    The p-value shown on the plot comes from the paired patient-level test
    (patient_signature_test_pb_vs_csf), not from testing these per-cell
    values directly.
    """
    sub = adata[adata.obs["cxcr3_group"] == "CXCR3-"].copy()

 

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

    df = sub.obs[["tissue", "condition", "subject_id"]].copy()
    df["cell_id"] = sub.obs_names
    df["zinc_signature"] = sig
    df = df[df["tissue"].isin(["PB", "CSF"])].copy()
    return df


def patient_signature_test_pb_vs_csf(
    adata: sc.AnnData,
    zinc_genes: list[str],
    group_label: str,
    gene_signs: dict[str, float] | None = None,
) -> dict:
    """
    Paired, patient-level test of the composite zinc/MT signature for PB vs
    CSF CXCR3- cells within one group (e.g. 'HC' or 'MS'). n = number of
    subjects with both PB and CSF cells present, not number of cells.

    gene_signs: see compute_cell_signature_pb_vs_csf -- None (default here)
    for the unsigned/plain-mean variant used as this figure's primary
    result, or a sign dict for the sign-corrected variant (supplementary).
    """
    sub = adata[adata.obs["cxcr3_group"] == "CXCR3-"].copy()
    
    sub = sub[(sub.obs["condition"] == group_label) & sub.obs["tissue"].isin(["PB", "CSF"])].copy()
    genes = [g for g in zinc_genes if g in sub.var_names]

    return paired_pseudobulk_signature_test(
        sub,
        groupby="tissue",
        group1="PB",
        group2="CSF",
        signature_genes=genes,
        patient_col="subject_id",
        use_raw=False,  # sub.X here is the same matrix compute_cell_signature_pb_vs_csf uses
        gene_signs=gene_signs,
        min_pairs=MIN_PAIRS,
    )


def check_pb_vs_csf_dataset_confounding(
    adata: sc.AnnData,
    zinc_genes: list[str],
    group_label: str,
    gene_signs: dict[str, float] | None = None,
    min_pairs_per_group: int = 3,
) -> dict:
    """
    Check whether the paired PB-vs-CSF composite signature result (Fig 3C/D)
    is being driven by, or differs between, dataset of origin -- same
    rationale as fig2.py's check_dataset_confounding and
    plot_figure1_umap_bar.py's check_bar_dataset_confounding (reviewer
    concern #4), adapted for a PAIRED design.

    Two checks, both operating on each subject's own (CSF - PB) difference
    rather than raw values (since the paired design's whole point is to
    remove between-patient variation -- comparing raw values across
    datasets would reintroduce exactly the noise pairing was meant to
    cancel):

    1. STRATIFIED: rerun the same paired Wilcoxon signed-rank test WITHIN
       each dataset separately (only subjects with both PB and CSF present
       contribute to each stratum).
    2. BETWEEN-DATASET COMPARISON OF THE PAIRED DIFFERENCES: an unpaired
       Mann-Whitney U test comparing each subject's (CSF - PB) difference
       between datasets. A significant result means the MAGNITUDE of the
       within-patient PB-to-CSF shift itself differs by dataset -- direct
       evidence of confounding in a paired design (there is no
       "interaction term" analogue here since each subject contributes
       only one difference value, not a full condition x dataset grid).
    """
    from scipy.stats import wilcoxon, mannwhitneyu

    sub = adata[adata.obs["cxcr3_group"] == "CXCR3-"].copy()


    sub = sub[(sub.obs["condition"] == group_label) & sub.obs["tissue"].isin(["PB", "CSF"])].copy()
    genes = [g for g in zinc_genes if g in sub.var_names]

    # Per-(subject, tissue) pseudobulk composite signature. NOTE: cannot use
    # pseudobulk_matrix here -- it requires each patient to map to exactly
    # ONE group value, which is violated by design here (a subject has BOTH
    # a PB and a CSF value). Aggregate directly instead, same pattern as
    # stats_utils.paired_pseudobulk_signature_test.
    source = sub  # use_raw=False, matching compute_cell_signature_pb_vs_csf / patient_signature_test_pb_vs_csf
    X = source[:, genes].X
    if sp.issparse(X):
        X = X.toarray()
    sign_vec = np.array([(gene_signs.get(g, 1.0) if gene_signs is not None else 1.0) for g in genes])
    signature = (np.asarray(X) * sign_vec[np.newaxis, :]).mean(axis=1)

    df = pd.DataFrame({
        "subject_id": sub.obs["subject_id"].values,
        "tissue": sub.obs["tissue"].values,
        "dataset": sub.obs["dataset"].values,
        "signature": signature,
    })
    pb_by_subject_tissue = df.groupby(["subject_id", "tissue"])["signature"].mean()
    wide = pb_by_subject_tissue.unstack("tissue")
    wide = wide.dropna(subset=[c for c in ["PB", "CSF"] if c in wide.columns])

    # attach dataset per subject (assumes 1:1 subject -> dataset)
    subject_dataset = df.drop_duplicates("subject_id").set_index("subject_id")["dataset"]
    wide["dataset"] = subject_dataset.reindex(wide.index)
    wide["diff"] = wide["CSF"] - wide["PB"]

    label = "signed" if gene_signs is not None else "unsigned"
    print(f"\n  === Dataset confounding check: PB vs CSF paired, {group_label}, {label} signature ===")

    per_dataset_results = []
    for ds in sorted(wide["dataset"].dropna().unique()):
        ds_wide = wide[wide["dataset"] == ds]
        n_pairs = len(ds_wide)
        if n_pairs < min_pairs_per_group:
            print(f"    [{ds}] n={n_pairs} paired subjects -- too few to test "
                  f"(< {min_pairs_per_group}); skipping.")
            per_dataset_results.append({"dataset": ds, "n_pairs": n_pairs,
                                         "median_diff": np.nan, "pvalue": np.nan})
            continue
        x, y = ds_wide["PB"].values, ds_wide["CSF"].values
        if np.all(x == y):
            stat, p = np.nan, 1.0
        else:
            stat, p = wilcoxon(x, y)
        median_diff = ds_wide["diff"].median()
        print(f"    [{ds}] n={n_pairs} paired subjects: median(CSF-PB)={median_diff:+.3f}, "
              f"Wilcoxon signed-rank p={p:.3g}")
        per_dataset_results.append({"dataset": ds, "n_pairs": n_pairs,
                                     "median_diff": median_diff, "pvalue": p})

    per_dataset_df = pd.DataFrame(per_dataset_results)
    testable = per_dataset_df.dropna(subset=["pvalue"])
    directions_agree = testable["median_diff"].apply(np.sign).nunique() <= 1 if len(testable) > 1 else None
    if directions_agree is False:
        print(f"    [WARNING] Direction of the paired CSF-vs-PB difference is INCONSISTENT "
              f"across datasets -- investigate before reporting the pooled paired test as "
              f"dataset-independent.")
    elif directions_agree is True:
        print(f"    Direction is consistent across all testable datasets.")

    # Between-dataset comparison of the paired differences themselves
    between_ds_p = np.nan
    ds_groups = [g["diff"].values for _, g in wide.groupby("dataset") if len(g) >= min_pairs_per_group]
    if len(ds_groups) == 2:
        stat, between_ds_p = mannwhitneyu(ds_groups[0], ds_groups[1], alternative="two-sided")
        verdict = "SIGNIFICANT -- the magnitude of the paired shift differs by dataset" if between_ds_p < 0.05 \
            else "not significant -- paired shift magnitude is consistent across datasets"
        print(f"    Between-dataset comparison of (CSF-PB) differences: p={between_ds_p:.3g} ({verdict})")
    elif len(ds_groups) > 2:
        print(f"    [note] >2 datasets with enough pairs -- between-dataset Mann-Whitney only "
              f"handles 2; use per_dataset table above for a >2-dataset comparison.")

    return {
        "group_label": group_label, "signature_label": label,
        "per_dataset": per_dataset_df,
        "directions_agree": directions_agree,
        "between_dataset_pvalue": between_ds_p,
    }


def _run_de_pb_vs_csf_paired(
    adata: sc.AnnData,
    zinc_genes: list[str],
    conditions: list[str],
    label: str,
    min_pct: float = 0.10,
) -> pd.DataFrame | None:
    """
    Paired DE for PB vs CSF in CXCR3- B cells, within the given set of raw
    `condition` values (e.g. ["HC"], or [ "MS"] pooled together as one
    demyelinating group). Returns CSF vs PB, so negative log2FC means lower
    in CSF -- same convention as the original per-cell version.

    Replaces sc.tl.rank_genes_groups (unpaired, per-cell) with
    paired_pseudobulk_de (paired, per-patient): the same patient contributes
    both PB and CSF cells here, so this is a within-patient comparison.
    """
    print(f"\nRunning paired DE: PB vs CSF CXCR3-, conditions={conditions} …")

    sub = adata[
        (adata.obs["cxcr3_group"] == "CXCR3-")
        & adata.obs["condition"].isin(conditions)
        & adata.obs["tissue"].isin(["PB", "CSF"])
    ].copy()

    if sub.n_obs == 0 or not {"PB", "CSF"}.issubset(set(sub.obs["tissue"].unique())):
        print(f"  [skip] {label}: no CXCR3- PB+CSF cells for conditions {conditions}.")
        return None

    n_subjects = sub.obs.loc[sub.obs["tissue"] == "PB", "subject_id"].nunique()
    n_subjects_csf = sub.obs.loc[sub.obs["tissue"] == "CSF", "subject_id"].nunique()
    n_both = len(set(sub.obs.loc[sub.obs["tissue"] == "PB", "subject_id"])
                 & set(sub.obs.loc[sub.obs["tissue"] == "CSF", "subject_id"]))
    print(f"    {label}: {n_subjects} subjects w/ PB, {n_subjects_csf} w/ CSF, "
          f"{n_both} with BOTH ({sub.n_obs} cells total)")
    if n_both < MIN_PAIRS:
        print(f"    [skip] {label}: only {n_both} paired subjects (< {MIN_PAIRS}) -- "
              f"not enough for a paired test.")
        return None

    if not sub.var_names.is_unique:
        print(f"    [fix] {label}: duplicate gene symbols in var_names -- making unique")
        sub.var_names_make_unique()

    missing_from_data = [g for g in zinc_genes if g not in sub.var_names]
    if missing_from_data:
        print(f"    [warn] {label}: {len(missing_from_data)} curated zinc/MT gene(s) not in "
              f"the dataset at all: {missing_from_data}")

    sub = _apply_min_pct_filter_tissue(sub, min_pct=min_pct, always_keep=zinc_genes)

    df = paired_pseudobulk_de(
        sub,
        groupby="tissue",
        group1="PB",
        group2="CSF",
        patient_col="subject_id",
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
    #                        The zinc panel is small and pre-specified, so
    #                        correcting it against thousands of unrelated
    #                        background genes (appropriate for the volcano)
    #                        is needlessly conservative for this narrower,
    #                        confirmatory claim.
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
    was actually tested -- see _apply_min_pct_filter_tissue's always_keep),
    sorted by log2FC descending. Any curated gene missing from `df` entirely
    (not in the dataset) 
    """
    tested = df[df["gene"].isin(zinc_genes)].copy()
    tested = tested.sort_values("log2FC", ascending=False)
    genes_present = tested["gene"].tolist()
    genes_missing = [g for g in zinc_genes if g not in set(genes_present)]
    return genes_present, genes_missing


# ── Plot helpers ─────────────────────────────────────────────────────────────

def _draw_volcano_pb_vs_csf(
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

    ax.set_xlabel("log2FC (CSF vs PB)")
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


def _draw_bars_pb_vs_csf(
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
    # the genome-wide padj used for the volcano's threshold line -- see
    # _run_de_pb_vs_csf_paired for the rationale.
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
        y_neg = max(fc_plot[i], 0) + y_offset
        ax.text(i, y_neg, sig, ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(genes, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("log2FC (CSF vs PB)")
    ax.set_xlabel("")
    ax.set_title(title, fontsize=10)
    _panel_letter(ax, label)


def draw_paired_patient_plot(ax, sig_test_result, group_label, letter, signed: bool = False):
    """
    Paired slope plot of PATIENT-LEVEL pseudobulk composite signature
    values: each patient's own PB and CSF pseudobulk values connected by a
    line, so the within-patient shift the paired Wilcoxon signed-rank test
    actually operates on is directly visible.

    This is deliberately NOT two independent-looking distributions side by
    side (e.g. two box plots or violins with no connecting lines) --  PB
    and CSF cells here come from the SAME patients, and the paired test
    only cares about each patient's own PB->CSF difference, not the two
    tissues' marginal distributions in isolation. Showing the paired lines
    directly reveals whether patients shift consistently (most lines the
    same direction) or heterogeneously (lines crossing / mixed direction),
    which two separate distributions cannot convey, and avoids the same
    display/test mismatch that motivated replacing the per-cell violin
    with patient-level plots in fig2.py / figS4.py.

    Box plots are drawn at each tissue position as an aggregate summary
    only (no fliers, since every point is already shown via the lines).

    `signed` only controls the axis/title label -- `sig_test_result` must
    already have been computed with the matching gene_signs by the caller.
    """
    per_patient = sig_test_result.get("per_patient_values")
    p = sig_test_result.get("wilcoxon_p", np.nan)
    n_pairs = sig_test_result.get("n_pairs", "?")

    if (per_patient is None or per_patient.empty
            or "PB" not in per_patient.columns or "CSF" not in per_patient.columns):
        ax.text(0.5, 0.5, "No paired per-patient data available", ha="center", va="center",
                transform=ax.transAxes)
        _panel_letter(ax, letter)
        return

    pb_vals = per_patient["PB"].values
    csf_vals = per_patient["CSF"].values

    ax.boxplot(
        [pb_vals, csf_vals], positions=[1, 2], widths=0.5,
        showfliers=False, patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        boxprops=dict(facecolor="#D9D9D9", alpha=0.5, edgecolor="black"),
        whiskerprops=dict(color="black"), capprops=dict(color="black"),
        zorder=1,
    )

    rng = np.random.default_rng(42)
    jitter = rng.normal(0, 0.04, size=len(pb_vals))
    for i in range(len(pb_vals)):
        direction_color = "#4C78A8" if csf_vals[i] >= pb_vals[i] else "#E45756"
        ax.plot(
            [1 + jitter[i], 2 + jitter[i]], [pb_vals[i], csf_vals[i]],
            color=direction_color, alpha=0.5, linewidth=1.2, zorder=5,
        )
        ax.scatter(
            [1 + jitter[i], 2 + jitter[i]], [pb_vals[i], csf_vals[i]],
            color=direction_color, s=24, alpha=0.8, edgecolors="black", linewidths=0.3, zorder=10,
        )

    if np.isfinite(p):
        print(
            f"[{group_label}] PAIRED PATIENT-LEVEL plot: n={n_pairs} subject pairs "
            f"(each line = one patient's PB->CSF trajectory), Wilcoxon signed-rank p={p:.3e}"
        )
    else:
        print(f"[{group_label}] n={n_pairs} pairs -- too few to test")

    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"PB\n(n={n_pairs})", f"CSF\n(n={n_pairs})"])
    score_label = "signed" if signed else "unsigned (plain mean)"
    ax.set_ylabel(f"Patient-level mean zinc/MT signature ({score_label})")
    title_p = f"{p:.2e}" if np.isfinite(p) else "n/a (too few paired subjects)"
    ax.set_title(
        f"{group_label} CXCR3- – PB vs CSF ({score_label})\n"
        f"Paired Wilcoxon signed-rank p = {title_p} (n={n_pairs} subject pairs)",
        fontsize=9,
    )
    ax.spines[["top", "right"]].set_visible(False)
    _panel_letter(ax, letter)


def draw_violin_pb_vs_csf(ax, df_sig, sig_test_result, group_label, letter, signed: bool = False):
    """
    Draws the per-cell violin/jitter (visual only) and annotates it with the
    PAIRED, patient-level test result passed in via `sig_test_result`, NOT a
    Mann-Whitney test on the per-cell values plotted here.

    `signed` only controls the axis/title label (which composite score is
    being shown) -- the data in df_sig / sig_test_result must already be
    computed accordingly by the caller (see compute_cell_signature_pb_vs_csf /
    patient_signature_test_pb_vs_csf's gene_signs argument). Defaults to
    False here since unsigned is this figure's primary choice (see
    compute_cell_signature_pb_vs_csf's docstring for why).
    """
    df = df_sig[df_sig["condition"] == group_label].copy()
    groups = ["PB", "CSF"]
    colors = {"PB": "#4C78A8", "CSF": "#4C78A8"}
    data = [df.loc[df["tissue"] == g, "zinc_signature"].values for g in groups]

    parts = ax.violinplot(data, positions=[1, 2], showmeans=False, showmedians=False)
    for pc in parts["bodies"]:
        pc.set_facecolor("#D9D9D9")
        pc.set_edgecolor("black")
        pc.set_alpha(0.7)

    for i, (g, vals) in enumerate(zip(groups, data), start=1):
        jitter = np.random.normal(0, 0.04, size=len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter,
            vals,
            s=8,
            alpha=0.30,
            color="black",
            zorder=10,
        )

        med = np.median(vals)
        ax.hlines(
            y=med,
            xmin=i - 0.16,
            xmax=i + 0.16,
            color=colors[g],
            linewidth=2.0,
            zorder=20,
        )

    pb_vals, csf_vals = data[0], data[1]
    p = sig_test_result.get("wilcoxon_p", np.nan)
    n_pairs = sig_test_result.get("n_pairs", "?")

    if np.isfinite(p):
        print(
            f"[{group_label}] cell-level PB median={np.median(pb_vals):.3f}, "
            f"CSF median={np.median(csf_vals):.3f} "
            f"({len(pb_vals)} PB cells, {len(csf_vals)} CSF cells) | "
            f"PAIRED PATIENT-LEVEL test: n={n_pairs} subject pairs, "
            f"Wilcoxon signed-rank p={p:.3e}"
        )
    else:
        print(
            f"[{group_label}] PAIRED PATIENT-LEVEL test: n={n_pairs} subject pairs "
            f"-- too few pairs to test"
        )
    

    ax.set_xticks([1, 2])
    ax.set_xticklabels(groups)
    score_label = "signed" if signed else "unsigned (plain mean)"
    ax.set_ylabel(f"Zinc/MT signature ({score_label}, per cell)")
    title_p = f"{p:.2e}" if np.isfinite(p) else "n/a (too few paired subjects)"
    ax.set_title(
        f"{group_label} CXCR3- – PB vs CSF ({score_label})\n"
        f"Paired Wilcoxon p = {title_p} (n={n_pairs} subject pairs)",
        fontsize=9,
    )
    _panel_letter(ax, letter)


def _save_single_volcano_pb_vs_csf(df, title, label, out_png, out_pdf):
    fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))
    _draw_volcano_pb_vs_csf(ax, df, title=title, label=label)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────
def make_figS5_pb_vs_csf() -> None:
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
    
    print("Checking PB/CSF subject-ID pairing …")
    adata.obs["subject_id"] = resolve_pb_csf_subject_ids(
        adata, patient_col="patient", tissue_col="tissue"
    ).values

    out_dir = FIG_DIRS.get("supp_pb_vs_csf_paired", RESULTS_DIR / "supp" / "pb_vs_csf_paired")
    out_dir.mkdir(parents=True, exist_ok=True)

    zinc_genes = load_zinc_gene_list()
    gene_signs = load_zinc_gene_signs()
    print(f"Loaded {len(zinc_genes)} curated zinc/MT genes: {zinc_genes}")
    print(f"Gene signs for composite signature: {gene_signs}")
    
 
    
    # MS paired DE
    df_de_ms = _run_de_pb_vs_csf_paired(
        adata, zinc_genes=zinc_genes, conditions=["MS"], label="MS", min_pct=0.10,
    )
    bar_genes_ms: list[str] = []
    if df_de_ms is not None:
        print(f" {df_de_ms.shape[0]} genes in PB vs CSF DE table (MS)")
        bar_genes_ms, ms_missing = _select_genes_for_bars(df_de_ms, zinc_genes=zinc_genes)
        print(f"MS PB vs CSF bar genes ({len(bar_genes_ms)}/{len(zinc_genes)} of curated panel):", bar_genes_ms)
        if ms_missing:
            print(f" [note] MS: {len(ms_missing)} curated gene(s) not in dataset: {ms_missing}")
    
    # HC paired DE
    df_de_hc = _run_de_pb_vs_csf_paired(
        adata, zinc_genes=zinc_genes, conditions=["HC"], label="HC", min_pct=0.10,
    )
    bar_genes_hc: list[str] = []
    if df_de_hc is not None:
        print(f" {df_de_hc.shape[0]} genes in PB vs CSF DE table (HC)")
        bar_genes_hc, hc_missing = _select_genes_for_bars(df_de_hc, zinc_genes=zinc_genes)
        print(f"HC PB vs CSF bar genes ({len(bar_genes_hc)}/{len(zinc_genes)} of curated panel):", bar_genes_hc)
        if hc_missing:
            print(f" [note] HC: {len(hc_missing)} curated gene(s) not in dataset: {hc_missing}")
    
    # Sanity check: HC and MS panels should show the SAME gene set
    if df_de_ms is not None and df_de_hc is not None:
        set_diff = set(bar_genes_ms).symmetric_difference(set(bar_genes_hc))
        if set_diff:
            print(f" [WARNING] HC and MS bar-chart gene sets differ: {sorted(set_diff)} -- panels are not directly comparable.")
    
    # Compute signatures (UNSIGNED is primary for this figure)
    df_sig = compute_cell_signature_pb_vs_csf(
        adata, zinc_genes=zinc_genes, gene_signs=None
    )
    
    # Save per-cell signature data
    if "HC" in df_sig["condition"].unique():
        df_sig[df_sig["condition"] == "HC"].to_csv(
            out_dir / "FigS5_CXCR3neg_PB_vs_CSF_HC_zinc_signature_per_cell.csv",
            index=False
        )
    if "MS" in df_sig["condition"].unique():
        df_sig[df_sig["condition"] == "MS"].to_csv(
            out_dir / "FigS5_CXCR3neg_PB_vs_CSF_MS_zinc_signature_per_cell.csv",
            index=False
        )
    
    # Paired patient-level signature tests
    sig_test_hc = patient_signature_test_pb_vs_csf(
        adata, zinc_genes=zinc_genes, gene_signs=None, group_label="HC"
    )
    sig_test_ms = patient_signature_test_pb_vs_csf(
        adata, zinc_genes=zinc_genes, gene_signs=None, group_label="MS"
    )
    
    for name, res in [("HC", sig_test_hc), ("MS", sig_test_ms)]:
        pd.DataFrame([{k: v for k, v in res.items() if k != "per_patient_values"}]).to_csv(
            out_dir / f"FigS5_CXCR3neg_PB_vs_CSF_{name}_zinc_signature_patient_level_test.csv",
            index=False,
        )
    
    # Dataset-of-origin confounding check
    confound_hc = check_pb_vs_csf_dataset_confounding(
        adata, zinc_genes=zinc_genes, group_label="HC", gene_signs=None
    )
    confound_ms = check_pb_vs_csf_dataset_confounding(
        adata, zinc_genes=zinc_genes, group_label="MS", gene_signs=None
    )
    
    # ===== CREATE THE 6-PANEL FIGURE =====
    fig, axes = plt.subplots(3, 2, figsize=(12, 14))
    
    # Panel A: MS Volcano
    if df_de_ms is not None:
        _draw_volcano_pb_vs_csf(
            axes[0, 0], df_de_ms,
            title="MS CXCR3- B cells: CSF vs PB",
            label="A"
        )
    else:
        axes[0, 0].text(0.5, 0.5, "No MS data", ha="center", va="center", transform=axes[0, 0].transAxes)
    
    # Panel B: MS Bar chart
    if df_de_ms is not None:
        _draw_bars_pb_vs_csf(
            axes[0, 1], df_de_ms, bar_genes_ms,
            title="MS Zinc/MT genes: CSF vs PB",
            label="B",
            outlier_cap=3.0
        )
    else:
        axes[0, 1].text(0.5, 0.5, "No MS data", ha="center", va="center", transform=axes[0, 1].transAxes)
    
    # Panel C: HC Volcano
    if df_de_hc is not None:
        _draw_volcano_pb_vs_csf(
            axes[1, 0], df_de_hc,
            title="HC CXCR3- B cells: CSF vs PB",
            label="C"
        )
    else:
        axes[1, 0].text(0.5, 0.5, "No HC data", ha="center", va="center", transform=axes[1, 0].transAxes)
    
    # Panel D: HC Bar chart
    if df_de_hc is not None:
        _draw_bars_pb_vs_csf(
            axes[1, 1], df_de_hc, bar_genes_hc,
            title="HC Zinc/MT genes: CSF vs PB",
            label="D",
            outlier_cap=3.0
        )
    else:
        axes[1, 1].text(0.5, 0.5, "No HC data", ha="center", va="center", transform=axes[1, 1].transAxes)
    
    # Panel E: HC paired slope plot (PB vs CSF signature)
    draw_paired_patient_plot(
        axes[2, 0], sig_test_hc,
        group_label="HC",
        letter="E",
        signed=False
    )
    
    # Panel F: MS paired slope plot (PB vs CSF signature)
    draw_paired_patient_plot(
        axes[2, 1], sig_test_ms,
        group_label="MS",
        letter="F",
        signed=False
    )
    
    fig.tight_layout()
    
    # Save outputs
 

    stem = "FigS5_CXCR3neg_PB_vs_CSF_6panel"
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


def compute_cell_signature_pb_vs_csf_pos(
    adata: sc.AnnData,
    zinc_genes: list[str],
    grouping: str,
    gene_signs: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Per-cell zinc/MT signature in CXCR3+ B cells. Columns:
        cell_id, tissue, condition, group_label, subject_id, zinc_signature

    Saved as supplementary per-cell CSV output only (in figS5.py, which
    imports this function) -- the paired slope plot (draw_paired_patient_plot)
    uses patient-level pseudobulk values from patient_signature_test_pb_vs_csf's
    per_patient_values instead, not this per-cell data directly.

    gene_signs: if given, sign-corrects each gene (+1 transporter, -1
    metallothionein) before averaging -- see stats_utils.py. If None
    (the default here, unlike fig2.py), computes a plain unsigned mean
    instead. Unsigned is the default/primary choice for THIS comparison
    (PB vs CSF) because there's no clean a priori directional hypothesis
    for a compartment shift the way there is for MS vs HC disease state --
    your own manuscript text (Fig 3B) describes a mixed-direction
    compartmental pattern even within the transporter category (e.g.
    SLC39A9/SLC30A9 lower in CSF while others don't move much), so applying
    a disease-direction sign scheme here would encode an assumption that
    isn't well supported for this specific comparison. The signed version is
    still computed and plotted in the supplementary figure for comparison.

    The p-value shown on the plot comes from the paired patient-level test
    (patient_signature_test_pb_vs_csf), not from testing these per-cell
    values directly.
    """
    sub = adata[adata.obs["cxcr3_group"] == "CXCR3+"].copy()

    if grouping == "demyelinating":
        sub = sub[sub.obs["condition"].isin(["HC", "MS"])].copy()
        # .astype(str): obs["condition"] is a pandas Categorical (from
        # AnnData/h5ad); casting to plain string avoids Categorical dtype
        # restrictions elsewhere in this function.
        sub.obs["group_label"] = sub.obs["condition"].astype(str)
    elif grouping == "MS_only":
        sub = sub[sub.obs["condition"].isin(["HC", "MS"])].copy()
        sub.obs["group_label"] = sub.obs["condition"].astype(str)
    elif grouping == "all":
        sub = sub[sub.obs["condition"].isin(["HC", "MS"])].copy()
        sub.obs["group_label"] = sub.obs["condition"].astype(str)
    else:
        raise ValueError(f"Unknown GROUPING_MODE: {grouping}")

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

    df = sub.obs[["tissue", "condition", "group_label", "subject_id"]].copy()
    df["cell_id"] = sub.obs_names
    df["zinc_signature"] = sig
    df = df[df["tissue"].isin(["PB", "CSF"])].copy()
    return df


def patient_signature_test_pb_vs_csf_pos(
    adata: sc.AnnData,
    zinc_genes: list[str],
    grouping: str,
    group_label: str,
    gene_signs: dict[str, float] | None = None,
) -> dict:
    """
    Paired, patient-level test of the composite zinc/MT signature for PB vs
    CSF CXCR3+ cells within one group (e.g. 'HC' or 'MS'). n = number of
    subjects with both PB and CSF cells present, not number of cells.

    gene_signs: see compute_cell_signature_pb_vs_csf -- None (default here)
    for the unsigned/plain-mean variant used as this figure's primary
    result, or a sign dict for the sign-corrected variant (supplementary).
    """
    sub = adata[adata.obs["cxcr3_group"] == "CXCR3+"].copy()
    if grouping == "demyelinating":
        sub = sub[sub.obs["condition"].isin(["HC", "MS"])].copy()
        sub.obs["group_label"] = sub.obs["condition"].astype(str)
    elif grouping == "MS_only":
        sub = sub[sub.obs["condition"].isin(["HC", "MS"])].copy()
        sub.obs["group_label"] = sub.obs["condition"].astype(str)
    elif grouping == "all":
        sub = sub[sub.obs["condition"].isin(["HC", "MS"])].copy()
        sub.obs["group_label"] = sub.obs["condition"].astype(str)
    else:
        raise ValueError(f"Unknown GROUPING_MODE: {grouping}")

    sub = sub[(sub.obs["group_label"] == group_label) & sub.obs["tissue"].isin(["PB", "CSF"])].copy()
    genes = [g for g in zinc_genes if g in sub.var_names]

    return paired_pseudobulk_signature_test(
        sub,
        groupby="tissue",
        group1="PB",
        group2="CSF",
        signature_genes=genes,
        patient_col="subject_id",
        use_raw=False,  # sub.X here is the same matrix compute_cell_signature_pb_vs_csf uses
        gene_signs=gene_signs,
        min_pairs=MIN_PAIRS,
    )


def check_pb_vs_csf_dataset_confounding_pos(
    adata: sc.AnnData,
    zinc_genes: list[str],
    grouping: str,
    group_label: str,
    gene_signs: dict[str, float] | None = None,
    min_pairs_per_group: int = 3,
) -> dict:
    """
    Check whether the paired PB-vs-CSF composite signature result (Fig 3C/D)
    is being driven by, or differs between, dataset of origin -- same
    rationale as fig2.py's check_dataset_confounding and
    plot_figure1_umap_bar.py's check_bar_dataset_confounding (reviewer
    concern #4), adapted for a PAIRED design.

    Two checks, both operating on each subject's own (CSF - PB) difference
    rather than raw values (since the paired design's whole point is to
    remove between-patient variation -- comparing raw values across
    datasets would reintroduce exactly the noise pairing was meant to
    cancel):

    1. STRATIFIED: rerun the same paired Wilcoxon signed-rank test WITHIN
       each dataset separately (only subjects with both PB and CSF present
       contribute to each stratum).
    2. BETWEEN-DATASET COMPARISON OF THE PAIRED DIFFERENCES: an unpaired
       Mann-Whitney U test comparing each subject's (CSF - PB) difference
       between datasets. A significant result means the MAGNITUDE of the
       within-patient PB-to-CSF shift itself differs by dataset -- direct
       evidence of confounding in a paired design (there is no
       "interaction term" analogue here since each subject contributes
       only one difference value, not a full condition x dataset grid).
    """
    from scipy.stats import wilcoxon, mannwhitneyu

    sub = adata[adata.obs["cxcr3_group"] == "CXCR3+"].copy()
    if grouping == "demyelinating":
        sub = sub[sub.obs["condition"].isin(["HC", "MS"])].copy()
        sub.obs["group_label"] = sub.obs["condition"].astype(str)
    elif grouping == "MS_only":
        sub = sub[sub.obs["condition"].isin(["HC", "MS"])].copy()
        sub.obs["group_label"] = sub.obs["condition"].astype(str)
    elif grouping == "all":
        sub = sub[sub.obs["condition"].isin(["HC", "MS"])].copy()
        sub.obs["group_label"] = sub.obs["condition"].astype(str)
    else:
        raise ValueError(f"Unknown GROUPING_MODE: {grouping}")

    sub = sub[(sub.obs["group_label"] == group_label) & sub.obs["tissue"].isin(["PB", "CSF"])].copy()
    genes = [g for g in zinc_genes if g in sub.var_names]

    # Per-(subject, tissue) pseudobulk composite signature. NOTE: cannot use
    # pseudobulk_matrix here -- it requires each patient to map to exactly
    # ONE group value, which is violated by design here (a subject has BOTH
    # a PB and a CSF value). Aggregate directly instead, same pattern as
    # stats_utils.paired_pseudobulk_signature_test.
    source = sub  # use_raw=False, matching compute_cell_signature_pb_vs_csf / patient_signature_test_pb_vs_csf
    X = source[:, genes].X
    if sp.issparse(X):
        X = X.toarray()
    sign_vec = np.array([(gene_signs.get(g, 1.0) if gene_signs is not None else 1.0) for g in genes])
    signature = (np.asarray(X) * sign_vec[np.newaxis, :]).mean(axis=1)

    df = pd.DataFrame({
        "subject_id": sub.obs["subject_id"].values,
        "tissue": sub.obs["tissue"].values,
        "dataset": sub.obs["dataset"].values,
        "signature": signature,
    })
    pb_by_subject_tissue = df.groupby(["subject_id", "tissue"])["signature"].mean()
    wide = pb_by_subject_tissue.unstack("tissue")
    wide = wide.dropna(subset=[c for c in ["PB", "CSF"] if c in wide.columns])

    # attach dataset per subject (assumes 1:1 subject -> dataset)
    subject_dataset = df.drop_duplicates("subject_id").set_index("subject_id")["dataset"]
    wide["dataset"] = subject_dataset.reindex(wide.index)
    wide["diff"] = wide["CSF"] - wide["PB"]

    label = "signed" if gene_signs is not None else "unsigned"
    print(f"\n  === Dataset confounding check: PB vs CSF paired, {group_label}, {label} signature ===")

    per_dataset_results = []
    for ds in sorted(wide["dataset"].dropna().unique()):
        ds_wide = wide[wide["dataset"] == ds]
        n_pairs = len(ds_wide)
        if n_pairs < min_pairs_per_group:
            print(f"    [{ds}] n={n_pairs} paired subjects -- too few to test "
                  f"(< {min_pairs_per_group}); skipping.")
            per_dataset_results.append({"dataset": ds, "n_pairs": n_pairs,
                                         "median_diff": np.nan, "pvalue": np.nan})
            continue
        x, y = ds_wide["PB"].values, ds_wide["CSF"].values
        if np.all(x == y):
            stat, p = np.nan, 1.0
        else:
            stat, p = wilcoxon(x, y)
        median_diff = ds_wide["diff"].median()
        print(f"    [{ds}] n={n_pairs} paired subjects: median(CSF-PB)={median_diff:+.3f}, "
              f"Wilcoxon signed-rank p={p:.3g}")
        per_dataset_results.append({"dataset": ds, "n_pairs": n_pairs,
                                     "median_diff": median_diff, "pvalue": p})

    per_dataset_df = pd.DataFrame(per_dataset_results)
    testable = per_dataset_df.dropna(subset=["pvalue"])
    directions_agree = testable["median_diff"].apply(np.sign).nunique() <= 1 if len(testable) > 1 else None
    if directions_agree is False:
        print(f"    [WARNING] Direction of the paired CSF-vs-PB difference is INCONSISTENT "
              f"across datasets -- investigate before reporting the pooled paired test as "
              f"dataset-independent.")
    elif directions_agree is True:
        print(f"    Direction is consistent across all testable datasets.")

    # Between-dataset comparison of the paired differences themselves
    between_ds_p = np.nan
    ds_groups = [g["diff"].values for _, g in wide.groupby("dataset") if len(g) >= min_pairs_per_group]
    if len(ds_groups) == 2:
        stat, between_ds_p = mannwhitneyu(ds_groups[0], ds_groups[1], alternative="two-sided")
        verdict = "SIGNIFICANT -- the magnitude of the paired shift differs by dataset" if between_ds_p < 0.05 \
            else "not significant -- paired shift magnitude is consistent across datasets"
        print(f"    Between-dataset comparison of (CSF-PB) differences: p={between_ds_p:.3g} ({verdict})")
    elif len(ds_groups) > 2:
        print(f"    [note] >2 datasets with enough pairs -- between-dataset Mann-Whitney only "
              f"handles 2; use per_dataset table above for a >2-dataset comparison.")

    return {
        "group_label": group_label, "signature_label": label,
        "per_dataset": per_dataset_df,
        "directions_agree": directions_agree,
        "between_dataset_pvalue": between_ds_p,
    }


def _run_de_pb_vs_csf_paired_pos(
    adata: sc.AnnData,
    zinc_genes: list[str],
    conditions: list[str],
    label: str,
    min_pct: float = 0.10,
) -> pd.DataFrame | None:
    """
    Paired DE for PB vs CSF in CXCR3+ B cells, within the given set of raw
    `condition` values (e.g. ["HC"], or ["MS"]). Returns CSF vs PB, so
    negative log2FC means lower in CSF -- same convention as the original
    per-cell version.

    Replaces sc.tl.rank_genes_groups (unpaired, per-cell) with
    paired_pseudobulk_de (paired, per-patient): the same patient contributes
    both PB and CSF cells here, so this is a within-patient comparison.
    """
    print(f"\nRunning paired DE: PB vs CSF CXCR3+, conditions={conditions} …")

    sub = adata[
        (adata.obs["cxcr3_group"] == "CXCR3+")
        & adata.obs["condition"].isin(conditions)
        & adata.obs["tissue"].isin(["PB", "CSF"])
    ].copy()

    if sub.n_obs == 0 or not {"PB", "CSF"}.issubset(set(sub.obs["tissue"].unique())):
        print(f"  [skip] {label}: no CXCR3+ PB+CSF cells for conditions {conditions}.")
        return None

    n_subjects = sub.obs.loc[sub.obs["tissue"] == "PB", "subject_id"].nunique()
    n_subjects_csf = sub.obs.loc[sub.obs["tissue"] == "CSF", "subject_id"].nunique()
    n_both = len(set(sub.obs.loc[sub.obs["tissue"] == "PB", "subject_id"])
                 & set(sub.obs.loc[sub.obs["tissue"] == "CSF", "subject_id"]))
    print(f"    {label}: {n_subjects} subjects w/ PB, {n_subjects_csf} w/ CSF, "
          f"{n_both} with BOTH ({sub.n_obs} cells total)")
    if n_both < MIN_PAIRS:
        print(f"    [skip] {label}: only {n_both} paired subjects (< {MIN_PAIRS}) -- "
              f"not enough for a paired test.")
        return None

    if not sub.var_names.is_unique:
        print(f"    [fix] {label}: duplicate gene symbols in var_names -- making unique")
        sub.var_names_make_unique()

    missing_from_data = [g for g in zinc_genes if g not in sub.var_names]
    if missing_from_data:
        print(f"    [warn] {label}: {len(missing_from_data)} curated zinc/MT gene(s) not in "
              f"the dataset at all: {missing_from_data}")

    sub = _apply_min_pct_filter_tissue(sub, min_pct=min_pct, always_keep=zinc_genes)

    df = paired_pseudobulk_de(
        sub,
        groupby="tissue",
        group1="PB",
        group2="CSF",
        patient_col="subject_id",
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
    #                        The zinc panel is small and pre-specified, so
    #                        correcting it against thousands of unrelated
    #                        background genes (appropriate for the volcano)
    #                        is needlessly conservative for this narrower,
    #                        confirmatory claim.
    zinc_mask = df["is_zinc"].values
    df["padj_zinc_panel"] = np.nan
    if zinc_mask.sum() > 0:
        df.loc[zinc_mask, "padj_zinc_panel"] = multipletests(
            df.loc[zinc_mask, "pvalue"].fillna(1.0), method="fdr_bh"
        )[1]

    return df


def _select_genes_for_bars_pos(df: pd.DataFrame, zinc_genes: list[str]) -> tuple[list[str], list[str]]:
    """
    Bar-chart gene list = the FULL curated zinc/MT panel (every gene that
    was actually tested -- see _apply_min_pct_filter_tissue's always_keep),
    sorted by log2FC descending. Any curated gene missing from `df` entirely
    (not in the dataset) is returned separately so the caller can report it
    rather than silently omit it -- and so the two panels (HC vs MS) can
    be checked for showing the same gene set.
    """
    tested = df[df["gene"].isin(zinc_genes)].copy()
    tested = tested.sort_values("log2FC", ascending=False)
    genes_present = tested["gene"].tolist()
    genes_missing = [g for g in zinc_genes if g not in set(genes_present)]
    return genes_present, genes_missing


# ── Plot helpers ─────────────────────────────────────────────────────────────


def _draw_volcano_pb_vs_csf_pos(
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

    ax.set_xlabel("log2FC (CSF vs PB)")
    ax.set_ylabel("−log10(padj)")
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper left", fontsize=7)
    _panel_letter(ax, label)


def _draw_bars_pb_vs_csf_pos(
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
    # the genome-wide padj used for the volcano's threshold line -- see
    # _run_de_pb_vs_csf_paired for the rationale.
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
    ax.set_ylabel("log2FC (CSF vs PB)")
    ax.set_xlabel("")
    ax.set_title(title, fontsize=10)
    _panel_letter(ax, label)


def draw_paired_patient_plot_pos(ax, sig_test_result, group_label, letter, signed: bool = False):
    """
    Paired slope plot of PATIENT-LEVEL pseudobulk composite signature
    values: each patient's own PB and CSF pseudobulk values connected by a
    line, so the within-patient shift the paired Wilcoxon signed-rank test
    actually operates on is directly visible.

    This is deliberately NOT two independent-looking distributions side by
    side (e.g. two box plots or violins with no connecting lines) --  PB
    and CSF cells here come from the SAME patients, and the paired test
    only cares about each patient's own PB->CSF difference, not the two
    tissues' marginal distributions in isolation. Showing the paired lines
    directly reveals whether patients shift consistently (most lines the
    same direction) or heterogeneously (lines crossing / mixed direction),
    which two separate distributions cannot convey, and avoids the same
    display/test mismatch that motivated replacing the per-cell violin
    with patient-level plots in fig2.py / figS4.py.

    Box plots are drawn at each tissue position as an aggregate summary
    only (no fliers, since every point is already shown via the lines).

    `signed` only controls the axis/title label -- `sig_test_result` must
    already have been computed with the matching gene_signs by the caller.
    """
    per_patient = sig_test_result.get("per_patient_values")
    p = sig_test_result.get("wilcoxon_p", np.nan)
    n_pairs = sig_test_result.get("n_pairs", "?")

    if (per_patient is None or per_patient.empty
            or "PB" not in per_patient.columns or "CSF" not in per_patient.columns):
        ax.text(0.5, 0.5, "No paired per-patient data available", ha="center", va="center",
                transform=ax.transAxes)
        _panel_letter(ax, letter)
        return

    pb_vals = per_patient["PB"].values
    csf_vals = per_patient["CSF"].values

    ax.boxplot(
        [pb_vals, csf_vals], positions=[1, 2], widths=0.5,
        showfliers=False, patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        boxprops=dict(facecolor="#D9D9D9", alpha=0.5, edgecolor="black"),
        whiskerprops=dict(color="black"), capprops=dict(color="black"),
        zorder=1,
    )

    rng = np.random.default_rng(42)
    jitter = rng.normal(0, 0.04, size=len(pb_vals))
    for i in range(len(pb_vals)):
        direction_color = "#4C78A8" if csf_vals[i] >= pb_vals[i] else "#E45756"
        ax.plot(
            [1 + jitter[i], 2 + jitter[i]], [pb_vals[i], csf_vals[i]],
            color=direction_color, alpha=0.5, linewidth=1.2, zorder=5,
        )
        ax.scatter(
            [1 + jitter[i], 2 + jitter[i]], [pb_vals[i], csf_vals[i]],
            color=direction_color, s=24, alpha=0.8, edgecolors="black", linewidths=0.3, zorder=10,
        )

    if np.isfinite(p):
        print(
            f"[{group_label}] PAIRED PATIENT-LEVEL plot: n={n_pairs} subject pairs "
            f"(each line = one patient's PB->CSF trajectory), Wilcoxon signed-rank p={p:.3e}"
        )
    else:
        print(f"[{group_label}] n={n_pairs} pairs -- too few to test")

    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"PB\n(n={n_pairs})", f"CSF\n(n={n_pairs})"])
    score_label = "signed" if signed else "unsigned (plain mean)"
    ax.set_ylabel(f"Patient-level mean zinc/MT signature ({score_label})")
    title_p = f"{p:.2e}" if np.isfinite(p) else "n/a (too few paired subjects)"
    ax.set_title(
        f"{group_label} CXCR3+ – PB vs CSF ({score_label})\n"
        f"Paired Wilcoxon signed-rank p = {title_p} (n={n_pairs} subject pairs)",
        fontsize=9,
    )
    ax.spines[["top", "right"]].set_visible(False)
    _panel_letter(ax, letter)


def make_figS5_cxcr3pos_pb_vs_csf() -> None:
    """
    CXCR3+ counterpart to make_figS5_pb_vs_csf() (which covers CXCR3-),
    using the CXCR3+-specific pipeline functions duplicated in this file
    (with a _pos suffix) from fig3.py -- NOT imported cross-file, since
    figure scripts in this project each live in their own directory and
    aren't reliably importable from one another.

    This is the content that used to be Figure 3's main 4-panel figure --
    moved here because those per-gene/composite-signature results are
    non-significant for CXCR3+ cells (0/14 genes at either correction
    level; composite signature null in both HC and MS). Figure 3 itself
    now shows only the competitive rank test. This function, combined with
    make_figS5_pb_vs_csf() (CXCR3-), gives Supplementary Figure 5 full
    coverage of both populations' PB-vs-CSF comparisons.
    """
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

    print("Checking PB/CSF subject-ID pairing (CXCR3+) …")
    adata.obs["subject_id"] = resolve_pb_csf_subject_ids(
        adata, patient_col="patient", tissue_col="tissue"
    ).values

    out_dir = FIG_DIRS.get("supp_pb_vs_csf_paired", RESULTS_DIR / "supp" / "pb_vs_csf_paired")
    out_dir.mkdir(parents=True, exist_ok=True)

    zinc_genes = load_zinc_gene_list()
    gene_signs = load_zinc_gene_signs()
    print(f"Loaded {len(zinc_genes)} curated zinc/MT genes: {zinc_genes}")
    print(f"Gene signs for composite signature: {gene_signs}")

    # MS paired DE
    df_de_ms = _run_de_pb_vs_csf_paired_pos(
        adata, zinc_genes=zinc_genes, conditions=["MS"], label="MS", min_pct=0.10,
    )
    bar_genes_ms: list[str] = []
    if df_de_ms is not None:
        print(f" {df_de_ms.shape[0]} genes in PB vs CSF DE table (MS)")
        bar_genes_ms, ms_missing = _select_genes_for_bars_pos(df_de_ms, zinc_genes=zinc_genes)
        print(f"MS PB vs CSF bar genes ({len(bar_genes_ms)}/{len(zinc_genes)} of curated panel):", bar_genes_ms)
        if ms_missing:
            print(f" [note] MS: {len(ms_missing)} curated gene(s) not in dataset: {ms_missing}")

    # HC paired DE
    df_de_hc = _run_de_pb_vs_csf_paired_pos(
        adata, zinc_genes=zinc_genes, conditions=["HC"], label="HC", min_pct=0.10,
    )
    bar_genes_hc: list[str] = []
    if df_de_hc is not None:
        print(f" {df_de_hc.shape[0]} genes in PB vs CSF DE table (HC)")
        bar_genes_hc, hc_missing = _select_genes_for_bars_pos(df_de_hc, zinc_genes=zinc_genes)
        print(f"HC PB vs CSF bar genes ({len(bar_genes_hc)}/{len(zinc_genes)} of curated panel):", bar_genes_hc)
        if hc_missing:
            print(f" [note] HC: {len(hc_missing)} curated gene(s) not in dataset: {hc_missing}")

    if df_de_ms is not None and df_de_hc is not None:
        set_diff = set(bar_genes_ms).symmetric_difference(set(bar_genes_hc))
        if set_diff:
            print(f" [WARNING] HC and MS bar-chart gene sets differ: {sorted(set_diff)} -- panels are not directly comparable.")

    df_sig = compute_cell_signature_pb_vs_csf_pos(
        adata, zinc_genes=zinc_genes, gene_signs=None, grouping="demyelinating"
    )

    if "HC" in df_sig["group_label"].unique():
        df_sig[df_sig["group_label"] == "HC"].to_csv(
            out_dir / "FigS5_CXCR3pos_PB_vs_CSF_HC_zinc_signature_per_cell.csv",
            index=False
        )
    if "MS" in df_sig["group_label"].unique():
        df_sig[df_sig["group_label"] == "MS"].to_csv(
            out_dir / "FigS5_CXCR3pos_PB_vs_CSF_MS_zinc_signature_per_cell.csv",
            index=False
        )

    sig_test_hc = patient_signature_test_pb_vs_csf_pos(
        adata, zinc_genes=zinc_genes, gene_signs=None, grouping="demyelinating", group_label="HC"
    )
    sig_test_ms = patient_signature_test_pb_vs_csf_pos(
        adata, zinc_genes=zinc_genes, gene_signs=None, grouping="demyelinating", group_label="MS"
    )

    for name, res in [("HC", sig_test_hc), ("MS", sig_test_ms)]:
        pd.DataFrame([{k: v for k, v in res.items() if k != "per_patient_values"}]).to_csv(
            out_dir / f"FigS5_CXCR3pos_PB_vs_CSF_{name}_zinc_signature_patient_level_test.csv",
            index=False,
        )

    confound_hc = check_pb_vs_csf_dataset_confounding_pos(
        adata, zinc_genes=zinc_genes, grouping="demyelinating", group_label="HC", gene_signs=None
    )
    confound_ms = check_pb_vs_csf_dataset_confounding_pos(
        adata, zinc_genes=zinc_genes, grouping="demyelinating", group_label="MS", gene_signs=None
    )

    # ===== CREATE THE 6-PANEL FIGURE =====
    fig, axes = plt.subplots(3, 2, figsize=(12, 14))

    if df_de_ms is not None:
        _draw_volcano_pb_vs_csf_pos(
            axes[0, 0], df_de_ms,
            title="MS CXCR3+ B cells: CSF vs PB",
            label="A"
        )
    else:
        axes[0, 0].text(0.5, 0.5, "No MS data", ha="center", va="center", transform=axes[0, 0].transAxes)

    if df_de_ms is not None:
        _draw_bars_pb_vs_csf_pos(
            axes[0, 1], df_de_ms, bar_genes_ms,
            title="MS Zinc/MT genes: CSF vs PB",
            label="B",
            outlier_cap=3.0
        )
    else:
        axes[0, 1].text(0.5, 0.5, "No MS data", ha="center", va="center", transform=axes[0, 1].transAxes)

    if df_de_hc is not None:
        _draw_volcano_pb_vs_csf_pos(
            axes[1, 0], df_de_hc,
            title="HC CXCR3+ B cells: CSF vs PB",
            label="C"
        )
    else:
        axes[1, 0].text(0.5, 0.5, "No HC data", ha="center", va="center", transform=axes[1, 0].transAxes)

    if df_de_hc is not None:
        _draw_bars_pb_vs_csf_pos(
            axes[1, 1], df_de_hc, bar_genes_hc,
            title="HC Zinc/MT genes: CSF vs PB",
            label="D",
            outlier_cap=3.0
        )
    else:
        axes[1, 1].text(0.5, 0.5, "No HC data", ha="center", va="center", transform=axes[1, 1].transAxes)

    draw_paired_patient_plot_pos(
        axes[2, 0], sig_test_hc,
        group_label="HC",
        letter="E",
        signed=False
    )

    draw_paired_patient_plot_pos(
        axes[2, 1], sig_test_ms,
        group_label="MS",
        letter="F",
        signed=False
    )

    fig.tight_layout()

    stem = "FigS5_CXCR3pos_PB_vs_CSF_6panel"
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
    # Supplementary Figure 5 now covers BOTH CXCR3+ and CXCR3- PB-vs-CSF
    # comparisons -- CXCR3+ content moved here from Figure 3 (now
    # non-significant, so no longer the main figure), CXCR3- content is
    # this file's original analysis.
    make_figS5_cxcr3pos_pb_vs_csf()
    make_figS5_pb_vs_csf()

