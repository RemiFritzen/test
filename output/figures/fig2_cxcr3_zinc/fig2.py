#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
fig2.py — PB + CSF CXCR3+ MS vs HC zinc signature

Outputs:
  1. Main 4-panel figure:
     A = PB zinc/MT bar plot
     B = CSF zinc/MT bar plot
     C = PB violin (HC vs MS)
     D = CSF violin (HC vs MS)

  2. Supplementary scatter figures saved separately in the supp directory:
     - PB volcano/scatter
     - CSF volcano/scatter

Also exports DE tables and selected bar-gene CSVs.

CHANGE (patient-level statistics):
Both the per-gene DE (_run_de) and the composite signature comparison
(draw_violin) previously ran their statistical test directly on per-cell
values (sc.tl.rank_genes_groups / mannwhitneyu on every cell), even though
each patient contributes many correlated cells and each patient belongs to
only ONE condition (HC or MS). That's pseudoreplication: the effective n
used by the test is n_cells, not n_patients, which can turn a real but
modest patient-level effect into an artificially tiny p-value.

Both now use `pseudobulk_de` / `pseudobulk_signature_test` from
stats_utils.py: cells are first aggregated to one value per patient, and the
test runs on those ~n_patients values. This is the *unpaired* pseudobulk
form, which is correct here because HC vs MS is a between-patient
comparison (unlike CXCR3+/- or PB/CSF, where the same patient appears in
both arms and the *paired* form in stats_utils.py is needed instead -- see
run_all_zinc_analyses.py).

The per-cell scatter/violin plot itself is left as-is (it's a valid and
informative visualization of the cell-level distribution) -- only the
p-value annotation and gene-level statistics now come from the patient-level
test.
"""

import scipy.sparse as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
from statsmodels.stats.multitest import multipletests

from config.config import RESULTS_DIR, FIG_DIRS, DATASETS_YAML
from data_io import load_cfg
from stats_utils import pseudobulk_de, pseudobulk_signature_test, compute_gene_signs


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

SCATTER_COLOR_OTHER = "#CCCCCC"
SCATTER_COLOR_ZINC_PB = "#1f77b4"
SCATTER_COLOR_ZINC_CSF = "#E45756"
BAR_COLOR_PB = "#4C78A8"
BAR_COLOR_CSF = "#E45756"
SIG_PADJ_THRESHOLD = 0.05
MIN_PATIENTS_PER_GROUP = 3
PANEL_KW = dict(fontsize=14, fontweight="bold", va="top", ha="left")


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
    A priori sign for each gene in the zinc/MT panel (see compute_gene_signs
    in stats_utils.py): +1 for transporters, -1 for metallothioneins. Used to
    build the composite signature so up- and down-regulated genes contribute
    in the same direction instead of cancelling in a plain mean.
    """
    cfg = load_cfg(DATASETS_YAML)
    markers = cfg.get("markers", {})
    artefact = set(markers.get("exclude_artefacts", []))
    transporters = [g for g in markers.get("zinc_transporters", []) if g not in artefact]
    metallothioneins = [g for g in markers.get("metallothioneins", []) if g not in artefact]
    return compute_gene_signs(transporters, metallothioneins)


def _apply_min_pct_filter(sub: sc.AnnData, min_pct: float, always_keep: list[str] | None = None) -> sc.AnnData:
    """
    Drop lowly-expressed genes before DE (mainly to declutter the volcano
    plot and avoid testing genes with almost no signal). `always_keep`
    (typically the curated zinc/MT panel) is exempted from this filter --
    those genes are the entire point of the figure, so they're always
    tested and available for the bar chart regardless of expression
    prevalence, even if that means they end up correctly non-significant.
    """
    always_keep = set(always_keep or [])
    X = sub.X if not sp.issparse(sub.X) else sub.X.toarray()
    ms_m = sub.obs["condition"] == "MS"
    hc_m = sub.obs["condition"] == "HC"
    pct_ms = (X[ms_m] > 0).mean(axis=0)
    pct_hc = (X[hc_m] > 0).mean(axis=0)
    keep = (pct_ms >= min_pct) & (pct_hc >= min_pct)

    forced_idx = [sub.var_names.get_loc(g) for g in always_keep if g in sub.var_names]
    n_forced_back_in = int((~keep[forced_idx]).sum()) if forced_idx else 0
    for idx in forced_idx:
        keep[idx] = True

    print(f"    min_pct={min_pct:.0%}: {keep.sum()}/{len(keep)} genes retained"
          + (f" ({n_forced_back_in} curated zinc/MT gene(s) kept despite low prevalence)"
             if n_forced_back_in else ""))
    return sub[:, keep].copy()


def _check_gene_expression(adata: sc.AnnData, tissue: str, gene: str) -> None:
    sub = adata[
        (adata.obs["tissue"] == tissue)
        & (adata.obs["cxcr3_group"] == "CXCR3+")
        & adata.obs["condition"].isin(["HC", "MS"])
    ]
    if gene not in sub.var_names:
        print(f"  [diag] {gene} not in var_names")
        return
    print(f"\n  [diag] {gene} in {tissue} CXCR3+ cells:")
    for cond in ["HC", "MS"]:
        cells = sub[sub.obs["condition"] == cond]
        expr = cells[:, gene].X
        if sp.issparse(expr):
            expr = expr.toarray()
        expr = expr.flatten()
        print(
            f"    {cond}: n_cells={len(cells):4d}  "
            f"pct_expr={np.mean(expr > 0):.1%}  "
            f"mean={expr.mean():.4f}  "
            f"median={np.median(expr):.4f}"
        )


def compute_cell_signature(adata, zinc_genes: list[str], tissue: str, gene_signs: dict[str, float] | None = None) -> pd.DataFrame:
    """
    Per-cell zinc/MT signature, used only for the violin plot's visual
    distribution (jittered points, violin shape).

    gene_signs: if given (dict of {gene: +1/-1}), each gene's expression is
    sign-corrected before averaging (transporters +1, metallothioneins -1),
    so up- and down-regulated genes don't cancel toward zero -- see
    stats_utils.py for the full rationale. If None, computes a plain
    (unsigned) mean instead -- used for the supplementary comparison panel,
    since sign-correction encodes a directional assumption that isn't
    equally justified for every comparison (see draw_violin/make_fig2_combined).

    The p-value shown on the violin plot comes from the patient-level test
    (patient_signature_test), not from testing these per-cell values.
    """
    sub = adata[
        (adata.obs["tissue"] == tissue)
        & (adata.obs["cxcr3_group"] == "CXCR3+")
        & adata.obs["condition"].isin(["HC", "MS"])
    ].copy()

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

    df = sub.obs[["condition", "patient"]].copy()
    df["cell_id"] = sub.obs_names
    df["tissue"] = tissue
    df["zinc_signature"] = sig
    return df


def patient_signature_test(adata, zinc_genes: list[str], tissue: str, gene_signs: dict[str, float] | None = None) -> dict:
    """
    Patient-level (unpaired) test of the composite zinc/MT signature for HC
    vs MS CXCR3+ cells in one tissue. Each patient contributes one aggregated
    value; the test therefore has n = n_patients, not n_cells.

    gene_signs: see compute_cell_signature -- pass None for the unsigned
    (plain mean) variant, or a sign dict for the sign-corrected variant.
    """
    sub = adata[
        (adata.obs["tissue"] == tissue)
        & (adata.obs["cxcr3_group"] == "CXCR3+")
        & adata.obs["condition"].isin(["HC", "MS"])
    ].copy()
    genes = [g for g in zinc_genes if g in sub.var_names]

    return pseudobulk_signature_test(
        sub,
        groupby="condition",
        group1="HC",
        group2="MS",
        signature_genes=genes,
        patient_col="patient",
        use_raw=False,  # sub.X here is the same scaled matrix compute_cell_signature uses
        gene_signs=gene_signs,
    )


def check_dataset_confounding(
    adata,
    tissue: str,
    zinc_genes: list[str],
    gene_signs: dict[str, float] | None,
    signature_label: str,
    min_patients_per_group: int = 3,
) -> dict:
    """
    Check whether the MS-vs-HC composite zinc/MT signature result (the
    pooled test in patient_signature_test) is being driven by, or differs
    meaningfully between, dataset of origin (GSE133028 vs GSE138266) --
    addresses reviewer concern #4 (harmonization / study-of-origin
    confounding), which UMAP visualization (Fig 1B) alone cannot rule out
    for a specific downstream statistical result.

    Two complementary checks:

    1. STRATIFIED: rerun the same patient-level composite signature test
       WITHIN each dataset separately (n patients, direction, raw p per
       dataset). If the direction/significance pattern is wildly
       inconsistent between datasets, the pooled result may be driven
       disproportionately by one study rather than reflecting a shared
       disease effect. This is exploratory (not corrected for multiple
       testing) -- it's a diagnostic, not a replacement for the main
       pooled/corrected test.

    2. FORMAL INTERACTION TEST: fit OLS on patient-level pseudobulk
       signature values: signature ~ C(condition) * C(dataset). A
       significant condition:dataset interaction term is direct statistical
       evidence that the disease effect differs by dataset (i.e. genuine
       confounding/heterogeneity), rather than relying on eyeballing two
       separate p-values.
    """
    import statsmodels.formula.api as smf
    from scipy.stats import mannwhitneyu
    from stats_utils import pseudobulk_matrix

    sub = adata[
        (adata.obs["tissue"] == tissue)
        & (adata.obs["cxcr3_group"] == "CXCR3+")
        & adata.obs["condition"].isin(["HC", "MS"])
    ].copy()
    genes = [g for g in zinc_genes if g in sub.var_names]

    pb = pseudobulk_matrix(sub, groupby="condition", patient_col="patient", genes=genes, use_raw=False)
    sign_vec = np.array([
        (gene_signs.get(g, 1.0) if gene_signs is not None else 1.0) for g in genes
    ])
    pb["signature"] = (pb[genes].values * sign_vec[np.newaxis, :]).mean(axis=1)

    # attach dataset per patient (assumes 1:1 patient -> dataset, same
    # assumption pseudobulk_matrix already makes for condition)
    patient_dataset = sub.obs.drop_duplicates("patient").set_index("patient")["dataset"]
    pb["dataset"] = patient_dataset.reindex(pb.index)

    print(f"\n  === Dataset confounding check: {tissue}, {signature_label} signature ===")

    per_dataset_results = []
    for ds in sorted(pb["dataset"].dropna().unique()):
        ds_pb = pb[pb["dataset"] == ds]
        hc = ds_pb.loc[ds_pb["_group"] == "HC", "signature"]
        ms = ds_pb.loc[ds_pb["_group"] == "MS", "signature"]
        if len(hc) < min_patients_per_group or len(ms) < min_patients_per_group:
            print(f"    [{ds}] n={len(hc)} HC, {len(ms)} MS -- too few patients to test "
                  f"(< {min_patients_per_group}/group); skipping.")
            per_dataset_results.append({
                "dataset": ds, "n_HC": len(hc), "n_MS": len(ms),
                "median_diff": np.nan, "pvalue": np.nan,
            })
            continue
        stat, p = mannwhitneyu(hc, ms, alternative="two-sided")
        median_diff = ms.median() - hc.median()
        print(f"    [{ds}] n={len(hc)} HC, {len(ms)} MS: median(MS-HC)={median_diff:+.3f}, "
              f"Mann-Whitney p={p:.3g}")
        per_dataset_results.append({
            "dataset": ds, "n_HC": len(hc), "n_MS": len(ms),
            "median_diff": median_diff, "pvalue": p,
        })

    per_dataset_df = pd.DataFrame(per_dataset_results)

    # Consistency flag: do all datasets that could be tested agree on direction?
    testable = per_dataset_df.dropna(subset=["pvalue"])
    directions_agree = testable["median_diff"].apply(np.sign).nunique() <= 1 if len(testable) > 1 else None
    if directions_agree is False:
        print(f"    [WARNING] Direction of the MS-vs-HC effect is INCONSISTENT across datasets "
              f"-- the pooled result may not reflect a shared disease effect. Investigate before "
              f"reporting the pooled signature test as dataset-independent.")
    elif directions_agree is True:
        print(f"    Direction is consistent across all testable datasets.")

    # Formal interaction test
    interaction_p = np.nan
    if pb["dataset"].nunique() >= 2 and pb["_group"].nunique() >= 2:
        model_df = pb.dropna(subset=["signature", "dataset", "_group"]).rename(columns={"_group": "condition"})
        try:
            ols = smf.ols("signature ~ C(condition) * C(dataset)", data=model_df).fit()
            interaction_terms = [t for t in ols.pvalues.index if ":" in t]
            if interaction_terms:
                interaction_p = ols.pvalues[interaction_terms[0]]
                verdict = "SIGNIFICANT interaction -- effect differs by dataset" if interaction_p < 0.05 \
                    else "no significant interaction -- effect is consistent across datasets"
                print(f"    Interaction test (condition x dataset): p={interaction_p:.3g} ({verdict})")
        except Exception as exc:  # noqa: BLE001
            print(f"    [warn] interaction model failed to fit: {exc}")

    return {
        "tissue": tissue, "signature_label": signature_label,
        "per_dataset": per_dataset_df,
        "directions_agree": directions_agree,
        "interaction_pvalue": interaction_p,
    }


def _run_de(
    adata: sc.AnnData,
    tissue: str,
    zinc_genes: list[str],
    min_pct: float = 0.10,
) -> pd.DataFrame:
    sub = adata[
        (adata.obs["tissue"] == tissue)
        & (adata.obs["cxcr3_group"] == "CXCR3+")
        & adata.obs["condition"].isin(["HC", "MS"])
    ].copy()

    if sub.n_obs == 0:
        raise ValueError(f"No {tissue} CXCR3+ HC/MS cells found.")
    if not {"HC", "MS"}.issubset(set(sub.obs["condition"].unique())):
        raise ValueError(f"{tissue} CXCR3+ missing HC or MS condition.")
    if "patient" not in sub.obs.columns:
        raise KeyError("Column 'patient' missing in adata.obs -- required for patient-level DE.")

    if not sub.var_names.is_unique:
        print(f"    [fix] {tissue}: duplicate gene symbols in var_names -- making unique "
              f"(otherwise DE indexing can crash or misalign genes/bars downstream)")
        sub.var_names_make_unique()

    missing_from_data = [g for g in zinc_genes if g not in sub.var_names]
    if missing_from_data:
        print(f"    [warn] {tissue}: {len(missing_from_data)} curated zinc/MT gene(s) not in "
              f"the dataset at all (not just filtered): {missing_from_data}")

    sub = _apply_min_pct_filter(sub, min_pct=min_pct, always_keep=zinc_genes)

    n_hc = sub.obs.loc[sub.obs["condition"] == "HC", "patient"].nunique()
    n_ms = sub.obs.loc[sub.obs["condition"] == "MS", "patient"].nunique()
    print(f"    {tissue} CXCR3+ patients: {n_hc} HC, {n_ms} MS "
          f"({sub.n_obs} cells total)")

    # Cells-per-patient distribution, split by condition. This is what
    # actually determines how noisy each patient's pseudobulk value is --
    # fewer CXCR3+ cells per patient (as expected in PB, where CXCR3+ cells
    # are rarer than in CSF per Fig 1E) means each patient's aggregated
    # value averages over less signal, which is a direct, checkable
    # explanation for weaker per-gene significance in one tissue vs another
    # (independent of whether the underlying patient-level effect size
    # itself also differs between tissues).
    cells_per_patient = sub.obs.groupby(["condition", "patient"]).size()
    for cond in ["HC", "MS"]:
        if cond in cells_per_patient.index.get_level_values("condition"):
            vals = cells_per_patient.loc[cond]
            print(f"      {cond} CXCR3+ cells/patient: "
                  f"median={vals.median():.0f}, min={vals.min()}, max={vals.max()} "
                  f"(n={len(vals)} patients)")
    if n_hc < MIN_PATIENTS_PER_GROUP or n_ms < MIN_PATIENTS_PER_GROUP:
        print(f"    [warn] fewer than {MIN_PATIENTS_PER_GROUP} patients in a group -- "
              f"treat {tissue} results as exploratory, not confirmatory.")

    # Each patient belongs to exactly ONE condition (HC or MS): this is an
    # unpaired, between-patient comparison. pseudobulk_de aggregates cells to
    # one value per patient first, so the Mann-Whitney test below runs on
    # ~n_patients values, not ~n_cells values.
    # use_raw=True (default) pulls from adata.raw, matching what
    # sc.tl.rank_genes_groups used by default (log1p-scale data); the
    # log2fc is therefore computed as a difference of mean log values,
    # converted to log2 units (log2fc_from_log_means=True, the default) --
    # NOT log2 of a ratio of means-of-logs.
    df = pseudobulk_de(
        sub,
        groupby="condition",
        group1="HC",
        group2="MS",
        patient_col="patient",
        min_patients_per_group=MIN_PATIENTS_PER_GROUP,
    )
    df = df.rename(columns={"log2fc": "log2FC", "pval": "pvalue"})
    df = df[np.isfinite(df["log2FC"])].copy()
    df["is_zinc"] = df["gene"].isin(zinc_genes)

    # Two FDR corrections, for two different claims:
    #   'padj'            -- genome-wide (all ~n_background_genes tested),
    #                        appropriate for the volcano plot, which is an
    #                        exploratory scan across the whole tested set.
    #   'padj_zinc_panel' -- BH correction restricted to ONLY the curated
    #                        zinc/MT genes. The zinc panel is a small,
    #                        pre-specified, hypothesis-driven set (not
    #                        discovered by screening the transcriptome), so
    #                        correcting it against thousands of unrelated
    #                        background genes is the wrong correction
    #                        family for that specific, narrower claim and
    #                        can suppress genuinely significant panel genes
    #                        purely due to the size of the background gene
    #                        set. Used for the bar-chart significance stars.
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
    was actually tested -- see _apply_min_pct_filter's always_keep), sorted
    by log2FC descending. This shows every curated gene (not a top-N
    subset), just ordered by effect size rather than the arbitrary config
    listing order. Any curated gene missing from `df` entirely (not in the
    dataset) is returned separately so the caller can report it rather than
    silently omit it.
    """
    tested = df[df["gene"].isin(zinc_genes)].copy()
    tested = tested.sort_values("log2FC", ascending=False)
    genes_present = tested["gene"].tolist()
    genes_missing = [g for g in zinc_genes if g not in set(genes_present)]
    return genes_present, genes_missing


def _draw_volcano(ax: plt.Axes, df: pd.DataFrame, title: str, label: str, zinc_color: str) -> None:
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
            s=28, color=zinc_color, alpha=0.4,
            edgecolors="black", linewidths=0.4,
            label="Zinc/MT (ns)",
        )
    if not zinc_sig.empty:
        ax.scatter(
            zinc_sig["log2FC"], zinc_sig["-log10padj"],
            s=45, color=zinc_color, alpha=0.9,
            edgecolors="black", linewidths=0.6,
            label=f"Zinc/MT (padj < {SIG_PADJ_THRESHOLD})",
        )

    ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.axhline(-np.log10(SIG_PADJ_THRESHOLD), color="grey", linestyle=":", linewidth=0.8)
    ax.set_xlabel("log2FC (MS vs HC)")
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


def _draw_bars(
    ax: plt.Axes,
    df: pd.DataFrame,
    genes: list[str],
    title: str,
    label: str,
    color: str,
    outlier_cap: float | None = None,
) -> None:
    if not genes:
        ax.text(
            0.5, 0.5,
            "No zinc/metallothionein genes available",
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
    # Stars use padj corrected within the curated zinc/MT panel only (see
    # _run_de) -- NOT the genome-wide padj used for the volcano's threshold
    # line -- since correcting this small, pre-specified panel against
    # thousands of unrelated background genes is overly conservative for
    # what this chart is actually claiming.
    padj_v = df_b.set_index("gene").loc[genes, "padj_zinc_panel"].values

    fc_plot = fc_raw.copy()
    capped_mask = np.zeros(len(fc_raw), dtype=bool)
    if outlier_cap is not None:
        capped_mask = np.abs(fc_raw) > outlier_cap
        fc_plot[capped_mask] = np.sign(fc_raw[capped_mask]) * outlier_cap * 0.9

    for i in range(len(genes)):
        hatch = "//" if capped_mask[i] else None
        ax.bar(x[i], fc_plot[i], width=0.65, color=color, edgecolor="black", linewidth=0.6, hatch=hatch)

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
    ax.set_ylabel("log2FC (MS vs HC)")
    ax.set_title(title, fontsize=10)
    _panel_letter(ax, label)


def draw_patient_bar(ax, sig_test_result, tissue_label, letter, bar_color, signed: bool = True):
    """
    Box plot + jittered patient dots of PATIENT-LEVEL pseudobulk composite
    signature values: median/IQR/whiskers per group (HC, MS), with each
    patient's own pseudobulk value plotted as an individual dot.

    Box plot rather than a bar chart or violin: with n as low as 6-8
    patients in the HC groups here, a violin's kernel density estimate is
    mostly smoothing artifact, not a real distribution shape, and can
    visually suggest structure (bimodality, skew) that isn't really there.
    A box plot makes no such shape assumption -- median/IQR/range come
    directly from the actual points, which are also all shown as dots on
    top regardless, so nothing about the raw data is hidden either way.

    This shows exactly the data the Mann-Whitney test operates on (patient-
    level pseudobulk means), NOT the thousands of raw cells a per-cell
    violin would show -- a per-cell summary (even a median) can look
    unremarkable or misleading relative to the actual test, since it's
    computed from a completely different, much larger and differently
    -weighted set of numbers (e.g. a per-cell median can tie across groups
    due to a shared zero-inflated floor, while patient-level means differ
    clearly and significantly).

    `signed` only controls the axis/title label -- `sig_test_result` must
    already have been computed with the matching gene_signs by the caller.
    """
    per_patient = sig_test_result.get("per_patient_values")
    p = sig_test_result.get("mannwhitney_p", np.nan)
    n_hc = sig_test_result.get("n_patients_HC", "?")
    n_ms = sig_test_result.get("n_patients_MS", "?")

    if per_patient is None or per_patient.empty:
        ax.text(0.5, 0.5, "No per-patient data available", ha="center", va="center",
                transform=ax.transAxes)
        _panel_letter(ax, letter)
        return

    groups = ["HC", "MS"]
    rng = np.random.default_rng(42)
    box_data = [per_patient.loc[per_patient["_group"] == g, "signature"].values for g in groups]

    ax.boxplot(
        box_data, positions=[1, 2], widths=0.5,
        showfliers=False, patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        boxprops=dict(facecolor=bar_color, alpha=0.35, edgecolor="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
    )

    for i, vals in enumerate(box_data, start=1):
        if len(vals) == 0:
            continue
        jitter = rng.normal(0, 0.06, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, color="black", s=28,
                   alpha=0.75, edgecolors="black", linewidths=0.4, zorder=10)

    if np.isfinite(p):
        print(
            f"[{tissue_label}] PATIENT-LEVEL box plot: n={n_hc} HC / {n_ms} MS patients "
            f"(each dot = one patient's pseudobulk mean), Mann-Whitney p={p:.3e}"
        )
    else:
        print(f"[{tissue_label}] n={n_hc} HC / {n_ms} MS -- too few to test")

    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"HC\n(n={n_hc})", f"MS\n(n={n_ms})"])
    score_label = "signed" if signed else "unsigned (plain mean)"
    ax.set_ylabel(f"Patient-level mean zinc/MT signature ({score_label})")
    title_p = f"{p:.2e}" if np.isfinite(p) else "n/a (too few patients)"
    ax.set_title(
        f"{tissue_label} CXCR3+ B cells – MS vs HC\n"
        f"Patient-level Mann–Whitney p = {title_p} (n={n_hc} HC, {n_ms} MS patients)",
        fontsize=9,
    )
    ax.spines[["top", "right"]].set_visible(False)
    _panel_letter(ax, letter)


def draw_violin(ax, df_sig, sig_test_result, tissue_label, letter, bar_color, signed: bool = True):
    """
    Draws the per-cell violin/jitter (visual only) and annotates it with the
    patient-level (pseudobulk) test result passed in via `sig_test_result`,
    NOT a Mann-Whitney test on the per-cell values plotted here.

    `signed` only controls the axis/title label (which composite score is
    being shown) -- the data in df_sig / sig_test_result must already be
    computed accordingly by the caller (see compute_cell_signature /
    patient_signature_test's gene_signs argument).
    """
    groups = ["HC", "MS"]
    data = [df_sig.loc[df_sig["condition"] == g, "zinc_signature"].values for g in groups]

    parts = ax.violinplot(data, positions=[1, 2], showmeans=False, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("#CCCCCC")
        pc.set_edgecolor("black")
        pc.set_alpha(0.7)
    parts["cmedians"].set_color("black")

    for i, vals in enumerate(data, start=1):
        jitter = np.random.normal(0, 0.04, size=len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter,
            vals,
            s=6,
            alpha=0.3,
            color="black",
            zorder=10,
        )
        med = np.median(vals)
        ax.hlines(med, i - 0.16, i + 0.16, color=bar_color, linewidth=3.0, zorder=20)

    p = sig_test_result.get("mannwhitney_p", np.nan)
    n_hc = sig_test_result.get("n_patients_HC", "?")
    n_ms = sig_test_result.get("n_patients_MS", "?")
    hc_vals, ms_vals = data[0], data[1]
    if np.isfinite(p):
        print(
            f"[{tissue_label}] cell-level HC median={np.median(hc_vals):.3f}, "
            f"MS median={np.median(ms_vals):.3f} "
            f"({len(hc_vals)} HC cells, {len(ms_vals)} MS cells) | "
            f"PATIENT-LEVEL test: n={n_hc} HC / {n_ms} MS patients, "
            f"Mann-Whitney p={p:.3e}"
        )
    else:
        print(
            f"[{tissue_label}] PATIENT-LEVEL test: n={n_hc} HC / {n_ms} MS patients "
            f"-- too few patients to test"
        )

    ax.set_xticks([1, 2])
    ax.set_xticklabels(groups)
    score_label = "signed" if signed else "unsigned (plain mean)"
    ax.set_ylabel(f"Zinc/MT signature ({score_label}, per cell)")
    title_p = f"{p:.2e}" if np.isfinite(p) else "n/a (too few patients)"
    ax.set_title(
        f"{tissue_label} CXCR3+ B cells – MS vs HC ({score_label})\n"
        f"Patient-level Mann–Whitney p = {title_p} (n={n_hc} HC, {n_ms} MS patients)",
        fontsize=9,
    )
    _panel_letter(ax, letter)


def _save_single_volcano(df, title, label, zinc_color, out_png, out_pdf):
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    _draw_volcano(ax, df, title=title, label=label, zinc_color=zinc_color)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def make_fig2_combined() -> None:
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

    zinc_genes = load_zinc_gene_list()
    gene_signs = load_zinc_gene_signs()
    print(f"Loaded {len(zinc_genes)} curated zinc/MT genes: {zinc_genes}")
    print(f"Gene signs for composite signature: {gene_signs}")

    print("\nRunning patient-level DE: PB CXCR3+ MS vs HC …")
    df_pb = _run_de(adata, tissue="PB", zinc_genes=zinc_genes, min_pct=0.10)
    print(f"  {df_pb.shape[0]} genes in PB DE table")

    print("\nRunning patient-level DE: CSF CXCR3+ MS vs HC …")
    df_csf = _run_de(adata, tissue="CSF", zinc_genes=zinc_genes, min_pct=0.15)
    print(f"  {df_csf.shape[0]} genes in CSF DE table")

    df_sig_pb = compute_cell_signature(adata, zinc_genes, tissue="PB", gene_signs=gene_signs)
    df_sig_csf = compute_cell_signature(adata, zinc_genes, tissue="CSF", gene_signs=gene_signs)
    # Save per-cell zinc signatures used for the violin plot visuals (Fig 2C, 2D)
    df_sig_pb.to_csv(RESULTS_DIR / "Fig2_PB_CXCR3pos_zinc_signature_per_cell.csv", index=False)
    df_sig_csf.to_csv(RESULTS_DIR / "Fig2_CSF_CXCR3pos_zinc_signature_per_cell.csv", index=False)

    # Patient-level signature test -- this is what's actually reported as the
    # Fig 2C/2D p-value now, replacing the per-cell Mann-Whitney test.
    sig_test_pb = patient_signature_test(adata, zinc_genes, tissue="PB", gene_signs=gene_signs)
    sig_test_csf = patient_signature_test(adata, zinc_genes, tissue="CSF", gene_signs=gene_signs)
    pd.DataFrame([{k: v for k, v in sig_test_pb.items() if k != "per_patient_values"}]).to_csv(
        RESULTS_DIR / "Fig2_PB_CXCR3pos_zinc_signature_patient_level_test.csv", index=False
    )
    pd.DataFrame([{k: v for k, v in sig_test_csf.items() if k != "per_patient_values"}]).to_csv(
        RESULTS_DIR / "Fig2_CSF_CXCR3pos_zinc_signature_patient_level_test.csv", index=False
    )

    # Dataset-of-origin confounding check (reviewer concern #4): does the
    # pooled MS-vs-HC signature result hold within each dataset separately,
    # or is it being driven disproportionately by one study?
    confound_pb = check_dataset_confounding(adata, tissue="PB", zinc_genes=zinc_genes,
                                             gene_signs=gene_signs, signature_label="signed")
    confound_csf = check_dataset_confounding(adata, tissue="CSF", zinc_genes=zinc_genes,
                                              gene_signs=gene_signs, signature_label="signed")
    confound_pb["per_dataset"].to_csv(
        RESULTS_DIR / "Fig2_PB_CXCR3pos_dataset_confounding_check.csv", index=False
    )
    confound_csf["per_dataset"].to_csv(
        RESULTS_DIR / "Fig2_CSF_CXCR3pos_dataset_confounding_check.csv", index=False
    )
    pd.DataFrame([
        {"tissue": "PB", "directions_agree": confound_pb["directions_agree"],
         "interaction_pvalue": confound_pb["interaction_pvalue"]},
        {"tissue": "CSF", "directions_agree": confound_csf["directions_agree"],
         "interaction_pvalue": confound_csf["interaction_pvalue"]},
    ]).to_csv(RESULTS_DIR / "Fig2_dataset_confounding_summary.csv", index=False)

    # Unsigned (plain-mean) version of the same signature, computed
    # alongside the sign-corrected one for transparency -- see
    # make_fig2_combined's supplementary figure below and the discussion of
    # why sign-correction is a defensible-but-not-free-of-assumptions choice
    # (the sign scheme is a priori/biology-based, not derived from this
    # comparison's own DE result, but a reader should still be able to see
    # the plain average for comparison).
    df_sig_pb_unsigned = compute_cell_signature(adata, zinc_genes, tissue="PB", gene_signs=None)
    df_sig_csf_unsigned = compute_cell_signature(adata, zinc_genes, tissue="CSF", gene_signs=None)
    df_sig_pb_unsigned.to_csv(RESULTS_DIR / "Fig2_PB_CXCR3pos_zinc_signature_UNSIGNED_per_cell.csv", index=False)
    df_sig_csf_unsigned.to_csv(RESULTS_DIR / "Fig2_CSF_CXCR3pos_zinc_signature_UNSIGNED_per_cell.csv", index=False)

    sig_test_pb_unsigned = patient_signature_test(adata, zinc_genes, tissue="PB", gene_signs=None)
    sig_test_csf_unsigned = patient_signature_test(adata, zinc_genes, tissue="CSF", gene_signs=None)
    pd.DataFrame([{k: v for k, v in sig_test_pb_unsigned.items() if k != "per_patient_values"}]).to_csv(
        RESULTS_DIR / "Fig2_PB_CXCR3pos_zinc_signature_UNSIGNED_patient_level_test.csv", index=False
    )
    pd.DataFrame([{k: v for k, v in sig_test_csf_unsigned.items() if k != "per_patient_values"}]).to_csv(
        RESULTS_DIR / "Fig2_CSF_CXCR3pos_zinc_signature_UNSIGNED_patient_level_test.csv", index=False
    )
    print(
        f"\nSigned vs unsigned signature p-values (sanity check -- both should point the "
        f"same direction even if only one reaches significance):\n"
        f"  PB : signed p={sig_test_pb.get('mannwhitney_p', float('nan')):.3e}   "
        f"unsigned p={sig_test_pb_unsigned.get('mannwhitney_p', float('nan')):.3e}\n"
        f"  CSF: signed p={sig_test_csf.get('mannwhitney_p', float('nan')):.3e}   "
        f"unsigned p={sig_test_csf_unsigned.get('mannwhitney_p', float('nan')):.3e}"
    )

    OUTLIER_FC_THRESHOLD = 5.0
    csf_extreme = df_csf[df_csf["is_zinc"] & (df_csf["log2FC"].abs() > OUTLIER_FC_THRESHOLD)]
    if not csf_extreme.empty:
        print(f"\n[WARNING] {len(csf_extreme)} zinc gene(s) with |log2FC| > {OUTLIER_FC_THRESHOLD} in CSF:")
        for _, row in csf_extreme.iterrows():
            print(f"  {row['gene']}: log2FC={row['log2FC']:.2f}, padj={row['padj']:.3e}")
            _check_gene_expression(adata, tissue="CSF", gene=row["gene"])

    pb_bar_genes, pb_missing = _select_genes_for_bars(df_pb, zinc_genes=zinc_genes)
    csf_bar_genes, csf_missing = _select_genes_for_bars(df_csf, zinc_genes=zinc_genes)
    print(f"PB bar genes  ({len(pb_bar_genes)}/{len(zinc_genes)} of curated panel):", pb_bar_genes)
    if pb_missing:
        print(f"  [note] PB: {len(pb_missing)} curated gene(s) not in dataset, excluded from bar chart: {pb_missing}")
    print(f"CSF bar genes ({len(csf_bar_genes)}/{len(zinc_genes)} of curated panel):", csf_bar_genes)
    if csf_missing:
        print(f"  [note] CSF: {len(csf_missing)} curated gene(s) not in dataset, excluded from bar chart: {csf_missing}")

    for name, df_de in (("PB", df_pb), ("CSF", df_csf)):
        zinc_rows = df_de[df_de["is_zinc"]]
        n_sig_panel = int((zinc_rows["padj_zinc_panel"] < SIG_PADJ_THRESHOLD).sum())
        n_sig_genomewide = int((zinc_rows["padj"] < SIG_PADJ_THRESHOLD).sum())
        print(f"  [{name}] zinc/MT genes significant: {n_sig_panel}/{len(zinc_rows)} "
              f"(panel-restricted FDR, used for bar-chart stars) vs "
              f"{n_sig_genomewide}/{len(zinc_rows)} (genome-wide FDR, used for volcano)")
        print(f"  [{name}] full per-gene table (verify star count against these numbers):")
        print(zinc_rows[["gene", "log2FC", "pvalue", "padj", "padj_zinc_panel"]]
              .sort_values("log2FC", ascending=False)
              .to_string(index=False))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axA, axB = axes[0]
    axC, axD = axes[1]

    n_hc_pb = sig_test_pb.get("n_patients_HC", "?")
    n_ms_pb = sig_test_pb.get("n_patients_MS", "?")
    n_hc_csf = sig_test_csf.get("n_patients_HC", "?")
    n_ms_csf = sig_test_csf.get("n_patients_MS", "?")

    _draw_bars(
        axA, df_pb, pb_bar_genes,
        title=f"PB CXCR3+ – zinc/metallothionein genes\n(n={n_hc_pb} HC, {n_ms_pb} MS)",
        label="A",
        color=BAR_COLOR_PB,
    )
    _draw_bars(
        axB, df_csf, csf_bar_genes,
        title=f"CSF CXCR3+ – zinc/metallothionein genes\n(n={n_hc_csf} HC, {n_ms_csf} MS)",
        label="B",
        color=BAR_COLOR_CSF,
    )
    draw_patient_bar(axC, sig_test_pb, tissue_label="PB", letter="C", bar_color=BAR_COLOR_PB)
    draw_patient_bar(axD, sig_test_csf, tissue_label="CSF", letter="D", bar_color=BAR_COLOR_CSF)

    fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.10, wspace=0.30, hspace=0.38)

    out_dir = FIG_DIRS.get("fig2", RESULTS_DIR / "fig2")
    out_dir.mkdir(parents=True, exist_ok=True)
    supp_dir = out_dir / "supp"
    supp_dir.mkdir(parents=True, exist_ok=True)

    stem = "Fig2_CXCR3pos_PB_CSF_MS_vs_HC_4panel"
    out_png = out_dir / f"{stem}.png"
    out_pdf = out_dir / f"{stem}.pdf"

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    _save_single_volcano(
        df_pb,
        title=f"PB CXCR3+ B cells – MS vs HC (n={n_hc_pb} HC, {n_ms_pb} MS)",
        label="A",
        zinc_color=SCATTER_COLOR_ZINC_PB,
        out_png=supp_dir / "Fig2_supp_PB_volcano.png",
        out_pdf=supp_dir / "Fig2_supp_PB_volcano.pdf",
    )
    _save_single_volcano(
        df_csf,
        title=f"CSF CXCR3+ B cells – MS vs HC (n={n_hc_csf} HC, {n_ms_csf} MS)",
        label="B",
        zinc_color=SCATTER_COLOR_ZINC_CSF,
        out_png=supp_dir / "Fig2_supp_CSF_volcano.png",
        out_pdf=supp_dir / "Fig2_supp_CSF_volcano.pdf",
    )

    # Supplementary: unsigned (plain-mean) composite signature, side by side
    # with the sign-corrected version used in the main figure (panels C/D
    # above). Shown so a reader can check that sign-correction isn't doing
    # something the plain average disagrees with in direction -- see the
    # console printout above ("Signed vs unsigned signature p-values") for
    # the numeric comparison this figure illustrates.
    fig_supp_sig, (axC_u, axD_u) = plt.subplots(1, 2, figsize=(10, 4.5))
    draw_patient_bar(
        axC_u, sig_test_pb_unsigned,
        tissue_label="PB", letter="C", bar_color=BAR_COLOR_PB, signed=False,
    )
    draw_patient_bar(
        axD_u, sig_test_csf_unsigned,
        tissue_label="CSF", letter="D", bar_color=BAR_COLOR_CSF, signed=False,
    )
    fig_supp_sig.suptitle(
        "Supplementary: unsigned (plain-mean) zinc/MT signature\n"
        "(compare against main Fig 2C/D, which use the sign-corrected version)",
        fontsize=10, y=1.04,
    )
    fig_supp_sig.tight_layout()
    fig_supp_sig.savefig(supp_dir / "Fig2_supp_unsigned_signature_violin.png", dpi=300, bbox_inches="tight")
    fig_supp_sig.savefig(supp_dir / "Fig2_supp_unsigned_signature_violin.pdf", bbox_inches="tight")
    plt.close(fig_supp_sig)

    df_pb.to_csv(out_dir / "Fig2_PB_CXCR3pos_MS_vs_HC_full_DE.csv", index=False)
    df_csf.to_csv(out_dir / "Fig2_CSF_CXCR3pos_MS_vs_HC_full_DE.csv", index=False)
    pd.DataFrame({"gene": pb_bar_genes}).to_csv(out_dir / "Fig2_PB_CXCR3pos_zinc_bar_genes.csv", index=False)
    pd.DataFrame({"gene": csf_bar_genes}).to_csv(out_dir / "Fig2_CSF_CXCR3pos_zinc_bar_genes.csv", index=False)

    print(f"\nSaved: {out_png}")
    print(f"Saved: {out_pdf}")
    print(f"Saved supplementary scatter plots and unsigned-signature comparison in: {supp_dir}")


if __name__ == "__main__":
    make_fig2_combined()