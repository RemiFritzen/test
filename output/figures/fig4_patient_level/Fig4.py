#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig4.py
Patient-level Figure 4 for CXCR3 / zinc project.

Main figure:
- A/B: zinc signature vs EDSS in CSF / PB
- C/D: HC vs MS boxplots with on-panel Mann-Whitney p-values

Supplementary figures:
- HC / MS boxplots for zinc signature in CSF / PB
- One PDF per gene, with 2 panels each: CSF vs EDSS and PB vs EDSS

Runs for TWO populations (CXCR3+ B cells, primary; all B cells regardless
of CXCR3 status, supplementary) -- see run_fig4_for_population(). Output
is organized as:

    <fig_dir>/<file_tag>/main/   -- main 4-panel figure + its panel CSVs
    <fig_dir>/<file_tag>/supp/   -- boxplots, per-gene EDSS PDFs, LOO
                                     robustness checks, unsigned-signature
                                     comparison, correlation tables

where <file_tag> is "CXCR3pos" or "AllBcells", so both the population and
the main-vs-supplementary distinction are visible directly in the folder
path rather than only in filename prefixes.
"""

from __future__ import annotations

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from scipy.stats import spearmanr, mannwhitneyu, kruskal
from statsmodels.stats.multitest import multipletests

from config.config import FIG_DIRS, DATASETS_YAML, RESULTS_DIR
from data_io import load_cfg
from stats_utils import compute_gene_signs

sns.set(style="whitegrid", context="talk")
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

TARGET_DATASETS = {"GSE133028", "GSE138266"}
DIAG_MAP = {"RRMS": "MS", "MS": "MS", "HC": "HC"}
EXCLUDE_TREATED = {"18", "26"}

PAL3 = {"HC": "#757575", "MS": "#1b9e77"}
PAL2 = {"HC": "#757575", "Disease": "#1b9e77"}


def load_markers(cfg: dict) -> list[str]:
    markers = cfg.get("markers", {})
    mt = markers.get("metallothioneins", [])
    zt = markers.get("zinc_transporters", [])
    artefacts = set(markers.get("exclude_artefacts", []))
    return [g for g in (mt + zt) if g not in artefacts]


def load_marker_categories(cfg: dict) -> tuple[list[str], list[str]]:
    """
    Transporter and metallothionein gene lists kept separate (not merged),
    needed to build the a priori gene_signs dict for the sign-corrected
    composite signature -- see compute_gene_signs in stats_utils.py.
    """
    markers = cfg.get("markers", {})
    artefacts = set(markers.get("exclude_artefacts", []))
    transporters = [g for g in markers.get("zinc_transporters", []) if g not in artefacts]
    mts = [g for g in markers.get("metallothioneins", []) if g not in artefacts]
    return transporters, mts


def load_clinical_metadata() -> pd.DataFrame:
    rows = [
        ["1", 43, "F", "RRMS", 0, np.nan, "Untreated", 16],
        ["2", 33, "M", "HC", np.nan, np.nan, "Untreated", np.nan],
        ["3", 27, "F", "HC", np.nan, np.nan, "Untreated", np.nan],
        ["6", 44, "M", "RRMS", 2, np.nan, "Untreated", 1],
        ["7", 37, "M", "RRMS", 1.5, np.nan, "Untreated", 14],
        ["8", 32, "M", "RRMS", 1.5, np.nan, "Untreated", 4],
        ["9", 27, "F", "RRMS", 3.5, np.nan, "Untreated", 27],
        ["10", 45, "F", "RRMS", 2, np.nan, "Untreated", 3],
        ["16", 22, "M", "RRMS", 4, np.nan, "Untreated", 0],
        ["18", 29, "F", "RRMS", 2, np.nan, "Steroids", 32],
        ["22", 35, "F", "RRMS", 4, np.nan, "Untreated", 1],
        ["24", 45, "F", "RRMS", 0, np.nan, "Untreated", 11],
        ["25", 42, "F", "RRMS", 1.5, np.nan, "Untreated", 4],
        ["26", 32, "F", "RRMS", 2, np.nan, "Steroids", 2],
        ["27", 37, "F", "RRMS", 1.5, np.nan, "Untreated", 166],
        ["28", 50, "F", "RRMS", 4, np.nan, "Untreated", 9],
        ["29", 54, "M", "RRMS", 2, np.nan, "Untreated", 49],
        ["31", 53, "F", "RRMS", 2.5, np.nan, "Untreated", 3],
        ["32", 41, "M", "HC", np.nan, np.nan, "Untreated", np.nan],
        ["PST95809", 43, "F", "HC", np.nan, np.nan, "Untreated", np.nan],
        ["PST83775", 43, "M", "HC", np.nan, np.nan, "Untreated", np.nan],
        ["PTC41540", 32, "F", "HC", np.nan, np.nan, "Untreated", np.nan],
        ["PTC85037", 25, "F", "HC", np.nan, np.nan, "Untreated", np.nan],
        ["PTC32190", 33, "M", "HC", np.nan, np.nan, "Untreated", np.nan],
        ["MS45044", 25, "F", "RRMS", 1.0, np.nan, "Untreated", np.nan],
        ["MS60249", 28, "M", "RRMS", 0.0, np.nan, "Untreated", np.nan],
        ["MS74594", 42, "M", "RRMS", 0.0, np.nan, "Untreated", np.nan],
        ["MS19270", 35, "F", "RRMS", 0.5, np.nan, "Untreated", np.nan],
        ["MS71658", 47, "F", "RRMS", 6.0, np.nan, "Untreated", np.nan],
        ["MS49131", 47, "F", "RRMS", 2.5, np.nan, "Untreated", np.nan],
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "patient", "Age", "Sex", "Diagnosis", "EDSS", "Latest_FU",
            "Treatment", "months_from_onset"
        ],
    )
    df["patient"] = df["patient"].astype(str)
    df["diagnosis_group"] = df["Diagnosis"].map(DIAG_MAP)
    return df


def harmonise_patient_num(obs: pd.DataFrame) -> pd.DataFrame:
    obs = obs.copy()
    obs["patient"] = obs["patient"].astype(str)
    obs["dataset"] = obs["dataset"].astype(str)
    obs["patient_num"] = obs["patient"]

    mask028 = obs["dataset"] == "GSE133028"
    mask266 = obs["dataset"] == "GSE138266"

    obs.loc[mask028, "patient_num"] = (
        obs.loc[mask028, "patient"]
        .str.extract(r"(\d+)$", expand=False)
        .fillna(obs.loc[mask028, "patient"])
        .astype(str)
    )

    p266 = obs.loc[mask266, "patient"].str.split("_", n=1, expand=True)
    if p266.shape[1] == 1:
        coreid = p266[0].str.replace(r"PB|CSF", "", regex=True)
    else:
        tmp0 = p266[0].fillna("").str.replace(r"PB|CSF", "", regex=True)
        tmp1 = p266[1].fillna("").str.replace(r"PB|CSF", "", regex=True)
        coreid = tmp0.where(tmp0.ne(""), tmp1)
    obs.loc[mask266, "patient_num"] = coreid.astype(str)

    return obs


def check_patient_num_mapping(obs: pd.DataFrame, clinical: pd.DataFrame) -> None:
    """
    Sanity-check the regex-derived `patient_num` (see harmonise_patient_num)
    against the clinical metadata's patient IDs before trusting the merge.

    harmonise_patient_num extracts patient_num via dataset-specific string
    parsing (trailing digits for GSE133028; underscore-split + PB/CSF-strip
    for GSE138266). If that parsing doesn't actually match the clinical
    metadata's patient IDs, affected patients are silently dropped from the
    EDSS/diagnosis merge (an inner/left join just produces NaN rows) rather
    than raising an error -- so this needs to be checked explicitly, the
    same way PB/CSF subject-ID pairing was checked in fig3.py.
    """
    derived = set(obs["patient_num"].dropna().unique())
    clinical_ids = set(clinical["patient"].unique())
    unmatched_derived = sorted(derived - clinical_ids)
    unmatched_clinical = sorted(clinical_ids - derived)

    print(f"  [patient_num check] {len(derived)} derived patient_num values in expression data, "
          f"{len(clinical_ids)} patients in clinical metadata")
    if unmatched_derived:
        print(f"  [WARNING] {len(unmatched_derived)} derived patient_num value(s) have NO match in "
              f"clinical metadata -- these patients' cells will be silently dropped from EDSS/diagnosis "
              f"merges (EDSS, Diagnosis, diagnosis_group all become NaN for them): {unmatched_derived}")
    if unmatched_clinical:
        print(f"  [note] {len(unmatched_clinical)} clinical metadata patient(s) not found among derived "
              f"patient_num values (expected if excluded/treated or simply absent from this h5ad): "
              f"{unmatched_clinical}")
    if not unmatched_derived and not unmatched_clinical:
        print("  [patient_num check] Perfect match between derived patient_num and clinical metadata IDs.")

    sample_map = (
        obs[["dataset", "patient", "patient_num"]]
        .drop_duplicates()
        .sort_values(["dataset", "patient_num"])
    )
    print("  [patient_num check] derived mapping, by dataset (VERIFY this looks correct):")
    print(sample_map.to_string(index=False))


def get_expression_matrix(a: sc.AnnData, genes: list[str]) -> pd.DataFrame:
    present = [g for g in genes if g in a.var_names]
    missing = [g for g in genes if g not in a.var_names]
    if missing:
        warnings.warn(f"Missing genes skipped: {missing}")
    if not present:
        raise ValueError("None of the zinc genes are present in var_names.")
    X = a[:, present].X
    if hasattr(X, "toarray"):
        X = X.toarray()
    return pd.DataFrame(X, columns=present, index=a.obs_names)


def get_patient_gene_means_from_raw(
    sub: sc.AnnData,
    gene: str,
    tissues: list[str] | None = None,
) -> pd.DataFrame:
    if sub.raw is None:
        raise ValueError("AnnData.raw is None; log-normalised layer not available.")
    if gene not in sub.raw.var_names:
        raise ValueError(f"{gene} not found in sub.raw.var_names.")

    obs = sub.obs[["patient_num", "tissue"]].copy()
    if tissues is not None:
        obs = obs[obs["tissue"].isin(tissues)].copy()
        sub_use = sub[obs.index].copy()
    else:
        sub_use = sub

    X = sub_use.raw[:, [gene]].X
    if hasattr(X, "toarray"):
        X = X.toarray()
    df = pd.DataFrame(X, columns=[f"{gene}_raw"], index=sub_use.obs_names)
    df = obs.join(df)

    return (
        df.groupby(["patient_num", "tissue"])[f"{gene}_raw"]
        .mean()
        .reset_index()
    )


def _format_p(p):
    if pd.isna(p):
        return "NA"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def make_zinc_vs_edss_main_collapsed(
    out: pd.DataFrame,
    fig_dir,
    signature_col: str = "zinc_signature",
    signature_label: str = "signed",
    file_suffix: str = "",
    population_label: str = "CXCR3+",
    file_tag: str = "CXCR3pos",
) -> None:
    """
    signature_col/signature_label: which composite score to plot (signed is
    primary -- see main() -- unsigned is the supplementary comparison).

    population_label/file_tag: which cell population this was computed on
    ("CXCR3+" = primary, matching Figs 2-3; "All" = supplementary, the
    original whole-B-cell population). Purely a labeling parameter -- the
    caller is responsible for passing in `out` already restricted to the
    right population.

    Multiple testing: the 4 panels here (2 Spearman correlations, 2
    Mann-Whitney tests) are a single pre-specified family of primary-figure
    tests. Raw p-values for all 4 are computed first, BH-corrected together,
    and both raw and corrected p are shown on each panel -- rather than
    reporting 4 uncorrected p-values with no acknowledgement of the
    multiple-comparison exposure.
    """
    df = out.copy()
    df["EDSS"] = pd.to_numeric(df["EDSS"], errors="coerce")
    df = df[df[signature_col].notna()].copy()
    df["diag_collapsed"] = (
        df["diagnosis_group"]
        .replace({"MS": "Disease"})
        .infer_objects(copy=False)
    )

    # ---- Pass 1: compute all 4 raw p-values (no plotting yet) ----
    panel_stats = {}  # letter -> dict(rho, pval, kind)
    for tissue, letter in zip(["CSF", "PB"], ["A", "B"]):
        tdf = df[(df["tissue"] == tissue) & df["EDSS"].notna()]
        rho = pval = np.nan
        if len(tdf) >= 3 and tdf["EDSS"].nunique() >= 2:
            rho, pval = spearmanr(tdf["EDSS"], tdf[signature_col])
        panel_stats[letter] = {"rho": rho, "pval": pval, "kind": "spearman"}

    for tissue, letter in zip(["CSF", "PB"], ["C", "D"]):
        bdf = df[(df["tissue"] == tissue) & (df["diag_collapsed"].isin(["HC", "Disease"]))]
        hc = bdf[bdf["diag_collapsed"] == "HC"][signature_col].dropna()
        dis = bdf[bdf["diag_collapsed"] == "Disease"][signature_col].dropna()
        pval = np.nan
        if len(hc) >= 1 and len(dis) >= 1:
            _, pval = mannwhitneyu(hc, dis, alternative="two-sided")
        panel_stats[letter] = {"rho": np.nan, "pval": pval, "kind": "mannwhitney"}

    raw_pvals = [panel_stats[l]["pval"] for l in ["A", "B", "C", "D"]]
    valid_mask = np.isfinite(raw_pvals)
    padj = np.full(4, np.nan)
    if valid_mask.sum() > 0:
        padj[valid_mask] = multipletests(np.array(raw_pvals)[valid_mask], method="fdr_bh")[1]
    for i, letter in enumerate(["A", "B", "C", "D"]):
        panel_stats[letter]["padj"] = padj[i]

    print(f"  [{signature_label}] Fig4 main-panel p-values (BH-corrected across all 4 panels):")
    for letter in ["A", "B", "C", "D"]:
        s = panel_stats[letter]
        print(f"    Panel {letter}: raw p={_format_p(s['pval'])}, padj={_format_p(s['padj'])}")

    # ---- Pass 2: plot, using the precomputed/corrected stats ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey="row")
    fig.subplots_adjust(wspace=0.35, hspace=0.45)

    for ax, tissue, letter in zip(axes[0], ["CSF", "PB"], ["A", "B"]):
        tdf = df[(df["tissue"] == tissue) & df["EDSS"].notna()].copy()

        for grp, gdf in tdf.groupby("diagnosis_group"):
            ax.scatter(
                gdf["EDSS"], gdf[signature_col], s=90,
                label=grp, color=PAL3.get(grp, "grey"), alpha=0.9,
                edgecolors="white", linewidths=1.2,
            )
            for _, r in gdf.iterrows():
                ax.text(
                    r["EDSS"], r[signature_col], str(r["patient_num"]),
                    fontsize=8, ha="left", va="bottom"
                )

        rho, pval, adj = panel_stats[letter]["rho"], panel_stats[letter]["pval"], panel_stats[letter]["padj"]
        if np.isfinite(rho):
            z = np.polyfit(tdf["EDSS"], tdf[signature_col], 1)
            xs = np.linspace(tdf["EDSS"].min(), tdf["EDSS"].max(), 100)
            ax.plot(xs, np.poly1d(z)(xs), linestyle="--", color="black", alpha=0.6)
            ax.set_title(
                f"{letter}. {tissue} {population_label} B cells ({signature_label})\n"
                f"ρ={rho:.2f}, p={_format_p(pval)}, padj={_format_p(adj)}",
                fontsize=12, weight="bold"
            )
        else:
            ax.set_title(f"{letter}. {tissue} {population_label} B cells ({signature_label})", fontsize=12, weight="bold")
        ax.set_xlabel("EDSS", fontsize=11)

    axes[0, 0].set_ylabel(f"Mean zinc signature ({signature_label})", fontsize=11)

    for ax, tissue, letter in zip(axes[1], ["CSF", "PB"], ["C", "D"]):
        bdf = df[
            (df["tissue"] == tissue) &
            (df["diag_collapsed"].isin(["HC", "Disease"]))
        ].copy()

        sns.boxplot(
            data=bdf, x="diag_collapsed", y=signature_col,
            order=["HC", "Disease"], hue="diag_collapsed",
            palette=PAL2, ax=ax, fliersize=0, legend=False,
        )
        sns.stripplot(
            data=bdf, x="diag_collapsed", y=signature_col,
            order=["HC", "Disease"], hue="diag_collapsed",
            palette=PAL2, dodge=False, ax=ax,
            linewidth=0.6, edgecolor="black", size=6, alpha=0.9, legend=False,
        )

        pval, adj = panel_stats[letter]["pval"], panel_stats[letter]["padj"]

        ax.set_title(f"{letter}. {tissue} {population_label} B cells ({signature_label})", fontsize=12, weight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(f"Mean zinc signature ({signature_label})", fontsize=11)

        # Dynamically position the p/padj annotation above the actual data
        # range in THIS panel, instead of a fixed axes-fraction position
        # (0.5, 0.95) -- the fixed position could land on top of data points
        # whenever the "Disease" group's spread reaches close to the top of
        # the axes, which is exactly what happened in the first version of
        # this figure. Also draws a bracket connecting HC/Disease, matching
        # the established style in plot_figure1_umap_bar.py's _sig_bracket.
        y_data_max = bdf[signature_col].max()
        y_data_min = bdf[signature_col].min()
        data_range = y_data_max - y_data_min if y_data_max > y_data_min else 1.0
        bracket_y = y_data_max + data_range * 0.08
        bracket_h = data_range * 0.03

        ax.plot([0, 0, 1, 1], [bracket_y, bracket_y + bracket_h, bracket_y + bracket_h, bracket_y],
                lw=1.3, color="black", clip_on=False)
        ax.text(
            0.5, bracket_y + bracket_h * 1.3,
            f"HC vs MS: p={_format_p(pval)}, padj={_format_p(adj)}",
            ha="center", va="bottom", fontsize=9, clip_on=False,
        )
        # Give the bracket/text room to breathe above the tallest data point
        ax.set_ylim(top=bracket_y + data_range * 0.20)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if len(handles) > 1:
        axes[0, 0].legend(handles, labels, frameon=False, title="Diagnosis")

    fig.suptitle(
        f"Figure 4. Zinc signature ({signature_label}) vs EDSS and disease status\nin PB and CSF {population_label} B cells",
        fontsize=16, weight="bold", y=0.98,
    )

    png = fig_dir / f"figure4_{file_tag}_zinc_signature_EDSS_main_collapsed_pb_csf{file_suffix}.png"
    pdf_fig = fig_dir / f"figure4_{file_tag}_zinc_signature_EDSS_main_collapsed_pb_csf{file_suffix}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.2)
    fig.savefig(pdf_fig, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)

    print(f"Saved main figure: {png}")
    print(f"Saved main figure: {pdf_fig}")


def make_zinc_boxplots_hc_ms_supp(
    out: pd.DataFrame,
    fig_dir,
    signature_col: str = "zinc_signature",
    signature_label: str = "signed",
    file_suffix: str = "",
    population_label: str = "CXCR3+",
    file_tag: str = "CXCR3pos",
) -> None:
    df = out.copy()
    df = df[df[signature_col].notna()].copy()
    box_df = df[df["diagnosis_group"].isin(["HC", "MS"])].copy()

    # Compute both tissues' KW p-values first so they can be BH-corrected
    # together (2 tests, same family: HC/MS comparison per tissue).
    kw_pvals = {}
    for tissue in ["CSF", "PB"]:
        bdf = box_df[box_df["tissue"] == tissue]
        groups = [bdf[bdf["diagnosis_group"] == g][signature_col].dropna() for g in ["HC", "MS"]]
        kw_p = np.nan
        if all(len(g) > 0 for g in groups):
            _, kw_p = kruskal(*groups)
        kw_pvals[tissue] = kw_p

    raw = np.array([kw_pvals["CSF"], kw_pvals["PB"]])
    valid = np.isfinite(raw)
    kw_padj = {"CSF": np.nan, "PB": np.nan}
    if valid.sum() > 0:
        corrected = multipletests(raw[valid], method="fdr_bh")[1]
        for tissue, p in zip(np.array(["CSF", "PB"])[valid], corrected):
            kw_padj[tissue] = p

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.subplots_adjust(wspace=0.3)

    for ax, tissue in zip(axes, ["CSF", "PB"]):
        bdf = box_df[box_df["tissue"] == tissue].copy()

        sns.boxplot(
            data=bdf, x="diagnosis_group", y=signature_col,
            order=["HC", "MS"], hue="diagnosis_group",
            palette=PAL3, ax=ax, fliersize=0, legend=False,
        )
        sns.stripplot(
            data=bdf, x="diagnosis_group", y=signature_col,
            order=["HC", "MS"], hue="diagnosis_group",
            palette=PAL3, dodge=False, ax=ax,
            linewidth=0.6, edgecolor="black", size=6, alpha=0.9, legend=False,
        )

        ax.set_title(
            f"{tissue} {population_label} B cells ({signature_label})\n"
            f"KW p={_format_p(kw_pvals[tissue])}, padj={_format_p(kw_padj[tissue])}",
            fontsize=12, weight="bold"
        )
        ax.set_xlabel("")
        ax.set_ylabel(f"Mean zinc signature ({signature_label})", fontsize=11)

    fig.suptitle(
        f"Supplementary: zinc signature ({signature_label}) by HC/MS diagnosis group ({population_label} B cells)",
        fontsize=15, weight="bold", y=1.02
    )

    png = fig_dir / f"supp_zinc_signature_boxplots_HC_MS_{file_tag}_pb_csf{file_suffix}.png"
    pdf_fig = fig_dir / f"supp_zinc_signature_boxplots_HC_MS_{file_tag}_pb_csf{file_suffix}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.2)
    fig.savefig(pdf_fig, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)

    print(f"Saved supplementary boxplots: {png}")
    print(f"Saved supplementary boxplots: {pdf_fig}")


def build_gene_edss_table(
    sub: sc.AnnData,
    gene: str,
    tissue: str,
    meta: pd.DataFrame,
) -> pd.DataFrame:
    try:
        gene_means = get_patient_gene_means_from_raw(
            sub=sub,
            gene=gene,
            tissues=[tissue],
        )
    except ValueError:
        return pd.DataFrame()

    gene_col = f"{gene}_raw"
    df = meta[meta["tissue"] == tissue].copy()
    df = df.merge(gene_means, on=["patient_num", "tissue"], how="inner")

    if gene_col not in df.columns:
        return pd.DataFrame()

    df["EDSS"] = pd.to_numeric(df["EDSS"], errors="coerce")
    df = df[df["EDSS"].notna() & df[gene_col].notna()].copy()
    return df


def compute_all_gene_tissue_correlations(
    sub: sc.AnnData,
    genes: list[str],
    meta: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute Spearman gene-vs-EDSS correlations for every (gene, tissue)
    combination up front, as ONE family of tests, so they can be
    BH-corrected together. Without this, the per-gene supplementary PDFs
    (n_genes x 2 tissues = ~26 tests here) would each report an uncorrected
    p-value with no acknowledgement of the multiple-comparison exposure
    across the full panel.
    """
    rows = []
    for gene in genes:
        for tissue in ["CSF", "PB"]:
            gdf = build_gene_edss_table(sub, gene, tissue, meta)
            rho = pval = np.nan
            n = len(gdf)
            if n >= 3 and gdf["EDSS"].nunique() >= 2:
                rho, pval = spearmanr(gdf["EDSS"], gdf[f"{gene}_raw"])
            rows.append({"gene": gene, "tissue": tissue, "n": n, "rho": rho, "pvalue": pval})

    result = pd.DataFrame(rows)
    result["padj"] = np.nan
    valid = result["pvalue"].notna()
    if valid.sum() > 0:
        result.loc[valid, "padj"] = multipletests(result.loc[valid, "pvalue"], method="fdr_bh")[1]
    return result


def leave_one_out_correlation(gdf: pd.DataFrame, gene_col: str) -> pd.DataFrame:
    """
    Leave-one-out sensitivity check for a Spearman correlation: for each
    patient, recompute rho/p with that ONE patient excluded, to check
    whether a significant correlation depends heavily on a single
    (possibly high-leverage) point -- e.g. a patient sitting alone at an
    extreme EDSS value.

    This checks robustness of the RAW correlation only; it does not
    recompute the family-wise BH correction for each dropped-patient
    subset (that would require rerunning the entire gene x tissue panel
    once per dropped patient, which changes the correction for every OTHER
    gene too -- a much heavier and less standard check). Use this
    alongside, not instead of, the panel-corrected padj already reported by
    compute_all_gene_tissue_correlations.
    """
    rows = []
    for idx in gdf.index:
        dropped_id = gdf.loc[idx, "patient_num"]
        remaining = gdf.drop(index=idx)
        rho, p = np.nan, np.nan
        if len(remaining) >= 3 and remaining["EDSS"].nunique() >= 2:
            rho, p = spearmanr(remaining["EDSS"], remaining[gene_col])
        rows.append({"dropped_patient": dropped_id, "n_remaining": len(remaining), "rho": rho, "pvalue": p})
    return pd.DataFrame(rows)


def summarize_loo_robustness(
    loo_df: pd.DataFrame, orig_rho: float, orig_pvalue: float, alpha: float = 0.05
) -> dict:
    """
    Summarize a leave-one-out table: does the significance call (raw p vs
    alpha) flip when any single patient is dropped? Reports which patient(s)
    would need to be excluded to flip the result, and the range of rho/p
    across all leave-one-out subsets, so a single influential point is
    visible rather than hidden inside a single reported correlation.
    """
    valid = loo_df.dropna(subset=["rho", "pvalue"])
    if valid.empty:
        return {
            "orig_rho": orig_rho, "orig_pvalue": orig_pvalue,
            "n_loo": 0, "robust": None,
            "rho_min": np.nan, "rho_max": np.nan, "p_min": np.nan, "p_max": np.nan,
            "flips_on_removing": [],
        }

    orig_sig = orig_pvalue < alpha
    flips = valid[(valid["pvalue"] < alpha) != orig_sig]

    return {
        "orig_rho": orig_rho, "orig_pvalue": orig_pvalue,
        "n_loo": len(valid),
        "rho_min": valid["rho"].min(), "rho_max": valid["rho"].max(),
        "p_min": valid["pvalue"].min(), "p_max": valid["pvalue"].max(),
        "n_flips": len(flips),
        "flips_on_removing": flips["dropped_patient"].tolist(),
        "robust": len(flips) == 0,
    }


def run_loo_check_for_significant_genes(
    sub: sc.AnnData,
    corr_table: pd.DataFrame,
    meta: pd.DataFrame,
    fig_dir,
    file_tag: str,
    padj_threshold: float = 0.05,
) -> pd.DataFrame:
    """
    For every (gene, tissue) that clears the panel-wide BH correction
    (padj < padj_threshold) in `corr_table`, run the leave-one-out
    robustness check and save a summary table + per-patient detail table.
    Prints a plain-language flag for any gene/tissue whose significance
    depends on a single patient.
    """
    sig_rows = corr_table[corr_table["padj"] < padj_threshold]
    if sig_rows.empty:
        print(f"  [LOO check] No gene x tissue correlations cleared padj < {padj_threshold} "
              f"-- nothing to robustness-check.")
        return pd.DataFrame()

    summary_rows = []
    detail_rows = []
    for _, row in sig_rows.iterrows():
        gene, tissue = row["gene"], row["tissue"]
        gdf = build_gene_edss_table(sub, gene, tissue, meta)
        gene_col = f"{gene}_raw"
        loo_df = leave_one_out_correlation(gdf, gene_col)
        loo_df["gene"] = gene
        loo_df["tissue"] = tissue
        detail_rows.append(loo_df)

        summary = summarize_loo_robustness(loo_df, orig_rho=row["rho"], orig_pvalue=row["pvalue"])
        summary["gene"] = gene
        summary["tissue"] = tissue
        summary["orig_padj"] = row["padj"]
        summary_rows.append(summary)

        if summary["n_loo"] == 0:
            print(f"  [LOO check] {gene} ({tissue}): CANNOT ASSESS -- every leave-one-out subset "
                  f"dropped below the minimum n needed for a Spearman test (n={row['n']} to start with). "
                  f"Treat the original padj={row['padj']:.3f} result as very low-powered regardless.")
        elif summary["robust"]:
            print(f"  [LOO check] {gene} ({tissue}): ROBUST -- significance does not depend on "
                  f"any single patient (rho range [{summary['rho_min']:.2f}, {summary['rho_max']:.2f}] "
                  f"across {summary['n_loo']} leave-one-out subsets).")
        else:
            print(f"  [LOO check] {gene} ({tissue}): NOT ROBUST -- removing patient(s) "
                  f"{summary['flips_on_removing']} flips significance (p vs {padj_threshold} threshold). "
                  f"rho range [{summary['rho_min']:.2f}, {summary['rho_max']:.2f}], "
                  f"p range [{summary['p_min']:.3f}, {summary['p_max']:.3f}]. "
                  f"Treat the original padj={row['padj']:.3f} result as fragile, not confirmatory.")

    summary_df = pd.DataFrame(summary_rows)
    detail_df = pd.concat(detail_rows, ignore_index=True)

    summary_path = fig_dir / f"supp_LOO_robustness_summary_significant_genes_{file_tag}.csv"
    detail_path = fig_dir / f"supp_LOO_robustness_detail_significant_genes_{file_tag}.csv"
    summary_df.to_csv(summary_path, index=False)
    detail_df.to_csv(detail_path, index=False)
    print(f"  Saved LOO robustness summary: {summary_path}")
    print(f"  Saved LOO robustness detail: {detail_path}")

    return summary_df


def _scatter_one_panel(
    ax, df: pd.DataFrame, gene_col: str, tissue_label: str,
    rho: float | None = None, pval: float | None = None, padj: float | None = None,
) -> None:
    """
    If rho/pval/padj are provided (precomputed via
    compute_all_gene_tissue_correlations, so they reflect BH correction
    across the full gene x tissue panel), those are used for the title and
    fit line. Otherwise falls back to computing them locally (uncorrected --
    only used if this is called outside the corrected-panel workflow).
    """
    gene_name = gene_col.replace("_raw", "")

    for grp, gdf in df.groupby("diagnosis_group"):
        ax.scatter(
            gdf["EDSS"], gdf[gene_col], s=90,
            label=grp, color=PAL3.get(grp, "grey"), alpha=0.9,
            edgecolors="white", linewidths=1.2,
        )
        for _, r in gdf.iterrows():
            ax.text(
                r["EDSS"], r[gene_col], str(r["patient_num"]),
                fontsize=8, ha="left", va="bottom"
            )

    if rho is None or pval is None:
        rho = pval = np.nan
        if len(df) >= 3 and df["EDSS"].nunique() >= 2:
            rho, pval = spearmanr(df["EDSS"], df[gene_col])

    if np.isfinite(rho) if rho is not None else False:
        z = np.polyfit(df["EDSS"], df[gene_col], 1)
        xs = np.linspace(df["EDSS"].min(), df["EDSS"].max(), 100)
        ax.plot(xs, np.poly1d(z)(xs), linestyle="--", color="black", alpha=0.6)

    title = f"{tissue_label} {gene_name}"
    if rho is not None and np.isfinite(rho):
        title += f" (ρ={rho:.2f}, p={_format_p(pval)}"
        if padj is not None:
            title += f", padj={_format_p(padj)}"
        title += ")"
    title += f"\nN={len(df)}"

    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_xlabel("EDSS", fontsize=11)
    ax.set_ylabel(f"{tissue_label} {gene_name} (mean log-normalised expr)", fontsize=11)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False, title="Diagnosis")


def make_gene_two_panel_pdf(
    sub: sc.AnnData,
    gene: str,
    meta: pd.DataFrame,
    fig_dir,
    corr_table: pd.DataFrame | None = None,
    population_label: str = "CXCR3+",
    file_tag: str = "CXCR3pos",
) -> None:
    csf_df = build_gene_edss_table(sub, gene, "CSF", meta)
    pb_df = build_gene_edss_table(sub, gene, "PB", meta)

    if csf_df.empty and pb_df.empty:
        print(f"Skipped {gene}: no EDSS-linked CSF or PB data")
        return

    def _lookup(tissue: str) -> dict:
        if corr_table is None:
            return {"rho": None, "pval": None, "padj": None}
        row = corr_table[(corr_table["gene"] == gene) & (corr_table["tissue"] == tissue)]
        if row.empty:
            return {"rho": None, "pval": None, "padj": None}
        r = row.iloc[0]
        return {"rho": r["rho"], "pval": r["pvalue"], "padj": r["padj"]}

    gene_col = f"{gene}_raw"
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.subplots_adjust(wspace=0.35)

    if not csf_df.empty:
        s = _lookup("CSF")
        _scatter_one_panel(axes[0], csf_df, gene_col, "CSF", rho=s["rho"], pval=s["pval"], padj=s["padj"])
    else:
        axes[0].set_title(f"CSF {gene}\nNo EDSS-linked data", fontsize=12, weight="bold")
        axes[0].set_xlabel("EDSS", fontsize=11)
        axes[0].set_ylabel(f"CSF {gene} (mean log-normalised expr)", fontsize=11)

    if not pb_df.empty:
        s = _lookup("PB")
        _scatter_one_panel(axes[1], pb_df, gene_col, "PB", rho=s["rho"], pval=s["pval"], padj=s["padj"])
    else:
        axes[1].set_title(f"PB {gene}\nNo EDSS-linked data", fontsize=12, weight="bold")
        axes[1].set_xlabel("EDSS", fontsize=11)
        axes[1].set_ylabel(f"PB {gene} (mean log-normalised expr)", fontsize=11)

    fig.suptitle(f"{gene} vs EDSS in CSF and PB ({population_label} B cells)", fontsize=15, weight="bold", y=1.02)

    pdf_path = fig_dir / f"supp_{gene}_CSF_PB_vs_EDSS_{file_tag}.pdf"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)

    print(f"Saved gene PDF: {pdf_path}")


def run_fig4_for_population(
    adata: sc.AnnData,
    obs: pd.DataFrame,
    keep_mask: pd.Series,
    cfg: dict,
    genes: list[str],
    fig_dir,
    population_label: str,
    file_tag: str,
) -> None:
    """
    Run the full Figure 4 pipeline (patient-level aggregation, signed/
    unsigned composite signature, main figure, HC/MS boxplots, per-gene
    EDSS correlation table + PDFs) for ONE cell population, defined by
    `keep_mask`. Called twice from main(): once for CXCR3+ B cells (primary,
    matching Figs 2-3), once for all B cells regardless of CXCR3 status
    (supplementary, the population this figure originally used).

    All outputs are tagged with `file_tag` so the two populations' files
    never collide, and all titles use `population_label` so it's always
    visually obvious which population a given figure represents.
    """
    n_cells = int(keep_mask.sum())
    n_patients = obs.loc[keep_mask, "patient_num"].nunique()
    print(f"\n=== Population: {population_label} B cells ({file_tag}) -- "
          f"{n_cells:,} cells, {n_patients} patients ===")

    sub = adata[keep_mask].copy()
    sub.obs = obs.loc[sub.obs_names].copy()

    expr = get_expression_matrix(sub, genes)
    df = sub.obs[["dataset", "patient_num", "tissue", "condition", "cxcr3_group"]].copy()
    df = df.join(expr)

    patient_means = (
        df.groupby(["dataset", "patient_num", "tissue"])[genes]
        .mean()
        .reset_index()
    )

    # Sign-corrected composite signature is primary (consistent with Fig 2,
    # which established this for the same kind of disease-severity axis:
    # +1 for transporters, -1 for metallothioneins, assigned a priori from
    # biological role -- see compute_gene_signs in stats_utils.py). Plain
    # unsigned mean kept alongside for the supplementary comparison figure.
    transporters, mts = load_marker_categories(cfg)
    gene_signs = compute_gene_signs(transporters, mts)
    sign_vec = np.array([gene_signs.get(g, 1.0) for g in genes])
    print(f"Gene signs for composite signature: {gene_signs}")

    patient_means["zinc_signature_unsigned"] = patient_means[genes].mean(axis=1)
    patient_means["zinc_signature_signed"] = (
        patient_means[genes].values * sign_vec[np.newaxis, :]
    ).mean(axis=1)
    patient_means["zinc_signature"] = patient_means["zinc_signature_signed"]  # primary, for backward compat

    clinical = load_clinical_metadata()
    check_patient_num_mapping(obs.loc[keep_mask], clinical)

    n_cells_tbl = (
        df.groupby(["patient_num", "tissue"])
        .size()
        .rename("n_cells")
        .reset_index()
    )

    out = patient_means.merge(n_cells_tbl, on=["patient_num", "tissue"], how="left")
    out = out.merge(
        clinical[["patient", "Diagnosis", "diagnosis_group", "EDSS", "months_from_onset"]],
        left_on="patient_num",
        right_on="patient",
        how="left",
    )

    out["EDSS"] = pd.to_numeric(out["EDSS"], errors="coerce")
    out["months_from_onset"] = pd.to_numeric(out["months_from_onset"], errors="coerce")
    out = out.sort_values(["tissue", "diagnosis_group", "dataset", "patient_num"])

    # Output folder structure: fig_dir/<file_tag>/main/ and .../supp/ --
    # separates the two populations (CXCR3pos vs AllBcells) AND separates
    # the main figure from supplementary outputs within each, so neither
    # dimension has to be inferred from filename prefixes alone.
    population_dir = Path(fig_dir) / file_tag
    main_dir = population_dir / "main"
    supp_dir = population_dir / "supp"
    main_dir.mkdir(parents=True, exist_ok=True)
    supp_dir.mkdir(parents=True, exist_ok=True)

    summary_path = main_dir / f"patient_level_zinc_edss_{file_tag}_pb_csf.csv"
    out.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")

    # Panel-specific CSVs for Fig 4 (main figure panel data -> main_dir)
    df_4A = out[
        (out["tissue"] == "CSF")
        & out["diagnosis_group"].isin(["MS"])
        & out["zinc_signature"].notna()
        & out["EDSS"].notna()
    ][["patient_num", "diagnosis_group", "tissue", "EDSS", "zinc_signature", "n_cells"]]
    df_4A.to_csv(main_dir / f"Fig4A_CSF_zinc_signature_vs_EDSS_patient_level_{file_tag}.csv",
                 index=False)

    df_4B = out[
        (out["tissue"] == "PB")
        & out["diagnosis_group"].isin(["MS"])
        & out["zinc_signature"].notna()
        & out["EDSS"].notna()
    ][["patient_num", "diagnosis_group", "tissue", "EDSS", "zinc_signature", "n_cells"]]
    df_4B.to_csv(main_dir / f"Fig4B_PB_zinc_signature_vs_EDSS_patient_level_{file_tag}.csv",
                 index=False)

    df_4C = out[
        (out["tissue"] == "CSF")
        & out["diagnosis_group"].isin(["HC", "MS"])
        & out["zinc_signature"].notna()
    ][["patient_num", "diagnosis_group", "tissue", "zinc_signature", "n_cells"]]
    df_4C.to_csv(main_dir / f"Fig4C_CSF_HC_vs_MS_zinc_signature_patient_level_{file_tag}.csv",
                 index=False)

    df_4D = out[
        (out["tissue"] == "PB")
        & out["diagnosis_group"].isin(["HC", "MS"])
        & out["zinc_signature"].notna()
    ][["patient_num", "diagnosis_group", "tissue", "zinc_signature", "n_cells"]]
    df_4D.to_csv(main_dir / f"Fig4D_PB_HC_vs_MS_zinc_signature_patient_level_{file_tag}.csv",
                 index=False)

    # Supplementary boxplot's underlying data (filename already says "FigSx") -> supp_dir
    box_df = out[
        out["diagnosis_group"].isin(["HC", "MS"])
        & out["zinc_signature"].notna()
    ][["patient_num", "diagnosis_group", "tissue", "zinc_signature", "n_cells"]]
    box_df.to_csv(supp_dir / f"FigSx_HC_MS_zinc_signature_patient_level_{file_tag}.csv",
                  index=False)

    # Main figure: sign-corrected composite (primary) -> main_dir
    make_zinc_vs_edss_main_collapsed(
        out, main_dir, signature_col="zinc_signature", signature_label="signed",
        file_suffix="", population_label=population_label, file_tag=file_tag,
    )
    # Boxplot is conceptually supplementary regardless of signed/unsigned -> supp_dir
    make_zinc_boxplots_hc_ms_supp(
        out, supp_dir, signature_col="zinc_signature", signature_label="signed",
        file_suffix="", population_label=population_label, file_tag=file_tag,
    )

    # Supplementary: unsigned composite, for direct comparison against the
    # signed version above (same rationale as Figs 2-3) -> supp_dir, even
    # though it's generated by the same function as the main figure above,
    # since this specific (unsigned) version is the supplementary one.
    make_zinc_vs_edss_main_collapsed(
        out, supp_dir, signature_col="zinc_signature_unsigned", signature_label="unsigned",
        file_suffix="_UNSIGNED", population_label=population_label, file_tag=file_tag,
    )
    make_zinc_boxplots_hc_ms_supp(
        out, supp_dir, signature_col="zinc_signature_unsigned", signature_label="unsigned",
        file_suffix="_UNSIGNED", population_label=population_label, file_tag=file_tag,
    )

    meta = out[["patient_num", "diagnosis_group", "EDSS", "tissue"]].drop_duplicates()

    # Per-gene EDSS correlations: compute ALL (gene x tissue) tests as one
    # family FIRST, BH-correct together, save as a supplementary table, and
    # only then render the per-gene PDFs using the corrected values.
    corr_table = compute_all_gene_tissue_correlations(sub, genes, meta)
    corr_table_path = supp_dir / f"supp_gene_EDSS_correlations_all_genes_BHcorrected_{file_tag}.csv"
    corr_table.to_csv(corr_table_path, index=False)
    print(f"Saved per-gene EDSS correlation table (BH-corrected across "
          f"{len(corr_table)} gene x tissue tests): {corr_table_path}")
    n_sig = int((corr_table["padj"] < 0.05).sum())
    print(f"  {n_sig}/{len(corr_table)} gene x tissue correlations significant after correction (padj < 0.05)")

    # Leave-one-out robustness check for anything that cleared correction --
    # a corrected p-value can still be driven by a single high-leverage
    # patient (e.g. one person alone at an extreme EDSS value); this checks
    # for that directly rather than reporting padj at face value.
    run_loo_check_for_significant_genes(sub, corr_table, meta, supp_dir, file_tag=file_tag)

    for gene in genes:
        make_gene_two_panel_pdf(
            sub=sub, gene=gene, meta=meta, fig_dir=supp_dir, corr_table=corr_table,
            population_label=population_label, file_tag=file_tag,
        )


def main() -> None:
    cfg = load_cfg(DATASETS_YAML)
    genes = load_markers(cfg)

    fig_dir = FIG_DIRS["fig4"]
    fig_dir.mkdir(parents=True, exist_ok=True)

    adata_path = RESULTS_DIR / "merged_bcells.h5ad"
    adata = sc.read_h5ad(adata_path)

    obs = adata.obs.copy()
    obs["dataset"] = obs["dataset"].astype(str)
    obs["tissue"] = obs["tissue"].astype(str)
    obs["condition"] = obs["condition"].astype(str)
    obs["cxcr3_group"] = obs["cxcr3_group"].astype(str)
    obs = harmonise_patient_num(obs)

    keep_all_bcells = (
        obs["dataset"].isin(TARGET_DATASETS)
        & obs["tissue"].isin(["PB", "CSF"])
        & obs["patient_num"].notna()
        & (~obs["patient_num"].isin(EXCLUDE_TREATED))
    )
    keep_cxcr3pos = keep_all_bcells & (obs["cxcr3_group"] == "CXCR3+")

    n_cells_all = int(keep_all_bcells.sum())
    n_cells_cxcr3 = int(keep_cxcr3pos.sum())
    n_patients_all = obs.loc[keep_all_bcells, "patient_num"].nunique()
    n_patients_cxcr3 = obs.loc[keep_cxcr3pos, "patient_num"].nunique()
    print(f"CXCR3+ restriction: {n_cells_cxcr3:,}/{n_cells_all:,} cells retained "
          f"({100*n_cells_cxcr3/max(n_cells_all,1):.1f}%); "
          f"{n_patients_cxcr3}/{n_patients_all} patients retained "
          f"(a patient can be lost entirely if they have zero CXCR3+ B cells "
          f"in a given tissue -- check EDSS/HC-vs-disease panels for reduced n).")

    # PRIMARY: CXCR3+ B cells, matching Figures 2-3's population.
    run_fig4_for_population(
        adata, obs, keep_cxcr3pos, cfg, genes, fig_dir,
        population_label="CXCR3+", file_tag="CXCR3pos",
    )

    # SUPPLEMENTARY: all B cells regardless of CXCR3 status -- the
    # population this figure originally used, kept for direct comparison
    # against the CXCR3+-restricted primary result above.
    run_fig4_for_population(
        adata, obs, keep_all_bcells, cfg, genes, fig_dir,
        population_label="All", file_tag="AllBcells",
    )


if __name__ == "__main__":
    main()