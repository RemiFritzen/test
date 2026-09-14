#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
fig3_pb_vs_csf_cxcr3pos.py — PB vs CSF comparison in CXCR3+ B cells (Figure 3)

Main output (4-panel, 2×2):
  A  Barplot: CIS+MS zinc/MT genes (CSF vs PB)
  B  Barplot: HC zinc/MT genes (CSF vs PB)
  C  Violin: HC CXCR3+ PB vs CSF zinc signature
  D  Violin: CIS+MS CXCR3+ PB vs CSF zinc signature

Supplementary outputs (saved in fig3/supp/):
  Supp_1  Volcano: CXCR3+ B cells – CSF vs PB (CIS+MS)
  Supp_2  Volcano: CXCR3+ B cells – CSF vs PB (HC)
"""

import scipy.sparse as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
from scipy.stats import mannwhitneyu

from config.config import RESULTS_DIR, FIG_DIRS, DATASETS_YAML
from data_io import load_cfg


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
PANEL_KW               = dict(fontsize=14, fontweight="bold", va="top", ha="left")

GROUPING_MODE = "demyelinating"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _panel_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(-0.12, 1.05, letter, transform=ax.transAxes, **PANEL_KW)


def load_zinc_gene_list() -> list[str]:
    cfg = load_cfg(DATASETS_YAML)
    markers = cfg.get("markers", {})
    artefact = set(markers.get("exclude_artefacts", []))
    all_g = markers.get("zinc_transporters", []) + markers.get("metallothioneins", [])
    return [g for g in all_g if g not in artefact]


def _apply_min_pct_filter_tissue(
    sub: sc.AnnData,
    min_pct: float = 0.10,
) -> sc.AnnData:
    """
    Keep genes expressed in at least one tissue above min_pct.
    """
    X = sub.X if not sp.issparse(sub.X) else sub.X.toarray()
    pb = sub.obs["tissue"] == "PB"
    csf = sub.obs["tissue"] == "CSF"

    pct_pb = (X[pb] > 0).mean(axis=0)
    pct_csf = (X[csf] > 0).mean(axis=0)
    keep = (pct_pb >= min_pct) | (pct_csf >= min_pct)

    print(f"    min_pct={min_pct:.0%}: {keep.sum()}/{len(keep)} genes retained")
    return sub[:, keep].copy()


def compute_cell_signature_pb_vs_csf(
    adata: sc.AnnData,
    zinc_genes: list[str],
    grouping: str,
) -> pd.DataFrame:
    """
    Return per-cell zinc signature in CXCR3+ B cells with columns:
        cell_id, tissue, condition, group_label, zinc_signature
    """
    sub = adata[adata.obs["cxcr3_group"] == "CXCR3+"].copy()

    if grouping == "demyelinating":
        sub = sub[sub.obs["condition"].isin(["HC", "CIS", "MS"])].copy()
        sub.obs["group_label"] = sub.obs["condition"].replace({
            "CIS": "CIS+MS",
            "MS":  "CIS+MS",
        })
    elif grouping == "MS_only":
        sub = sub[sub.obs["condition"].isin(["HC", "MS"])].copy()
        sub.obs["group_label"] = sub.obs["condition"]
    elif grouping == "all":
        sub = sub[sub.obs["condition"].isin(["HC", "CIS", "MS"])].copy()
        sub.obs["group_label"] = sub.obs["condition"]
    else:
        raise ValueError(f"Unknown GROUPING_MODE: {grouping}")

    genes = [g for g in zinc_genes if g in sub.var_names]
    X = sub[:, genes].X
    if sp.issparse(X):
        X = X.toarray()

    sig = X.mean(axis=1)

    df = sub.obs[["tissue", "condition", "group_label"]].copy()
    df["cell_id"] = sub.obs_names
    df["zinc_signature"] = sig
    df = df[df["tissue"].isin(["PB", "CSF"])].copy()
    return df



def _run_de_pb_vs_csf(
    adata: sc.AnnData,
    zinc_genes: list[str],
    grouping: str,
    min_pct: float = 0.10,
) -> pd.DataFrame:
    """
    DE for PB vs CSF in CXCR3+ B cells within a chosen grouping.
    Returns CSF vs PB, so negative log2FC means lower in CSF.
    """
    print(f"\nRunning DE: PB vs CSF CXCR3+, grouping='{grouping}' …")

    sub = adata[adata.obs["cxcr3_group"] == "CXCR3+"].copy()
    if grouping == "demyelinating":
        sub = sub[sub.obs["condition"].isin(["CIS", "MS"])].copy()
    elif grouping == "MS_only":
        sub = sub[sub.obs["condition"] == "MS"].copy()
    elif grouping == "all":
        sub = sub[sub.obs["condition"].isin(["HC", "CIS", "MS"])].copy()
    else:
        raise ValueError(f"Unknown GROUPING_MODE: {grouping}")

    sub = sub[sub.obs["tissue"].isin(["PB", "CSF"])].copy()
    if not {"PB", "CSF"}.issubset(set(sub.obs["tissue"].unique())):
        raise ValueError("CXCR3+ PB and CSF cells not both present in selected grouping.")

    sub = _apply_min_pct_filter_tissue(sub, min_pct=min_pct)

    key = f"de_PB_vs_CSF_CXCR3pos_{grouping}"
    sc.tl.rank_genes_groups(
        sub,
        groupby="tissue",
        groups=["CSF"],
        reference="PB",
        method="wilcoxon",
        key_added=key,
    )
    df = sc.get.rank_genes_groups_df(sub, group="CSF", key=key)
    df = df.rename(columns={
        "names": "gene",
        "logfoldchanges": "log2FC",
        "pvals": "pvalue",
        "pvals_adj": "padj",
    })
    df = df[np.isfinite(df["log2FC"])].copy()
    df["is_zinc"] = df["gene"].isin(zinc_genes)
    return df


def _run_de_pb_vs_csf_single_condition(
    adata: sc.AnnData,
    zinc_genes: list[str],
    condition: str,
    min_pct: float = 0.10,
) -> pd.DataFrame:
    """
    DE for PB vs CSF in CXCR3+ B cells for one condition, e.g. HC.
    """
    print(f"\nRunning DE: PB vs CSF CXCR3+, condition='{condition}' …")

    sub = adata[
        (adata.obs["cxcr3_group"] == "CXCR3+")
        & (adata.obs["condition"] == condition)
        & (adata.obs["tissue"].isin(["PB", "CSF"]))
    ].copy()

    if sub.n_obs == 0 or not {"PB", "CSF"}.issubset(set(sub.obs["tissue"].unique())):
        raise ValueError(f"No CXCR3+ PB+CSF cells for condition '{condition}'.")

    sub = _apply_min_pct_filter_tissue(sub, min_pct=min_pct)

    key = f"de_PB_vs_CSF_CXCR3pos_{condition}"
    sc.tl.rank_genes_groups(
        sub,
        groupby="tissue",
        groups=["CSF"],
        reference="PB",
        method="wilcoxon",
        key_added=key,
    )
    df = sc.get.rank_genes_groups_df(sub, group="CSF", key=key)
    df = df.rename(columns={
        "names": "gene",
        "logfoldchanges": "log2FC",
        "pvals": "pvalue",
        "pvals_adj": "padj",
    })
    df = df[np.isfinite(df["log2FC"])].copy()
    df["is_zinc"] = df["gene"].isin(zinc_genes)
    return df


def _select_genes_for_bars(df: pd.DataFrame, zinc_genes, max_genes=20):
    zinc = df[df["is_zinc"]].sort_values("log2FC", ascending=False)
    genes = zinc["gene"].head(max_genes).tolist()
    return genes, False


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
    padj_v = df_b.set_index("gene").loc[genes, "padj"].values

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


def draw_violin_pb_vs_csf(ax, df_sig, group_label, letter):
    df = df_sig[df_sig["group_label"] == group_label].copy()
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
    stat, p = mannwhitneyu(pb_vals, csf_vals, alternative="two-sided")

    print(
        f"[{group_label}] PB median={np.median(pb_vals):.3f}, "
        f"CSF median={np.median(csf_vals):.3f}, "
        f"PB mean={np.mean(pb_vals):.3f}, "
        f"CSF mean={np.mean(csf_vals):.3f}, "
        f"MW U={stat:.1f}, p={p:.3e}"
    )

    ax.set_xticks([1, 2])
    ax.set_xticklabels(groups)
    ax.set_ylabel("Zinc/MT signature (per cell)")
    ax.set_title(f"{group_label} CXCR3+ – PB vs CSF\nMann–Whitney p = {p:.2e}", fontsize=9)
    _panel_letter(ax, letter)


def _save_single_volcano_pb_vs_csf(df, title, label, out_png, out_pdf):
    fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))
    _draw_volcano_pb_vs_csf(ax, df, title=title, label=label)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def make_fig3_pb_vs_csf() -> None:
    merged_path = RESULTS_DIR / "merged_bcells.h5ad"
    if not merged_path.exists():
        raise FileNotFoundError(f"merged_bcells.h5ad not found at {merged_path}")

    adata = sc.read_h5ad(merged_path)
    if not adata.obs_names.is_unique:
        print("[fix] Making obs_names unique")
        adata.obs_names_make_unique()

    for col in ("condition", "cxcr3_group", "tissue"):
        if col not in adata.obs.columns:
            raise KeyError(f"Column '{col}' missing in adata.obs.")

    zinc_genes = load_zinc_gene_list()
    print(f"Loaded {len(zinc_genes)} curated zinc/MT genes: {zinc_genes}")

    # CIS+MS DE
    df_de = _run_de_pb_vs_csf(
        adata,
        zinc_genes=zinc_genes,
        grouping=GROUPING_MODE,
        min_pct=0.10,
    )
    print(f"  {df_de.shape[0]} genes in PB vs CSF DE table")
    bar_genes, _ = _select_genes_for_bars(df_de, zinc_genes=zinc_genes)
    print("CIS+MS PB vs CSF bar genes:", bar_genes)

    # HC-only DE
    try:
        df_de_hc = _run_de_pb_vs_csf_single_condition(
            adata,
            zinc_genes=zinc_genes,
            condition="HC",
            min_pct=0.10,
        )
        hc_bar_genes, _ = _select_genes_for_bars(df_de_hc, zinc_genes=zinc_genes)
        print("HC PB vs CSF bar genes:", hc_bar_genes)
    except ValueError as e:
        print(f"[HC DE skipped] {e}")
        df_de_hc = None
        hc_bar_genes = []

    # Signatures
    df_sig = compute_cell_signature_pb_vs_csf(
        adata, zinc_genes=zinc_genes, grouping=GROUPING_MODE
    )
    # Save per-cell zinc signatures used for Mann–Whitney PB vs CSF comparisons (Fig 3C, 3D)
    # HC (panel 3C)
    if "HC" in df_sig["group_label"].unique():
        df_sig_hc = df_sig[df_sig["group_label"] == "HC"].copy()
        df_sig_hc.to_csv(RESULTS_DIR / "Fig3_CXCR3pos_PB_vs_CSF_HC_zinc_signature_per_cell.csv",
                         index=False)

    # CIS+MS (panel 3D)
    if "CIS+MS" in df_sig["group_label"].unique():
        df_sig_cisms = df_sig[df_sig["group_label"] == "CIS+MS"].copy()
        df_sig_cisms.to_csv(RESULTS_DIR / "Fig3_CXCR3pos_PB_vs_CSF_CISMS_zinc_signature_per_cell.csv",
                            index=False)


    # Main figure: 2 × 2
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    (axA, axB), (axC, axD) = axes



    # A= HC bars
    if df_de_hc is not None:
        _draw_bars_pb_vs_csf(
            axA,
            df_de_hc,
            hc_bar_genes,
            title="HC CXCR3+ – zinc/MT genes (CSF vs PB)",
            label="A",
            color=BAR_COLOR_CMP,
        )
    else:
        axA.axis("off")
        axA.text(0.5, 0.5, "No HC CXCR3+ PB+CSF cells", ha="center", va="center")
        
    # B = CIS+MS bars
    _draw_bars_pb_vs_csf(
        axB,
        df_de,
        bar_genes,
        title="CIS+MS CXCR3+ – zinc/MT genes (CSF vs PB)",
        label="B",
        color=BAR_COLOR_CMP,
    )

    # C = HC violin
    if "HC" in df_sig["group_label"].unique():
        draw_violin_pb_vs_csf(axC, df_sig, group_label="HC", letter="C")
    else:
        axC.axis("off")
        axC.text(0.5, 0.5, "No HC CXCR3+ PB+CSF cells", ha="center", va="center")

    # D = CIS+MS violin
    if "CIS+MS" in df_sig["group_label"].unique():
        draw_violin_pb_vs_csf(axD, df_sig, group_label="CIS+MS", letter="D")
    else:
        axD.axis("off")
        axD.text(0.5, 0.5, "No CIS+MS CXCR3+ PB+CSF cells", ha="center", va="center")

    fig.subplots_adjust(
        left=0.08, right=0.98,
        top=0.93, bottom=0.10,
        wspace=0.35, hspace=0.50,
    )

    # Save outputs
    out_dir = FIG_DIRS.get("fig3", RESULTS_DIR / "fig3")
    out_dir.mkdir(parents=True, exist_ok=True)
    supp_dir = out_dir / "supp"
    supp_dir.mkdir(parents=True, exist_ok=True)

    stem = f"Fig3_CXCR3pos_PB_vs_CSF_{GROUPING_MODE}_4panel"
    out_png = out_dir / f"{stem}.png"
    out_pdf = out_dir / f"{stem}.pdf"

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    # Supplementary volcanoes
    _save_single_volcano_pb_vs_csf(
        df_de,
        title="CXCR3+ B cells – CSF vs PB (CIS+MS)",
        label="A",
        out_png=supp_dir / "Fig3_supp_CISMS_volcano.png",
        out_pdf=supp_dir / "Fig3_supp_CISMS_volcano.pdf",
    )

    if df_de_hc is not None:
        _save_single_volcano_pb_vs_csf(
            df_de_hc,
            title="CXCR3+ B cells – CSF vs PB (HC)",
            label="B",
            out_png=supp_dir / "Fig3_supp_HC_volcano.png",
            out_pdf=supp_dir / "Fig3_supp_HC_volcano.pdf",
        )

    # Save DE tables and bar-gene lists
    df_de.to_csv(out_dir / f"{stem}_CISMS_full_DE.csv", index=False)
    pd.DataFrame({"gene": bar_genes}).to_csv(
        out_dir / f"{stem}_CISMS_zinc_bar_genes.csv", index=False
    )

    if df_de_hc is not None:
        df_de_hc.to_csv(out_dir / f"{stem}_HC_full_DE.csv", index=False)
        pd.DataFrame({"gene": hc_bar_genes}).to_csv(
            out_dir / f"{stem}_HC_zinc_bar_genes.csv", index=False
        )

    print(f"\nSaved main figure: {out_png}")
    print(f"Saved main figure: {out_pdf}")
    print(f"Saved supplementary volcanoes in: {supp_dir}")


if __name__ == "__main__":
    make_fig3_pb_vs_csf()
   