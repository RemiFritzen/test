#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 18:29:44 2026

@author: remi
"""
from __future__ import annotations

import pathlib

import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc

from config.config import RESULTS_DIR, QC_DIR, DATASETS_YAML
from data_io import load_cfg

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

def _ensure_qc_dir() -> pathlib.Path:
    qc_dir = QC_DIR if QC_DIR is not None else (RESULTS_DIR / "qc")
    qc_dir.mkdir(parents=True, exist_ok=True)
    return qc_dir

def _load_adata() -> sc.AnnData:
    merged_path = RESULTS_DIR / "merged_bcells.h5ad"
    if not merged_path.exists():
        raise FileNotFoundError(f"merged_bcells.h5ad not found at {merged_path}")
    adata = sc.read_h5ad(merged_path)
    if not adata.obs_names.is_unique:
        adata.obs_names_make_unique()
    return adata

def _debug_obs_qc_columns(adata: sc.AnnData) -> None:
    print("[qc] obs columns:", list(adata.obs.columns))
    for col in [
        "n_genes_by_counts",
        "n_genes",
        "n_counts",
        "total_counts",
        "pct_counts_mt",
        "percent_mito",
        "pct_mito",
    ]:
        if col in adata.obs.columns:
            v = adata.obs[col]
            print(
                f"[qc] {col}: min={v.min()}, max={v.max()}, "
                f"n_nonzero={(v>0).sum()}, n_na={v.isna().sum()}"
            )
        else:
            print(f"[qc] {col}: NOT PRESENT")

def qc_cells_per_sample(adata: sc.AnnData, qc_dir: pathlib.Path) -> None:
    # pick columns if they exist
    cols = []
    for candidate in ("n_genes_by_counts", "n_genes"):
        if candidate in adata.obs.columns:
            cols.append(candidate)
            break
    for candidate in ("n_counts", "total_counts"):
        if candidate in adata.obs.columns:
            cols.append(candidate)
            break
    for candidate in ("pct_counts_mt", "percent_mito", "pct_mito"):
        if candidate in adata.obs.columns:
            cols.append(candidate)
            break

    if not cols:
        print("[qc] No standard QC columns found in adata.obs")
        return

    # sample id
    if "sample" in adata.obs.columns:
        sample_col = "sample"
    elif "donor_id" in adata.obs.columns:
        sample_col = "donor_id"
    else:
        sample_col = None

    tissues = sorted(adata.obs["tissue"].unique())

    # Violin per tissue for each metric, but skip all-zero/all-NaN
    for col in cols:
        vals_all = adata.obs[col]
        if vals_all.isna().all() or np.nanmax(vals_all.values) == 0:
            print(f"[qc] Skipping {col}: all zeros or NaN")
            continue

        tissues = sorted(adata.obs["tissue"].unique())
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))

        positions = []
        labels = []
        for i, tissue in enumerate(tissues, start=1):
            vals = adata.obs.loc[adata.obs["tissue"] == tissue, col].dropna()
            if vals.empty:
                continue

            positions.append(i)
            labels.append(tissue)

            parts = ax.violinplot(
                [vals.values],
                positions=[i],
                showmeans=False,
                showmedians=True,
            )
            for pc in parts["bodies"]:
                pc.set_alpha(0.4)
            parts["cmedians"].set_color("black")

            # add a few jittered points so something is visibly there
            jitter = np.random.normal(0, 0.05, size=min(100, len(vals)))
            sample_vals = vals.sample(min(100, len(vals)), random_state=0)
            ax.scatter(
                np.full(len(sample_vals), i) + jitter,
                sample_vals,
                s=6,
                alpha=0.4,
                color="black",
            )

        if not positions:
            print(f"[qc] No non-empty tissues for {col}, skipping plot")
            plt.close(fig)
            continue

        ax.set_xticks(positions)
        ax.set_xticklabels(labels)

        # log-scale for counts to avoid everything at the bottom
        if col in ("n_genes_by_counts", "total_counts", "n_counts"):
            ax.set_yscale("log")

        ax.set_ylabel(col)
        ax.set_title(f"{col} by tissue")
        fig.tight_layout()
        out_path = qc_dir / f"qc_{col}_by_tissue.pdf"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[qc] Wrote {out_path}")

def qc_cells_total_per_tissue(adata: sc.AnnData, qc_dir: pathlib.Path) -> None:
    df = adata.obs.groupby("tissue").size().reset_index(name="n_cells")
    fig, ax = plt.subplots(1, 1, figsize=(2, 3))
    ax.bar(df["tissue"], df["n_cells"], color="#4C78A8")
    ax.set_ylabel("cells")
    ax.set_title("Total cells per tissue")
    fig.tight_layout()
    out_path = qc_dir / "qc_cells_total_per_tissue.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[qc] Wrote {out_path}")



def qc_umap(adata: sc.AnnData, qc_dir: pathlib.Path) -> None:
    if "X_umap" not in adata.obsm_keys():
        print("[qc] No X_umap found, computing a quick UMAP for QC")
        sc.pp.pca(adata, n_comps=30)
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)

    for color in ("tissue", "condition", "cxcr3_group"):
        if color not in adata.obs.columns:
            continue
        fig = sc.pl.umap(
            adata,
            color=color,
            size=10,
            show=False,
            return_fig=True,
        )
        fig.savefig(qc_dir / f"qc_umap_by_{color}.pdf", bbox_inches="tight")
        plt.close(fig)

def qc_zinc_genes(adata: sc.AnnData, qc_dir: pathlib.Path) -> None:
    cfg = load_cfg(DATASETS_YAML)
    markers = cfg.get("markers", {})
    zinc_genes = markers.get("zinc_transporters", []) + markers.get("metallothioneins", [])
    genes = [g for g in zinc_genes if g in adata.var_names]
    if not genes:
        print("[qc] No zinc genes found in var_names"); return

    sub = adata[adata.obs["cxcr3_group"] == "CXCR3+"].copy()

    sc.pl.dotplot(
        sub,
        var_names=genes,
        groupby="tissue",
        standard_scale="var",
        show=False,
    )
    
    fig = plt.gcf()
    fig.savefig(
        qc_dir / "qc_zinc_genes_dotplot_CXCR3pos_by_tissue.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)



from output.figures.fig2_cxcr3_zinc.fig2 import compute_cell_signature  # if you expose it at module level

def qc_zinc_signature(adata: sc.AnnData, qc_dir: pathlib.Path) -> None:
    cfg = load_cfg(DATASETS_YAML)
    markers = cfg.get("markers", {})
    zinc_genes = markers.get("zinc_transporters", []) + markers.get("metallothioneins", [])
    df_pb = compute_cell_signature(adata, zinc_genes, tissue="PB")
    df_csf = compute_cell_signature(adata, zinc_genes, tissue="CSF")

    for label, df in (("PB", df_pb), ("CSF", df_csf)):
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        ax.hist(df["zinc_signature"], bins=40, alpha=0.7, color="#4C78A8")
        ax.set_xlabel("zinc signature")
        ax.set_ylabel("cells")
        ax.set_title(f"Zinc signature distribution – {label} CXCR3+")
        fig.tight_layout()
        fig.savefig(qc_dir / f"qc_zinc_signature_hist_{label}.pdf", bbox_inches="tight")
        plt.close(fig)


def make_qc_figs() -> None:
    qc_dir = _ensure_qc_dir()
    adata = _load_adata()

    _debug_obs_qc_columns(adata)  # <<< add this line

    qc_cells_per_sample(adata, qc_dir)
    qc_umap(adata, qc_dir)
    qc_zinc_genes(adata, qc_dir)
    qc_zinc_signature(adata, qc_dir)
    qc_cells_per_sample(adata, qc_dir)
    qc_cells_total_per_tissue(adata, qc_dir)


    print(f"[qc] Wrote QC outputs to {qc_dir}")


if __name__ == "__main__":
    make_qc_figs()
