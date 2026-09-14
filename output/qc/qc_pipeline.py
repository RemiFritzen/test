#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qc_pipeline.py
==============
QC + cluster-stability + metadata-completeness pipeline for the CSF vs plasma
B-cell scRNA-seq comparison (CXCR3 / MS project).

Covers GSE133028 (per-patient CSF/PB), GSE138266 (per-patient CSF/PB),
and GSE239626 (global CITE-seq object).

Writes everything into output/qc/:
  - qc_metrics.csv            per-sample QC table with PASS/WARN/FAIL flags
  - cluster_stability.csv     per-sample clustering stability table
  - metadata_audit.csv        per-sample metadata completeness + anomaly flags
  - qc_summary.json           machine-readable summary consumed by the dashboard
  - plots/<sample>_qc.png      per-sample QC violin/scatter panels
  - plots/<sample>_umap.png    per-sample UMAP coloured by leiden cluster
  - plots/overview_*.png       cohort-level overview plots

Run with the spyder-runtime python. Cache dirs must point to a writable place.
"""
import os, sys, json, glob, warnings, traceback
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scipy.sparse as sp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scanpy as sc
from sklearn.metrics import silhouette_score, adjusted_rand_score

sc.settings.verbosity = 0
np.random.seed(0)

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
BASE = "/Users/remi/Library/Mobile Documents/com~apple~CloudDocs/Work/Papers/Research papers/2026 CXCR3"
PROC = os.path.join(BASE, "data", "processed")
QC_DIR = os.path.join(BASE, "output", "qc")
PLOT_DIR = os.path.join(QC_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# ----------------------------------------------------------------------------
# QC thresholds (pass/fail markers). Tunable; documented in dashboard.
# ----------------------------------------------------------------------------
THRESH = {
    "min_cells":            200,    # FAIL below; WARN below 2x
    "min_cells_warn":       500,
    "min_median_genes":     500,    # FAIL below; WARN below 2x-ish
    "min_median_genes_warn":750,
    "max_median_pct_mt":    10.0,   # WARN above; FAIL above hard
    "max_median_pct_mt_fail":20.0,
    "min_median_counts":    1000,
    "min_median_counts_warn":1500,
    # cluster stability
    "min_silhouette":       0.10,   # FAIL below; WARN below 0.20
    "min_silhouette_warn":  0.20,
    "min_ari_stability":    0.60,   # mean ARI across resolution neighbours
    "min_ari_stability_warn":0.75,
}

MT_PREFIX = ("MT-", "mt-")

# ----------------------------------------------------------------------------
# Sample registry
# ----------------------------------------------------------------------------
def discover_samples():
    samples = []
    # GSE133028 per-patient
    for f in sorted(glob.glob(os.path.join(PROC, "GSE133028", "*.h5ad"))):
        name = os.path.splitext(os.path.basename(f))[0]
        samples.append({"dataset": "GSE133028", "sample_id": name, "path": f,
                        "expected_tissue": "CSF" if name.startswith("CSF") else "PB"})
    # GSE138266 per-patient (exclude merged/bad)
    for f in sorted(glob.glob(os.path.join(PROC, "GSE138266", "*.h5ad"))):
        name = os.path.splitext(os.path.basename(f))[0]
        if "processed" in name or "bad" in name:
            continue
        exp = "CSF" if name.endswith("_CSF") else ("PB" if name.endswith("_PB") else "?")
        samples.append({"dataset": "GSE138266", "sample_id": name, "path": f,
                        "expected_tissue": exp})
    # GSE239626 global - treat as a single object; will split by 'sample' if informative
    g = os.path.join(PROC, "GSE239626", "GSE239626_processed.h5ad")
    if os.path.exists(g):
        samples.append({"dataset": "GSE239626", "sample_id": "GSE239626_global", "path": g,
                        "expected_tissue": "?"})
    return samples


# ----------------------------------------------------------------------------
# QC metric computation
# ----------------------------------------------------------------------------
def compute_qc_metrics(adata):
    """Ensure n_genes_by_counts, total_counts, pct_counts_mt exist on .obs."""
    X = adata.X
    needs = not all(c in adata.obs.columns for c in
                    ["n_genes_by_counts", "total_counts", "pct_counts_mt"])
    if needs:
        adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
        sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True,
                                   percent_top=None, log1p=False)
    # always (re)derive pct_counts_mt if mt genes exist but col missing
    if "pct_counts_mt" not in adata.obs.columns:
        adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
        sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True,
                                   percent_top=None, log1p=False)
    return adata


def qc_flags(row):
    flags = {}
    # cells
    if row["n_cells"] < THRESH["min_cells"]:
        flags["cells"] = "FAIL"
    elif row["n_cells"] < THRESH["min_cells_warn"]:
        flags["cells"] = "WARN"
    else:
        flags["cells"] = "PASS"
    # genes
    if row["median_genes"] < THRESH["min_median_genes"]:
        flags["genes"] = "FAIL"
    elif row["median_genes"] < THRESH["min_median_genes_warn"]:
        flags["genes"] = "WARN"
    else:
        flags["genes"] = "PASS"
    # counts
    if row["median_counts"] < THRESH["min_median_counts"]:
        flags["counts"] = "FAIL"
    elif row["median_counts"] < THRESH["min_median_counts_warn"]:
        flags["counts"] = "WARN"
    else:
        flags["counts"] = "PASS"
    # mito
    mt = row["median_pct_mt"]
    if mt != mt:  # nan -> no mito genes detected
        flags["mito"] = "WARN"
    elif mt > THRESH["max_median_pct_mt_fail"]:
        flags["mito"] = "FAIL"
    elif mt > THRESH["max_median_pct_mt"]:
        flags["mito"] = "WARN"
    else:
        flags["mito"] = "PASS"
    return flags


def overall_flag(flag_dict):
    vals = list(flag_dict.values())
    if "FAIL" in vals:
        return "FAIL"
    if "WARN" in vals:
        return "WARN"
    return "PASS"


# ----------------------------------------------------------------------------
# Clustering + stability
# ----------------------------------------------------------------------------
def cluster_and_stability(adata, sample_id, n_top=2000, max_cells=6000):
    """Standard scanpy pipeline + resolution sweep stability via ARI.
    Returns dict of stability metrics and a clustered (subsampled) adata for plotting."""
    res = {"sample_id": sample_id}
    a = adata.copy()
    # subsample very large objects for speed/stability comparability
    if a.n_obs > max_cells:
        sc.pp.subsample(a, n_obs=max_cells, random_state=0)
    res["n_cells_clustered"] = int(a.n_obs)
    # Detect whether data already looks log-normalised
    Xmax = a.X.max()
    is_lognorm = Xmax < 30
    a.layers["counts"] = a.X.copy()
    if not is_lognorm:
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
    res["assumed_lognorm_input"] = bool(is_lognorm)
    try:
        sc.pp.highly_variable_genes(a, n_top_genes=min(n_top, a.n_vars - 1),
                                    flavor="seurat")
        a.raw = a
        a = a[:, a.var.highly_variable]
    except Exception:
        pass
    sc.pp.scale(a, max_value=10)
    n_pcs = int(min(50, a.n_obs - 1, a.n_vars - 1))
    sc.tl.pca(a, n_comps=n_pcs, svd_solver="arpack")
    n_neighbors = int(min(15, max(5, a.n_obs // 50)))
    sc.pp.neighbors(a, n_neighbors=n_neighbors, n_pcs=min(n_pcs, 40))
    # resolution sweep
    resolutions = [0.4, 0.6, 0.8, 1.0]
    labels = {}
    for r in resolutions:
        key = f"leiden_{r}"
        sc.tl.leiden(a, resolution=r, key_added=key, flavor="igraph",
                     n_iterations=2, directed=False, random_state=0)
        labels[r] = a.obs[key].values
    # number of clusters at primary resolution 0.6
    primary = "leiden_0.6"
    a.obs["leiden"] = a.obs[primary]
    res["n_clusters_res06"] = int(a.obs[primary].nunique())
    res["n_clusters_range"] = f"{min(a.obs[k].nunique() for k in [f'leiden_{r}' for r in resolutions])}-{max(a.obs[k].nunique() for k in [f'leiden_{r}' for r in resolutions])}"
    # ARI between adjacent resolutions = stability proxy
    aris = []
    for i in range(len(resolutions) - 1):
        aris.append(adjusted_rand_score(labels[resolutions[i]], labels[resolutions[i + 1]]))
    res["mean_ari_adjacent_res"] = float(np.mean(aris)) if aris else float("nan")
    # silhouette on PCA using primary clustering
    try:
        emb = a.obsm["X_pca"][:, :min(20, n_pcs)]
        lab = a.obs[primary].astype(str).values
        if len(set(lab)) > 1 and a.n_obs > 10:
            # subsample silhouette for speed if large
            idx = np.arange(a.n_obs)
            if a.n_obs > 4000:
                idx = np.random.choice(a.n_obs, 4000, replace=False)
            res["silhouette_res06"] = float(silhouette_score(emb[idx], lab[idx]))
        else:
            res["silhouette_res06"] = float("nan")
    except Exception:
        res["silhouette_res06"] = float("nan")
    # UMAP for plotting
    try:
        sc.tl.umap(a, random_state=0)
        res["_umap_adata"] = a
    except Exception:
        res["_umap_adata"] = None
    return res


def stability_flags(row):
    flags = {}
    s = row.get("silhouette_res06", float("nan"))
    if s != s:
        flags["silhouette"] = "WARN"
    elif s < THRESH["min_silhouette"]:
        flags["silhouette"] = "FAIL"
    elif s < THRESH["min_silhouette_warn"]:
        flags["silhouette"] = "WARN"
    else:
        flags["silhouette"] = "PASS"
    ari = row.get("mean_ari_adjacent_res", float("nan"))
    if ari != ari:
        flags["ari"] = "WARN"
    elif ari < THRESH["min_ari_stability"]:
        flags["ari"] = "FAIL"
    elif ari < THRESH["min_ari_stability_warn"]:
        flags["ari"] = "WARN"
    else:
        flags["ari"] = "PASS"
    return flags


# ----------------------------------------------------------------------------
# Metadata audit
# ----------------------------------------------------------------------------
EXPECTED_META = ["patient", "donor", "tissue", "condition", "celltype", "dataset", "sample"]

def normalise_tissue(val):
    if val is None:
        return None
    v = str(val).strip().upper()
    if v in ("PB", "PBMC", "PBMCS", "BLOOD", "PLASMA", "PERIPHERAL BLOOD"):
        return "PB"
    if v in ("CSF", "CEREBROSPINAL FLUID"):
        return "CSF"
    return str(val)


def metadata_audit(adata, smeta):
    obs = adata.obs
    row = {"sample_id": smeta["sample_id"], "dataset": smeta["dataset"]}
    present = [c for c in EXPECTED_META if c in obs.columns]
    missing = [c for c in EXPECTED_META if c not in obs.columns]
    row["meta_fields_present"] = ";".join(present)
    row["meta_missing"] = ";".join(missing)
    row["completeness_pct"] = round(100 * len(present) / len(EXPECTED_META), 1)
    # null fractions per present field
    nullinfo = {}
    for c in present:
        nullinfo[c] = round(100 * obs[c].isna().mean(), 1)
    row["null_pct_per_field"] = json.dumps(nullinfo)
    # tissue anomaly check
    anomalies = []
    if "tissue" in obs.columns:
        uniq = [str(x) for x in pd.unique(obs["tissue"].dropna())]
        row["tissue_values"] = ";".join(uniq)
        norm = set(normalise_tissue(x) for x in uniq)
        # raw label inconsistency (e.g. PBMCs vs PB)
        if any(u not in ("CSF", "PB") for u in uniq):
            anomalies.append(f"nonstandard_tissue_label({';'.join(uniq)})")
        # mismatch vs expected from filename
        exp = smeta.get("expected_tissue")
        if exp in ("CSF", "PB") and norm and norm != {exp}:
            anomalies.append(f"tissue_obs={';'.join(uniq)}_vs_filename={exp}")
    else:
        row["tissue_values"] = ""
        anomalies.append("no_tissue_column")
    if "celltype" not in obs.columns:
        anomalies.append("no_celltype_annotation")
    if "condition" not in obs.columns:
        anomalies.append("no_condition_column")
    # single-value patient/sample where multiple expected
    row["anomalies"] = ";".join(anomalies) if anomalies else ""
    row["meta_flag"] = "FAIL" if any("no_tissue" in a or "vs_filename" in a for a in anomalies) \
        else ("WARN" if anomalies else "PASS")
    return row


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def plot_qc_panel(adata, sample_id):
    obs = adata.obs
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.4))
    fig.suptitle(f"{sample_id}  QC", fontsize=12, fontweight="bold")
    # genes
    axes[0].violinplot([obs["n_genes_by_counts"].values], showmedians=True)
    axes[0].set_title("genes / cell"); axes[0].set_xticks([])
    # counts
    axes[1].violinplot([obs["total_counts"].values], showmedians=True)
    axes[1].set_title("counts / cell"); axes[1].set_xticks([])
    # pct mito
    if "pct_counts_mt" in obs.columns:
        axes[2].violinplot([obs["pct_counts_mt"].values], showmedians=True)
    axes[2].set_title("% mito"); axes[2].set_xticks([])
    axes[2].axhline(THRESH["max_median_pct_mt"], color="orange", ls="--", lw=1)
    axes[2].axhline(THRESH["max_median_pct_mt_fail"], color="red", ls="--", lw=1)
    # counts vs genes scatter coloured by mito
    c = obs["pct_counts_mt"] if "pct_counts_mt" in obs.columns else None
    sca = axes[3].scatter(obs["total_counts"], obs["n_genes_by_counts"],
                          c=c, s=3, cmap="viridis", alpha=0.5)
    axes[3].set_xlabel("total_counts"); axes[3].set_ylabel("n_genes")
    axes[3].set_title("counts vs genes")
    if c is not None:
        fig.colorbar(sca, ax=axes[3], label="% mito", fraction=0.046)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(PLOT_DIR, f"{sample_id}_qc.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_umap(a, sample_id, stab):
    if a is None or "X_umap" not in a.obsm:
        return None
    fig, ax = plt.subplots(figsize=(5, 4.2))
    sc.pl.umap(a, color="leiden", ax=ax, show=False, frameon=False,
               title=f"{sample_id}\nleiden res0.6  k={stab.get('n_clusters_res06')}  "
                     f"sil={stab.get('silhouette_res06', float('nan')):.2f}")
    out = os.path.join(PLOT_DIR, f"{sample_id}_umap.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    samples = discover_samples()
    print(f"[info] discovered {len(samples)} samples")
    qc_rows, stab_rows, meta_rows = [], [], []
    summary = {"thresholds": THRESH, "samples": []}

    for i, sm in enumerate(samples):
        sid = sm["sample_id"]
        print(f"[{i+1}/{len(samples)}] {sid} ...", flush=True)
        try:
            adata = sc.read_h5ad(sm["path"])
        except Exception as e:
            print(f"   READ FAIL: {e}")
            continue
        try:
            adata = compute_qc_metrics(adata)
        except Exception as e:
            print(f"   QC metric fail: {e}")
        obs = adata.obs
        qc = {
            "sample_id": sid, "dataset": sm["dataset"],
            "expected_tissue": sm["expected_tissue"],
            "n_cells": int(adata.n_obs), "n_genes_total": int(adata.n_vars),
            "median_genes": float(np.median(obs["n_genes_by_counts"])) if "n_genes_by_counts" in obs else float("nan"),
            "median_counts": float(np.median(obs["total_counts"])) if "total_counts" in obs else float("nan"),
            "median_pct_mt": float(np.median(obs["pct_counts_mt"])) if "pct_counts_mt" in obs else float("nan"),
        }
        fl = qc_flags(qc)
        qc.update({f"flag_{k}": v for k, v in fl.items()})
        qc["qc_overall"] = overall_flag(fl)
        qc_rows.append(qc)

        # metadata
        mrow = metadata_audit(adata, sm)
        meta_rows.append(mrow)

        # plots
        try:
            plot_qc_panel(adata, sid)
        except Exception as e:
            print(f"   qc plot fail: {e}")

        # clustering + stability
        try:
            stab = cluster_and_stability(adata, sid)
            uplot = plot_umap(stab.pop("_umap_adata", None), sid, stab)
            sfl = stability_flags(stab)
            stab.update({f"flag_{k}": v for k, v in sfl.items()})
            stab["stability_overall"] = overall_flag(sfl)
            stab["dataset"] = sm["dataset"]
            stab_rows.append(stab)
        except Exception as e:
            print(f"   clustering fail: {e}")
            traceback.print_exc()
            stab_rows.append({"sample_id": sid, "dataset": sm["dataset"],
                              "stability_overall": "FAIL",
                              "error": str(e)})
        del adata

    qc_df = pd.DataFrame(qc_rows)
    stab_df = pd.DataFrame(stab_rows)
    meta_df = pd.DataFrame(meta_rows)
    qc_df.to_csv(os.path.join(QC_DIR, "qc_metrics.csv"), index=False)
    stab_df.to_csv(os.path.join(QC_DIR, "cluster_stability.csv"), index=False)
    meta_df.to_csv(os.path.join(QC_DIR, "metadata_audit.csv"), index=False)

    # cohort overview plot
    try:
        plot_overview(qc_df)
    except Exception as e:
        print("overview plot fail", e)

    # merged summary json for dashboard
    merged = qc_df.merge(stab_df, on=["sample_id", "dataset"], how="left", suffixes=("", "_stab"))
    merged = merged.merge(meta_df[["sample_id", "completeness_pct", "meta_missing",
                                   "tissue_values", "anomalies", "meta_flag"]],
                          on="sample_id", how="left")
    summary["records"] = json.loads(merged.to_json(orient="records"))
    summary["generated"] = pd.Timestamp.now().isoformat()
    with open(os.path.join(QC_DIR, "qc_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("[done] wrote qc_metrics.csv, cluster_stability.csv, metadata_audit.csv, qc_summary.json")
    print(f"[done] plots in {PLOT_DIR}")


def plot_overview(qc_df):
    if qc_df.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    df = qc_df.copy()
    df["grp"] = df["dataset"] + "\n" + df["expected_tissue"]
    for ax, col, title in zip(axes,
                              ["n_cells", "median_genes", "median_pct_mt"],
                              ["cells per sample", "median genes/cell", "median % mito"]):
        order = sorted(df["sample_id"])
        colors = {"PASS": "#2e9e5b", "WARN": "#e0a106", "FAIL": "#d23b3b"}
        flagcol = {"n_cells": "flag_cells", "median_genes": "flag_genes",
                   "median_pct_mt": "flag_mito"}[col]
        bar_c = [colors.get(f, "#888") for f in df[flagcol]]
        ax.bar(range(len(df)), df[col].values, color=bar_c)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df["sample_id"], rotation=90, fontsize=6)
        ax.set_title(title)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "overview_qc.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
