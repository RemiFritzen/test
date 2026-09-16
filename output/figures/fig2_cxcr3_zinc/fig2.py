# Fig2 · PY
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
 
"""
fig2.py — PB + CSF CXCR3+ MS vs HC zinc signature
 
Outputs:
  1. Main 4-panel figure:
     A = PB zinc/MT bar plot
     B = CSF zinc/MT bar plot
     C = PB patient-level box plot (HC vs MS)
     D = CSF patient-level box plot (HC vs MS)
 
  2. Supplementary scatter figures saved separately in the supp directory:
     - PB volcano/scatter
     - CSF volcano/scatter
 
Also exports DE tables and selected bar-gene CSVs.
 
PATIENT-LEVEL STATISTICS
Both the per-gene DE (_run_de) and the composite signature comparison
previously ran their statistical test directly on per-cell values, even
though each patient contributes many correlated cells and each patient
belongs to only ONE condition (HC or MS). That's pseudoreplication: the
effective n used by the test is n_cells, not n_patients, which can turn a
real but modest patient-level effect into an artificially tiny p-value.
 
Both now use `pseudobulk_de` / `pseudobulk_signature_test` from
stats_utils.py: cells are first aggregated to one value per patient, and
the test runs on those ~n_patients values. This is the *unpaired*
pseudobulk form, correct here because HC vs MS is a between-patient
comparison (unlike CXCR3+/- or PB/CSF, where the same patient appears in
both arms and the *paired* form is needed instead).
 
PRESENTATION CHANGES (this revision)
  - Panel titles removed; sample sizes moved into the y-axis labels and
    the composite p-value annotated inside the box-plot axes.
  - Bars coloured by gene class: ZIP (SLC39A, import) and ZnT (SLC30A,
    export) as two tints of the same teal, metallothioneins in a
    contrasting red. Makes the coordinated transporter-up/MT-down pattern
    visible before the caption is read, which is what justifies the
    sign-corrected composite in panels C/D.
  - Significance stars now sit below negative bars instead of being
    pinned to y=0, where they appeared detached from the bar they marked.
  - Box-plot row made shorter than the bar row and the boxes narrowed.
  - Fonts enlarged throughout for legibility at figure scale.
"""
 
import scipy.sparse as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
from matplotlib.patches import Patch
from statsmodels.stats.multitest import multipletests
 
from config.config import RESULTS_DIR, FIG_DIRS, DATASETS_YAML
from data_io import load_cfg
from stats_utils import pseudobulk_de, pseudobulk_signature_test, compute_gene_signs
 
 
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.edgecolor": "#1F2933",
    "text.color": "#1F2933",
    "axes.labelcolor": "#1F2933",
    "xtick.color": "#1F2933",
    "ytick.color": "#1F2933",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
 
# ---------------------------------------------------------------------
# Palette matches the Approach document's other figures: teal for
# zinc/biochemistry, purple for B cells/MS, muted grey-blue for controls,
# neutral ink for text.
#
# Gene classes: ZIP and ZnT share a hue at two tints because they are one
# functional group (transporters) with opposite directionality;
# metallothioneins take a contrasting colour because the composite
# signature inverts their sign.
# ---------------------------------------------------------------------
INK = "#1F2933"
MUTED = "#6B757C"
LINE = "#A7B0B7"
ZINC = "#1F7A8C"
BCELL = "#6B4E9B"
PALE_B = "#E3DCEF"
PALE_GREY = "#E4E6E8"
 
HC_COLOR = MUTED
HC_PALE = PALE_GREY
MS_COLOR = BCELL
MS_PALE = PALE_B
 
ZIP_COLOR = "#14606F"    # dark teal  — SLC39A / ZIP (import)
ZNT_COLOR = "#7FBECB"    # light teal — SLC30A / ZnT (export)
MT_COLOR = "#C1666B"     # warm red   — metallothioneins
TICK_TEAL = "#14606F"    # tick labels: the light tint is illegible as text
 
CLASS_COLORS = {"ZIP": ZIP_COLOR, "ZnT": ZNT_COLOR, "MT": MT_COLOR}
CLASS_LABELS = {
    "ZIP": "ZIP (SLC39A)",
    "ZnT": "ZnT (SLC30A)",
    "MT": "Metallothionein",
}
 
SCATTER_COLOR_OTHER = "#C7CBCE"
SCATTER_COLOR_ZINC_PB = ZINC
SCATTER_COLOR_ZINC_CSF = "#2B4C7E"
SIG_PADJ_THRESHOLD = 0.05
MIN_PATIENTS_PER_GROUP = 3
PANEL_KW = dict(fontsize=17, fontweight="bold", va="top", ha="left", color=INK)
 
 
def _panel_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(-0.13, 1.06, letter, transform=ax.transAxes, **PANEL_KW)
 
 
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
 
 
def _gene_class(gene: str, gene_signs: dict[str, float]) -> str:
    """
    Classify a panel gene as 'ZIP', 'ZnT' or 'MT' for colouring.
 
    Symbol prefix is used first (SLC39A* = ZIP importers, SLC30A* = ZnT
    exporters, MT* = metallothioneins), falling back to the a priori sign
    from compute_gene_signs for anything unrecognised, so an unexpected
    alias cannot silently drop out of the colour scheme.
    """
    g = gene.upper()
    if g.startswith("SLC39A"):
        return "ZIP"
    if g.startswith("SLC30A"):
        return "ZnT"
    if g.startswith("MT"):
        return "MT"
    return "ZIP" if gene_signs.get(gene, 1.0) > 0 else "MT"
 
 
def _apply_min_pct_filter(sub: sc.AnnData, min_pct: float, always_keep: list[str] | None = None) -> sc.AnnData:
    """
    Drop lowly-expressed genes before DE (mainly to declutter the volcano
    plot and avoid testing genes with almost no signal). `always_keep`
    (typically the curated zinc/MT panel) is exempted -- those genes are the
    entire point of the figure, so they're always tested and available for
    the bar chart regardless of expression prevalence, even if that means
    they end up correctly non-significant.
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
    Per-cell zinc/MT signature. Retained for the exported CSVs and for the
    supplementary per-cell visualisation; the main figure's panels C/D use
    patient-level values instead.
 
    gene_signs: if given, each gene's expression is sign-corrected before
    averaging (transporters +1, metallothioneins -1) so up- and
    down-regulated genes don't cancel toward zero. If None, computes a plain
    (unsigned) mean -- used for the supplementary comparison panel, since
    sign-correction encodes a directional assumption that isn't equally
    justified for every comparison.
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
 
def signature_outlier_robustness(
    sig_test_result: dict,
    tissue: str,
    label: str = "signed",
) -> dict:
    """
    Sensitivity of the HC-vs-MS composite signature test to individual
    patients.
 
    Returns a dict with the full-sample p, the leave-one-out p range and
    which exclusion produced the worst case, and the Tukey-fence result.
    `per_patient_df` in the return value is the per-exclusion table, for
    export.
 
    A result that holds under both checks is robust to any single patient.
    A result that survives the full-sample test but crosses 0.05 under
    leave-one-out is carried by one observation and should be reported
    with that stated.
    """
    from scipy.stats import mannwhitneyu
 
    per_patient = sig_test_result.get("per_patient_values")
    p_full = sig_test_result.get("mannwhitney_p", np.nan)
 
    out = {
        "tissue": tissue,
        "signature_label": label,
        "p_full": p_full,
        "loo_p_min": np.nan,
        "loo_p_max": np.nan,
        "loo_worst_patient": None,
        "loo_worst_group": None,
        "loo_all_significant": None,
        "n_tukey_excluded": 0,
        "tukey_excluded_patients": [],
        "p_tukey": np.nan,
        "tukey_significant": None,
        "per_patient_df": pd.DataFrame(),
    }
 
    if per_patient is None or per_patient.empty:
        print(f"  [{tissue}] robustness: no per-patient values available")
        return out
 
    df = per_patient.copy()
    if "patient" not in df.columns:
        df = df.reset_index().rename(columns={df.index.name or "index": "patient"})
 
    def _test(frame: pd.DataFrame) -> float:
        hc = frame.loc[frame["_group"] == "HC", "signature"].values
        ms = frame.loc[frame["_group"] == "MS", "signature"].values
        if len(hc) < MIN_PATIENTS_PER_GROUP or len(ms) < MIN_PATIENTS_PER_GROUP:
            return np.nan
        return mannwhitneyu(hc, ms, alternative="two-sided")[1]
 
    print(f"\n  === Outlier robustness: {tissue}, {label} signature ===")
    print(f"    Full-sample Mann-Whitney p = {p_full:.4g}"
          if np.isfinite(p_full) else "    Full-sample p not available")
 
    # --- 1. leave-one-out -------------------------------------------------
    rows = []
    for idx, row in df.iterrows():
        p_loo = _test(df.drop(index=idx))
        rows.append({
            "excluded_patient": row.get("patient", idx),
            "excluded_group": row["_group"],
            "excluded_value": row["signature"],
            "p_without": p_loo,
        })
    loo = pd.DataFrame(rows)
    out["per_patient_df"] = loo
 
    valid = loo.dropna(subset=["p_without"])
    if not valid.empty:
        worst = valid.loc[valid["p_without"].idxmax()]
        out["loo_p_min"] = valid["p_without"].min()
        out["loo_p_max"] = valid["p_without"].max()
        out["loo_worst_patient"] = worst["excluded_patient"]
        out["loo_worst_group"] = worst["excluded_group"]
        out["loo_all_significant"] = bool((valid["p_without"] < SIG_PADJ_THRESHOLD).all())
 
        print(f"    Leave-one-out p range: {out['loo_p_min']:.4g} to {out['loo_p_max']:.4g}")
        print(f"    Worst case: excluding {worst['excluded_patient']} "
              f"({worst['excluded_group']}, signature = {worst['excluded_value']:+.3f}) "
              f"gives p = {worst['p_without']:.4g}")
        if out["loo_all_significant"]:
            print(f"    -> significant with any single patient removed")
        else:
            n_lost = int((valid["p_without"] >= SIG_PADJ_THRESHOLD).sum())
            print(f"    -> [CAUTION] significance lost when {n_lost} particular "
                  f"patient(s) removed; the result is carried by individual "
                  f"observations and must be reported as such")
 
    # --- 2. Tukey fence ---------------------------------------------------
    keep = pd.Series(True, index=df.index)
    excluded = []
    for grp, sub in df.groupby("_group"):
        q1, q3 = np.percentile(sub["signature"], [25, 75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = sub[(sub["signature"] < lo) | (sub["signature"] > hi)]
        for idx, row in outliers.iterrows():
            keep[idx] = False
            excluded.append({
                "patient": row.get("patient", idx),
                "group": grp,
                "value": row["signature"],
            })
 
    out["n_tukey_excluded"] = len(excluded)
    out["tukey_excluded_patients"] = [e["patient"] for e in excluded]
 
    if excluded:
        for e in excluded:
            print(f"    Tukey fence flags {e['patient']} ({e['group']}, "
                  f"signature = {e['value']:+.3f})")
        p_tukey = _test(df[keep])
        out["p_tukey"] = p_tukey
        if np.isfinite(p_tukey):
            out["tukey_significant"] = bool(p_tukey < SIG_PADJ_THRESHOLD)
            verdict = "holds" if out["tukey_significant"] else "DOES NOT hold"
            print(f"    Excluding {len(excluded)} flagged value(s): p = {p_tukey:.4g} "
                  f"-> result {verdict}")
        else:
            print(f"    Too few patients remain after exclusion to retest")
    else:
        print(f"    No values beyond the 1.5*IQR fence in either group")
        out["p_tukey"] = p_full
        out["tukey_significant"] = bool(p_full < SIG_PADJ_THRESHOLD) if np.isfinite(p_full) else None
 
    return out
def check_dataset_confounding(
    adata,
    tissue: str,
    zinc_genes: list[str],
    gene_signs: dict[str, float] | None,
    signature_label: str,
    min_patients_per_group: int = 3,
) -> dict:
    """
    Check whether the MS-vs-HC composite zinc/MT signature result is being
    driven by, or differs meaningfully between, dataset of origin
    (GSE133028 vs GSE138266) -- addresses the harmonization /
    study-of-origin confounding concern, which UMAP visualization alone
    cannot rule out for a specific downstream statistical result.
 
    Two complementary checks:
 
    1. STRATIFIED: rerun the same patient-level composite signature test
       WITHIN each dataset separately. If the direction/significance
       pattern is wildly inconsistent between datasets, the pooled result
       may be driven disproportionately by one study rather than reflecting
       a shared disease effect. Exploratory (not corrected for multiple
       testing) -- a diagnostic, not a replacement for the main test.
 
    2. FORMAL INTERACTION TEST: OLS on patient-level pseudobulk signature
       values, signature ~ C(condition) * C(dataset). A significant
       interaction term is direct evidence that the disease effect differs
       by dataset, rather than relying on eyeballing two separate p-values.
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
 
    # attach dataset per patient (assumes 1:1 patient -> dataset, the same
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
 
    testable = per_dataset_df.dropna(subset=["pvalue"])
    directions_agree = testable["median_diff"].apply(np.sign).nunique() <= 1 if len(testable) > 1 else None
    if directions_agree is False:
        print("    [WARNING] Direction of the MS-vs-HC effect is INCONSISTENT across datasets "
              "-- the pooled result may not reflect a shared disease effect. Investigate before "
              "reporting the pooled signature test as dataset-independent.")
    elif directions_agree is True:
        print("    Direction is consistent across all testable datasets.")
 
    interaction_p = np.nan
    if pb["dataset"].nunique() >= 2 and pb["_group"].nunique() >= 2:
        model_df = pb.dropna(subset=["signature", "dataset", "_group"]).rename(columns={"_group": "condition"})
        try:
            ols = smf.ols("signature ~ C(condition) * C(dataset)", data=model_df).fit()
            interaction_terms = [t for t in ols.pvalues.index if ":" in t]
            if interaction_terms:
                interaction_p = ols.pvalues[interaction_terms[0]]
                verdict = ("SIGNIFICANT interaction -- effect differs by dataset"
                           if interaction_p < 0.05
                           else "no significant interaction -- effect is consistent across datasets")
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
    print(f"    {tissue} CXCR3+ patients: {n_hc} HC, {n_ms} MS ({sub.n_obs} cells total)")
 
    # Cells-per-patient distribution, split by condition. This determines
    # how noisy each patient's pseudobulk value is -- fewer CXCR3+ cells per
    # patient means each patient's aggregated value averages over less
    # signal, a direct and checkable contributor to weaker per-gene
    # significance in one tissue versus another (independent of whether the
    # underlying patient-level effect size also differs).
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
 
    # Each patient belongs to exactly ONE condition: unpaired, between-patient
    # comparison. pseudobulk_de aggregates cells to one value per patient
    # first, so the test runs on ~n_patients values, not ~n_cells.
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
    #   'padj'            -- genome-wide, for the volcano plot, which is an
    #                        exploratory scan across the whole tested set.
    #   'padj_zinc_panel' -- BH restricted to the curated zinc/MT genes. The
    #                        panel is small, pre-specified and
    #                        hypothesis-driven (not discovered by screening
    #                        the transcriptome), so correcting it against
    #                        thousands of unrelated background genes is the
    #                        wrong correction family for that narrower claim.
    #                        Used for the bar-chart significance stars.
    zinc_mask = df["is_zinc"].values
    df["padj_zinc_panel"] = np.nan
    if zinc_mask.sum() > 0:
        df.loc[zinc_mask, "padj_zinc_panel"] = multipletests(
            df.loc[zinc_mask, "pvalue"].fillna(1.0), method="fdr_bh"
        )[1]
 
    return df
 
 
def _select_genes_for_bars(df: pd.DataFrame, zinc_genes: list[str]) -> tuple[list[str], list[str]]:
    """
    Bar-chart gene list = the FULL curated zinc/MT panel (every gene that was
    actually tested), sorted by log2FC descending. Shows every curated gene
    rather than a top-N subset, ordered by effect size rather than config
    listing order. Any curated gene missing from `df` entirely is returned
    separately so the caller can report it rather than silently omit it.
    """
    tested = df[df["gene"].isin(zinc_genes)].copy()
    tested = tested.sort_values("log2FC", ascending=False)
    genes_present = tested["gene"].tolist()
    genes_missing = [g for g in zinc_genes if g not in set(genes_present)]
    return genes_present, genes_missing
 
 
def _draw_volcano(ax: plt.Axes, df: pd.DataFrame, label: str, zinc_color: str) -> None:
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
            s=32, color=zinc_color, alpha=0.4,
            edgecolors="black", linewidths=0.4,
            label="Zinc/MT (ns)",
        )
    if not zinc_sig.empty:
        ax.scatter(
            zinc_sig["log2FC"], zinc_sig["-log10padj"],
            s=52, color=zinc_color, alpha=0.9,
            edgecolors="black", linewidths=0.6,
            label=f"Zinc/MT (padj < {SIG_PADJ_THRESHOLD})",
        )
 
    ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.axhline(-np.log10(SIG_PADJ_THRESHOLD), color="grey", linestyle=":", linewidth=0.8)
    ax.set_xlabel("log2FC (MS vs HC)")
    ax.set_ylabel("−log10(padj)")
    ax.legend(frameon=False, loc="upper left", fontsize=10)
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
 
def _sig_stars(p_val: float) -> str:
    if pd.isna(p_val) or not np.isfinite(p_val):
        return "n/a"
    if p_val < 0.001:
        return "***"
    if p_val < 0.01:
        return "**"
    if p_val < SIG_PADJ_THRESHOLD:
        return "*"
    return "ns"
 
def _draw_bars(
    ax: plt.Axes,
    df: pd.DataFrame,
    genes: list[str],
    label: str,
    gene_signs: dict[str, float],
    ylabel: str,
    outlier_cap: float | None = None,
    show_legend: bool = False,
) -> None:
    """
    Per-gene log2FC bars, coloured by gene class.
 
    No title: the tissue and sample sizes live in `ylabel` instead, which
    keeps the panel compact and avoids a two-line heading above every plot.
    """
    if not genes:
        ax.text(0.5, 0.5, "No zinc/metallothionein genes available",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color="grey")
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
 
    classes = [_gene_class(g, gene_signs) for g in genes]
    bar_colors = [CLASS_COLORS[c] for c in classes]
 
    fc_plot = fc_raw.copy()
    capped_mask = np.zeros(len(fc_raw), dtype=bool)
    if outlier_cap is not None:
        capped_mask = np.abs(fc_raw) > outlier_cap
        fc_plot[capped_mask] = np.sign(fc_raw[capped_mask]) * outlier_cap * 0.9
 
    for i in range(len(genes)):
        hatch = "//" if capped_mask[i] else None
        ax.bar(x[i], fc_plot[i], width=0.65, color=bar_colors[i],
               edgecolor=INK, linewidth=0.7, hatch=hatch)
 
    ax.axhline(0, color=LINE, linestyle=(0, (2, 2)), linewidth=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(LINE)
    ax.grid(axis="y", color=LINE, linewidth=0.5, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
 
    for i in np.where(capped_mask)[0]:
        true_val = fc_raw[i]
        cap_val = fc_plot[i]
        direction = np.sign(cap_val)
        ax.annotate(
            f"{true_val:.1f}",
            xy=(x[i], cap_val),
            xytext=(x[i], cap_val + direction * abs(cap_val) * 0.08),
            ha="center", va="bottom" if direction > 0 else "top",
            fontsize=11, color=bar_colors[i], fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", lw=1, color=bar_colors[i]),
            clip_on=False,
        )
 
    # Stars above positive bars, below negative bars. Previously every star
    # was placed at max(fc, 0) + offset, so stars on downregulated genes sat
    # at the zero line, visually detached from the bar they marked.
    y_range = max(np.abs(fc_plot).max(), 0.5)
    y_offset = y_range * 0.05
    for i in range(len(genes)):
        sig = _sig_label(padj_v[i])
        if sig is None:
            continue
        if fc_plot[i] >= 0:
            ax.text(i, fc_plot[i] + y_offset, sig, ha="center", va="bottom",
                    fontsize=14, fontweight="bold", color=INK)
        else:
            ax.text(i, fc_plot[i] - y_offset, sig, ha="center", va="top",
                    fontsize=14, fontweight="bold", color=INK)
 
    ax.margins(y=0.16)
    ax.set_xticks(x)
    ax.set_xticklabels(genes, rotation=45, ha="right", fontsize=11)
    # Tick labels: dark teal for both transporter classes (the light ZnT
    # tint is too faint as text on white) and red for MTs, so the
    # transporter/MT split still reads from the axis alone.
    for tick_label, cls in zip(ax.get_xticklabels(), classes):
        tick_label.set_color(MT_COLOR if cls == "MT" else TICK_TEAL)
    ax.set_ylabel(ylabel)
 
    if show_legend:
        present = [c for c in ("ZIP", "ZnT", "MT") if c in set(classes)]
        ax.legend(
            handles=[Patch(facecolor=CLASS_COLORS[c], edgecolor=INK, label=CLASS_LABELS[c])
                     for c in present],
            frameon=False, loc="lower left", fontsize=11,
        )
 
    _panel_letter(ax, label)
 
def draw_patient_bar(ax, sig_test_result, tissue_label, letter, signed: bool = True):
    """
    Box plot + jittered patient dots of PATIENT-LEVEL pseudobulk composite
    signature values: median/IQR/whiskers per group, with each patient's own
    value as a dot.
 
    Box plot rather than violin: with n as low as 6-8 patients per group, a
    kernel density estimate is mostly smoothing artefact and can suggest
    structure (bimodality, skew) that isn't there. A box plot makes no such
    shape assumption -- median/IQR/range come directly from the points,
    which are all shown regardless.
 
    This shows exactly the data the Mann-Whitney test operates on, NOT the
    thousands of raw cells a per-cell violin would show -- a per-cell median
    can tie across groups due to a shared zero-inflated floor while
    patient-level means differ clearly.
 
    The test result is drawn as a bracket with stars, matching the
    convention used for the per-gene bars in panels A/C, so significance is
    read the same way everywhere in the figure. The exact p-value goes to
    stdout and into the exported CSV rather than onto the plot.
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
    group_colors = [HC_COLOR, MS_COLOR]
    group_pales = [HC_PALE, MS_PALE]
    rng = np.random.default_rng(42)
    box_data = [per_patient.loc[per_patient["_group"] == g, "signature"].values for g in groups]
 
    bp = ax.boxplot(
        box_data, positions=[1, 2], widths=0.42,
        showfliers=False, patch_artist=True,
        medianprops=dict(color=INK, linewidth=2.2),
        boxprops=dict(edgecolor=INK, linewidth=1.3),
        whiskerprops=dict(color=INK, linewidth=1.1),
        capprops=dict(color=INK, linewidth=1.1),
    )
    # Colour each box by group (HC vs MS): that is the comparison this panel
    # tests, so that is the distinction colour should carry. Tissue is
    # conveyed by which panel this is.
    for patch, edge_c, face_c in zip(bp["boxes"], group_colors, group_pales):
        patch.set_facecolor(face_c)
        patch.set_edgecolor(edge_c)
 
    for i, vals in enumerate(box_data, start=1):
        if len(vals) == 0:
            continue
        jitter = rng.normal(0, 0.055, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, color=INK, s=32,
                   alpha=0.72, edgecolors="white", linewidths=0.6, zorder=10)
 
    if np.isfinite(p):
        print(f"[{tissue_label}] PATIENT-LEVEL box plot: n={n_hc} HC / {n_ms} MS patients "
              f"(each dot = one patient's pseudobulk mean), Mann-Whitney p={p:.3e} "
              f"-> {_sig_stars(p)}")
    else:
        print(f"[{tissue_label}] n={n_hc} HC / {n_ms} MS -- too few to test")
 
    # Significance bracket spanning the two groups.
    all_vals = np.concatenate([v for v in box_data if len(v)])
    if all_vals.size:
        y_top = all_vals.max()
        y_bot = all_vals.min()
        span = max(y_top - y_bot, 1e-6)
        bar_y = y_top + span * 0.10
        tick_h = span * 0.035
        ax.plot([1, 1, 2, 2],
                [bar_y, bar_y + tick_h, bar_y + tick_h, bar_y],
                lw=1.4, color=INK, clip_on=False, zorder=12)
        stars = _sig_stars(p)
        # "ns"/"n/a" are words, not glyphs, so they need a smaller size and
        # a little more clearance than the asterisks.
        is_word = stars in ("ns", "n/a")
        ax.text(1.5, bar_y + tick_h + (span * 0.02 if is_word else 0),
                stars, ha="center", va="bottom",
                fontsize=12 if is_word else 17,
                fontweight="bold", color=INK, clip_on=False, zorder=12)
        ax.set_ylim(y_bot - span * 0.14, bar_y + tick_h + span * 0.20)
 
    ax.set_xlim(0.4, 2.6)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"HC\n(n={n_hc})", f"MS\n(n={n_ms})"], fontsize=12)
    for tick_label, c in zip(ax.get_xticklabels(), group_colors):
        tick_label.set_color(c)
        tick_label.set_fontweight("bold")
 
    score_label = "signed" if signed else "unsigned"
    ax.set_ylabel(f"Mean zinc/MT signature\n({score_label}, per patient)")
 
    ax.axhline(0, color=LINE, linestyle=(0, (2, 2)), linewidth=0.9, zorder=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(LINE)
    ax.tick_params(colors=INK)
    ax.grid(axis="y", color=LINE, linewidth=0.5, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    _panel_letter(ax, letter)
 
 
def _save_single_volcano(df, label, zinc_color, out_png, out_pdf):
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.5))
    _draw_volcano(ax, df, label=label, zinc_color=zinc_color)
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
    print("Gene classes for colouring: "
          f"{ {g: _gene_class(g, gene_signs) for g in zinc_genes} }")
 
    print("\nRunning patient-level DE: PB CXCR3+ MS vs HC …")
    df_pb = _run_de(adata, tissue="PB", zinc_genes=zinc_genes, min_pct=0.10)
    print(f"  {df_pb.shape[0]} genes in PB DE table")
 
    print("\nRunning patient-level DE: CSF CXCR3+ MS vs HC …")
    df_csf = _run_de(adata, tissue="CSF", zinc_genes=zinc_genes, min_pct=0.15)
    print(f"  {df_csf.shape[0]} genes in CSF DE table")
 
    df_sig_pb = compute_cell_signature(adata, zinc_genes, tissue="PB", gene_signs=gene_signs)
    df_sig_csf = compute_cell_signature(adata, zinc_genes, tissue="CSF", gene_signs=gene_signs)
    df_sig_pb.to_csv(RESULTS_DIR / "Fig2_PB_CXCR3pos_zinc_signature_per_cell.csv", index=False)
    df_sig_csf.to_csv(RESULTS_DIR / "Fig2_CSF_CXCR3pos_zinc_signature_per_cell.csv", index=False)
 
    # Patient-level signature test -- this is what's reported as the panel
    # C/D p-value, replacing the per-cell Mann-Whitney test.
    sig_test_pb = patient_signature_test(adata, zinc_genes, tissue="PB", gene_signs=gene_signs)
    sig_test_csf = patient_signature_test(adata, zinc_genes, tissue="CSF", gene_signs=gene_signs)
    pd.DataFrame([{k: v for k, v in sig_test_pb.items() if k != "per_patient_values"}]).to_csv(
        RESULTS_DIR / "Fig2_PB_CXCR3pos_zinc_signature_patient_level_test.csv", index=False
    )
    pd.DataFrame([{k: v for k, v in sig_test_csf.items() if k != "per_patient_values"}]).to_csv(
        RESULTS_DIR / "Fig2_CSF_CXCR3pos_zinc_signature_patient_level_test.csv", index=False
    )
 
    robust_csf = signature_outlier_robustness(sig_test_csf, tissue="CSF")
    robust_pb = signature_outlier_robustness(sig_test_pb, tissue="PB")
    
    robust_csf["per_patient_df"].to_csv(
        RESULTS_DIR / "Fig2_CSF_CXCR3pos_signature_leave_one_out.csv", index=False
    )
    robust_pb["per_patient_df"].to_csv(
        RESULTS_DIR / "Fig2_PB_CXCR3pos_signature_leave_one_out.csv", index=False
    )
    pd.DataFrame([
        {k: v for k, v in r.items() if k != "per_patient_df"}
        for r in (robust_csf, robust_pb)
    ]).to_csv(RESULTS_DIR / "Fig2_signature_outlier_robustness_summary.csv", index=False)
    # Dataset-of-origin confounding check: does the pooled MS-vs-HC
    # signature result hold within each dataset separately?
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
    # alongside the sign-corrected one for transparency: the sign scheme is
    # a priori and biology-based, not derived from this comparison's own DE
    # result, but a reader should still be able to see the plain average.
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
        "\nSigned vs unsigned signature p-values (sanity check -- both should point the "
        "same direction even if only one reaches significance):\n"
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
 
    # Bar row taller than the box-plot row: the bar panels carry 14 genes
    # each and need the vertical space; the box plots carry two groups.
    #
    # To lead with CSF instead of PB, swap the two _draw_bars calls and the
    # two draw_patient_bar calls below (and their A/B/C/D letters).

    fig, axes = plt.subplots(
        2, 2, figsize=(13, 9),
        gridspec_kw=dict(width_ratios=[1.75, 1.0]),
    )
    axA, axB = axes[0]   # CSF: bars, composite
    axC, axD = axes[1]   # PB:  bars, composite
 
    n_hc_pb = sig_test_pb.get("n_patients_HC", "?")
    n_ms_pb = sig_test_pb.get("n_patients_MS", "?")
    n_hc_csf = sig_test_csf.get("n_patients_HC", "?")
    n_ms_csf = sig_test_csf.get("n_patients_MS", "?")
 
    _draw_bars(
        axA, df_csf, csf_bar_genes, label="A",
        gene_signs=gene_signs,
        ylabel=f"log2FC, MS vs HC\nCSF (n={n_hc_csf} HC, {n_ms_csf} MS)",
        show_legend=True,
    )
    draw_patient_bar(axB, sig_test_csf, tissue_label="CSF", letter="B")
 
    _draw_bars(
        axC, df_pb, pb_bar_genes, label="C",
        gene_signs=gene_signs,
        ylabel=f"log2FC, MS vs HC\nPB (n={n_hc_pb} HC, {n_ms_pb} MS)",
    )
    draw_patient_bar(axD, sig_test_pb, tissue_label="PB", letter="D")
 
    fig.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.12,
                        wspace=0.30, hspace=0.60)
 
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
        df_pb, label="A", zinc_color=SCATTER_COLOR_ZINC_PB,
        out_png=supp_dir / "Fig2_supp_PB_volcano.png",
        out_pdf=supp_dir / "Fig2_supp_PB_volcano.pdf",
    )
    _save_single_volcano(
        df_csf, label="B", zinc_color=SCATTER_COLOR_ZINC_CSF,
        out_png=supp_dir / "Fig2_supp_CSF_volcano.png",
        out_pdf=supp_dir / "Fig2_supp_CSF_volcano.pdf",
    )
 
    # Supplementary: unsigned (plain-mean) composite signature, side by side
    # with the sign-corrected version used in the main figure, so a reader
    # can check that sign-correction isn't producing a direction the plain
    # average disagrees with.
    fig_supp_sig, (axC_u, axD_u) = plt.subplots(1, 2, figsize=(10, 4.5))
    draw_patient_bar(axC_u, sig_test_pb_unsigned, tissue_label="PB", letter="A", signed=False)
    draw_patient_bar(axD_u, sig_test_csf_unsigned, tissue_label="CSF", letter="B", signed=False)
    fig_supp_sig.suptitle(
        "Unsigned (plain-mean) zinc/MT signature — compare with main Fig 2C/D",
        fontsize=12, y=1.03,
    )
    fig_supp_sig.tight_layout()
    fig_supp_sig.savefig(supp_dir / "Fig2_supp_unsigned_signature.png", dpi=300, bbox_inches="tight")
    fig_supp_sig.savefig(supp_dir / "Fig2_supp_unsigned_signature.pdf", bbox_inches="tight")
    plt.close(fig_supp_sig)
 
    df_pb.to_csv(out_dir / "Fig2_PB_CXCR3pos_MS_vs_HC_full_DE.csv", index=False)
    df_csf.to_csv(out_dir / "Fig2_CSF_CXCR3pos_MS_vs_HC_full_DE.csv", index=False)
    pd.DataFrame({"gene": pb_bar_genes}).to_csv(out_dir / "Fig2_PB_CXCR3pos_zinc_bar_genes.csv", index=False)
    pd.DataFrame({"gene": csf_bar_genes}).to_csv(out_dir / "Fig2_CSF_CXCR3pos_zinc_bar_genes.csv", index=False)
 
    print(f"\nSaved: {out_png}")
    print(f"Saved: {out_pdf}")
    print(f"Saved supplementary volcanoes and unsigned-signature comparison in: {supp_dir}")
 
 
if __name__ == "__main__":
    make_fig2_combined()
 



