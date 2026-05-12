import io, re, zipfile
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

st.set_page_config(page_title="Interference Analysis App", layout="wide")
st.title("Interference Analysis: Control vs Interference")
st.caption("Compare analytes between control and interference conditions with assumption checks, % shift, CIs, and downloadable outputs.")

ID_HINTS = ["batch_id", "bloodSampleId", "bloodSampleID", "sampleId", "deviceId", "serialNumber", "patientUserId"]
DEFAULT_ANALYTE_HINTS = ["WBC", "RBC", "HGB", "HCT", "MCV", "MCH", "MCHC", "PLT", "RDW_SD", "RDW_CV", "NEUT", "LYMPH", "MONO", "EOS", "BASO", "NEUT_per", "LYMPH_per", "MONO_per", "EOS_per", "BASO_per"]

# ------------------------- helpers -------------------------
def read_upload(uploaded):
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded)
    return pd.read_excel(uploaded, engine="openpyxl")

def numeric_cols(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        if c in ID_HINTS:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= 3:
            cols.append(c)
    return cols

def find_col(df, candidates):
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None

def extract_pair_key_from_sample(x: str, condition_value: str = "") -> str:
    """Extracts comparable replicate/sample key, e.g. APT-...-Glu-C1 and APT-...-Glu-I1 -> 1.
    Falls back to last numeric token, else full value.
    """
    s = str(x)
    m = re.search(r"[-_](?:C|I|CTRL|CONTROL|INT|INTERFERENCE)?\s*(\d+)\s*$", s, flags=re.I)
    if m:
        return m.group(1)
    nums = re.findall(r"(\d+)", s)
    return nums[-1] if nums else s

def mad_sd(x):
    x = np.asarray(pd.to_numeric(pd.Series(x), errors="coerce").dropna(), dtype=float)
    if len(x) == 0: return np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad

def ci_quantiles(vals, alpha=0.05):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, np.nan
    return np.percentile(vals, [100*alpha/2, 100*(1-alpha/2)])

def bootstrap_unpaired(x, y, stat_func, n_boot=2000, seed=1):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan, np.nan
    out = []
    for _ in range(n_boot):
        xb = rng.choice(x, size=len(x), replace=True)
        yb = rng.choice(y, size=len(y), replace=True)
        out.append(stat_func(xb, yb))
    return ci_quantiles(out)

def bootstrap_paired(diffs, stat_func=np.mean, n_boot=2000, seed=1):
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) < 2:
        return np.nan, np.nan
    out = []
    for _ in range(n_boot):
        db = rng.choice(diffs, size=len(diffs), replace=True)
        out.append(stat_func(db))
    return ci_quantiles(out)

def permutation_pvalue_unpaired(x, y, n_perm=5000, seed=1):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan
    obs = abs(np.mean(y) - np.mean(x))
    z = np.concatenate([x, y])
    nx = len(x)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(z)
        stat = abs(np.mean(z[nx:]) - np.mean(z[:nx]))
        if stat >= obs:
            count += 1
    return (count + 1) / (n_perm + 1)

def benjamini_hochberg(pvals):
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    mask = np.isfinite(p)
    pm = p[mask]
    if len(pm) == 0:
        return q
    order = np.argsort(pm)
    ranked = pm[order]
    m = len(pm)
    adj = ranked * m / (np.arange(1, m+1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    qmask = np.empty_like(pm)
    qmask[order] = adj
    q[mask] = qmask
    return q

def assumption_checks(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    residuals = np.concatenate([x - np.mean(x), y - np.mean(y)]) if len(x) and len(y) else np.array([])
    shapiro_p = stats.shapiro(residuals).pvalue if 3 <= len(residuals) <= 5000 else np.nan
    lev_mean_p = stats.levene(x, y, center="mean").pvalue if len(x) >= 2 and len(y) >= 2 else np.nan
    brown_p = stats.levene(x, y, center="median").pvalue if len(x) >= 2 and len(y) >= 2 else np.nan
    return shapiro_p, lev_mean_p, brown_p

def compare_unpaired(x, y, control_label, int_label, n_boot, do_boot, seed):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    row = {}
    row[f"n_{control_label}"] = len(x); row[f"n_{int_label}"] = len(y)
    row[f"mean_{control_label}"] = np.mean(x) if len(x) else np.nan
    row[f"sd_{control_label}"] = np.std(x, ddof=1) if len(x) > 1 else np.nan
    row[f"median_{control_label}"] = np.median(x) if len(x) else np.nan
    row[f"iqr_{control_label}"] = np.subtract(*np.percentile(x, [75,25])) if len(x) else np.nan
    row[f"mean_{int_label}"] = np.mean(y) if len(y) else np.nan
    row[f"sd_{int_label}"] = np.std(y, ddof=1) if len(y) > 1 else np.nan
    row[f"median_{int_label}"] = np.median(y) if len(y) else np.nan
    row[f"iqr_{int_label}"] = np.subtract(*np.percentile(y, [75,25])) if len(y) else np.nan
    row["mean_difference_I_minus_C"] = row[f"mean_{int_label}"] - row[f"mean_{control_label}"]
    row["median_difference_I_minus_C"] = row[f"median_{int_label}"] - row[f"median_{control_label}"]
    row["percent_shift_mean"] = 100 * row["mean_difference_I_minus_C"] / row[f"mean_{control_label}"] if row[f"mean_{control_label}"] != 0 else np.nan
    row["percent_shift_median"] = 100 * row["median_difference_I_minus_C"] / row[f"median_{control_label}"] if row[f"median_{control_label}"] != 0 else np.nan
    sh, lev, bf = assumption_checks(x, y)
    row["shapiro_wilk_p_residuals"] = sh
    row["levene_mean_p"] = lev
    row["brown_forsythe_median_p"] = bf
    row["residuals_normal"] = bool(sh >= 0.05) if np.isfinite(sh) else None
    row["equal_variance_levene"] = bool(lev >= 0.05) if np.isfinite(lev) else None
    row["equal_variance_brown_forsythe"] = bool(bf >= 0.05) if np.isfinite(bf) else None
    try:
        row["student_t_p_equal_var"] = stats.ttest_ind(x, y, equal_var=True).pvalue
    except Exception: row["student_t_p_equal_var"] = np.nan
    try:
        row["welch_t_p_primary"] = stats.ttest_ind(x, y, equal_var=False).pvalue
    except Exception: row["welch_t_p_primary"] = np.nan
    try:
        row["mann_whitney_p_robust"] = stats.mannwhitneyu(x, y, alternative="two-sided").pvalue
    except Exception: row["mann_whitney_p_robust"] = np.nan
    row["permutation_p_mean_diff"] = permutation_pvalue_unpaired(x, y, seed=seed) if len(x) >= 2 and len(y) >= 2 else np.nan
    if do_boot:
        lo, hi = bootstrap_unpaired(x, y, lambda xb,yb: np.mean(yb)-np.mean(xb), n_boot=n_boot, seed=seed)
        row["mean_difference_95CI_low"] = lo; row["mean_difference_95CI_high"] = hi
        lo, hi = bootstrap_unpaired(x, y, lambda xb,yb: 100*(np.mean(yb)-np.mean(xb))/np.mean(xb) if np.mean(xb)!=0 else np.nan, n_boot=n_boot, seed=seed+11)
        row["percent_shift_mean_95CI_low"] = lo; row["percent_shift_mean_95CI_high"] = hi
        lo, hi = bootstrap_unpaired(x, y, lambda xb,yb: np.median(yb)-np.median(xb), n_boot=n_boot, seed=seed+22)
        row["median_difference_95CI_low"] = lo; row["median_difference_95CI_high"] = hi
        lo, hi = bootstrap_unpaired(x, y, lambda xb,yb: 100*(np.median(yb)-np.median(xb))/np.median(xb) if np.median(xb)!=0 else np.nan, n_boot=n_boot, seed=seed+33)
        row["percent_shift_median_95CI_low"] = lo; row["percent_shift_median_95CI_high"] = hi
    return row

def compare_paired(df_sub, analyte, cond_col, control_val, int_val, pair_cols, n_boot, do_boot, seed):
    work = df_sub[pair_cols + [cond_col, analyte]].copy()
    work[analyte] = pd.to_numeric(work[analyte], errors="coerce")
    wide = work.pivot_table(index=pair_cols, columns=cond_col, values=analyte, aggfunc="mean")
    if control_val not in wide.columns or int_val not in wide.columns:
        return None, pd.DataFrame()
    paired = wide[[control_val, int_val]].dropna().reset_index()
    x = paired[control_val].to_numpy(dtype=float)
    y = paired[int_val].to_numpy(dtype=float)
    diffs = y - x
    row = compare_unpaired(x, y, str(control_val), str(int_val), n_boot, do_boot, seed)
    row["paired_n"] = len(diffs)
    row["paired_mean_difference_I_minus_C"] = np.mean(diffs) if len(diffs) else np.nan
    row["paired_percent_shift_mean"] = 100*np.mean(diffs)/np.mean(x) if len(diffs) and np.mean(x)!=0 else np.nan
    try: row["paired_t_p"] = stats.ttest_rel(x, y).pvalue
    except Exception: row["paired_t_p"] = np.nan
    try: row["wilcoxon_p"] = stats.wilcoxon(diffs).pvalue if len(diffs) >= 2 else np.nan
    except Exception: row["wilcoxon_p"] = np.nan
    if do_boot and len(diffs) >= 2:
        lo, hi = bootstrap_paired(diffs, np.mean, n_boot, seed+44)
        row["paired_mean_difference_95CI_low"] = lo; row["paired_mean_difference_95CI_high"] = hi
    return row, paired

def make_download_zip(result_tables: Dict[str, pd.DataFrame], meta_text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("README_summary.txt", meta_text)
        for name, table in result_tables.items():
            zf.writestr(f"{name}.csv", table.to_csv(index=False).encode("utf-8"))
        excel_buf = io.BytesIO()
        with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
            for name, table in result_tables.items():
                sheet = name[:31]
                table.to_excel(writer, sheet_name=sheet, index=False)
        zf.writestr("interference_analysis_results.xlsx", excel_buf.getvalue())
    return buf.getvalue()

# ------------------------- UI -------------------------
uploaded = st.file_uploader("Upload Excel or CSV", type=["xlsx", "xls", "csv"])
if uploaded is None:
    st.info("Upload your interference file to begin. Expected design: condition column with control/interference rows, optional deviceId, and selected numeric analytes.")
    st.stop()

try:
    df = read_upload(uploaded)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

st.subheader("1) Data preview and detected columns")
st.write(f"Rows: **{len(df)}** | Columns: **{len(df.columns)}**")
st.dataframe(df.head(30), use_container_width=True)

condition_guess = find_col(df, ["Condition", "condition", "Group", "group"])
device_guess = find_col(df, ["deviceId", "Device", "device", "serialNumber"])
batch_guess = find_col(df, ["batch_id", "batchId", "Batch"])
sample_guess = find_col(df, ["bloodSampleId", "sampleId", "SampleID", "sample"])

c1, c2, c3, c4 = st.columns(4)
with c1:
    cond_col = st.selectbox("Condition column", options=list(df.columns), index=list(df.columns).index(condition_guess) if condition_guess in df.columns else 0)
with c2:
    device_col = st.selectbox("Device column (optional)", options=["None"] + list(df.columns), index=(["None"] + list(df.columns)).index(device_guess) if device_guess in df.columns else 0)
with c3:
    batch_col = st.selectbox("Batch ID column", options=["None"] + list(df.columns), index=(["None"] + list(df.columns)).index(batch_guess) if batch_guess in df.columns else 0)
with c4:
    sample_col = st.selectbox("Sample/replicate ID column", options=["None"] + list(df.columns), index=(["None"] + list(df.columns)).index(sample_guess) if sample_guess in df.columns else 0)

conditions = sorted([str(x) for x in df[cond_col].dropna().unique()])
if len(conditions) < 2:
    st.error("Need at least two condition values, e.g. c and i.")
    st.stop()

c1, c2 = st.columns(2)
with c1:
    control_val = st.selectbox("Control/reference condition", options=conditions, index=0)
with c2:
    int_val = st.selectbox("Interference/test condition", options=conditions, index=1 if len(conditions)>1 else 0)

numcols = numeric_cols(df)
def_default = [c for c in DEFAULT_ANALYTE_HINTS if c in numcols]
if not def_default: def_default = numcols[:10]
analytes = st.multiselect("Analyte columns to compare", options=numcols, default=def_default)

st.subheader("2) Analysis settings")
c1, c2, c3, c4 = st.columns(4)
with c1:
    device_mode = st.selectbox("Device handling", ["Pool all devices", "Analyze each device separately + pooled"])
with c2:
    paired_mode = st.checkbox("Try paired analysis using sample key and/or device", value=True)
with c3:
    do_boot = st.checkbox("Bootstrap 95% CIs", value=True)
with c4:
    n_boot = st.number_input("Bootstrap iterations", min_value=200, max_value=20000, value=2000, step=200)

c1, c2 = st.columns(2)
with c1:
    alpha = st.number_input("Significance alpha", min_value=0.001, max_value=0.2, value=0.05, step=0.01)
with c2:
    seed = st.number_input("Random seed", min_value=1, max_value=999999, value=123, step=1)

st.markdown("""
**Primary test recommendation:** Welch two-sample t-test for mean shift because it does not assume equal variances.  
**Robust/non-parametric check:** Mann-Whitney U for distribution/median-like shift.  
**If paired by replicate/sample/device:** paired t-test + Wilcoxon signed-rank are also reported.
""")

if not analytes:
    st.warning("Select at least one analyte.")
    st.stop()

run = st.button("Run interference analysis", type="primary")
if not run:
    st.stop()

# prepare data
df2 = df.copy()
df2[cond_col] = df2[cond_col].astype(str)
df2 = df2[df2[cond_col].isin([control_val, int_val])].copy()
if device_col == "None":
    df2["__device__"] = "pooled"
    device_col_use = "__device__"
else:
    device_col_use = device_col
    df2[device_col_use] = df2[device_col_use].astype(str)

if sample_col != "None":
    df2["__pair_key__"] = df2.apply(lambda r: extract_pair_key_from_sample(r[sample_col], r[cond_col]), axis=1)
else:
    df2["__pair_key__"] = df2.groupby(cond_col).cumcount().astype(str)

scopes = [("pooled_all_devices", df2.copy())]
if device_mode.startswith("Analyze each"):
    for dev, sdf in df2.groupby(device_col_use):
        scopes.append((f"device_{dev}", sdf.copy()))

summary_rows = []
paired_rows = []
paired_detail_tables = {}
for scope_name, sdf in scopes:
    for analyte in analytes:
        work = sdf[[cond_col, analyte, device_col_use, "__pair_key__"] + ([batch_col] if batch_col != "None" else []) + ([sample_col] if sample_col != "None" else [])].copy()
        work[analyte] = pd.to_numeric(work[analyte], errors="coerce")
        x = work.loc[work[cond_col] == control_val, analyte].dropna().values
        y = work.loc[work[cond_col] == int_val, analyte].dropna().values
        row = compare_unpaired(x, y, str(control_val), str(int_val), int(n_boot), bool(do_boot), int(seed))
        row.update({"scope": scope_name, "analyte": analyte, "control_condition": control_val, "interference_condition": int_val})
        row["primary_test"] = "Welch t-test"
        row["recommended_p_value"] = row.get("welch_t_p_primary", np.nan)
        row["significant_at_alpha"] = bool(row["recommended_p_value"] < alpha) if np.isfinite(row["recommended_p_value"]) else None
        summary_rows.append(row)
        if paired_mode:
            pair_cols = ["__pair_key__"] if scope_name == "pooled_all_devices" else ["__pair_key__", device_col_use]
            prow, pdetail = compare_paired(sdf, analyte, cond_col, control_val, int_val, pair_cols, int(n_boot), bool(do_boot), int(seed))
            if prow is not None:
                prow.update({"scope": scope_name, "analyte": analyte, "control_condition": control_val, "interference_condition": int_val})
                paired_rows.append(prow)
                paired_detail_tables[f"paired_details_{scope_name}_{analyte}"[:60]] = pdetail

summary = pd.DataFrame(summary_rows)
if not summary.empty:
    summary["welch_t_q_BH_FDR_within_run"] = benjamini_hochberg(summary["welch_t_p_primary"].values)
    summary["mann_whitney_q_BH_FDR_within_run"] = benjamini_hochberg(summary["mann_whitney_p_robust"].values)
paired_summary = pd.DataFrame(paired_rows)
if not paired_summary.empty and "paired_t_p" in paired_summary.columns:
    paired_summary["paired_t_q_BH_FDR_within_run"] = benjamini_hochberg(paired_summary["paired_t_p"].values)
    paired_summary["wilcoxon_q_BH_FDR_within_run"] = benjamini_hochberg(paired_summary["wilcoxon_p"].values)

counts = df2.groupby([cond_col, device_col_use], dropna=False).size().reset_index(name="n_rows")

st.subheader("3) Results: pooled/all devices primary table")
pooled = summary[summary["scope"] == "pooled_all_devices"].copy()
st.dataframe(pooled, use_container_width=True)

st.subheader("4) Paired results, if matched")
if paired_summary.empty:
    st.info("No paired result could be created, or paired analysis was off.")
else:
    st.dataframe(paired_summary, use_container_width=True)

st.subheader("5) Condition/device counts")
st.dataframe(counts, use_container_width=True)

meta = f"""Interference Analysis App\nRows analyzed: {len(df2)}\nCondition column: {cond_col}\nControl condition: {control_val}\nInterference condition: {int_val}\nDevice mode: {device_mode}\nDevice column: {device_col}\nAnalytes: {', '.join(analytes)}\nPrimary test: Welch two-sample t-test; robust check: Mann-Whitney U; optional paired t/Wilcoxon if sample keys match.\nAssumption checks: Shapiro-Wilk on residuals; Levene mean-centered; Brown-Forsythe median-centered.\nEffect sizes: mean difference, median difference, percent shift in mean and median, bootstrap 95% CIs where enabled.\n"""
result_tables = {"interference_summary_unpaired": summary, "paired_summary": paired_summary, "condition_device_counts": counts}
# avoid too many sheets but include details if manageable
for k, v in list(paired_detail_tables.items())[:20]:
    result_tables[k] = v
zip_bytes = make_download_zip(result_tables, meta)

c1, c2, c3 = st.columns(3)
with c1:
    st.download_button("Download unpaired summary CSV", summary.to_csv(index=False), "interference_summary_unpaired.csv", "text/csv")
with c2:
    st.download_button("Download paired summary CSV", paired_summary.to_csv(index=False), "interference_paired_summary.csv", "text/csv")
with c3:
    st.download_button("Download ALL results ZIP", zip_bytes, "interference_analysis_results.zip", "application/zip")

st.success("Done. Download the CSVs or full ZIP above.")
