import io, re, zipfile
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

st.set_page_config(page_title="Interference Analysis App", layout="wide")
st.title("Interference Analysis: Control vs Interference")
st.caption("Compare control vs interference conditions with mean/median shift, assumption checks, optional outlier removal, bootstrap CIs, and downloadable outputs.")

ID_HINTS = ["batch_id", "bloodSampleId", "bloodSampleID", "sampleId", "deviceId", "serialNumber", "patientUserId"]
DEFAULT_ANALYTE_HINTS = ["WBC", "RBC", "HGB", "HCT", "MCV", "MCH", "MCHC", "PLT", "RDW_SD", "RDW_CV", "NEUT", "LYMPH", "MONO", "EOS", "BASO", "NEUT_per", "LYMPH_per", "MONO_per", "EOS_per", "BASO_per"]

# ------------------------- input helpers -------------------------
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
    s = str(x)
    m = re.search(r"[-_](?:C|I|CTRL|CONTROL|INT|INTERFERENCE)?\s*(\d+)\s*$", s, flags=re.I)
    if m:
        return m.group(1)
    nums = re.findall(r"(\d+)", s)
    return nums[-1] if nums else s

# ------------------------- stats helpers -------------------------
def mad_sd(x):
    x = np.asarray(pd.to_numeric(pd.Series(x), errors="coerce").dropna(), dtype=float)
    if len(x) == 0:
        return np.nan
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
    for _ in range(int(n_boot)):
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
    for _ in range(int(n_boot)):
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
    for _ in range(int(n_perm)):
        zp = rng.permutation(z)
        stat = abs(np.mean(zp[nx:]) - np.mean(zp[:nx]))
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

# ------------------------- outlier helpers -------------------------
def _id_value(row, col):
    return row[col] if col and col != "None" and col in row.index else ""

def detect_outliers_one_condition(work: pd.DataFrame, analyte: str, condition_value: str,
                                  method: str, max_remove: int, gcrit: float,
                                  modified_z_threshold: float, robust_interval_z: float,
                                  batch_col: str, sample_col: str, device_col: str,
                                  scope_name: str) -> Tuple[List[int], List[Dict]]:
    """Detect up to max_remove outliers within one condition/analyte/scope. Returns original DataFrame indices."""
    sub = work.loc[work["__condition_str__"] == str(condition_value), [analyte]].copy()
    sub[analyte] = pd.to_numeric(sub[analyte], errors="coerce")
    sub = sub.dropna()
    remaining = list(sub.index)
    removed = []
    logs = []
    if method == "None" or int(max_remove) <= 0 or len(remaining) < 3:
        return removed, logs

    for step in range(int(max_remove)):
        vals = work.loc[remaining, analyte].astype(float)
        if vals.notna().sum() < 3:
            break
        x = vals.to_numpy(dtype=float)

        chosen_idx = None
        metric = np.nan
        threshold = np.nan
        direction = ""
        details = ""

        if method.startswith("Gcrit"):
            mu = float(np.mean(x)); sd = float(np.std(x, ddof=1))
            if not np.isfinite(sd) or sd == 0:
                break
            gvals = np.abs(x - mu) / sd
            k = int(np.argmax(gvals))
            metric = float(gvals[k]); threshold = float(gcrit)
            if metric >= gcrit:
                chosen_idx = remaining[k]
                direction = "high" if x[k] > mu else "low"
                details = f"G={metric:.4g}; mean={mu:.4g}; sd={sd:.4g}; threshold={gcrit}"
            else:
                break

        elif method.startswith("Robust MAD"):
            med = float(np.median(x)); mad = float(np.median(np.abs(x - med)))
            if not np.isfinite(mad) or mad == 0:
                break
            modz = 0.6745 * (x - med) / mad
            k = int(np.argmax(np.abs(modz)))
            metric = float(abs(modz[k])); threshold = float(modified_z_threshold)
            if metric >= modified_z_threshold:
                chosen_idx = remaining[k]
                direction = "high" if x[k] > med else "low"
                details = f"modified_z={modz[k]:.4g}; median={med:.4g}; MAD={mad:.4g}; threshold={modified_z_threshold}"
            else:
                break

        elif method.startswith("95% robust interval"):
            med = float(np.median(x)); rsd = mad_sd(x)
            if not np.isfinite(rsd) or rsd == 0:
                break
            lo = med - robust_interval_z * rsd
            hi = med + robust_interval_z * rsd
            distances = np.maximum(lo - x, x - hi)
            k = int(np.argmax(distances))
            metric = float(distances[k]); threshold = 0.0
            if metric > 0:
                chosen_idx = remaining[k]
                direction = "high" if x[k] > hi else "low"
                details = f"value outside robust interval [{lo:.4g}, {hi:.4g}]; median={med:.4g}; robust_SD={rsd:.4g}; z={robust_interval_z}"
            else:
                break

        if chosen_idx is None:
            break

        row = work.loc[chosen_idx]
        logs.append({
            "scope": scope_name,
            "condition": condition_value,
            "analyte": analyte,
            "removed_order": step + 1,
            "outlier_method": method,
            "row_index": int(chosen_idx) if isinstance(chosen_idx, (int, np.integer)) else str(chosen_idx),
            "batch_id": _id_value(row, batch_col),
            "sample_id": _id_value(row, sample_col),
            "device_id": _id_value(row, device_col),
            "value_removed": row[analyte],
            "direction": direction,
            "outlier_metric": metric,
            "outlier_threshold": threshold,
            "details": details,
        })
        removed.append(chosen_idx)
        remaining.remove(chosen_idx)

    return removed, logs

def apply_outlier_removal_for_analyte(work: pd.DataFrame, analyte: str, control_val: str, int_val: str,
                                      method: str, max_remove_per_condition: int, gcrit: float,
                                      modified_z_threshold: float, robust_interval_z: float,
                                      batch_col: str, sample_col: str, device_col: str,
                                      scope_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    work = work.copy()
    work["__condition_str__"] = work["__condition_str__"].astype(str)
    all_removed = []
    all_logs = []
    for cond in [str(control_val), str(int_val)]:
        removed, logs = detect_outliers_one_condition(
            work, analyte, cond, method, max_remove_per_condition, gcrit,
            modified_z_threshold, robust_interval_z, batch_col, sample_col, device_col, scope_name
        )
        all_removed.extend(removed)
        all_logs.extend(logs)
    cleaned = work.drop(index=list(dict.fromkeys(all_removed)), errors="ignore").copy()
    return cleaned, pd.DataFrame(all_logs)

# ------------------------- comparison helpers -------------------------
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
    row["mad_sd_control"] = mad_sd(x)
    row["mad_sd_interference"] = mad_sd(y)
    sh, lev, bf = assumption_checks(x, y)
    row["shapiro_wilk_p_residuals"] = sh
    row["levene_mean_p"] = lev
    row["brown_forsythe_median_p"] = bf
    row["residuals_normal"] = bool(sh >= 0.05) if np.isfinite(sh) else None
    row["equal_variance_levene"] = bool(lev >= 0.05) if np.isfinite(lev) else None
    row["equal_variance_brown_forsythe"] = bool(bf >= 0.05) if np.isfinite(bf) else None
    try: row["student_t_p_equal_var"] = stats.ttest_ind(x, y, equal_var=True).pvalue
    except Exception: row["student_t_p_equal_var"] = np.nan
    try: row["welch_t_p_primary"] = stats.ttest_ind(x, y, equal_var=False).pvalue
    except Exception: row["welch_t_p_primary"] = np.nan
    try: row["mann_whitney_p_robust"] = stats.mannwhitneyu(x, y, alternative="two-sided").pvalue
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
            used = set()
            for name, table in result_tables.items():
                sheet = re.sub(r"[^A-Za-z0-9_]+", "_", name)[:31] or "sheet"
                base = sheet
                i = 1
                while sheet in used:
                    suffix = f"_{i}"
                    sheet = (base[:31-len(suffix)] + suffix)
                    i += 1
                used.add(sheet)
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
    st.error("Need at least two condition values, e.g. C and I.")
    st.stop()

c1, c2 = st.columns(2)
with c1:
    control_val = st.selectbox("Control/reference condition", options=conditions, index=0)
with c2:
    int_val = st.selectbox("Interference/test condition", options=conditions, index=1 if len(conditions)>1 else 0)

numcols = numeric_cols(df)
def_default = [c for c in DEFAULT_ANALYTE_HINTS if c in numcols]
if not def_default:
    def_default = numcols[:10]
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

st.subheader("3) Optional outlier sensitivity analysis")
outlier_method = st.selectbox(
    "Outlier method for cleaned/sensitivity results",
    [
        "None",
        "Gcrit Grubbs-like: remove largest |value-mean|/SD if >= Gcrit",
        "Robust MAD modified-z: remove largest robust z if >= threshold",
        "95% robust interval: remove most extreme outside median ± z*MAD_SD",
    ],
)
c1, c2, c3, c4 = st.columns(4)
with c1:
    max_remove_per_condition = st.selectbox("Max outliers to remove per condition/analyte/scope", [0, 1, 2], index=1)
with c2:
    gcrit = st.number_input("Gcrit value", min_value=0.0, value=3.135, step=0.001, format="%.3f")
with c3:
    modified_z_threshold = st.number_input("MAD modified-z threshold", min_value=0.1, value=3.5, step=0.1)
with c4:
    robust_interval_z = st.number_input("Robust interval z", min_value=0.5, value=1.96, step=0.01)

st.markdown("""
**Primary test recommendation:** Welch two-sample t-test for mean shift because it does not assume equal variances.  
**Robust/non-parametric check:** Mann-Whitney U for distribution/median-like shift.  
**Assumption checks:** Shapiro-Wilk on residuals, classic Levene, and Brown-Forsythe/median-centered Levene.  
**Outlier outputs:** the app always reports raw results; if outlier removal is selected, it also reports cleaned results, a raw-vs-cleaned shift table, and an outlier log with batch/sample/device/value.
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
df2 = df2[df2[cond_col].isin([str(control_val), str(int_val)])].copy()
df2["__condition_str__"] = df2[cond_col].astype(str)
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

summary_raw_rows = []
summary_clean_rows = []
outlier_logs = []
paired_raw_rows = []
paired_clean_rows = []
paired_detail_tables = {}

for scope_name, sdf in scopes:
    for analyte in analytes:
        cols = [cond_col, "__condition_str__", analyte, device_col_use, "__pair_key__"]
        for extra in [batch_col, sample_col]:
            if extra != "None" and extra in sdf.columns and extra not in cols:
                cols.append(extra)
        work = sdf[cols].copy()
        work[analyte] = pd.to_numeric(work[analyte], errors="coerce")

        # raw result
        x_raw = work.loc[work["__condition_str__"] == str(control_val), analyte].dropna().values
        y_raw = work.loc[work["__condition_str__"] == str(int_val), analyte].dropna().values
        row_raw = compare_unpaired(x_raw, y_raw, str(control_val), str(int_val), int(n_boot), bool(do_boot), int(seed))
        row_raw.update({"scope": scope_name, "analyte": analyte, "result_type": "raw_no_outlier_removal", "control_condition": control_val, "interference_condition": int_val, "outlier_method": "None", "n_outliers_removed": 0})
        row_raw["primary_test"] = "Welch t-test"
        row_raw["recommended_p_value"] = row_raw.get("welch_t_p_primary", np.nan)
        row_raw["significant_at_alpha"] = bool(row_raw["recommended_p_value"] < alpha) if np.isfinite(row_raw["recommended_p_value"]) else None
        summary_raw_rows.append(row_raw)

        # cleaned/sensitivity result
        cleaned = work.copy()
        log_df = pd.DataFrame()
        if outlier_method != "None" and int(max_remove_per_condition) > 0:
            cleaned, log_df = apply_outlier_removal_for_analyte(
                work, analyte, control_val, int_val, outlier_method, int(max_remove_per_condition), float(gcrit),
                float(modified_z_threshold), float(robust_interval_z), batch_col, sample_col, device_col_use, scope_name
            )
            if not log_df.empty:
                outlier_logs.append(log_df)

        x_clean = cleaned.loc[cleaned["__condition_str__"] == str(control_val), analyte].dropna().values
        y_clean = cleaned.loc[cleaned["__condition_str__"] == str(int_val), analyte].dropna().values
        row_clean = compare_unpaired(x_clean, y_clean, str(control_val), str(int_val), int(n_boot), bool(do_boot), int(seed))
        row_clean.update({"scope": scope_name, "analyte": analyte, "result_type": "cleaned_after_outlier_rule", "control_condition": control_val, "interference_condition": int_val, "outlier_method": outlier_method, "n_outliers_removed": int(len(work) - len(cleaned))})
        row_clean["primary_test"] = "Welch t-test"
        row_clean["recommended_p_value"] = row_clean.get("welch_t_p_primary", np.nan)
        row_clean["significant_at_alpha"] = bool(row_clean["recommended_p_value"] < alpha) if np.isfinite(row_clean["recommended_p_value"]) else None
        summary_clean_rows.append(row_clean)

        # paired raw and cleaned
        if paired_mode:
            pair_cols = ["__pair_key__"] if scope_name == "pooled_all_devices" else ["__pair_key__", device_col_use]
            prow, pdetail = compare_paired(work, analyte, cond_col, str(control_val), str(int_val), pair_cols, int(n_boot), bool(do_boot), int(seed))
            if prow is not None:
                prow.update({"scope": scope_name, "analyte": analyte, "result_type": "raw_no_outlier_removal", "control_condition": control_val, "interference_condition": int_val})
                paired_raw_rows.append(prow)
                paired_detail_tables[f"paired_raw_{scope_name}_{analyte}"[:60]] = pdetail
            pcrow, pcdetail = compare_paired(cleaned, analyte, cond_col, str(control_val), str(int_val), pair_cols, int(n_boot), bool(do_boot), int(seed))
            if pcrow is not None:
                pcrow.update({"scope": scope_name, "analyte": analyte, "result_type": "cleaned_after_outlier_rule", "control_condition": control_val, "interference_condition": int_val, "outlier_method": outlier_method})
                paired_clean_rows.append(pcrow)

summary_raw = pd.DataFrame(summary_raw_rows)
summary_clean = pd.DataFrame(summary_clean_rows)
for tbl in [summary_raw, summary_clean]:
    if not tbl.empty:
        tbl["welch_t_q_BH_FDR_within_table"] = benjamini_hochberg(tbl["welch_t_p_primary"].values)
        tbl["mann_whitney_q_BH_FDR_within_table"] = benjamini_hochberg(tbl["mann_whitney_p_robust"].values)

paired_raw = pd.DataFrame(paired_raw_rows)
paired_clean = pd.DataFrame(paired_clean_rows)
for tbl in [paired_raw, paired_clean]:
    if not tbl.empty and "paired_t_p" in tbl.columns:
        tbl["paired_t_q_BH_FDR_within_table"] = benjamini_hochberg(tbl["paired_t_p"].values)
        tbl["wilcoxon_q_BH_FDR_within_table"] = benjamini_hochberg(tbl["wilcoxon_p"].values)

outlier_log = pd.concat(outlier_logs, ignore_index=True) if outlier_logs else pd.DataFrame(columns=["scope", "condition", "analyte", "removed_order", "outlier_method", "row_index", "batch_id", "sample_id", "device_id", "value_removed", "direction", "outlier_metric", "outlier_threshold", "details"])

# Raw-vs-cleaned shift table
shift_rows = []
if not summary_raw.empty and not summary_clean.empty:
    keys = ["scope", "analyte"]
    raw_small = summary_raw[keys + ["mean_difference_I_minus_C", "percent_shift_mean", "welch_t_p_primary", "mann_whitney_p_robust"]].rename(columns={
        "mean_difference_I_minus_C": "raw_mean_difference_I_minus_C",
        "percent_shift_mean": "raw_percent_shift_mean",
        "welch_t_p_primary": "raw_welch_p",
        "mann_whitney_p_robust": "raw_mann_whitney_p",
    })
    clean_small = summary_clean[keys + ["mean_difference_I_minus_C", "percent_shift_mean", "welch_t_p_primary", "mann_whitney_p_robust", "n_outliers_removed"]].rename(columns={
        "mean_difference_I_minus_C": "cleaned_mean_difference_I_minus_C",
        "percent_shift_mean": "cleaned_percent_shift_mean",
        "welch_t_p_primary": "cleaned_welch_p",
        "mann_whitney_p_robust": "cleaned_mann_whitney_p",
    })
    sensitivity = pd.merge(raw_small, clean_small, on=keys, how="outer")
    sensitivity["delta_mean_difference_cleaned_minus_raw"] = sensitivity["cleaned_mean_difference_I_minus_C"] - sensitivity["raw_mean_difference_I_minus_C"]
    sensitivity["delta_percent_shift_cleaned_minus_raw"] = sensitivity["cleaned_percent_shift_mean"] - sensitivity["raw_percent_shift_mean"]
else:
    sensitivity = pd.DataFrame()

counts = df2.groupby([cond_col, device_col_use], dropna=False).size().reset_index(name="n_rows")

st.subheader("4) Results: raw pooled/all devices primary table")
st.dataframe(summary_raw[summary_raw["scope"] == "pooled_all_devices"], use_container_width=True)

st.subheader("5) Results: cleaned/outlier-sensitivity pooled/all devices primary table")
st.dataframe(summary_clean[summary_clean["scope"] == "pooled_all_devices"], use_container_width=True)

st.subheader("6) Raw vs cleaned sensitivity table")
st.dataframe(sensitivity[sensitivity["scope"] == "pooled_all_devices"] if not sensitivity.empty else sensitivity, use_container_width=True)

st.subheader("7) Outlier log")
if outlier_log.empty:
    st.info("No outliers removed, or outlier removal was set to None/0.")
else:
    st.dataframe(outlier_log, use_container_width=True)

st.subheader("8) Paired results, if matched")
if paired_raw.empty and paired_clean.empty:
    st.info("No paired result could be created, or paired analysis was off.")
else:
    st.write("Raw paired results")
    st.dataframe(paired_raw, use_container_width=True)
    st.write("Cleaned paired results")
    st.dataframe(paired_clean, use_container_width=True)

st.subheader("9) Condition/device counts")
st.dataframe(counts, use_container_width=True)

meta = f"""Interference Analysis App
Rows analyzed: {len(df2)}
Condition column: {cond_col}
Control condition: {control_val}
Interference condition: {int_val}
Device mode: {device_mode}
Device column: {device_col}
Analytes: {', '.join(analytes)}
Primary test: Welch two-sample t-test; robust check: Mann-Whitney U; optional paired t/Wilcoxon if sample keys match.
Assumption checks: Shapiro-Wilk on residuals; Levene mean-centered; Brown-Forsythe median-centered.
Effect sizes: mean difference, median difference, percent shift in mean and median, bootstrap 95% CIs where enabled.
Outlier sensitivity method: {outlier_method}
Max outliers removed per condition/analyte/scope: {max_remove_per_condition}
Gcrit: {gcrit}; MAD modified-z threshold: {modified_z_threshold}; robust interval z: {robust_interval_z}
The ZIP contains raw results, cleaned results, raw-vs-cleaned sensitivity, and outlier log.
"""
result_tables = {
    "interference_summary_raw": summary_raw,
    "interference_summary_cleaned": summary_clean,
    "raw_vs_cleaned_sensitivity": sensitivity,
    "outlier_log": outlier_log,
    "paired_summary_raw": paired_raw,
    "paired_summary_cleaned": paired_clean,
    "condition_device_counts": counts,
}
for k, v in list(paired_detail_tables.items())[:20]:
    result_tables[k] = v
zip_bytes = make_download_zip(result_tables, meta)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.download_button("Download raw summary CSV", summary_raw.to_csv(index=False), "interference_summary_raw.csv", "text/csv")
with c2:
    st.download_button("Download cleaned summary CSV", summary_clean.to_csv(index=False), "interference_summary_cleaned.csv", "text/csv")
with c3:
    st.download_button("Download outlier log CSV", outlier_log.to_csv(index=False), "interference_outlier_log.csv", "text/csv")
with c4:
    st.download_button("Download ALL results ZIP", zip_bytes, "interference_analysis_results.zip", "application/zip")

st.success("Done. Download the CSVs or full ZIP above.")
