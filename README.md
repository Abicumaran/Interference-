# Interference Analysis App

Streamlit app for comparing **Control vs Interference** conditions across selected analytes.

## Features

- Upload Excel or CSV.
- Select condition, device, batch, and sample columns.
- Select analytes to compare.
- Default analysis pools all devices; optional per-device + pooled analysis.
- Reports raw results with no outlier removal.
- Optional outlier sensitivity analysis with:
  - Gcrit / Grubbs-like largest-deviation removal
  - Robust MAD modified-z removal
  - 95% robust interval removal: median ± z × MAD-scaled SD
- User can remove up to 0, 1, or 2 outliers per condition × analyte × scope.
- Saves an outlier log with batch ID, sample ID, device ID, condition, analyte, value removed, and metric.
- Reports cleaned results after outlier removal and a raw-vs-cleaned sensitivity table.
- Assumption checks:
  - Shapiro-Wilk on residuals
  - Levene mean-centered equal-variance test
  - Brown-Forsythe median-centered equal-variance test
- Statistical tests:
  - Welch two-sample t-test as primary test
  - Student t-test for reference
  - Mann-Whitney U robust check
  - Permutation p-value for mean difference
  - Optional paired t-test/Wilcoxon if sample keys match
- Effect sizes:
  - mean difference
  - median difference
  - percent mean shift
  - percent median shift
  - bootstrap 95% confidence intervals
- Download CSVs or full ZIP with CSV + Excel workbook.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

Use `app.py` as the main file path.
