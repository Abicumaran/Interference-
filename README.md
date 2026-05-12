# Interference Analysis App

Streamlit app for comparing Control vs Interference conditions across selected analytes.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Expected input
Excel or CSV with:
- a condition column, e.g. `Condition` with values like `c` and `i`
- numeric analyte columns, e.g. WBC, RBC, HGB, HCT, PLT
- optional `deviceId`, `batch_id`, and `bloodSampleId`

## Outputs
- Mean ± SD by condition
- Median and IQR by condition
- Mean difference and median difference
- Percent shift from control to interference
- Shapiro-Wilk residual normality test
- Levene and Brown-Forsythe variance equality tests
- Welch t-test, Student t-test, Mann-Whitney U, permutation p-value
- Optional paired t-test and Wilcoxon if sample keys match
- Bootstrap 95% confidence intervals
- CSVs + full ZIP/Excel workbook
