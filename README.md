# Interference Analysis: Control vs Interference — updated

https://interferenceblood.streamlit.app/

The existing Welch/Mann-Whitney, assumption-check, outlier, effect-size, paired-sensitivity and FDR calculations are retained. The requested automatic branch/output behavior is now explicit.

- Global-flag selector and “treat all rows as FALSE” override.
- `global_flag=TRUE` rows are excluded before analysis and exported in their own worksheet.
- Default analyte order: RBC, WBC_2, PLT_3, HCT, HGB, MCV_3, RDW_3, MCH, MCHC, NEUT_2, LYMPH_2, MXD_2, PLT, MCV, RDW.
- Paired analysis is off by default.
- Bootstrap 95% CIs are off by default.
- Automatic outlier mode is default: Shapiro-Wilk residual normality selects existing Gcrit when normal and existing Robust MAD otherwise.
- Automatic Gcrit is default.
- Single primary inferential outcome: normal residuals -> Welch t-test; non-normal/not-testable residuals -> Mann-Whitney U. Both diagnostic p-values remain available for audit, but `Statistical Test Applied` / `Selected p value` are the single reported branch outcome.
- One downloadable Excel workbook only, including raw/cleaned summaries, outlier log, `global flag TRUE`, condition/device counts, sensitivity table, optional paired sheets, and settings.

Run with:

```bash
pip install -r requirements.txt
streamlit run app.py
```
