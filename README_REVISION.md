# heartloc – analyses added in revision (Scientific Reports, 2026)

**Repository layout.** This folder set (`kod/revision/`, `veri/`, `sonuclar/revision/`) is added to the existing
heartloc repository (https://github.com/kahramanalias/heartloc). The main analysis script from release v1.0, which
retrieves all source data (CDC, USDA, USGS, PLACES), runs the leakage audit and the original analysis, stays in the
repository root; running it produces its outputs, including the audit table. The revision scripts read
`veri/analiz_tablosu_setB.csv` (included here), the county analysis table produced by that script.

Run every script from the repository root, e.g. `python kod/revision/a1_arms.py`.
Python 3, scikit-learn, pandas, numpy, scipy, matplotlib. Long runs are checkpointed
(re-running a script continues where it stopped). `N_REP` sets the number of repetitions.

| Script | Purpose | Manuscript |
|---|---|---|
| `common.py` | data loading, rotated equal-count blocks, randomised state folds, environmental blocks | all |
| `a1_arms.py` | predictor sets A/B (± coordinates), four location-only baselines, nonsense arm × 3 regimes × 10 seeds | Tables 1–2, S3, S7 |
| `b9_gb_ridge.py` | gradient boosting and ridge (Table 1); `SONUC=yas_std TAG=_w_std` for WONDER | Table 1, S11 |
| `b3_tuned.py` | nested tuning of kNN / GP / thin-plate spline baselines, grid and oracle results | Table 2, S4 |
| `a2_wonder.py` | unsmoothed WONDER replication (`SONUC=yas_std TAG=w_std` or `SONUC=ham_hiz TAG=w_ham`) | Table 4, S11 |
| `b5_buffer.py` | buffered CV 0–500 km with matched random-removal controls, 5 seeds | Table 3, S8 |
| `b1_variogram.py` | correlograms, variograms, block extents | S1–S2 figure/table |
| `b4_aoa.py` | dissimilarity index, area of applicability, per-block table (block numbers as in Figure 2), local Moran | Fig. 4b, S9, Fig. S4 |
| `b6_env.py` | environmental (k-means feature-space) blocking | S10 |
| `b7_smooth.py` | smoothing simulation with an independent known signal | S12 |
| `b8_weighted.py` | population-weighted fitting / evaluation | S13 |
| `b10_reduced.py` | set A without the three state-level inputs (24 predictors) | Table 1, S3 |
| `c1_summary.py`, `c2_moran.py`, `c3_more.py`, `e1_tables.py` | summaries, confidence intervals, block bootstrap, table contents | Tables, S5, S14 |
| `b11_blocksize.py` | sensitivity to block size (5, 10, 20 blocks) | Fig. S3 |
| `b12_purenoise.py` | pure-noise smoothing scenario | Table S12b |
| `b13_importance.py` | permutation importance on held-out blocks, sets A and B | Table S6, Fig. S6 |
| `b14_foldwise.py` | fold-wise versus pooled R² | Fig. S5 |
| `d1_figs.py` | Figures 2–4 and Supplementary Figures S1, S2 and S4 (600 dpi; file names follow the Supplementary Information numbering) | figures |

## Data added in revision (`veri/`)
* `wonder_ilce.csv` – CDC WONDER, Underlying Cause of Death 2018–2024 Single Race; years 2018–2022;
  ICD-10 113 cause list "#Diseases of heart (I00-I09,I11,I13,I20-I51)"; ages 35-44 … 85+; grouped by county.
  Columns: `olum` deaths 2018–2022, `nufus_35arti` population aged 35+ (mean of 1 July 2020–2022),
  `beklenen` expected deaths from national age-specific rates, `SMR`, `ham_hiz` crude rate,
  `yas_std` indirectly age-standardised rate (per 100,000 per year). The 22 suppressed counties are not included,
  in line with the CDC WONDER data use restrictions. Oglala Lakota (WONDER code 46113) is stored as 46102.
* `nufus_duzeltilmis.csv` – Census Bureau county population estimates, Vintage 2024 (cc-est2024-agesex-all),
  five-year age groups 35+, mean of YEAR codes 2–4 (1 July 2020, 2021, 2022).
* National age-specific heart disease death rates 2018–2022 (NCHS, Health, United States; data.cdc.gov 7aq9-prdf)
  are hard-coded in `kod/revision/prep_wonder.py`, which builds both files above from the raw exports: 35–44: 27.42; 45–54: 80.60; 55–64: 200.66; 65–74: 405.54;
  75–84: 1015.50; 85+: 3875.94 per 100,000 (means of the annual values).

All tables and figures of the revised manuscript and Supplementary Information, except Figure 1 (outcome map), are produced by the scripts above. The main script (`kalp_ve_loc.py`) builds the county analysis table and writes the audit table `cizelge0_sizinti_denetimi`.

`sonuclar/revision/` holds all checkpoint files, out-of-fold predictions of the first repetition, table contents and figures.

Software used for the revision analyses: Python 3.13, scikit-learn 1.9.1, NumPy 2.5.3, pandas 3.0.5, SciPy 1.18.1.
