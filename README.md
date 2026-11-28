# heartloc

Code and derived data for:

**Spatial cross-validation and location-only baselines for area-level health models: a case study of US county heart disease mortality**
Abdulgani Kahraman, Department of Computer Engineering, Balıkesir University (Scientific Reports, revision 2026).

Archived at Zenodo, concept DOI [10.5281/zenodo.24511944](https://doi.org/10.5281/zenodo.24511944) (always resolves to the latest version).

## Contents

| Path | Content |
|---|---|
| `kalp_ve_loc.py` | Main analysis script (single Google Colab cell). Retrieves the CDC, USDA, USGS and PLACES data, builds the county analysis table, runs the leakage audit (table `cizelge0_sizinti_denetimi`) and the original analysis. Produces Supplementary Figures S3, S5 and S6 and Supplementary Tables S6 and S12b. |
| `kod/revision/` | Scripts for all analyses added in revision (see `README_REVISION.md` for the script-to-table map). |
| `veri/` | County analysis table (`analiz_tablosu_setB.csv`), unsmoothed CDC WONDER county rates for unsuppressed counties (`wonder_ilce.csv`) and Census population denominators (`nufus_duzeltilmis.csv`). |
| `sonuclar/revision/` | Checkpoint files, out-of-fold predictions of the first repetition, table contents (`tab/`) and Figures 2–4 and Supplementary Figures S1, S2 and S4 (`fig/`, 600 dpi). |

## Running

1. Main analysis: open `kalp_ve_loc.py` in Google Colab and run it as one cell; outputs are written to `/content/sonuclar`.
2. Revision analyses: from the repository root, run e.g. `python kod/revision/a1_arms.py`. Scripts are checkpointed and continue where they stopped.

Python 3.13, scikit-learn 1.9.1, NumPy 2.5.3, pandas 3.0.5, SciPy 1.18.1, Matplotlib.

CDC WONDER extracts are subject to the CDC WONDER data use restrictions; county rates are distributed only for counties with unsuppressed counts.

## Licence

Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/
