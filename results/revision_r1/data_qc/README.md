# Dataset quality control — revision R1

Outputs of `scripts/build_dataset.py` (revision pipeline) for the 18-well corpus of the
revised article, anonymised for release.

**Well identifiers.** The 18 benchmark wells are W01–W18 in the order of `folds_loo.json`,
the same numbering as `../error_stats/` (`well = W{fold+1:02d}`). The well removed by the
depth-step QC has no identifier and appears as `EXCLUDED`. No well name, file name, depth,
coordinate or source path is included, and no correspondence table is published. The
W01–W18 numbering is not the W01–W19 numbering of the original submission.

| file | content |
|---|---|
| `depth_qc.csv` | depth-step check per well: median/min/max step (m), non-positive steps, jumps and their size, `kept` / `EXCLUDED`. 18 benchmark wells and the excluded well. |
| `quality_report.csv` | per well and log: source mnemonic, cleaning action (unit conversion, out-of-bounds samples removed), coverage (%). 18 benchmark wells and the excluded well. |
| `missingness_real.json` | observed missingness of the 18 wells: co-occurrence of missing logs, length quantiles by combination of absent logs, `gaps_per_well` (count per W-id), and `intervals`, the list of (absent-log combination, length in samples) sampled by the `blackout_emp` scenario. The intervals carry no depth. Their order is kept because the empirical-blackout sampler draws from them by index. |
| `screened_wells_summary.csv` | counts only: wells screened, kept, excluded by the depth QC, and the wells without the complete GR+RHOB+NPHI triplet, by missing or rejected log. These wells are not listed individually. |
