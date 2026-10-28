# Figures of the revised article (R1)

| figure | in the article | content | how to regenerate |
|---|---|---|---|
| `02_log_distributions_r1` | supplement | histograms of GR, RHOB and NPHI pooled over the 18 benchmark wells (sample count, median) | `scripts/make_revision_figures.py --data DIR` (needs the corpus) |
| `03_log_crossplots_r1` | supplement | GR–RHOB, GR–NPHI and RHOB–NPHI density cross-plots pooled over the 18 wells, with Pearson r | `scripts/make_revision_figures.py --data DIR` (needs the corpus) |
| `fig4_aggregation` | supplement | per-fold vs pooled R² for MEAN·single, GNN·blackout and ST-GNN·blackout. Source: `results/results_initial_run.json`, the initial single-seed run of the original submission (9 methods, 19 folds); the five-seed tables are in `results/aggregated.csv` | `scripts/make_revision_figures.py --results-json results/results_initial_run.json` |
| `missingness_scenarios_synthetic` | — (illustration) | the four masking scenarios (single, block, profile, blackout) on one window of the **synthetic** example corpus. This is an illustration on synthetic data, **not the article figure**: the article's version shows a window of a real well and is not released | `scripts/make_revision_figures.py --slices examples/synthetic/data/processed/slices.npz` |

Apart from the synthetic illustration, these files are the ones included in the revised
manuscript (identical PDFs). The two
pooled data figures carry no well identifier, depth or location; individual samples are
not attributed to a well.

Figures of the article that are **not** released here, because they reveal well
locations or individual log traces: the well map (`fig5_wellmap_r1`), the per-well depth
coverage (`04_depth_coverage_r1`), the example windows (`05_example_slices_r1`) and the
masking illustration on a real window (`06_missingness_scenarios`, replaced here by its
synthetic counterpart).

`08_convergence_curves` (results/figures/) and `fig_sensitivity_probe`
(results/robustness/) are cited by the revised supplement from their v1.0 locations; the
released files are the ones in the manuscript.
