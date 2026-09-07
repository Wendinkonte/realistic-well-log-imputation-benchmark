# Data format

What the code expects, so the benchmark can be run on any well-log collection.

---

## 1. Input: the LAS corpus

One LAS file per well, in a single directory (`las_dir`). Read with
[`lasio`](https://lasio.readthedocs.io/), so LAS 2.0 and 3.0 both work.

**Depth index.** The file's index is the depth in metres. The study corpus is sampled at
**0.1524 m (0.5 ft)**, the value in `data/pipeline.py :: STEP_M`. The windowing itself
only assumes a *regular* step; the spatial models additionally assume every well shares
the same step, because they resample onto one common depth axis.

**Curve mnemonics.** For each of the three logs, the first mnemonic found in this priority
order is used (`data/pipeline.py :: PREFS`):

| Log | Accepted mnemonics, in priority order |
|---|---|
| `GR` (gamma ray) | `GR`, `SGR`, `CGR`, `GRD`, `GRN` |
| `RHOB` (bulk density) | `RHOB`, `RHOZ`, `DEN`, `DENS` |
| `NPHI` (neutron porosity) | `NPHI`, `TNPH`, `NPOR`, `PHIN` |

`NEUT` is deliberately **not** accepted for NPHI: it is a neutron count, not a porosity.

A well missing one of the three logs is read and reported, but excluded from the folds.

**Well identifier.** Taken from the filename without its extension. It is the key used by
the folds, by the coordinate lookup and by the anonymisation map, so keep it stable.

**Example.** `examples/synthetic/data/las/SYN01.las`, written by
`examples/synthetic/make_synthetic.py`, is a minimal valid file.

---

## 2. Cleaning applied by the pipeline

| Step | Rule |
|---|---|
| Unit harmonisation, NPHI | if the 99th percentile exceeds 1.5, values are read as percent and divided by 100. The decision uses the observed RANGE, not the LAS header unit, because the header is unreliable in practice. |
| Aberrant GR | a well whose GR median is below 6 GAPI is treated as recorded in a wrong unit, and its GR is dropped |
| Physical bounds | out-of-range samples become NaN: GR [0, 400] GAPI; RHOB [1.2, 3.1] g/cm³; NPHI [-0.05, 1.0] fraction |
| Normalisation | **none here.** The scaler is fitted per fold on training windows only, which is what makes the protocol leakage-free. |

Only MEASURED curves are used. Composite curves — depth-merged products that blend
measured wireline data with model-derived infill — are excluded on purpose: using them as
ground truth makes the evaluation circular (article Section 4.1).

---

## 3. Output: the processed dataset

`scripts/build_dataset.py` writes five artefacts into `data_dir`.

### `slices.npz` — the windows every method operates on

| Key | Type | Meaning |
|---|---|---|
| `X` | float32 `(N, 256, 3)` | intact windows; channel order `GR, RHOB, NPHI` |
| `wells` | str `(N,)` | well identifier of each window |
| `starts` | int `(N,)` | start index of each window within its well |
| `logs` | str `(3,)` | `["GR", "RHOB", "NPHI"]` |
| `slice_len` | int | 256 |
| `stride` | int | 64 |

A window is kept only where all three logs are present over its full length, so the
ground truth is never itself imputed.

### `well_logs.npz` — the cleaned continuous logs

One entry per well: a float array `(L, 4)` whose columns are `depth, GR, RHOB, NPHI`,
cleaned but NOT normalised, NaN where missing. Used by the spatial models to build the
common depth grid.

### `folds_loo.json` — the leave-one-well-out folds

```json
{
  "wells": ["W01", "W02", "..."],
  "n_folds": 19,
  "folds": {
    "fold_0": {"test": ["W01"], "train": ["W02", "W03", "..."]},
    "fold_1": {"test": ["W02"], "train": ["W01", "W03", "..."]}
  }
}
```

One fold per well carrying the complete triplet, in sorted order.

### `missingness_real.json` — the measured missingness statistics

```json
{
  "n_wells_triplet": 19,
  "cooccur_pct": {"1_logs": 3.5, "2_logs": 75.9, "3_logs": 20.6},
  "gap_samples": {"p50": 260.0, "p75": 0.0, "p90": 0.0, "p99": 0.0},
  "gap_median_m": 39.6
}
```

`cooccur_pct` is what calibrates the `blackout` scenario: on the study corpus 96.5 % of
gaps involve two or three logs missing together, which is the empirical fact the paper
is built on.

### `quality_report.csv` — traceability

`well, log, mnemo_source, action, coverage_%` — one row per (well, log), recording which
mnemonic was used, what unit action was applied, and the resulting coverage.

---

## 4. Well coordinates — spatial models only

Required by GNN and ST-GNN, ignored by the other nine methods. A CSV at `coords_csv`:

```csv
Name,SIGLE,Surf_LocX_L93,Surf_LocY_L93
SYN01,SYN01,100000.0,200000.0
SYN02,SYN02,102000.0,200000.0
```

| Column | Meaning |
|---|---|
| `Name` | well identifier; matched to the log identifiers after normalisation |
| `SIGLE` | short code; read but not used |
| `Surf_LocX_L93`, `Surf_LocY_L93` | projected surface easting and northing, **in metres** |

The column names retain the Lambert-93 (RGF93) naming of the study corpus, but nothing in
the code is specific to that projection: distances are Euclidean in the supplied
coordinates, so any metric projection works. Do not supply degrees.

Identifier matching is tolerant (`models/spatial.py :: _key`): it strips a trailing
archive-index suffix, collapses separators, drops a trailing deviation letter, and expands
the French abbreviation `SS` to `SOUS`. If a well is missing from the table, `load_coords`
raises rather than guessing.

If your log files and your coordinate table spell some wells differently in a way the
generic rules do not cover, add the substitutions to `configs/well_aliases.yaml`:

```yaml
aliases:
  SPELLING-IN-THE-LAS-FILENAME: SPELLING-IN-THE-COORDINATE-TABLE
```

Start from `configs/well_aliases.example.yaml`. The file is **git-ignored**, because a
list of well names is itself identifying information — which is exactly why no such alias
is hard-coded in the source.

---

## 5. Optional: an anonymisation map

If you need figures labelled with anonymous identifiers, provide a CSV at `anon_map`:

```csv
anon_id,well_name
W01,SOME-WELL-1
W02,ANOTHER-WELL-2
```

Without it, `figstyle.anon()` returns identifiers unchanged. With it, an identifier
absent from the map raises rather than silently falling back to the raw name — so a real
identifier cannot slip into a published figure.

**Never commit a real mapping table.** `.gitignore` blocks it by name and
`tests/test_privacy_guards.py` fails the build if one appears.

---

## 6. Configuring the paths

```bash
cp configs/paths.example.yaml configs/paths.yaml   # then edit; the file is git-ignored
```

or set environment variables, which take priority:

```bash
export WELLOG_LAS_DIR=/path/to/las
export WELLOG_DATA_DIR=/path/to/processed
export WELLOG_COORDS_CSV=/path/to/well_coordinates.csv
export WELLOG_RESULTS_DIR=/path/to/results
```

Most scripts also take `--data-dir` / `--out` for a one-off override.
