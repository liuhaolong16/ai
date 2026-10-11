# Where cross-temperature generalization breaks — code, derived data and figures

Reproducibility package for the manuscript:

> **Where cross-temperature generalization breaks: an interpretable machine-learning study of
> lithium-ion battery state-of-health from early-cycle features**
> (submitted to *Journal of Energy Storage*)

This repository contains the analysis code, the derived analysis dataset and the figure sources
behind every number reported in the paper.

---

## 1. Data provenance — please read before reusing

| | |
|---|---|
| **Primary source** | NASA Prognostics Center of Excellence (PCoE), *Battery Data Set*, NASA Ames Research Center (cells B0005–B0056) |
| **Retrieval channel used here** | mirror `devhaaana/nasa-battery-aging-dataset` |

**The mirror is a retrieval channel only — it is not the object to be cited.** The raw `.mat`
files are **not redistributed here**: they are NASA's data and should be obtained from the
primary source. `code/download_data.py` documents the retrieval used and writes into `data/`.

The ambient temperature of every cycle was read from the raw `ambient_temperature` field inside
each `.mat` cycle record — **never** from filenames or from the literature. This is what the paper
means by a *temperature-verified* subset.

---

## 2. Repository contents

```
code/          analysis pipeline (see §3)
data/          filtered_v2.csv  — the derived analysis dataset (16 columns, 2,096 rows)
figures/       fig1–fig4, vector PDF for the manuscript + PNG previews
results/       machine-generated result tables that back the paper's numbers
```

`data/filtered_v2.csv` is the analysis dataset after the within-cell main-group filter (§4).
Its columns are:

```
cell, temp_c, cycle_idx, cap_ah, soh, duration_s, mean_abs_i, rest_frac,
v_mean, v_std, v_min, v_max, t_mean, t_max, t_rise, n_points
```

---

## 3. Pipeline

| Step | Script | Role |
|---|---|---|
| 1 | `code/download_data.py` | retrieve the raw `.mat` cells into `data/` |
| 2 | `code/explore_data.py` | inspect raw cycle records |
| 3 | `code/filter_v2.py` | **produces `data/filtered_v2.csv`** — per-cell MAD main-group filter (§4) |
| 4 | `code/model_cv.py` | **dual-protocol cross-validation + three-tier feature ablation** → `results/results_cv.md` |
| 5 | `code/shap_analysis.py` | SHAP attribution → `results/shap_results.md` |
| 6 | `code/find_failure_source.py` | 11 leave-one-feature-out experiments + temperature-feature intervention → `results/failure_source.md` |
| 7 | `code/verify_attribution.py` | the pre-registered intervention test that **falsified** the SHAP-derived causal inference → `results/verify_attribution.md` |
| 8 | `code/make_figures.py` | regenerate the four figures (uses the vendored `code/figkit.py`) |

Auxiliary: `filter_cycles.py` (the superseded v1 filter), `extract_features.py`,
`diag_cycles.py`, `run_baseline.py`.

### Running it

```bash
python -m pip install -r requirements.txt

# full pipeline from raw NASA data
python code/download_data.py     # writes raw .mat into data/  (large download)
python code/filter_v2.py         # writes data/filtered_v2.csv
python code/model_cv.py          # writes results/results_cv.md
python code/shap_analysis.py     # writes results/shap_results.md
python code/find_failure_source.py
python code/verify_attribution.py
python code/make_figures.py      # writes figures/*.pdf and *.png
```

All paths are resolved **relative to the repository root**, so a fresh clone runs in place.

### Verified environment

Verified with Python 3.14.8 / numpy 2.5.3 / scipy 1.18.1 / xgboost 3.4.1 / shap 0.53.0 /
matplotlib 3.11.2 (see `requirements.txt`). The model is an `XGBRegressor` with a fixed
`random_state=0`, so the reported numbers are deterministic.

**Self-check.** Running `python code/model_cv.py` from a fresh clone reproduces
`results/results_cv.md` **byte-for-byte** (identical SHA-256) from the deposited
`data/filtered_v2.csv`. The deposit is therefore sufficient to reproduce the paper's headline
numbers without the raw NASA download.

---

## 4. Filtering and exclusion decisions

These are the only judgement calls in the pipeline; each is applied in code, not by hand.

- **Within-cell MAD main-group filter (`filter_v2.py`).** Protocol consistency is a property of a
  cell, not of the dataset, so for **each cell separately** the cycles are kept only if
  `log(duration_s)`, `log(mean_abs_i)` and `rest_frac` all fall within
  `median ± 3 · 1.4826 · MAD`. Cells whose reference cycles are not homogeneous are dropped.
- **`B0041` excluded entirely** — internally two populations, with the earlier one numerically
  dominant, so any main-group criterion selects it. See `code/filter_v2.py`.
- **`MIXED = {B0038, B0039, B0040, B0042, B0043, B0044}` skipped** — mixed protocol.
- **`B0050` excluded in `model_cv.py`** (`EXCLUDE_CELLS`) — its SOH **rises** with cycle index,
  i.e. capacity recovery beyond the monotonicity tolerance, which makes it unusable as a
  degradation reference.

After these steps: **2,096 reference discharge cycles over 27 cells** in
`data/filtered_v2.csv`, reduced to the paper's analysis set of **26 cells / 2,084 cycles**
by the `B0050` exclusion.

| Ambient | Cells | Cycles (deposited) | Cells | Cycles (analysis set) |
|---|---|---|---|---|
| 4 °C | 12 | 648 | 11 | 636 |
| 24 °C | 11 | 1,297 | 11 | 1,297 |
| 43 °C | 4 | 151 | 4 | 151 |
| **total** | **27** | **2,096** | **26** | **2,084** |

---

## 5. Protocols and feature sets

- **Protocol A — within-temperature leave-one-cell-out (LOCO).**
- **Protocol B — cross-temperature leave-one-temperature-out (LOTO)**: train on two ambient
  temperatures, test on the held-out third.
- **Feature sets:** `F1` = 11 indirect physical descriptors (no capacity); `F2` = `F1` +
  `cycle_idx`; `F3` = `F2` + `cap_ah`.

**`F3` is intentionally a near-tautology**: SOH is defined as `cap_ah / 2.0`, so including
`cap_ah` as a predictor inflates protocol A (`R² = 0.992` at 24 °C). The paper reports it as a
**diagnostic of the tautology**, not as a result.

---

## 6. Headline numbers (from `results/results_cv.md`)

`F1`, protocol A: RMSE / R² = 0.0839 / **0.398** (4 °C), 0.0336 / **0.874** (24 °C),
0.0238 / 0.527 (43 °C).

`F1`, protocol B: **negative R² for every held-out temperature** — −4.459 (4 °C),
−4.133 (24 °C), −8.002 (43 °C). A cliff, not a gradual decay.

`F3`, protocol A at 24 °C rises to R² = 0.992 while protocol B still largely fails.

> ⚠ **Confounder, stated in the paper and repeated here:** in this dataset **temperature and
> discharge rate are confounded**, so protocol B is more precisely a
> **cross-operating-condition** protocol. Cross-temperature generalization as such is not
> isolated by this design.

---

## 7. Figures

`figures/fig1_soh_curves`, `fig2_dual_protocol`, `fig3_shap`, `fig4_ablation` — each as vector
**PDF** (for typesetting) and **PNG** (preview). Regenerate with `code/make_figures.py`.

---

## 8. Citation

See `CITATION.cff`. The manuscript is under submission; the journal DOI and the author list are
**not yet assigned** and still need to be filled in.

## 9. License

Code: MIT (`LICENSE`). Derived data `data/filtered_v2.csv` and the figures: CC BY 4.0. The
underlying NASA PCoE measurements remain subject to NASA's terms and are not relicensed here.
