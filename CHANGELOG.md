# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [2.0.0] - Unreleased

### Added
- **Baseline (reference) period**: `calculate(..., baseline_start=1961, baseline_end=1990)` fits
  the distributions (and the probability of zero precipitation) on a reference period only, then
  transforms the entire record — the standard setup for climate-projection work (#18).
- **Annual / whole-series mode**: `calculate(..., freq=None)` fits a single distribution to the
  whole series, the correct semantics for annual totals (#25).
- **Fitted parameters**: `calculate(..., return_params=True)` additionally returns a tidy
  dataframe of fitted parameters per (column, frequency group), including `p_zero` (#20).
- **SPEI class re-added** (`from standard_precip import SPEI`), with documentation on supplying
  the water balance P − PET and the recommended `glo`/`lmom` fit per Vicente-Serrano et al.
  2010 (#15).
- L-moments fitting for the Generalized Logistic and Generalized Normal distributions
  (previously MLE-only), and Kappa (may be unsolvable for some samples, yielding NaN + warning).
- `standard_precip/__init__.py` with `__version__` and re-exports: `from standard_precip import
  SPI, SPEI` now works.
- GitHub Actions CI (Python 3.10–3.13) and a PyPI trusted-publishing release workflow, replacing
  the long-dead Travis setup.

### Fixed
- **pandas ≥ 2.0**: `freq="W"` no longer crashes (`Series.dt.week` was removed; now uses
  `isocalendar().week`) (#24).
- **numpy ≥ 1.24**: `best_fit_distribution` no longer crashes (`np.histogram(normed=)` was
  removed); it also had a missing import that made it raise `NameError` since its introduction
  (#19).
- **pe3 with MLE** works again: it had recursed infinitely for years because scipy's
  `pearson3.fit` breaks when subclassed, as the vendored l-moments code did.
- Small frequency groups with non-gamma distributions no longer crash (`params=None` guard);
  insufficient data now consistently yields NaN plus a `UserWarning`.
- `calculate()` and `rolling_window_sum()` no longer mutate the caller's DataFrame.
- With `scale > 1`, rows are sorted by date before the rolling window is applied; unsorted
  input previously produced silently wrong sums. A warning is emitted when the dates are not
  regularly spaced (the window would span gaps in the record).
- With multiple precipitation columns, a NaN in one column no longer removes that row from the
  other columns' distribution fits and outputs.
- Observations outside the support of the fitted distribution map to large finite index values
  (about ±8.2; CDF clipped at float64 resolution) instead of NaN — important when transforming
  records against a baseline period. All previously finite values are unchanged.
- A gamma/Pearson-III frequency group with no observations (e.g. a partial-year baseline with
  daily or weekly frequency) yields NaN instead of raising ZeroDivisionError.
- A group whose fit fails to converge (lmoments3 raises a bare `Exception` for some samples)
  yields NaN instead of crashing the whole calculation.
- Input DataFrames with duplicate index labels (e.g. from `pd.concat` without `ignore_index`)
  no longer break the baseline-period path; the working copy's index is reset internally.
- `baseline_start`/`baseline_end` accept numpy integers (e.g. `df['year'].max()`) as years;
  previously they were misinterpreted as nanosecond timestamps.
- `best_fit_distribution` now fits each candidate exactly as `calculate()` does (zero-stripping
  and mixed-CDF weighting for gamma/Pearson III, gamma MLE `floc=0` default), so its ranking
  corresponds to the model the index calculation actually uses.
- `return_params`: `n_fit` counts the observations actually used in the fit (zeros stripped for
  gamma/Pearson III are excluded).
- Unfittable frequency groups produce one aggregated warning per column instead of one warning
  per group (sparse daily records previously emitted hundreds).
- A user-supplied `freq_col` literally named `"freq"` is no longer dropped from the output.
- The zero-stripping distribution set lives on the immutable distribution registry; the mutable
  class-level `non_zero_distr` list (which leaked mutations across all instances) is removed.
- The internal l-moments code no longer sorts the input array in place.
- `freq_col` is now usable: the column is carried into the calculation (it previously always
  raised `KeyError`) and is validated.
- Unknown `freq`/`dist_type`/`fit_type` values raise `ValueError` with the available options.

### Changed
- **Python ≥ 3.10 required**; dependencies now have tested lower bounds (numpy ≥ 1.24,
  pandas ≥ 2.2, scipy ≥ 1.10, matplotlib ≥ 3.7).
- **Vendored GPL-3 l-moments code removed**; L-moment fitting now uses the maintained
  [lmoments3](https://github.com/Ouranosinc/lmoments3) package (#23), with `>=1.0.7` required:
  1.0.6 produces different (incorrect) Wakeby and Generalized Normal L-moment fits. A regression
  gate verifies numerical equivalence to 1e-9 across the swap. MLE fitting uses plain scipy
  distributions. The full test suite is verified against the declared minimum versions
  (numpy 1.24 / scipy 1.10 / pandas 2.2 / matplotlib 3.7 / lmoments3 1.0.7) and the current
  locked versions.
- **Gamma MLE fixes `loc=0` by default** (pass `floc`/`loc` to override): unconstrained-loc
  gamma MLE is ill-posed on zero-stripped precipitation data and produced extreme index
  values (#22). For the same reason, unconstrained Weibull MLE became degenerate on modern
  scipy; pass `floc=0` for stable Weibull MLE fits.
- `wak` with `fit_type="mle"` raises a clear `ValueError` (it never actually worked).
- `fit_distribution` returns `(distribution, params, p_zero)` and `cdf_to_ppf` takes the
  distribution explicitly; instances no longer carry per-call state and are safely reusable.
- The duplicate-date message is a `UserWarning` instead of `print()`.
- Packaging: `pyproject.toml` (hatchling) replaces `setup.py`; the project is developed with
  [uv](https://docs.astral.sh/uv/) (`uv sync --group dev`).

## [1.0] - 2021

Historical release; see git history.
