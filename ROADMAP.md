# ROADMAP.md

# Current Development Roadmap

This document tracks the current engineering status of the repository.

Its purpose is to organize software development, not to describe long-term research ideas or commercial applications.

Completed work should be moved to the project history or Git history rather than removed.

Future speculative research should **not** be included here.

---

# Repository Status

Current Version

Development

Current Focus

General score-driven modelling framework for probabilistic forecasting.

Current Dataset

Configured by the active notebook.

Current Active Notebook

Configured by the user.

---

# Current Milestone

The current milestone is to transform the repository into a reusable scientific framework while completing the precipitation modelling experiments required for the thesis.

The priority order is:

1. Framework correctness.
2. Scientific reproducibility.
3. Experimental results.
4. Report generation.
5. Documentation.

---

# Active Tasks

## Core Framework

* [ ] General dynamic parameter interface.
* [ ] Distribution factory.
* [ ] Score scaling factory.
* [ ] Dynamic recursion factory.
* [ ] DataLoader abstraction.
* [ ] Configuration system.
* [ ] Artifact manager.

---

## Mathematical Models

### Standard GAS

* [ ] Generalize dynamic parameter selection.
* [ ] Support arbitrary parameter subsets.
* [ ] Support multiple score scaling methods.
* [ ] Preserve analytical Fisher information.

### Harvey Long-Short

* [ ] Replace current implementation with Harvey two-component recursion.
* [ ] Support separate long and short covariates.
* [ ] Enforce persistence restrictions.
* [ ] Maintain analytical score implementation.

### Regime Models

* [ ] Implement regime-sensitive score updates.
* [ ] Support threshold definitions such as (q_{0.95}) and (q_{0.98}).
* [ ] Support applying regime effects to selected dynamic parameters.
* [ ] Keep regime logic inside the model/dynamics layer, not notebooks.
* [ ] Add configuration flag `run_regime_models = False` by default.
* [ ] Ensure regime models save the same artifacts and diagnostics as other models.

Status:

Implemented as part of the modelling framework, but disabled by default.

Do not execute regime-change experiments unless explicitly requested.

---

## Optimization

* [ ] Improve convergence monitoring.
* [ ] Runtime monitoring.
* [ ] Memory monitoring.
* [ ] Retry framework.
* [ ] Convergence classification.
* [ ] Approximate standard errors from optimizer Hessian.
* [ ] Numerical safety improvements.

---

## Evaluation

* [ ] Separate wet/dry-day metrics.
* [ ] Improve tail diagnostics.
* [ ] Standardize model comparison tables.
* [ ] Validate PIT diagnostics.
* [ ] Validate residual diagnostics.

---

## Reporting

* [ ] Automated report generation.
* [ ] Editable LaTeX report.
* [ ] Report validation.
* [ ] Automatic README summary.
* [ ] Artifact-based report generation.

---

## Documentation

* [ ] Complete documentation files.
* [ ] Module docstrings.
* [ ] Package OVERVIEW sections.
* [ ] Examples.
* [ ] README improvements.

---

# Current Thesis Pipeline

The current experimental pipeline is:

## Stage 1

Baseline GAS

* phi only
* phi + xi

Short lags

↓

Seasonal lags

↓

Compare score scaling

↓

Select winner

---

## Stage 2

Weather covariates

↓

Dew point

* short lags
* seasonal lags

↓

Select best dew point

↓

Temperature

* short lags
* seasonal lags, only if justified

↓

Select best weather model

---

## Stage 3

Harvey Long-Short

Short component

↓

Best weather specification

Long component

↓

ENSO

1. 90-day rolling mean

2. 90-day + 30-day rolling means

3. 90-day + 30-day rolling means + lagged daily ENSO

↓

Select final Harvey model

---

## Stage 4

Generate

* diagnostics
* comparison tables
* figures
* report

---

## Optional Stage

Regime-change models

Implemented and available, but not run by default.

Only execute if explicitly requested.

---

# Current Technical Debt

The following issues should eventually be removed or refactored.

* [ ] Duplicate model implementations.
* [ ] Hard-coded assumptions.
* [ ] Legacy long-short implementation.
* [ ] Components not following factory pattern.
* [ ] Missing documentation.
* [ ] Missing tests.
* [ ] Legacy notebooks that duplicate functionality.

Do not remove working code until the replacement has been validated.

---

# Execution Priorities

When multiple tasks are possible, prioritize:

1. Mathematical correctness.
2. Numerical stability.
3. Code quality.
4. Computational efficiency.
5. Documentation.
6. Report quality.

Never prioritize speed over correctness.

---

# Blocked

Use this section to record why a task cannot currently move forward.

* [ ] No active blockers.

---

# Definition of Done

A feature is considered complete only when:

* [ ] Mathematical implementation is correct.
* [ ] Unit tests pass when available.
* [ ] Documentation is updated.
* [ ] Module walkthrough has been updated.
* [ ] Artifacts are saved correctly.
* [ ] Reports use the new functionality when relevant.
* [ ] Existing functionality remains compatible unless intentionally deprecated.

---

# Updating This Document

This document should evolve throughout the project.

---

# Multi-Location Experiment Status (2026-07-06)

Six locations have been run through the full ZA-GAS Stages 1–4 pipeline.

## Completed

| Location | Stage 1 winner | Stage 3 accepted | Stage 4 accepted | Notes |
|---|---|---|---|---|
| Belo Horizonte | phixi_seasonal_diagfi | Yes (harvey_enso_90d_30d) | Not run | Stage 4 pending |
| Cruzeiro do Sul | phixi_short_diagfi (OOS) | No | No (4 timeout) | fullfi loglik winner has NaN CRPS |
| Darwin Airport | stage3_harvey_no_enso | Yes | No (unit diverge) | Stage 2 CRPS diverged |
| Garanhuns | phixi_short_unit (OOS) | No | No | fullfi loglik winner has NaN CRPS |
| Manaus | phixi_seasonal_unit | No | No (diagfi diverge, Bug6b post-hoc) | Unit fallback not attempted (pre-fix) |
| Salvador | phixi_seasonal_unit | No | Pending (unit batch running) | |

## Known Bugs Fixed During This Experiment

1. **Bug 5** — `_gb2_fi_cross` ZeroDivisionError for Harvey models with fullfi scaling
2. **Bug 6** — Stage 4 always uses diagfi even when Stage 1 winner is unit-scaled
3. **Bug 6b** — Fallback check accepted finite-but-catastrophically-bad CRPS as "success"

## Next Steps (Prioritised)

* [ ] Generate final multi-location report (`python generate_report_extended.py`)
* [ ] Re-run Garanhuns Stage 3 with Bug 5 fix applied (harvey_enso_90d_30d may now converge)
* [ ] Run Belo Horizonte Stage 4 (Harvey model base, unit scaling)
* [ ] Run Manaus Stage 4 with unit scaling (Bug 6b fix now in place)
* [ ] Add Toronto (temperate climate diversity) — run_id: run_20260705_toronto
* [ ] Evaluate fullfi OOS stability fix (parameter guard in GB2 PPF call)

## Deferred (Out of Scope for Current Experiment)

* São Paulo — climatologically similar to Belo Horizonte; deprioritised to avoid duplication
* Riyadh — data file quality issues; excluded entirely

Completed tasks should be checked and retained as part of the project's development history.

Do not add speculative future research, commercial ideas, startup plans, or unrelated applications.

The roadmap should remain focused on the software currently being built.

---

# Pi-Dynamics Alternatives Experiment (2026-07-14)

A Stage 1 side-experiment (not part of the Stage 1-4 pipeline objectives):
compares four occurrence-probability ($\pi_t$) dynamics specifications --
the existing AR-logistic model (seasonal lags), an AR-logistic model with
short lags {1,2,3}, a no-AR short-lag variant, and a model where $\eta_t$
is driven by the frozen magnitude GAS(p,q) scale path $\varphi_{t|t-1}$ --
at all six locations, entirely in-sample.

Implemented, not yet run (per docs/EXECUTION.md, execution requires user
authorization after code review):

* `pi_dynamics/ar_logistic_custom_lags.py` -- configurable short-lag /
  no-AR occurrence dynamics.
* `pi_dynamics/phi_linked.py` -- phi-linked occurrence dynamics.
* `diagnostics/occurrence_pit.py` -- randomised Bernoulli PIT/quantile
  residuals.
* `diagnostics/pi_dynamics_eval.py` -- in-sample RMSE/CRPS evaluation
  against a frozen magnitude baseline.
* `run_pi_dynamics_alternatives.py` -- fits all four variants at all six
  locations, saves artifacts under `artifacts/pi_dynamics_experiment/`.
* `generate_pi_dynamics_report.py` -- builds `reports/pi_dynamics_alternatives/`
  (tables, figures, report.tex/PDF) from those artifacts only.

None of the existing Stage 1-4 code, artifacts, or reports were modified to
build this experiment.

---

# Multi-Location Stage 1-4 Diagnostic Mosaics (2026-07-31)

A reporting-only deliverable (not a new experiment): cross-location mosaic
figures -- PIT histogram, normal QQ plot, and 400-lag quantile-residual
ACF -- showing each stage's own winning model (Stage 1 baseline GAS,
Stage 2 weather covariates, Stage 3 Harvey long-short, Stage 4
tail-sensitive xi regime) at all six completed locations, side by side.
Stages 1-3 are shown for both the phi+xi branch and the phi-only branch
separately (21 mosaic PNG/PDF files total: 7 stage/branch combinations x
3 chart types). Every stage's own winner is shown regardless of whether it
became the pipeline's overall final model.

Implemented, not yet run (per docs/EXECUTION.md, execution requires user
authorization after code review):

* `diagnostics/plots.py` -- added `pit_histogram_mosaic`, `qq_plot_mosaic`,
  `acf_mosaic_400` (cross-location grid builders; ACF fixed to y in
  [-1, 1] with a shaded +/-2/sqrt(n) null band per panel).
* `diagnostics/multilocation_mosaics.py` -- stage/branch winner resolution
  (`resolve_stage_winner`, `resolve_stage4_winner`) and per-location IS
  diagnostic assembly, reusing `generate_report_extended.py`'s existing
  model-reconstruction/CDF logic rather than duplicating it.
* `multilocation_diagnostic_mosaics.ipynb` -- orchestration notebook;
  writes to `reports/multi_location/mosaics/`.

No existing Stage 1-4 code, artifacts, or the main multi-location report
were modified to build this.
