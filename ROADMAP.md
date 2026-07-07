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
