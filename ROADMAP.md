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

---

# Harvey Long-Short Persistence/Initialisation Diagnostics (2026-08-31)

A diagnostic side-experiment (not part of the Stage 1-4 pipeline
objectives): isolates the MODELS.md §20 Harvey persistence restriction
(0 < phi_S,j < phi_L,j < 1) and the initialisation strategy for
phi_S,j/phi_L,j/kappa_L,j/kappa_S,j/Gamma_L,j/Gamma_S,j from every other
numerical choice in the production Stage-3 pipeline, using a minimal
single-lag configuration (X_short = dewpoint_lag1 only, X_long =
enso_dummy_current only) at all six completed locations, separately for
the phi-only and phi+xi branches: 2 persistence configurations (full
restriction vs. dropping only the phi<1 upper bound) x 3 Gamma_L
initial-value variants (1, 0.2, 0.02) x 2 branches x 6 locations = 72 fits.
A separate, pure-artifact-reuse report also compares existing Stage 1/2
short-vs-seasonal-lag out-of-sample RMSE/CRPS across all six locations and
both branches (no new fitting).

Implemented, not yet run (per docs/EXECUTION.md, execution requires user
authorization after code review):

* `models/harvey_gas.py` -- added `restrict_persistence_upper` and
  `use_auxiliary_penalties` constructor flags (both default `True`,
  reproducing prior behaviour exactly for every existing caller), the
  `persistence_init_from_filtered_path()` helper, and `overrides`/
  `static_overrides` parameters on `initial_theta()`.
* `docs/MODELS.md` §20 -- fixed an internal notation inconsistency (the
  "optimizer estimates" bullet list used A/B where the equations above it
  already used kappa/phi).
* `run_longshort_diagnostics.py` -- the 72-fit experiment; writes to a new
  `artifacts/longshort_diagnostics_experiment/` tree only.
* `generate_longshort_diagnostics_report.py` -- builds
  `reports/longshort experiment/long short diagnostics.tex`/`.pdf`.
* `generate_seasonality_experiment_report.py` -- builds
  `reports/multi_location/seasonality experiment/report.tex`/`.pdf`
  (no new fitting; can run immediately once authorized).

No existing Stage 1-3 code behaviour, artifacts, or the main multi-location
report were changed by this work.

---

# Pi-Dynamics Alternatives -- GAS(1,1) Magnitude Variant (2026-09-09)

A re-run of the 2026-07-14 pi-dynamics side-experiment with one change to
the underlying model. The observation model is still the zero-augmented
mixture M_t = D_t * Y_t, D_t ~ Bernoulli(pi_{t|t-1}),
Y_t ~ GB2(varphi, gamma, xi, zeta); the four occurrence-dynamics
alternatives (existing AR+seasonal, AR+short {1,2,3}, no-AR short {1,2,3},
phi-linked), their standalone estimation, the common in-sample window, and
every diagnostic are unchanged. The only difference: the frozen scale path
varphi_{t|t-1} now follows a freshly-estimated phi-only GAS(1,1)
(gas_lags=[1] -- one score coefficient A_varphi, one persistence
coefficient B_varphi) instead of a cached Stage 1 short/seasonal-lag GAS
artifact. The GAS(1,1) magnitude model is fit under all three score
scalings; the one frozen as the magnitude path is chosen by (1) the
most-preferred non-empty gradient tier (||grad||_inf < 1; < 100; not
failed; any), using the independent numerical_gradient_checked gradient at
the optimum, then (2) lowest AIC within that tier. New deliverable
alongside the original tables/figures: a per-location "gradient of the
optimised models" table (magnitude GAS(1,1) + all four pi_t fits,
per-parameter) plus a cross-location gradient-norm summary.

Implemented, not yet run (per docs/EXECUTION.md, execution requires user
authorization after code review):

* `pi_dynamics/standalone_fit.py` -- added the public alias
  `standalone_neg_loglik` (= the private `_neg_loglik`) so callers can
  evaluate the exact standalone objective at the fitted optimum for an
  independent gradient check. No behavioural change; `_neg_loglik` and
  `fit_pi_dynamics_standalone` are untouched.
* `run_pi_dynamics_gas11.py` -- the experiment; writes to a new
  `artifacts/pi_dynamics_gas11_experiment/` tree only. Per location: 3
  magnitude scaling fits (all saved under `candidates/`) + 4 standalone
  occurrence fits, each with a `numerical_gradient_checked` gradient.
* `generate_pi_dynamics_gas11_report.py` -- builds
  `reports/pi_dynamics_gas11/report.tex`/`.pdf` from those artifacts only
  (no optimisation), including the magnitude scaling-selection table, the
  gradient-of-the-optimised-models tables, and the cross-location
  gradient-norm summary.
* `generate_pi_dynamics_gas11_mosaics.py` -- builds two cross-location x
  cross-variant mosaics (`reports/pi_dynamics_gas11/mosaics/`): a normal
  QQ-plot grid and a 400-lag ACF grid of the occurrence quantile
  residuals, rows = the six locations, columns = the four pi_t
  alternatives (one grid each, not one figure per location). Reuses
  `diagnostics/plots.py`'s generic `stage_qq_grid_mosaic` /
  `stage_acf_grid_mosaic` (the "stages" axis carries the variant keys);
  `diagnostics/plots.py` is not modified -- the only cosmetic tweak
  (shorter left-column y-label, legend placement) is applied on the
  returned Figure inside the script.

RUN STATUS (2026-09-09): executed at all six locations. Every fit
converged (24/24 pi_t fits `grad_converged=true`; every GAS(1,1)
magnitude fit selected with a checked gradient). Magnitude score-scaling
picked per location: unit (Belo Horizonte, Darwin, Manaus), diagonal
inverse Fisher (Cruzeiro do Sul, Garanhuns, Salvador). Occurrence tally
unchanged from the 2026-07-14 experiment: variant B (AR + short lags
{1,2,3}) wins dry/wet separation, CRPS and RMSE at 4/6 locations, A at
2/6, C and D never. The ACF mosaic shows a sub-annual autocorrelation
wave at Darwin and Garanhuns for variants C (no AR term) and D
(phi-linked), i.e. unmodelled annual recurrence in the occurrence process
when the AR feedback term is dropped.

No existing pi_dynamics_alternatives code, artifacts, or report were
modified; the 2026-07-14 experiment is untouched.

---

# Pi-Dynamics GAS(1,1) -- 8y-fit / 2y-validation Split (2026-09-10)

Same four occurrence alternatives (A..D), same GAS(1,1) phi-only magnitude
model, same score-scaling selection rule as the 2026-09-09 experiment, but
the whole-series in-sample evaluation is replaced by a held-out protocol:
the training series (exactly 10 calendar years at every location) is split
by date into the first 8 years (fit) and the final 2 years (validation).
Every model -- the magnitude GAS(1,1) and each standalone pi_t fit -- is
estimated only on the fit block. The summary table reports RMSE / CRPS /
dry-wet occurrence split on the validation block, 1-step-ahead, no re-fit
(magnitude varphi via ZAGASModel.simulate_oos, each variant's pi via
eta_path_full over concat(y_fit, y_val)). The *_test.csv held-out set is
not touched.

Executed at all six locations (2026-09-10). Every fit converged
(gradients checked, all `grad_converged=true`). Magnitude scaling picked:
unit at Belo Horizonte / Cruzeiro do Sul / Darwin / Garanhuns / Manaus,
diagonal inverse Fisher at Salvador. On *validation* CRPS the ranking
FLIPS relative to the in-sample tally: variant A (existing AR + seasonal
lags) is best at 5/6 locations (all but Darwin, where B wins); on
validation RMSE, A wins at 4/6 and C (no-AR short lags) at Darwin and
Salvador. B (AR + short lags), which dominated in-sample, wins nothing on
validation CRPS. The QQ/ACF mosaics (now over the 8-year fit block, n=2921)
keep the earlier qualitative picture: C and D leave a sub-annual
autocorrelation wave at Darwin and Garanhuns; A and B stay in the null
band.

Files:

* `run_pi_dynamics_gas11_validation.py` -- the experiment; writes to a new
  `artifacts/pi_dynamics_gas11_validation_experiment/` tree only. Reuses
  the magnitude-model helpers from `run_pi_dynamics_gas11.py` verbatim
  (they take the fit series as an argument).
* `generate_pi_dynamics_gas11_summary.py` -- writes
  `reports/pi_dynamics_gas11/tables/summary/summary_of_alternatives.tex`
  and `.csv` (snake_case columns, `pd.read_csv`-ready) from the validation
  metadata. Replaces the earlier hand-made A/B-only summary file with an
  A--D, validation-basis version.
* `generate_pi_dynamics_gas11_mosaics.py` -- `ARTIFACTS_DIR` retargeted to
  the validation tree; now also has `SHOW_SUPTITLE` / `SHOW_LEGEND` knobs
  (both `False`) so the mosaics render as bare panel grids.

The earlier whole-series `artifacts/pi_dynamics_gas11_experiment/` tree and
`reports/pi_dynamics_gas11/report.pdf` are left in place but are now stale
relative to the summary table and mosaics, which were overwritten on the
validation basis.

---

# Marginal Effects of External Covariates (2026-09-16)

A reporting/diagnostics deliverable (`docs/claude_code_marginal_effects_instructions.md`),
not a new experiment: analytical one-step-ahead and paired-CRN-Monte-Carlo
multi-step-ahead marginal effects of dew point (DPT), air temperature (AT),
and ENSO on the ZA-GB2 rolling forecast mean/variance, for the selected
GAS(p,q) and Harvey long-short model at each of the 6 completed locations,
separately for 1-TVP ($\varphi$) and 2-TVP ($\varphi,\xi$) branches.

Model selection (`marginal_effects/model_selection.py`) applies the
instructions doc's rule (grad_norm_inf<1, prefer a 3-variable
DPT+AT+ENSO config, minimize OOS CRPS then RMSE) to
`artifacts/stage_comparison_20260911/metrics.json`. 22 of the 24
(location x branch x family) cells have a usable model; two (Cruzeiro do
Sul and Manaus, 2-TVP Harvey) have none clearing the gradient bar at any
covariate config and are left absent, per the user's own instruction not
to use badly-optimized models. Two further cells whose literal
RMSE/CRPS-minimising model had ZERO external covariates (Darwin 1-TVP
Harvey; Cruzeiro do Sul 2-TVP standard GAS) were, on the user's explicit
2026-09-16 decision, replaced by the best covariate-bearing grad<1
alternative instead, at a small RMSE/CRPS cost.

New analytical derivative formulas (`GB2LogLink.R_moment_ratio`,
`GB2LogLink.psi_diff`, and the one-step marginal-effect chain-rule
derivations in `marginal_effects/onestep.py`) are proved in
`docs/math_proofs.tex` Sec.11 (also fixed a pre-existing, unrelated
`\newcommand{\psi}` duplicate-definition bug in that file that prevented
it from compiling at all).

The multi-step design was revised with the user 2026-09-16 (overriding the
instructions doc's literal "every rolling forecast origin" reading, at the
user's explicit request): ONE forecast origin per model -- the end of the
in-sample training period, not a rolling scan through the test period --
and "5 predictions per location": a shared baseline path plus 4 named
one-time-impulse scenarios (DPT +1 degC, AT +1 degC, ENSO state switch to
El Nino, ENSO state switch to La Nina -- El Nino/La Nina/Neutral being
mutually exclusive categorical states), all built on the same common-
random-numbers baseline draw. This cut the compute cost from a multi-hour
rolling-origin scan to ~22 models x 5 simulations x ~35s (~1h single
process, a few minutes parallelised).

Implemented, **not yet run** (per docs/EXECUTION.md, execution requires
user authorization after code review):

* `data/station_loader.py` -- purely additive: `load_all()` now also
  returns `covariate_scalers` (per-block training mean/std), needed to
  convert a raw-unit covariate perturbation into the standardised units
  the fitted Gamma coefficients actually multiply. No existing key/value
  changed.
* `distributions/gb2_log_link.py` -- added `R_moment_ratio(k,...)` and
  `psi_diff(k,...)`, the GB2 moment-ratio and digamma terms the
  marginal-effect formulas need; both derived from the existing
  `raw_moment` machinery, no new distributional assumptions.
* `marginal_effects/` (new package) -- `model_selection.py` (the 24-cell
  selection above), `model_loader.py` (reconstructs a selected fitted
  model + theta + data with NO re-optimisation, reusing
  `run_rerun_10092026.py`'s / `run_stage34_phiL_sigmoid.py`'s own
  construction code), `coefficients.py` (raw-unit $c_v^\varphi, c_v^\xi,
  c_v^\pi$ extraction), `onestep.py` (analytical rolling one-step
  effects, unaffected by the multi-step redesign above), `multistep.py`
  (the single-origin/5-scenario paired-CRN Monte Carlo effects, reusing
  `diagnostics/multistep_forecast.py`'s replay/simulate-forward
  machinery), `tables.py` (LaTeX table builders).
* `run_marginal_effects_onestep.py` -- writes
  `artifacts/marginal_effects/onestep/`; cheap (~seconds total, purely
  analytical, no simulation).
* `run_marginal_effects_multistep.py` -- writes
  `artifacts/marginal_effects/multistep/`; one task per available model
  (5 simulations each) -- resumable, checkpointed per (location, branch,
  family), parallel over independent tasks.
* `generate_marginal_effects_report.py` -- builds
  `reports/marginal effects/report.tex`/`.pdf` (Tables 1-6 of the
  instructions doc, plus a supplementary baseline-forecast table) from the
  two artifact trees above only, no computation.

Verified this session with small-scale integration smoke tests (few
hundred to few thousand Monte Carlo paths, a handful of representative
models) confirming the full model-selection -> simulation -> CSV -> LaTeX
pipeline runs correctly end-to-end; no full-scale (10,000-path, all-22-
model) run has been performed.

No existing Stage 1-5 code, artifacts, or any other report were modified
by this work, aside from the two purely-additive library changes listed
above.
