"""
Marginal effects of external covariates (dew point "DPT", air temperature
"AT", ENSO) on the ZA-GB2 rolling forecasts, for the standard-GAS(p,q) and
Harvey long-short model families, at every fitted location.

=============================================================================
OVERVIEW
=============================================================================

This package answers one question per (location, model family, number of
time-varying parameters): given the SELECTED fitted model (docs/MODELS.md's
existing ZAGASModel / CovZAGASModel / HarveyZAGASModel, never re-implemented
here), how much does a one-unit change in an external covariate move the
one-step-ahead conditional mean and variance of rainfall (analytically), and
how much does it move the terminal and cumulative h-step-ahead forecast
(via paired Monte Carlo simulation)?

It never fits a model. Every computation here starts from a cached theta
(artifacts/rerun_10092026[_phiL_sigmoid]/<loc>/<stage>/<cfg>/theta.npy) and
the model construction that produced it -- see model_loader.py.

=============================================================================
CODE WALKTHROUGH
=============================================================================

model_selection.select_all()
    Reads artifacts/stage_comparison_20260911/metrics.json (already computed
    by collect_stage_comparison_metrics.py -- no refitting) and applies the
    docs/claude_code_marginal_effects_instructions.md "TAKE CARE" rule
    (grad_norm_inf < 1, prefer 3-variable dew_temp_enso configs, minimize
    OOS CRPS then RMSE) to pick one model per (location, family, branch).
    |
model_loader.load(selection_entry)
    Reconstructs the exact fitted model object + theta + input data (reusing
    run_rerun_10092026.py's / run_stage34_phiL_sigmoid.py's own construction
    code, never a re-implementation) -- returns everything onestep.py and
    multistep.py need.
    |
coefficients.covariate_coefficients(model, theta)
    Reads off c_v^phi, c_v^xi (raw-physical-unit, converting through the
    training standardisation moments) for DPT/AT/ENSO from the model's own
    decoded Gamma / Gamma_L / Gamma_S arrays.
    |
onestep.rolling_one_step_effects(...)
    Analytical marginal effects (docs/claude_code_marginal_effects_instructions.md
    Sec.1-2) at every OOS rolling forecast origin, from the model's own
    filtered one-step-ahead states (model.simulate_oos).
    |
multistep.multistep_effects(...)
    Paired common-random-numbers Monte Carlo simulation (Sec.3), reusing
    diagnostics/multistep_forecast.py's replay/simulate-forward machinery,
    for horizons {1,2,5,10,30,60,90}.
    |
tables.py / generate_marginal_effects_report.py
    Build the required LaTeX tables from the CACHED onestep/multistep
    outputs only (docs/REPORTING.md: reports never trigger computation).
"""
