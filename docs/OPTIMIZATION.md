# OPTIMIZATION.md

# Optimization, Numerical Stability, and Model Acceptance

This document defines the optimization philosophy for the score-driven modelling framework.

It should be read together with:

* `docs/ARCHITECTURE.md`
* `docs/MODELS.md`
* `docs/EXECUTION.md`
* `docs/REPORTING.md`

The purpose of this document is to make optimization reproducible, diagnosable, and safe for expensive nonlinear likelihood models.

---

# 1. General Philosophy

Optimization is part of the statistical model implementation.

A model estimate should not be accepted or rejected only because an optimizer returns a success or failure flag.

Nonlinear likelihood models may produce useful and stable estimates even when the optimizer reports failure, especially when the likelihood surface is flat, numerical gradients are noisy, or stopping tolerances are strict.

Therefore, optimization status must always be interpreted together with post-fit numerical and statistical diagnostics.

---

# 2. Preferred Optimizer

The preferred production optimizer is:

```python
method = "BFGS"
```

using penalties for constraints whenever possible.

Hard bounds should be avoided unless they are mathematically unavoidable or explicitly requested.

BFGS is preferred because:

* it is flexible for unconstrained penalized likelihoods;
* it returns an approximate inverse Hessian;
* it works well with smooth objectives;
* it avoids bound-induced artifacts when penalties are carefully designed.

---

# 3. Bounds and Penalties

The preferred approach is:

```text
unconstrained optimization + penalty terms
```

rather than hard parameter bounds.

Penalties should be used for:

* invalid parameter domains;
* stationarity restrictions;
* explosive filtered states;
* invalid Fisher matrices;
* non-finite likelihoods;
* non-positive distribution parameters if applicable;
* invalid predictive quantiles.

Hard bounds may be used only when:

* the optimizer repeatedly enters impossible regions;
* penalties are insufficient;
* numerical safety requires it;
* or the user explicitly requests bounded optimization.

If hard bounds are introduced, document why.

---

# 4. Objective Function

The optimization objective is the negative log-likelihood,

[
Q(\theta)
=========

-\ell(\theta),
]

where the log-likelihood is computed from the one-step-ahead predictive distribution.

For zero-augmented models,

[
\ell(\theta)
============

\sum_t
\log p(y_t\mid\mathcal F_{t-1};\theta).
]

The optimizer should always optimize the likelihood itself whenever the current parameter vector satisfies the model assumptions.

The optimization routine should never evaluate invalid parameterizations. Instead, before evaluating the likelihood, verify that the proposed parameter vector satisfies all mathematical and numerical requirements of the model.

Examples include:

* valid parameter domains;
* stationarity restrictions;
* persistence restrictions;
* ordered parameters when required;
* finite transformed parameters;
* valid distribution parameters;
* valid state recursion.

If any of these conditions are violated, **do not compute the likelihood**. Instead, immediately return a large finite objective value (for example (10^{12})) together with any safe default values required by the optimizer.

Avoid returning `NaN` or `inf` whenever a large finite value is sufficient to steer the optimizer away from invalid regions.

These large objective values are **numerical safeguards**, not statistical penalty terms. They are used only to prevent the optimizer from exploring impossible parameterizations and should not be interpreted as part of the statistical model.

Consequently, statistical inference—including approximate standard errors, Hessian-based diagnostics, and Fisher-information calculations—should always be based on the likelihood itself rather than on these numerical safeguards whenever possible.


---

# 5. Numerical Safety During Filtering

During likelihood evaluation, monitor whether filtered states remain finite and reasonable.

Immediately penalize or reject evaluations where:

* a state becomes `NaN`;
* a state becomes infinite;
* a transformed parameter leaves the distribution domain;
* predictive probabilities are invalid;
* predictive quantiles are not ordered;
* Fisher information is singular or non-positive when needed;
* likelihood contributions are non-finite.

The objective should fail safely.

A single invalid iteration should not crash the entire experiment unless the error indicates a programming bug.

---

# 6. Parameter Restrictions

Parameter restrictions should be explicit.

Examples:

Standard GAS persistence should be controlled so that state dynamics do not explode.

For Harvey long-short models, the persistence restriction is:

[
0 < B_S < B_L < 1.
]

This should be enforced by penalties or a stable reparameterization.

If a reparameterization is used, document the mapping clearly.

Also, GB2 required moments only exist if
[
    $exp(\zeta) > 4*exp(\gamma)$
]

---

# 7. Starting Values

Starting values matter.

The optimization pipeline should support structured initialization.

Preferred initialization hierarchy:

1. Use known stable default starting values.
2. Use estimates from simpler models when available.
3. Use estimates from nested models when moving to richer specifications.
4. Use previous stage winners as starting values for dependent stages.
5. Use multiple starts only when necessary.

Examples:

* initialize seasonal GAS from short-lag GAS when possible;
* initialize weather covariate models from best no-covariate GAS;
* initialize Harvey long-short models from the selected standard GAS/weather model when possible;
* initialize phi-xi models from phi-only models where appropriate.

Every starting value strategy should be saved in metadata.

---

# 8. Optimizer Output to Save

For every optimization run, save:

* optimizer method;
* starting values;
* final parameter vector;
* final objective value;
* final log-likelihood;
* penalty value;
* number of iterations;
* number of function evaluations;
* optimizer message;
* optimizer success flag;
* gradient if available;
* gradient norm;
* approximate inverse Hessian if available;
* runtime;
* memory usage if monitored;
* warnings;
* retry history;
* post-fit validity class.

Do not discard failed runs.

A failed run can be useful for debugging and for selecting better starting values.

---

# 9. Hessians and Standard Errors

Do **not** compute a fresh numerical finite-difference Hessian for every model.

This is too expensive when the number of parameters grows.

Instead:

* for BFGS, use `result.hess_inv` as a quasi-Newton approximation to the inverse Hessian;
* save it whenever available;
* compute approximate standard errors from it when meaningful;
* clearly label these standard errors as approximate.

Numerical Hessians should only be computed for final selected models and only when explicitly requested.

Standard errors should be classified as:

* reliable;
* approximate;
* unreliable;
* unavailable.

A standard error should be marked unreliable if:

* the optimizer did not stabilize;
* the inverse Hessian is ill-conditioned;
* the diagonal contains negative or non-finite entries;
* penalties dominate the objective;
* the gradient norm remains large;
* the model is classified as failed.

---

# 10. Gradient Diagnostics

The optimizer success flag is not enough.

Always save and inspect the gradient when available.

Useful diagnostics include:

[
|\nabla Q(\hat\theta)|_\infty
]

and

[
|\nabla Q(\hat\theta)|_2.
]

A model with `scipy.success=False` may still be valid if:

* the objective has stabilized;
* the gradient norm is small or acceptable;
* post-fit diagnostics are good;
* restarting from the final parameters does not materially improve the objective.

A model with `scipy.success=True` may still be problematic if:

* filtered states are unstable;
* forecasts are invalid;
* PIT values are degenerate;
* the Hessian approximation is unusable;
* parameter estimates are nonsensical.

---

# 11. Model Acceptance Classes

Each fitted model should be assigned a post-fit validity class.

## 11.1 Valid Converged

A model is `valid_converged` when:

* optimizer reports success;
* likelihood is finite;
* parameters are finite;
* filtered states are stable;
* gradient norm is acceptable;
* predictive quantities are finite;
* diagnostics do not reveal major failures.

## 11.2 Valid With Warning

A model is `valid_with_warning` when:

* optimizer reports failure or warning;
* but likelihood is finite;
* parameters are finite;
* filtered states are stable;
* objective has stabilized;
* gradient norm is acceptable or only mildly elevated;
* predictions are valid;
* PIT, coverage, CRPS, and tail metrics are reasonable;
* restart/polish does not materially improve the objective.

This class is important.

Do not discard a model only because `scipy.success=False`.

## 11.3 Failed

A model is `failed` when one or more of the following occurs:

* non-finite likelihood;
* non-finite parameters;
* exploding filtered states;
* invalid predictive distribution;
* invalid quantiles;
* repeated optimizer failure with no stable objective;
* large gradient norm after retries;
* model cannot produce diagnostics;
* memory or runtime safety limits are exceeded;
* forecasts are obviously nonsensical.

Failed models should be saved with metadata but excluded from primary ranking tables unless a failure table is being shown.

---

# 12. Polish Step

A polish step should be available.

After the main optimization:

1. take the final parameter vector;
2. restart the optimizer from that vector;
3. use the same model specification;
4. optionally use slightly tighter tolerances;
5. compare the objective improvement.

If the relative improvement is negligible, the original solution can be accepted as stable even if the optimizer reported failure.

Suggested relative improvement threshold:

[
\frac{|Q_{old}-Q_{new}|}{1+|Q_{old}|} < 10^{-6}.
]

This threshold should be configurable.

---

# 13. Retry Strategy

The optimization pipeline should support controlled retries.

Possible retry sequence:

1. baseline BFGS with standard penalties;
2. BFGS restart from final parameters;
3. BFGS with stronger penalties;
4. initialization from simpler nested model;
5. optional bounded optimizer only if explicitly allowed;
6. mark failed if still unstable.

Retries must be logged.

Do not silently replace a failed run with a successful fallback without recording what happened.

---

# 14. Penalty Escalation

If instability occurs, penalty escalation may be used.

Examples:

* stronger penalty for persistence close to or above one;
* stronger penalty for invalid distribution parameters;
* stronger penalty for explosive filtered states;
* stronger penalty for invalid Fisher information;
* stronger penalty for non-finite likelihood contributions.

Penalty escalation should not change the scientific model.

It should only improve numerical stability.

If penalty escalation changes the effective feasible region, document this clearly.

---

# 15. Runtime Monitoring

Every model fit should record runtime.

The pipeline should support soft and hard runtime thresholds.

Suggested defaults:

* soft warning: 30–45 minutes per model;
* hard timeout: 90 minutes per model for exploratory pipeline runs;
* final selected models may use longer limits if explicitly requested.

These thresholds should be configurable.

If a model exceeds the soft threshold, log a warning.

If a model exceeds the hard threshold, stop it cleanly if possible, save partial diagnostics, and classify it as failed or interrupted.

A global pipeline timeout may also be used, but per-model monitoring is more informative.

---

# 16. Memory Monitoring

Every model fit should monitor memory where practical.

Suggested policy:

* log memory usage per model;
* warn if memory use is unexpectedly high;
* reduce worker count if multiprocessing causes memory pressure;
* stop a model if it exceeds a configured memory safety threshold.

For a 32 GB RAM machine, do not allow multiprocessing to consume nearly all RAM.

The pipeline should reserve memory for the operating system and interactive work.

If memory cannot be monitored reliably on the platform, log that memory monitoring is unavailable.

---

# 17. Progress Monitoring

When possible, log objective progress during optimization.

Useful quantities:

* current iteration;
* current objective;
* current penalty;
* gradient norm;
* best objective so far;
* elapsed time;
* number of invalid objective evaluations.

This helps detect stuck models.

A model may be stopped early if:

* objective is non-finite repeatedly;
* filtered states explode repeatedly;
* objective has not improved meaningfully for many iterations;
* runtime limit is exceeded;
* memory limit is exceeded.

Stopping rules must be configurable.

---

# 18. Multiprocessing Safety

Optimization may be run in parallel only for independent models.

Parallel jobs must write to separate artifact directories.

Never allow two processes to write to the same model output file.

Multiprocessing should affect runtime only.

It must not change:

* parameter estimates;
* random seeds;
* model definitions;
* optimization settings;
* evaluation metrics.

Sequential execution must always remain available for debugging.

---

# 19. Checkpointing

Every completed model should be checkpointed immediately.

Do not wait until the full stage finishes to save results.

If a long pipeline is interrupted, already completed models should be reusable.

Checkpoint at least:

* model configuration;
* final parameters;
* optimizer output;
* post-fit diagnostics;
* forecasts;
* metrics.

If a model fails, checkpoint the failure metadata.

---

# 20. Reproducibility

Every optimization run should be reproducible from saved configuration and data references.

Save:

* code version if available;
* random seed if randomness is used;
* model specification;
* distribution name;
* dynamic parameters;
* score scaling method;
* covariates used;
* lag structure;
* train/test split;
* optimizer settings;
* penalty settings.

Do not rely on implicit notebook state.

---

# 21. Optimization and Model Comparison

Optimization diagnostics and forecasting diagnostics answer different questions.

A model with slightly worse optimizer diagnostics may still forecast better.

A model with perfect optimizer success may still forecast poorly.

Therefore, model comparison should combine:

* post-fit validity class;
* likelihood and information criteria;
* calibration;
* CRPS;
* tail scores;
* quantile scores;
* coverage;
* OOS forecasting metrics.

Do not rank models using optimizer status alone.

---

# 22. Handling SciPy Failure Messages

SciPy failure messages should be saved but not treated as final judgement.

Common harmless or semi-harmless reasons for failure include:

* precision loss;
* maximum iterations reached near optimum;
* noisy numerical gradient;
* flat likelihood surface;
* strict tolerance;
* small objective improvements below tolerance;
* approximate Hessian instability.

These should trigger inspection, not automatic rejection.

The post-fit validity classification determines how the model is used.

---

# 23. Expensive Diagnostics

Some diagnostics are expensive.

Default policy:

* compute essential metrics for all valid models;
* compute full diagnostic plots for selected models;
* compute expensive numerical Hessians only for final selected models and only when explicitly requested.

Do not make the pipeline slow by computing unnecessary diagnostics for every candidate model.

However, save enough outputs so diagnostics can be computed later without rerunning optimization.

---

# 24. Standard Error Reporting

Report standard errors only when they are meaningful.

For broad model comparison tables, standard errors are usually secondary.

For final selected models, standard errors are more important.

When reporting standard errors, include a note describing whether they come from:

* quasi-Newton inverse Hessian;
* numerical Hessian;
* robust/sandwich estimator;
* unavailable source.

If using quasi-Newton inverse Hessian, label them approximate.

---

# 25. Failure Tables

Reports should include failure summaries when relevant.

A failure summary may include:

* model id;
* optimizer message;
* validity class;
* reason for failure;
* whether retry was attempted;
* whether partial outputs were saved.

This is useful for transparency and debugging.

---

# 26. Recommended Defaults

Recommended default optimization configuration:

```python
optimizer = "BFGS"
use_hard_bounds = False
use_penalties = True
compute_numerical_hessian = False
save_hess_inv = True
polish_solution = True
max_retries = 2
runtime_soft_warning_minutes = 45
runtime_hard_timeout_minutes = 90
monitor_memory = True
```

These defaults may be overridden by experiment configuration.

---

# 27. Things To Avoid

Avoid:

* computing numerical Fisher information at each iteration;
* computing numerical Hessians for every model;
* discarding models only because `scipy.success=False`;
* accepting models only because `scipy.success=True`;
* running huge model grids without checkpointing;
* allowing one failed model to stop the entire pipeline;
* silently changing model specifications during retries;
* overwriting previous optimization outputs;
* hard-coding optimizer settings inside model classes;
* hiding penalty settings from metadata.

---

# 28. Final Principle

Optimization is not a black box.

Every model estimate should be reproducible, diagnosable, and interpretable.

The pipeline should make it clear:

* what was optimized;
* how it was optimized;
* whether optimization was numerically stable;
* whether the resulting model is statistically usable;
* and whether the model should enter scientific comparisons.
