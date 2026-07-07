# CLAUDE.md

# Repository Constitution

This document defines how Claude Code should work inside this repository.

It is **not** intended to duplicate the documentation contained in `docs/`.

Instead, it defines the operational rules that should be followed before modifying the codebase.

The detailed specifications are contained in the documentation files.

---

# Repository Purpose

This repository develops a reusable scientific framework for score-driven (GAS/SD) probabilistic models.

Although the current application is precipitation modelling, the framework should remain application-agnostic whenever possible.

The repository should evolve into a reusable scientific library rather than a collection of thesis scripts.

Scientific correctness, reproducibility, modularity and maintainability take priority over short-term convenience.

---

# Before Writing Any Code

Before modifying any code, read the repository documentation in the following order:

1. `docs/ARCHITECTURE.md`
2. `docs/MODELS.md`
3. `docs/OPTIMIZATION.md`
4. `docs/EXECUTION.md`
5. `docs/REPORTING.md`
6. `docs/DOCUMENTATION.md`
7. `ROADMAP.md`

Do not begin implementation until these documents have been understood.

If implementation conflicts with documentation,

**stop**

and explain the inconsistency before changing either the code or the documentation.

Never silently change repository architecture.

---

# Development Philosophy

The objective is to build a reusable scientific modelling framework.

Always prefer:

* correctness over speed;
* analytical methods over numerical approximations;
* modularity over duplication;
* reusable interfaces over application-specific code;
* readability over clever implementations.

Avoid unnecessary complexity.

Implement only what is required for the current milestone.

---

# Existing Code

Always inspect existing code before implementing new functionality.

Do not assume the current implementation is incorrect.

Determine whether functionality already exists before creating new modules.

When changes are necessary:

* refactor rather than duplicate;
* preserve working functionality whenever possible;
* remove obsolete implementations only after the replacement has been validated.

Do not create files such as

* `model_v2.py`
* `model_new.py`
* `experiment_final.py`

Integrate functionality into the existing architecture.

---

# Repository Organization

The repository is organized as follows:

* `docs/` contains the authoritative technical documentation.
* `src/` contains reusable implementation.
* `artifacts/` contains saved experimental outputs.
* `reports/` contains generated reports.
* `examples/` contains minimal usage examples.
* notebooks orchestrate experiments but should not contain reusable implementation.

Business logic belongs inside reusable modules.

---

# Documentation

Documentation is part of the implementation.

Every major module should contain:

1. Module docstring.
2. Code walkthrough.
3. Mathematical overview.
4. Important implementation notes.

Complex modules should begin with a short execution-flow description before the implementation.

Refer to `docs/DOCUMENTATION.md` for documentation standards.

---

# Code Walkthrough Requirement

Every important module should begin with a short walkthrough explaining the algorithm at a conceptual level.

The walkthrough should describe:

* execution flow;
* responsibilities of major stages;
* interaction with other modules.

It should help a researcher understand the implementation before reading the source code.

---

# Mathematical Implementation

Mathematical specifications are defined in

`docs/MODELS.md`.

Do not implement alternative mathematical formulations unless explicitly requested.

Whenever analytical expressions exist,

prefer them over numerical approximations.

In particular:

* analytical score;
* analytical Fisher information;
* analytical derivatives;
* analytical moments.

Avoid numerical differentiation whenever analytical expressions are available.

---

# Optimization

Optimization rules are defined in

`docs/OPTIMIZATION.md`.

In summary:

* use the preferred optimizer specified there;
* avoid numerical instability;
* classify convergence independently of optimizer success;
* save optimizer outputs;
* avoid unnecessary numerical Hessians;
* use numerical safeguards rather than allowing invalid evaluations.

---

# Execution

Execution rules are defined in

`docs/EXECUTION.md`.

Important principles:

* implement models independently from execution;
* execute only models requested by the notebook or configuration;
* checkpoint after every completed model;
* reuse artifacts whenever possible;
* isolate failures;
* support resumable execution.

The execution engine should follow dependency-aware execution rather than blindly evaluating all possible models.

---

# Reporting

Reporting rules are defined in

`docs/REPORTING.md`.

Reports should:

* be generated entirely from saved artifacts;
* never trigger optimization;
* remain editable through LaTeX;
* produce reproducible figures and tables;
* follow a coherent scientific narrative.

Every generated report should also produce reusable figures and tables.

---

# Current Development

Current engineering priorities are maintained in

`ROADMAP.md`.

Do not implement speculative features that are not part of the current milestone unless explicitly requested.

Implement features that are part of the framework even if they are disabled by default (for example, regime models), but only execute them when requested.

---

# Configuration Philosophy

Framework behaviour should be determined by configuration rather than duplicated implementations.

Examples include:

* dynamic parameter selection;
* score scaling;
* distributions;
* dynamic recursions;
* occurrence models;
* optimization settings;
* report generation.

Avoid creating separate implementations that differ only by configuration.

---

# Data Philosophy

Model implementations should remain independent of the underlying application.

Data loading, preprocessing and feature engineering belong to dedicated data modules.

Models should receive standardized model-ready inputs.

Avoid hard-coded:

* datasets;
* stations;
* variable names;
* file paths.

Input data will be described in file docs/DATA.md, you should read application specific input data paths and what dat arepresents from there, it will explicitly show what

[
    y_t
]

data is, that is, path to precipitation files data. Also will show

[
    X_t
]

that is, GAS(p,q) covariates

Also

[
    X_short
]

And

[
    X_long
]

For the Harvey long-short models

---

# Artifacts

Every completed model should save enough information to avoid repeating expensive estimation.

Save at least:

* configuration;
* fitted parameters;
* optimizer outputs;
* diagnostics;
* forecasts;
* evaluation metrics;
* residuals;
* state estimates;
* figures;
* tables.

Reports should use these artifacts rather than recomputing results.

---

# Multiprocessing

Parallel execution should only be used for independent tasks.

Dependent stages should remain sequential.

Parallel execution must never change numerical results.

Only execution time should change.

---

# Safety

The framework should monitor:

* runtime;
* memory usage;
* optimization progress;
* numerical stability;
* execution failures.

Problems should be logged.

Whenever possible, failed models should not interrupt the remainder of the execution pipeline.

---

# Code Quality

Write code as if another researcher will maintain it in five years.

Prefer:

* explicit variable names;
* small functions;
* modular classes;
* clear interfaces;
* extensive mathematical comments where appropriate.

Avoid unnecessary abstraction.

Avoid clever but opaque implementations.

---

# Testing

Whenever practical:

* validate new functionality;
* preserve backward compatibility unless intentionally refactoring;
* verify generated reports;
* verify saved artifacts;
* verify reproducibility.

Do not remove existing functionality without replacing it.

---

# Final Principle

This repository is intended to become a reusable scientific modelling framework.

Every implementation decision should improve at least one of the following:

* scientific correctness;
* reproducibility;
* modularity;
* maintainability;
* computational efficiency;
* clarity.

If a proposed modification improves one objective while significantly harming another, explain the trade-off before implementing it.

When in doubt, favour the solution that produces the most maintainable long-term scientific software.
