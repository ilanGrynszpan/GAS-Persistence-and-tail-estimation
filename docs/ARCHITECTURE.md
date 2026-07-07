# DESIGN_PRINCIPLES.md

# Design Principles

This repository develops a reusable statistical modelling framework based on score-driven (GAS/SD) models.

The objective is to build a scientific software library whose components are mathematically correct, modular, reproducible, and reusable across different application domains.

This document defines the long-term design principles of the repository. These principles should remain stable even if specific models, distributions, applications, or experiments change over time.

---

# 1. Scientific Correctness Comes First

The primary objective of the project is mathematical and statistical correctness.

Computational convenience should never justify mathematically incorrect implementations.

Whenever possible,

* derive analytical results;
* document assumptions;
* validate implementations against the literature.

If a mathematical shortcut changes the model being implemented, it should not be adopted.

---

# 2. Analytical Methods Are Preferred

Whenever analytical expressions are available, they should be preferred over numerical approximations.

Examples include

* score vectors;
* Fisher information matrices;
* parameter transformations;
* gradients;
* moments.

Numerical differentiation should be avoided unless analytical expressions are unavailable or clearly impractical.

Analytical implementations are generally preferred because they improve

* computational efficiency;
* numerical stability;
* reproducibility;
* interpretability.

---

# 3. Separate Mathematics From Applications

Core mathematical models should remain independent from any specific application.

The mathematical framework should not contain assumptions about

* datasets;
* stations;
* assets;
* variables;
* domains.

Application-specific logic belongs in the data-loading and experiment layers.

A mathematical model should be reusable without modification across multiple domains.

---

# 4. Separate Models From Data

Models should operate on standardized model-ready data.

Reading raw files, cleaning data, aligning timestamps, creating lagged variables, and application-specific preprocessing belong outside the model implementation.

The model should only receive the information required for estimation and prediction.

---

# 5. Configuration Over Duplication

Different model specifications should be controlled through configuration rather than separate implementations.

Model behaviour should be determined by configurable options whenever possible.

Avoid creating multiple files that differ only by

* dynamic parameters;
* score scaling;
* optimization settings;
* lag structures;
* distributions.

The same implementation should support multiple configurations.

---

# 6. Modularity

Every component should have a single, well-defined responsibility.

Typical responsibilities include

* distributions;
* dynamic recursions;
* optimization;
* data loading;
* evaluation;
* reporting.

Avoid unnecessary coupling between modules.

Modules should communicate through clear interfaces rather than hidden assumptions.

---

# 7. Separation of Concerns

The repository should clearly separate

1. mathematical models;
2. data preparation;
3. experiment execution;
4. evaluation;
5. reporting.

These layers should evolve independently whenever possible.

Changes in one layer should require minimal changes elsewhere.

---

# 8. Reproducibility

Scientific results must be reproducible.

Every model should save sufficient information to reproduce

* parameter estimates;
* forecasts;
* diagnostics;
* evaluation metrics;
* figures;
* reports

without repeating computationally expensive estimation whenever possible.

Reports should be generated from saved artifacts rather than directly from optimization routines.

---

# 9. Progressive Development

The framework should evolve through careful refactoring rather than continual duplication.

Existing implementations should be understood before modification.

Working code should not be rewritten without clear justification.

Incorrect implementations should be corrected rather than silently preserved.

When major redesigns occur, obsolete implementations should be explicitly deprecated or archived.

---

# 10. Scalability

The architecture should support increasing model complexity without requiring major redesign.

Adding

* new distributions;
* additional dynamic parameters;
* new score scaling methods;
* alternative recursions;
* new applications

should primarily involve extending existing interfaces rather than rewriting the framework.

The architecture should avoid combinatorial growth in the number of implementations.

---

# 11. Computational Efficiency

Efficiency should be considered during design.

Avoid unnecessary numerical computation.

Prefer analytical expressions whenever available.

Reuse previously computed quantities whenever possible.

Store computationally expensive results for future reuse.

Parallel computation should accelerate independent work while preserving identical numerical results.

Performance improvements must never change the underlying statistical model.

---

# 12. Transparency

Scientific software should be understandable.

Every important mathematical module should explain

* the implemented model;
* notation;
* assumptions;
* references;
* numerical considerations.

Code should favour clarity over cleverness.

Important mathematical operations should be documented.

Future researchers should be able to understand the implementation without reverse engineering the code.

---

# 13. Extensibility

The framework should be designed to accommodate future research.

Adding new functionality should require extending existing interfaces rather than modifying unrelated components.

Factory patterns, configuration objects, and modular interfaces should be preferred over hard-coded assumptions.

---

# 14. Robustness

Optimization routines should be monitored for numerical stability.

Model validity should not depend solely on optimizer termination messages.

Model quality should be assessed using both numerical diagnostics and statistical diagnostics.

Failures should be isolated whenever possible so that unsuccessful models do not interrupt larger experimental pipelines.

---

# 15. Documentation

Documentation is considered part of the software.

Major modules should include

* mathematical description;
* implementation notes;
* expected inputs and outputs;
* important assumptions.

README files should describe the purpose of each major module.

Docstrings and comments should explain mathematical reasoning rather than obvious programming syntax.

---

# 16. Long-Term Vision

The objective is to build a reusable scientific framework for score-driven statistical modelling.

Applications, distributions, and experiments will evolve over time.

The design principles described in this document should remain valid regardless of those future developments.

When implementation decisions are uncertain, prefer the solution that better preserves

* correctness;
* modularity;
* reproducibility;
* scalability;
* clarity;
* reusability.
