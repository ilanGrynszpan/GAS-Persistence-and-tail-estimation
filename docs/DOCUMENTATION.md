# DOCUMENTATION.md

# Documentation Standards

This document defines the documentation philosophy for the repository.

Documentation is considered part of the software.

Every module should be understandable without reverse engineering the implementation.

The objective is to make the framework maintainable, reusable, and scientifically transparent.

---

# 1. General Philosophy

Documentation is a first-class component of the project.

Code that cannot be understood should be considered incomplete.

Documentation should explain

- why something exists;
- what mathematical model is implemented;
- what assumptions are made;
- how modules interact.

Do not document only Python syntax.

Document scientific intent.

---

# 2. Documentation Hierarchy

Documentation exists at multiple levels.

Repository

↓

Module

↓

Class

↓

Function

↓

Important code blocks

Each level should explain different information.

Avoid repeating identical text.

---

# 3. Repository Documentation

The repository should contain

README.md

↓

CLAUDE.md

↓

ROADMAP.md

↓

docs/

The README explains

what the project is.

The documentation inside docs explains

how the project works.

---

# 4. Module Documentation

Every major module should begin with a module-level docstring.

The module docstring should explain

- purpose;
- implemented mathematical models;
- important assumptions;
- references when appropriate;
- interaction with other modules.

Example:

"""
Implements score-driven recursions.

Supports

- standard GAS
- Harvey long-short

This module updates latent states only.

Distribution-specific quantities such as scores and Fisher
information are provided by the distribution classes.
"""

---

# 5. Mathematical Modules

Modules implementing mathematical models require additional documentation.

Examples include

- distributions;
- score calculations;
- Fisher information;
- dynamic recursions;
- optimization.

These modules should explain

- notation;
- equations;
- parameterization;
- transformations;
- references;
- numerical considerations.

Readers should understand the implemented mathematics before reading the code.

---

# 6. Class Documentation

Every public class should explain

- purpose;
- inputs;
- outputs;
- assumptions;
- configurable options.

Avoid describing private implementation details.

---

# 7. Function Documentation

Every public function should document

Parameters

Returns

Raises

Side effects

Notes

Example usage where appropriate

Use a consistent docstring style throughout the repository.

---

# 8. Mathematical Comments

Important mathematical operations should always be commented.

Especially comment

- parameter transformations;
- score calculations;
- Fisher calculations;
- likelihood contributions;
- filtering recursions;
- state updates;
- optimization penalties or numerical safeguards.

Comments should explain

why

rather than

what.

Bad example

```python
x += 1
```

Good example

```python
# Shift recursion to obtain one-step-ahead prediction.
```

In case any calculation to compute analytical things such as score of FI matrix elements were performed for code building, create a md document inside the docs/ folder called "math_proofs.tex", which should contain detailed explanation of the process, so I can review it for corectness, put the maths in latex notation, so that I can generate a PDF from it and read, generate the PDF in the same folder and give me instructions on how to regenerate it from CMD prompt, no AI, if necessary.

---

# 9. Avoid Obvious Comments

Do not write comments that merely restate Python syntax.

Avoid

```python
# Increment i
i += 1
```

Prefer comments that explain scientific reasoning.

---

# 10. README Files

Every major package should contain a small README.

Examples

src/distributions/

src/models/

src/evaluation/

src/data/

Each README should explain

- purpose;
- important classes;
- mathematical references;
- extension points.

General project README should refer to these internal ones and contain major project explanation, incuding main maths

---

# 11. Examples

Every major component should include at least one minimal example.

Examples should demonstrate

- construction;
- fitting;
- prediction;
- evaluation.

Examples should be short and executable.

---

# 12. Configuration Documentation

Configuration options should be documented.

Users should understand

- default values;
- valid options;
- expected behaviour.

Avoid undocumented configuration flags.

---

# 13. Logging

Logging messages should help users understand execution.

Good logs explain

- what is running;
- current stage;
- warnings;
- failures.

Avoid excessive logging.

---

# 14. Error Messages

Errors should be informative.

Avoid

```
ValueError
```

Prefer

```
ValueError:
Diagonal Fisher scaling requires at least one dynamic parameter.
```

Error messages should help users fix the problem.

---

# 15. Notebook Documentation

Notebooks should be readable as scientific documents.

Every notebook should contain

1. Objective
2. Mathematical background
3. Configuration
4. Data
5. Models
6. Results
7. Discussion
8. Conclusions

Avoid notebooks containing only executable cells.

---

# 16. Generated Documentation

Generated reports should not replace source documentation.

Reports explain experiments.

Documentation explains software.

Keep these responsibilities separate.

---

# 17. API Stability

Public interfaces should change only when necessary.

If interfaces change

- update documentation;
- update examples;
- update README files.

Documentation should never lag behind the code.

---

# 18. Documentation Review

Whenever a significant feature is implemented, verify that the corresponding documentation has also been updated.

Implementation is not complete until:

- code is documented;
- examples work;
- README is updated if necessary;
- module documentation reflects the implementation.

---

# 19. Documentation Style

Documentation should be

- precise;
- concise;
- mathematically correct;
- implementation-aware;
- application-independent where possible.

Avoid marketing language.

Avoid unnecessary jargon.

Prefer complete explanations over short descriptions.

---

# 20. Final Principle

Future researchers should be able to understand, modify, and extend the framework by reading the documentation before reading the implementation.

Good documentation should reduce the need to inspect source code while remaining consistent with it.

# 21. Code Walkthrough Philosophy

Complex scientific software should explain its execution flow before presenting implementation details.

Every major module should begin with a short "Code Walkthrough" section immediately after the module docstring.

The purpose is to allow a researcher to understand the complete execution logic in one or two minutes before reading the implementation.

A walkthrough should describe:

- the sequence of major operations;
- the responsibility of each step;
- interactions with other modules;
- important mathematical stages;
- where outputs are generated.

Avoid implementation details.

The walkthrough should explain the algorithm, not the Python syntax.

---

## Example: GAS Model

```text
Execution Flow

fit()

↓

Initialize model configuration

↓

Initialize static and dynamic parameters

↓

Initialize latent states

↓

For each observation

    ↓

Compute predictive distribution

    ↓

Evaluate log-likelihood

    ↓

Compute analytical score

    ↓

Compute analytical Fisher information

    ↓

Apply selected score scaling

    ↓

Update latent states

↓

Return objective value

↓

Optimizer

↓

Post-fit diagnostics

↓

Save artifacts
```

---

## Example: Harvey Long-Short Model

```text
Execution Flow

fit()

↓

Initialize long component

↓

Initialize short component

↓

For each observation

    ↓

Compute predictive distribution

    ↓

Evaluate likelihood

    ↓

Compute score

    ↓

Update long component

    ↓

Update short component

↓

Combine

f_t = ω + L_t + S_t

↓

Store filtered states

↓

Return objective

↓

Optimizer

↓

Diagnostics

↓

Artifacts
```

---

## Example: Report Generation

```text
Execution Flow

Read experiment configuration

↓

Locate artifacts

↓

Load metrics

↓

Load predictions

↓

Generate tables

↓

Generate figures

↓

Generate LaTeX

↓

Compile PDF

↓

Update README summary
```

---

## Example: Experiment Execution

```text
Execution Flow

Read configuration

↓

Determine execution graph

↓

Check existing artifacts

↓

Skip completed models

↓

Estimate resources

↓

Launch workers

↓

Monitor execution

↓

Checkpoint completed models

↓

Generate report
```

---

## Walkthrough Style

Walkthroughs should be:

- concise;
- high-level;
- implementation-independent;
- mathematically meaningful.

Avoid describing every helper function.

Instead, describe the conceptual stages of the algorithm.

---

## Updating Walkthroughs

Whenever the execution logic of a module changes, its walkthrough should be updated together with the implementation.

A walkthrough that no longer reflects the implemented algorithm should be considered outdated documentation.

Keeping walkthroughs synchronized with the code is part of maintaining the software.