# EXECUTION.md

# Experiment Execution Framework

This document defines how experiments should be executed within the modelling framework.

It should be read together with

* ARCHITECTURE.md
* MODELS.md
* OPTIMIZATION.md
* REPORTING.md

The purpose of this document is to ensure that experiments are reproducible, computationally efficient, fault tolerant, and scientifically consistent.

---

# 1. General Philosophy

Experiments should be viewed as scientific workflows rather than isolated model fits.

Each experiment answers one scientific question.

The execution framework should:

* execute only the required models;
* reuse previous results whenever possible;
* avoid duplicate computation;
* remain reproducible;
* isolate failures;
* produce reusable artifacts.

The objective is not to execute as many models as possible.

The objective is to answer scientific questions efficiently.

---

# 2. Separation Between Implementation and Execution

The framework should distinguish between

**implemented models**

and

**executed models**.

All implemented models should exist within the library.

However, only models explicitly requested by the notebook, configuration file, or execution prompt should be executed.

Implementing a model does **not** imply running it.

This allows future models to exist in the codebase without increasing execution time.

Also, after implementing everything, you should pause and ask me for permission to run it, while giving me instrucitons for code review. Only run things after I authorize, which I will after code review has been approved.

---

# 3. Configuration Driven Execution

Experiments should be controlled by configuration.

Typical configuration options include

* distribution;
* dynamic parameters;
* score scaling;
* lag structure;
* optimizer;
* covariates;
* long-short dynamics;
* regime models;
* train/test split;
* evaluation metrics.

The execution pipeline should avoid hard-coded model specifications.

---

# 4. Sequential Scientific Pipeline

The thesis experiments are sequential.

Each stage answers one scientific question before moving to the next.

The current pipeline is

Stage 1

Baseline GAS

↓

Stage 2

Weather covariates

↓

Stage 3

Harvey Long-Short


Stage 4 is a separate thing:

Regime models (future work, but you should implement now)

Each stage depends on the conclusions of the previous stage.

Do not execute later stages before selecting the best model from earlier stages.

---

# 5. Stage 1 — Baseline GAS

Scientific question:

Can score-driven dynamics model the predictive distribution?

Current comparisons:

* phi only; (1)
* phi + xi; (2)
* short lags; (3)
* seasonal lags; (4)
* score scaling alternatives. (5)

No external covariates.

The objective is to identify the best baseline specification.

Objectives (1,2), (3,4), (5) are separate sets, (3,4) should both be tried bot hboth 1 and 2. all alternatives in 5 should be tried for both 1 and 2 time varying parameters, you can choose between scling alternatives from the model with no covariates, and follow to covarites with the best approach by likelihood.

---

# 6. Stage 2 — Weather Covariates

Scientific question:

Do local atmospheric variables improve predictive performance?

The execution order is fixed.

1.

Best baseline GAS

↓

2.

Dew point short lags

↓

3.

Dew point seasonal lags

↓

4.

Select best dew point

↓

5.

Temperature short lags

↓

6.

Temperature seasonal lags (only if justified)

↓

7.

Select best weather specification

Do not reopen previous model searches.

Do not search all weather combinations simultaneously. Do not build interactions, like D_t * T_t, its beyond the scope and sill make execution endless.

---

# 7. Stage 3 — Harvey Long-Short

Scientific question:

Does persistent climate information improve the selected weather model?

The short component should inherit the selected weather specification.

The long component should compare

* 90-day ENSO;
* 90-day + 30-day ENSO;
* 90-day + 30-day + lagged daily ENSO.

The weather search is **not** repeated.

Only the long-term climate information is evaluated.

---

# 8. Stage 4 — Regime Models

Future work.

Not part of the default execution pipeline.

Only execute when explicitly requested.

Nevertheless you should build the full code for it, I will review while the requested run codes are running.

---

# 9. Dependency Graph

Execution should follow a dependency graph rather than a flat model list.

Example

Baseline GAS

↓

winner

↓

Weather

↓

winner

↓

Harvey

↓

winner

↓

Report

Each stage depends on the previous stage.

The execution engine should understand these dependencies.

---

# 10. Fork-Join Parallelism

Parallel execution should occur only inside independent branches.

Example

phi

phi + xi

↓

unit

FI

diag FI

↓

run independently

↓

join

↓

compare

↓

continue

Do not parallelize dependent stages.

The execution graph should remain deterministic.

---

# 11. Multiprocessing

Multiprocessing should be optional.

Configuration should include

```python
parallel = True
workers = 4
```

Worker count should remain configurable.

Default worker count should be conservative.

Leave sufficient CPU and memory available for the operating system.

Parallel execution must not change numerical results.

Only execution speed should change.

Most impotantly, you can only paralelize things that are in the same stage, because best results from ine stage are necessary for the execution of the other ones.

Needless to say but parallelism runs different models on different cores, it is stupid and you should not attempt to parallelize the same model, these are time series models, therefore sequential.

---

# 12. Runtime Monitoring

Every running model should be monitored.

Record

* elapsed time;
* iterations;
* objective value;
* gradient norm;
* memory usage when available.

Suggested defaults

Soft warning

45 minutes

Hard timeout

90 minutes

These limits should remain configurable.

---

# 13. Memory Monitoring

Monitor memory whenever practical.

If memory consumption becomes excessive

* log a warning;
* stop the model cleanly if necessary;
* save partial outputs;
* continue with remaining independent jobs.

The framework should avoid exhausting system memory.

---

# 14. Execution Safety

The framework should actively monitor

* optimization progress;
* runtime;
* memory;
* objective stability;
* filtered states;
* numerical failures.

If problems occur

attempt

* restart;
* stronger numerical safeguards;
* reuse previous estimates;

before declaring failure.

One failed model must never terminate the entire experiment.

---

# 15. Checkpoint Philosophy

Checkpoint after every completed model.

Never wait until the end of a stage.

Each checkpoint should contain

* parameters;
* optimizer output;
* diagnostics;
* forecasts;
* metrics;
* configuration.

If execution stops unexpectedly

completed models should never need to be re-estimated.

---

# 16. Artifact Philosophy

Artifacts are part of the scientific workflow.

Every model should produce its own artifact directory.

Example

artifacts/

```
model_0001/

model_0002/

model_0003/
```

This is just the structure, the names of the models, including their folders whould be relevant to what the model represents, example "GASpq_phi_xi_tv/"

Each artifact directory should contain everything necessary to reproduce diagnostics and reports without repeating optimization.

---

# 17. Reuse Philosophy

Before fitting a model

check whether

an identical configuration

already exists.

If the model already exists

reuse

its artifacts

instead of repeating optimization.

Allow users to force re-estimation if explicitly requested.

---

# 18. Execution Logging

Maintain detailed execution logs.

For every model record

* configuration;
* execution time;
* optimizer;
* convergence class;
* warnings;
* retries;
* runtime;
* memory;
* artifact location.

Execution logs should allow interrupted pipelines to resume safely.

---

# 19. Experiment Metadata

Every experiment should receive a unique identifier.

Metadata should include

* timestamp;
* git commit if available;
* configuration hash;
* dataset version;
* notebook;
* software version.

Experiments should remain reproducible months or years later.

---

# 20. Failure Isolation

Failures should be isolated.

A failed model should

* save failure metadata;
* save partial diagnostics when possible;
* release resources;
* continue the remaining pipeline whenever dependencies allow.

Never terminate the entire experiment because of a single failed model.

YOU SHOULD ABORT EVEYTHING AND LET ME KNOW THROUGH LOGS WHAT IS HAPPENING IF ALL MODELS, OR MOST OF THEM ARE FAILING.

AS STATED IN THE OPTIMIZATION.md FILE, FAILURE DOES NOT MEAN THE SCIPY ANSWER OR SOMETHING, SOMETHING FAILS IF TIMEOUT IS REACHED OR ACCORDING TO SPECIFICATIONS FROM OPTIMIZATION.md, ITEM 11. 

---

# 21. Resume Capability

The framework should support resuming interrupted experiments.

When restarting

* detect completed models;
* skip completed work;
* continue from the first unfinished dependency.

Never recompute successful models unless explicitly requested.

---

# 22. Report Execution

Reports should never call optimization.

Report generation should read only

artifacts

↓

tables

↓

figures

↓

latex

↓

pdf.

Users should be able to modify the LaTeX report manually and regenerate the PDF without rerunning experiments.

---

# 23. Notebook Philosophy

Notebooks should orchestrate experiments.

They should

* configure;
* execute;
* inspect;
* discuss.

They should not contain model implementations.

Business logic belongs in reusable modules.

---

# 24. Execution Modes

The framework should support multiple execution modes.

Examples

Development

Small datasets

Fast diagnostics

↓

Research

Selected experiments

↓

Production

Complete thesis pipeline

↓

Report only

No optimization

Only artifact reuse

Execution mode should be configurable.

---

# 25. Current Default Pipeline

The current thesis execution order is

1. Standard GAS
2. Select best baseline
3. Dew point
4. Select best dew point
5. Temperature
6. Select best weather specification
7. Harvey Long-Short
8. Compare ENSO specifications
9. Select final model
10. Generate diagnostics
11. Generate report

This ordering should remain unchanged unless explicitly modified.

---

# 26. Things To Avoid

Avoid

* rerunning existing models;
* executing dependent stages in parallel;
* overwriting artifacts;
* generating reports directly from optimization;
* hard-coded execution paths;
* notebook-specific logic inside the library;
* storing execution state only in notebook memory;
* recomputing expensive diagnostics unnecessarily.

---

# 27. Final Principle

Execution should be viewed as a reproducible scientific workflow.

The execution engine should answer scientific questions while minimizing unnecessary computation.

The framework should favor

* reproducibility;
* checkpointing;
* modularity;
* deterministic execution;
* safe parallelism;
* complete artifact reuse.

---

# 24.5 Resource Estimation

Before launching any execution stage, the framework should estimate the computational resources required.

The objective is to help the researcher understand the expected computational cost before committing to a potentially long experiment.

Whenever possible, estimate:

* number of models to be executed;
* dependency graph for the current stage;
* expected runtime;
* expected memory usage;
* expected disk usage for artifacts;
* expected number of CPU workers;
* whether completed artifacts can be reused.

The estimation should use previous execution logs whenever available.

For example, if similar models have already been fitted, estimate runtime using the average runtime of comparable models.

If no previous information exists, estimate using pilot runs or conservative defaults.

Example execution summary:

```text
====================================================

Stage:
Weather Covariates

Models to estimate:
12

Previously completed:
4

Models remaining:
8

Estimated runtime:
2 h 35 min

Estimated peak RAM:
11.8 GB

Estimated disk usage:
4.2 GB

Parallel workers:
4

Estimated reusable artifacts:
67 %

====================================================
```

This information should be displayed before execution begins and recorded in the execution log.

---

# 24.6 Adaptive Resource Management

The execution engine should adapt resource usage when necessary.

Examples include:

* reducing the number of worker processes if estimated memory exceeds available system memory;
* executing large models sequentially while allowing smaller models to run in parallel;
* postponing expensive diagnostic generation until after model estimation;
* temporarily disabling non-essential outputs when storage becomes limited.

Adaptive resource management should **never** change the statistical model or optimization settings.

It may only change execution scheduling.

---

# 24.7 Execution Health Monitoring

The execution engine should continuously monitor the health of every running model.

Monitor whenever practical:

* elapsed runtime;
* memory consumption;
* CPU utilization;
* optimizer iterations;
* objective value progression;
* gradient norm;
* optimizer warnings;
* state stability;
* disk usage.

When abnormal behaviour is detected, the framework should respond progressively.

Examples include:

1. Log a warning.
2. Increase monitoring frequency.
3. Save an intermediate checkpoint.
4. Attempt a safe restart if appropriate.
5. Stop only the problematic model if necessary.

One problematic model should never terminate the entire experimental pipeline.

---

# 24.8 Resource-Based Warnings

The execution engine should proactively warn the user when resource usage becomes unusual.

Examples include:

* one model is taking substantially longer than comparable models;
* memory usage is increasing continuously;
* repeated numerical failures occur;
* disk usage is approaching configured limits;
* checkpoint generation is unusually slow.

Warnings should be informative rather than interrupting execution.

For example:

```text
WARNING

Model:
harvey_phi_xi_diagonalFI

Elapsed time:
71 minutes

Expected runtime:
24 ± 8 minutes

Current gradient norm:
1.8e-2

Memory usage:
14.7 GB

Recommendation:
Continue monitoring. Model is progressing, but runtime is substantially above the expected range.
```

Warnings should be saved together with the execution metadata.

---

# 24.9 Automatic Resource Recovery

When possible, the framework should recover automatically from resource-related problems.

Possible recovery actions include:

* reducing parallel workers for remaining jobs;
* releasing cached intermediate objects that are no longer required;
* checkpointing completed work immediately;
* postponing expensive diagnostics;
* skipping already completed models during restart.

Recovery mechanisms should preserve scientific reproducibility.

They should never modify:

* the statistical model;
* the optimization objective;
* parameter estimates;
* evaluation methodology.

Only execution scheduling and resource allocation may be adapted.

---

# IMPORTANT

- EXECUTION MUST BE LOGGED, LOGS SHOULD BE UNDERSTANDABLE AND DETAILED.
- DO NOT RUN ANY CODE YOU BUILT BEFORE i AUTHORIZE, AFTER YOU FINISHED BUILDING CODE, ASK ME IF IT IS OK TO RUN WHILE GIVING ME CODE REVIEW INSTRUCTIONS, i HAVE TO REVIEW THE CODE AND CHANGE THINGS IF NECESSARY BEFORE YOU RUN ANYTHING. 
- IT I AUTHORIZE YOU SHOULD RUN THE CODE ON BACKGROUND, YOU SHOULD ALSO TELL ME, IN CASE I FIND SOMETHING OUT LATER, HOW I CAN RUN EVERYTHING ON THE BACKGROUND WITHOUT ASKING YOU AND WASTING PROMPTS WITH SILLY THINGS.
