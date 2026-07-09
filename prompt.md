# Experiment: Full Multi-Location Model Evaluation and Tail Diagnostics

Before running anything, read the repository documentation exactly as described in `CLAUDE.md` and the architecture and strategy explicited by the instructions in the docs/ folder (ignore AUDIT, or non markdown files there, I am talking about the data and other capital letter markdown files here).

Follow the existing repository architecture and reuse existing modules whenever possible. Do not duplicate functionality.

More than anything, dont change current modelling unless something is actually wrong or requested in this document.

Input data file paths should be in docs/data.md

---

# Objective

We have to implement some fixes to the existing pipeline. The last execution was from run_all_location.iynb and generated the report in reports/multi_location. We need the same locations that were used for that report but with the following changes:

1. You do not have to reestimate these models but the evaluation criteria is different: 

1a. phi-only tv models should be evaluated against thhemselves according to OOS RMSE criterion. This means, stages 1, 2 and 3 comparing only phi with only phi.

1b. 2 tvp (phi and xi) models should be compared within and between stages against themselves too, using CRPS and the main criterion, and also twCRPS at 95% and 98%.

2. Stage 3 models used the wrong data, somehow you were not able to catch that the input CSV had Nino SST, not anomalies, and the range did not even include key values like 0.5. This caused an evaluation as if the whole time series was under El Nino, which is obviously not correct, and should have been questioned before optimization even started. Therefore, all stage 3 models need to be reestimated. But, I will introduce some changes:

2a. The important for the slow moving part (Long) is not the ENSO anomalies themselves, but if we are in El Nino, La Nina, or neutral times. Therefore, we will rather work with dummies to capture such behavior, as in:

L^θ_t    =  α^θ_L · L^θ_{t-1}  +  β^θ_L' · X^long_t

where vector $\beta$ contains multipliers to orthogonal dummies that represent each of these states, and possibly their lags.

2b. The input tables should be evaluated if the refering data corresponds (makes sense) to the observed variable, do not use data that does not make sense, activelly question it.

3. Stage 4 (regime sensitive GAS) only makes sense as tail sensitive dynamics. Therefore, it will only be used with xi, not with the scale parameter phi. The dynamics of phi within these models will follow the best calibration for phi and xi tvps from earlier stages, and therefore, the same scalling should be used. Phi dynamics here should not be reestimated, as that model already works. I know that xi dynamics will impact phi more or less depending on the scalling, but computational cost and margin for errors increases with phi reestimation unnecessarily.

3a. Stage 4 models will be evaluated on twCRPS agains their best of preivous stages counterparts.

3b. Stage 4 models should only be sensitive to different dynamics for q95 and q98, therefore, models with q90 or other quatile differentiation lss than q95 should be terminated.

4. Current reporting as of reports/multi_location is a bit confusing and hard to read. I need the following structure:

4a. For all models in all stages the following metrics: log-likelihood, AIC, BIC.

4b. For only-phi tvp models, all stages and all models, OOS RMSE and MAD.

4c. For both phi and xi tvp models, all stages and all models, OOS CRPS and twCRPS for q95 and q98.

4d. For all stages, the in-stage winner according to these metrics for the corresponding set of tvp, with the winner being the best key metric or metrics for that set of tvp, or, if improvement is less than 2% in key metric(s), the simplest model. This smae logic should be used for inter-stage decision.

4e. For winner model in each stage: IS PIT histogram, IS ACF wih 400 lags.

4f. For winner model of all stages for both sets of tvp: table with OOS RMSE, MAD, CRPS, AIC, BIC. Another table with twCRPS for q95 and q98, christopherssen and kupiec at q95.

4g. Climatological return levels for 

- 2-year
- 5-year
- 10-year
- 25-year
- 50-year
- 75-year
- 100-year
- 500-year
- 1000-year

in separate charts of OOS time series, against time series. Below the chart it should show the rate of theoretical and empirical violations.

4f.

For each quantile level (q50, q75, q90, q95, q98, q99) compute empirical exceedance frequencies.

Evaluate exceedance calibration:

- overall;
- by month;
- by ENSO regime;
- by wet season;
- by dry season.

Compare observed exceedance frequencies against theoretical exceedance probabilities for each model, comparing their only-phi, and phi and xi counterpart, and the stage 4 vs the best in preivous stages.

4g. For time series OOS, shade in light red the El Nino moments, and in light blue the La Nina ones.

---

## Climatological return levels

Compute conditional return levels for:

- 2-year
- 5-year
- 10-year
- 25-year
- 50-year
- 75-year
- 100-year

using

p(T) = (1 - 1/T)^(1/365)

where T is measured in years.

These should be interpreted as dynamic conditional return levels rather than assuming a stationary rainfall distribution.

---

# Wet and dry season decomposition

Compute average rainfall for every calendar month.

Months with average rainfall above the overall monthly mean should be classified as wet months.

The remaining months should be classified as dry months.

Repeat all tail diagnostics separately for wet and dry months.

For each location, you should list which months were categorized as wet or dry. There could be statistical anomalies like a dry month in the middle of wet ones for example. If this is the case categorize this month as wet. Vice-versa too. 

If its not possible to clearly distinguish wet and dry seasons (for instance, if all months hover within an interval of +-2% of the average), say that this analysis is not even possible for the location.

This should be ehxibited by location, and used to interpret results within the objectives section.

---

# Outputs

Produce:

- reusable CSV files;
- reusable figures;
- publication-quality PDF report;
- editable LaTeX report;
- summary tables for every station;
- global comparison tables.

All outputs should be generated from saved artifacts and remain fully reproducible.

As with previous experiments, before running any estimation, create an implementation summary document explaining your interpretation of the requested work, the data used, any assumptions made, and the expected outputs, so that I can audit the implementation before authorising execution.

SAVE ALL TABLES IN CSVS AND LATEX FOR FURTHER USE, AS WELL AS FIGURES PRODUCED. I WILL NEED TO USE THEM LATER IN MY THESIS.

ALSO, INCLUDE IN THE APPENDIX OF THE REPORT ALL TABLES, THIS ONES AND OTHER ONES THAT YOU DID NOT INCLUDE IN THE PREVIOUS ONE, LIKE STD ERRORS OF PARAMETERS. MAKE ALL TABLES, APPENDIX OR NOT CLEARLY READABLE IN THE REPORT, NOT OVERFLOWING THE PDF PAGE. ONE TABLE PER TOPIC (1. OOS CRPS, TWCRPS, QUANTILE SCORE. 2 - IS METRICS. 3 - RMSE. 4 - STD ERRORS... DO NOT MIX THINGS).

---

# IMPORTANT

YOU SHOULD NOT HAVE TO RUN EVERY SINGLE MODEL TO DO WHAT I ASKED YOU HERE. MOST MODELS I PREVIOUSLY ASKED YOU ALREADY RAN AND SHOULD HAVE DOCUMENTED ALL OUTPUTS INCLUDING OPTIMIZED PARAMETERS, SO ONLY RUN THEM AGAIN IF THERE WAS AN ERROR THE FIRST TIME. AS FAR AS I AM CONCERNED YOU WILL ONLY HAVE TO RUN AGAIN STAGES 3 AND 4.

MONITOR IT ALL WHILE RUNNING, KEEP ME INFORMED OF WHAT IS HAPPENING OF RELEVANCE.

YOU DID NOT RUN ALL LOCATIONS IN THE FOLDERS AND FILES FOR THE REPORT YOU ARE BASING YOURSELF ON. FOR THIS RUN, SINCE YOU ARE NOT RE-RUNNING ALL MODELS, USE THE SAME LOCATIONS YOU ALREADY RAN, IGNORE THE ONES YOU DIDNT.