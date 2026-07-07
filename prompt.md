# Experiment: Full Multi-Location Model Evaluation and Tail Diagnostics

Before running anything, read the repository documentation exactly as described in `CLAUDE.md` and the architecture and strategy explicited by the instructions in the docs/ folder (ignore AUDIT, or non markdown files there, I am talking about the data and other capital letter markdown files here).

Follow the existing repository architecture and reuse existing modules whenever possible. Do not duplicate functionality.

More than anything, dont change current modelling unless something is actually wrong or requested in this document.

---

# Objective

Extend the current experimental pipeline so that it evaluates every available location, performs comprehensive probabilistic and tail-focused evaluation, and generates publication-quality diagnostics demonstrating the behaviour of the dynamic predictive distribution.

The goal is not only to determine which model achieves the lowest CRPS, but also to demonstrate whether dynamic scale, dynamic tail shape, and the Harvey long-short decomposition produce meaningful improvements in the modelling of precipitation extremes.

The implementation should remain fully modular and report generation should continue to be artifact-driven.

---

# Model estimation

Run the complete three-stage pipeline for **every location** available in the precipitation dataset. YOU SHOULD NOT HAVE TO RUN FOR **BELO HORIZONTE** AGAIN, YOU ALREADY DID, I TESTED AND APPROVED. YOU SHOULD ONLY RUN IT AGAIN IF IT IS NOT POSSIBLE TO OBTAIN THE METRICS AND DIAGNOSTICS I AM ASKING WITHOUT RUNNING AGAIN, BUT THIS SHOULD NOT BE NECESSARY IF YOU DID WHAT I TOLD YOU TO DO AND SAVED THE OPTIMIZED PARAMETERS.

RESULTS,CHARTS, DIAGNOSTICS, ... SHOULD SEPARATE BETWEEN LOCATIONS, NOT BE GROUPED.

The files for these locations should be found in the locations described inside the data.md folder. You will see that inside those folders there are numerous location files, and inside the precipitation folder, they are also divided in train/test sets. You should follow this division for all files and modelling.

For every location:

1. Estimate every Stage 1 baseline ZA-GAS specification.

2. Select the Stage 1 winner using **out-of-sample CRPS**.

3. Use that winning specification as the base model for Stage 2.

4. Estimate every Stage 2 covariate specification.

5. Accept Stage 2 only if OOS CRPS improves by at least **2%** relative to the selected Stage 1 model. Otherwise continue with the Stage 1 winner.

6. Use the selected Stage 1/Stage 2 model as the base for Stage 3.

7. Estimate every Harvey long-short specification.

8. Accept Stage 3 only if OOS CRPS improves by at least **2%** relative to the currently selected model.

Produce both per-location summaries and an overall summary comparing all locations.

---

# NOTE

The requests below DO NOT replace the reporting style and information you are already producing, and that resulted in file reports/report.*. Nor do they replace REPORTING.md, I like that report style, the explanations, everything. Everything described below is addition to what has already been done. That is why I am saying, do not delete or change unnecessary working things.

Pay attention to everything, including how you are generating the PIT, it has to be IS and as you are already doing. The ACF has to be with 400 lags. Every metrics you are using, report in the PDF and tex in prior sections, including all the mathematics as in your code, report the mathematics of the models as in the code, you can use the current report as basis for styling, indexing, formatting, ... but the content HAS TO BE faithful to what you did in the code, I want to know what was done.

---

# Out-of-sample evaluation

Compute every evaluation metric using the **time-varying predictive distribution** produced by the model.

Do not evaluate static fitted distributions.

Retain existing metrics.

Additionally compute:

- CRPS
- Threshold-weighted CRPS (twCRPS)
- Quantile Score (pinball loss)
- Log Score
- RMSE
- MAD
- Brier Score
- Kupiec unconditional coverage
- Christoffersen conditional coverage / independence
- PIT diagnostics
- Quantile residual diagnostics

For Quantile Score, Kupiec and Christoffersen evaluate the following quantiles:

- 0.50
- 0.75
- 0.90
- 0.95
- 0.975
- 0.99
- 0.995
- 0.999
- 0.9995
- 0.9999

For twCRPS compute versions emphasising at least:

- upper 90%
- upper 95%
- upper 99%

using appropriate threshold weighting functions.

---

# Dynamic quantile diagnostics

The defining characteristic of these models is that the predictive distribution changes through time.

Therefore diagnostics should evaluate the evolution of conditional quantiles rather than treating the model as a single fitted distribution.

For every selected model generate dynamic conditional quantiles for:

- 0.50
- 0.75
- 0.90
- 0.95
- 0.975
- 0.99
- 0.995
- 0.999
- 0.9995
- 0.9999

Generate figures showing:

- observed rainfall
- dynamic conditional quantiles

through time.

These figures should clearly illustrate how the predictive distribution adapts to changing meteorological and climatic conditions.

---

# Return-level diagnostics

Compute two complementary families of return levels.

## Daily rarity / return-level diagnostics

For a return period of \(T\) days, the corresponding conditional return level is the rainfall amount expected to be exceeded with probability \(1/T\) on a given day.

For each day \(t\), the model produces a conditional predictive CDF:

\[
F_t(y) = P(Y_t \le y \mid \mathcal{F}_{t-1})
\]

The \(T\)-day conditional return level is therefore:

\[
RL_{T,t} = F_t^{-1}\left(1-\frac{1}{T}\right)
\]

Compute this for:

- \(T=30\)
- \(T=60\)
- \(T=90\)
- \(T=100\)
- \(T=500\)
- \(T=1000\)

So, for example:

\[
RL_{100,t} = F_t^{-1}(0.99)
\]

is the model-implied rainfall amount on day \(t\) that should be exceeded with probability 1% under the predictive distribution for that day.

These are not stationary climatological return levels. They are dynamic conditional return levels, changing over time because \(F_t\) changes over time.

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

# Tail calibration

For every quantile level compute empirical exceedance frequencies.

Evaluate exceedance calibration:

- overall;
- by month;
- by ENSO regime;
- by wet season;
- by dry season.

Compare observed exceedance frequencies against theoretical exceedance probabilities.

---

# Seasonal decomposition

For every location compute diagnostics grouped by calendar month.

Generate:

- monthly boxplots of dynamic quantiles;
- monthly boxplots of dynamic return levels;
- monthly exceedance frequencies;
- monthly Quantile Scores;
- monthly twCRPS.

These should demonstrate whether the model correctly adapts to seasonal rainfall behaviour.

---

# ENSO decomposition

Evaluate the behaviour of the predictive distribution as a function of ENSO using two complementary approaches.

## 1. Standard ENSO regimes

Classify each day using the Niño 3.4 index according to the standard NOAA thresholds:

- El Niño: Niño 3.4 ≥ +0.5°C
- Neutral: -0.5°C < Niño 3.4 < +0.5°C
- La Niña: Niño 3.4 ≤ -0.5°C

For each regime compute:

- dynamic conditional quantiles;
- dynamic return levels;
- Quantile Scores;
- twCRPS;
- empirical exceedance frequencies.

## 2. ENSO intensity

Rather than only using discrete categories, also analyse ENSO continuously.

Divide the observed Niño 3.4 values into approximately equal-frequency bins (e.g. deciles or quintiles).

For each bin compute:

- mean and distribution of the estimated shape parameter (xi or its equivalent eexponentiated version in the distribution);
- mean dynamic conditional quantiles;
- mean dynamic return levels;
- Quantile Scores;
- twCRPS;
- empirical exceedance frequencies.

This analysis should determine whether increasing ENSO intensity is associated with systematically increasing extreme rainfall risk, providing direct evidence for the effectiveness of the Harvey long-short decomposition.

---

# Wet and dry season decomposition

Compute average rainfall for every calendar month.

Months with average rainfall above the overall monthly mean should be classified as wet months.

The remaining months should be classified as dry months.

Repeat all tail diagnostics separately for wet and dry months.

For each location, you should list which months were categorized as wet or dry. There could be statistical anomalies like a dry month in the middle of wet ones for example. If this is the case categorize this month as wet. Vice-versa too. 

If its not possible to clearly distinguish wet and dry seasons (for instance, if all months hover within an interval of +-2% of the average), say that this analysis is not even possible for the location.

---

# Signature diagnostic figures

Generate publication-quality figures showing:

1. observed rainfall;
2. dynamic 95% quantile;
3. dynamic 99% quantile;
4. dynamic 99.9% quantile;

through time.

Shade El Niño and La Niña periods in the background with different colors: a light red representing El Nino, and light blue La Nina.

On a second aligned panel, plot the estimated dynamic tail parameter (and long/short components where applicable).

The objective is to visually demonstrate the chain:

climate conditions → latent dynamics → tail parameter → predictive distribution → observed extremes.

These figures should become the primary visual evidence supporting the contribution of the thesis.

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

# Regime models

Implement if not implemented yet, and run the regime sensitive models specified in MODELS.md sections 22 and 23. Use as covariates the best version between stages 1 to 3, you can even implement it as a long short decomposition if stage 3 was the best call. Put this as a last chapter in the end of the report, separated from everything elese, and run all diagnostics and metrics you ran for all else for these models.

---

# Inspection

Make necessary code alterations if necessary, but ask me before you run. Dont make me run things, you will run, and dont make me prompt again to run, just ask me within this same prompt something like "I finished, here is all I did, answer yes if I should run". After you generate the report, tell me what is the tex file generating it and how to regenerate pdf from it without asking you. Tell me too what notebook or python file you are running to run evereything in the background, in case later I have to run it with changes without wasting prompts.