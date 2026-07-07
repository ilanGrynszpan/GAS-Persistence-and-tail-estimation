# MODELS.md

# Model Design and Mathematical Specifications

This document defines the mathematical models implemented in this repository.

It should be read together with:

* `docs/ARCHITECTURE.md`
* `docs/OPTIMIZATION.md`
* `docs/EXECUTION.md`
* `docs/REPORTING.md`

The purpose of this document is to define the modelling framework, the model hierarchy, and the current thesis model pipeline.

This document should not contain application-specific file paths, station names, or report formatting instructions.

---

# 1. Modelling Objective

The repository implements score-driven statistical models for conditional distribution forecasting.

The objective is to model the full predictive distribution:

$$
Y_t \mid \mathcal F_{t-1}.
$$

For precipitation applications, this means modelling:

* rainfall occurrence;
* wet-day intensity;
* distributional calibration;
* upper-tail behaviour.

Point forecasts are evaluated, but the project is primarily distributional.

The same modelling framework should remain reusable with different distributions, covariates, and data loaders.

---

# 2. Model Hierarchy

The framework should be understood as a hierarchy of interchangeable modelling choices.

```text
Observation model
│
├── Occurrence component
│     └── pi_dynamics/  (already implemented; reuse as is)
│
├── Positive-part distribution
│     ├── GB2LogLink  (current implementation)
│     └── future distributions
│
├── Dynamic parameters
│     ├── phi
│     ├── xi
│     └── arbitrary supported subsets
│
├── Score scaling
│     ├── unit
│     ├── inverse_fisher
│     └── diagonal_inverse_fisher
│
├── Dynamic recursion
│     ├── standard_gas
│     ├── harvey_long_short
│     └── regime_score_update
│
└── Covariates
      ├── none
      ├── X_t              standard GAS covariates
      ├── X_short,t        short-component covariates
      └── X_long,t         long-component covariates
```

The framework should support these choices through configuration rather than separate model files.

---

# 3. Observation Model

The current precipitation model is a zero-augmented positive distribution model.

Let

$$
\pi_t = P(Y_t>0\mid\mathcal F_{t-1})
$$

denote the conditional probability of positive rainfall.

The zero-augmented likelihood is

$$
p(y_t\mid\mathcal F_{t-1})
=

(1-\pi_t)\mathbf 1(y_t=0)
+
\pi_t g(y_t\mid\mathcal F_{t-1};\theta_t)\mathbf 1(y_t>0),
$$

where:

* \(\pi_t\) is the occurrence probability;
* \(g(\cdot)\) is the positive-part density;
* \(\theta_t\) contains the positive-part distribution parameters.

The occurrence dynamics already exist in `pi_dynamics/`.

Use the existing `pi_dynamics/` implementation as is unless explicitly instructed otherwise.

Do not redesign or rewrite the occurrence model during framework refactoring.

---

# 4. Positive-Part Distribution

The current positive-part distribution is the GB2 with log-link parameterisation.

The current implementation is `GB2LogLink`.

The distribution module owns all distribution-specific mathematics:

* log-density;
* CDF;
* PPF/quantiles;
* random simulation;
* analytical score;
* analytical Fisher information.

The modelling layer should not duplicate distribution-specific mathematics.

---

# 5. GB2LogLink Parameterisation

Let the PDF of $g_y(y_t \mid \mathcal{F}_{t-1}; \theta)$ have the following parametrisation

$$
g_y(y_t \mid \mathcal{F}_{t-1}; \theta)
=
\frac{
\dfrac{1}{\gamma}
\left(
\frac{y_{t}}{\phi}
\right)^{\frac{\xi}{\gamma}-1}
}{
\phi B(\xi,\zeta)
\left[
1 +
\left(
\frac{y_{t}}{\varphi}
\right)^{\frac{1}{\gamma}}
\right]^{\xi+\zeta}
}
$$

The current dynamic parameters of interest are:

* \(\phi_t\), transformed scale;
* \(\xi_t\), transformed shape/tail-flexibility parameter.

The parameters \(\gamma\) and \(\zeta\) remain static unless explicitly changed.

Important implementation warning:

The current `GB2LogLink` implementation already works in the transformed/log parameterisation. Do **not** apply an additional link function or exponentiation in the GAS recursion.

Future work may separate distributions from link functions, but that is not part of the current implementation.

You might just have to complete fisher information or scores, in which case you could check if these are correct, and if they are use them:

score:

The following variable can be defined for further use:

$$
z_{t|t-1}
=
\left(
\frac{y_{t}}{\varphi}
\right)^{\frac{1}{\gamma}}
$$

The score with respect to the scale parameter $\varphi$ is:

$$
\frac{\partial l}{\partial \varphi}
=
-\frac{1}{\varphi}\left(\frac{\xi}{\gamma}\right)
+
\frac{\xi + \zeta}{\gamma \varphi}
\frac{z_{t|t-1}}{1 + z_{t|t-1}}
$$

The score with respect to the shape parameter $\gamma$ is:

$$
\frac{\partial l}{\partial \gamma}
=
-\frac{1}{\gamma}
-
\frac{\xi}{\gamma^2}
\log\left(
\frac{y_{t}}{\varphi}
\right)
+
\frac{\xi + \zeta}{\gamma^2}
\frac{
z_{t|t-1}\log(z_{t|t-1})
}{
1 + z_{t|t-1}
}
$$

The score with respect to the shape parameter $\xi$ is:

$$
\frac{\partial l}{\partial \xi}
=
\frac{1}{\gamma}
\log\left(
\frac{y_{t}}{\varphi}
\right)
-
\psi(\xi)
+
\psi(\xi + \zeta)
-
\log\left(
1 + z_{t|t-1}
\right)
$$

where $\psi(\cdot)$ is the digamma function.

The score with respect to the shape parameter $\zeta$ is:

$$
\frac{\partial l}{\partial \zeta}
=
-\psi(\zeta)
+
\psi(\xi + \zeta)
-
\log\left(
1 + z_{t|t-1}
\right)
$$

FI:

The matrix is:

$$
I =
\begin{bmatrix}
I_{\varphi\varphi} & I_{\varphi\gamma} & I_{\varphi\xi} & I_{\varphi\zeta} \\
I_{\gamma\varphi} & I_{\gamma\gamma} & I_{\gamma\xi} & I_{\gamma\zeta} \\
I_{\xi\varphi} & I_{\xi\gamma} & I_{\xi\xi} & I_{\xi\zeta} \\
I_{\zeta\varphi} & I_{\zeta\gamma} & I_{\zeta\xi} & I_{\zeta\zeta}
\end{bmatrix}
$$

Let $\psi(\cdot)$ denote the digamma function and
$\psi_1(\cdot)$ denote the trigamma function.

\noindent
The elements of the Fisher Information Matrix are:

$$
\begin{aligned}
\mathcal{I}_{\varphi\varphi}
&=
\frac{\xi}{\varphi^2 \gamma^2}
\frac{\xi+\zeta}{\xi+\zeta+1},
\\[1em]
%
\mathcal{I}_{\varphi\gamma}
&=
-\frac{\xi}{\gamma^2 \varphi}
\frac{\xi+\zeta}{\xi+\zeta+1},
\\[1em]
%
\mathcal{I}_{\varphi\xi}
&=
-\frac{1}{\gamma \varphi}
\frac{\zeta}{\xi+\zeta+1},
\\[1em]
%
\mathcal{I}_{\varphi\zeta}
&=
\frac{1}{\gamma \varphi}
\frac{\xi}{\xi+\zeta+1},
\\[1em]
%
\mathcal{I}_{\gamma\gamma}
&=
\frac{1}{\gamma^2}
\left[
1
+
\xi \psi_1(\xi)
+
\zeta \psi_1(\zeta)
-
(\xi+\zeta)\psi_1(\xi+\zeta)
\right],
\\[1em]
%
\mathcal{I}_{\gamma\xi}
&=
-\frac{1}{\gamma}
\left[
\psi(\xi)
-
\psi(\xi+\zeta)
\right],
\\[1em]
%
\mathcal{I}_{\gamma\zeta}
&=
-\frac{1}{\gamma}
\left[
\psi(\zeta)
-
\psi(\xi+\zeta)
\right],
\\[1em]
%
\mathcal{I}_{\xi\xi}
&=
\psi_1(\xi)
-
\psi_1(\xi+\zeta),
\\[1em]
%
\mathcal{I}_{\xi\zeta}
&=
-\psi_1(\xi+\zeta),
\\[1em]
%
\mathcal{I}_{\zeta\zeta}
&=
\psi_1(\zeta)
-
\psi_1(\xi+\zeta).
\end{aligned}
$$


Once again, check if this is correct before blindly implementing

---

# 6. Dynamic Parameters

Dynamic parameters must be chosen by configuration.

Do not create separate active files for each subset of time-varying parameters.

Avoid implementations such as:

* `gb2_phi_only.py`;
* `gb2_xi_only.py`;
* `gb2_phi_xi.py`.

These create combinatorial growth when additional dynamic parameters are introduced.

Instead use:

```python
dynamic_params = ["phi"]
dynamic_params = ["xi"]
dynamic_params = ["phi", "xi"]
```

The distribution should expose analytical score and Fisher information for all supported parameters.

The model layer should extract the required components according to `dynamic_params`.

---

# 7. Score

I said earlier, when I described the GB2loglink, what according to my calculations the FI and scores are, but double check.

---

# 9. Fisher Information

The distribution module must provide analytical Fisher information.

Do not compute Fisher information numerically at each iteration.

I said earlier, when I described the GB2loglink, what according to my calculations the FI and scores are, but double check.

The distribution should return the full analytical Fisher matrix for all supported parameters.

The model layer extracts the relevant submatrix.

Examples:

* `dynamic_params=["phi"]` uses \(I_{\phi\phi}\);
* `dynamic_params=["xi"]` uses \(I_{\xi\xi}\);
* `dynamic_params=["phi","xi"]` uses the matrix values.

---

# 10. Score Scaling

Score scaling is a model configuration.

Supported options:

```python
scaling = "unit"
scaling = "inverse_fisher"
scaling = "diagonal_inverse_fisher"
```

## 10.1 Unit Scaling

$$
S_t=I.
$$

Thus:

$$
s_t=\nabla_t.
$$

## 10.2 Full Inverse Fisher Scaling

$$
S_t=\mathcal I_t^{-1}.
$$

Thus:

$$
s_t=\mathcal I_t^{-1}\nabla_t.
$$

For multiple dynamic parameters, this includes Fisher cross-scaling.

## 10.3 Diagonal Inverse Fisher Scaling

$$
S_t=\operatorname{diag}(\mathcal I_t)^{-1}.
$$

This removes Fisher cross-scaling while preserving parameter-specific scaling.

---

# 11. Interpretation of Dynamic Xi

When both \(\phi_t\) and \(\xi_t\) are dynamic, the GB2 likelihood couples scale and shape.

Therefore, \(\xi_t\) should not be interpreted as an autonomous tail process independent of \(\phi_t\).

The correct scientific question is:

> Does allowing \(\xi_t\) to vary improve tail-sensitive predictive performance beyond models where only \(\phi_t\) varies?

Full inverse Fisher scaling is statistically natural but mixes raw score components through the Fisher matrix.

Therefore, Diagonal inverse Fisher scaling and unit scaling should be tested against full inverse FI, and results tabled side by side on report, because these reduce cross-parameter influence.

---

# 12. Standard GAS Without Covariates

The baseline dynamic recursion is the standard GAS model without external covariates.

For each transformed dynamic parameter \(f_{j,t}\),

$$
f_{j,t+1}
=

\omega_j
+
\sum_{\ell\in\mathcal L}
A_{j,\ell}s_{j,t-\ell+1}
+
\sum_{\ell\in\mathcal L}
B_{j,\ell}f_{j,t-\ell+1}.
$$

Here:

* \(j\) indexes the dynamic parameter;
* \(\mathcal L\) is the selected lag set;
* \(s_{j,t}\) is the scaled score component;
* \(A_{j,\ell}\) controls score response;
* \(B_{j,\ell}\) controls persistence.

This is the first model class to estimate.

It answers:

> Can score-driven dynamics alone capture the rainfall distribution?

---

# 13. Standard GAS With Covariates

The standard GAS model with covariates adds an exogenous term:

$$
f_{j,t+1}
=

\omega_j
+
\sum_{\ell\in\mathcal L}
A_{j,\ell}s_{j,t-\ell+1}
+
\sum_{\ell\in\mathcal L}
B_{j,\ell}f_{j,t-\ell+1}
+
\Gamma_j'X_t.
$$

Here:

* \(X_t\) is the standard GAS covariate vector;
* \(\Gamma_j\) is the covariate coefficient vector for parameter \(j\).

The model should support arbitrary covariate matrices supplied by the data layer.

The model should not know whether \(X_t\) contains dew point, temperature, ENSO, finance variables, or other application-specific variables.

---

# 14. Lag Structures for Standard GAS

The current thesis pipeline uses controlled lag sets.

Short GAS memory:

$$
\mathcal L_{short}
=

\{1,2,3\}.
$$

Seasonal GAS memory:

$$
\mathcal L_{seasonal}
=

\{1,2,3,364,365,366,367\}.
$$

Avoid uncontrolled lag searches.

The scientific comparison is:

$$
\mathcal L_{short}
\quad
\text{versus}
\quad
\mathcal L_{seasonal}.
$$

This tests whether annual recurrence improves the score-driven model beyond short memory.

---

# 15. Weather Covariate Blocks

Weather covariates are introduced only after the best no-covariate standard GAS setting is selected.

Let \(D_t\) denote dew point and \(T_t\) denote air temperature.

Use lagged values only, to avoid future leakage.

## 15.1 Dew Point Short Block

$$
X^{D,short}_t
=

\left(
D_{t-1},
D_{t-2},
D_{t-3}
\right)'.
$$

## 15.2 Dew Point Seasonal Block

$$
X^{D,seasonal}_t
=

\left(
D_{t-1},
D_{t-2},
D_{t-3},
D_{t-364},
D_{t-365},
D_{t-366},
D_{t-367}
\right)'.
$$

The comparison is:

$$
X^{D,short}_t
\quad
\text{versus}
\quad
X^{D,seasonal}_t.
$$

This tests whether annual dew-point recurrence improves the model beyond short dew-point memory.

---

# 16. Temperature Covariate Blocks

Temperature is tested only after selecting the best dew point specification.

Do not test only \(T_{t-1}\).

That creates ambiguity about whether later improvements come from adding \(T_{t-2}\), \(T_{t-3}\), or seasonal lags.

Let \(X^{D,*}_t\) denote the selected dew point block.

## 16.1 Temperature Short Extension

$$
X^{D+T,short}_t
=

\left[
X^{D,*}_t,
T_{t-1},
T_{t-2},
T_{t-3}
\right].
$$

## 16.2 Temperature Seasonal Extension

$$
X^{D+T,seasonal}_t
=

\left[
X^{D,*}_t,
T_{t-1},
T_{t-2},
T_{t-3},
T_{t-364},
T_{t-365},
T_{t-366},
T_{t-367}
\right].
$$

The seasonal temperature extension should only be tested if the short temperature extension is justified.

---

# 17. Best Weather Specification

After the weather stage, select one best short-run weather specification:

$$
X^{weather,*}_t.
$$

This may be:

* dew point short;
* dew point seasonal;
* dew point plus temperature short;
* dew point plus temperature seasonal.

This selected weather block is then passed to the Harvey long-short model as the short-component covariate matrix:

$$
X^{short}_t = X^{weather,*}_t.
$$

Do not reopen the weather covariate search inside the Harvey stage.

---

# 18. ENSO Covariate Blocks

ENSO is introduced in the Harvey long component.

Let \(E_t\) denote the Niño 3.4 index.

Use lagged values relative to the target rainfall observation to avoid future leakage.

Define:

$$
E^{90}_{t-1}
=

\text{90-day rolling mean of Niño 3.4 available at } t-1,
$$

$$
E^{30}_{t-1}
=

\text{30-day rolling mean of Niño 3.4 available at } t-1.
$$

The three ENSO long-component specifications are:

## 18.1 ENSO 90-Day

$$
X^{long,1}_t
=

\left(
E^{90}_{t-1}
\right)'.
$$

## 18.2 ENSO 90-Day + 30-Day

$$
X^{long,2}_t
=

\left(
E^{90}_{t-1},
E^{30}_{t-1}
\right)'.
$$

## 18.3 ENSO 90-Day + 30-Day + Daily Lag

$$
X^{long,3}_t
=

\left(
E^{90}_{t-1},
E^{30}_{t-1},
E_{t-1}
\right)'.
$$

These are rolling ENSO features, not simply repeated daily lags.

---

# 19. Harvey Long-Short Model

The Harvey long-short model replaces the single standard GAS recursion with two score-driven latent components.

Do not implement the active Harvey model as:

$$
f_t
=

\omega
+
GAS_t
+
L_t
+
S_t.
$$

That is not the intended model.

The intended model is:

$$
f_{j,t}
=

\omega_j
+
L_{j,t}
+
S_{j,t}.
$$

Both (L_{j,t}) and (S_{j,t}) are score-driven.

---

# 20. Harvey Long-Short Dynamics

For each transformed dynamic parameter \(f_{j,t}\),

$$
L_{j,t+1}
=

B_{L,j}L_{j,t}
+
A_{L,j}s_{j,t}
+
\Gamma_{L,j}'X^{long}_t,
$$

and

$$
S_{j,t+1}
=

B_{S,j}S_{j,t}
+
A_{S,j}s_{j,t}
+
\Gamma_{S,j}'X^{short}_t.
$$

The combined dynamic parameter is:

$$
f_{j,t}
=

\omega_j
+
L_{j,t}
+
S_{j,t}.
$$

The persistence restriction is:

$$
0 < B_{S,j} < B_{L,j} < 1.
$$

The restriction ensures that the long component is more persistent than the short component.

The optimizer estimates:

* \(A_{L,j}\), long-component score response;
* \(B_{L,j}\), long-component persistence;
* \(\Gamma_{L,j}\), long-component covariate effects;
* \(A_{S,j}\), short-component score response;
* \(B_{S,j}\), short-component persistence;
* \(\Gamma_{S,j}\), short-component covariate effects.

---

# 21. Harvey Interpretation

Harvey's two-component score-driven model approximates persistent dynamics using components with different persistence rather than many explicit lag coefficients.

In the precipitation application:

The short component captures local atmospheric variation.

The long component captures persistent climate-scale variation.

The main Harvey comparison is between the best covariate models from previous stages, and Harvey decomposition.

This tests whether ENSO adds persistent climate information beyond the selected short-run weather specification.

---

# 22. Regime-Sensitive Score Updates

Regime-sensitive models should be implemented as configurable dynamics but disabled by default.

They should only run when explicitly requested.

The intended regime model is not a hidden Markov model.

It is not a generic switching model.

It modifies the score response when the previous observation is extreme.

Let

$$
R_t(c)
=

\mathbf 1(y_t>c),
$$

where \(c\) is a high threshold such as \(q_{0.95}\) or \(q_{0.98}\).

For a first-order standard GAS model, the normal update is:

$$
f_{j,t+1}
=

\omega_j
+
B_j f_{j,t}
+
A_j s_{j,t}.
$$

The regime-sensitive update is:

$$
f_{j,t+1}
=

\omega_j
+
B_j f_{j,t}
+
\left(
A_j
+
A^{ext}_j R_t(c)
\right)
s_{j,t}.
$$

Equivalently:

If \(y_t \leq c\),

$$
f_{j,t+1}
=

\omega_j
+
B_j f_{j,t}
+
A_j s_{j,t}.
$$

If \(y_t > c\),

$$
f_{j,t+1}
=

\omega_j
+
B_j f_{j,t}
+
(A_j + A^{ext}_j)s_{j,t}.
$$

Thus extreme observations modify the score response.

This tests whether extreme rainfall carries different dynamic information from ordinary rainfall.

---

# 23. Regime Extensions

A covariate version may be written as:

$$
f_{j,t+1}
=

\omega_j
+
B_j f_{j,t}
+
\left(
A_j
+
A^{ext}_j R_t(c)
\right)
s_{j,t}
+
\Gamma_j'X_t.
$$

A Harvey-compatible future extension may allow the extreme indicator to modify the score response in the short component, long component, or both.

These extensions should be implemented only if compatible with the framework design and enabled explicitly.

Default execution flag:

```python
run_regime_models = False
```

---

# 24. Occurrence Model

The occurrence model for \(\pi_t\) is separate from the positive-part distribution.

Use the existing `pi_dynamics/` folder.

Do not rewrite the occurrence model unless explicitly requested.

The occurrence module should eventually behave like a factory-configurable component, but the current implementation should be reused.

---

# 25. Factory Pattern

The framework should use factories or equivalent configuration-driven construction for:

* distributions;
* occurrence dynamics;
* score scaling;
* dynamic recursions;
* evaluation components where useful.

Example:

```python
distribution = DistributionFactory.create("GB2LogLink")
dynamics = DynamicsFactory.create("standard_gas")
scaling = ScalingFactory.create("diagonal_inverse_fisher")
```

Do not force this exact API if the existing code suggests a cleaner implementation.

Preserve the principle:

> model behaviour should be controlled by configuration, not by duplicated implementations.

---

# 26. Standard Thesis Pipeline

The current thesis pipeline should follow this model sequence.

## Stage 1 — Standard GAS, No Covariates

Estimate:

* \(\phi_t\)-only models;
* \((\phi_t,\xi_t)\) models.

Compare:

* short GAS memory;
* seasonal GAS memory;
* score scaling variants.

Select the best no-covariate standard GAS setting.

## Stage 2 — Standard GAS With Weather Covariates

Using the selected standard GAS setting, estimate:

1. dew point short;
2. dew point seasonal.

Select best dew point block.

Then estimate:

3. dew point best + temperature short;
4. dew point best + temperature seasonal only if justified.

Select one best weather block:

$$
X^{weather,*}_t.
$$

## Stage 3 — Harvey Long-Short

Use:

$$
X^{short}_t=X^{weather,*}_t.
$$

Compare Harvey models with:

* no ENSO long covariate;
* \(X^{long,1}_t\);
* \(X^{long,2}_t\);
* \(X^{long,3}_t\).

Select the best Harvey specification.

## Stage 4 — Regime-Sensitive Dynamics

Implemented but disabled by default.

Run only if explicitly requested.

---

# 27. Model Comparison Philosophy

Models should be compared according to the scientific question of each stage.

Do not select models by RMSE alone.

Preferred model evaluation priority:

1. validity and calibration;
2. CRPS;
3. twCRPS and high quantile scores;
4. coverage and exceedance diagnostics;
5. RMSE and MAE/MAD;
6. AIC/BIC as in-sample fit-complexity evidence.

Point forecasting is secondary to distributional and tail performance.

---

# 28. Required Outputs From Every Model

Every fitted model should save enough information to avoid re-running optimization.

At minimum, save:

* model configuration;
* dynamic parameters;
* static parameters;
* optimizer result;
* optimizer status;
* post-fit validity class;
* objective value;
* gradient if available;
* gradient norm;
* inverse Hessian approximation if available;
* approximate standard errors if available;
* filtered states;
* raw scores;
* scaled scores;
* Fisher information summaries;
* fitted probabilities;
* predictive means;
* predictive quantiles;
* PIT values;
* quantile residuals;
* residual ACF data;
* In-sample fit, for expected value, variance, assymetry, kurtosis, q50, q75, q90, q95, q99
* OOS predictions;
* OOS predictive distributions where feasible;
* OOS for expected value, variance, assymetry, kurtosis, q50, q75, q90, q95, q99
* metrics;
* diagnostics;
* warnings;
* runtime;
* memory usage.

Do not discard intermediate or diagnostic outputs.

---

# 29. Standard Errors

For standard errors, use the inverse Hessian approximation returned by the optimizer when available.

For BFGS, SciPy's `result.hess_inv` can be used as an approximate covariance estimate.

Do not compute a fresh numerical finite-difference Hessian for every model.

Expensive numerical Hessians should only be computed for final selected models and only when explicitly requested.

Standard errors should be classified as:

* reliable;
* approximate;
* unreliable.

standard errors for all optimized parameters should also be saved, make sure it is easily retrievable which parameter refers to which error 

---

# 30. Implementation Principle

The codebase should implement this model hierarchy in a reusable way.

Adding a new distribution, new dynamic parameter, new scaling method, or new application should not require duplicating existing model files.

The correct implementation direction is:

* one configurable distribution interface;
* one configurable dynamics interface;
* one configurable score scaling interface;
* one configurable experiment layer.

Avoid combinatorial growth in model files.

The repository should evolve into a general score-driven modelling framework, not a collection of one-off scripts.
