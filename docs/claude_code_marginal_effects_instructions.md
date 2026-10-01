# Task: marginal effects of external variables in rolling ZA-GB2 forecasts

Inspect the existing project and implement calculation and export of marginal effects for every fitted location and every external variable included in the selected model. Reuse the existing fitted-model objects, forecasting functions, parameter transformations, covariate preprocessing, train/test split, and location loop. Do not create a separate model implementation.

## 1. Mathematical definitions

For the positive-rainfall component,

\[
Y_t\mid(Y_t>0,\mathcal F_{t-1})\sim
\operatorname{GB2}(\varphi_t,\gamma,\xi_t,\zeta),
\qquad
R_{k,t}=\frac{B(\xi_t+k\gamma,\zeta-k\gamma)}{B(\xi_t,\zeta)}.
\]

Define

\[
m_{1,t}=\varphi_tR_{1,t},\qquad
m_{2,t}=\varphi_t^2R_{2,t},\qquad
\sigma_t^2=m_{2,t}-m_{1,t}^2,
\]

and

\[
O_{t}^{k}=\psi(\xi_t+k\gamma)-\psi(\xi_t).
\]

Assume the external variable \(x_{v,t-1}\) enters the log dynamics of \(\varphi_t\) and \(\xi_t\) with coefficients \(c_v^\varphi\) and \(c_v^\xi\). For models in which only \(\varphi_t\) varies, use

\[
\frac{\partial m_{1,t}}{\partial x_{v,t-1}}=c_v^\varphi m_{1,t},
\qquad
\frac{\partial\sigma_t^2}{\partial x_{v,t-1}}=2c_v^\varphi\sigma_t^2.
\]

For models in which both \(\varphi_t\) and \(\xi_t\) vary, use

\[
\frac{\partial m_{1,t}}{\partial x_{v,t-1}}
=m_{1,t}\left(c_v^\varphi+\xi_tc_v^\xi O_t^1\right),
\]

\[
\frac{\partial\sigma_t^2}{\partial x_{v,t-1}}
=2c_v^\varphi\sigma_t^2
+\xi_tc_v^\xi\left(m_{2,t}O_t^2-2m_{1,t}^2O_t^1\right).
\]

Use zero for a coefficient when the variable does not enter the corresponding parameter dynamics. Respect the actual link function in the code; the formulas above assume log links. If a different link is used, apply its derivative explicitly.

For the zero-augmented distribution, let \(\pi_t=P(Y_t>0\mid\mathcal F_{t-1})\). Its moments are

\[
M_{1,t}=\pi_tm_{1,t},\qquad
V_{M,t}=\pi_tm_{2,t}-\pi_t^2m_{1,t}^2
=\pi_t\sigma_t^2+\pi_t(1-\pi_t)m_{1,t}^2.
\]

If the external variable does not enter the occurrence equation, calculate

\[
\frac{\partial M_{1,t}}{\partial x}=\pi_t\frac{\partial m_{1,t}}{\partial x},
\]

\[
\frac{\partial V_{M,t}}{\partial x}
=\pi_t\frac{\partial\sigma_t^2}{\partial x}
+2\pi_t(1-\pi_t)m_{1,t}\frac{\partial m_{1,t}}{\partial x}.
\]

If the variable also enters the occurrence equation, include the additional \(\partial\pi_t/\partial x\) terms by applying the product rule; for a logit predictor with coefficient \(c_v^\pi\), \(\partial\pi_t/\partial x=c_v^\pi\pi_t(1-\pi_t)\).

## 2. Rolling one-step-ahead effects

At every out-of-sample rolling forecast origin and for every location:

1. Obtain the one-step-ahead states \(\varphi_{t|t-1}\), \(\xi_{t|t-1}\), and \(\pi_{t|t-1}\) from the existing filter.
2. Compute the analytical marginal effects above for every external variable in the fitted specification, holding the previous filtered state and realized score fixed. The marginal effect happens at time t, not anytime in the projected sample or the future.
3. Store both positive-GB2 effects and full ZA-GB2 effects.
4. Preserve one row per location, forecast date, and external variable. Also produce average marginal effects across the rolling evaluation period, with the mean, median, standard deviation, and the 2.5%, 25%, 75%, and 97.5% empirical quantiles.

## 3. Multi-step effects

Estimate effects for horizons

\[
h\in\{1,2,5,10,30,60,90\}
\]

using paired Monte Carlo simulation from every rolling forecast origin for which the required horizon fits inside the evaluation period.

For each location, origin, external variable, and horizon:

1. Run a baseline forecast path and a counterfactual path in which the selected external variable at the intervention date is increased by one unit. Keep all other observed or forecast external-variable values unchanged. Treat this as a one-time impulse unless the existing model explicitly defines a sustained scenario.
2. Use at least 10,000 paired paths. Use common random numbers: the baseline and counterfactual member of each pair must use identical underlying uniform draws for occurrence and GB2 innovations at every step. Each scenario must nevertheless update its own GAS states and scores.
3. Use fixed fitted parameters; this exercise measures conditional predictive effects, not parameter uncertainty.
4. For terminal rainfall \(Y_{t+h}\), calculate the effects
   \[
   \Delta\mu_h=E_1(Y_{t+h})-E_0(Y_{t+h}),\qquad
   \Delta V_h=\operatorname{Var}_1(Y_{t+h})-\operatorname{Var}_0(Y_{t+h}).
   \]
   These are finite one-unit effects. Do not label them exact derivatives.
5. If the existing forecast target is cumulative rainfall, additionally calculate the same effects for \(\sum_{r=1}^{h}Y_{t+r}\), clearly distinguishing terminal from cumulative results.
6. Estimate Monte Carlo uncertainty. For the mean effect, use the standard error of paired differences. For the variance effect, use a paired-path bootstrap or several independent paired batches. Save the Monte Carlo standard error and a 95% Monte Carlo interval.
7. Use 123 as seed, or whatecver you already used for the multistep ahead prediction. Add a convergence check that repeats a representative subset with more paths and warns when the Monte Carlo standard error is large relative to the estimated effect.

## 4. Outputs and validation

Write in reports/marginal effects, a latex file containing the following tables:

1. for each location (6 rows), two header columns: GAS(p,q), long-short models. Then each one divided in 3 columns: DPT (dew point temperature), AT (air temperature), ENSO, and the marginal effect on the mean first, and then the variance for each of them. mean first then 6 rows for the variance. first for the models with one time varying parameter.

2. same for two time varying parameters.

3. long tables with the projected h steps ahead for each location for the mean and then a second table for the variance, so 4 tables for one and two time-varying parameters.

## TAKE CARE

Use the model that has gradinet norm of less than one and that minimized RMSE and CRPS, prefer a model with 3 external variables. If it is not possible, try to get at least one model that contains each variable. Tell me which ones you picked. If there are no suitable models for one set of dynamics, number of time varying parameters, or external variables, let me know, dont use badly optimized models if you cant find one that suits.
