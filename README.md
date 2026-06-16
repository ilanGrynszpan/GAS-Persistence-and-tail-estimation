# ZA-GAS Framework

Zero-Augmented Generalised Autoregressive Score (ZA-GAS) model for precipitation time series, following **Creal, Koopman & Lucas (2012)**.

## Model overview

The observation model mixes a point mass at zero with a positive continuous distribution:

$$p(y_t \mid \mathcal{F}_{t-1}) = (1 - \pi_t)\,\mathbf{1}[y_t = 0] \;+\; \pi_t\,g(y_t;\,f_t,\,\theta)\,\mathbf{1}[y_t > 0]$$

**Three cleanly separated components:**

| Component | Location | Description |
|-----------|----------|-------------|
| **Distribution** | `distributions/` | Positive-part density (GB2 log-link) |
| **Pi dynamics** | `pi_dynamics/` | AR-logistic model for π_t |
| **GAS filter** | `models/` | Creal et al. GAS(L,L) update for f_t |

### GAS update equation

$$f_{t+1} = \omega + \sum_{l \in L} A_l\,s_{t-l+1} + \sum_{l \in L} B_l\,f_{t-l+1}$$

where $s_t = \mathcal{I}(f_t)^{-1}\,\nabla_t$ is the scaled score (scaling type 1, Creal et al. eq. 7).

### Seasonal lag sets

| Cadence | Lag set L |
|---------|-----------|
| Daily   | {1, 2, 3, 364, 365, 366, 367} |
| Monthly | {1, 2, 3, 11, 12, 13} |

## Package structure

```
FurtherTopics/
├── distributions/
│   ├── base.py              # Abstract Distribution interface
│   ├── gb2_log_link.py      # GB2 log-link (phi & xi TV)
│   ├── gb2_phi_only.py      # GB2 log-link, phi TV only
│   └── gb2_xi_only.py       # GB2 log-link, xi TV only
├── pi_dynamics/
│   ├── base.py              # Abstract PiDynamics interface
│   ├── ar_logistic.py       # AR(1) logistic with seasonal y-lags
│   └── factory.py           # PiDynamicsFactory
├── models/
│   ├── gas_filter.py        # GAS(L,L) filter + _ParamCodec
│   ├── za_gas_model.py      # Baseline ZA-GAS model
│   ├── factory.py           # build_zagas_model convenience factory
│   ├── cov_gas_model.py     # Exogenous GAS-X model (covariates in update equation)
│   ├── long_short_gas.py    # Long-short ZA-GAS (L_t / S_t decomposition)
│   └── inference.py         # Numerical Hessian → SE, CI, z-test
├── covariates/
│   ├── __init__.py          # Package exports
│   ├── standardizer.py      # CovariateStandardizer (train-set moments only)
│   └── builder.py           # ERA5/MJO/Niño3.4 loaders + lag block builders
├── diagnostics/
│   ├── residuals.py         # Randomised PIT, quantile residuals
│   ├── information.py       # AIC, BIC
│   ├── tests.py             # Kupiec, Christoffersen, Jarque-Bera
│   ├── plots.py             # Diagnostic plots
│   ├── latex.py             # LaTeX table helpers
│   └── scoring.py           # CRPS, twCRPS, quantile scores, Brier scores
├── simulation/
│   └── simulator.py         # OOS rolling forecast + basic metrics
├── report/
│   ├── summarize.py         # Baseline PDF/LaTeX report
│   └── cov_report.py        # Covariate/long-short PDF+LaTeX report
├── constants.py             # SEASONAL_LAGS
├── analysis.ipynb           # Baseline GAS comparison notebook
├── model_comparison.ipynb   # Monthly/daily model comparison notebook
├── covariates_long_short.ipynb  # Exogenous + long-short analysis notebook
└── requirements.txt
```

## Quick start

```python
from distributions.gb2_log_link import GB2LogLink
from pi_dynamics.factory        import PiDynamicsFactory
from models.za_gas_model        import ZAGASModel

model = ZAGASModel(
    distribution = GB2LogLink(),
    pi_dynamics  = PiDynamicsFactory.get('ar_logistic'),
    seasonal     = 'daily',   # or 'monthly'
)

fit = model.fit(y_train)
print(f"loglik = {fit['loglik']:.2f}")

paths = model.filter(fit['theta'], y_train)
```

---
## Covariate and Long-Short Models

### Exogenous GAS (GAS-X)

The standard exogenous GAS update augments the state equation with contemporaneous
covariates $X_t$:

$$f_{j,t+1} = \omega_j + \sum_{l \in L} A_{j,l}\,s_{j,t-l+1} + \sum_{l \in L} B_{j,l}\,f_{j,t-l+1} + \Gamma_j' X_t$$

where $j \in \{\phi, \xi\}$ depends on which parameters are time-varying.

Implemented in `models/cov_gas_model.py` (`CovZAGASModel`).

### Long-Short Decomposition

The long-short model decomposes the time-varying filter into GAS, long-run (ENSO), and
short-run (local weather) components:

$$h(\theta_t) = \omega_\theta + \text{GAS}^\theta_t + L^\theta_t + S^\theta_t$$

$$L^\theta_t = \alpha^\theta_L L^\theta_{t-1} + (\beta^\theta_L)' X^{\text{long}}_t, \qquad S^\theta_t = (\Gamma^\theta_S)' X^{\text{short}}_t$$

**Long component** $L_t$: persistent AR(1) state driven by Niño 3.4 ($N34_t$, rolling means).  
**Short component** $S_t$: contemporaneous ERA5 dew point $D_t$, temperature $T_t$, MJO $RMM1_t, RMM2_t$.

Implemented in `models/long_short_gas.py` (`LongShortZAGASModel`).

### Covariate Sources

| Variable | Source | Path |
|----------|--------|------|
| ERA5 dew point $D_t$ | ERA5 reanalysis | `data/input/ERA5/humidity/` |
| ERA5 temperature $T_t$ | ERA5 reanalysis | `data/input/ERA5/temperature/` |
| MJO RMM1/2 | BOM (Wheeler & Hendon 2004) | `data/input/pacific/MJO.csv` |
| Niño 3.4 $N34_t$ | NOAA/PSL | `data/processed/pacific/NINO34_daily.csv` |

### Model Naming Convention

| Prefix | Target TV parameter(s) |
|--------|------------------------|
| `xi_*` | $\xi_t$ only |
| `phi_*` | $\phi_t$ only |
| `phi_xi_*` | both $\phi_t$ and $\xi_t$ |
| `long_short_xi_*` | long-short on $\xi_t$ |
| `long_short_phi_*` | long-short on $\phi_t$ |
| `long_short_phi_xi_*` | long-short on both |

Suffix (covariate block) examples: `dewpoint_t`, `nino34_lags`, `weather_climate`, `interactions`.

### Notebook: `covariates_long_short.ipynb`

Main analysis notebook. Configuration cell controls all paths and run flags:

```python
FORCE_REFIT       = False   # Load from cache if available
RUN_XI            = True    # Standard GAS with xi_t TV
RUN_PHI           = True    # Standard GAS with phi_t TV
RUN_PHI_XI        = True    # Standard GAS with phi_t, xi_t TV
RUN_LS_XI         = True    # Long-short on xi_t
RUN_LS_PHI        = True    # Long-short on phi_t
RUN_LS_PHI_XI     = True    # Long-short on phi_t, xi_t
```

Outputs:
- **Cache**: `artifacts/cache_covariates_long_short/{model_id}/` — estimated parameters,
  filtered paths, PIT series, metrics, inference tables, covariance matrices.
- **Report**: `artifacts/reports/covariates_long_short/report.pdf` + `report.tex`.

To regenerate the PDF from an edited `report.tex` without re-running models:

```python
from report.cov_report import recompile_pdf_from_latex
recompile_pdf_from_latex("artifacts/reports/covariates_long_short/report.tex")
```

Or open `report.tex` in Overleaf and compile there.

---
## Running the comparison notebook

Use `model_comparison.ipynb` for the thesis comparison report. The main
switch is in the **Configuration** cell:

```python
RUN_SCOPE = "monthly"  # choose "monthly", "daily", or "both"
```

- `"monthly"` runs or loads only the monthly models.
- `"daily"` runs or loads only the daily models.
- `"both"` runs or loads both and renders the combined report.

The monthly thesis trial specification uses standard BFGS with no bounds:

```python
MONTHLY_FIT_METHOD = "BFGS"
MONTHLY_USE_BOUNDS = False
MONTHLY_BFGS_CACHE_DIR = Path("artifacts/cache_monthly_output_bfgs_216")
```

Keep `FORCE_REFIT = False` to load cached `estimated_parameters.csv` files and
regenerate diagnostics/reports without running optimization again. Set
`FORCE_REFIT = True` only when you intentionally want to re-estimate models.
When changing the dataset path, use a new cache directory or intentionally set
`FORCE_REFIT = True`; otherwise the notebook may load estimates from a previous
dataset with the same model id.

The report is regenerated by rerunning the notebook cells from the data-loading
cell through **Render PDF and LaTeX report**. The report renderer itself is in
`report/summarize.py`; the notebook passes the selected cached/fitted model
results to `render_model_report`.

## Changing model specifications

### Time-varying positive-part parameters

The positive distribution is GB2 with log-link parameters:

- `phi`: log scale, time-varying in every comparison model.
- `xi`: log shape, either static or time-varying depending on the model.
- `gamma`, `zeta`: static GB2 shape parameters.

The standard comparison models are built in `models/factory.py`:

```python
build_zagas_model("phi", seasonal="monthly")
build_zagas_model("phi_xi", seasonal="monthly")
```

For `model_type="phi"`, the factory uses `GB2LogLinkPhiOnly`, whose
`tv_param_names` are `["phi"]`, and estimates `xi`, `gamma`, and `zeta` as
static parameters.

For `model_type="phi_xi"`, the factory uses `GB2LogLink`, whose
`tv_param_names` are `["phi", "xi"]`, and estimates `gamma` and `zeta` as
static parameters.

To add a new set of time-varying parameters, create or subclass a distribution
in `distributions/` and change its `tv_param_names`, then register a new branch
in `models/factory.py`.

### GAS dynamics and seasonal lags

The GAS state recursion is implemented in `models/gas_filter.py`. For each
time-varying parameter `j`, the filter estimates:

```text
omega_j, f0_j, A_j_l for l in L, B_j_l for l in L
```

The update is:

```text
f_{t+1|t} = omega + sum_{l in L} A_l s_{t-l+1}
                  + sum_{l in L} B_l f_{t-l+1|t-l}
```

The lag set `L` is shared by the GAS filter and the pi dynamics and is defined
in `constants.py`:

```python
SEASONAL_LAGS = {
    "daily": [1, 2, 3, 364, 365, 366, 367],
    "monthly": [1, 2, 3, 11, 12, 13],
}
```

Reducing the monthly lag set is the quickest way to reduce the number of
parameters and improve stability when the monthly sample is short.

### Pi dynamics

The zero/positive probability is modeled separately from the positive GB2
distribution. The current implementation is `ARLogisticPiDynamics` in
`pi_dynamics/ar_logistic.py`:

```text
pi_t = Lambda(eta_t)
eta_t = omega0 + rho eta_{t-1} + sum_{l in L} omega_y_l y_{t-l}
```

where `Lambda(x) = 1 / (1 + exp(-x))`. Its parameters are:

```text
omega0, rho, omega_y_l for each lag l in L
```

To change the pi dynamics, either edit `ARLogisticPiDynamics` or create a new
class that subclasses `pi_dynamics.base.PiDynamics`, implements
`param_names`, `default_bounds`, `initial_params`, and `compute_eta`, and then
use it in `models/factory.py`.

### Optimizer and bounds

The optimizer is controlled by `ZAGASModel.fit` in `models/za_gas_model.py`.
The bounded default is `L-BFGS-B`. The monthly thesis trial in
`model_comparison.ipynb` intentionally uses:

```python
model.fit(y_train, method="BFGS", use_bounds=False)
```

This can improve in-sample likelihood and PIT behavior, but it can also create
unstable out-of-sample forecasts when the model is overparameterized. Use a
separate cache directory for these trials so bounded and unbounded estimates do
not overwrite each other.

## Extending the framework

### New distribution

Subclass `distributions.base.Distribution` and implement:
- `tv_param_names` — list of time-varying parameter names
- `logpdf`, `score`, `fisher_info_diag`, `cdf`, `rvs`

### New pi dynamics

Subclass `pi_dynamics.base.PiDynamics`, implement `param_names`, `default_bounds`, `initial_params`, `compute_eta`, then register:

```python
from pi_dynamics.factory import PiDynamicsFactory
PiDynamicsFactory.register('my_name', MyDynamics)
```

## Setup

### Python 3.13 virtual environment (Windows)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Docker-ready (Linux/macOS)

```dockerfile
FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
```

Then run notebooks with:

```bash
docker run --rm -p 8888:8888 -v $(pwd)/artifacts:/app/artifacts \
  <image> jupyter notebook --ip=0.0.0.0 --no-browser --allow-root
```

## Dependencies

```
numpy>=2.0  scipy>=1.14  pandas>=2.2  matplotlib>=3.9
jupyter>=1.1  statsmodels>=0.14
```

See `requirements.txt` for pinned versions compatible with Python 3.13.

## Reference

Creal, D., Koopman, S. J., & Lucas, A. (2012). *Generalized autoregressive score models with applications*. Journal of Applied Econometrics, 28(5), 777–795.
