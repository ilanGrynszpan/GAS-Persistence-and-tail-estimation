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
│   ├── base.py           # Abstract Distribution interface
│   └── gb2_log_link.py   # GB2 with log-link (phi, xi time-varying; gamma, zeta static)
├── pi_dynamics/
│   ├── base.py           # Abstract PiDynamics interface
│   ├── ar_logistic.py    # AR(1) logistic with seasonal y-lags
│   └── factory.py        # PiDynamicsFactory (register custom dynamics here)
├── models/
│   ├── lags.py           # SEASONAL_LAGS constant
│   ├── gas_filter.py     # GAS(L,L) filter
│   └── za_gas_model.py   # Combined ZA-GAS model
├── diagnostics/
│   ├── residuals.py      # Quantile residuals, PIT
│   ├── information.py    # AIC, BIC
│   ├── tests.py          # Kupiec, Christoffersen, Jarque-Bera
│   ├── plots.py          # Diagnostic plots + mosaic helper
│   └── latex.py          # LaTeX table generation
├── simulation/
│   └── simulator.py      # OOS evaluation (RMSE, MAD, CRPS)
└── analysis.ipynb        # Main analysis notebook
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

## Dependencies

```
numpy scipy pandas matplotlib statsmodels
```

## Reference

Creal, D., Koopman, S. J., & Lucas, A. (2012). *Generalized autoregressive score models with applications*. Journal of Applied Econometrics, 28(5), 777–795.
