"""
Reads off c_v^phi, c_v^xi, c_v^pi -- the log-link GAS covariate
coefficients used throughout
docs/claude_code_marginal_effects_instructions.md's marginal-effect
formulas -- from a reconstructed fitted model (marginal_effects.model_loader),
converted to RAW physical units (e.g. per +1 degC of dew point, or per one
El-Nino-dummy flip) rather than the standardised units the model actually
fits in.

=============================================================================
WHY THE RAW-UNIT CONVERSION IS NEEDED
=============================================================================

Every covariate fed into CovZAGASModel/HarveyZAGASModel is z-scored on
TRAINING moments only (data/station_loader.py's `_fit_scaler`/`_apply_scaler`,
x_std = (x_raw - mu_train) / std_train). The fitted Gamma coefficients
therefore multiply x_std, not x_raw:

    f_{j,t+1} = ... + Gamma_j' x_std_t = ... + Gamma_j' (x_raw_t - mu)/std

so d f_j / d x_raw = Gamma_j / std_train. Reporting "the marginal effect of
+1 raw unit of DPT/AT/ENSO" (as the instructions doc's Sec.3 "increased by
one unit" and every LaTeX table column require) means every c_v below is
ALREADY divided by that column's training std -- callers never need to
apply the conversion themselves.

=============================================================================
NAMED-VARIABLE -> COLUMN MAPPING
=============================================================================

DPT  -> "dewpoint_lag1"     (block "dewtemp_short")
AT   -> "temperature_lag1"  (block "dewtemp_short")
ENSO -> "el_nino_t"         (block "enso_dummy_current")

ENSO is fit as two dummies (el_nino_t, la_nina_t; neutral is the omitted
baseline -- data/station_loader.py Sec. "ENSO -- El Nino / La Nina
dummies"). A single-column "ENSO marginal effect" is not uniquely defined
for a 2-dummy categorical variable, so this module makes an explicit,
documented choice: the reported ENSO effect is the effect of the
Neutral -> El Nino transition (the el_nino_t coefficient), holding
la_nina_t fixed at its observed/forecast value. This is a genuine "one-time
impulse" (dummy flips 0->1, matching the instructions doc's "+1 unit"
literally) rather than a linearised derivative. The symmetric La Nina
effect (the la_nina_t coefficient, generally NOT the negative of the El
Nino one) is available via `la_nina_coefficients()` below for a
supplementary check but is not part of the required tables.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from marginal_effects.model_loader import LoadedModel

VARIABLE_COLUMN: Dict[str, str] = {
    "DPT": "dewpoint_lag1",
    "AT": "temperature_lag1",
    "ENSO": "el_nino_t",
}
COLUMN_BLOCK: Dict[str, str] = {
    "dewpoint_lag1": "dewtemp_short",
    "temperature_lag1": "dewtemp_short",
    "el_nino_t": "enso_dummy_current",
    "la_nina_t": "enso_dummy_current",
}
VARIABLES = tuple(VARIABLE_COLUMN.keys())


@dataclass
class CovariateCoefficients:
    """Raw-physical-unit coefficients for one named variable in one model.
    c_phi/c_xi are 0.0 (not None) when the variable does not enter that
    parameter's dynamics -- matches the instructions doc's "Use zero for a
    coefficient when the variable does not enter the corresponding
    parameter dynamics."""
    variable: str
    column: str
    in_model: bool          # whether `column` is among this model's covariates at all
    c_phi: float
    c_xi: float
    c_pi: float
    std_train: Optional[float]  # training std used for the raw-unit conversion (None if not in model)


def _column_std(data: dict, column: str) -> float:
    block = COLUMN_BLOCK[column]
    return float(data["covariate_scalers"][block]["std"][column])


def _cov_gamma_std(model, theta: np.ndarray) -> Dict[str, np.ndarray]:
    """CovZAGASModel: {tv_name: Gamma_j array (standardised-covariate units)}."""
    _, _, cov_theta = model._split_theta(theta)
    return model._decode_cov(cov_theta)


def _harvey_decoded(model, theta: np.ndarray) -> Dict[str, dict]:
    """HarveyZAGASModel: {tv_name: {..., "Gamma_L":..., "Gamma_S":...}}."""
    _, _, hv_theta = model._split_theta(theta)
    return model._decode_harvey(hv_theta)


def occurrence_coefficient(model, column: str) -> float:
    """c_v^pi: the covariate's coefficient in the occurrence (pi_t) logit
    predictor. Every pi_dynamics/ class in this repo (ar_logistic,
    ar_logistic_custom_lags, phi_linked) is driven only by lagged y (and/or
    eta_{t-1}, and/or the frozen magnitude path) -- none accepts exogenous
    covariates (docs/MODELS.md Sec.24 explicitly scopes covariates out of
    the occurrence model). This always returns 0.0 today; it is a real
    function (not a stub) so the ZA product-rule terms in onestep.py stay
    mathematically general and require no change if a covariate occurrence
    model is ever added."""
    return 0.0


def covariate_coefficients(loaded: LoadedModel, variable: str) -> CovariateCoefficients:
    """Raw-unit c_phi/c_xi/c_pi for `variable` in `loaded.model`, or all
    zeros with in_model=False if this model does not use that variable."""
    if variable not in VARIABLE_COLUMN:
        raise ValueError(f"unknown variable {variable!r}; expected one of {VARIABLES}")
    column = VARIABLE_COLUMN[variable]
    model, theta, data = loaded.model, loaded.theta, loaded.data
    model_type = type(model).__name__

    c_phi_std = c_xi_std = 0.0
    in_model = False

    if model_type == "CovZAGASModel":
        cov_names = list(model.cov_names)
        if column in cov_names:
            idx = cov_names.index(column)
            gamma = _cov_gamma_std(model, theta)
            in_model = True
            if "phi" in model.cov_target:
                c_phi_std = float(gamma["phi"][idx])
            if "xi" in model.cov_target:
                c_xi_std = float(gamma["xi"][idx])
    elif model_type == "HarveyZAGASModel":
        long_names = list(model.long_names)
        short_names = list(model.short_names)
        # include_long=False (the stage4S "structural" Harvey variant) hard-sets
        # Gamma_L to the CONSTANT 0 for every dynamic parameter -- not a fitted
        # value that happens to be small, a parameter the optimiser never had a
        # theta slot for (see models/harvey_gas.py's _decode_harvey and
        # marginal_effects/model_selection.py's _variables_in_cfg docstring for
        # the full story). A long_names column is therefore only really "in the
        # model" when include_long is also True; the short block has no such
        # override and is always structurally active.
        in_long = column in long_names and getattr(model, "include_long", True)
        in_short = column in short_names
        if in_long or in_short:
            in_model = True
            hvd = _harvey_decoded(model, theta)
            for tv in model.tv_names:
                if tv not in hvd:
                    continue  # tv is static in this model (e.g. xi in a phi-only branch)
                if in_long:
                    val = float(hvd[tv]["Gamma_L"][long_names.index(column)])
                elif in_short:
                    val = float(hvd[tv]["Gamma_S"][short_names.index(column)])
                else:
                    val = 0.0
                if tv == "phi":
                    c_phi_std = val
                elif tv == "xi":
                    c_xi_std = val
    elif model_type == "ZAGASModel":
        pass  # no covariates at all -- in_model stays False, all coefficients 0
    else:
        raise ValueError(f"unrecognised model type {model_type!r}")

    c_pi = occurrence_coefficient(model, column)

    if not in_model:
        return CovariateCoefficients(variable, column, False, 0.0, 0.0, c_pi, None)

    std_train = _column_std(data, column)
    return CovariateCoefficients(
        variable=variable, column=column, in_model=True,
        c_phi=c_phi_std / std_train, c_xi=c_xi_std / std_train, c_pi=c_pi,
        std_train=std_train,
    )


def all_covariate_coefficients(loaded: LoadedModel) -> Dict[str, CovariateCoefficients]:
    return {v: covariate_coefficients(loaded, v) for v in VARIABLES}


def column_std(data: dict, column: str) -> float:
    """Public accessor for a raw covariate column's training std (used by
    marginal_effects.multistep for columns not in VARIABLE_COLUMN, e.g.
    "la_nina_t")."""
    return _column_std(data, column)


def column_in_model(model, column: str) -> bool:
    """Whether `column` (any raw covariate name, e.g. "la_nina_t") is part
    of this model's fitted covariate set at all. See covariate_coefficients'
    Harvey branch: include_long=False (stage4S) hard-zeroes the long block,
    so a long_names column there is not really "in" the model."""
    model_type = type(model).__name__
    if model_type == "CovZAGASModel":
        return column in list(model.cov_names)
    if model_type == "HarveyZAGASModel":
        in_long = column in list(model.long_names) and getattr(model, "include_long", True)
        in_short = column in list(model.short_names)
        return in_long or in_short
    return False
