"""
Core action surface model implementation and fitting routines.

This module defines the input-modulated Hill equation used to describe
optogenetic dose-response relationships as a function of both wavelength
and irradiance, as well as fitting routines for parameter estimation.
"""

import numpy as np
import pandas as pd
import patsy
from pydantic import BaseModel
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold




def action_surface_response(data, K, n, log10_amp_max, lambda_max, sigma, baseline):
    """
    Calculate the action surface response.

    This function implements the core mathematical model:
    Response = A_max * (I_scaled^n) / (I_scaled^n + K^n) + baseline
    where I_scaled = I * exp(-(λ - λ_max)² / (2σ²))

    Args:
        data: Tuple of (wavelength, irradiance) arrays or lists
        K: Half-saturation constant
        n: Hill coefficient
        log10_amp_max: Log10 of maximum amplitude
        lambda_max: Peak wavelength sensitivity
        sigma: Spectral width
        baseline: Baseline response

    Returns:
        Array of predicted responses
    """
    wavelength, irradiance = data

    # Convert to numpy arrays if needed
    wavelength = np.asarray(wavelength)
    irradiance = np.asarray(irradiance)

    # Apply spectral scaling
    irradiance_scaled = irradiance * np.exp(-((wavelength - lambda_max) ** 2) / (2 * (sigma**2 + 1e-8)))

    # Convert log amplitude to linear scale
    amp_max = 10**log10_amp_max

    # Apply Hill equation
    return amp_max * (irradiance_scaled**n) / (irradiance_scaled**n + K**n) + baseline


def find_threshold_irradiance(wavelength, threshold_current, K, n, log10_amp_max, lambda_max, sigma, baseline):
    """
    Calculate the irradiance needed to reach a threshold current.

    Uses closed-form solution to the Hill equation to find the irradiance
    at which the response equals a specified threshold.

    Args:
        wavelength: Wavelength to evaluate at (nm) - scalar or array
        threshold_current: Target current threshold (pA)
        K, n, log10_amp_max, lambda_max, sigma, baseline: Model parameters

    Returns:
        Irradiance at threshold (mW/mm²), or None if unreachable
        For array inputs, returns array with None for unreachable wavelengths
    """
    # Convert to numpy array if needed
    wavelength = np.asarray(wavelength)
    scalar_input = wavelength.ndim == 0
    wavelength = np.atleast_1d(wavelength)

    amp_max = 10**log10_amp_max

    # Account for baseline
    threshold_above_baseline = threshold_current - baseline

    # Check if threshold is achievable
    if threshold_above_baseline > amp_max or threshold_above_baseline < 0:
        return None if scalar_input else np.full_like(wavelength, None, dtype=float)

    # Calculate spectral scaling factor
    scaling_factor = np.exp(-((wavelength - lambda_max) ** 2) / (2 * (sigma**2 + 1e-8)))

    # Solve for scaled irradiance using rearranged Hill equation
    I_scaled = K * ((threshold_above_baseline) / (amp_max - threshold_above_baseline)) ** (1 / n)

    # Correct for spectral scaling
    irradiance = I_scaled / scaling_factor

    # Return scalar if input was scalar
    if scalar_input:
        return float(irradiance[0])

    return irradiance





class Parameter(BaseModel):
    """
    Represents a parameter for the model.
    """

    initial_guess: float
    bounds_lower: float
    bounds_upper: float


class ParameterSet(BaseModel):
    """Base class for a set of parameters."""

    @property
    def parameters(self):
        """Returns a list of parameter names defined in the model fields."""
        return [k for k, v in self.model_fields.items() if v.annotation == Parameter]

    @property
    def initial_guess(self):
        """Returns a list of initial guess values for each parameter."""
        return [getattr(self, field).initial_guess for field in self.parameters]

    @property
    def lower_bounds(self):
        """Returns a list of lower bounds for each parameter."""
        return [getattr(self, field).bounds_lower for field in self.parameters]

    @property
    def upper_bounds(self):
        """Returns a list of upper bounds for each parameter."""
        return [getattr(self, field).bounds_upper for field in self.parameters]

    @property
    def bounds(self):
        """Returns a tuple of lower and upper bounds for all parameters."""
        return (self.lower_bounds, self.upper_bounds)


class ActionSurfaceParameters(ParameterSet):
    """
    Parameters for the input-modulated Hill model (action surface).

    Attributes:
        K: Half-saturation constant (mW/mm²)
        n: Hill coefficient (cooperativity)
        log10_amp_max: Log10 of maximum amplitude (pA)
        lambda_max: Peak wavelength sensitivity (nm)
        sigma: Spectral width (nm)
        baseline: Baseline response (pA)
    """

    K: Parameter = Parameter(initial_guess=1, bounds_lower=1e-4, bounds_upper=1e2)
    n: Parameter = Parameter(initial_guess=1, bounds_lower=0.6, bounds_upper=1.2)
    log10_amp_max: Parameter = Parameter(initial_guess=3.0, bounds_lower=1.7, bounds_upper=4.5)
    lambda_max: Parameter = Parameter(initial_guess=500, bounds_lower=375, bounds_upper=800)
    sigma: Parameter = Parameter(initial_guess=40, bounds_lower=20, bounds_upper=55)
    baseline: Parameter = Parameter(initial_guess=0.0, bounds_lower=-150, bounds_upper=150)


class HillModelConfig(BaseModel):
    """
    Configuration for Hill model fitting.

    Attributes:
        formula: Patsy formula defining model structure
        n_folds: Number of cross-validation folds
        hill_model_params: Parameter configuration
    """

    formula: str = "peak_current ~ wavelength + irradiance - 1"
    n_folds: int = 5
    hill_model_params: ActionSurfaceParameters = ActionSurfaceParameters()

class HillModel:
    """
    Hill model for fitting and analyzing dose-response data.
    """

    def __init__(self, config: HillModelConfig):
        self.config = config
        self.mse_scores = []
        self.r2_scores = []
        self.fold_results = []
        self.popts = {}
        self.pcovs = {}

    def fit(self, df):
        """
        Fit the Hill model using k-fold cross-validation.

        Args:
            df: DataFrame with columns matching the formula specification
        """
        self.df = df
        self.y, self.X = patsy.dmatrices(self.config.formula, df, return_type="matrix")

        kf = KFold(n_splits=self.config.n_folds, shuffle=True, random_state=42)

        for ii, (train_index, test_index) in enumerate(kf.split(self.X)):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            # Dynamic initialization based on data
            amp_max_init = np.quantile(y_train, 0.95) * 1.2
            amp_max_init = max(amp_max_init, 200)  # at least 200 pA
            amp_max_init = min(amp_max_init, 20000)  # at most 20000 pA
            log10_amp_max_init = np.log10(amp_max_init)

            self.config.hill_model_params.log10_amp_max.initial_guess = log10_amp_max_init
            self.config.hill_model_params.log10_amp_max.bounds_upper = min(np.log10(amp_max_init * 4), 4.5)

            # Fit the model
            popt, pcov = curve_fit(
                action_surface_response,
                (X_train[:, 0], X_train[:, 1]),
                np.ravel(y_train),
                p0=self.config.hill_model_params.initial_guess,
                bounds=self.config.hill_model_params.bounds,
                maxfev=20000,
            )

            self.popts[ii] = popt
            self.pcovs[ii] = pcov

            # Evaluate on test set
            y_pred = action_surface_response((X_test[:, 0], X_test[:, 1]), *popt)

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            self.mse_scores.append(mse)
            self.r2_scores.append(r2)

            # Store results
            fold_result = {
                "fold": ii,
                "train_index": train_index,
                "test_index": test_index,
                "popt": popt,
                "pcov": pcov,
                "mse": mse,
                "r2": r2,
            }
            self.fold_results.append(fold_result)

    @property
    def parameters_by_fold(self):
        """Get parameters for each fold as a DataFrame."""
        params = []
        param_names = self.config.hill_model_params.parameters

        for fold_idx, popt in self.popts.items():
            for param_name, param_value in zip(param_names, popt):
                params.append(
                    {
                        "fold": fold_idx,
                        "parameter": param_name,
                        "estimate": param_value,
                    }
                )

        return pd.DataFrame(params)

    @property
    def parameters_avg(self):
        """Get average parameters across folds."""
        df = self.parameters_by_fold
        return df.groupby("parameter")["estimate"].mean().to_dict()

    @property
    def performance_metrics(self):
        """Get performance metrics for each fold."""
        metrics = []
        for fold_idx, (mse, r2) in enumerate(zip(self.mse_scores, self.r2_scores)):
            metrics.append(
                {
                    "fold": fold_idx,
                    "mse": mse,
                    "r2": r2,
                }
            )
        return pd.DataFrame(metrics)

    @property
    def mean_performance_metrics(self):
        """Get mean performance metrics across folds."""
        return {
            "mse_mean": np.mean(self.mse_scores),
            "mse_std": np.std(self.mse_scores),
            "r2_mean": np.mean(self.r2_scores),
            "r2_std": np.std(self.r2_scores),
        }

    @property
    def sample_level_results(self):
        """Get predictions for each sample in the dataset."""
        results = []

        for fold_idx, fold_result in enumerate(self.fold_results):
            test_index = fold_result["test_index"]
            popt = fold_result["popt"]

            X_test = self.X[test_index]
            y_test = self.y[test_index]

            y_pred = action_surface_response((X_test[:, 0], X_test[:, 1]), *popt)

            for i, idx in enumerate(test_index):
                results.append(
                    {
                        "sample_index": idx,
                        "fold": fold_idx,
                        "wavelength": X_test[i, 0],
                        "irradiance": X_test[i, 1],
                        "observed": y_test[i, 0],
                        "predicted": y_pred[i],
                    }
                )

        return pd.DataFrame(results)

def fit_action_surface(df, config=None):
    """
    Fit an action surface model to dose-response data.

    Args:
        df: DataFrame with columns 'wavelength', 'irradiance', 'peak_current'
        config: HillModelConfig object (optional, uses defaults if not provided)

    Returns:
        Dictionary containing:
            - model: Fitted HillModel object
            - parameters_avg: Average parameters across folds
            - performance_metrics: DataFrame of performance metrics
            - sample_level_results: DataFrame of predictions
            - mean_performance_metrics: Mean metrics across folds
    """
    if config is None:
        config = HillModelConfig()

    model = HillModel(config)
    model.fit(df)

    return {
        "model": model,
        "parameters_avg": model.parameters_avg,
        "parameters_by_fold": model.parameters_by_fold,
        "performance_metrics": model.performance_metrics,
        "sample_level_results": model.sample_level_results,
        "mean_performance_metrics": model.mean_performance_metrics,
    }
