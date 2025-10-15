"""
Synthetic data generation for action surface models.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from .model import action_surface_response


def generate_synthetic_data(
    wavelengths=None,
    irradiances=None,
    lambda_max=510,
    sigma=40,
    K=0.5,
    n=1.0,
    amp_max=1000,
    baseline=0,
    noise_level=0.1,
    n_replicates=3,
    seed=None,
):
    """
    Generate synthetic action surface data for testing.

    Args:
        wavelengths: List of wavelengths to sample (nm)
        irradiances: Array of irradiances to sample (mW/mm²)
        lambda_max: Peak wavelength sensitivity (nm)
        sigma: Spectral width (nm)
        K: Half-saturation constant (mW/mm²)
        n: Hill coefficient
        amp_max: Maximum response amplitude (pA)
        baseline: Baseline response (pA)
        noise_level: Fractional noise level (0.1 = 10% noise)
        n_replicates: Number of replicates per condition
        seed: Random seed for reproducibility

    Returns:
        DataFrame with columns: wavelength, irradiance, peak_current, replicate
    """
    if wavelengths is None:
        wavelengths = [440, 475, 510, 555, 575, 637, 748]

    if irradiances is None:
        # Log-spaced from 0.0001 to 10 mW/mm²
        irradiances = np.logspace(-4, 1, 15)

    if seed is not None:
        np.random.seed(seed)

    data = []
    log10_amp_max = np.log10(amp_max)

    for wavelength in wavelengths:
        for irradiance in irradiances:
            for replicate in range(n_replicates):
                # Calculate true response
                true_response = action_surface_response(
                    ([wavelength], [irradiance]), K, n, log10_amp_max, lambda_max, sigma, baseline
                )[0]

                # Add noise
                noise = np.random.normal(0, noise_level * true_response)
                observed_response = true_response + noise

                # Ensure non-negative responses
                observed_response = max(0, observed_response)

                data.append(
                    {
                        "wavelength": wavelength,
                        "irradiance": irradiance,
                        "peak_current": observed_response,
                        "replicate": replicate,
                        "true_response": true_response,
                    }
                )

    return pd.DataFrame(data)


def generate_action_surface_grid(
    lambda_max=510,
    sigma=40,
    K=0.5,
    n=1.0,
    amp_max=1000,
    baseline=0,
    wavelength_range=(400, 750),
    irradiance_range=(-4, 1),
    n_wavelengths=50,
    n_irradiances=50,
):
    """
    Generate a dense grid of action surface values for visualization.

    Args:
        lambda_max: Peak wavelength sensitivity (nm)
        sigma: Spectral width (nm)
        K: Half-saturation constant (mW/mm²)
        n: Hill coefficient
        amp_max: Maximum response amplitude (pA)
        baseline: Baseline response (pA)
        wavelength_range: (min, max) wavelength in nm
        irradiance_range: (min, max) log10 irradiance
        n_wavelengths: Number of wavelength points
        n_irradiances: Number of irradiance points

    Returns:
        Tuple of (wavelengths, log10_irradiances, response_grid)
    """
    wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], n_wavelengths)
    log10_irradiances = np.linspace(irradiance_range[0], irradiance_range[1], n_irradiances)

    # Create meshgrid
    W, I = np.meshgrid(wavelengths, 10**log10_irradiances)

    # Calculate responses
    log10_amp_max = np.log10(amp_max)
    responses = action_surface_response(
        (W.flatten(), I.flatten()), K, n, log10_amp_max, lambda_max, sigma, baseline
    ).reshape(W.shape)

    return wavelengths, log10_irradiances, responses
