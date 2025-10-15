"""
Action Surface: Input-modulated dose-response function fitting for optogenetics
"""

from .model import (
    HillModel,
    HillModelConfig,
    ActionSurfaceParameters,
    action_surface_response,
    find_threshold_irradiance,
    fit_action_surface,
)
from .synthetic import generate_synthetic_data
from .visualization import (
    WAVELENGTH_COLORS,
    ScatterContourPlotter,
    plot_contour,
    plot_dose_response_curves,
    plot_scatter,
    plot_single_dose_response_curve,
    visualize_action_surface,
)

__version__ = "0.1.0"

__all__ = [
    "HillModelConfig",
    "ActionSurfaceParameters",
    "action_surface_response",
    "find_threshold_irradiance",
    "fit_action_surface",
    "HillModel",
    "generate_synthetic_data",
    "visualize_action_surface",
    "plot_dose_response_curves",
    "plot_scatter",
    "plot_contour",
    "plot_single_dose_response_curve",
    "ScatterContourPlotter",
    "WAVELENGTH_COLORS",
]
