"""
Visualization utilities for action surface models.

This module provides plotting functions matching the style of the existing codebase,
including the ScatterContourPlotter for publication-quality action surface plots.
"""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter, LogLocator

from .model import find_threshold_irradiance, action_surface_response
from .synthetic import generate_action_surface_grid

# Standard wavelength color mapping
WAVELENGTH_COLORS = {
    390: [0.22, 0.0, 0.251, 1.0],
    440: [0.0, 0.0, 0.694, 1.0],
    475: [0.0, 0.469, 0.867, 1.0],
    510: [0.0, 0.655, 0.702, 1.0],
    555: [0.0, 0.624, 0.0, 1.0],
    575: [0.747, 0.862, 0.0, 1.0],
    637: [0.8, 0.298, 0.298, 1.0],
    748: [0.6, 0.6, 0.6, 1.0],
}


class ScatterContourPlotter:
    """
    Plots action surface data with contours and threshold lines.

    Reproduces the style from proteng.ambient_light_manuscript.plot.ScatterContourPlotter
    """

    def __init__(self):
        """Initialize plotter with default configuration."""
        self.config = {
            "y_axis_var": "irradiance",
            "dep_var": "peak_current",
            "reduction": "median",
            "reduce_conditions": True,
            "color_scale_min": 50,
            "color_scale_max": 4000,
            "cbar_label": None,
            "s": 40,
            "levels": 20,
            "min_log10_irradiance": -3.2,
            "max_log10_irradiance": 0.55,
            "min_wavelength": 400,
            "max_wavelength": 760,
            "vmin": 1e1,
            "vmax": 2e3,
            "title_fontsize": 8,
            "clabel_fontsize": 10,
            "remove_baseline_from_scatter": True,
            # Contour configuration
            "fill_contours": True,
            "line_contours": True,
            "fill_cmap": "Purples",
            "line_color": 'white',
            "line_cmap": "Purples",
            "line_alpha": 0.3,
            "line_linewidth": 0.8,
            "scatter_cmap": "Purples",
            # Contour levels
            "contour_levels": [0, 100] + list(np.arange(200, 10000, 100)),
            # Threshold configuration
            "threshold_linewidth": 1.2,
            "threshold_markersize": 12,
            "threshold_lines": [
                {
                    "current": 200,
                    "show_line": True,
                    "show_marker": False,
                    "line_color": "orange",
                    "marker_color": "orange",
                    "line_style": "-",
                    "label": "T200",
                },
            ],
            "show_K_threshold": False,
            "K_line_color": "green",
            "K_marker_color": "green",
            "K_line_style": "--",
            "K_show_line": True,
            "K_show_marker": True,
            "K_label": "K (half-max)",
            # Y-axis configuration
            "y_axis_major_ticks": 5,
        }
        self._setup_style()

    def _setup_style(self):
        """Configure plot styling."""
        sns.set_style("white")
        sns.set_context("paper")
        plt.rcParams.update(
            {
                "xtick.bottom": True,
                "ytick.left": True,
                "xtick.major.size": 3.5,
                "ytick.major.size": 3.5,
            }
        )

    def update_config(self, **kwargs):
        """Update configuration parameters."""
        self.config.update(kwargs)

    def set_threshold_lines(self, threshold_configs):
        """Set threshold line configurations."""
        self.config["threshold_lines"] = threshold_configs

    def add_threshold_line(
        self,
        current,
        line_color="black",
        marker_color=None,
        line_style="-",
        show_line=True,
        show_marker=False,
        label=None,
    ):
        """Add a single threshold line to the configuration."""
        if marker_color is None:
            marker_color = line_color

        threshold_config = {
            "current": current,
            "show_line": show_line,
            "show_marker": show_marker,
            "line_color": line_color,
            "marker_color": marker_color,
            "line_style": line_style,
            "label": label or f"T{current}",
        }
        self.config["threshold_lines"].append(threshold_config)

    def clear_threshold_lines(self):
        """Clear all threshold lines."""
        self.config["threshold_lines"] = []

    def _generate_prediction_grid(self, params):
        """Generate meshgrid and predictions for contour plot."""
        log10_irr_range = np.linspace(self.config["min_log10_irradiance"], self.config["max_log10_irradiance"], 300)
        irr_range = 10**log10_irr_range
        wave_range = np.linspace(self.config["min_wavelength"], self.config["max_wavelength"], 300)
        wavelengths, irradiances = np.meshgrid(wave_range, irr_range)

        y_preds = action_surface_response(
            data=(wavelengths, irradiances),
            K=params["K"],
            n=params["n"],
            log10_amp_max=params["log10_amp_max"],
            lambda_max=params["lambda_max"],
            sigma=params["sigma"],
            baseline=params["baseline"],
        )
        y_preds -= params["baseline"]
        y_preds_reshaped = y_preds.reshape(len(irr_range), len(wave_range))

        return wave_range, irr_range, y_preds_reshaped

    def _plot_contours(self, ax, wave_range, irr_range, y_preds):
        """Add contour plot to axes."""
        levels = self.config["contour_levels"]
        
        # Filter out zero or negative values to avoid log10 warnings
        levels = np.array([l for l in levels if l > 0])
        
        if len(levels) == 0:
            return None
        
        # Clip predictions to avoid log10 of zero or negative values
        y_preds_clipped = np.clip(y_preds, 1e-10, None)
        log_y_preds = np.log10(y_preds_clipped)

        if self.config["fill_contours"]:
            cp_fill = ax.contourf(
                wave_range,
                irr_range,
                log_y_preds,
                levels=np.log10(levels),
                cmap=self.config["fill_cmap"],
                vmin=np.log10(self.config["color_scale_min"]),
                vmax=np.log10(self.config["color_scale_max"]),
            )

        if self.config["line_contours"]:
            if self.config["line_color"]:
                cp_line = ax.contour(
                    wave_range,
                    irr_range,
                    log_y_preds,
                    levels=np.log10(levels),
                    colors=self.config["line_color"],
                    linewidths=self.config["line_linewidth"],
                    alpha=self.config["line_alpha"],
                )
            else:
                cp_line = ax.contour(
                    wave_range,
                    irr_range,
                    log_y_preds,
                    levels=np.log10(levels),
                    cmap=self.config["line_cmap"],
                    linewidths=self.config["line_linewidth"],
                    vmin=np.log10(self.config["color_scale_min"]),
                    vmax=np.log10(self.config["color_scale_max"]),
                    alpha=self.config["line_alpha"],
                )

        return None

    def _plot_scatter(self, ax, single_exp, params):
        """Add scatter plot of experimental data."""
        values = single_exp[self.config["dep_var"]].copy()
        if self.config["remove_baseline_from_scatter"]:
            values -= params["baseline"]

        # Clip to be above 0
        values = np.clip(values, 10, None)

        # Log transform
        values = np.log10(values)

        scatter = ax.scatter(
            single_exp["wavelength"],
            single_exp[self.config["y_axis_var"]],
            s=self.config["s"],
            c=values,
            cmap=self.config["scatter_cmap"],
            alpha=1,
            vmin=np.log10(self.config["color_scale_min"]),
            vmax=np.log10(self.config["color_scale_max"]),
            edgecolors="k",
            zorder=100,
            linewidths=1.5,
        )
        ax._scatter_plot = scatter
        return scatter

    def _plot_single_threshold_line(
        self,
        ax,
        wave_range,
        threshold_current,
        line_color,
        marker_color,
        params,
        line_style="-",
        show_line=True,
        show_marker=True,
        label=None,
    ):
        """Plot a single threshold line and optional marker at lambda_max."""
        if not show_line and not show_marker:
            return

        try:
            threshold_irradiances = find_threshold_irradiance(
                wavelength=wave_range,
                threshold_current=threshold_current,
                K=params["K"],
                n=params["n"],
                log10_amp_max=params["log10_amp_max"],
                lambda_max=params["lambda_max"],
                sigma=params["sigma"],
                baseline=0,
            )

            if show_line:
                valid_idx = (threshold_irradiances > 10 ** self.config["min_log10_irradiance"]) & (
                    threshold_irradiances < 10 ** self.config["max_log10_irradiance"]
                )
                ax.plot(
                    wave_range[valid_idx],
                    threshold_irradiances[valid_idx],
                    linestyle=line_style,
                    color=line_color,
                    linewidth=self.config["threshold_linewidth"],
                    label=label,
                )

            if show_marker:
                threshold_at_lmax = find_threshold_irradiance(
                    wavelength=params["lambda_max"],
                    threshold_current=threshold_current,
                    K=params["K"],
                    n=params["n"],
                    log10_amp_max=params["log10_amp_max"],
                    lambda_max=params["lambda_max"],
                    sigma=params["sigma"],
                    baseline=0,
                )
                ax.plot(
                    params["lambda_max"],
                    threshold_at_lmax,
                    marker="*",
                    markerfacecolor=marker_color,
                    markeredgecolor="k",
                    markeredgewidth=1.5,
                    markersize=self.config["threshold_markersize"],
                )
        except Exception as e:
            print(f"Error plotting threshold line: {e}")

    def _plot_all_threshold_lines(self, ax, wave_range, params):
        """Plot all configured threshold lines."""
        for threshold_config in self.config["threshold_lines"]:
            self._plot_single_threshold_line(
                ax,
                wave_range,
                threshold_config["current"],
                threshold_config["line_color"],
                threshold_config["marker_color"],
                params,
                line_style=threshold_config.get("line_style", "-"),
                show_line=threshold_config.get("show_line", True),
                show_marker=threshold_config.get("show_marker", False),
                label=threshold_config.get("label", None),
            )

        if self.config.get("show_K_threshold", False):
            half_max_current = params.get("amp_max", 10 ** params["log10_amp_max"]) / 2
            self._plot_single_threshold_line(
                ax,
                wave_range,
                half_max_current,
                self.config["K_line_color"],
                self.config["K_marker_color"],
                params,
                line_style=self.config.get("K_line_style", "--"),
                show_line=self.config.get("K_show_line", True),
                show_marker=self.config.get("K_show_marker", True),
                label=self.config.get("K_label", "K (half-max)"),
            )

    def _configure_axes(self, ax, show_ylabel=True, show_xlabel=True):
        """Configure axis properties."""
        ax.set_yscale("log")
        ax.set_ylim(
            10 ** self.config["min_log10_irradiance"],
            10 ** self.config["max_log10_irradiance"] * 1.10,
        )

        tick_values = np.logspace(-3, 0, num=4)
        ax.set_yticks(tick_values)

        ax.yaxis.set_major_locator(LogLocator(numticks=self.config["y_axis_major_ticks"]))
        ax.yaxis.set_minor_locator(LogLocator(numticks=self.config["y_axis_major_ticks"], subs=list(np.arange(2, 10))))
        ax.tick_params(axis="y", which="both", direction="out", length=4)
        ax.tick_params(axis="y", which="minor", labelbottom=False)
        ax.tick_params(axis="y", which="major", length=4, width=1)
        ax.tick_params(axis="y", which="minor", length=2, width=0.5)

        # Add gridlines
        ax.grid(True, which="major", axis="y", alpha=0.8, linestyle="-", linewidth=0.7, zorder=0)
        ax.grid(True, which="minor", axis="y", alpha=0.6, linestyle=":", linewidth=0.5, zorder=0)
        ax.grid(True, which="major", axis="x", alpha=0.8, linestyle="-", linewidth=0.7, zorder=0)

        # Ensure ticks and spines are on top
        ax.set_axisbelow(True)

        if show_xlabel:
            ax.set_xlabel("Wavelength (nm)")
        if show_ylabel:
            ax.set_ylabel("Irradiance\n(mW·mm$^{-2}$)")

        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(2.0)

        ax.tick_params(axis="both", colors="black", width=1.5)
        ax.xaxis.label.set_color("black")
        ax.yaxis.label.set_color("black")

        sns.despine()

        # Re-enable ticks after despine
        ax.tick_params(axis="y", which="major", left=True, labelleft=True, length=4, width=1.5)
        ax.tick_params(axis="y", which="minor", left=True, length=2, width=0.75)

    def plot(
        self,
        params,
        single_exp,
        ax=None,
        figsize=(3, 3),
        show_ylabel=True,
        show_xlabel=True,
        title=None,
        show_legend=False,
    ):
        """
        Create the complete plot.

        Parameters
        ----------
        params : dict
            Dictionary containing model parameters
        single_exp : pandas.DataFrame
            Experimental data
        ax : matplotlib.axes.Axes, optional
            Axes to plot on
        figsize : tuple, optional
            Figure size if creating new figure
        show_ylabel : bool, optional
            Whether to show y-axis label
        show_xlabel : bool, optional
            Whether to show x-axis label
        title : str, optional
            Title for the subplot
        show_legend : bool, optional
            Whether to show legend for threshold lines

        Returns
        -------
        fig, ax : matplotlib figure and axes objects
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        # Aggregate data if needed
        if single_exp is not None and not single_exp.empty:
            if self.config["reduce_conditions"]:
                if self.config["reduction"] == "median":
                    single_exp = (
                        single_exp.groupby(["wavelength", self.config["y_axis_var"]])[self.config["dep_var"]]
                        .median()
                        .reset_index()
                    )
                elif self.config["reduction"] == "mean":
                    single_exp = (
                        single_exp.groupby(["wavelength", self.config["y_axis_var"]])[self.config["dep_var"]]
                        .mean()
                        .reset_index()
                    )

        # Generate and plot contours
        wave_range, irr_range, y_preds = self._generate_prediction_grid(params)
        self._plot_contours(ax, wave_range, irr_range, y_preds)

        # Add scatter plot only if data is available
        if single_exp is not None and not single_exp.empty:
            self._plot_scatter(ax, single_exp, params)

        # Plot all threshold lines
        self._plot_all_threshold_lines(ax, wave_range, params)

        # Configure axes
        self._configure_axes(ax, show_ylabel=show_ylabel, show_xlabel=show_xlabel)

        if title:
            ax.set_title(title, fontsize=self.config["title_fontsize"])

        if show_legend and any(t.get("label") for t in self.config["threshold_lines"]):
            ax.legend(loc="upper right", fontsize=8)

        return fig, ax

    def plot_multiple(
        self,
        data_list,
        figsize=(12, 4),
        sharex=True,
        sharey=True,
        titles=None,
        fig_title=None,
        colorbar_position="last",
        colorbar_ax_width=0.02,
        colorbar_label=None,
        show_legend=False,
    ):
        """
        Create multiple plots with shared axes.

        Parameters
        ----------
        data_list : list of tuples
            Each tuple should contain (params, single_exp) for a subplot
        figsize : tuple
            Overall figure size
        sharex : bool
            Whether to share x-axes
        sharey : bool
            Whether to share y-axes
        titles : list of str, optional
            Titles for each subplot
        fig_title : str, optional
            Overall figure title
        colorbar_position : str or int, optional
            Where to place colorbar: 'last', 'first', 'none', or subplot index
        colorbar_ax_width : float
            Width of colorbar axes as fraction of figure width
        colorbar_label : str, optional
            Label for the colorbar
        show_legend : bool, optional
            Whether to show legend on plots

        Returns
        -------
        fig, axes : matplotlib figure and axes array
        """
        n_plots = len(data_list)
        show_colorbar = colorbar_position != "none"

        fig = plt.figure(figsize=figsize)

        if show_colorbar:
            width_ratios = [1.0] * n_plots
            width_ratios.append(colorbar_ax_width * 3)

            gs = gridspec.GridSpec(1, len(width_ratios), width_ratios=width_ratios, wspace=0.3)

            axes = []
            for i in range(n_plots):
                if i == 0:
                    ax = fig.add_subplot(gs[i])
                else:
                    ax = fig.add_subplot(
                        gs[i],
                        sharex=axes[0] if sharex else None,
                        sharey=axes[0] if sharey else None,
                    )
                axes.append(ax)

            cbar_ax = fig.add_subplot(gs[-1])
        else:
            axes = []
            for i in range(n_plots):
                ax = plt.subplot(
                    1,
                    n_plots,
                    i + 1,
                    sharex=axes[0] if (sharex and i > 0) else None,
                    sharey=axes[0] if (sharey and i > 0) else None,
                )
                axes.append(ax)
            cbar_ax = None

        colorbar_idx = n_plots - 1 if colorbar_position == "last" else 0
        if isinstance(colorbar_position, int):
            colorbar_idx = colorbar_position

        for i, (params, single_exp) in enumerate(data_list):
            show_ylabel = i == 0
            show_xlabel = True
            show_subplot_legend = show_legend and (i == n_plots - 1)

            title = titles[i] if titles else None

            self.plot(
                params,
                single_exp,
                ax=axes[i],
                show_ylabel=show_ylabel,
                show_xlabel=show_xlabel,
                title=title,
                show_legend=show_subplot_legend,
            )

        if show_colorbar and cbar_ax is not None:
            scatter_plot = getattr(axes[colorbar_idx], "_scatter_plot", None)

            if scatter_plot is not None:
                cbar = fig.colorbar(scatter_plot, cax=cbar_ax)

                if colorbar_label:
                    cbar.set_label(colorbar_label)
                else:
                    cbar.set_label(f'{self.config["dep_var"].replace("_", " ").title()} (pA)')

                def format_func(value, tick_number):
                    linear_val = 10**value
                    if linear_val >= 1000:
                        return f"{linear_val:.0f}"
                    elif linear_val >= 100:
                        return f"{linear_val:.0f}"
                    elif linear_val >= 10:
                        return f"{linear_val:.1f}"
                    else:
                        return f"{linear_val:.2f}"

                cbar.ax.yaxis.set_major_formatter(FuncFormatter(format_func))

                nice_values = [50, 100, 200, 500, 1000, 2000, 4000]
                tick_positions = [
                    np.log10(v)
                    for v in nice_values
                    if self.config["color_scale_min"] <= v <= self.config["color_scale_max"]
                ]
                cbar.set_ticks(tick_positions)

        for ax in axes:
            for spine in ax.spines.values():
                spine.set_color("black")
                spine.set_linewidth(2.0)

            ax.tick_params(axis="both", colors="black", width=1.5)
            if ax.get_xlabel():
                ax.xaxis.label.set_color("black")
            if ax.get_ylabel():
                ax.yaxis.label.set_color("black")

        if show_colorbar:
            plt.subplots_adjust(wspace=0.3)
        else:
            plt.tight_layout()

        if fig_title:
            fig.suptitle(fig_title, y=1.02)

        return fig, axes


def plot_single_dose_response_curve(
    wavelength,
    data,
    model_params=None,
    ax=None,
    color=None,
    show_fit=True,
    show_data=True,
    normalize=True,
    subtract_baseline=True,
    label_midpoint=True,
    label_wavelength=True,
    dep_var="peak_current",
    xmin=-3.3,
    xmax=1,
    reduction="median",
    threshold_current=None,
):
    """
    Plot a single dose-response curve for a specific wavelength.

    Closely matches the style from doseresp.plot.plot_single_curve

    Args:
        wavelength: Wavelength to plot (nm)
        data: DataFrame with wavelength, irradiance, peak_current columns
        model_params: Fitted model parameters (dict)
        ax: Matplotlib axes (creates new if None)
        color: Color for plot (uses wavelength color map if None)
        show_fit: Whether to show fitted curve
        show_data: Whether to show data points
        normalize: Whether to normalize response
        subtract_baseline: Whether to subtract baseline
        label_midpoint: Whether to label K value or threshold
        label_wavelength: Whether to label wavelength
        dep_var: Dependent variable column name
        xmin: Minimum log10 irradiance
        xmax: Maximum log10 irradiance
        reduction: How to aggregate data points ('median', 'mean', or None for all points)
        threshold_current: Optional threshold current value to mark instead of K

    Returns:
        matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    # Get color from wavelength map if not specified
    if color is None:
        color = WAVELENGTH_COLORS.get(wavelength, "black")

    # Filter data for this wavelength
    wavelength_data = data[data["wavelength"] == wavelength].copy()

    if model_params is not None and subtract_baseline:
        wavelength_data[dep_var] = wavelength_data[dep_var] - model_params.get("baseline", 0)
    if model_params is not None and normalize:
        amp_max = 10 ** model_params.get("log10_amp_max", 3.0)
        wavelength_data[dep_var] = wavelength_data[dep_var] / amp_max

    # Aggregate data if requested
    if show_data and not wavelength_data.empty:
        if reduction is None:
            scatter_data = wavelength_data
        elif reduction == "median":
            scatter_data = wavelength_data.groupby("irradiance")[dep_var].median().reset_index()
        elif reduction == "mean":
            scatter_data = wavelength_data.groupby("irradiance")[dep_var].mean().reset_index()
        else:
            raise ValueError("reduction must be 'mean', 'median', or None")

        # Plot data points
        sns.scatterplot(
            data=scatter_data,
            x="irradiance",
            y=dep_var,
            ax=ax,
            legend=False,
            color=color,
            alpha=1,
        )

    # Plot fitted curve if parameters provided
    if show_fit and model_params is not None:
        log10_irr_range = np.linspace(xmin, xmax, 200)
        irr_range = 10**log10_irr_range
        wavelengths_arr = np.ones_like(irr_range) * wavelength

        y_pred = action_surface_response(
            (wavelengths_arr, irr_range),
            model_params.get("K", 0.5),
            model_params.get("n", 1.0),
            model_params.get("log10_amp_max", 3.0),
            model_params.get("lambda_max", 510),
            model_params.get("sigma", 40),
            model_params.get("baseline", 0),
        )

        if subtract_baseline:
            y_pred -= model_params.get("baseline", 0)
        if normalize:
            amp_max = 10 ** model_params.get("log10_amp_max", 3.0)
            y_pred = y_pred / amp_max

        ax.plot(
            irr_range,
            y_pred,
            color=color,
            linewidth=1.5,
        )

        # Calculate and mark threshold or K value
        if label_midpoint:
            if threshold_current is not None:
                # Mark threshold current
                try:
                    threshold_irr = find_threshold_irradiance(
                        wavelength=wavelength,
                        threshold_current=threshold_current,
                        K=model_params["K"],
                        n=model_params["n"],
                        log10_amp_max=model_params["log10_amp_max"],
                        lambda_max=model_params["lambda_max"],
                        sigma=model_params["sigma"],
                        baseline=model_params.get("baseline", 0),
                    )

                    if threshold_irr is not None and 10**xmin < threshold_irr < 10**xmax:
                        # Calculate y value at threshold
                        y_at_threshold = action_surface_response(
                            ([wavelength], [threshold_irr]),
                            model_params["K"],
                            model_params["n"],
                            model_params["log10_amp_max"],
                            model_params["lambda_max"],
                            model_params["sigma"],
                            model_params.get("baseline", 0),
                        )[0]

                        if subtract_baseline:
                            y_at_threshold -= model_params.get("baseline", 0)
                        if normalize:
                            y_at_threshold = y_at_threshold / (10 ** model_params["log10_amp_max"])

                        # Draw vertical line to threshold irradiance
                        ax.vlines(threshold_irr, 0, y_at_threshold, linestyle="--", color="black", alpha=0.7)
                        # Draw horizontal line to threshold current (no label)
                        ax.hlines(y_at_threshold, 10**xmin, threshold_irr, linestyle="--", color="black", alpha=0.7)
                except Exception:
                    pass
            elif "K" in model_params:
                # Mark K value (original behavior)
                K = model_params["K"]
                lambda_max = model_params["lambda_max"]
                sigma = model_params["sigma"]

                # Calculate wavelength-specific K
                scaling = np.exp(-((wavelength - lambda_max) ** 2) / (2 * (sigma**2 + 1e-8)))
                scaled_K = K / scaling

                # Only show if within plot range
                if 10**xmin < scaled_K < 10**xmax:
                    # Calculate y value at K
                    y_at_K = action_surface_response(
                        ([wavelength], [scaled_K]),
                        K,
                        model_params["n"],
                        model_params["log10_amp_max"],
                        lambda_max,
                        sigma,
                        model_params.get("baseline", 0),
                    )[0]

                    if subtract_baseline:
                        y_at_K -= model_params.get("baseline", 0)
                    if normalize:
                        y_at_K = y_at_K / amp_max

                    # Draw vertical line and label
                    ax.vlines(scaled_K, 0, y_at_K, linestyle="--", color="black", alpha=0.7)
                    ax.text(scaled_K * 0.07, y_at_K, f"$K$ = {scaled_K:.3f}", fontsize=8, ha="center", va="bottom")

    # Add wavelength label if requested
    if label_wavelength:
        ax.text(
            0.0,
            0.965,
            f"$\\bf{{{int(wavelength)}}}$nm",
            fontsize=8,
            ha="left",
            va="top",
            transform=ax.transAxes,
            color=color,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.5),
        )

    ax.set_xscale("log")
    tick_values = np.logspace(-3, 1, num=5)
    ax.set_xticks(tick_values)
    ax.set_xlabel("Irradiance (mW/mm²)")

    if normalize:
        ax.set_ylabel("Normalized peak current")
    else:
        ax.set_ylabel("Peak current (pA)")

    ax.grid(True, alpha=0.3)

    return ax


def plot_dose_response_curves(
    data,
    model_params=None,
    wavelengths=None,
    figsize=(10, 6),
    normalize=True,
    subtract_baseline=True,
    reduction="median",
    threshold_current=None,
):
    """
    Plot dose-response curves for specific wavelengths.

    Args:
        data: DataFrame with columns wavelength, irradiance, peak_current
        model_params: Fitted model parameters (optional)
        wavelengths: List of wavelengths to plot (optional, uses all if None)
        figsize: Figure size
        normalize: Whether to normalize responses
        subtract_baseline: Whether to subtract baseline
        reduction: How to aggregate ('median', 'mean', or None)
        threshold_current: Optional threshold current to mark instead of K

    Returns:
        matplotlib Figure object
    """
    if wavelengths is None:
        wavelengths = sorted(data["wavelength"].unique())

    n_wavelengths = len(wavelengths)
    n_cols = min(4, n_wavelengths)
    n_rows = int(np.ceil(n_wavelengths / n_cols))

    sns.set(style="whitegrid", context="paper")
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, sharey=True)

    # Handle single subplot case
    if n_wavelengths == 1:
        axs = np.array([axs])
    else:
        axs = axs.flatten() if n_rows > 1 or n_cols > 1 else np.array([axs])

    for i, wavelength in enumerate(wavelengths):
        if i < len(axs):
            plot_single_dose_response_curve(
                wavelength,
                data,
                model_params=model_params,
                ax=axs[i],
                normalize=normalize,
                subtract_baseline=subtract_baseline,
                label_wavelength=True,
                label_midpoint=True,
                reduction=reduction,
                threshold_current=threshold_current,
            )

            # Only show y-label on leftmost column
            if i % n_cols != 0:
                axs[i].set_ylabel("")

    # Remove empty subplots
    for i in range(n_wavelengths, len(axs)):
        axs[i].set_visible(False)

    plt.tight_layout()
    return fig


def plot_contour(
    model_params,
    data=None,
    figsize=(8, 6),
    levels=20,
    wavelength_range=(400, 750),
    irradiance_range=(-3, 1),
    mark_lmax_and_k=True,
    threshold_current=None,
    title=None,
    vmin=None,
    vmax=None,
    cmap="viridis",
):
    """
    Create a contour plot of the action surface.

    Matches the style from doseresp.plot.plot_contours

    Args:
        model_params: Dictionary of fitted model parameters
        data: Optional experimental data to overlay as scatter points
        figsize: Figure size
        levels: Number of contour levels
        wavelength_range: (min, max) wavelength in nm
        irradiance_range: (min, max) log10 irradiance
        mark_lmax_and_k: Whether to mark λ_max and K with an X
        threshold_current: If provided, mark threshold at λ_max
        title: Plot title
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        cmap: Colormap name for contour plot (default: 'viridis')

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Generate prediction grid
    log10_irr_range = np.linspace(irradiance_range[0], irradiance_range[1], 50)
    irr_range = 10**log10_irr_range
    wave_range = np.linspace(wavelength_range[0], wavelength_range[1], 100)
    wavelengths, irradiances = np.meshgrid(wave_range, irr_range)

    y_preds = action_surface_response(
        (wavelengths, irradiances),
        model_params.get("K", 0.5),
        model_params.get("n", 1.0),
        model_params.get("log10_amp_max", 3.0),
        model_params.get("lambda_max", 510),
        model_params.get("sigma", 40),
        model_params.get("baseline", 0),
    )
    y_preds -= model_params.get("baseline", 0)
    y_preds_reshaped = y_preds.reshape(len(irr_range), len(wave_range))

    # Plot contours
    sns.set(style="whitegrid", context="paper")
    ax.set_yscale("log")
    tick_values = np.logspace(irradiance_range[0], irradiance_range[1], num=5)
    ax.set_yticks(tick_values)

    cp = ax.contour(
        wave_range,
        irr_range,
        y_preds_reshaped,
        levels=levels,
        cmap=cmap,
        linewidths=1.5,
        vmin=vmin,
        vmax=vmax,
        norm="log",
    )

    # Label select contours
    def fmt(x):
        return f"{x:3.0f}"

    level_idxs = [1, -3, -2]
    valid_idxs = [i for i in level_idxs if abs(i) < len(cp.levels)]
    if valid_idxs:
        ax.clabel(cp, [cp.levels[i] for i in valid_idxs], inline=True, fmt=fmt, fontsize=10)

    # Overlay data if provided
    if data is not None:
        ax.scatter(
            data["wavelength"],
            data["irradiance"],
            c="red",
            s=20,
            alpha=0.5,
            edgecolors="white",
            linewidths=0.5,
            zorder=100,
        )

    # Mark λ_max and K
    if mark_lmax_and_k:
        lambda_max = model_params.get("lambda_max", 510)
        K = model_params.get("K", 0.5)
        ax.plot(
            lambda_max,
            K,
            marker="x",
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=2.0,
            markersize=12,
        )

    # Mark threshold if requested
    if threshold_current is not None:
        lambda_max = model_params.get("lambda_max", 510)
        threshold_irr = find_threshold_irradiance(lambda_max, threshold_current, **model_params)
        if threshold_irr is not None:
            ax.plot(
                lambda_max,
                threshold_irr,
                marker="x",
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=2.0,
                markersize=12,
            )

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Irradiance (mW/mm²)")

    if title:
        ax.set_title(title)

    plt.tight_layout()
    return fig


def plot_scatter(
    data,
    model_params=None,
    figsize=(8, 6),
    cmap="rocket",
    vmin=None,
    vmax=None,
    s=50,
    reduction="median",
):
    """
    Create a scatter plot of peak current vs wavelength and irradiance.

    Args:
        data: DataFrame with columns wavelength, irradiance, peak_current
        model_params: Fitted model parameters (optional, for overlay)
        figsize: Figure size
        cmap: Colormap for peak current
        vmin: Minimum value for color scale
        vmax: Maximum value for color scale
        s: Point size
        reduction: How to aggregate data ('median', 'mean', or None)

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Aggregate data if requested
    if reduction == "median":
        plot_data = data.groupby(["wavelength", "irradiance"])["peak_current"].median().reset_index()
    elif reduction == "mean":
        plot_data = data.groupby(["wavelength", "irradiance"])["peak_current"].mean().reset_index()
    else:
        plot_data = data

    # Create scatter plot
    scatter = ax.scatter(
        plot_data["wavelength"],
        plot_data["irradiance"],
        c=plot_data["peak_current"],
        s=s,
        cmap=cmap,
        alpha=0.7,
        vmin=vmin,
        vmax=vmax,
        edgecolors="k",
        linewidths=0.5,
    )

    ax.set_yscale("log")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Irradiance (mW/mm²)")
    ax.set_title("Action Surface Data")

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Peak Current (pA)")

    # Overlay model prediction contours if parameters provided
    if model_params is not None:
        wave_range = np.linspace(400, 750, 100)
        log10_irr_range = np.linspace(-4, 1, 100)
        irr_range = 10**log10_irr_range

        W, I = np.meshgrid(wave_range, irr_range)

        responses = action_surface_response(
            (W, I),
            model_params.get("K", 0.5),
            model_params.get("n", 1.0),
            model_params.get("log10_amp_max", 3.0),
            model_params.get("lambda_max", 510),
            model_params.get("sigma", 40),
            model_params.get("baseline", 0),
        )

        ax.contour(W, I, responses, levels=5, colors="white", alpha=0.5, linewidths=1)

    plt.tight_layout()
    return fig


def visualize_action_surface(data=None, model_params=None, figsize=(12, 5)):
    """
    Visualize an action surface as a heatmap and dose-response curves.

    Args:
        data: DataFrame with experimental data (optional)
        model_params: Dictionary of model parameters (optional)
        figsize: Figure size tuple

    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=figsize)

    # Default parameters if not provided
    if model_params is None:
        model_params = {
            "lambda_max": 510,
            "sigma": 40,
            "K": 0.5,
            "n": 1.0,
            "log10_amp_max": 3.0,
            "baseline": 0,
        }

    # Generate action surface grid
    wavelengths, log10_irradiances, responses = generate_action_surface_grid(
        lambda_max=model_params.get("lambda_max", 510),
        sigma=model_params.get("sigma", 40),
        K=model_params.get("K", 0.5),
        n=model_params.get("n", 1.0),
        amp_max=10 ** model_params.get("log10_amp_max", 3.0),
        baseline=model_params.get("baseline", 0),
    )

    # Create subplots
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)

    # Plot heatmap
    im = ax1.imshow(
        responses,
        aspect="auto",
        origin="lower",
        extent=[wavelengths[0], wavelengths[-1], log10_irradiances[0], log10_irradiances[-1]],
        cmap="viridis",
    )
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("Log₁₀ Irradiance (mW/mm²)")
    ax1.set_title("Action Surface")
    plt.colorbar(im, ax=ax1, label="Response (pA)")

    # Plot dose-response curves for different wavelengths
    test_wavelengths = [440, 475, 510, 555, 575, 637, 748]

    for wavelength in test_wavelengths:
        if wavelength in WAVELENGTH_COLORS:
            color = WAVELENGTH_COLORS[wavelength]
            irradiances = 10**log10_irradiances
            responses_at_wavelength = []

            for irr in irradiances:
                resp = action_surface_response(
                    ([wavelength], [irr]),
                    model_params.get("K", 0.5),
                    model_params.get("n", 1.0),
                    model_params.get("log10_amp_max", 3.0),
                    model_params.get("lambda_max", 510),
                    model_params.get("sigma", 40),
                    model_params.get("baseline", 0),
                )[0]
                responses_at_wavelength.append(resp)

            ax2.plot(
                log10_irradiances,
                responses_at_wavelength,
                label=f"{wavelength} nm",
                color=color,
                linewidth=2,
            )

    # Overlay experimental data if provided
    if data is not None:
        for wavelength in data["wavelength"].unique():
            subset = data[data["wavelength"] == wavelength]
            ax2.scatter(
                np.log10(subset["irradiance"]),
                subset["peak_current"],
                alpha=0.3,
                s=20,
            )

    ax2.set_xlabel("Log₁₀ Irradiance (mW/mm²)")
    ax2.set_ylabel("Response (pA)")
    ax2.set_title("Dose-Response Curves")
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
