from typing import Callable

import arviz
import cola
import cola.ops
from gpjax.distributions import GaussianDistribution
import jax.numpy as jnp
from kohgpjax.parameters import ModelParameterPriorDict
import matplotlib.pyplot as plt
import numpy as np

# import numpyro.distributions as dist


plot_style = {
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "axes.linewidth": 0.5,
    "lines.linewidth": 0.5,
    "axes.labelpad": 2.0,
    "figure.dpi": 150,
}


def plot_sim_sample(
    xf: np.ndarray,
    yf: np.ndarray,
    xc: np.ndarray,
    yc: np.ndarray,
    tc: np.ndarray,
    num_samples: int = 5,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a sample of the simulation data.
    Args:
        xf: x-coordinates of the observations.
        yf: y-coordinates of the observations.
        xc: x-coordinates of the simulation data.
        yc: y-coordinates of the simulation data.
        tc: parameter values for the simulation data.
        num_samples: number of samples to plot.
    Returns:
        fig: matplotlib Figure object containing the plot.
    """
    fig, ax = plt.subplots(1, 1)

    ax.scatter(xf, yf, label="Observations")

    rng = np.random.default_rng()
    ts = rng.permutation(np.unique(tc, axis=0))[:num_samples]
    for t in ts:
        rows = np.all(tc == t, axis=1)
        label = [f"{ti:.2f}" for ti in t]
        ax.plot(xc[rows], yc[rows], "--", label=f"$t$=({', '.join(label)})")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()

    return fig, ax


def plot_pairwise_samples(
    traces,
    var_names: list[str],
    figsize=(6, 6),
    plot_style=plot_style,
) -> plt.Axes:
    with plt.style.context(plot_style):
        axes = arviz.plot_pair(
            traces,
            var_names=var_names,
            figsize=figsize,
        )
        axes[0, 0].figure.tight_layout()

    return axes


def plot_posterior_chains_with_priors(
    traces,
    model_parameters: ModelParameterPriorDict = None,
    tracer_index_dict: dict[str, int] = None,
    true_values: dict[str, float] = None,
    figsize=(9, 14),
    plot_style=plot_style,
    **kwargs,
):
    if true_values is not None:
        lines = [(name, {}, value) for name, value in true_values.items()]
    else:
        lines = []
    # Plot the prior, posterior, true values (if known) and chains
    with plt.style.context(plot_style):
        axes = arviz.plot_trace(
            traces,
            figsize=figsize,
            legend=True,
            compact=False,
            lines=lines,
            **kwargs,
        )

    if model_parameters is not None and tracer_index_dict is not None:
        for i in range(axes.shape[0]):
            left, right = axes[i, 0].get_xlim()
            left, right = left * 0.9, right * 1.1
            x = np.linspace(left, right, 1000)

            title = axes[i, 0].get_title()
            prior_dist = model_parameters.priors_flat[
                tracer_index_dict[title]
            ].distribution
            pdf = jnp.exp(prior_dist.log_prob(x))

            axes[i, 0].plot(x, pdf, color="red", linestyle="--", label="Prior")
            axes[i, 0].legend()

    return axes


def plot_GP(
    ax: plt.Axes,
    x: np.ndarray,
    distribution: GaussianDistribution,
    label: str = "GP",
    alpha: float = 0.5,
    **kwargs,
) -> plt.Axes:
    """
    Plot the GP distribution on the given axes.

    Args:
        ax: The matplotlib Axes to plot on.
        x: The input values for the GP.
        distribution: The GaussianDistribution object containing the GP mean and covariance.

    Returns:
        The updated Axes with the GP plot.
    """
    mean = distribution.mean
    cov = distribution.variance
    sd = jnp.sqrt(cov)

    ax.plot(x, mean, label=label, **kwargs)
    ax.fill_between(
        x,
        mean - 1.96 * sd,
        mean + 1.96 * sd,
        alpha=alpha,
        **kwargs,
    )

    return ax


def plot_f_eta(
    x_full: np.ndarray,
    x_GP: np.ndarray,
    thetas: np.ndarray,
    thetas_full: np.ndarray,
    eta: Callable,
    GP_eta: GaussianDistribution,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(nrows=1, ncols=1)

    x_plot = x_full[:, 0]
    x_calc = x_full[:, [0, 1]]
    label = [f"{theta:.2f}" for theta in thetas]

    ax.plot(x_plot, eta(x_calc, thetas_full), label=rf"$\eta({', '.join(label)})$")
    plot_GP(
        ax,
        x_GP,
        GP_eta,
        label=r"$f_\eta(x)$",
        color="orange",
    )

    ax.legend()
    fig.suptitle("Computer simulation and GP model of simulator")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    return fig, ax


def plot_f_zeta(
    x_full: np.ndarray,
    x_GP: np.ndarray,
    zeta: Callable,
    GP_zeta: GaussianDistribution,
    GP_zeta_epsilon: GaussianDistribution,
    scatter_xf: np.ndarray,
    scatter_yf: np.ndarray,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(nrows=1, ncols=1)

    x_plot = x_full[:, 0]
    x_calc = x_full[:, [0, 1]]

    ax.plot(x_plot, zeta(x_calc), label=r"$\zeta(x)$ True process")

    ax = plot_GP(
        ax,
        x_GP,
        GP_zeta,
        label=r"$f_\zeta(x)$ GP reconstruction",
        color="orange",
    )
    # ax.plot(x_test_GP[:, 0], obs_pred_m, label=r"$f_{\zeta+\epsilon}(x)$")
    # ax.fill_between(
    #     x_test_GP[:, 0],
    #     obs_pred_m - 1.96 * obs_pred_sd,
    #     obs_pred_m + 1.96 * obs_pred_sd,
    #     alpha=0.3,
    #     color="orange",
    # )

    ax = plot_GP(
        ax,
        x_GP,
        GP_zeta_epsilon,
        label=r"$f_{\zeta+\epsilon}(x)$ GP reconstruction",
        color="green",
    )
    # ax.plot(x_test_GP[:, 0], zeta_pred_m, label=r"$f_{\zeta}(x)$")
    # ax.fill_between(
    #     x_test_GP[:, 0],
    #     zeta_pred_m - 1.96 * zeta_pred_sd,
    #     zeta_pred_m + 1.96 * zeta_pred_sd,
    #     alpha=0.3,
    #     color="green",
    # )

    ax.scatter(scatter_xf, scatter_yf, label="Observations")

    ax.legend()
    fig.suptitle("True process, observations and GP reconstruction of $\zeta$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return fig, ax


def plot_f_delta(
    x_full: np.ndarray,
    x_GP: np.ndarray,
    thetas: np.ndarray,
    thetas_full: np.ndarray,
    eta: Callable,
    zeta: Callable,
    delta: Callable,
    delta_gp_mean: np.ndarray,
    delta_gp_cov: np.ndarray,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(nrows=1, ncols=1)

    x_plot = x_full[:, 0]
    x_calc = x_full[:, [0, 1]]
    label = [f"{theta:.2f}" for theta in thetas]

    ax.plot(
        x_plot,
        zeta(x_calc) - eta(x_calc, thetas_full),
        label=rf"$\delta(x) = \zeta(x) - \eta(x, {', '.join(label)})$",
    )

    plot_GP(
        ax,
        x_GP,
        GaussianDistribution(
            loc=delta_gp_mean,
            scale=cola.ops.Dense(delta_gp_cov),
        ),
        label=r"$f_\delta(x)$",
        color="orange",
    )

    ax.plot(
        x_plot, delta(x_calc), color="black", linestyle="--", label="True discrepancy"
    )

    ax.legend()
    fig.suptitle("GP model of discrepancy function $\delta(x)$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    return fig, ax
