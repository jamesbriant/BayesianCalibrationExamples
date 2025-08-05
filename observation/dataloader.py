# ONLY WORKS FOR 1D X VARIABLES
from jax import config

config.update("jax_enable_x64", True)  # Enable 64-bit precision for JAX

from typing import Dict, Tuple

import numpy as np
import jax.numpy as jnp
import kohgpjax as kgx
import gpjax as gpx
import arviz
import os


def load(
    sim_file_path_csv: str, obs_file_path_csv: str, num_calib_params: int
) -> Tuple[kgx.KOHDataset, Dict[str, Tuple[float, float]], float]:
    DATAFIELD = np.loadtxt(obs_file_path_csv, delimiter=",", dtype=np.float32)
    DATACOMP = np.loadtxt(sim_file_path_csv, delimiter=",", dtype=np.float32)

    yf = jnp.reshape(DATAFIELD[:, 0], (-1, 1)).astype(jnp.float64)
    yc = jnp.reshape(DATACOMP[:, 0], (-1, 1)).astype(jnp.float64)
    xf = jnp.reshape(DATAFIELD[:, 1], (-1, 1)).astype(jnp.float64)
    xc = jnp.reshape(DATACOMP[:, 1], (-1, 1)).astype(jnp.float64)
    tc = jnp.reshape(DATACOMP[:, 2:], (-1, num_calib_params)).astype(jnp.float64)

    # normalising the output is not required provided they are all of a similar scale.
    # But subtracting the mean is sensible as our GP priors assume zero mean.
    ycmean = jnp.mean(yc)
    yc_centered = yc - ycmean  # Centre so that E[yc] = 0
    yf_centered = yf - ycmean

    # normalising the inputs is not required provided they are all of a similar scale.

    tmin = jnp.min(tc, axis=0)
    tmax = jnp.max(tc, axis=0)
    # print(f"tmin: {tmin}, tmax: {tmax}")
    tc_normalized = (tc - tmin) / (tmax - tmin)  # Normalize to [0, 1]

    tminmax = {
        f"theta_{i}": (tmin[i], tmax[i]) for i in range(num_calib_params)
    }  # Create a dictionary for each calibration parameter

    field_dataset = gpx.Dataset(xf, yf_centered)
    comp_dataset = gpx.Dataset(jnp.hstack((xc, tc_normalized)), yc_centered)

    kohdataset = kgx.KOHDataset(field_dataset, comp_dataset)

    return kohdataset, tminmax, ycmean


def thin_runs_by_div(data: np.ndarray, div: int) -> np.ndarray:
    """Thin the runs by a specified divisor.
    Args:
        data (np.ndarray): The data to be thinned.
        div (int): The divisor for thinning.
    Returns:
        np.ndarray: The thinned data.
    """
    if div <= 1:
        return data

    unique_params = np.unique(data[:, 2:], axis=0)

    thinned_data = np.empty((0, data.shape[1]), dtype=data.dtype)
    for params in unique_params:
        mask = np.all(data[:, 2:] == params, axis=1)
        thinned_subset = data[mask][::div]
        thinned_data = np.vstack((thinned_data, thinned_subset))
    return thinned_data


def save_chains_to_netcdf(
    traces,
    traces_transformed,
    file_name: str,
    n_warm_up_iter: int,
    n_main_iter: int,
    n_sim: int,
) -> None:
    """
    Save the MCMC traces to NetCDF files.

    Args:
        traces: The raw MCMC traces
        traces_transformed: The transformed MCMC traces
        file_name: The base name for the output files
        n_warm_up_iter: Number of warm-up iterations
        n_main_iter: Number of main iterations
        n_sim: Number of simulation output points
    """
    # check if chains directory exists, if not create it
    if not os.path.exists("chains"):
        os.makedirs("chains")

    arviz.convert_to_inference_data(traces).to_netcdf(
        f"chains/{file_name}-W{n_warm_up_iter}-N{n_main_iter}-Nsim{n_sim}-raw.nc"
    )
    arviz.convert_to_inference_data(traces_transformed).to_netcdf(
        f"chains/{file_name}-W{n_warm_up_iter}-N{n_main_iter}-Nsim{n_sim}.nc"
    )
