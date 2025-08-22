import json
import os
from datetime import datetime

import arviz
import gpjax as gpx
import jax.numpy as jnp
import kohgpjax as kgx
import numpy as np
from jax import config

config.update("jax_enable_x64", True)  # Enable 64-bit precision for JAX


def load(file_name, data_dir):
    """Load the simulation and observation data from CSV files and prepare them for modeling.
    Args:
        file_name (str): Base name of the dataset (e.g., 'sin-a').
        data_dir (str): Directory where the data files are located.
    Returns:
        Tuple[kgx.KOHDataset, Dict[str, Tuple[float, float]], float]:
            - kohdataset: A KOHDataset containing the field and component datasets.
            - tminmax: A dictionary with the min and max values for each calibration parameter.
            - ycmean: The mean of the centered output data.
    """
    sim_file = os.path.join(data_dir, file_name, "simulation.json")
    obs_file = os.path.join(data_dir, file_name, "observation.json")

    try:
        with open(sim_file, "r") as f:
            sim_data = json.load(f)
        with open(obs_file, "r") as f:
            obs_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Could not find data files. {e}")
        return

    # --- Prepare field data ---
    xf = jnp.array(obs_data["x_obs"])
    yf = jnp.array(obs_data["observations"])

    # --- Prepare computer model data ---
    x_grid = np.array(sim_data["x_sim"])
    simulations = sim_data["simulations"]

    xc_list, tc_list, yc_list = [], [], []
    num_grid_points = x_grid.shape[0]

    for sim in simulations:
        xc_list.append(x_grid)
        theta_values = np.array(list(sim["theta"].values()))
        tc_list.append(np.tile(theta_values, (num_grid_points, 1)))
        yc_list.append(np.array(sim["output"]))

    xc = jnp.vstack(xc_list)
    tc = jnp.vstack(tc_list)
    yc = jnp.vstack(yc_list)

    # Normalize calibration parameters
    tmin = jnp.min(tc, axis=0)
    tmax = jnp.max(tc, axis=0)
    tc_normalized = (tc - tmin) / (tmax - tmin)

    # --- Center the outputs ---
    yc_mean = jnp.mean(yc, axis=0)
    yc_centered = yc - yc_mean
    yf_centered = yf - yc_mean

    # --- Create KOHDataset ---
    field_dataset = gpx.Dataset(X=xf, y=yf_centered)
    comp_dataset = gpx.Dataset(X=jnp.hstack([xc, tc_normalized]), y=yc_centered)
    koh_dataset = kgx.KOHDataset(field_dataset, comp_dataset)

    tminmax = {
        f"theta_{i}": (tmin[i], tmax[i]) for i in range(koh_dataset.num_calib_params)
    }  # Create a dictionary for each calibration parameter

    return koh_dataset, tminmax, yc_mean


def transform_chains(traces, model_parameters, prior_dict, tminmax):
    """Transforms the MCMC chains."""
    traces_transformed = {}
    for var, trace in traces.items():
        if var == "hamiltonian":
            continue
        index = next(
            i for i, p in enumerate(model_parameters.priors_flat) if p.name == var
        )
        traces_transformed[var] = model_parameters.priors_flat[index].forward(
            np.array(trace)
        )
        if var in prior_dict["thetas"].keys():
            trace_val = traces_transformed[var]
            tmin, tmax = tminmax[var]
            traces_transformed[var] = list(
                (jnp.array(trace_val) * (tmax - tmin)) + tmin
            )
    return traces_transformed


def thin_runs_by_div(data: np.ndarray, div: int, x_dim: int = 1) -> np.ndarray:
    """Thin the runs by a specified divisor.
    Args:
        data (np.ndarray): The data to be thinned.
        div (int): The divisor for thinning.
        x_dim (int): The number of control/regression variables.
    Returns:
        np.ndarray: The thinned data.
    """
    if div <= 1:
        return data

    unique_params = np.unique(data[:, (x_dim + 1) :], axis=0)

    thinned_data = np.empty((0, data.shape[1]), dtype=data.dtype)
    for params in unique_params:
        mask = np.all(data[:, (x_dim + 1) :] == params, axis=1)
        thinned_subset = data[mask][::div]
        thinned_data = np.vstack((thinned_data, thinned_subset))
    return thinned_data


def save_chains_to_netcdf(
    raw_traces,
    transformed_traces,
    file_name: str,
    n_warm_up_iter: int,
    n_main_iter: int,
    n_sim: int,
    ycmean: float,
    inference_library_name: str,
) -> None:
    """
    Save the MCMC traces to a NetCDF file in Arviz InferenceData format.

    Args:
        raw_traces: The raw MCMC traces.
        transformed_traces: The transformed MCMC traces.
        file_name: The base name for the output file.
        n_warm_up_iter: Number of warm-up iterations.
        n_main_iter: Number of main iterations.
        n_sim: Number of simulation output points.
        ycmean: The mean of the centered output data.
        inference_library_name: The name of the inference library used.
    """
    # Create the directory for the experiment if it doesn't exist
    output_dir = os.path.join("chains", file_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    inference_data = arviz.from_dict(
        posterior=transformed_traces,
        unconstrained_posterior=raw_traces,
    )
    inference_data.attrs["ycmean"] = str(ycmean)
    inference_data.attrs["inference_library"] = inference_library_name
    inference_data.attrs["created_at"] = datetime.now().isoformat()

    inference_data.to_netcdf(
        os.path.join(output_dir, f"W{n_warm_up_iter}-N{n_main_iter}-Nsim{n_sim}.nc")
    )
