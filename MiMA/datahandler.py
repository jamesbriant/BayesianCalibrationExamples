import glob
import os
from datetime import datetime
from typing import Tuple

import arviz
import gpjax as gpx
import jax.numpy as jnp
import kohgpjax as kgx
import numpy as np
import pandas as pd
import xarray as xr
from jax import config

try:
    from config_schema import ExperimentConfig
except ImportError:
    from .config_schema import ExperimentConfig

config.update("jax_enable_x64", True)  # Enable 64-bit precision for JAX


def load(
    experiment_config: ExperimentConfig, data_root: str
) -> Tuple[kgx.KOHDataset, dict, float]:
    """Load the simulation and observation data from CSV files and prepare them for modeling.
    Args:
        experiment_config (ExperimentConfig): Configuration object.
        data_root (str): Root directory where the data files are located.
    Returns:
        Tuple[kgx.KOHDataset, Dict[str, Tuple[float, float]], float]:
            - kohdataset: A KOHDataset containing the field and component datasets.
            - tminmax: A dictionary with the min and max values for each calibration parameter.
            - ycmean: The mean of the centered output data.
    """
    # Construct paths using config
    base_dir = os.path.join(data_root, experiment_config.data.experiment_name)

    obs_path = os.path.join(base_dir, experiment_config.data.obs_data_path)

    variable_name = experiment_config.data.variable_name

    # Branch: single NetCDF file containing all runs
    if getattr(experiment_config.data, "sim_nc_path", None):
        sim_nc_path = os.path.join(base_dir, experiment_config.data.sim_nc_path)
        ds = xr.open_dataset(sim_nc_path, decode_times=False)
        # Expect variable with coords (rh, lat); rename to (t, lat) to match prior code
        da = ds[variable_name]
        da = da.rename({"rh": "t"})
        # Build a dataset with expected dims
        sim_data = xr.Dataset({variable_name: da})
        # Create t_sim_df-like structure from the 't' coordinate for later assignment
        t_values = sim_data["t"].to_numpy()
        t_sim_df = pd.DataFrame({0: t_values})
    else:
        # --- Load simulation parameters ---
        sim_params_path = os.path.join(base_dir, experiment_config.data.sim_params_path)
        # Assuming the parameters file has columns corresponding to parameters
        # If header=None, we assume columns 0, 1, ... match parameters 0, 1, ...
        t_sim_df = pd.read_csv(sim_params_path, header=None)

        # --- Load data files ---
        sim_pattern = os.path.join(base_dir, experiment_config.data.sim_data_pattern)
        sim_files = sorted(glob.glob(sim_pattern))

        # --- Define the preprocessing function ---
        def select_vars(ds):
            """Selects only the desired variable(s) from a dataset."""
            return ds[[variable_name]]

        # --- Load datasets ---
        sim_data = xr.open_mfdataset(
            sim_files,
            combine="nested",
            concat_dim="t",
            preprocess=select_vars,
        )

    # Assign coordinates.
    # If we have 1 parameter, we can assign it as 't'.
    # If we have multiple, 't' might be just an index or one of them.
    # The original code assigned t=t_sim.
    # Let's assume for now we assign the first parameter as 't' coordinate for xarray alignment,
    # but we will use the full parameter set for Xc.

    # If we have multiple parameters, we might need a dummy index or use the first one.
    # For backward compatibility with T21 which has 1 param, we use column 0.
    if t_sim_df.shape[1] == 1:
        sim_data = sim_data.assign_coords(t=t_sim_df[0])
    else:
        # If multiple parameters, we might need a dummy index or use the first one.
        # For backward compatibility with T21 which has 1 param, we use column 0.
        sim_data = sim_data.assign_coords(t=t_sim_df[0])

    obs_data = xr.open_dataset(obs_path)

    # --- Prepare field data ---
    # Assuming 'lat' is the spatial coordinate.
    xf = jnp.array(obs_data["lat"]).reshape(-1, 1).astype(jnp.float64)
    yf = jnp.array(obs_data[variable_name]).reshape(-1, 1).astype(jnp.float64)

    # --- Build Xc matrix robustly for any number of calibration parameters ---
    # Identify all relevant coordinates: lat + calibration parameters
    coords = sim_data.coords
    lat_vals = coords["lat"].to_numpy()

    # Map config parameter names to actual NetCDF coordinate names
    # Assume: first calibration parameter in config matches 't', second matches 'tau', etc.
    # Find all coordinates except 'lat' and variable_name
    all_coord_names = list(coords)
    non_lat_coords = [c for c in all_coord_names if c != "lat" and c != variable_name]
    # Sort to ensure deterministic order (t, tau, etc.)
    non_lat_coords_sorted = sorted(non_lat_coords)
    # If 't' is present, always put it first (for legacy)
    if "t" in non_lat_coords_sorted:
        non_lat_coords_sorted.remove("t")
        non_lat_coords_sorted = ["t"] + non_lat_coords_sorted

    # Map config param order to coord order
    param_coord_names = non_lat_coords_sorted[: len(experiment_config.parameters)]

    # Stack only over actual dimensions (not coordinates)
    # Exclude variable_name from stack dims
    stack_dims = [d for d in sim_data.dims if d != variable_name]
    sim_data_flat = sim_data.stack(sample=stack_dims)

    # Extract coordinate values for each stacked sample
    lat = sim_data_flat["lat"].to_numpy().reshape(-1, 1).astype(jnp.float64)
    tminmax = {p.name: tuple(p.range) for p in experiment_config.parameters}
    t_normalized_list = []
    for i, param in enumerate(experiment_config.parameters):
        coord_name = param_coord_names[i]
        # If the calibration parameter is a coordinate, extract it; otherwise, try to get it from the stacked coordinates
        if coord_name in sim_data_flat.coords:
            vals = sim_data_flat[coord_name].to_numpy().reshape(-1, 1)
        else:
            # Try to extract from the original sim_data (e.g., if it's a scalar or broadcasted)
            vals = np.full_like(lat, sim_data[coord_name].item())
        pmin, pmax = param.range
        p_norm = (vals - pmin) / (pmax - pmin)
        t_normalized_list.append(p_norm)

    # Compose Xc: [lat, norm_param1, norm_param2, ...]
    Xc = jnp.hstack([lat] + t_normalized_list)

    # Extract simulation outputs to match Xc
    yc = sim_data_flat[variable_name].to_numpy().reshape(-1, 1).astype(jnp.float64)

    # --- Center the outputs ---
    yc_mean = jnp.mean(yc, axis=0)
    yc_centered = yc - yc_mean  # Center the outputs
    yf_centered = yf - yc_mean  # Center the outputs

    # --- Create KOHDataset ---
    field_dataset = gpx.Dataset(X=xf, y=yf_centered)
    comp_dataset = gpx.Dataset(X=Xc, y=yc_centered)
    koh_dataset = kgx.KOHDataset(field_dataset, comp_dataset)

    return koh_dataset, tminmax, yc_mean


def transform_chains(traces, model_parameters, prior_dict, tminmax):
    """Transforms the MCMC chains."""
    traces_transformed = {}
    for var, trace in traces.items():
        if var == "hamiltonian":
            continue

        # Find the prior corresponding to this variable
        # model_parameters.priors_flat is a list of ParameterPrior objects
        # We need to find the one with name == var
        prior = next((p for p in model_parameters.priors_flat if p.name == var), None)

        if prior:
            # Transform from unconstrained to constrained space (e.g. log to linear)
            traces_transformed[var] = prior.forward(np.array(trace))
        else:
            # If not found in priors (shouldn't happen for sampled vars), keep as is
            traces_transformed[var] = np.array(trace)

        # If it's a calibration parameter (theta), un-normalize it
        # The prior_dict structure is typically {'thetas': {'theta_0': ...}, ...}
        if "thetas" in prior_dict and var in prior_dict["thetas"]:
            trace_val = traces_transformed[var]
            if var in tminmax:
                tmin, tmax = tminmax[var]
                # Scale from [0, 1] back to [tmin, tmax]
                traces_transformed[var] = (np.array(trace_val) * (tmax - tmin)) + tmin

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
    output_dir: str = None,
) -> None:
    """
    Save the MCMC traces to a NetCDF file in Arviz InferenceData format.

    Args:
        raw_traces: The raw MCMC traces.
        transformed_traces: The transformed MCMC traces.
        file_name: The base name for the output file (used if output_dir is None).
        n_warm_up_iter: Number of warm-up iterations.
        n_main_iter: Number of main iterations.
        n_sim: Number of simulation output points.
        ycmean: The mean of the centered output data.
        inference_library_name: The name of the inference library used.
        output_dir: The directory to save the output file.
    """
    # Create the directory for the experiment if it doesn't exist
    if output_dir is None:
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

    # Use a fixed name "posterior.nc" if saving to a specific run directory,
    # otherwise keep the descriptive name for backward compatibility or if output_dir is generic.
    # But to be safe and consistent with the new plan, let's use the descriptive name inside the run folder.
    # Or better: just use "posterior.nc" inside the run folder to make it easy to find.

    # Let's stick to the descriptive name for now to avoid confusion, but place it in the run folder.
    filename = f"W{n_warm_up_iter}-N{n_main_iter}-Nsim{n_sim}.nc"
    inference_data.to_netcdf(os.path.join(output_dir, filename))
    print(f"Saved chains to {os.path.join(output_dir, filename)}")
