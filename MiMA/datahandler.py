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

        # Determine the run dimension and parameter coordinates
        dims = list(ds.dims)
        # Assuming 'lat' is spatial
        run_dims = [
            d
            for d in dims
            if d != "lat"
            and d != variable_name
            and d != "phalf"
            and d != "pfull"
            and d != "time"
            and d != "lon"
        ]

        # Heuristic: if 'N' is present, it's the run dim. Else if 'rh' is present, it's the run dim.
        if "N" in dims:
            run_dim = "N"
        elif "rh" in dims:
            run_dim = "rh"
        elif "t" in dims:
            run_dim = "t"
        else:
            # Fallback: take the first non-lat dim
            run_dim = run_dims[0] if run_dims else "rh"

        da = ds[variable_name]

        # Ensure we have a dataset with the variable
        sim_data = xr.Dataset({variable_name: da})

        # If parameters are not in sim_data coords (e.g. if loaded from separate file?), but here we loaded from NC.
        # We assume rh/tau are in coords if they define the run.

        # Define stack dimensions: (run_dim, 'lat')
        # This ensures we get Size = N_runs * N_lat
        stack_dims = [run_dim, "lat"]
        # Verify these dims exist
        stack_dims = [d for d in stack_dims if d in sim_data.dims]

        sim_data_flat = sim_data.stack(sample=stack_dims)

    else:
        # --- Load simulation parameters ---
        sim_params_path = os.path.join(base_dir, experiment_config.data.sim_params_path)
        t_sim_df = pd.read_csv(sim_params_path, header=None)

        # --- Load data files ---
        sim_pattern = os.path.join(base_dir, experiment_config.data.sim_data_pattern)
        sim_files = sorted(glob.glob(sim_pattern))

        def select_vars(ds):
            return ds[[variable_name]]

        sim_data = xr.open_mfdataset(
            sim_files,
            combine="nested",
            concat_dim="t",
            preprocess=select_vars,
        )

        # Assign coords from t_sim_df
        # If t_sim_df has 1 col, assign to 't'.
        # If multiple, assign appropriately.
        # For legacy compatibility, assume 't' is the run index/dim.
        sim_data = sim_data.assign_coords(t=np.arange(len(t_sim_df)))

        # Add parameter values as coordinates
        for i in range(t_sim_df.shape[1]):
            # Name them param_0, param_1 etc or try to match config?
            # Let's assign them to the 't' dimension
            sim_data = sim_data.assign_coords({f"param_{i}": ("t", t_sim_df[i])})

        stack_dims = ["t", "lat"]  # standard for this branch
        sim_data_flat = sim_data.stack(sample=stack_dims)

    # --- Prepare field data ---
    obs_data = xr.open_dataset(obs_path)
    xf = jnp.array(obs_data["lat"]).reshape(-1, 1).astype(jnp.float64)
    yf = jnp.array(obs_data[variable_name]).reshape(-1, 1).astype(jnp.float64)

    # --- Build Xc matrix ---
    # We need to construct Xc = [lat, param1, param2, ...]

    # Extract lat
    lat = sim_data_flat["lat"].to_numpy().reshape(-1, 1).astype(jnp.float64)

    tminmax = {p.name: tuple(p.range) for p in experiment_config.parameters}
    t_normalized_list = []

    # Heuristic mapping for T21_land_2D and T21
    # T21 (1D): theta_0 -> rh
    # T21_land_2D (2D): theta_0 -> rh, theta_1 -> tau

    # We will try to find the coordinate in sim_data_flat that matches the parameter logic
    # If the parameter name is theta_0, we look for 'rh' or 'param_0'.
    # If theta_1, we look for 'tau' or 'param_1'.

    for i, param in enumerate(experiment_config.parameters):
        vals = None

        # 1. Direct name match
        if param.name in sim_data_flat.coords:
            vals = sim_data_flat[param.name].to_numpy()

        # 2. Heuristic for theta_0 -> rh
        elif param.name == "theta_0" and "rh" in sim_data_flat.coords:
            vals = sim_data_flat["rh"].to_numpy()

        # 3. Heuristic for theta_1 -> tau
        elif param.name == "theta_1" and "tau" in sim_data_flat.coords:
            vals = sim_data_flat["tau"].to_numpy()

        # 4. Fallback: param_i (from CSV loading branch)
        elif f"param_{i}" in sim_data_flat.coords:
            vals = sim_data_flat[f"param_{i}"].to_numpy()

        # 5. Fallback for T21 legacy: if 't' contains the parameter value (e.g. rh implies t)
        elif (
            "t" in sim_data_flat.coords
            and i == 0
            and len(experiment_config.parameters) == 1
        ):
            vals = sim_data_flat["t"].to_numpy()

        if vals is None:
            raise ValueError(
                f"Could not locate simulation values for parameter {param.name}"
            )

        vals = vals.reshape(-1, 1)

        # Normalize
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

    # --- Apply Filters if defined in config ---
    if getattr(experiment_config, "filters", None):
        koh_dataset = filter_dataset(
            koh_dataset, experiment_config.filters, experiment_config, tminmax
        )

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


def filter_dataset(
    kohdataset: kgx.KOHDataset,
    filters: dict,
    experiment_config: ExperimentConfig,
    tminmax: dict,
) -> kgx.KOHDataset:
    """
    Filter the KOHDataset based on variable ranges.

    Args:
        kohdataset: The dataset to filter.
        filters: Dictionary defining filters. Keys can be 'lat' or parameter names.
                 Values are tuples (min, max). Use None for no bound.
                 Example: {'lat': [-70, 70], 'theta_0': [0.35, None]}
        experiment_config: Experiment configuration.
        tminmax: Dictionary of parameter ranges (min, max) for denormalization.

    Returns:
        Filtered KOHDataset.
    """
    import numpy as np

    print(f"\n[Filter Dataset] Applying filters: {filters}")
    print(f"Original Xc: {kohdataset.Xc.shape}, y: {kohdataset.y.shape}")
    print(f"Original Xf: {kohdataset.Xf.shape}, z: {kohdataset.z.shape}")

    # --- Filter Simulation Data (Xc, y) ---
    # Start with all true
    keep_sim = np.ones(kohdataset.Xc.shape[0], dtype=bool)

    # 1. Filter by Latitude (Column 0 of Xc)
    lat_sim = kohdataset.Xc[:, 0]
    if "lat" in filters:
        fmin, fmax = filters["lat"]
        if fmin is not None:
            keep_sim &= lat_sim >= fmin
        if fmax is not None:
            keep_sim &= lat_sim <= fmax

    # 2. Filter by Parameters (Columns 1..N of Xc)
    # Map parameter names to column indices
    # Xc columns: [Lat, Param1_Norm, Param2_Norm, ...]
    # Params are in order of experiment_config.parameters
    for i, param in enumerate(experiment_config.parameters):
        if param.name in filters:
            # Column index is i + 1 (since 0 is Lat)
            col_idx = i + 1
            norm_vals = kohdataset.Xc[:, col_idx]

            # Denormalize
            if param.name in tminmax:
                pmin, pmax = tminmax[param.name]
                phys_vals = norm_vals * (pmax - pmin) + pmin
            else:
                # Should strict fail or warn? tminmax should have it.
                phys_vals = norm_vals  # fallback

            fmin, fmax_val = filters[param.name]
            if fmin is not None:
                keep_sim &= phys_vals >= fmin
            if fmax_val is not None:
                keep_sim &= phys_vals <= fmax_val

    # Apply simulation mask
    Xc_filtered = kohdataset.Xc[keep_sim]
    y_filtered = kohdataset.y[keep_sim]

    # --- Filter Field Data (Xf, z) ---
    keep_field = np.ones(kohdataset.Xf.shape[0], dtype=bool)

    # 1. Filter by Latitude (Column 0 of Xf)
    lat_field = kohdataset.Xf[:, 0]
    if "lat" in filters:
        fmin, fmax = filters["lat"]
        if fmin is not None:
            keep_field &= lat_field >= fmin
        if fmax is not None:
            keep_field &= lat_field <= fmax

    # Apply field mask
    Xf_filtered = kohdataset.Xf[keep_field]
    z_filtered = kohdataset.z[keep_field]

    # Create new KOHDataset
    field_dataset_filtered = gpx.Dataset(X=Xf_filtered, y=z_filtered)
    comp_dataset_filtered = gpx.Dataset(X=Xc_filtered, y=y_filtered)
    new_dataset = kgx.KOHDataset(field_dataset_filtered, comp_dataset_filtered)

    print(f"Filtered Xc: {new_dataset.Xc.shape}, y: {new_dataset.y.shape}")
    print(f"Filtered Xf: {new_dataset.Xf.shape}, z: {new_dataset.z.shape}\n")

    return new_dataset
