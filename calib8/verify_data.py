import argparse
import json
import os

import gpjax as gpx
import jax.numpy as jnp
import kohgpjax as kgx
import numpy as np
from jax import config

config.update("jax_enable_x64", True)  # Enable 64-bit precision for JAX


def load_and_verify(file_name, data_dir):
    """Loads a specific dataset and verifies it by creating a KOHDataset."""
    sim_file = os.path.join(data_dir, f"{file_name}_simulation.json")
    obs_file = os.path.join(data_dir, f"{file_name}_observation.json")

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

    print("yc_mean.shape:", yc_mean.shape)
    print("yc.shape:", yc.shape)
    print("yf.shape:", yf.shape)

    yc_centered = yc - yc_mean
    yf_centered = yf - yc_mean

    # --- Create KOHDataset ---
    try:
        field_dataset = gpx.Dataset(X=xf, y=yf_centered)
        comp_dataset = gpx.Dataset(X=jnp.hstack([xc, tc_normalized]), y=yc_centered)
        koh_dataset = kgx.KOHDataset(field_dataset, comp_dataset)

        print(f"Successfully verified dataset '{file_name}'.")
        print(f"Field dataset size: {field_dataset.n}")
        print(f"Computer model dataset size: {comp_dataset.n}\n")

        print(koh_dataset)

    except Exception as e:
        print(f"An error occurred during KOHDataset creation for '{file_name}': {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify a generated dataset.")
    parser.add_argument(
        "--file-name",
        type=str,
        required=True,
        help="Base name of the dataset to verify (e.g., sin-a)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory where the data is stored.",
    )
    args = parser.parse_args()

    load_and_verify(args.file_name, args.data_dir)
