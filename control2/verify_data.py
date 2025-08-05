import json
import jax.numpy as jnp
import gpjax as gpx
import kohgpjax as kgx
import numpy as np

def load_and_verify():
    """Loads the newly structured data and verifies it by creating a KOHDataset."""
    # Load data from JSON files
    with open("control2/data/simulation_data.json", "r") as f:
        sim_data = json.load(f)
    with open("control2/data/observation_data.json", "r") as f:
        obs_data = json.load(f)

    # --- Prepare field data ---
    xf = jnp.array(obs_data["x_obs"])
    # NOTE: KOHDataset currently only supports 1D output. We select the first dimension for verification.
    yf = jnp.array(obs_data["outputs"])[:, 0:1]

    # --- Prepare computer model data ---
    x_grid = np.array(sim_data["x_grid"])
    simulations = sim_data["simulations"]

    # Reconstruct the xc, tc, and yc arrays from the new structure
    num_sim_runs = len(simulations)
    num_grid_points = x_grid.shape[0]

    xc_list = []
    tc_list = []
    yc_list = []

    for sim in simulations:
        # Append the grid for each simulation run
        xc_list.append(x_grid)

        # Extract theta values and repeat for each grid point
        theta_values = np.array(list(sim["theta"].values()))
        tc_list.append(np.tile(theta_values, (num_grid_points, 1)))

        # Append the outputs
        yc_list.append(np.array(sim["output"]))

    xc = jnp.vstack(xc_list)
    tc = jnp.vstack(tc_list)
    yc = jnp.vstack(yc_list)

    # Normalize calibration parameters
    tmin = jnp.min(tc, axis=0)
    tmax = jnp.max(tc, axis=0)
    tc_normalized = (tc - tmin) / (tmax - tmin)

    # --- Center the outputs ---
    # NOTE: KOHDataset currently only supports 1D output. We select the first dimension for verification.
    yc = yc[:, 0:1]
    yc_mean = jnp.mean(yc, axis=0)
    yc_centered = yc - yc_mean
    yf_centered = yf - yc_mean

    # --- Create KOHDataset ---
    try:
        field_dataset = gpx.Dataset(X=xf, y=yf_centered)
        comp_dataset = gpx.Dataset(X=jnp.hstack([xc, tc_normalized]), y=yc_centered)

        koh_dataset = kgx.KOHDataset(field_dataset, comp_dataset)

        print("Successfully created kohgpjax.KOHDataset object from the new data structure.")
        print(f"Field dataset size: {field_dataset.n}")
        print(f"Computer model dataset size: {comp_dataset.n}")
        print(f"Output dimension: {field_dataset.y.shape[1]}")

    except Exception as e:
        print(f"An error occurred during KOHDataset creation: {e}")

if __name__ == "__main__":
    load_and_verify()
