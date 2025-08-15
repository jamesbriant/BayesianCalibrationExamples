import argparse
import json
import os
import sys

import numpy as np
from scipy.stats.qmc import LatinHypercube
from ..utils import load_config_from_path


def generate_simulation_data(config):
    """Generates and saves the simulation data based on the provided config."""
    x_ranges = [p["range"] for p in config.CONTROL_PARAMETERS]
    x_grids = [np.linspace(r[0], r[1], config.N_SIMULATION_POINTS) for r in x_ranges]
    x_mesh = np.meshgrid(*x_grids)
    x_grid = np.vstack([m.flatten() for m in x_mesh]).T

    t_ranges = np.array(
        [p["range"] for p in config.PARAMETERS[: config.N_CALIB_PARAMS]]
    )
    sampler = LatinHypercube(d=config.N_CALIB_PARAMS)
    t_samples_scaled = sampler.random(n=config.N_SIMULATION_RUNS)
    t_samples = t_samples_scaled * (t_ranges[:, 1] - t_ranges[:, 0]) + t_ranges[:, 0]

    simulations_list = []
    param_names = [p["name"] for p in config.PARAMETERS[: config.N_CALIB_PARAMS]]
    for t_sample in t_samples:
        t_sample_repeated = np.tile(t_sample, (x_grid.shape[0], 1))
        output = config.eta(x_grid, t_sample_repeated)

        sim_run = {
            "theta": dict(zip(param_names, t_sample.tolist())),
            "output": output.tolist(),
        }
        simulations_list.append(sim_run)

    data = {
        "x_sim": x_grid.tolist(),
        "simulations": simulations_list,
    }

    return data


def generate_observation_data(config):
    """Generates and saves the multi-dimensional observation data."""
    rng = np.random.default_rng(42)
    x_ranges = [p["range"] for p in config.CONTROL_PARAMETERS]
    x_dim = len(x_ranges)

    x_obs = rng.uniform(
        low=[r[0] for r in x_ranges],
        high=[r[1] for r in x_ranges],
        size=(config.N_OBSERVATION_POINTS, x_dim),
    )

    obs = config.zeta(x_obs)

    noise = rng.normal(
        0,
        config.OBS_NOISE_STD,
        (config.N_OBSERVATION_POINTS, len(config.OBS_NOISE_STD)),
    )
    obs += noise

    data = {
        "x_obs": x_obs.tolist(),
        "observations": obs.tolist(),
    }

    return data


def main(config_path: str, output_dir: str):
    """
    Main function to generate and save all data.

    Args:
        config_path (str): Path to the configuration module.
        output_dir (str): Directory to save the output files.
    """
    # Dynamically import the specified config module
    try:
        config = load_config_from_path(config_path)
    except ImportError:
        print(f"Error: Could not import configuration module from '{config_path}'.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Generate and save simulation data
    sim_data = generate_simulation_data(config)
    sim_file_path = os.path.join(output_dir, f"{config.FILE_NAME}_simulation.json")
    with open(sim_file_path, "w") as f:
        json.dump(sim_data, f, indent=4)
    print(f"Simulation data saved to {sim_file_path}")

    # Generate and save observation data
    obs_data = generate_observation_data(config)
    obs_file_path = os.path.join(output_dir, f"{config.FILE_NAME}_observation.json")
    with open(obs_file_path, "w") as f:
        json.dump(obs_data, f, indent=4)
    print(f"Observation data saved to {obs_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate simulation and observation data."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration module (e.g., configs/calib8.py)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Directory to save the output files.",
    )
    args = parser.parse_args()

    main(config_path=args.config, output_dir=args.output_dir)
