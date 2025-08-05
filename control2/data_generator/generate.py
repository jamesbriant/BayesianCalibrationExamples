import json
import os
import numpy as np
from scipy.stats.qmc import LatinHypercube
from . import config
from . import simulator

def generate_simulation_data():
    """Generates and saves the simulation data with the new structure."""
    # Generate control parameter grid
    x_ranges = [p['range'] for p in config.CONTROL_PARAMETERS]
    x_grids = [np.linspace(r[0], r[1], config.N_SIMULATION_POINTS) for r in x_ranges]
    x_mesh = np.meshgrid(*x_grids)
    x_grid = np.vstack([m.flatten() for m in x_mesh]).T

    # Generate calibration parameter samples
    t_ranges = np.array([p['range'] for p in config.PARAMETERS[:config.N_CALIB_PARAMS]])
    sampler = LatinHypercube(d=config.N_CALIB_PARAMS)
    t_samples_scaled = sampler.random(n=config.N_SIMULATION_RUNS)
    t_samples = t_samples_scaled * (t_ranges[:, 1] - t_ranges[:, 0]) + t_ranges[:, 0]

    # Run simulator and build simulation list
    simulations_list = []
    param_names = [p['name'] for p in config.PARAMETERS[:config.N_CALIB_PARAMS]]
    for t_sample in t_samples:
        t_sample_repeated = np.tile(t_sample, (x_grid.shape[0], 1))
        output = simulator.eta(x_grid, t_sample_repeated)

        sim_run = {
            "theta": dict(zip(param_names, t_sample.tolist())),
            "output": output.tolist()
        }
        simulations_list.append(sim_run)

    # Prepare final data structure for JSON
    data = {
        "x_grid": x_grid.tolist(),
        "simulations": simulations_list,
    }

    return data

def generate_observation_data():
    """Generates and saves the multi-dimensional observation data."""
    rng = np.random.default_rng(42)  # for reproducibility
    x_ranges = [p['range'] for p in config.CONTROL_PARAMETERS]
    x_dim = len(x_ranges)

    # Generate random observation points
    x_obs = rng.uniform(
        low=[r[0] for r in x_ranges],
        high=[r[1] for r in x_ranges],
        size=(config.N_OBSERVATION_POINTS, x_dim)
    )

    # Generate observations
    obs = simulator.zeta(x_obs)

    # Add noise to observations
    noise = rng.normal(0, config.OBS_NOISE_STD, (config.N_OBSERVATION_POINTS, len(config.OBS_NOISE_STD)))
    obs += noise

    # Prepare data for JSON
    data = {
        "x_obs": x_obs.tolist(),
        "outputs": obs.tolist(),
    }

    return data

def main():
    """Main function to generate and save all data."""
    output_dir = "control2/data"
    os.makedirs(output_dir, exist_ok=True)

    sim_data = generate_simulation_data()
    with open(os.path.join(output_dir, "simulation_data.json"), "w") as f:
        json.dump(sim_data, f, indent=4)
    print("Simulation data generated and saved with new structure.")

    obs_data = generate_observation_data()
    with open(os.path.join(output_dir, "observation_data.json"), "w") as f:
        json.dump(obs_data, f, indent=4)
    print("Observation data generated and saved.")

if __name__ == "__main__":
    main()
