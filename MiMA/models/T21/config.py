import os
import sys

# Add parent directory to path to allow importing config_schema
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config_schema import DataConfig, ExperimentConfig, Parameter

experiment_config = ExperimentConfig(
    name="T21",
    parameters=[Parameter(name="theta_0", range=(0.1, 0.9))],
    data=DataConfig(
        experiment_name="T21",
        sim_params_path="Rh.csv",
        sim_data_pattern="moist_vars/*/moist_vars.nc",
        obs_data_path="gpcp_target.nc",
        variable_name="precip_mean",
    ),
    n_calib_params=1,
    n_control_params=1,
    output_dims=1,
    n_simulation_runs=15,  # 15 * 1
    n_simulation_points=25,
    n_observation_points=100,
)

FILE_NAME = experiment_config.name

# --- General Settings ---
# Number of calibration parameters to use in the simulation.
N_CALIB_PARAMS = experiment_config.n_calib_params
N_CONTROL_PARAMS = experiment_config.n_control_params
OUTPUT_DIMS = experiment_config.output_dims


# --- Parameter Definitions ---
# Using a list of dictionaries to define parameters generically.
PARAMETERS = [
    {"name": p.name, "true_value": p.true_value, "range": list(p.range)}
    for p in experiment_config.parameters
]

# # --- Control Parameter Definitions ---
# CONTROL_PARAMETERS = [
#     {"name": "x0", "range": [0.02, 3.98]},
# ]

# --- Data Generation Settings ---
N_SIMULATION_RUNS = experiment_config.n_simulation_runs
N_SIMULATION_POINTS = experiment_config.n_simulation_points
N_OBSERVATION_POINTS = experiment_config.n_observation_points

# --- Observation Noise ---
# Standard deviation of the observation noise.
# OBS_NOISE_STD = [0.5]
