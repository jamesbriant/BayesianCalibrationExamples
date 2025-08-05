import numpy as np

# --- General Settings ---
# Number of calibration parameters to use in the simulation.
# This can be changed to generate data for a different number of parameters.
N_CALIB_PARAMS = 2

# --- Parameter Definitions ---
# Using a list of dictionaries to define parameters generically.
PARAMETERS = [
    {'name': 't0', 'true_value': 0.4, 'range': [0.25, 0.45]},
    {'name': 't1', 'true_value': -3.14, 'range': [-3.3, -3.0]},
    {'name': 't2', 'true_value': 1.0, 'range': [0.8, 1.2]},
    {'name': 't3', 'true_value': 1.0, 'range': [0.6, 1.1]},
    {'name': 't4', 'true_value': 0.5, 'range': [0.4, 0.6]},
]

# --- Control Parameter Definitions ---
CONTROL_PARAMETERS = [
    {'name': 'x0', 'range': [0.02, 3.98]},
    {'name': 'x1', 'range': [-0.92, 2.98]},
]

# --- Data Generation Settings ---
N_SIMULATION_RUNS = 10 * N_CALIB_PARAMS  # Number of simulation runs (r)
N_SIMULATION_POINTS = 25  # Number of simulation output points per run (n_sim)
N_OBSERVATION_POINTS = 100  # Number of observation points (n_obs)

# --- Observation Noise ---
# Standard deviation of the observation noise.
# This can be a single value or a list of values for each output dimension.
OBS_NOISE_STD = [0.05, 0.02]
