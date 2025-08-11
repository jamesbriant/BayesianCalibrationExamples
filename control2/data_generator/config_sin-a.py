FILE_NAME = "sin-a"

# --- General Settings ---
# Number of calibration parameters to use in the simulation.
N_CALIB_PARAMS = 1
N_CONTROL_PARAMS = 2  # Number of control parameters
OUTPUT_DIMS = 1  # Number of output dimensions


# --- Parameter Definitions ---
# Using a list of dictionaries to define parameters generically.
PARAMETERS = [
    {"name": "t0", "true_value": 0.4, "range": [0.25, 0.45]},
    {"name": "t1", "true_value": -3.14, "range": [-3.3, -3.0]},
    {"name": "t2", "true_value": 1.0, "range": [0.8, 1.2]},
    {"name": "t3", "true_value": 1.0, "range": [0.6, 1.1]},
    {"name": "t4", "true_value": 0.5, "range": [0.4, 0.6]},
]

# --- Control Parameter Definitions ---
CONTROL_PARAMETERS = [
    {"name": "x0", "range": [0.02, 3.98]},
    {"name": "x1", "range": [-0.92, 2.98]},
]

# --- Data Generation Settings ---
N_SIMULATION_RUNS = 10 * N_CALIB_PARAMS  # Number of simulation runs (r)
N_SIMULATION_POINTS = 25  # Number of simulation output points per run (n_sim)
N_OBSERVATION_POINTS = 100  # Number of observation points (n_obs)

# --- Observation Noise ---
# Standard deviation of the observation noise.
OBS_NOISE_STD = [0.05]
