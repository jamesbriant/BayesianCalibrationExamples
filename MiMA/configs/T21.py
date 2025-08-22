FILE_NAME = "T21"

# --- General Settings ---
# Number of calibration parameters to use in the simulation.
N_CALIB_PARAMS = 1  # Number of calibration parameters
N_CONTROL_PARAMS = 1  # Number of control parameters
OUTPUT_DIMS = 1  # Number of output dimensions


# --- Parameter Definitions ---
# Using a list of dictionaries to define parameters generically.
PARAMETERS = [
    {"name": "theta_0", "true_value": 0.4, "range": [0.25, 0.45]},
]

# --- Control Parameter Definitions ---
CONTROL_PARAMETERS = [
    {"name": "x0", "range": [0.02, 3.98]},
]

# --- Data Generation Settings ---
N_SIMULATION_RUNS = 15 * N_CALIB_PARAMS  # Number of simulation runs (r)
N_SIMULATION_POINTS = 25  # Number of simulation output points per run (n_sim)
N_OBSERVATION_POINTS = 100  # Number of observation points (n_obs)

# --- Observation Noise ---
# Standard deviation of the observation noise.
# OBS_NOISE_STD = [0.5]
