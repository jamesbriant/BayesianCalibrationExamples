FILE_NAME = "calib8"

# --- General Settings ---
# Number of calibration parameters to use in the simulation.
N_CALIB_PARAMS = 8  # Number of calibration parameters
N_CONTROL_PARAMS = 1  # Number of control parameters
OUTPUT_DIMS = 1  # Number of output dimensions


# --- Parameter Definitions ---
# Using a list of dictionaries to define parameters generically.
PARAMETERS = [
    {"name": "t0", "true_value": 0.4, "range": [0.25, 0.45]},
    {"name": "t1", "true_value": -0.22, "range": [0.0, -0.3]},
    {"name": "t2", "true_value": 1.25, "range": [1.1, 1.35]},
    {"name": "t3", "true_value": 1.0, "range": [0.95, 1.05]},
    {"name": "t4", "true_value": -2.45, "range": [-2.65, -2.4]},
    {"name": "t5", "true_value": 2.0, "range": [1.9, 2.1]},
    {"name": "t6", "true_value": 0.7, "range": [0.6, 0.85]},
    {"name": "t7", "true_value": 12.2, "range": [10.0, 20.0]},
]

# (
#     t0
#     + t1 * x0**t2
#     + t3 * (x0 - t4) ** t5 * np.sin(t6 * x0 * np.pi)
#     - t7 * np.exp(-2 * x0)
# )

# --- Control Parameter Definitions ---
CONTROL_PARAMETERS = [
    {"name": "x0", "range": [0.02, 3.98]},
]

# --- Data Generation Settings ---
N_SIMULATION_RUNS = 10 * N_CALIB_PARAMS  # Number of simulation runs (r)
N_SIMULATION_POINTS = 25  # Number of simulation output points per run (n_sim)
N_OBSERVATION_POINTS = 100  # Number of observation points (n_obs)

# --- Observation Noise ---
# Standard deviation of the observation noise.
OBS_NOISE_STD = [0.5]
