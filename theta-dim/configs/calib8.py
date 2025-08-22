import numpy as np

FILE_NAME = "calib8"

# --- General Settings ---
# Number of calibration parameters to use in the simulation.
N_CALIB_PARAMS = 8  # Number of calibration parameters
N_CONTROL_PARAMS = 1  # Number of control parameters
OUTPUT_DIMS = 1  # Number of output dimensions


# --- Parameter Definitions ---
# Using a list of dictionaries to define parameters generically.
PARAMETERS = [
    {"name": "theta_0", "true_value": 0.4, "range": [0.25, 0.45]},
    {"name": "theta_1", "true_value": -0.22, "range": [-0.3, 0.0]},
    {"name": "theta_2", "true_value": 1.25, "range": [1.1, 1.35]},
    {"name": "theta_3", "true_value": 1.0, "range": [0.95, 1.05]},
    {"name": "theta_4", "true_value": -2.45, "range": [-2.65, -2.4]},
    {"name": "theta_5", "true_value": 2.0, "range": [1.9, 2.1]},
    {"name": "theta_6", "true_value": 0.7, "range": [0.6, 0.85]},
    {"name": "theta_7", "true_value": 12.2, "range": [10.0, 20.0]},
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


def eta(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Computer model/simulator function.
    Handles generic control (x) and calibration (t) parameters and
    produces a multi-dimensional output.

    Args:
        x (np.ndarray): Control parameters, shape (n_points, n_control_params).
        t (np.ndarray): Calibration parameters, shape (n_points, n_calib_params).

    Returns:
        np.ndarray: Simulator output, shape (n_points, n_output_dims).
    """
    n_points = x.shape[0]
    n_calib_params = t.shape[1]

    # Get all parameter values, using true values from the config as defaults
    all_params = np.array([p["true_value"] for p in PARAMETERS])

    # Create a parameter matrix of shape (n_all_params, n_points)
    params_matrix = np.tile(all_params[:, np.newaxis], (1, n_points))

    # Overwrite with provided calibration parameters
    params_matrix[:n_calib_params, :] = t.T

    t0, t1, t2, t3, t4, t5, t6, t7 = params_matrix

    x0 = x[:, 0]
    # x1 = x[:, 1]

    # Output 1: Original sine function
    output1 = (
        t0
        + t1 * x0**t2
        + t3 * (x0 - t4) ** t5 * np.cos(t6 * x0 * np.pi) * np.exp(-0.5 * x0)
        - t7 * np.exp(-2 * x0)
    )

    # Output 2: A simple quadratic function for demonstration
    # output2 = t0 * (x0 - 2)**2 + t1 * x1

    # return np.vstack([output1, output2]).T
    return output1[:, np.newaxis]  # Return as a 2D array with one column


def discrepancy(x: np.ndarray) -> np.ndarray:
    """
    Discrepancy function. This represents the difference between the
    computer model and the true physical process. For multi-dimensional
    output, this should also be multi-dimensional.

    Args:
        x (np.ndarray): Control parameters, shape (n_points, n_control_params).

    Returns:
        np.ndarray: Discrepancy output, shape (n_points, n_output_dims).
    """
    x0 = x[:, 0]
    # x1 = x[:, 1]

    # Discrepancy for the first output
    discrepancy1 = np.exp(0.14 * x0) / 10

    # Discrepancy for the second output (e.g., a small linear trend)
    # discrepancy2 = 0.05 * x0 + 0.02 * x1

    # return np.vstack([discrepancy1, discrepancy2]).T
    return discrepancy1[:, np.newaxis]  # Return as a 2D array with one column


def zeta(x: np.ndarray) -> np.ndarray:
    """
    True physical process function. This is the sum of the computer model
    run with the true calibration parameters and the discrepancy function.

    Args:
        x (np.ndarray): Control parameters, shape (n_points, n_control_params).

    Returns:
        np.ndarray: True process output, shape (n_points, n_output_dims).
    """
    true_theta_values = np.array([p["true_value"] for p in PARAMETERS])
    t_true = np.tile(true_theta_values, (x.shape[0], 1))

    return eta(x, t_true) + discrepancy(x)
