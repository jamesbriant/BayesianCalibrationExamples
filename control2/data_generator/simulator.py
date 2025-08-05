import numpy as np
from . import config

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

    # Get all parameter values, using true values as defaults
    all_params = np.array([p['true_value'] for p in config.PARAMETERS])

    # Create a parameter matrix of shape (n_all_params, n_points)
    params_matrix = np.tile(all_params[:, np.newaxis], (1, n_points))

    # Overwrite with provided calibration parameters
    params_matrix[:n_calib_params, :] = t.T

    t0, t1, t2, t3, t4 = params_matrix

    x0 = x[:, 0]
    x1 = x[:, 1]

    # Output 1: Original sine function
    output1 = t0 * np.sin(t1 * x0 + t4 * x1 + t2) + t3

    # Output 2: A simple quadratic function for demonstration
    output2 = t0 * (x0 - 2)**2 + t1 * x1

    return np.vstack([output1, output2]).T

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
    x1 = x[:, 1]

    # Discrepancy for the first output
    discrepancy1 = np.exp(0.14 * x0 - 0.14 * x1) / 10

    # Discrepancy for the second output (e.g., a small linear trend)
    discrepancy2 = 0.05 * x0 + 0.02 * x1

    return np.vstack([discrepancy1, discrepancy2]).T

def zeta(x: np.ndarray) -> np.ndarray:
    """
    True physical process function. This is the sum of the computer model
    run with the true calibration parameters and the discrepancy function.

    Args:
        x (np.ndarray): Control parameters, shape (n_points, n_control_params).

    Returns:
        np.ndarray: True process output, shape (n_points, n_output_dims).
    """
    true_theta_values = np.array([p['true_value'] for p in config.PARAMETERS])
    t_true = np.tile(true_theta_values, (x.shape[0], 1))

    return eta(x, t_true) + discrepancy(x)
