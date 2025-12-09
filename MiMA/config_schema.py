from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Parameter:
    """
    Represents a model parameter for calibration or control.

    Attributes:
        name (str): Name of the parameter.
        range (Tuple[float, float]): Allowed range for the parameter.
        true_value (Optional[float]): True value (if known, for synthetic experiments).
    """

    name: str
    range: Tuple[float, float]
    true_value: Optional[float] = None


@dataclass
class DataConfig:
    """
    Configuration for data sources and variable names.

    Attributes:
        experiment_name (str): Name of the experiment/data folder.
        sim_params_path (str): Path to simulation parameters CSV file.
        sim_data_pattern (str): Glob pattern for simulation data files.
        obs_data_path (str): Path to observation data file.
        variable_name (str): Name of the variable to extract from data files.
        sim_nc_path (Optional[str]): Optional path to a single NetCDF file containing
            all simulation runs (e.g., "moist_variables_all.nc"). When provided,
            this takes precedence over `sim_data_pattern` and `sim_params_path`.
    """

    experiment_name: str
    # Relative to the main data directory or absolute paths
    sim_params_path: str = "Rh.csv"
    sim_data_pattern: str = "moist_vars/*/moist_vars.nc"
    obs_data_path: str = "gpcp_target.nc"
    variable_name: str = "precip_mean"
    sim_nc_path: Optional[str] = None


@dataclass
class ExperimentConfig:
    """
    Configuration for a full experiment, including parameters and data sources.

    Attributes:
        name (str): Name of the experiment/model.
        parameters (List[Parameter]): List of calibration/control parameters.
        data (DataConfig): Data configuration object.
        n_calib_params (int): Number of calibration parameters.
        n_control_params (int): Number of control parameters (default 0).
        output_dims (int): Number of output dimensions (default 1).
        n_simulation_runs (int): Number of simulation runs (default 0).
        n_simulation_points (int): Number of simulation output points (default 0).
        n_observation_points (int): Number of observation points (default 0).
        obs_noise_std (Optional[List[float]]): Observation noise standard deviation(s).
    """

    name: str
    parameters: List[Parameter]
    data: DataConfig

    # Derived or explicit counts
    n_calib_params: int
    n_control_params: int = 0
    output_dims: int = 1

    # MCMC / Simulation settings
    n_simulation_runs: int = 0
    n_simulation_points: int = 0
    n_observation_points: int = 0

    obs_noise_std: Optional[List[float]] = None
