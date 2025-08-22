# Calib8 Experiment Workflow

This directory contains a modular workflow for running Bayesian calibration experiments. The workflow is designed to be configuration-driven, allowing for easy definition and execution of new experiments.

## Workflow Overview

The typical workflow is as follows:

1. Define an experiment in a config file.
2. Generate simulation and observation data.
3. Run HMC sampling to obtain posterior distributions.
4. Plot the results.

All commands below should be run from within the `calib8/` directory.

## 1. Defining an Experiment

An experiment is defined by a Python configuration file located in the `configs/` directory (e.g., `configs/calib8.py`). This file contains all the necessary parameters and functions for the experiment, including:

- **Parameter Definitions**: `PARAMETERS` and `CONTROL_PARAMETERS` define the calibration and control parameters, their true values, and their ranges.
- **Data Generation Settings**: `N_SIMULATION_RUNS`, `N_SIMULATION_POINTS`, etc.
- **Simulator Functions**: The config file must also contain the Python functions for the computer model (`eta`), the discrepancy (`discrepancy`), and the true physical process (`zeta`).

## 2. Generating Data

Once a configuration file is defined, you can generate the simulation and observation data.

```bash
python -m data_generator.generate --config configs/calib8.py
```

This command will create `calib8_simulation.json` and `calib8_observation.json` in the `data/` directory within `calib8`.

### Verifying the Data

After generating the data, you can verify that it has been created correctly and can be loaded. Create a simple Python script or use an interactive Python session within the `calib8/` directory to run the following:

```python
from utils import verify_data

# Replace 'calib8' with the FILE_NAME from your config
verify_data("calib8")
```

This will print details about the loaded dataset if successful.

## 3. Running HMC Sampling

With the data generated, you can run Hamiltonian Monte Carlo (HMC) sampling to calibrate the model. Two inference backends are available: Mici and BlackJax.

**Using Mici:**

```bash
python HMC-mici.py --config configs/calib8.py -W 1000 -N 1000 --n_chain 2 --n_processes 2 --max_tree_depth 10
```

**Using BlackJax:**

```bash
python HMC-blackjax.py --config configs/calib8.py -W 1000 -N 1000 --n_chain 2 --max_num_doublings 10
```

- `-W`: Number of warm-up iterations.
- `-N`: Number of main sampling iterations.
- `--n_chain`: Number of MCMC chains to run.
- `--n_processes`: (Mici only) Number of processes to use for parallel computation.
- `--max_tree_depth`: (Mici only) Maximum tree depth for the NUTS sampler.
- `--max_num_doublings`: (BlackJax only) Maximum number of doublings for the NUTS sampler.

The results will be saved in a structured directory. For an experiment named `calib8` (defined by `FILE_NAME` in the config), the output will be in `chains/calib8/`. The saved file is a NetCDF file containing an ArviZ `InferenceData` object.

## 4. Plotting Results

After sampling is complete, you can generate plots from the results file.

```bash
python plotting-script.py --config configs/calib8.py chains/calib8/W1000-N1000-Nsim80.nc
```

Make sure to replace the `.nc` file path with the actual path to your results file. The `Nsim` value may vary depending on your config.

The plots will be saved in a corresponding structured directory. For an experiment named `calib8`, the figures will be in `figures/calib8/`.
