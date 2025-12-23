# MiMA Calibration Workflow

This directory contains a modular workflow for running Bayesian calibration experiments using the **MiMA** toolkit. The workflow is centralized around the `mima.py` Command Line Interface (CLI).

## Workflow Overview

The typical workflow involves:
1.  **Running HMC Sampling**: Calibrate the model using MCMC (Mici or BlackJax backends).
2.  **Analysis & Plotting**: Generate diagnostic plots, predictive checks, and posterior analysis.

## unified CLI: `mima.py`

All main operations are performed using the `mima.py` script.

### 1. Running MCMC (`run`)

Run Hamiltonian Monte Carlo sampling for a specific model.

```bash
python mima.py run <model_dir> -W <warmup> -N <main_iter> --n_chain <chains> --backend <backend>
```

**Arguments:**
*   `model_dir`: Path to the model directory (e.g., `models/T21`).
*   `-W`, `--warmup`: Number of warmup iterations (default: 100).
*   `-N`, `--main`: Number of main sampling iterations (default: 100).
*   `--n_chain`: Number of chains (default: 1).
*   `--backend`: MCMC backend to use, either `mici` (default) or `blackjax`.

**Example:**
```bash
python mima.py run models/T21 -W 100 -N 100 --backend mici
```

Output is saved to `experiments/<ModelName>/<Timestamp>_W<W>_N<N>/`.

### 2. Plotting Results (`plot`)

Generate various plots from the experiment results.

**Diagnostic Plots (Trace, Autocorr, ESS):**
```bash
python mima.py plot trace --model_dir <experiment_run_dir>/model --output_dir <experiment_run_dir>
```

**Posterior Predictive Check (PPC):**
```bash
python mima.py plot ppc --model_dir <experiment_run_dir>/model --file_path <experiment_run_dir>/<chain_file>.nc --output_dir <experiment_run_dir>/ppc
```

**Simulation Samples:**
```bash
python mima.py plot sim_sample --model_dir <experiment_run_dir>/model --output_dir <experiment_run_dir>
```

**Predictions (Experimental):**
```bash
python mima.py plot predictions --model_dir <experiment_run_dir>/model --output_dir <experiment_run_dir>
```

### 3. Analyzing Posterior (`analyze`)

Calculate statistics, find local optima, and export results to JSON.

```bash
python mima.py analyze <experiment_run_dir>
```

**Example:**
```bash
python mima.py analyze experiments/T21/20251212_120000_W100_N100
```

## Automated Pipeline: `run_analysis.sh`

The `run_analysis.sh` script automates the entire process: running MCMC, archiving model files, and generating all plots and analysis.

**Usage:**
```bash
./run_analysis.sh <model_dir> <warmup> <main_iter> <n_chain> <backend>
```

**Example:**
```bash
./run_analysis.sh models/T21 100 100 1 mici
```

## Directory Structure

*   `mima.py`: Main CLI entry point.
*   `runners/`: Contains MCMC runner implementations (`mici.py`, `blackjax.py`).
*   `analysis/`: Contains analysis and plotting scripts (`posterior.py`, `diagnostics.py`, etc.).
*   `models/`: Directory containing model definitions (each model has its own folder with `model.py` and `config.py`).
*   `experiments/`: Output directory where experiment results, plots, and analysis are saved.
