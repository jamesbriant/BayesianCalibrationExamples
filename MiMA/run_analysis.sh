#!/bin/bash

# Usage: ./run_analysis.sh [config_file] [warmup] [main_iter]
# Example: ./run_analysis.sh configs/T21.py 100 100

CONFIG=${1:-configs/T21.py}
WARMUP=${2:-60}
MAIN=${3:-60}
ENV="py311noipython"

# Ensure we are in the script's directory
cd "$(dirname "$0")"

echo "----------------------------------------------------------------"
echo "Starting Analysis Pipeline"
echo "Config: $CONFIG"
echo "Warmup: $WARMUP"
echo "Main Iterations: $MAIN"
echo "Environment: $ENV"
echo "----------------------------------------------------------------"

# 1. Run MCMC
echo "[1/4] Running MCMC Sampling..."
conda run -n $ENV python HMC-mici.py "$CONFIG" "$WARMUP" "$MAIN"
if [ $? -ne 0 ]; then
    echo "Error: MCMC sampling failed."
    exit 1
fi

# Extract experiment name from config filename
EXP_NAME=$(basename "$CONFIG" .py)

# Find the latest run directory in experiments/$EXP_NAME
# We look for directories starting with 20 (year)
LATEST_RUN_DIR=$(ls -td experiments/"$EXP_NAME"/20* 2>/dev/null | head -n 1)

if [ -z "$LATEST_RUN_DIR" ]; then
    echo "Error: No run directory found in experiments/$EXP_NAME"
    exit 1
fi

echo "Latest Run Directory: $LATEST_RUN_DIR"

# 2. Plot Data Samples
echo "[2/4] Plotting Data Samples..."
conda run -n $ENV python plot_sim_sample.py --config "$CONFIG" --output_dir "$LATEST_RUN_DIR"
if [ $? -ne 0 ]; then
    echo "Error: Data plotting failed."
    exit 1
fi

# 3. Run Diagnostics
echo "[3/4] Generating Diagnostic Plots..."
conda run -n $ENV python plot_diagnostics.py --config "$CONFIG" --output_dir "$LATEST_RUN_DIR"
if [ $? -ne 0 ]; then
    echo "Error: Diagnostics plotting failed."
    exit 1
fi

# 4. Run Posterior Predictive Checks (PPC)
echo "[4/4] Running Posterior Predictive Checks..."

# Find the NetCDF file in the run directory
CHAIN_FILE=$(ls "$LATEST_RUN_DIR"/*.nc 2>/dev/null | head -n 1)

if [ -z "$CHAIN_FILE" ]; then
    echo "Error: No chain file found in $LATEST_RUN_DIR to run PPC."
    exit 1
fi

echo "Using chain file: $CHAIN_FILE"
conda run -n $ENV python plot_ppc.py --config "$CONFIG" --file_path "$CHAIN_FILE" --output_dir "$LATEST_RUN_DIR/ppc"
if [ $? -ne 0 ]; then
    echo "Error: PPC plotting failed."
    exit 1
fi

echo "----------------------------------------------------------------"
echo "Pipeline Completed Successfully."
echo "Output Directory: $LATEST_RUN_DIR"
echo "----------------------------------------------------------------"
