#!/bin/bash

# This script runs a Python script sequentially on a local machine over a
# list of inputs. It captures timing information for each run and supports
# specifying a 'cpu' or 'gpu' mode for consistency with cluster scripts.
# The Python script's output is printed to the console in real-time.
#
# USAGE:
# ./run_local_timed_script.sh path/to/your_script.py cpu
# ./run_local_timed_script.sh path/to/your_script.py gpu
#

# --- Main Configuration ---
# Your Python script's input parameters.
W=80
N=50
# A list of the final parameter to loop over.
DIVISOR_LIST="12 10 7 6 5 4 3 2"
# --- End of Configuration ---

# --- Main Script Logic ---
# Check that a script path and a mode were provided.
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <path_to_python_script.py> <cpu|gpu>"
    exit 1
fi

PYTHON_SCRIPT=$1
MODE=$2

# Validate the mode argument.
if [[ "$MODE" != "cpu" && "$MODE" != "gpu" ]]; then
    echo "Error: Invalid mode '$MODE'. Use 'cpu' or 'gpu'."
    exit 1
fi

# Ensure the Python script exists before starting.
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: The specified Python script ('$PYTHON_SCRIPT') was not found."
    exit 1
fi

# --- Dynamic Output Filename Generation ---
# Get just the filename from the path (e.g., "HMC-sin-a.py")
script_filename=$(basename "$PYTHON_SCRIPT")
# Remove the .py extension (e.g., "HMC-sin-a")
script_name_no_ext="${script_filename%.py}"
# Remove the "HMC-" prefix, if it exists (e.g., "sin-a")
cleaned_name="${script_name_no_ext#HMC-}"

# Create the 'timings' directory if it doesn't exist.
mkdir -p timings
# Construct the final output filename, including the mode.
OUTPUT_FILE="timings/timing_results_${cleaned_name}_${MODE}.csv"
# --- End of Filename Generation ---


# Create the CSV file and write the header row.
echo "warm_up,chain_length,divisor,gpu_used,real_time_seconds,user_time_seconds,sys_time_seconds" > "$OUTPUT_FILE"

echo "Starting timing process for '$PYTHON_SCRIPT' in '$MODE' mode..."
echo "Results will be saved to '$OUTPUT_FILE'."

# --- Core Logic with Progress Display ---
# Create a temporary file to store the output from the 'time' command.
TMP_TIME_FILE=$(mktemp)
# Set up a trap to ensure the temporary file is removed when the script exits.
trap 'rm -f -- "$TMP_TIME_FILE"' EXIT

# Set the gpu_used flag based on the mode.
use_gpu=false
if [ "$MODE" == "gpu" ]; then
    use_gpu=true
fi

# Loop over the specified list of divisors.
for d in $DIVISOR_LIST
do
    echo "-------------------------------------"
    echo "Running with input: W=$W, N=$N, Divisor=$d"

    # Execute the time command.
    # The '-o' flag directs the time command's output directly to our temp file.
    # This leaves stdout and stderr free for the Python script's progress.
    /usr/bin/time -f "%e,%U,%S" -o "$TMP_TIME_FILE" python3 "$PYTHON_SCRIPT" "$W" "$N" "$d"

    # Read the timing data from the temporary file.
    time_data=$(tr -d '\n' < "$TMP_TIME_FILE")

    # Combine the input values, gpu_used flag, and time data, then append to the CSV.
    echo "$W,$N,$d,$use_gpu,$time_data" >> "$OUTPUT_FILE"
done

# Sort the final file numerically by the third column (divisor).
{ head -n 1 "$OUTPUT_FILE" && tail -n +2 "$OUTPUT_FILE" | sort -t, -k3,3n; } > "${OUTPUT_FILE}.tmp" && mv "${OUTPUT_FILE}.tmp" "$OUTPUT_FILE"

echo "-------------------------------------"
echo "Timing process finished successfully."
echo "Results are in '$OUTPUT_FILE'."
# The 'trap' set earlier will automatically clean up the temporary file upon exit.
