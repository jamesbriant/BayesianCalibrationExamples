#!/bin/bash

# This script is designed to run a Python script multiple times on a Slurm-based
# HPC cluster, with each run being a separate job. It captures timing information
# for each run and supports both CPU and GPU job submissions.
#
# WORKFLOW:
# 1. Submit jobs (choose 'cpu' or 'gpu'):
#    ./run_slurm_timed_script.sh submit path/to/your_script.py cpu
#    ./run_slurm_timed_script.sh submit path/to/your_script.py gpu
#
# 2. Check job status until they are complete:
#    squeue -u $USER
#
# 3. Once all jobs are done, collect the results:
#    ./run_slurm_timed_script.sh collect path/to/your_script.py cpu
#    ./run_slurm_timed_script.sh collect path/to/your_script.py gpu
#

# --- Main Configuration ---
# Your Python script's input parameters.
W=80
N=50
# A list of the final parameter to loop over.
DIVISOR_LIST="12 10 7 6 5 4 3 2"

# --- Slurm Job Configuration (EDIT THESE) ---
# Time limit for each individual job (e.g., 3 hours).
JOB_TIME="03:00:00"
# Memory for a CPU job.
CPU_MEM="16G"
# Memory for a GPU job (often higher).
GPU_MEM="16G"
# Your cluster's general partition (e.g., "batch", "compute"). Leave blank if not needed.
JOB_PARTITION=""
# Your cluster's GPU partition. Leave blank if GPUs are in the main partition.
JOB_GPU_PARTITION=""
# --- End of Configuration ---

# Create the 'timings' directory if it doesn't exist.
mkdir -p timings

# --- Function to derive names from the Python script path and mode ---
setup_names() {
    local mode=$1 # cpu or gpu

    # Get just the filename from the path (e.g., "HMC-sin-a.py")
    script_filename=$(basename "$PYTHON_SCRIPT")
    # Remove the .py extension (e.g., "HMC-sin-a")
    script_name_no_ext="${script_filename%.py}"
    # Remove the "HMC-" prefix, if it exists (e.g., "sin-a")
    cleaned_name="${script_name_no_ext#HMC-}"

    # Construct the final output filename, including the mode.
    OUTPUT_FILE="timings/timing_results_${cleaned_name}_${mode}.csv"
    # Construct a name for the directory that will hold temporary files.
    TEMP_DIR="slurm_temp_${cleaned_name}_${mode}"
}

# --- Function to submit jobs ---
submit_jobs() {
    local mode=$1
    local use_gpu=false
    local job_mem=$CPU_MEM
    local job_partition=$JOB_PARTITION

    if [ "$mode" == "gpu" ]; then
        use_gpu=true
        job_mem=$GPU_MEM
        # If a specific GPU partition is set, use it.
        [ -n "$JOB_GPU_PARTITION" ] && job_partition=$JOB_GPU_PARTITION
    fi

    echo "Preparing to submit jobs for '$PYTHON_SCRIPT' in '$mode' mode..."
    
    # Create temporary directories for job scripts, results, and slurm logs.
    mkdir -p "${TEMP_DIR}/slurm_scripts" "${TEMP_DIR}/results" "${TEMP_DIR}/slurm_logs"
    echo "Temporary files will be stored in '${TEMP_DIR}/'"

    # Loop over the specified list of divisors.
    for d in $DIVISOR_LIST
    do
        JOB_SCRIPT_PATH="${TEMP_DIR}/slurm_scripts/job_${d}.slurm"
        RESULT_FILE_PATH="${TEMP_DIR}/results/result_${d}.csv"
        
        # Use a "here document" to write the Slurm job script.
        cat > "$JOB_SCRIPT_PATH" << EOF
#!/bin/bash
#SBATCH --job-name=time_${cleaned_name}_${d}
#SBATCH --output=${TEMP_DIR}/slurm_logs/job_${d}.out
#SBATCH --error=${TEMP_DIR}/slurm_logs/job_${d}.err
#SBATCH --time=${JOB_TIME}
#SBATCH --mem=${job_mem}
$( [ -n "$job_partition" ] && echo "#SBATCH --partition=${job_partition}" )
$( [ "$use_gpu" = true ] && echo "#SBATCH --gpus=1" )

# This command runs the python script and captures the timing data.
# The output of 'time' is redirected from stderr to a variable.
# The python script's own stdout is sent to /dev/null.
time_data=\$(/usr/bin/time -f "%e,%U,%S" python3 "$PYTHON_SCRIPT_ABSPATH" "$W" "$N" "$d" 2>&1 >/dev/null)

# Write the input values and the captured time data to a unique result file.
echo "$W,$N,$d,\$time_data" > "$RESULT_FILE_PATH"

EOF
        # Submit the generated script to the Slurm queue.
        sbatch "$JOB_SCRIPT_PATH"
    done

    echo "--------------------------------------------------"
    echo "All jobs submitted."
    echo "Monitor their progress with: squeue -u \$USER"
    echo "Once all jobs are complete, collect the results by running:"
    echo "$0 collect '$PYTHON_SCRIPT' '$mode'"
    echo "--------------------------------------------------"
}

# --- Function to collect results ---
collect_results() {
    local mode=$1
    local use_gpu=false
    if [ "$mode" == "gpu" ]; then
        use_gpu=true
    fi

    echo "Collecting results from '${TEMP_DIR}/' for mode '$mode'..."

    if [ ! -d "${TEMP_DIR}/results" ]; then
        echo "Error: The results directory '${TEMP_DIR}/results' was not found."
        exit 1
    fi
    
    # Create the final CSV file and write the header, including the new gpu_used column.
    echo "warm_up,chain_length,divisor,gpu_used,real_time_seconds,user_time_seconds,sys_time_seconds" > "$OUTPUT_FILE"
    
    # Loop through each result file, insert the gpu_used status, and append to the main CSV.
    for f in "${TEMP_DIR}/results/"*.csv
    do
        # check if file exists to avoid error with empty dir
        [ -e "$f" ] || continue
        # Robustly insert the 'gpu_used' column after the 3rd column.
        col1_3=$(cut -d, -f1-3 "$f")
        col4_end=$(cut -d, -f4- "$f")
        echo "$col1_3,$use_gpu,$col4_end" >> "$OUTPUT_FILE"
    done
    
    # Sort the final file numerically by the third column (divisor).
    # This is optional but makes the final output cleaner.
    { head -n 1 "$OUTPUT_FILE" && tail -n +2 "$OUTPUT_FILE" | sort -t, -k3,3n; } > "${OUTPUT_FILE}.tmp" && mv "${OUTPUT_FILE}.tmp" "$OUTPUT_FILE"


    echo "--------------------------------------------------"
    echo "Results successfully collected into '$OUTPUT_FILE'."
    
    # Ask the user if they want to clean up the temporary files.
    read -p "Do you want to remove the temporary directory '${TEMP_DIR}/'? (y/n) " -n 1 -r
    echo # Move to a new line
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -r "$TEMP_DIR"
        echo "Removed temporary directory."
    fi
    echo "Process complete."
}


# --- Main Script Logic ---
# Check that a command, script path, and mode were provided.
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <submit|collect> <path_to_python_script.py> <cpu|gpu>"
    exit 1
fi

COMMAND=$1
PYTHON_SCRIPT=$2
MODE=$3

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

# Get the absolute path to the python script to ensure Slurm jobs can find it.
PYTHON_SCRIPT_ABSPATH=$(realpath "$PYTHON_SCRIPT")

# Call the function to set up filenames based on the mode.
setup_names "$MODE"

# Execute the appropriate function based on the command.
case "$COMMAND" in
    submit)
        submit_jobs "$MODE"
        ;;
    collect)
        collect_results "$MODE"
        ;;
    *)
        echo "Error: Invalid command '$COMMAND'. Use 'submit' or 'collect'."
        exit 1
        ;;
esac
