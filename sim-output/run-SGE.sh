#!/bin/bash

# This script is designed to run a Python script multiple times on an SGE-based
# HPC cluster (like UCL's Myriad), with each run being a separate job. It
# captures timing information for each run.
#
# WORKFLOW:
# 1. Submit jobs (choose 'cpu' or 'gpu'):
#    ./run_sge_timed_script.sh submit path/to/your_script.py cpu
#    ./run_sge_timed_script.sh submit path/to/your_script.py gpu
#
# 2. Check job status until they are complete:
#    qstat
#
# 3. Once all jobs are done, collect the results:
#    ./run_sge_timed_script.sh collect path/to/your_script.py cpu
#    ./run_sge_timed_script.sh collect path/to/your_script.py gpu
#

# --- Main Configuration ---
# Your Python script's input parameters.
W=80
N=50
# A list of the final parameter to loop over.
DIVISOR_LIST="15 12 10 8 7 6 5 4 3 2"

# --- SGE Job Configuration (EDIT THESE) ---
# See: https://www.rc.ucl.ac.uk/docs/Example_Jobscripts/
# Time limit for each individual job (e.g., 3 hours).
JOB_TIME="12:00:00"
# Memory for a CPU job.
CPU_MEM="16G"
# Memory for a GPU job (often higher).
GPU_MEM="16G"
# --- End of Configuration ---

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
    TEMP_DIR="sge_temp_${cleaned_name}_${mode}"
}

# --- Function to submit jobs ---
submit_jobs() {
    local mode=$1
    local use_gpu=false
    local job_mem=$CPU_MEM
    local node_type="L" # Accepts any of E, F, J or L - only L nodes work.
    if [ "$mode" == "gpu" ]; then
        use_gpu=true
        job_mem=$GPU_MEM
    fi

    echo "Preparing to submit jobs for '$PYTHON_SCRIPT' in '$mode' mode..."
    
    # Create temporary directories for job scripts and results.
    mkdir -p "${TEMP_DIR}/sge_scripts" "${TEMP_DIR}/results"
    echo "Temporary files will be stored in '${TEMP_DIR}/'"

    # Loop over the specified list of divisors.
    for d in $DIVISOR_LIST
    do
        JOB_SCRIPT_PATH="${TEMP_DIR}/sge_scripts/job_${d}.sge"
        RESULT_FILE_PATH="${TEMP_DIR}/results/result_${d}.csv"
        
        # Use a "here document" to write the SGE job script.
        cat > "$JOB_SCRIPT_PATH" << EOF
#!/bin/bash -l
#$ -N ${cleaned_name}_${d}_${mode}  # Job name
#$ -l h_rt=${JOB_TIME}
#$ -l mem=${job_mem}
# Add GPU request if needed
$( [ "$use_gpu" = true ] && echo "#$ -l gpu=1" )
$( [ "$use_gpu" = true ] && echo "#$ -ac allow=${node_type}" ) # Accepts any of E, F, J or L

# Check the node type and GPU
echo "--------------------------------------------------"
cat /proc/cpuinfo
if [ "$use_gpu" = true ]; then
    echo "Using GPU for this job."
    nvidia-smi
else
    echo "Running on CPU."
fi
echo "--------------------------------------------------"

# Set the working directory to somewhere in your scratch space.  
#  This is a necessary step as compute nodes cannot write to $HOME.
# Replace "<your_UCL_id>" with your UCL user ID.
#$ -wd /home/ucakjcb/Scratch/BayesianCalibrationExamples/sim-output

# activate the virtual environment
# source /home/ucakjcb/jax-venv.sh
module unload compilers mpi
module load compilers/gnu/4.9.2
if [ "$use_gpu" = true ]; then
    module load cuda/12.2.2/gnu-10.2.0
    module load cudnn/9.2.0.82/cuda-12
fi
module load python3/3.11
source /home/ucakjcb/venvs/jax/bin/activate

# This command runs the python script and captures the timing data.
# The output of 'time' is redirected from stderr to a variable.
# The python script's own stdout is sent to /dev/null.
time_data=\$(/usr/bin/time -f "%e,%U,%S" python3 "$PYTHON_SCRIPT_ABSPATH" "$W" "$N" "$d" 2>&1 >/dev/null)

# Write the input values and the captured time data to a unique result file.
echo "$W,$N,$d,\$time_data" > "$RESULT_FILE_PATH"

# copy the posterior sample chains to the results directory
cp -r chains/* "${TEMP_DIR}/results/"

EOF
        # Submit the generated script to the SGE queue.
        qsub "$JOB_SCRIPT_PATH"
    done

    echo "--------------------------------------------------"
    echo "All jobs submitted."
    echo "Monitor their progress with: qstat"
    echo "Once all jobs are complete, collect the results by running:"
    echo "$0 collect $PYTHON_SCRIPT $mode"
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
    
    # CORRECTED: Header with 'gpu_used' as the first column.
    echo "gpu_used,warm_up,chain_length,divisor,real_time_seconds,user_time_seconds,sys_time_seconds" > "$OUTPUT_FILE"
    
    # Loop through each result file and prepend the gpu_used status.
    for f in "${TEMP_DIR}/results/"*.csv
    do
        # check if file exists to avoid error with empty dir
        [ -e "$f" ] || continue
        result_data=$(cat "$f")
        # CORRECTED: Prepend the gpu_used flag to the data from the file.
        echo "$use_gpu,$result_data" >> "$OUTPUT_FILE"
    done
    
    # CORRECTED: Sort by the fourth column (k4) now that 'gpu_used' is first.
    { head -n 1 "$OUTPUT_FILE" && tail -n +2 "$OUTPUT_FILE" | sort -t, -k4,4n; } > "${OUTPUT_FILE}.tmp" && mv "${OUTPUT_FILE}.tmp" "$OUTPUT_FILE"


    echo "--------------------------------------------------"
    echo "Results successfully collected into '$OUTPUT_FILE'."

    echo "Copy the posterior sample chains to the output directory."
    # Copy the posterior sample chains to the output directory.
    if [ -d "${TEMP_DIR}/results/chains" ]; then
        mkdir -p chains
        cp -r "${TEMP_DIR}/results/chains/"* chains/
        echo "Posterior sample chains copied to 'chains/' directory."
    else
        echo "No posterior sample chains found in '${TEMP_DIR}/results/chains/'."
    fi
    
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

# Get the absolute path to the python script to ensure SGE jobs can find it.
PYTHON_SCRIPT_ABSPATH=$(realpath "$PYTHON_SCRIPT")

# Call the function to set up filenames based on the mode.
setup_names "$MODE"

# Create the 'timings' directory if it doesn't exist.
mkdir -p timings
# Create the temporary directory for SGE jobs.
mkdir -p "$TEMP_DIR"

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
