#!/bin/bash
# parallel_lib.sh

# Parallel Execution Library
# Author: Naufal Suryanto
# Description: A bash library for running commands in parallel using tmux.
# Added Features: Support for Conda environments and CUDA_VISIBLE_DEVICES

# Initialize the parallel execution environment
init_parallel_execution() {
    # Number of parallel processes
    n=$1  # Pass the number of parallel processes as the first argument
    custom_session_name=$2  # Optional session name

    # Ensure 'tmux' is installed
    if ! command -v tmux &> /dev/null; then
        echo "tmux is not installed. Please install tmux to use this script."
        exit 1
    fi

    # Create a unique tmux session name if not provided
    if [ -z "$custom_session_name" ]; then
        session_name="parallel_session_$(date +%s)_$$"
    else
        session_name="$custom_session_name"
    fi

    # Create the tmux session
    tmux new-session -d -s "$session_name"

    # Create a temporary directory for intermediate files
    temp_dir=$(mktemp -d -t parallel_lib_XXXXXX)

    # Function to clean up FIFOs, worker scripts, status files, temporary directory, and tmux session
    cleanup() {
        tmux kill-session -t "$session_name"
        rm -rf "$temp_dir"
    }

    trap cleanup EXIT

    # Export variables for use in other functions
    export n
    export session_name
    export temp_dir
}

# Start worker panes and loops
start_workers() {
    for ((i=0; i<n; i++)); do
        if [ $i -gt 0 ]; then
            tmux split-window -t "$session_name"
            tmux select-layout -t "$session_name" tiled
        fi
        pane_index=$i
        fifo="$temp_dir/fifo_$$_$i"
        ready_file="$temp_dir/worker_$$_$i.ready"

        # Create FIFOs for communication
        mkfifo "$fifo"
        
        # Mark worker as ready initially
        touch "$ready_file"

        # Calculate GPU index for the worker
        if [ -n "${gpu_indexes[*]}" ]; then
            gpu_index=${gpu_indexes[$((i % ${#gpu_indexes[@]}))]}
        else
            gpu_index=""
        fi

        # Create a temporary worker script
        worker_script="$temp_dir/worker_$$_$i.sh"
        worker_status_file="$temp_dir/worker_$$_$i.status"
        cat > "$worker_script" <<EOF
#!/bin/bash
fifo="$fifo"
pane_index=$pane_index
status_file="$worker_status_file"
ready_file="$ready_file"
touch "\$status_file"
touch "\$ready_file"
echo "Worker \$pane_index starting"

# Activate Conda environment if specified
if [ -n "$conda_env" ]; then
    echo "Worker \$pane_index activating Conda environment: $conda_env"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$conda_env"
fi

# Set CUDA_VISIBLE_DEVICES if specified
if [ -n "$gpu_index" ]; then
    export CUDA_VISIBLE_DEVICES="$gpu_index"
    echo "Worker \$pane_index using GPU: \$CUDA_VISIBLE_DEVICES"
fi

while true; do
    if read -r cmd < "\$fifo"; then
        # Remove the ready file to indicate we're busy
        rm -f "\$ready_file" 
        
        echo "Worker \$pane_index received command: \$cmd"
        if [ "\$cmd" == "exit" ]; then
            echo "Worker \$pane_index exiting"
            break
        fi
        eval "\$cmd"
        
        # Create ready file to indicate we're available for more work
        touch "\$ready_file"
        echo "Worker \$pane_index ready for next task"
    else
        sleep 0.5
    fi
done
echo "Worker \$pane_index finished"
rm -f "\$status_file"
rm -f "\$ready_file"
exit 0
EOF
        chmod +x "$worker_script"

        # Start the worker script in the tmux pane using exec to replace the shell
        tmux send-keys -t "${session_name}.$pane_index" "exec \"$worker_script\"" C-m
    done
}

# Run commands in parallel
run_commands_in_parallel() {
    commands=("$@")  # Accept the commands as arguments to the function
    total_commands=${#commands[@]}
    command_index=0
    
    # Set to track assigned commands
    assigned_commands=0
    
    echo "Total commands to execute: $total_commands"
    
    # Continue until all commands are completed
    while [ $assigned_commands -lt $total_commands ]; do
        # Check for available workers
        for ((i=0; i<n; i++)); do
            # Skip if we've already assigned all commands
            if [ $assigned_commands -ge $total_commands ]; then
                break
            fi
            
            ready_file="$temp_dir/worker_$$_$i.ready"
            fifo="$temp_dir/fifo_$$_$i"
            
            # If worker is ready (file exists), assign new work
            if [ -f "$ready_file" ]; then
                echo "Worker $i is ready. Sending command: ${commands[$command_index]}"
                echo "${commands[$command_index]}" > "$fifo"
                command_index=$((command_index + 1))
                assigned_commands=$((assigned_commands + 1))
            fi
        done
        
        # Small sleep to avoid high CPU usage when polling
        sleep 0.2
    done
    
    echo "All $total_commands commands have been assigned to workers"
}

# Signal workers to exit
stop_workers() {
    # Wait for all workers to be ready (meaning they've completed their tasks)
    all_ready=false
    while [ "$all_ready" = false ]; do
        all_ready=true
        for ((i=0; i<n; i++)); do
            ready_file="$temp_dir/worker_$$_$i.ready"
            if [ ! -f "$ready_file" ]; then
                all_ready=false
                echo "Waiting for worker $i to complete its task..."
                break
            fi
        done
        if [ "$all_ready" = false ]; then
            sleep 1
        fi
    done
    
    # Signal workers to exit after all commands are completed
    for ((i=0; i<n; i++)); do
        fifo="$temp_dir/fifo_$$_$i"
        echo "Sending 'exit' to worker $i"
        echo "exit" > "$fifo"
    done
}

# Wait for all workers to finish
wait_for_completion() {
    # Wait until all worker processes have finished
    while true; do
        all_done=true
        for ((i=0; i<n; i++)); do
            worker_status_file="$temp_dir/worker_$$_$i.status"
            if [ -f "$worker_status_file" ]; then
                echo "Worker $i is still running"
                all_done=false
                break
            else
                echo "Worker $i has finished"
            fi
        done
        if [ "$all_done" = true ]; then
            echo "All workers have finished"
            break
        fi
        sleep 1
    done
}