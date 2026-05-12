#!/bin/bash
set -euo pipefail

# Run FineWebSecurity BERT filtering jobs in parallel tmux panes.

year_to_filter=${1:-2023}
current_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd "$current_dir/.." && pwd)

if ! command -v tmux &> /dev/null; then
    echo "tmux could not be found. Please install tmux to use this script."
    exit 1
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    hf_token_file="$project_dir/.env"
    if [[ -f "$hf_token_file" ]]; then
        source "$hf_token_file"
    else
        echo "Error: HF_TOKEN is not set and $hf_token_file was not found."
        exit 1
    fi
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "Error: HF_TOKEN is not set after loading environment."
    exit 1
fi

source "$current_dir/parallel_lib.sh"

max_processes=8
if command -v nvidia-smi &> /dev/null; then
    gpu_count=$(nvidia-smi --list-gpus | wc -l)
    if [ "$gpu_count" -eq 0 ]; then
        num_parallel_processes=1
        gpu_indexes=(0)
    elif [ "$gpu_count" -gt "$max_processes" ]; then
        num_parallel_processes=$max_processes
        gpu_indexes=()
        for ((i=0; i<num_parallel_processes; i++)); do gpu_indexes+=("$i"); done
    else
        num_parallel_processes=$gpu_count
        gpu_indexes=()
        for ((i=0; i<num_parallel_processes; i++)); do gpu_indexes+=("$i"); done
    fi
else
    num_parallel_processes=1
    gpu_indexes=(0)
fi

tmux_session_name="fineweb_security_filter"
conda_env=""
export conda_env
export gpu_indexes

init_parallel_execution "$num_parallel_processes" "$tmux_session_name"
start_workers

config_txt="$project_dir/config/fineweb_config.txt"
mapfile -t config_lines < "$config_txt"

commands=()
for fineweb_config in "${config_lines[@]}"; do
    if [[ "$fineweb_config" == *"$year_to_filter"* ]]; then
        if command -v uv &> /dev/null; then
            runner="PYTHONPATH=src uv run python -m fineweb_security.cli.filter_bert"
        else
            runner="PYTHONPATH=src python -m fineweb_security.cli.filter_bert"
        fi
        command="cd \"$project_dir\" && $runner --dataset_subset \"$fineweb_config\" --parallel_worker 2 --hf_token \"$HF_TOKEN\""
        commands+=("$command")
    fi
done

if [ "${#commands[@]}" -eq 0 ]; then
    echo "No FineWeb configs matched year: $year_to_filter"
    stop_workers
    wait_for_completion
    exit 0
fi

run_commands_in_parallel "${commands[@]}"
stop_workers
wait_for_completion

