#!/usr/bin/env python3
"""
RedSage Evaluation Runner

This script provides a convenient wrapper around lighteval for running
cybersecurity benchmark evaluations on RedSage models.

Usage:
    python eval/run_lighteval.py --model RISys-Lab/RedSage-Qwen3-8B-Ins --tasks cybermetrics:80

For more options:
    python eval/run_lighteval.py --help
"""
import os
import argparse
import sys
import subprocess
from pathlib import Path

def get_available_tasks():
    """
    Return a dictionary of available task categories and their tasks.
    
    This function defines all supported evaluation tasks across multiple benchmark suites
    including CyberMetrics, CTI-Bench, MMLU, SECURE, SecBench, and SecEval. Tasks can be
    run individually or in combination.
    
    Returns:
        dict: A dictionary where keys are category names (str) and values are lists of
              task identifiers (str). Task identifiers are either file paths (for curated
              tasks) or lighteval task specifications (e.g., 'cybermetrics:80').
    """
    return {
        "CuratedTasks": [
            "tasks/redsage_mcqs_base.txt",
            "tasks/redsage_mcqs_ins.txt",
            "tasks/related_benchmarks_base.txt",
            "tasks/related_benchmarks_ins.txt",
        ],
        "CyberMetrics": [
            "cybermetrics:80",
            "cybermetrics:500",
            "cybermetrics:2000",
            "cybermetrics:10000",
            "cybermetrics:80_em",
            "cybermetrics:500_em",
            "cybermetrics:2000_em",
            "cybermetrics:10000_em",
        ],
        "CTI-Bench": [
            "cti_bench:cti-mcq",
            "cti_bench:cti-mcq_em",
            "cti_bench:cti-mcq_em_direct",
            "cti_bench:cti-rcm_em",
            "cti_bench:cti-rcm_em_direct",
        ],
        "MMLU": [
            "mmlu:cs_security",
        ],
        "SECURE": [
            "secure:maet",
            "secure:cwet",
            "secure:maet_em",
            "secure:cwet_em",
            "secure:kcv_em",
        ],
        "SecBench": [
            "secbench:mcq-en",
            "secbench:mcq-en_em",
        ],
        "SecEval": [
            "seceval:mcqa",
            "seceval:mcqa_5s",
        ],
        "RedSage-MCQ": [
            "redsage_mcq:cybersecurity_knowledge_generals",
            "redsage_mcq:cybersecurity_knowledge_frameworks",
            "redsage_mcq:cybersecurity_skills",
            "redsage_mcq:cybersecurity_tools_cli",
            "redsage_mcq:cybersecurity_tools_kali",
            "redsage_mcq_em:cybersecurity_knowledge_generals",
            "redsage_mcq_em:cybersecurity_knowledge_frameworks",
            "redsage_mcq_em:cybersecurity_skills",
            "redsage_mcq_em:cybersecurity_tools_cli",
            "redsage_mcq_em:cybersecurity_tools_kali",
        ],
    }


def list_tasks():
    """
    Print all available tasks organized by category.
    
    Displays a formatted list of all available evaluation tasks grouped by their
    benchmark category. For curated tasks, displays file paths. For lighteval tasks,
    displays the task specification in the format required for command-line execution.
    
    Returns:
        None
    """
    tasks = get_available_tasks()
    print("\n=== Available Evaluation Tasks ===\n")
    for category, task_list in tasks.items():
        print(f"{category}:")
        if category == "CuratedTasks":
            for task_file in task_list:
                print(f"  -  {task_file}")
        else:
            for task in task_list:
                print(f"  - lighteval|{task}|n")
        print()


def main():
    """
    Main entry point for the lighteval evaluation runner.
    
    Parses command-line arguments, validates inputs, and constructs and executes a
    lighteval command to run cybersecurity benchmarks on RedSage models. Supports
    multiple backends (accelerate and vLLM) and flexible task specification.
    
    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    parser = argparse.ArgumentParser(
        description="Run RedSage cybersecurity benchmark evaluations using lighteval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a single task
  python eval/run_lighteval.py --model RISys-Lab/RedSage-Qwen3-8B-Ins --tasks cybermetrics:80
  
  # Run multiple tasks
  python eval/run_lighteval.py --model RISys-Lab/RedSage-Qwen3-8B-Ins \\
      --tasks cybermetrics:80,mmlu:cs_security,secbench:mcq-en

  # Run a curated RedSage MCQs task
  python eval/run_lighteval.py --model RISys-Lab/RedSage-Qwen3-8B-Ins --tasks tasks/redsage_mcqs.txt
  
  # Run with specific output directory
  python eval/run_lighteval.py --model RISys-Lab/RedSage-Qwen3-8B-Ins \\
      --tasks cybermetrics:500 --output-dir results/my_eval
  
  # List all available tasks
  python eval/run_lighteval.py --list-tasks
  
  # Run with vLLM backend (for faster inference)
  python eval/run_lighteval.py vllm --model RISys-Lab/RedSage-Qwen3-8B-Ins \\
      --tasks cybermetrics:80

  # Run with custom vLLM parameters
  python eval/run_lighteval.py vllm \\
      --model RISys-Lab/RedSage-Qwen3-8B-Ins \\
      --tasks cybermetrics:80 \\
      --vllm-gpu-memory-utilization 0.8 \\
      --max-model-len 8192
        """,
    )
    
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List all available tasks and exit",
    )
    
    parser.add_argument(
        "backend",
        nargs="?",
        default="accelerate",
        choices=["accelerate", "vllm"],
        help="Backend to use for inference (default: accelerate)",
    )
    
    parser.add_argument(
        "--model",
        type=str,
        help="Model name or path (e.g., RISys-Lab/RedSage-Qwen3-8B-Ins)",
    )
    
    parser.add_argument(
        "--tasks",
        type=str,
        help="Comma-separated list of task names (without 'lighteval|' prefix or '|0' suffix)",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save evaluation results (default: results)",
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to evaluate per task (for testing)",
    )
    
    parser.add_argument(
        "--num-fewshot",
        type=int,
        default=0,
        help="Number of few-shot examples (default: 0)",
    )

    parser.add_argument(
        "--save-details",
        action="store_true",
        help="Save detailed evaluation results",
    )

    chat_template_group = parser.add_mutually_exclusive_group()
    chat_template_group.add_argument(
        "--use-chat-template",
        action="store_true",
        help="Force lighteval to use the model chat template",
    )
    chat_template_group.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Disable the model chat template (for base models)",
    )
    
    # vLLM-specific arguments
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization for vLLM (default: 0.9)",
    )
    
    parser.add_argument(
        "--max-model-len",
        "--vllm-max-model-len",
        dest="max_model_len",
        type=int,
        help="Maximum model sequence length",
    )
    
    parser.add_argument(
        "--vllm-tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: 1)",
    )
    
    args = parser.parse_args()
    
    # Handle --list-tasks flag: display available tasks and exit
    if args.list_tasks:
        list_tasks()
        return 0
    
    # Validate required arguments: model and tasks are mandatory
    if not args.model:
        parser.error("--model is required (unless using --list-tasks)")
    
    if not args.tasks:
        parser.error("--tasks is required (unless using --list-tasks)")
    
    # Locate and validate the custom tasks module
    # This module contains the cybersecurity benchmark task definitions
    script_dir = Path(__file__).parent.absolute()
    custom_tasks_path = script_dir / "cybersecurity_benchmarks.py"
    
    if not custom_tasks_path.exists():
        print(f"Error: Custom tasks file not found at {custom_tasks_path}", file=sys.stderr)
        return 1
    
    # Parse and format tasks for lighteval execution
    # Tasks can be either: (1) file paths to curated task files, or
    # (2) lighteval task specifications with optional few-shot parameters
    task_list = []
    if args.tasks.strip().startswith("tasks/"):
        # Handle curated task file specification
        current_dir = os.path.dirname(os.path.abspath(__file__))
        task_file_path = os.path.join(current_dir, args.tasks.strip())
        if not os.path.isfile(task_file_path):
            print(f"Error: Task file not found at {task_file_path}", file=sys.stderr)
            return 1
        task_list.append(task_file_path)
    else:
        # Handle comma-separated lighteval task specifications
        for task in args.tasks.split(","):
            task = task.strip()
            if task:
                # Append few-shot parameter if not already specified
                # Task format: 'taskname' or 'taskname|num_fewshot'
                if not task.split("|")[-1].isdigit():
                    task = f"{task}|{args.num_fewshot}"
                task_list.append(task)
    
    # Convert task list to comma-separated string for lighteval
    tasks_str = ",".join(task_list)
    
    # Build the lighteval command to be executed
    # Format: lighteval <backend> <model_args> [options] <tasks>
    cmd = [
        "lighteval",
        args.backend,
    ]
    
    # Configure model parameters based on selected backend (vLLM or Accelerate)
    if args.backend == "vllm":
        # vLLM backend configuration for optimized GPU inference
        model_args = [
            f"model_name={args.model}",
            f"gpu_memory_utilization={args.vllm_gpu_memory_utilization}",
            f"tensor_parallel_size={args.vllm_tensor_parallel_size}",
        ]
        if args.max_model_len:
            model_args.append(f"max_model_length={args.max_model_len}")
        if args.use_chat_template:
            model_args.append("override_chat_template=True")
        elif args.no_chat_template:
            model_args.append("override_chat_template=False")
        cmd.append(",".join(model_args))
    else:
        # Accelerate backend configuration for standard distributed inference
        model_args = [f"model_name={args.model}"]
        if args.max_model_len:
            model_args.append(f"max_length={args.max_model_len}")
        if args.use_chat_template:
            model_args.append("override_chat_template=True")
        elif args.no_chat_template:
            model_args.append("override_chat_template=False")
        cmd.append(",".join(model_args))

    # Optional flag to save detailed per-sample evaluation results
    if args.save_details:
        cmd.append("--save-details")
    
    # Add common lighteval arguments
    cmd.extend([
        "--custom-tasks", str(custom_tasks_path),  # Path to custom cybersecurity benchmarks
        "--output-dir", args.output_dir,            # Output directory for results
        tasks_str                                    # Tasks to evaluate
    ])
    
    # Optional: limit number of samples per task (useful for testing)
    if args.max_samples:
        cmd.extend(["--max-samples", str(args.max_samples)])
    
    # Display the command being executed for transparency and debugging
    print("\n" + "="*80)
    print("Running lighteval with the following command:")
    print(" ".join(cmd))
    print("="*80 + "\n")
    
    # Execute lighteval and handle potential errors
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        # lighteval process failed with non-zero exit code
        print(f"\nError: lighteval failed with exit code {e.returncode}", file=sys.stderr)
        return e.returncode
    except FileNotFoundError:
        # lighteval command is not available in the environment
        print("\nError: lighteval command not found. Please ensure lighteval is installed:", file=sys.stderr)
        print("  cd eval/lighteval && pip install -e .", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
