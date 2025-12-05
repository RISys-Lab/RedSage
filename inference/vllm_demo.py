#!/usr/bin/env python3
"""
RedSage vLLM Client Demo

This script demonstrates how to interact with a RedSage model served via vLLM
using the OpenAI-compatible API.

Prerequisites:
    - vLLM server running (e.g., `vllm serve RISys-Lab/RedSage-8B-DPO --port 8000`)
    - openai Python package installed (`pip install openai`)

Usage:
    python vllm_demo.py
    python vllm_demo.py --base-url http://localhost:8000/v1
    python vllm_demo.py --model RISys-Lab/RedSage-8B-Ins
"""

import argparse
from typing import List, Dict

try:
    from openai import OpenAI
except ImportError:
    print("Error: The 'openai' package is required for this script.")
    print("Install it with: pip install openai")
    exit(1)


# Default system prompt for RedSage
DEFAULT_SYSTEM_PROMPT = "You are RedSage, a helpful cybersecurity assistant."


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Simple vLLM client for RedSage models"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000/v1",
        help="vLLM server base URL (default: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="RISys-Lab/RedSage-8B-DPO",
        help="Model name (must match server) (default: RISys-Lab/RedSage-8B-DPO)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="EMPTY",
        help="API key (use 'EMPTY' for local servers) (default: EMPTY)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (default: 0.2)",
    )
    return parser.parse_args()


def create_client(base_url: str, api_key: str) -> OpenAI:
    """
    Create an OpenAI client configured for the vLLM server.
    
    Args:
        base_url: The vLLM server's base URL
        api_key: API key (typically 'EMPTY' for local servers)
        
    Returns:
        OpenAI: Configured client instance
    """
    return OpenAI(
        base_url=base_url,
        api_key=api_key,
    )


def chat_completion(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 512,
    temperature: float = 0.2,
) -> str:
    """
    Send a chat completion request to the vLLM server.
    
    Args:
        client: OpenAI client instance
        model: Model name
        messages: List of message dictionaries with 'role' and 'content'
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        
    Returns:
        str: Generated response content
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    
    if not response.choices:
        raise ValueError("No response generated")
    
    return response.choices[0].message.content


def run_examples(
    client: OpenAI,
    model: str,
    max_tokens: int = 512,
    temperature: float = 0.2,
):
    """
    Run a series of example queries to demonstrate the API.
    
    Args:
        client: OpenAI client instance
        model: Model name
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
    """
    examples = [
        {
            "description": "SSRF Mitigation",
            "messages": [
                {
                    "role": "system",
                    "content": DEFAULT_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": "List three SSRF mitigations.",
                },
            ],
        },
        {
            "description": "CVSS Vector Explanation",
            "messages": [
                {
                    "role": "system",
                    "content": DEFAULT_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": "Explain AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H.",
                },
            ],
        },
        {
            "description": "SQL Injection Prevention",
            "messages": [
                {
                    "role": "system",
                    "content": DEFAULT_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": "What are best practices to prevent SQL injection in web applications?",
                },
            ],
        },
    ]
    
    print("\n" + "=" * 70)
    print("RedSage vLLM Demo - Running Example Queries")
    print("=" * 70 + "\n")
    
    for i, example in enumerate(examples, 1):
        print(f"Example {i}: {example['description']}")
        print("-" * 70)
        print(f"User: {example['messages'][1]['content']}")
        print()
        
        try:
            response = chat_completion(
                client=client,
                model=model,
                messages=example["messages"],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            print(f"RedSage: {response}")
        except Exception as e:
            print(f"Error: {e}")
        
        print("\n" + "=" * 70 + "\n")


def interactive_mode(
    client: OpenAI,
    model: str,
    max_tokens: int = 512,
    temperature: float = 0.2,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
):
    """
    Run an interactive chat session with conversation history.
    
    Args:
        client: OpenAI client instance
        model: Model name
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        system_prompt: System prompt for the assistant
    """
    print("\n" + "=" * 70)
    print("RedSage Interactive Chat (vLLM)")
    print("=" * 70)
    print("Type 'quit' or 'exit' to end the conversation.")
    print("=" * 70 + "\n")
    
    # Initialize conversation history with system prompt
    messages = [{"role": "system", "content": system_prompt}]
    
    while True:
        try:
            user_input = input("User: ").strip()
            
            if user_input.lower() in ["quit", "exit"]:
                print("Goodbye!")
                break
            
            # Skip empty input
            if user_input == "":
                continue
            
            # Add user message to conversation history
            messages.append({"role": "user", "content": user_input})
            
            print("\nRedSage: ", end="", flush=True)
            response = chat_completion(
                client=client,
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            # Add assistant's response to conversation history
            messages.append({"role": "assistant", "content": response})
            print(response)
            print()
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again.\n")


def main():
    """Main entry point."""
    args = parse_args()
    
    print(f"Connecting to vLLM server at: {args.base_url}")
    print(f"Using model: {args.model}")
    
    try:
        # Create client
        client = create_client(args.base_url, args.api_key)
        
        # Run example queries
        run_examples(
            client=client,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        
        # Start interactive mode
        print("Starting interactive mode...")
        interactive_mode(
            client=client,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        
    except Exception as e:
        print(f"Fatal error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
