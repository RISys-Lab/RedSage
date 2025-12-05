#!/usr/bin/env python3
"""
RedSage Transformers Chat Example

This script demonstrates how to use RedSage models (Ins/DPO variants) with
the Hugging Face Transformers library for interactive chat inference.

Usage:
    python hf_chat.py --model RISys-Lab/RedSage-8B-Ins
    python hf_chat.py --model RISys-Lab/RedSage-8B-DPO --max-tokens 512
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Interactive chat with RedSage using Transformers"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="RISys-Lab/RedSage-8B-Ins",
        help="Model name or path (default: RISys-Lab/RedSage-8B-Ins)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate (default: 512)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (default: 0.2)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: 'auto', 'cuda', 'cpu' (default: auto)",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are RedSage, a helpful cybersecurity assistant.",
        help="System prompt to use",
    )
    return parser.parse_args()


def load_model(model_name: str, device: str = "auto"):
    """
    Load the model and tokenizer.
    
    Args:
        model_name: HuggingFace model name or local path
        device: Device placement strategy
        
    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    
    print(f"Model loaded successfully on {model.device}")
    return model, tokenizer


def chat_single_turn(
    model,
    tokenizer,
    user_message: str,
    system_prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.2,
):
    """
    Generate a single-turn chat response.
    
    Args:
        model: The loaded language model
        tokenizer: The loaded tokenizer
        user_message: User's input message
        system_prompt: System prompt for the assistant
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        
    Returns:
        str: Generated response
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # Tokenize and move to device
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Decode the full output
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response (remove the prompt)
    response = full_response[len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)):]
    
    return response.strip()


def interactive_chat(
    model,
    tokenizer,
    system_prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.2,
):
    """
    Run an interactive chat session.
    
    Args:
        model: The loaded language model
        tokenizer: The loaded tokenizer
        system_prompt: System prompt for the assistant
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
    """
    print("\n" + "=" * 70)
    print("RedSage Interactive Chat")
    print("=" * 70)
    print(f"System: {system_prompt}")
    print("Type 'quit' or 'exit' to end the conversation.")
    print("=" * 70 + "\n")
    
    while True:
        try:
            user_input = input("User: ").strip()
            
            if user_input.lower() in ["quit", "exit", ""]:
                print("Goodbye!")
                break
            
            print("\nRedSage: ", end="", flush=True)
            response = chat_single_turn(
                model=model,
                tokenizer=tokenizer,
                user_message=user_input,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
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
    
    try:
        # Load model and tokenizer
        model, tokenizer = load_model(args.model, args.device)
        
        # Start interactive chat
        interactive_chat(
            model=model,
            tokenizer=tokenizer,
            system_prompt=args.system_prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        
    except Exception as e:
        print(f"Fatal error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
