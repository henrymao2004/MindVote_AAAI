#!/usr/bin/env python3
"""
Command-line script for running MindVote LLM evaluations.

Usage:
    python run_evaluation.py --provider openai --model gpt-4o --limit 10
    python run_evaluation.py --provider claude --model claude-3-7-sonnet-20250219
    python run_evaluation.py --provider qwen-local --model Qwen/Qwen2.5-32B --limit 5
"""

import argparse
import os
from pathlib import Path

from evaluation.data_loader import load_default_dataset
from evaluation.prompt_driver import (
    PromptDriver,
    PromptDriverConfig,
    OpenAIClient,
    GeminiClient,
    QwenClient,
    DeepSeekClient,
    LlamaClient,
    ClaudeClient,
    QwenLocalClient,
    LlamaLocalClient,
    MistralClient,
    GemmaClient,
    TransformersClient,
)
from evaluation.metrics import load_cache_file, evaluate_predictions, summarize_metrics


def get_client(provider: str, model_name: str, **kwargs):
    """Factory function to create the appropriate client."""
    provider = provider.lower()
    
    if provider == "openai" or provider == "gpt":
        return OpenAIClient(model_name=model_name)
    
    elif provider == "gemini":
        return GeminiClient(model_name=model_name)
    
    elif provider == "qwen" or provider == "qwen-api":
        return QwenClient(model_name=model_name)
    
    elif provider == "qwen-local":
        return QwenLocalClient(
            model_name=model_name,
            device_map=kwargs.get("device_map", "auto"),
            torch_dtype=kwargs.get("torch_dtype", "auto"),
            max_new_tokens=kwargs.get("max_new_tokens", 2048),
        )
    
    elif provider == "deepseek":
        return DeepSeekClient(model_name=model_name)
    
    elif provider == "claude":
        return ClaudeClient(model_name=model_name)
    
    elif provider == "llama" or provider == "llama-api":
        return LlamaClient(
            model_name=model_name,
            max_new_tokens=kwargs.get("max_new_tokens", 512),
        )
    
    elif provider == "llama-local":
        return LlamaLocalClient(
            model_name=model_name,
            device_map=kwargs.get("device_map", "auto"),
            torch_dtype=kwargs.get("torch_dtype", "bfloat16"),
            max_new_tokens=kwargs.get("max_new_tokens", 2048),
        )
    
    elif provider == "mistral":
        return MistralClient(
            model_name=model_name,
            device_map=kwargs.get("device_map", "auto"),
            torch_dtype=kwargs.get("torch_dtype", "auto"),
            max_new_tokens=kwargs.get("max_new_tokens", 2048),
        )
    
    elif provider == "gemma":
        return GemmaClient(
            model_name=model_name,
            device=kwargs.get("device", "cuda"),
            max_new_tokens=kwargs.get("max_new_tokens", 2048),
        )
    
    elif provider == "transformers":
        return TransformersClient(
            model_name=model_name,
            device_map=kwargs.get("device_map", "auto"),
            torch_dtype=kwargs.get("torch_dtype", "auto"),
            max_new_tokens=kwargs.get("max_new_tokens", 2048),
        )
    
    else:
        raise ValueError(f"Unknown provider: {provider}")


def run_evaluation(args):
    """Run the evaluation with specified parameters."""
    print(f"\n{'=' * 70}")
    print(f"MindVote Evaluation: {args.provider} - {args.model}")
    print(f"{'=' * 70}\n")
    
    # Load dataset
    print("Loading dataset...")
    data_dir = Path(args.data_dir)
    dataset = load_default_dataset(
        data_dir=data_dir,
        holdout_fraction=args.holdout_fraction,
        random_seed=args.seed,
    )
    print(f"  Train size: {len(dataset.train)}")
    print(f"  Eval size: {len(dataset.eval)}")
    
    # Create client
    print(f"\nInitializing {args.provider} client...")
    client_kwargs = {
        "device_map": args.device_map,
        "torch_dtype": args.torch_dtype,
        "max_new_tokens": args.max_new_tokens,
        "device": args.device,
    }
    client = get_client(args.provider, args.model, **client_kwargs)
    print(f"  Model: {client.model_name}")
    print(f"  Provider: {client.provider}")
    
    # Configure driver
    config = PromptDriverConfig(
        cache_dir=Path(args.cache_dir),
        temperature=args.temperature,
        resume_from_cache=args.resume,
    )
    driver = PromptDriver(config)
    
    # Run evaluation
    split = args.split.lower()
    if split == "train":
        records = dataset.train.to_dict("records")
    elif split == "eval":
        records = dataset.eval.to_dict("records")
    else:
        raise ValueError(f"Unknown split: {split}. Use 'train' or 'eval'.")
    
    print(f"\nRunning evaluation on {split} split...")
    if args.limit:
        print(f"  Limit: {args.limit} samples")
    
    completed = 0
    for entry in driver.run(records, client, limit=args.limit):
        completed += 1
        if completed % 10 == 0 or completed == args.limit:
            print(f"  Processed {completed} polls...")
    
    print(f"\n✓ Evaluation complete! Processed {completed} polls.")
    print(f"  Results cached to: {config.cache_dir / f'{client.model_name.replace(\"/\", \"_\")}.jsonl'}")
    
    # Compute and display metrics if requested
    if args.compute_metrics:
        print(f"\n{'=' * 70}")
        print("Computing Metrics")
        print(f"{'=' * 70}\n")
        
        cache_file = config.cache_dir / f"{client.model_name.replace('/', '_')}.jsonl"
        cache_entries = load_cache_file(cache_file)
        
        metrics_df = evaluate_predictions(dataset.records, cache_entries)
        summary = summarize_metrics(metrics_df)
        
        print("Average Metrics:")
        print(f"  1-Wasserstein Distance: {summary['1-wasserstein']:.4f}")
        print(f"  1-KL Divergence:        {summary['1-kl']:.4f}")
        print(f"  Spearman Correlation:   {summary['spearman']:.4f}")
        print(f"  One-hot Accuracy:       {summary['one_hot_accuracy']:.4f}")
        print(f"\nEvaluated {len(metrics_df)} predictions.")


def main():
    parser = argparse.ArgumentParser(
        description="Run MindVote LLM evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # OpenAI GPT-4o evaluation
  python run_evaluation.py --provider openai --model gpt-4o --limit 10

  # Claude evaluation
  python run_evaluation.py --provider claude --model claude-3-7-sonnet-20250219

  # Gemini evaluation
  python run_evaluation.py --provider gemini --model gemini-2.5-pro

  # Local Qwen evaluation
  python run_evaluation.py --provider qwen-local --model Qwen/Qwen2.5-32B --limit 5

  # Local Llama evaluation
  python run_evaluation.py --provider llama-local --model meta-llama/Llama-2-13b

  # Mistral evaluation
  python run_evaluation.py --provider mistral --model mistralai/Mistral-7B-v0.1

Supported providers:
  API: openai, gemini, qwen, deepseek, claude, llama
  Local: qwen-local, llama-local, mistral, gemma, transformers
        """,
    )
    
    # Required arguments
    parser.add_argument(
        "--provider",
        type=str,
        required=True,
        help="LLM provider (openai, gemini, claude, qwen, deepseek, llama, mistral, gemma, etc.)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name/identifier",
    )
    
    # Optional arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing poll data (default: data)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="results",
        help="Directory for caching results (default: results)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="eval",
        choices=["train", "eval"],
        help="Dataset split to evaluate (default: eval)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to process (default: all)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.2,
        help="Fraction of data for evaluation split (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset splitting (default: 42)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from cache (start fresh)",
    )
    parser.add_argument(
        "--compute-metrics",
        action="store_true",
        help="Compute and display metrics after evaluation",
    )
    
    # Local model arguments
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Device map for local models (default: auto)",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="auto",
        help="Torch dtype for local models (default: auto)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for pipeline models (default: cuda)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Max new tokens for generation (default: 2048)",
    )
    
    args = parser.parse_args()
    args.resume = not args.no_resume
    
    try:
        run_evaluation(args)
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

