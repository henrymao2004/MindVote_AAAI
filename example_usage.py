"""
Example usage script for MindVote LLM evaluation clients.

This script demonstrates how to use each of the supported LLM clients
for poll prediction evaluation.
"""

from pathlib import Path
from evaluation.data_loader import load_default_dataset
from evaluation.prompt_driver import (
    PromptDriver,
    PromptDriverConfig,
    # API-based clients
    OpenAIClient,
    GeminiClient,
    QwenClient,
    DeepSeekClient,
    LlamaClient,
    ClaudeClient,
    # Local transformers-based clients
    QwenLocalClient,
    LlamaLocalClient,
    MistralClient,
    GemmaClient,
    TransformersClient,
)
from evaluation.metrics import evaluate_predictions, load_cache_file, summarize_metrics


def example_openai_gpt():
    """Example: Using OpenAI GPT models (GPT-4.1, o3-medium, etc.)"""
    print("\n=== OpenAI GPT Example ===")
    
    # Initialize client
    client = OpenAIClient(
        model_name="gpt-4o",  # or "gpt-4.1", "o3-medium"
        # api_key="your-api-key"  # Optional if OPENAI_API_KEY env var is set
    )
    
    # Load dataset
    dataset = load_default_dataset(data_dir=Path("data"))
    
    # Configure prompt driver
    config = PromptDriverConfig(
        cache_dir=Path("results"),
        temperature=0.0,
        resume_from_cache=True,
    )
    driver = PromptDriver(config)
    
    # Run evaluation (limit to 5 for demo)
    print(f"Running evaluation with {client.model_name}...")
    for entry in driver.run(dataset.eval.to_dict("records"), client, limit=5):
        print(f"  Processed poll: {entry['poll_id']}")
    
    print("✓ OpenAI evaluation complete")


def example_gemini():
    """Example: Using Google Gemini models"""
    print("\n=== Google Gemini Example ===")
    
    client = GeminiClient(
        model_name="gemini-2.5-pro",
        # api_key="your-api-key"  # Optional if env var is set
    )
    
    dataset = load_default_dataset(data_dir=Path("data"))
    config = PromptDriverConfig(cache_dir=Path("results"))
    driver = PromptDriver(config)
    
    print(f"Running evaluation with {client.model_name}...")
    for entry in driver.run(dataset.eval.to_dict("records"), client, limit=5):
        print(f"  Processed poll: {entry['poll_id']}")
    
    print("✓ Gemini evaluation complete")


def example_qwen_api():
    """Example: Using Qwen models via API (dashscope)"""
    print("\n=== Qwen API Example ===")
    
    client = QwenClient(
        model_name="qwen2.5-32b-instruct",
        # api_key="your-api-key"  # Optional if env var is set
    )
    
    dataset = load_default_dataset(data_dir=Path("data"))
    config = PromptDriverConfig(cache_dir=Path("results"))
    driver = PromptDriver(config)
    
    print(f"Running evaluation with {client.model_name}...")
    for entry in driver.run(dataset.eval.to_dict("records"), client, limit=5):
        print(f"  Processed poll: {entry['poll_id']}")
    
    print("✓ Qwen API evaluation complete")


def example_qwen_local():
    """Example: Using Qwen models locally with transformers"""
    print("\n=== Qwen Local Example ===")
    
    client = QwenLocalClient(
        model_name="Qwen/Qwen2.5-32B",
        device_map="auto",
        torch_dtype="auto",
        max_new_tokens=2048,
    )
    
    dataset = load_default_dataset(data_dir=Path("data"))
    config = PromptDriverConfig(cache_dir=Path("results"))
    driver = PromptDriver(config)
    
    print(f"Running evaluation with {client.model_name}...")
    for entry in driver.run(dataset.eval.to_dict("records"), client, limit=2):
        print(f"  Processed poll: {entry['poll_id']}")
    
    print("✓ Qwen local evaluation complete")


def example_deepseek():
    """Example: Using DeepSeek models"""
    print("\n=== DeepSeek Example ===")
    
    client = DeepSeekClient(
        model_name="deepseek-reasoner",
        # api_key="your-api-key"  # Optional if DEEPSEEK_API_KEY env var is set
    )
    
    dataset = load_default_dataset(data_dir=Path("data"))
    config = PromptDriverConfig(cache_dir=Path("results"))
    driver = PromptDriver(config)
    
    print(f"Running evaluation with {client.model_name}...")
    for entry in driver.run(dataset.eval.to_dict("records"), client, limit=5):
        print(f"  Processed poll: {entry['poll_id']}")
    
    print("✓ DeepSeek evaluation complete")


def example_claude():
    """Example: Using Anthropic Claude models"""
    print("\n=== Claude Example ===")
    
    client = ClaudeClient(
        model_name="claude-3-7-sonnet-20250219",
        # api_key="your-api-key"  # Optional if ANTHROPIC_API_KEY env var is set
    )
    
    dataset = load_default_dataset(data_dir=Path("data"))
    config = PromptDriverConfig(cache_dir=Path("results"))
    driver = PromptDriver(config)
    
    print(f"Running evaluation with {client.model_name}...")
    for entry in driver.run(dataset.eval.to_dict("records"), client, limit=5):
        print(f"  Processed poll: {entry['poll_id']}")
    
    print("✓ Claude evaluation complete")


def example_llama_api():
    """Example: Using Llama models via Hugging Face Inference API"""
    print("\n=== Llama API Example ===")
    
    client = LlamaClient(
        model_name="meta-llama/Llama-3-70b-instruct",
        # api_token="your-hf-token"  # Optional if HF_TOKEN env var is set
        max_new_tokens=512,
    )
    
    dataset = load_default_dataset(data_dir=Path("data"))
    config = PromptDriverConfig(cache_dir=Path("results"))
    driver = PromptDriver(config)
    
    print(f"Running evaluation with {client.model_name}...")
    for entry in driver.run(dataset.eval.to_dict("records"), client, limit=5):
        print(f"  Processed poll: {entry['poll_id']}")
    
    print("✓ Llama API evaluation complete")


def example_llama_local():
    """Example: Using Llama models locally with transformers"""
    print("\n=== Llama Local Example ===")
    
    client = LlamaLocalClient(
        model_name="meta-llama/Llama-2-13b",
        device_map="auto",
        torch_dtype="bfloat16",
        max_new_tokens=2048,
    )
    
    dataset = load_default_dataset(data_dir=Path("data"))
    config = PromptDriverConfig(cache_dir=Path("results"))
    driver = PromptDriver(config)
    
    print(f"Running evaluation with {client.model_name}...")
    for entry in driver.run(dataset.eval.to_dict("records"), client, limit=2):
        print(f"  Processed poll: {entry['poll_id']}")
    
    print("✓ Llama local evaluation complete")


def example_mistral():
    """Example: Using Mistral models locally"""
    print("\n=== Mistral Example ===")
    
    client = MistralClient(
        model_name="mistralai/Mistral-7B-v0.1",
        device_map="auto",
        torch_dtype="auto",
        max_new_tokens=2048,
    )
    
    dataset = load_default_dataset(data_dir=Path("data"))
    config = PromptDriverConfig(cache_dir=Path("results"))
    driver = PromptDriver(config)
    
    print(f"Running evaluation with {client.model_name}...")
    for entry in driver.run(dataset.eval.to_dict("records"), client, limit=2):
        print(f"  Processed poll: {entry['poll_id']}")
    
    print("✓ Mistral evaluation complete")


def example_gemma():
    """Example: Using Gemma models locally"""
    print("\n=== Gemma Example ===")
    
    client = GemmaClient(
        model_name="google/gemma-2-9b",
        device="cuda",  # or "mps" for Mac, "cpu" for CPU
        max_new_tokens=2048,
    )
    
    dataset = load_default_dataset(data_dir=Path("data"))
    config = PromptDriverConfig(cache_dir=Path("results"))
    driver = PromptDriver(config)
    
    print(f"Running evaluation with {client.model_name}...")
    for entry in driver.run(dataset.eval.to_dict("records"), client, limit=2):
        print(f"  Processed poll: {entry['poll_id']}")
    
    print("✓ Gemma evaluation complete")


def example_custom_transformers():
    """Example: Using the generic TransformersClient for any HF model"""
    print("\n=== Custom Transformers Example ===")
    
    # You can use this for any transformers-compatible model
    client = TransformersClient(
        model_name="meta-llama/Llama-2-7b",  # or any other model
        device_map="auto",
        torch_dtype="auto",
        max_new_tokens=2048,
    )
    
    dataset = load_default_dataset(data_dir=Path("data"))
    config = PromptDriverConfig(cache_dir=Path("results"))
    driver = PromptDriver(config)
    
    print(f"Running evaluation with {client.model_name}...")
    for entry in driver.run(dataset.eval.to_dict("records"), client, limit=2):
        print(f"  Processed poll: {entry['poll_id']}")
    
    print("✓ Custom transformers evaluation complete")


def example_compute_metrics():
    """Example: Computing metrics from cached results"""
    print("\n=== Computing Metrics Example ===")
    
    # Load dataset
    dataset = load_default_dataset(data_dir=Path("data"))
    
    # Load cached results for a specific model
    cache_file = Path("results/gpt-4o.jsonl")
    if cache_file.exists():
        cache_entries = load_cache_file(cache_file)
        
        # Evaluate predictions
        metrics_df = evaluate_predictions(dataset.records, cache_entries)
        
        # Summarize metrics
        summary = summarize_metrics(metrics_df)
        
        print(f"Results for {cache_file.name}:")
        print(f"  1-Wasserstein: {summary['1-wasserstein']:.4f}")
        print(f"  1-KL Divergence: {summary['1-kl']:.4f}")
        print(f"  Spearman Correlation: {summary['spearman']:.4f}")
        print(f"  One-hot Accuracy: {summary['one_hot_accuracy']:.4f}")
        
        print("✓ Metrics computation complete")
    else:
        print(f"Cache file not found: {cache_file}")


if __name__ == "__main__":
    print("=" * 60)
    print("MindVote LLM Evaluation Examples")
    print("=" * 60)
    
    # Uncomment the examples you want to run:
    
    # API-based clients (require API keys)
    # example_openai_gpt()
    # example_gemini()
    # example_qwen_api()
    # example_deepseek()
    # example_claude()
    # example_llama_api()
    
    # Local transformers-based clients (require GPU/compute)
    # example_qwen_local()
    # example_llama_local()
    # example_mistral()
    # example_gemma()
    # example_custom_transformers()
    
    # Metrics computation
    # example_compute_metrics()
    
    print("\n" + "=" * 60)
    print("To run an example, uncomment the corresponding function call")
    print("in the __main__ section of this script.")
    print("=" * 60)

