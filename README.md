# Mindvote

## Overview

This pipeline evaluates Large Language Models (LLMs) on their ability to predict poll results by analyzing social contexts patterns. The task involves predicting probability distributions over answer choices for polls from Reddit and Weibo platforms.


## Supported LLM Providers

The evaluation framework supports multiple LLM providers through both API and local inference:

### API-Based Clients
- **OpenAI GPT**: GPT-4o, GPT-4.1, o3-medium
- **Google Gemini**: Gemini 2.5 Pro
- **Anthropic Claude**: Claude 3.7 Sonnet
- **DeepSeek**: DeepSeek-R1


### Local Clients
- **Qwen**: Qwen/Qwen2.5-32B 
- **Llama**: Meta-Llama-2/3/4 series
- **Mistral**: Mistral-7B 
- **Gemma**: Google Gemma-2-9B 

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

See `example_usage.py` for detailed examples of each client. Basic usage:

```python
from pathlib import Path
from evaluation.data_loader import load_default_dataset
from evaluation.prompt_driver import PromptDriver, PromptDriverConfig, ClaudeClient

# Load dataset
dataset = load_default_dataset(data_dir=Path("data"))

# Initialize client
client = ClaudeClient(model_name="claude-3-7-sonnet-20250219")

# Configure and run evaluation
config = PromptDriverConfig(cache_dir=Path("results"))
driver = PromptDriver(config)

for entry in driver.run(dataset.eval.to_dict("records"), client, limit=10):
    print(f"Processed poll: {entry['poll_id']}")
```

## Data Processing

The pipeline includes robust data processing features:
- Configurable holdout percentage for prediction evaluation
- Structured and minimal prompt templates
- Batch processing for efficient evaluation
- Automatic caching and resume capability
- Multiple evaluation metrics (Wasserstein, KL divergence, Spearman, accuracy)

## Evaluation Metrics

The framework computes multiple metrics to assess prediction quality:
- **1-Wasserstein Distance**: Measures distribution similarity
- **1-KL Divergence**: Information theoretic distance
- **Spearman Correlation**: Rank-order agreement
- **One-hot Accuracy**: Winner prediction accuracy

## Project Structure

```
mindvote/
├── data/                     # Poll datasets
│   ├── reddit_polls_en.json
│   └── weibo_polls_en.json
├── evaluation/               # Core evaluation modules
│   ├── data_loader.py       # Dataset loading and normalization
│   ├── prompt_driver.py     # LLM client implementations
│   └── metrics.py           # Evaluation metrics
├── results/                  # Cached model outputs
├── example_usage.py          # Usage examples for all clients
├── requirements.txt          # Python dependencies
└── README.md
```

## Client Usage Examples

### Using API-Based Clients

```python
# OpenAI GPT
from evaluation.prompt_driver import OpenAIClient
client = OpenAIClient(model_name="gpt-4o")

# Gemini
from evaluation.prompt_driver import GeminiClient
client = GeminiClient(model_name="gemini-2.5-pro")

# Claude
from evaluation.prompt_driver import ClaudeClient
client = ClaudeClient(model_name="claude-3-7-sonnet-20250219")

# DeepSeek
from evaluation.prompt_driver import DeepSeekClient
client = DeepSeekClient(model_name="deepseek-reasoner")
```

### Using Local Clients

```python
# Qwen (local)
from evaluation.prompt_driver import QwenLocalClient
client = QwenLocalClient(model_name="Qwen/Qwen2.5-32B")

# Llama (local)
from evaluation.prompt_driver import LlamaLocalClient
client = LlamaLocalClient(model_name="meta-llama/Llama-2-13b")

# Mistral
from evaluation.prompt_driver import MistralClient
client = MistralClient(model_name="mistralai/Mistral-7B-v0.1")

# Gemma
from evaluation.prompt_driver import GemmaClient
client = GemmaClient(model_name="google/gemma-2-9b", device="cuda")


## Computing Metrics

```python
from pathlib import Path
from evaluation.metrics import load_cache_file, evaluate_predictions, summarize_metrics
from evaluation.data_loader import load_default_dataset

# Load dataset and cached results
dataset = load_default_dataset(data_dir=Path("data"))
cache_entries = load_cache_file(Path("results/gpt-4o.jsonl"))

# Evaluate predictions
metrics_df = evaluate_predictions(dataset.records, cache_entries)
summary = summarize_metrics(metrics_df)

print(f"1-Wasserstein: {summary['1-wasserstein']:.4f}")
print(f"1-KL Divergence: {summary['1-kl']:.4f}")
print(f"Spearman: {summary['spearman']:.4f}")
print(f"Accuracy: {summary['one_hot_accuracy']:.4f}")
```

## Environment Variables

Set API keys as environment variables for authentication:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"
export DEEPSEEK_API_KEY="your-deepseek-key"
export DASHSCOPE_API_KEY="your-qwen-key"
export HF_TOKEN="your-huggingface-token"
```

