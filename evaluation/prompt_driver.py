"""Prompt construction and execution utilities for MindVote evaluations."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Protocol, Sequence, Union

from .data_loader import PollRecord


class ModelClient(Protocol):
    model_name: str
    provider: str

    def generate(self, prompt: str, temperature: float = 0.0) -> str:  # pragma: no cover - interface
        ...


@dataclass
class PromptDriverConfig:
    cache_dir: Path = Path("results")
    temperature: float = 0.0
    resume_from_cache: bool = True


class PromptDriver:
    """Builds prompts, dispatches them to model clients, and caches responses."""

    def __init__(self, config: PromptDriverConfig):
        self.config = config
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        records: Iterable[Union[PollRecord, Mapping[str, object]]],
        client: ModelClient,
        limit: Optional[int] = None,
    ) -> Iterator[Dict[str, object]]:
        cache_file = self.config.cache_dir / f"{client.model_name.replace('/', '_')}.jsonl"
        processed: Dict[str, Dict[str, object]] = {}

        if self.config.resume_from_cache and cache_file.exists():
            with cache_file.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    processed[entry["poll_id"]] = entry

        completed = 0
        with cache_file.open("a", encoding="utf-8") as sink:
            for record in records:
                if limit is not None and completed >= limit:
                    break

                payload = _ensure_dict(record)
                poll_id = str(payload["poll_id"])
                if poll_id in processed:
                    continue

                prompt_text = self._build_prompt(payload)
                response_text = client.generate(prompt_text, temperature=self.config.temperature)
                parsed_response = _try_parse_json(response_text)

                entry = {
                    "poll_id": poll_id,
                    "timestamp": time.time(),
                    "model": client.model_name,
                    "provider": client.provider,
                    "prompt": prompt_text,
                    "response_text": response_text,
                    "parsed_response": parsed_response,
                }
                sink.write(json.dumps(entry, ensure_ascii=False) + "\n")
                sink.flush()
                processed[poll_id] = entry
                completed += 1
                yield entry

    def _build_prompt(self, payload: Mapping[str, object]) -> str:
        options = [
            {
                "option_index": idx,
                "option_text": option_text,
                "percentage": "Guess Here",
            }
            for idx, option_text in enumerate(payload.get("options", []))
        ]

        metadata = {
            "platform": payload.get("platform"),
            "platform_context": payload.get("platform_context"),
            "topical_context": payload.get("topical_context"),
            "temporal_context": payload.get("temporal_context"),
            "topic": payload.get("topic"),
            "subtopic": payload.get("subtopic"),
        }
        metadata = {key: value for key, value in metadata.items() if value not in (None, "", [], {})}

        prompt_payload = {
            "poll_id": payload["poll_id"],
            "question": payload["question"],
            "total_votes": payload.get("total_votes"),
            "options": options,
        }
        if metadata:
            prompt_payload["metadata"] = metadata

        instruction = (
            "Role-Play Instructions:\n"
            "- Act as a poll prediction expert analyzing voting patterns and social dynamics.\n"
            "- Use only the authenticated poll text and metadata provided in the JSON object.\n"
            "\nPoll Question & Metadata:\n"
            "- The poll JSON below mirrors the content of poll_question.json and meta.json. Treat every field as factual.\n"
            "\nOutput Constraints:\n"
            "- Each percentage must be an integer or have one decimal place (no '%' sign).\n"
            "- All percentages must sum to exactly 100 (±0.1 rounding error).\n"
            "- Percentages must be non-negative. Do not infer vote counts beyond the provided total_votes.\n"
            "\nInput Processing:\n"
            "- Read the single JSON object inside the BEGIN/END delimiters.\n"
            "- Locate every option whose `percentage` is set to \"Guess Here\".\n"
            "- Predict the voting share for each incomplete option drawing on the poll context.\n"
            "\nReturn Format:\n"
            "- Output exactly one JSON block.\n"
            "- Preserve the schema, key names, ordering, and spacing from the input JSON.\n"
            "- Replace only the \"percentage\": \"Guess Here\" fields with numeric predictions.\n"
            "- Do NOT add commentary, extra keys, or any text outside the JSON block.\n"
            "\nStep-by-Step Reasoning:\n"
            "- Think through the drivers of the vote distribution step-by-step before answering.\n"
            "- Only emit the final JSON once you are confident in the percentages."
        )
        prompt = (
            f"{instruction}\n\n"
            f"--- BEGIN POLL JSON ---\n{json.dumps(prompt_payload, ensure_ascii=False, indent=2)}\n"
            f"--- END POLL JSON ---\n\n"
            "Return the same JSON object with numeric predictions in place of each \"Guess Here\" placeholder."
        )
        return prompt


def _ensure_dict(record: Union[PollRecord, Mapping[str, object]]) -> Dict[str, object]:
    if isinstance(record, PollRecord):
        return record.to_dict()
    return dict(record)


def _try_parse_json(response_text: str) -> Optional[object]:
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        return None


# client wrappers
class OpenAIClient:
    provider = "gpt"

    def __init__(self, model_name: str, api_key: Optional[str] = None, base_url: Optional[str] = None):
        from openai import OpenAI

        self.model_name = model_name
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        response = self._client.responses.create(
            model=self.model_name,
            input=prompt,
            temperature=temperature,
        )
        return response.output[0].content[0].text


class GeminiClient:
    provider = "gemini"

    def __init__(self, model_name: str, api_key: Optional[str] = None):
        import google.generativeai as genai

        if api_key:
            genai.configure(api_key=api_key)
        self.model_name = model_name
        self._model = genai.GenerativeModel(model_name)

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        result = self._model.generate_content(
            prompt,
            generation_config={"temperature": temperature},
        )
        return result.text


class QwenClient:
    provider = "qwen"

    def __init__(self, model_name: str, api_key: Optional[str] = None):
        from dashscope import Generation

        self.model_name = model_name
        self._generation = Generation
        if api_key:
            import dashscope

            dashscope.api_key = api_key

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        response = self._generation.call(
            model=self.model_name,
            prompt=prompt,
            temperature=temperature,
        )
        return response.output["text"]


class DeepSeekClient:
    provider = "deepseek"

    def __init__(self, model_name: str, api_key: Optional[str] = None, base_url: str = "https://api.deepseek.com"):
        from openai import OpenAI

        self.model_name = model_name
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        response = self._client.responses.create(
            model=self.model_name,
            input=prompt,
            temperature=temperature,
        )
        return response.output[0].content[0].text


class LlamaClient:
    provider = "llama"

    def __init__(self, model_name: str, api_token: Optional[str] = None, max_new_tokens: int = 512):
        from huggingface_hub import InferenceClient

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._client = InferenceClient(model=model_name, token=api_token)

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        return self._client.text_generation(
            prompt,
            temperature=temperature,
            max_new_tokens=self.max_new_tokens,
        )


class ClaudeClient:
    provider = "claude"

    def __init__(self, model_name: str, api_key: Optional[str] = None):
        import anthropic

        self.model_name = model_name
        self._client = anthropic.Anthropic(api_key=api_key)

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        response = self._client.messages.create(
            model=self.model_name,
            max_tokens=4096,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text



class MistralClient:
    provider = "mistral"

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-v0.1",
        device_map: str = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 2048,
    ):
        self.model_name = model_name
        self._transformers_client = TransformersClient(
            model_name=model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            max_new_tokens=max_new_tokens,
        )

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        return self._transformers_client.generate(prompt, temperature)


class GemmaClient:
    provider = "gemma"

    def __init__(
        self,
        model_name: str = "google/gemma-2-9b",
        device: str = "cuda",
        max_new_tokens: int = 2048,
    ):
        from transformers import pipeline

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._pipe = pipeline(
            "text-generation",
            model=model_name,
            device=device,
        )

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        outputs = self._pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
        )
        return outputs[0]["generated_text"][len(prompt) :]


class QwenLocalClient:
    """Local Qwen client."""
    provider = "qwen"

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-32B",
        device_map: str = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 2048,
    ):
        self.model_name = model_name
        self._transformers_client = TransformersClient(
            model_name=model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            max_new_tokens=max_new_tokens,
        )

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        return self._transformers_client.generate(prompt, temperature)


class LlamaLocalClient:
    """Local Llama client."""
    provider = "llama"

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-13b",
        device_map: str = "auto",
        torch_dtype: str = "bfloat16",
        max_new_tokens: int = 2048,
    ):
        self.model_name = model_name
        self._transformers_client = TransformersClient(
            model_name=model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            max_new_tokens=max_new_tokens,
        )

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        return self._transformers_client.generate(prompt, temperature)
