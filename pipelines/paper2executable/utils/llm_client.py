"""
LLM Client

Unified interface for interacting with Large Language Models.
Supports OpenAI-compatible APIs (Claude, GPT, DeepSeek, etc.)
"""

import json
import time
from typing import Any, Optional, Union

import openai
from openai import OpenAI, AsyncOpenAI
from rich.console import Console

from utils.logging_utils import get_logger

logger = get_logger(__name__)
console = Console()


class LLMClient:
    """Unified client for LLM interactions."""

    def __init__(self, config: dict):
        self.config = config
        self.llm_config = config.get("llm", {})
        self.models = self.llm_config.get("models", {})
        self.default_model = self.llm_config.get("default_model", "cds/Claude-4.6-opus")

        # Initialize clients for each configured model
        self.clients: dict[str, OpenAI] = {}
        for model_name, cfg in self.models.items():
            if cfg.get("api_type") == "openai":
                self.clients[model_name] = OpenAI(
                    api_key=cfg.get("api_key"),
                    base_url=cfg.get("base_url"),
                    timeout=cfg.get("timeout", 60),
                )

    def completion(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful AI assistant.",
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> str:
        """
        Get a completion from the LLM.

        Args:
            prompt: User prompt content
            system_prompt: System instruction
            model: Model identifier (defaults to config default)
            temperature: Sampling temperature
            max_tokens: Max output tokens
            json_mode: Whether to enforce JSON output format

        Returns:
            The text content of the response.
        """
        model = model or self.default_model
        if model not in self.clients:
            logger.warning(
                f"Model '{model}' not configured. Falling back to '{self.default_model}'"
            )
            model = self.default_model

        client = self.clients[model]
        response_format = {"type": "json_object"} if json_mode else None

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        start_time = time.time()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )
            content = response.choices[0].message.content
            elapsed = time.time() - start_time
            logger.debug(f"LLM completion ({model}) took {elapsed:.2f}s")
            return content

        except Exception as e:
            logger.error(f"LLM completion error ({model}): {e}")
            raise

    def structured_completion(
        self,
        prompt: str,
        output_schema: dict,
        system_prompt: str = "You are a helpful data extraction assistant.",
        model: Optional[str] = None,
    ) -> dict:
        """
        Get a structured JSON response guaranteed to match a schema.
        Note: Simple implementation using prompt engineering + json mode.
        """
        schema_str = json.dumps(output_schema, indent=2)
        enhanced_system_prompt = (
            f"{system_prompt}\n\n"
            f"You must output valid JSON matching this schema:\n{schema_str}"
        )

        content = self.completion(
            prompt,
            system_prompt=enhanced_system_prompt,
            model=model,
            json_mode=True,
        )

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON response: {content[:100]}...")
            # Simple retry logic could go here
            return {}
