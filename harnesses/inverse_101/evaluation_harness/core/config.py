"""Configuration dataclasses for the evaluation harness."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class LLMConfig:
    """OpenAI-compatible LLM endpoint configuration."""
    model: str = "gpt-4o"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    temperature: float = 0.2
    max_tokens: int = 32768

    @classmethod
    def from_yaml(cls, model_name: str, yaml_path: Optional[Path] = None) -> "LLMConfig":
        """Load LLM config from config_llm.yaml by model name."""
        if yaml_path is None:
            candidates = [
                Path("config_llm.yaml"),
                Path(__file__).resolve().parent.parent.parent / "config_llm.yaml",
            ]
            for c in candidates:
                if c.exists():
                    yaml_path = c
                    break
        if yaml_path is None or not yaml_path.exists():
            raise FileNotFoundError(
                "config_llm.yaml not found. Provide --base-url and --api-key, "
                "or create config_llm.yaml."
            )
        with open(yaml_path, encoding="utf-8") as f:
            configs = yaml.safe_load(f) or {}
        if model_name not in configs:
            available = list(configs.keys())
            raise KeyError(
                f"Model {model_name!r} not found in {yaml_path}. Available: {available}"
            )
        entry = configs[model_name]
        return cls(
            model=model_name,
            base_url=entry.get("base_url", "https://api.openai.com/v1"),
            api_key=entry.get("api_key", os.environ.get("OPENAI_API_KEY", "")),
            temperature=entry.get("temperature", 0.2),
            max_tokens=entry.get("max_tokens", 32768),
        )


E2E_LEVELS = ("L1", "L2", "L3")


@dataclass
class TaskConfig:
    """Which task to evaluate and in what mode."""
    task_name: str = ""
    task_dir: Path = field(default_factory=Path)
    mode: str = "end_to_end"
    target_function: Optional[str] = None
    level: str = "L1"


@dataclass
class RunConfig:
    """Top-level run configuration."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    max_iterations: int = 20
    docker_image: str = "imaging101-sandbox"
    timeout_seconds: int = 600
    output_dir: Path = field(default_factory=lambda: Path("results"))
    log_file: Optional[Path] = None
    framework: str = "react"
