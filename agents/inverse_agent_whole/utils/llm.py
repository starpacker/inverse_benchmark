import os
import json
import time
import re
import yaml
import openai
from typing import Dict, Any, Optional, List, Union
from config import settings

class LLMClient:
    def __init__(self, config_path: str = None):
        self.model_configs = {}
        self._load_configs(config_path)

    def _load_configs(self, config_path: str = None):
        """Load model configurations from YAML."""
        # Try standard locations if not provided
        paths = [
            config_path,
            "config/config2.yaml",
            os.path.join(settings.BASE_DIR, "config/config2.yaml"),
            os.path.join(os.path.dirname(settings.BASE_DIR), "config/config2.yaml"),
            "/home/yjh/inverse_agent_whole/config/config2.yaml"  # Absolute path fallback
        ]
        
        for p in paths:
            if p and os.path.exists(p):
                try:
                    with open(p, 'r') as f:
                        data = yaml.safe_load(f)
                        if data and "models" in data:
                            self.model_configs = data["models"]
                            # print(f"[LLMClient] Loaded configs from {p}")
                            return
                except Exception as e:
                    print(f"[LLMClient] Error loading config {p}: {e}")

    def _get_client(self, model_name: str):
        """Get OpenAI client for a specific model."""
        api_key = None
        base_url = None

        # 1. Check specific config
        if model_name in self.model_configs:
            conf = self.model_configs[model_name]
            api_key = conf.get("api_key")
            base_url = conf.get("base_url")
        
        # 2. Wildcard match
        if not api_key:
            import fnmatch
            for k, v in self.model_configs.items():
                if "*" in k and fnmatch.fnmatch(model_name, k):
                    api_key = v.get("api_key")
                    base_url = v.get("base_url")
                    break

        api_key = self._resolve_env_value(api_key)
        base_url = self._resolve_env_value(base_url)
        
        # 3. Environment variables
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")
            base_url = os.environ.get("OPENAI_BASE_URL")

        if api_key:
            return openai.OpenAI(api_key=api_key, base_url=base_url)
        
        return None

    @staticmethod
    def _resolve_env_value(value):
        if not isinstance(value, str):
            return value
        s = value.strip()
        if s.startswith("${") and s.endswith("}") and len(s) > 3:
            env_key = s[2:-1].strip()
            return os.environ.get(env_key)
        return value

    def call(self, system_prompt: str, user_prompt: str, model: str = "gpt-4o", json_mode: bool = False, max_retries: int = settings.MAX_LLM_RETRIES) -> Dict[str, Any]:
        """
        Calls the LLM and returns a dictionary with full metadata.
        Returns:
            {
                "content": str (parsed JSON or raw text),
                "thinking": str (if available),
                "raw_content": str (original text before parsing),
                "model": str,
                "usage": dict
            }
        """
        client = self._get_client(model)
        if not client:
            print(f"[LLMClient] Error: No client configuration for {model}")
            return {"content": {}, "error": "No Client"} if json_mode else {"content": "", "error": "No Client"}

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": 0.2
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(**kwargs)
                message = response.choices[0].message
                content = message.content or ""
                
                # Extract thinking if present in specific fields (DeepSeek/o1) or tags
                # Try standard OpenAI compatible fields first
                thinking = getattr(message, "reasoning_content", "") or ""
                
                # Try accessing via extra_fields if using a proxy that maps it there
                if not thinking and hasattr(message, "extra_fields"):
                     thinking = message.extra_fields.get("reasoning_content", "")
                
                # Fallback: Check for <thinking> tags in content
                if not thinking:
                    match = re.search(r"<thinking>(.*?)</thinking>", content, re.DOTALL)
                    if match:
                        thinking = match.group(1).strip()
                        # Clean content by removing thinking block to avoid interference
                        content = re.sub(r"<thinking>.*?</thinking>", "", content, flags=re.DOTALL).strip()
                
                # Fallback: Check for <reasoning> tags (Gemini sometimes uses this or similar)
                if not thinking:
                    match = re.search(r"<reasoning>(.*?)</reasoning>", content, re.DOTALL)
                    if match:
                        thinking = match.group(1).strip()
                        content = re.sub(r"<reasoning>.*?</reasoning>", "", content, flags=re.DOTALL).strip()

                result = {
                    "content": content,
                    "thinking": thinking,
                    "raw_content": content,
                    "model": model,
                    "usage": dict(response.usage) if response.usage else {}
                }

                if json_mode:
                    try:
                        parsed = self.safe_json_load(content)
                        result["content"] = parsed
                        return result
                    except json.JSONDecodeError as e:
                        print(f"[LLMClient] JSON Parse Error (Attempt {attempt+1}): {e}")
                        if attempt < max_retries - 1:
                            messages.append({"role": "assistant", "content": content})
                            messages.append({"role": "user", "content": f"ERROR: The previous response was not valid JSON. Error: {e}. Please fix and return ONLY valid JSON."})
                            continue
                
                return result
            
            except Exception as e:
                print(f"[LLMClient] Request Error (Attempt {attempt+1}): {e}")
                time.sleep(2)

        return {"content": {}, "error": "Max Retries"} if json_mode else {"content": "", "error": "Max Retries"}

    @staticmethod
    def safe_json_load(json_str: str) -> Dict[str, Any]:
        """Robust JSON parser."""
        try:
            # Strip markdown fences
            cleaned = json_str.replace("```json", "").replace("```", "").strip()
            # Strip thinking tags if they wrap the JSON
            cleaned = re.sub(r"<thinking>.*?</thinking>", "", cleaned, flags=re.DOTALL).strip()
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Naive repair for common latex escape issues
            try:
                fixed = cleaned.replace("\\", "\\\\")
                return json.loads(fixed)
            except:
                pass
            raise

# Global instance
llm_client = LLMClient()
