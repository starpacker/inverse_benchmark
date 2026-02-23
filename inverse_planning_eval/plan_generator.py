
import os
import yaml
import logging
from litellm import completion

logger = logging.getLogger(__name__)

PLAN_GEN_SYSTEM_INSTRUCTION = """
### Output Requirement 
Output a valid JSON object following this structure: 
{ 
  "algorithm_name": "string", 
  "initialization": ["list of strings describing init"], 
  "iteration_loop": [ 
    { 
      "step_order": int, 
      "step_name": "string", 
      "mathematical_description": "string (Latex compatible formula preferred)", 
      "required_operators": ["list of operator names"], 
      "inputs": ["list of variable names"], 
      "outputs": ["list of updated variable names"] 
    } 
  ] 
} 
"""

class PlanGenerator:
    def __init__(self, config_path, prompt_text):
        self.config = self._load_yaml(config_path)
        self.prompt_text = prompt_text
        
    def _load_yaml(self, path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)
            
    def generate_description(self, model_key, code_content, task_name=None, use_cache=False):
        """
        Generates a Task Description using the optimized prompt.
        """
        # Check for pre-generated task description
        if use_cache and task_name:
            description_path = f"/data/yjh/task_descriptions/{task_name}_description.md"
            if os.path.exists(description_path):
                logger.info(f"Loading pre-generated task description for {task_name} from {description_path}")
                try:
                    with open(description_path, 'r') as f:
                        return f.read()
                except Exception as e:
                    logger.error(f"Failed to read pre-generated description: {e}")
        
        # Optimized prompt is a system instruction.
        messages = [
            {"role": "system", "content": self.prompt_text},
            {"role": "user", "content": f"Here is the input code:\n\n{code_content}"}
        ]
        
        return self._call_llm(model_key, messages)

    def generate_plan(self, model_key, description):
        """
        Generates a Test Plan from the Task Description using a fixed instruction.
        """
        messages = [
            {"role": "system", "content": PLAN_GEN_SYSTEM_INSTRUCTION},
            {"role": "user", "content": f"Here is the Task Description:\n\n{description}"}
        ]
        return self._call_llm(model_key, messages)

    def _call_llm(self, model_key, messages):
        # Get model config
        if model_key not in self.config['models']:
            raise ValueError(f"Model {model_key} not found in config")
            
        model_conf = self.config['models'][model_key]
        
        # Set env vars temporarily
        os.environ["OPENAI_API_KEY"] = model_conf['api_key']
        os.environ["OPENAI_BASE_URL"] = model_conf['base_url']
        
        try:
            response = completion(
                model=f"openai/{model_key}", 
                messages=messages,
                temperature=0.0,
                timeout=1200 # Increase timeout to 20 minutes for thinking models
            )
            return response.choices[0].message.content
        except Exception as e:
            import traceback
            logger.error(f"Generation failed for {model_key}: {e}")
            logger.error(traceback.format_exc())
            return None
