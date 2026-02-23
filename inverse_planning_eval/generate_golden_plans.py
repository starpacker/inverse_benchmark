
import os
import yaml
import json
import logging
from litellm import completion

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Configuration Loading ---
def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def setup_llm_env(config_path, model_key):
    """
    Sets up the environment variables for the chosen model from config2.yaml.
    """
    config = load_yaml(config_path)
    if model_key not in config['models']:
        raise ValueError(f"Model {model_key} not found in {config_path}")
    
    model_conf = config['models'][model_key]
    os.environ["OPENAI_API_KEY"] = model_conf['api_key']
    os.environ["OPENAI_BASE_URL"] = model_conf['base_url']
    logger.info(f"Configured LLM Environment for {model_key} at {model_conf['base_url']}")
    return model_key

def generate_golden_plan(model_key, code_content):
    """
    Generates a Golden Plan using the specified model.
    We use a very detailed prompt to ensure high quality.
    """
    
    # Prompt designed to elicit a high-quality, comprehensive test plan (Golden Standard)
    GOLDEN_PROMPT = """
You are an Expert Algorithm Architect.
Your task is to reverse-engineer a "Golden Plan" (Algorithmic Logic) from the provided code.
The output must be a strict JSON object describing the algorithm with a complete schema.

Output JSON Schema (strict JSON, no markdown):
{{
  "algorithm_name": "string",

  "data_preprocessing": {{
    "description": "string (high-level purpose of preprocessing)",
    "steps": [
      {{
        "step_order": 1,
        "step_name": "string",
        "operation": "string (mathematical or algorithmic description)",
        "mathematical_formula": "string (LaTeX compatible, optional)",
        "input_data": ["list of inputs"],
        "output_data": ["list of outputs"],
        "assumptions": ["list of assumptions or conditions"]
      }}
    ]
  }},

  "objective_function": {{
    "full_expression": "string (complete mathematical objective)",
    "data_fidelity_term": "string (mathematical formula)",
    "regularization_term": "string (mathematical formula)",
    "constraints": ["list of explicit constraints, if any"]
  }},

  "variables": {{
    "primal": ["list of primal variables"],
    "dual": ["list of dual / auxiliary variables"],
    "constants": ["list of constants / hyperparameters"],
    "observations": ["list of observed data variables"]
  }},

  "initialization": [
    {{
      "variable": "string",
      "value": "string description",
      "shape": "string description",
      "source": "string (e.g., zeros, random, from data, analytical)"
    }}
  ],

  "main_inverse_process": {{
    "algorithm_framework": "string (e.g., GD, ADMM, PDHG, ISTA)",
    "iteration_loop": [
      {{
        "step_order": 1,
        "step_name": "string",
        "step_type": "string (e.g., primal update / dual update / proximal step)",
        "mathematical_formula": "string (LaTeX compatible)",
        "operator_requirements": ["list of required operators"],
        "input_variables": ["list of variables"],
        "output_variables": ["list of variables"],
        "computational_notes": "string (e.g., FFT-based, linear solve, proximal closed-form)"
      }}
    ],
    "stopping_criterion": {{
      "type": "string (e.g., max_iter / tolerance / relative_change)",
      "expression": "string (mathematical or logical condition)"
    }}
  }},

  "evaluation": {{
    "description": "string (purpose of evaluation)",
    "metrics": [
      {{
        "metric_name": "string (e.g., MSE, PSNR, SSIM, residual norm)",
        "definition": "string (mathematical definition)",
        "reference_data": "string (e.g., ground truth, observations)",
        "output": "string (scalar / vector / curve)"
      }}
    ],
    "post_processing": [
      {{
        "operation": "string (e.g., normalization, thresholding, visualization)",
        "input_variables": ["list of variables"],
        "output_variables": ["list of variables"]
      }}
    ]
  }}
}}
"""

    messages = [
        {"role": "system", "content": GOLDEN_PROMPT},
        {"role": "user", "content": f"Here is the Ground Truth Code:\n\n{code_content}"}
    ]
    
    try:
        # Use openai/ prefix for litellm with custom providers
        response = completion(
            model=f"openai/{model_key}",
            messages=messages,
            temperature=0.0 # Deterministic
        )
        content = response.choices[0].message.content
        
        # Strip markdown formatting if present
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        
        # Handle thinking/reasoning blocks which some models (like Claude-Thinking) might output
        # Look for the first '{' and the last '}'
        start_idx = content.find('{')
        end_idx = content.rfind('}')
        
        if start_idx != -1 and end_idx != -1:
            content = content[start_idx : end_idx+1]
        else:
            logger.warning("Could not find JSON object in response.")
            
        return content.strip()
        
    except Exception as e:
        logger.error(f"Failed to generate Golden Plan: {e}")
        return None

def main():
    base_dir = "/home/yjh/inverse_planning_eval"
    config2_path = os.path.join(base_dir, "config/config2.yaml")
    gt_index_path = os.path.join(base_dir, "config/gt_code_index.yaml")
    config_gen_path = os.path.join(base_dir, "config/config_golden_gen.yaml")
    
    # Load Generation Config
    gen_config = load_yaml(config_gen_path)
    golden_model = gen_config['golden_model']
    output_dir = gen_config['output_dir']
    
    # Load GT Code Index
    gt_index = load_yaml(gt_index_path)
    tasks = gt_index['tasks']
    
    # Setup LLM Env
    setup_llm_env(config2_path, golden_model)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    for task_entry in tasks:
        task_name = task_entry['name']
        
        # Only process successful tasks
        # if task_entry.get('status') != 'success':
        #     continue
            
        logger.info(f"Processing Task: {task_name}")
        gt_code_path = task_entry['gt_code_path']
        
        if not os.path.exists(gt_code_path):
            logger.error(f"GT Code not found at {gt_code_path}. Skipping.")
            continue
      
        output_path = os.path.join(output_dir, f"{task_name}_golden_plan.json")
        if os.path.exists(output_path):
            logger.info(f"  Golden Plan already exists at {output_path}. Skipping.")
            continue

        with open(gt_code_path, 'r') as f:
            code_content = f.read()
            
        logger.info(f"  Generating Golden Plan using {golden_model}...")
        golden_plan_json_str = generate_golden_plan(golden_model, code_content)
        
        if golden_plan_json_str:
            # Validate JSON
            try:
                json_obj = json.loads(golden_plan_json_str)
                
                # Save to file
                output_path = os.path.join(output_dir, f"{task_name}_golden_plan.json")
                with open(output_path, 'w') as f:
                    json.dump(json_obj, f, indent=2)
                
                logger.info(f"  Golden Plan saved to {output_path}")
                
            except json.JSONDecodeError as e:
                logger.error(f"  Generated content is not valid JSON: {e}")
                logger.debug(f"  Content preview: {golden_plan_json_str[:200]}...")
        else:
            logger.error(f"  Generation failed for {task_name}")

if __name__ == "__main__":
    main()
