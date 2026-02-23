import os
import yaml
import logging
import argparse
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

def generate_task_description(model_key, code_content):
    """
    Generates a Task Description using the specified model.
    """
    
    PROMPT = """You are a code analysis expert. Read the input code, extract its core logic, and generate a structured Task Description that will instruct another LLM to reproduce the algorithm.

**Purpose:** The description you generate serves as an executable specification—another system will use it to implement the algorithm and produce structured JSON output.

**Output Structure Requirements:**
- Use snake_case section headers (e.g., `## data_preprocessing`, `## objective_function`)
- Each section header maps to a JSON key; subsections become nested objects
- Use imperative language: specify what MUST be included, not just what exists
- Use consistent table formats with explicit column headers

**Sections** (include only applicable ones; starred sections are REQUIRED):

1. `overview`: Brief summary of the code's purpose and pipeline stages

2. `data_preprocessing`: One-time transformations before main computation (mark as initialization steps)

3. `variables`*: Categorize ALL variables in table format: | name | shape | dtype | role | description |
   - Create **distinct subsections** by role: `primal` (optimized), `dual` (multipliers), `constants`, `observations`

4. `objective_function`*: For optimization/inverse problems:
   - `full_expression`: Complete mathematical formulation
   - `data_fidelity_term`: Measurement consistency term with formula
   - `regularization_terms`: Each regularizer with name, formula, weight
   - `constraints`: Mathematical constraints (type: equality/inequality/bound + expression)

5. `initialization`: Table format: | variable | value | shape | source/justification |

6. `main_process`*:
   - `algorithm_framework`: Name the optimization method
   - `stopping_criterion`: Termination condition type and expression
   - `iteration_loop`: Numbered per-iteration update sequence
   - `validation_checks`: Post-hoc filters or sanity checks

7. `evaluation`*: Table of metrics: | metric | formula | reference_required | output_type |

8. `output_schema`*: Explicitly list all JSON keys that MUST appear in any implementation output, with expected types and structure. Map pipeline stages to their corresponding output sections.

**Format Guidelines:**
- Consolidate mathematical expressions before decomposing
- Clearly distinguish one-time setup from repeated iteration steps
- Cover all logical steps, edge cases, and constraints with precision
- Frame each section as requirements for reproduction, not just documentation
"""

    messages = [
        {"role": "system", "content": PROMPT},
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
        return content.strip()
        
    except Exception as e:
        logger.error(f"Failed to generate Task Description: {e}")
        return None

def main():
    base_dir = "/home/yjh/inverse_planning_eval"
    config2_path = os.path.join(base_dir, "config/config2.yaml")
    gt_index_path = os.path.join(base_dir, "config/gt_code_index.yaml")
    config_gen_path = os.path.join(base_dir, "config/config_golden_gen.yaml")
    
    # Load Generation Config to get model
    gen_config = load_yaml(config_gen_path)
    # You can also allow overriding the model via command line if needed, but for now we use the same as golden plan
    model_name = gen_config.get('golden_model', 'gpt-4') 
    
    # Define output directory
    output_dir = "/data/yjh/task_descriptions"
    
    # Load GT Code Index
    gt_index = load_yaml(gt_index_path)
    tasks = gt_index['tasks']
    
    # Setup LLM Env
    setup_llm_env(config2_path, model_name)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
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
        
        output_path = os.path.join(output_dir, f"{task_name}_description.md")
        if os.path.exists(output_path):
            logger.info(f"  Task Description already exists at {output_path}. Skipping.")
            continue
            
        with open(gt_code_path, 'r') as f:
            code_content = f.read()
            
        logger.info(f"  Generating Task Description using {model_name}...")
        description_content = generate_task_description(model_name, code_content)
        
        if description_content:
            # Save to file
            output_path = os.path.join(output_dir, f"{task_name}_description.md")
            with open(output_path, 'w') as f:
                f.write(description_content)
            
            logger.info(f"  Task Description saved to {output_path}")
        else:
            logger.error(f"  Generation failed for {task_name}")

if __name__ == "__main__":
    main()
