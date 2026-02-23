
import os
import argparse
import yaml
import random
import textgrad as tg
from textgrad import Variable, BlackboxLLM
from textgrad.optimizer import TextualGradientDescent
from textgrad.engine import LiteLLMEngine
from litellm import completion
import logging
import concurrent.futures
import threading

# Setup logging
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_LOG_PATH = os.path.join(BASE_DIR, "optimization_artifacts.log")
EXECUTION_LOG_PATH = os.path.join(BASE_DIR, "optimization_execution.log")
PROMPTS_DIR = os.path.join(BASE_DIR, "optimized_prompts")
os.makedirs(PROMPTS_DIR, exist_ok=True)

# Configure logging to file and console
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(EXECUTION_LOG_PATH, mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SafeLiteLLMEngine(LiteLLMEngine):
    def __init__(self, model_string, api_key=None, base_url=None, **kwargs):
        super().__init__(model_string, **kwargs)
        self.api_key = api_key
        self.base_url = base_url

    def lite_llm_generate(self, content, system_prompt=None, **kwargs) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        return completion(model=self.model_string,
                          messages=messages,
                          api_key=self.api_key,
                          base_url=self.base_url,
                          **kwargs)['choices'][0]['message']['content']

# --- Configuration Loading ---
def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def get_llm_config(config_path, model_key):
    """
    Returns the api_key and base_url for the chosen model from config2.yaml.
    """
    config = load_yaml(config_path)
    if model_key not in config['models']:
        raise ValueError(f"Model {model_key} not found in {config_path}")
    
    model_conf = config['models'][model_key]
    return model_conf['api_key'], model_conf['base_url']

# --- Data Loading ---
def load_dataset(task_config_path):
    """
    Loads (Code, Golden Plan) pairs.
    Uses config_task.yaml for GT Code paths.
    Uses hardcoded logic or search to find Golden Plans.
    """
    task_config = load_yaml(task_config_path)['task_io_paths']
    dataset = []
    
    # Mapping based on previous exploration
    # Adjust base path if necessary. Assuming script is run from /home/yjh
    golden_plan_base = "/data/yjh/golden_plans"
    
    for task_name, paths in task_config.items():
        gt_code_path = paths.get('gt_code')
        if not gt_code_path or not os.path.exists(gt_code_path):
            logger.warning(f"GT Code not found for {task_name}: {gt_code_path}")
            continue
            
        with open(gt_code_path, 'r') as f:
            code_content = f.read()
            
        # Try to find golden plan
        plan_path = os.path.join(golden_plan_base, f"{task_name}_golden_plan.json")
        if not os.path.exists(plan_path):
             logger.warning(f"Golden Plan not found for {task_name}: {plan_path}")
             continue
             
        with open(plan_path, 'r') as f:
            # Plan might be a JSON object or string. 
            # If it's the 'used_plan.json', it likely contains the plan structure.
            # We need a string representation for the prompt.
            import json
            plan_data = json.load(f)
            # Convert to formatted string if it's a dict
            if isinstance(plan_data, (dict, list)):
                golden_plan_str = json.dumps(plan_data, indent=2)
            else:
                golden_plan_str = str(plan_data)
        
        dataset.append({
            'task_name': task_name,
            'code': code_content,
            'golden_plan': golden_plan_str
        })
        
    logger.info(f"Loaded {len(dataset)} items into dataset.")
    return dataset

# --- Core Components ---

# 1. Prompts
INITIAL_PROMPT_TEXT = """
You are a code analysis expert. Please read the input code, extract its core logic, and generate a EXTREMELY detailed Task Description.
The description should be comprehensive, covering all logical steps, edge cases, and constraints, serving as a perfect guide for testers to write test plans.
Do not worry about information leakage; precision is the top priority.
"""

PLAN_COMPARISON_PROMPT_TEMPLATE = """
You are a strict Test Architect. Please compare the following two test plans:

1. **Golden Plan (Standard Answer):** {golden_plan}
2. **Generated Plan (Model Generated):** (Provided below)

And the basis for generation — **Task Description:** {task_description}

Please execute the following evaluation steps:
1. **Difference Analysis:** Point out the missing or incorrect steps in the Generated Plan compared to the Golden Plan.
2. **Attribution (Blame Assignment):**
    - Check if the Task Description contained the necessary information to avoid this error?
    - If the Description **already contained** the necessary information, but the Test LLM still got it wrong, explicitly state this is a reasoning error and **DO NOT** suggest modifying the Description.
    - If the Description **was indeed missing** key logic or constraints leading to this error, point out exactly what information needs to be added to the Description.

**Output Feedback:** Please provide specific modification suggestions for the Task Description (if needed).
"""

PLAN_GEN_SYSTEM_INSTRUCTION = """
### Output Requirement 
Output a valid JSON object following this structure: 
{{ 
  "algorithm_name": "string", 
  "initialization": ["list of strings describing init"], 
  "iteration_loop": [ 
    {{ 
      "step_order": int, 
      "step_name": "string", 
      "mathematical_description": "string (Latex compatible formula preferred)", 
      "required_operators": ["list of operator names"], 
      "inputs": ["list of variable names"], 
      "outputs": ["list of updated variable names"] 
    }} 
  ] 
}} 
"""

# --- Main Pipeline ---

# Logging helper
log_lock = threading.Lock()
def log_artifact(name, content):
    with log_lock:
        with open(ARTIFACTS_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*20} {name} {'='*20}\n")
            f.write(str(content))
            f.write("\n")

def process_single_item(args):
    item, model_name, desc_engines, test_engines, config_path, optimizer_model = args
    try:
        gt_code = item['code']
        golden_plan = item['golden_plan']
        
        # Get engines
        desc_engine = desc_engines[model_name]
        test_engine = test_engines[model_name]
        
        # --- Layer 1: Code to Description ---
        # desc_engine uses the shared system_prompt_var
        description = desc_engine(tg.Variable(gt_code, requires_grad=False, role_description="input_code"))
        description.set_role_description(f"generated_task_description_by_{model_name}")
        
        log_artifact(f"Generated Description ({item['task_name']} - {model_name})", description.value)
        
        # --- Layer 2: Description to Test Plan ---
        generated_plan = test_engine(description)
        generated_plan.set_role_description(f"test_plan_by_{model_name}")
        
        log_artifact(f"Generated Plan ({item['task_name']} - {model_name})", generated_plan.value)
        
        # --- Loss Calculation: Plan Comparison ---
        # Create Judge Engine with SafeLiteLLMEngine
        api_key, base_url = get_llm_config(config_path, optimizer_model)
        judge_engine = SafeLiteLLMEngine(
            model_string=f"openai/{optimizer_model}",
            api_key=api_key,
            base_url=base_url
        )
        
        comparison_loss_fn = tg.TextLoss(
            eval_system_prompt=PLAN_COMPARISON_PROMPT_TEMPLATE.format(
                golden_plan=golden_plan,
                task_description=description.value
            ),
            engine=judge_engine
        )
        
        comparison_loss = comparison_loss_fn(generated_plan)
        
        log_artifact(f"Comparison Feedback ({item['task_name']} - {model_name})", comparison_loss.value)
        return comparison_loss
        
    except Exception as e:
        logger.error(f"Error processing {item['task_name']} with {model_name}: {e}")
        return None

def train_step(batch_data, optimizer, desc_engines, test_engines, system_prompt_var, config_path, optimizer_model, models):
    optimizer.zero_grad()
    total_loss_nodes = []
    
    logger.info(f"Processing batch of size {len(batch_data)}")
    
    tasks = []
    for item in batch_data:
        for model_name in models:
            tasks.append((item, model_name, desc_engines, test_engines, config_path, optimizer_model))
            
    # Run in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(process_single_item, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res:
                total_loss_nodes.append(res)
    
    logger.info(f"Computed {len(total_loss_nodes)} feedback signals. Backpropagating...")
    
    for loss_node in total_loss_nodes:
        loss_node.backward()
        
    optimizer.step()
    log_artifact("Updated System Prompt", system_prompt_var.value)
    return system_prompt_var.value

def main():
    parser = argparse.ArgumentParser(description="TextGrad Prompt Optimization")
    parser.add_argument("--models", type=str, default="gemini-3-pro-preview,qwen3-max,deepseek-v3.2,gpt-5.2-thinking,claude-opus-4-5-20251101-thinking", help="Comma-separated list of models to use")
    parser.add_argument("--optimizer_model", type=str, default="gemini-3-pro-preview", help="Model used for optimization/backward pass")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(',')]
    optimizer_model = args.optimizer_model

    # Paths
    base_dir = "/home/yjh/inverse_planning_eval"
    config_path = os.path.join(base_dir, "config/config2.yaml")
    task_config_path = os.path.join(base_dir, "config/config_task.yaml")
    
    # Verify optimizer model
    api_key, base_url = get_llm_config(config_path, optimizer_model)
    optimizer_engine = SafeLiteLLMEngine(
        model_string=f"openai/{optimizer_model}",
        api_key=api_key,
        base_url=base_url
    )
    
    # Set backward engine globally for TextGrad
    tg.set_backward_engine(optimizer_engine)
    
    # Parameter to optimize
    system_prompt_v2 = Variable(
        INITIAL_PROMPT_TEXT.strip(),
        requires_grad=True,
        role_description="system_prompt_for_description_generation"
    )

    # Initialize Engines for all models
    desc_engines = {}
    test_engines = {}
    
    for model in models:
        api_key, base_url = get_llm_config(config_path, model)
        eng = SafeLiteLLMEngine(
            model_string=f"openai/{model}",
            api_key=api_key,
            base_url=base_url
        )
        
        desc_engines[model] = BlackboxLLM(engine=eng, system_prompt=system_prompt_v2)
        test_engines[model] = BlackboxLLM(engine=eng, system_prompt=PLAN_GEN_SYSTEM_INSTRUCTION)
        
    # Data
    dataset = load_dataset(task_config_path)
    if not dataset:
        logger.error("No data loaded. Exiting.")
        return
        
    logger.info(f"Loaded {len(dataset)} tasks. Using full dataset.")

    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    
    # Optimizer
    # We use the optimizer_engine
    optimizer = TextualGradientDescent(
        parameters=[system_prompt_v2],
        engine=optimizer_engine,
        constraints=["Keep the System Prompt concise", "Do not add hardcoded rules for specific Tasks"]
    )
    
    logger.info(f"Initial System Prompt:\n{system_prompt_v2.value}\n{'='*50}")
    log_artifact("Initial System Prompt", system_prompt_v2.value)
    
    def get_batches(dataset, size):
        dataset_shuffled = dataset[:] # Copy
        random.shuffle(dataset_shuffled)
        for i in range(0, len(dataset_shuffled), size):
            yield dataset_shuffled[i:i + size]
            
    for epoch in range(EPOCHS):
        logger.info(f"--- Epoch {epoch+1} ---")
        for batch_idx, batch in enumerate(get_batches(dataset, BATCH_SIZE)):
            logger.info(f"Processing Batch {batch_idx+1}...")
            new_prompt = train_step(
                batch, optimizer, 
                desc_engines, test_engines, 
                system_prompt_v2,
                config_path, optimizer_model, models
            )

            # Save prompt after each UPDATE (or each batch? or each epoch?)
            # User said "save 10 prompts", implying 1 per epoch.
            # But we update per batch.
            # We'll save the latest prompt at the end of the epoch.
            
            logger.info(f"Updated System Prompt (after Batch {batch_idx+1}):")
            print(new_prompt)
            print("-" * 30)
            
        # Save prompt after each full epoch
        prompt_path = os.path.join(PROMPTS_DIR, f"prompt_epoch_{epoch+1:02d}.txt")
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(system_prompt_v2.value)
        logger.info(f"Saved optimized prompt for epoch {epoch+1} to {prompt_path}")
            
    logger.info(f"{'='*50}\nFinal Optimized System Prompt:\n{system_prompt_v2.value}")
    log_artifact("Final Optimized System Prompt", system_prompt_v2.value)

if __name__ == "__main__":
    main()
