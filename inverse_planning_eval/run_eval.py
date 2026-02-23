
import os
# Set HF Mirror to avoid timeout in restricted network environments
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import yaml
import logging
import argparse
import json
from evaluation.similarity import SimilarityEvaluator
from evaluation.elo import EloEvaluator
from plan_generator import PlanGenerator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_dataset(task_config_path):
    config_data = load_yaml(task_config_path)
    tasks_list = config_data.get('tasks', [])
    dataset = {}
    
    golden_plan_base = "/data/yjh/inverse_planning_eval/data/golden_plans"
    if not os.path.exists(golden_plan_base):
        # Fallback to source if not in data dir
        golden_plan_base = "/home/yjh/inverse_planning_eval/data/golden_plans"
    
    for task_entry in tasks_list:
        task_name = task_entry.get('name')
        gt_code_path = task_entry.get('gt_code_path')
        
        if not gt_code_path or not os.path.exists(gt_code_path):
            logger.warning(f"GT Code not found for {task_name}")
            continue
            
        with open(gt_code_path, 'r') as f:
            code_content = f.read()
            
        plan_path = os.path.join(golden_plan_base, f"{task_name}_golden_plan.json")
        if not os.path.exists(plan_path):
             logger.warning(f"Golden Plan not found for {task_name}")
             continue
             
        with open(plan_path, 'r') as f:
            golden_plan = json.load(f)
            
        dataset[task_name] = {
            'code': code_content,
            'golden_plan': golden_plan
        }
    return dataset

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Planning Evaluation")
    parser.add_argument("--prompt_file", type=str, default = "final_prompt.md")
    parser.add_argument("--output_file", type=str, default = "eval_results.json")
    args = parser.parse_args()

    # Configuration
    base_dir = "/data/yjh/inverse_planning_eval"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)
    # Ensure config files are loaded from the source repo, not the output data dir
    source_dir = "/home/yjh/inverse_planning_eval"
    
    config_path = os.path.join(source_dir, "config/config2.yaml")
    task_config_path = os.path.join(source_dir, "config/gt_code_index.yaml")
    
    # Models to evaluate
    MODELS = [
        "gemini-3-pro-preview",
        "qwen3-max",
        "deepseek-v3.2",
        "gpt-5.2-thinking",
        "claude-opus-4-5-20251101-thinking",
        "kimi-k2-250905",
        "glm-4.7"
    ]
    
    # Load Prompt
    if os.path.exists(args.prompt_file):
        with open(args.prompt_file, 'r') as f:
            OPTIMIZED_PROMPT = f.read()
            print(f"using prompt:{OPTIMIZED_PROMPT}")
    else:
        # Fallback or error. For now, use a placeholder if file missing.
        # Ideally, we should have extracted it from the optimization log.
        # Since I cannot automate that extraction reliably without parsing logs in previous steps,
        # I will assume the user has put the prompt in 'optimized_prompt.txt' OR 
        # I will use the one I saw in the logs in previous turn.
        # "You are a code analysis expert. Analyze the input code and generate an EXTREMELY detailed task description..."
        logger.warning(f"Prompt file {args.prompt_file} not found. Using hardcoded optimized prompt from logs.")
        OPTIMIZED_PROMPT = """You are a code analysis expert. Analyze the input code and generate an EXTREMELY detailed task description that serves as a rigorous, mathematically complete blueprint. The description must:

        1. Formalize all key equations in LaTeX, including derivations of noise models, operator norms, and optimization steps.
        2. Specify hyperparameters with theoretical justification and include pseudocode with complexity analysis.
        3. Define testable assertions for each component and generate edge-case test templates.
        4. Describe dynamic monitoring requirements (convergence criteria, logging hooks).
        5. Resolve ambiguities by citing sources or proving uniqueness.
        6. State performance-accuracy tradeoffs and computational bottlenecks.
        7. Include stress-test scenarios and connections to relevant literature.

        Precision is paramount - treat this as an executable specification. Omit no critical details.

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
        }}"""

    # Initialize Components
    generator = PlanGenerator(config_path, OPTIMIZED_PROMPT)
    sim_evaluator = SimilarityEvaluator()
    
    # Judge Model Config for Voting ELO
    # Primary: claude-opus-4-5-20251101-thinking, gemini-3-pro-preview, deepseek-v3.2
    # Reserves: qwen3-max, gpt-5.2-thinking
    
    primary_judges = ["claude-opus-4-5-20251101-thinking", "gemini-3-pro-preview", "deepseek-v3.2"]
    reserve_judges = ["qwen3-max", "gpt-5.2-thinking"]
    
    config_data = load_yaml(config_path)
    
    # Collect API configs for all potential judges
    api_configs = {}
    all_judges = primary_judges + reserve_judges
    
    for j_key in all_judges:
        if j_key in config_data['models']:
            conf = config_data['models'][j_key]
            # Determine provider prefix if not explicit. 
            # Config has 'api_type', usually maps to litellm provider.
            # Here we assume most are openai-compatible as per config structure.
            prefix = "openai/" # Default
            if "deepseek" in j_key and "r1" not in j_key: # v3.2 might be different? Config says "openai" api_type
                 pass 
            
            api_configs[j_key] = {
                "api_key": conf['api_key'],
                "base_url": conf['base_url'],
                "provider_prefix": prefix
            }
        else:
            logger.warning(f"Judge model {j_key} not found in config. Skipping.")
    
    judge_config = {
        "primary": primary_judges,
        "reserves": reserve_judges,
        "api_configs": api_configs
    }

    elo_evaluator = EloEvaluator(judge_config=judge_config)
    
    # Load Data
    dataset = load_dataset(task_config_path)
    if not dataset:
        logger.error("No data loaded")
        return

    final_results = {
        "similarity_scores": {},
        "elo_ratings": {},
        "generated_plans": {}
    }

    # 1. Generate Plans and Run Eval for ALL Models
    # Archive Directory for Plans
    plan_archive_dir = os.path.join(base_dir, "archived_plans")
    os.makedirs(plan_archive_dir, exist_ok=True)
    
    # Ensure logs directory exists
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    all_gen_plans = {} # {task: {model: plan}}
    
    for task_name, data in dataset.items():
        # Setup Logger for this task
        task_log_file = os.path.join(log_dir, f"{task_name}_eval.log")
        file_handler = logging.FileHandler(task_log_file, mode='w')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        
        try:
            logger.info(f"Processing Task: {task_name}")
            all_gen_plans[task_name] = {}
            all_gen_plans[task_name]["Golden"] = json.dumps(data['golden_plan']) # Include Golden in the mix
            
            # Calculate Similarity for each model against Golden immediately
            if task_name not in final_results["similarity_scores"]:
                final_results["similarity_scores"][task_name] = {}
                
            for model in MODELS:
                logger.info(f"  Generating/Loading plan for {model}...")
                # Pass task_name to support loading pre-generated descriptions
                # Step 1: Generate Description
                description = generator.generate_description(model, data['code'], task_name=task_name)
                if not description:
                    logger.error(f"Failed to generate description for {model} on {task_name}")
                    continue

                # Step 2: Generate Plan
                plan_text = generator.generate_plan(model, description)
                
                if not plan_text:
                    logger.error(f"Failed to generate plan for {model} on {task_name}")
                    continue
                
                # Save to archive (overwrite)
                plan_filename = f"{model}_{task_name}_plan.txt"
                plan_filepath = os.path.join(plan_archive_dir, plan_filename)
                with open(plan_filepath, 'w') as f:
                    f.write(plan_text)
                
                all_gen_plans[task_name][model] = plan_text
                
                # Compute Similarity vs Golden
                sim_metrics = sim_evaluator.evaluate(plan_text, data['golden_plan'])
                final_results["similarity_scores"][task_name][model] = sim_metrics
            
            # Quick Save: Similarity Results
            similarity_output_file = args.output_file.replace(".json", "_similarity.json")
            with open(similarity_output_file, 'w') as f:
                json.dump(final_results, f, indent=2)
            logger.info(f"Similarity evaluation complete. Results saved to {similarity_output_file}")
            
            # Run ELO Tournament
            logger.info(f"Running ELO Tournament for Task: {task_name}")
            # Context is the code snippet
            task_desc = f"Code for task {task_name}:\n{dataset[task_name]['code'][:3000]}..."
            
            ratings, history = elo_evaluator.run_tournament(all_gen_plans[task_name], task_desc)
            
            final_results["elo_ratings"][task_name] = {
                "final_ratings": ratings,
                "match_history": history
            }
            
        finally:
            # Remove handler to avoid duplicate logs in next iteration
            root_logger.removeHandler(file_handler)
            file_handler.close()

    # Save Results
    with open(args.output_file, 'w') as f:
        json.dump(final_results, f, indent=2)
        
    logger.info(f"Evaluation complete. Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
