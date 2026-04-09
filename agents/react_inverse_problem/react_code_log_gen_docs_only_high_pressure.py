
import argparse
import aiofiles
import asyncio
import json
import logging
import os
import sys
import shutil
import yaml
import glob
from datetime import datetime
# from code_dev_env import code_development_env
from code_dev_env_info import code_development_env
from scripts.async_llm import LLMsConfig
from scripts.async_llm import create_llm_instance
import workspace.InverseProb.workflows.template.operator as operator
from filter_question import create_starter_code, load_code_from_file

# # Set CUDA Device to 7 as requested
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

Hardware_info= """
Hardware Environment Specification:

Device: 1x NVIDIA GeForce RTX 3090 (CUDA 7)
VRAM: 24 GB (High Availability)
CUDA Driver: 535.230.02
"""

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def discover_questions_from_docs(working_folder):
    """
    Discovers questions by looking for generated_docs_*.json files in the working folder.
    Returns a list of question dictionaries with 'function_name', 'question_prompt', and 'provided_code'.
    """
    questions = []
    # Find all generated_docs_*.json
    doc_files = glob.glob(os.path.join(working_folder, "generated_docs_*.json"))
    
    # Sort to ensure deterministic order
    doc_files.sort()
    
    for doc_file in doc_files:
        filename = os.path.basename(doc_file)
        # generated_docs_{func_name}.json
        if not filename.startswith("generated_docs_") or not filename.endswith(".json"):
            continue
            
        func_name = filename[len("generated_docs_"):-len(".json")]
        
        # Load docs
        try:
            with open(doc_file, 'r', encoding='utf-8') as f:
                docs = json.load(f)
        except Exception as e:
            logging.error(f"Error loading doc file {doc_file}: {e}")
            continue
            
        # Construct prompt from docs
        explanation = docs.get("function_explanation", "")
        docstring = docs.get("function_docstring", "")
        usage = docs.get("usage_example", "")
        
        prompt = f"Function Name: {func_name}\n\n"
        prompt += f"Function Explanation:\n{explanation}\n\n"
        prompt += f"Function Docstring:\n{docstring}\n\n"
        prompt += f"Usage Example:\n{usage}\n"
        
        # Generate skeleton code
        gt_save_path = os.path.join(working_folder, f"gt_{func_name}.py")
        agent_code_path = os.path.join(working_folder, f"agent_{func_name}.py")
        
        source_code = None
        if os.path.exists(gt_save_path):
             source_code = load_code_from_file(gt_save_path)
        elif os.path.exists(agent_code_path):
             source_code = load_code_from_file(agent_code_path)
        else:
             logging.warning(f"Could not find code for {func_name} in {working_folder} (checked gt_*.py and agent_*.py)")
             continue

        skeleton_code = create_starter_code(source_code, func_name)
        
        questions.append({
            "function_name": func_name,
            "question_prompt": prompt,
            "provided_code": skeleton_code,
            "docs": docs 
        })
        
    return questions

def parse_args():
    parser = argparse.ArgumentParser(description="react_code")
    parser.add_argument(
        "--config_path",
        type=str,
        default="config/config.yaml",
        help="Path to the configuration file",
    )
    # Arguments that can be overridden by config
    parser.add_argument(
        "--model_name",
        type=str,
        default=None, # Changed to None to detect if user provided it
        help="Specifies the name of the model used for optimization tasks.",
    )
    parser.add_argument("--max_rounds", type=int, default=None, help="Max iteration rounds")
    parser.add_argument("--gpu_id", type=str, default="7", help="CUDA Device ID (default: 7)")
    
    # Optional: Filter to run only specific tasks
    parser.add_argument("--task", type=str, help="Run only a specific task by name")
    parser.add_argument("--sandbox_root", type=str, help="Override sandbox root directory")
    
    return parser.parse_args()

def setup_logging(log_dir, log_name):
    """Configures the logging system."""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    log_file_name = f"{log_name}_{timestamp}_0.7_math.log"
    # log_file_name = f"{log_name}_{timestamp}.log"  #表示温度！
    log_file_path = os.path.join(log_dir, log_file_name)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers to avoid duplicates if called multiple times
    if logger.handlers:
        logger.handlers = []
        
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file_path}")
    return logger

def run_task(task_config, global_config, args, custom_code_generate):
    task_name = task_config['name']
    python_path = task_config['python_path']
    working_folder_template = task_config['working_folder']
    # questions_file = task_config['questions_file'] # DISCARDED
    
    sandbox_root = global_config['global']['sandbox_root']
    
    # Create Task Sandbox Root
    # task_sandbox_dir = os.path.join(sandbox_root, task_name)
    # Use model name in path to avoid overwriting
    task_sandbox_dir = os.path.join(sandbox_root, args.model_name, task_name)
    os.makedirs(task_sandbox_dir, exist_ok=True)
    
    # Setup Logging for this task
    logger = setup_logging(os.path.join(task_sandbox_dir, "logs"), f"run_{task_name}")
    
    logging.info(f"Starting Task: {task_name}")
    logging.info(f"Python Path: {python_path}")
    logging.info(f"Working Folder Template: {working_folder_template}")
    # logging.info(f"Questions File: (Discarded, using generated_docs from working folder)")
    
    try:
        # Load questions from docs instead of questions_file
        all_questions = discover_questions_from_docs(working_folder_template)
        
        logging.info(f"Discovered {len(all_questions)} questions from docs.")
        if len(all_questions) == 0:
            logging.warning("No questions found. Exiting task.")
            return

    except Exception as e:
        logging.error(f"Error processing questions from docs: {e}")
        return

    # Initialize Report Statistics
    report_stats = {
        "task_name": task_name,
        "total_questions": 0,
        "processed_questions": 0,
        "results": {
            "SUCCESS": 0,
            "Wronganswer": 0,
            "FAIL": 0
        },
        "details": []
    }
    
    # Filter out skipped questions first to get accurate count if needed, or just iterate
    valid_questions = [q for q in all_questions if q['function_name'] != "Loss_angle_diff"]
    report_stats["total_questions"] = len(valid_questions)

    for i, q in enumerate(all_questions):
        f_name = q['function_name']
            
        logging.info(f"--- Processing Question {i+1}: {f_name} ---")
        report_stats["processed_questions"] += 1
        
        # Create Per-Problem Sandbox
        problem_sandbox_dir = os.path.join(task_sandbox_dir, f"sandbox_{f_name}")
        
        # Copy template to problem sandbox
        if os.path.exists(problem_sandbox_dir):
            shutil.rmtree(problem_sandbox_dir) # Clean start
        shutil.copytree(working_folder_template, problem_sandbox_dir, dirs_exist_ok=True)
        
        working_folder_location = problem_sandbox_dir
        
        prompt = q['question_prompt']
        code = q['provided_code']
        
        # Note: We already used docs to generate prompt, so we don't need to load them again for docstring,
        # but the original code re-loaded them to check for existence in sandbox.
        # Here we assume they exist since we found them in template and copied template.
        
        ## set up working file path:
        gt_save_path = os.path.join(working_folder_location, f"gt_{f_name}.py")
        agent_code_path = os.path.join(working_folder_location, f"agent_{f_name}.py")
        
        # If agent code exists (from template?), use it as start, else ensure we have a place to write
        if os.path.exists(gt_save_path):
            working_file_path = agent_code_path
        else:
            if os.path.exists(agent_code_path):
                with open(agent_code_path, 'r', encoding='utf-8', errors='ignore') as f:
                    gt_code = f.read()
                with open(gt_save_path, 'w', encoding='utf-8', errors='ignore') as f:
                    f.write(gt_code)
            working_file_path = agent_code_path

        logging.info(f"Problem Description: {prompt}...") 
        
        # UPDATED PROMPT STRUCTURE
        CI_agent_prompt = f"""
        You are a Senior Computational Physicist specialized in Non-Linear Spectroscopy (CARS).
You are provided with a code development task to implement a rigorous Forward Model for N2 CARS thermometry.
You have access to these hardware:{Hardware_info}
        The target function is {f_name}
        The problem is {prompt}
        And the provided code is {code}. Using the packages imported and other helper function defined in the provided code.
        Your answer should contain the completed function and the other imported packages and provided helper function.
        """
        
        error_history = []
        final_status = "FAIL" # Default assumption
        
        for round_idx in range(args.max_rounds):
            logging.info(f"--- Round {round_idx} ---")
            logging.debug(f"Sending prompt: {CI_agent_prompt}")
            
            solution = asyncio.run(custom_code_generate(problem=CI_agent_prompt, instruction=""))
            
            # Log the full thinking process if available (from raw_response)
            if isinstance(solution, dict) and 'raw_response' in solution:
                logging.info(f"Model Thinking & Response:\n{solution['raw_response']}")
            else:
                logging.info(f"Model Response Code:\n{solution['response']}")

            current_results, current_code = code_development_env(
                solution['response'], 
                working_file_path, 
                None, # command is built inside using python_path 
                working_folder_location, 
                f_name,
                python_path=python_path
            )
            
            if current_results["status"] == "SUCCESS":
                current_status = current_results["status"]
                logging.info(f"Question status {current_status}!")
                logging.info(f"answer code {current_code}")
                final_status = current_status
                break
                
            elif current_results["status"] == "FAIL" or current_results["status"] == "Wronganswer":
                logging.error(f"Test failed. Error: {current_results['error']}")
                final_status = current_results["status"]
                
                # Construct error feedback prompt
                formatted_error_history = "\n".join(
                    [f"Attempt {idx+1} Error: {err}" for idx, err in enumerate(error_history)]
                )
                
                CI_agent_prompt = f"""
            You are a expert in computational imaging expert and you're provided with code development task tailed for developing scientific functions.
            You should write code to complete the function and your code would be written into a certain file and tested to check whether you produced the same results like the provided expert code
            You have access to these hardware:{Hardware_info}
            The target function is {f_name}
            The problem is {prompt}
            And the provided code is {code}. using the packages imported and other helper function defined in the provided code.
            The previous trail code is {solution['response']}
            The previous test error of the trail code is {current_results["error"]}
            You should revised the previous trail code with the provided test error info!
            Here is the full history of errors from all past attempts:
            {formatted_error_history}
            You should revised the previous trail code, taking into account the **latest test error** and the **full error history** to avoid repeating mistakes.
            """
                error_history.append(current_results["error"])
        
        # Write final code
        with open(working_file_path, 'w', encoding='utf-8') as f:
            f.write(current_code)
            
        # Update Statistics
        report_stats["results"][final_status] += 1
        report_stats["details"].append({
            "function_name": f_name,
            "status": final_status,
            "rounds_used": round_idx + 1
        })

    # Generate Report
    generate_report(task_sandbox_dir, report_stats)

def generate_report(task_dir, stats):

    """Generates a summary report for the task with a timestamped filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(task_dir, f"task_report_{timestamp}.json")

    # """Generates a summary report for the task."""
    # report_path = os.path.join(task_dir, "task_report.json")
    
    total = stats["processed_questions"]
    success = stats["results"]["SUCCESS"]
    success_rate = (success / total * 100) if total > 0 else 0
    
    stats["success_rate"] = f"{success_rate:.2f}%"
    stats["timestamp"] = datetime.now().isoformat()
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=4)
        
    logging.info(f"Task Completed. Report saved to {report_path}")
    logging.info(f"Summary: Total={total}, Success={success}, Rate={success_rate:.2f}%")


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config_path)
    
    if args.sandbox_root:
        if 'global' not in config:
            config['global'] = {}
        config['global']['sandbox_root'] = args.sandbox_root

    # Apply default settings from config if not provided in args
    default_settings = config.get('global', {}).get('default_settings', {})
    
    if args.model_name is None:
        args.model_name = default_settings.get('model_name', "deepseek-r1-250528")
        
    if args.max_rounds is None:
        args.max_rounds = default_settings.get('max_rounds', 10)
    
    logging.info(f"Using Model: {args.model_name}")
    logging.info(f"Max Rounds: {args.max_rounds}")
    
    models_config = LLMsConfig.default()
    models_config = models_config.get(args.model_name)
    custom_code_generate = operator.CustomCodeGenerate(create_llm_instance(models_config))
    
    if args.task:
        # Run specific task
        task_config = next((t for t in config['tasks'] if t['name'] == args.task), None)
        if task_config:
            run_task(task_config, config, args, custom_code_generate)
        else:
            print(f"Task {args.task} not found in config.")
    else:
        # Run all tasks
        for task_config in config['tasks']:
            run_task(task_config, config, args, custom_code_generate)
