import os
import json
import yaml
import sys
from openai import OpenAI
import re
import argparse
import shutil
from utils import run_command_streaming, get_full_response,_extract_code_from_markdown,validate_test_reliability
from utils import inject_double_decorators, DependencyExtractor,analyze_code_structure

# ==============================================================================
# Unified Testing Workflow
# ==============================================================================

def generate_and_run_tests(config_path, command, working_folder, code_path, target_functions=None):
    """
    Unified workflow that:
    1. Injects decorators to capture ground truth data (if not already captured).
    2. Runs the instrumented code to populate `std_data/`.
    3. Generates unit tests for each function using the captured data.
    4. Runs and validates these tests.
    """
    
    config = yaml.safe_load(open(config_path))
    
    if target_functions is None:
        target_functions = [
            "load_and_preprocess_data",
            "forward_operator", 
            "run_inversion", 
            "evaluate_results"
        ]
        
    client = OpenAI(api_key=config['llm']['code']['api_key'], base_url=config['llm']['code']['base_url'])
    model = config['llm']['code']['model']

    output_dir = os.path.join(working_folder, config['uni_test_output_dir'])
    os.makedirs(output_dir, exist_ok=True)
    
    std_data_dir = os.path.join(output_dir, "std_data")
    meta_data_path = os.path.join(output_dir, "meta_data.json")
    
    # --------------------------------------------------------------------------
    # Phase 1: Data Capture (Run only if needed)
    # --------------------------------------------------------------------------
    print("\n[Phase 1] Checking/Generating Ground Truth Data...")
    
    if os.path.exists(std_data_dir):
        shutil.rmtree(std_data_dir)
    os.makedirs(std_data_dir, exist_ok=True)

    verification_utils_file = "verification_utils.py"
    working_folder_verification_utils_file = os.path.join(output_dir, "verification_utils.py")
    shutil.copy(verification_utils_file, working_folder_verification_utils_file)

    # Load Reference Code
    with open(code_path, 'r', encoding='utf-8') as f:
        reference_code = f.read()

    print("  -> Injecting decorators...")

    try:
        gen_data_code = inject_double_decorators(
            original_code=reference_code,
            output_dir=std_data_dir.replace("\\", "/"), 
            metadata_path=meta_data_path,
            target_functions=target_functions
        )
    except Exception as e:
        print(f"Error generating injection code: {e}")
        return
    
    gen_script_path = os.path.join(working_folder, "gen_std_data.py")
    with open(gen_script_path, 'w', encoding='utf-8') as f:
        f.write(gen_data_code)
    
    print(f"  -> Saved instrumented script to {gen_script_path}")

    # Execute gen_std_data.py
    print(f"  -> Running Data Capture with command context...")
    
    cmd_parts = command.split()
    new_cmd_parts = []
    replaced = False
    for part in cmd_parts:
        if part.endswith('.py') and not replaced:
            new_cmd_parts.append("gen_std_data.py")
            replaced = True
        else:
            new_cmd_parts.append(part)
    
    if not replaced:
        print("  [Error] Failed to inject gen_std_data.py into command. Aborting.")
        sys.exit(1)
        
    gen_command = " ".join(new_cmd_parts)
    print(f"  -> Executing: {gen_command}")

    log_path = os.path.join(output_dir, "data_gen_log.txt")
    return_code, _ = run_command_streaming(
        gen_command,
        cwd=working_folder,
        log_file_path=log_path,
        timeout=None 
    )

    if return_code != 0:
        print(f"  [Error] Data generation failed with return code {return_code}!")
        print(f"  -> Check log file for details: {log_path}")
        return
    else:
        print("  -> Data generation successful.")
        data_exists = True

    if not data_exists:
        print("  [Error] No data captured. Aborting testing phase.")
        return

    # --------------------------------------------------------------------------
    # Phase 2: Save Function List (for Tool Generation)
    # --------------------------------------------------------------------------
    final_eval_function_lists = []
    for func in target_functions:
        # Pattern matching for data files
        func_pattern = re.compile(rf"^standard_data_(?:parent_function_.*_)?{re.escape(func)}\.pkl$")
        found = False
        for fname in os.listdir(std_data_dir):
            if func_pattern.match(fname):
                found = True
                break
        
        if found:
            final_eval_function_lists.append(func)
        else:
            print(f"  [Warning] No data found for {func}")

    output_list_file = os.path.join(working_folder, "final_function_list.json")
    try:
        with open(output_list_file, 'w', encoding='utf-8') as f:
            json.dump(final_eval_function_lists, f, ensure_ascii=False, indent=4)
        print(f"  -> Saved active function list to: {output_list_file}")
    except Exception as e:
        print(f"  [Error] Failed to save final function list: {e}")

    # --------------------------------------------------------------------------
    # Phase 3: Unit Test Generation & Verification
    # --------------------------------------------------------------------------
    print("\n[Phase 3] Generating and Running Unit Tests...")
    
    # Reload reference code in case we skipped Phase 1
    with open(code_path, 'r', encoding='utf-8') as f:
        reference_code = f.read()

    # Load gen_std_data code (needed for prompt context)
    # If we skipped Phase 1, we try to read it, or regenerate it in memory if missing
    gen_script_path = os.path.join(working_folder, "gen_std_data.py")
    if os.path.exists(gen_script_path):
        with open(gen_script_path, 'r', encoding='utf-8') as f:
            gen_data_code = f.read()
    else:
        # Regenerate in memory if file missing (edge case)
        gen_data_code = inject_double_decorators(
                original_code=reference_code,
                output_dir=std_data_dir.replace("\\", "/"), 
                metadata_path=meta_data_path,
                target_functions=target_functions
        )

    results = {}
    intermediate_root = os.path.join(working_folder, ".intermediate")
    os.makedirs(intermediate_root, exist_ok=True)

    print("\n[Prep] Extracting Standard Evaluation Logic...")
    eval_extractor = DependencyExtractor(reference_code)
    try:
        eval_extractor.get_dependencies("evaluate_results")
        standard_eval_code = eval_extractor.generate_code()
    except Exception as e:
        print(f"  [Warning] Could not extract 'evaluate_results': {e}")
        standard_eval_code = """"""

    for func in target_functions:
        if func not in final_eval_function_lists:
            results[func] = "SKIPPED_NO_DATA"
            continue

        print(f"\n--- Processing: {func} ---")
        is_optimization_task = (func == "run_inversion")
        func_intermediate_dir = os.path.join(intermediate_root, func)
        os.makedirs(func_intermediate_dir, exist_ok=True)

        # Collect data files
        func_pattern = re.compile(rf"^standard_data_(?:parent_function_.*_)?{re.escape(func)}\.pkl$")
        found_data_files = []
        for fname in os.listdir(std_data_dir):
            if func_pattern.match(fname):
                full_path = os.path.join(std_data_dir, fname).replace("\\", "/")
                found_data_files.append(full_path)
        data_path_str = str(found_data_files)

        # Extract Dependencies
        reference_extractor = DependencyExtractor(reference_code)
        gen_extractor = DependencyExtractor(gen_data_code)
        try:
            reference_extractor.get_dependencies(func)
            gen_extractor.get_dependencies(func)
            standard_code = reference_extractor.generate_code()
            gen_code = gen_extractor.generate_code()
        except Exception as e:
            print(f"  [Error] Failed to extract dependencies for {func}: {e}")
            results[func] = "ERROR_EXTRACTION"
            continue

        # Save Agent Code
        agent_file = f"agent_{func}.py"
        agent_path = os.path.join(working_folder, agent_file)
        os.makedirs(os.path.dirname(agent_path), exist_ok=True)
        with open(agent_path, 'w', encoding='utf-8') as f:
            f.write(standard_code)
        
        # Test Generation Loop
        max_retries = 15
        retry_count = 0
        passed = False  
        test_code = ""
        test_res_stderr = "" 
        
        while retry_count <= max_retries:
            attempt_dir = os.path.join(func_intermediate_dir, f"attempt_{retry_count}")
            os.makedirs(attempt_dir, exist_ok=True)

            if retry_count == 0:
                user_msg = f"Generate test_{func}.py now."
            else:
                if not test_code.strip():
                    user_msg = (
                        "The previous response did NOT contain any Python code block.\n"
                        f"Generate test_{func}.py now."
                    )
                else:
                    user_msg = (
                        f"The previous code had error:\n{test_res_stderr}\n\n"
                        f"Fix all issues and return the full, corrected test_{func}.py code."
                    )

            if is_optimization_task:
                msgs = [
                    {"role": "system", "content": config['prompts']['gen_inverse_test_script'].format(
                        func_name=func,
                        data_paths=data_path_str,
                        standard_code=standard_code,    
                        gen_data_code=gen_code,
                        eval_code=standard_eval_code,
                    )},
                    {"role": "user", "content": user_msg}
                ]
            else:
                msgs = [
                    {"role": "system", "content": config['prompts']['gen_test_script'].format(
                        func_name=func,
                        data_paths=data_path_str,
                        standard_code=standard_code,    
                        gen_data_code=gen_code,
                    )},
                    {"role": "user", "content": user_msg}
                ]
            
            raw_resp = get_full_response(client, model, msgs)
            test_code = _extract_code_from_markdown(raw_resp)
            
            # Save Test Script
            test_filename = f"test_{func}.py"
            test_filepath = os.path.join(working_folder, test_filename)
            with open(test_filepath, 'w', encoding='utf-8') as f:
                f.write(test_code if test_code else "# FAILED TO EXTRACT CODE")

            print(f"  -> Generated {test_filename} (attempt {retry_count})")

            # Run Test
            # Reset agent code to standard
            with open(agent_path, 'w', encoding='utf-8') as f:
                f.write(standard_code)

            target_python_path = command.split()[0]
            test_cmd = f"{target_python_path} {test_filename}"
            execution_log_path = os.path.join(attempt_dir, "execution_log.txt")

            return_code, _ = run_command_streaming(
                test_cmd,
                cwd=working_folder,
                log_file_path=execution_log_path,
                timeout=None 
            )

            test_output_content = ""
            if os.path.exists(execution_log_path):
                with open(execution_log_path, 'r', encoding='utf-8', errors='ignore') as f:
                    test_output_content = f.read()

            if return_code == 0:
                print("  -> Initial Run PASSED.")
                
                if is_optimization_task:
                    print("    -> Optimization task: Skipping strict Meta-Test.")
                    passed = True
                    results[func] = "PASS"
                    break

                # Meta-Test
                is_reliable, reason = validate_test_reliability(
                    working_folder, func, 
                    test_filename, standard_code,
                    target_python_path=target_python_path
                )
                
                if is_reliable:
                    print("  -> RELIABILITY CHECK PASSED.")
                    results[func] = "PASS"
                    passed = True
                    break 
                else:
                    print(f"  -> RELIABILITY CHECK FAILED: {reason}")
                    test_res_stderr = f"Reliability check failed: {reason}"
                    retry_count += 1
            else:
                print("  -> FAILED (Runtime/Assertion Error)")
                max_chars = 2000
                error_tail = test_output_content[-max_chars:] if len(test_output_content) > max_chars else test_output_content
                test_res_stderr = f"Execution failed. Log tail:\n...\n{error_tail}"
                results[func] = "FAIL"
                retry_count += 1

        final_status = "PASS" if passed else "FAIL"
        with open(os.path.join(func_intermediate_dir, "final_status.txt"), 'w') as f:
            f.write(final_status + "\n")

    # Final Report
    print("\n================ TEST REPORT ================")
    print(json.dumps(results, indent=2))
    with open(os.path.join(output_dir, "final_test_report.json"), 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--command", type=str, required=True)
    parser.add_argument("--working_folder", type=str, required=True)
    parser.add_argument("--code_path", type=str, required=True)

    args = parser.parse_args()

    if not os.path.exists(args.config_path):
        raise FileNotFoundError(f"Config file not found: {args.config_path}")
    os.makedirs(args.working_folder, exist_ok=True)
    target_functions = analyze_code_structure(args.code_path)
    generate_and_run_tests(
        config_path=args.config_path,
        command=args.command,
        working_folder=args.working_folder,
        code_path=args.code_path,
        target_functions=target_functions
    )
