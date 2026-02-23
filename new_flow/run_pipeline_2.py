      
import argparse
import os
import yaml
import subprocess
import sys
from clean_up_code import clean_up
import shutil
from utils import load_processed_functions,generate_tools

def main():
    parser = argparse.ArgumentParser(description="Auto Tutorial Flow Pipeline")
    parser.add_argument('--output_dir', default='output/', help='Output directory')
    parser.add_argument('--command', required=True, help='The command to run the code (e.g., "python train.py")')
    parser.add_argument('--code', required=True, help='Path to original Code file')
    parser.add_argument("--working_folder", required=True, help="The working folder for testing and temp files")
    parser.add_argument("--working_folder_file", required=True, help="The filename within working_folder to write cleaned code")
    parser.add_argument("--saving_folder", required=True, help="Folder to save historical clean-up code")
    parser.add_argument("--refactored_py_path", default=None, help="Path to refactored python file (optional, skips refactoring if provided)")
    
    args = parser.parse_args()

    # 0. 环境准备
    os.makedirs(args.saving_folder, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.working_folder, exist_ok=True)

    # Create sandbox folder
    original_working_folder = args.working_folder
    sandbox_folder = os.path.normpath(args.working_folder) + "_sandbox"

    if os.path.exists(sandbox_folder):
        shutil.rmtree(sandbox_folder)

    shutil.copytree(args.working_folder, sandbox_folder)
    print(f"  -> Created sandbox: {sandbox_folder}")
    args.working_folder = sandbox_folder
    
    # Update working_folder_file to point to sandbox
    if os.path.isabs(args.working_folder_file):
        try:
            rel_path = os.path.relpath(args.working_folder_file, original_working_folder)
            if not rel_path.startswith(".."):
                args.working_folder_file = os.path.join(sandbox_folder, rel_path)
        except ValueError:
            pass
    print(f"  -> Updated working_folder_file to: {args.working_folder_file}")
    
    verification_utils_file = "verification_utils.py"
    working_folder_verification_utils_file = os.path.join(args.working_folder, "verification_utils.py")
    shutil.copy(verification_utils_file, working_folder_verification_utils_file)
    
    config = yaml.safe_load(open('config.yaml'))

  
    # ==========================================================================
    # Step 2: Code Cleaning & Separation
    # ==========================================================================
    print("\n=== Step 2: Cleaning and Refactoring Code ===")
    if args.refactored_py_path is None:
        # # clean_up 返回的是整理后的 python 文件路径
        refactored_py_path = clean_up(
            args.code, 
            args.command, 
            config, 
            args.output_dir, 
            args.working_folder, 
            args.working_folder_file, 
            args.saving_folder
        )
        
        # refactored_py_path = os.path.join(args.saving_folder, "final_code.py")
    else:
        print(f"     Refactored finished in advance!")
        refactored_py_path = args.refactored_py_path
    print(f"  -> Refactored code saved to: {refactored_py_path}")
    

    # 实际上，这个路径是固定的： os.path.join(args.saving_folder, "final_code.py")

    # ==========================================================================
    # Step 2.1: Run 'uni_test_temp.py' (Generate Data & Temp Tests)
    # ==========================================================================
    print("\n=== Step 2.1: Running Unit Test Generation (Temp) ===")
    
    cmd_temp = [
        sys.executable,  "uni_test.py",
        "--config_path", 'config.yaml',
        "--command", args.command,
        "--working_folder", args.working_folder,
        "--code_path", refactored_py_path
    ]
    
    print(f"  [Exec] {' '.join(cmd_temp)}")
    json_final_function_list_path = os.path.join(args.working_folder, "final_function_list.json")
    
    if not os.path.exists(json_final_function_list_path):
        try:
            # check=True 确保如果脚本报错，主程序停止
            subprocess.run(cmd_temp, check=True)
            print("  -> Step 2.1 finished successfully.")
        except subprocess.CalledProcessError as e:
            print(f"  [Error] uni_test_temp.py failed with return code {e.returncode}")
            sys.exit(1) # 中断流程
    else:
        print(f"  -> final_function_list.json already exists. Skip uni_test_temp.py.")



    # ==========================================================================
    # Step 2.3: Generate Tools
    # ==========================================================================
    final_eval_function_lists = load_processed_functions(working_folder=args.working_folder)
    output_dir = os.path.join(args.working_folder, config['uni_test_output_dir'])
    standard_data_dir = os.path.join(output_dir, "std_data")
    generate_tools( config=config,
                    standard_data_dir=standard_data_dir,
                    working_folder=args.working_folder,
                    final_eval_function_lists=final_eval_function_lists
                      )


    print(f"\n✅ Pipeline complete!")
    # print(f"   Tutorial generated at: {tutorial_path}")

if __name__ == "__main__":
    main()

    