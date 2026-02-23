import argparse
import os
import yaml
import subprocess
import sys
from clean_up_code import clean_up
from tutorial_writer import tutorial_write_and_verified
import shutil
from utils import load_processed_functions,generate_tools, load_code_from_file
from make_up_question import generate_coding_questions
from openai import OpenAI
import json

def main():
    parser = argparse.ArgumentParser(description="Auto Tutorial Flow Pipeline")
    parser.add_argument('--pdf', default=None, help='Path to PDF')
    parser.add_argument('--paper_md', default=None, help='Path to paper md (optional, skips OCR if provided)')
    parser.add_argument('--markdown_output', default='output_2/', help='Path to generated paper md')
    parser.add_argument('--output_dir', default='output/', help='Output directory')
    parser.add_argument('--command', required=True, help='The command to run the code (e.g., "python train.py")')
    parser.add_argument('--code', required=True, help='Path to original Code file')
    parser.add_argument("--working_folder", required=True, help="The working folder for testing and temp files")
    parser.add_argument("--working_folder_file", required=True, help="The filename within working_folder to write cleaned code")
    parser.add_argument("--saving_folder", required=True, help="Folder to save historical clean-up code")
    parser.add_argument("--tutorial_name", required=True, help="Name of the output tutorial")
    parser.add_argument("--function_folder", required=True, help="Folder to store separated functions")
    parser.add_argument('--step', type=int, default=0, help='Start from specific step (0=Start, 1=OCR, 2=CodeClean/Test, 3=Tutorial, 4=Questions)')
    
    args = parser.parse_args()

    # 0. 环境准备
    # Clean up previous runs
    # if args.step <= 0:
    #     # for folder in [args.saving_folder, args.output_dir]:
    #     for folder in [args.output_dir]:
    #         if os.path.exists(folder):
    #             print(f"Cleaning up existing folder: {folder}")
    #             shutil.rmtree(folder)
    #         os.makedirs(folder, exist_ok=True)
    # else:
    #     # Ensure folders exist if resuming
    #     os.makedirs(args.saving_folder, exist_ok=True)
    #     os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.saving_folder, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    verification_utils_file = "verification_utils.py"
    working_folder_verification_utils_file = os.path.join(args.working_folder, "verification_utils.py")
    if not os.path.exists(working_folder_verification_utils_file):
        shutil.copy(verification_utils_file, working_folder_verification_utils_file)
    
    config = yaml.safe_load(open('config.yaml'))
    target_python_path = args.command.split()[0]


    # ==========================================================================
    # Step 1: PDF to Markdown (OCR)
    # ==========================================================================
    md_path = args.paper_md
    
    if args.step <= 1:
        print("\n=== Step 1: Processing PDF/Markdown ===")
        
        # 定义 Paddle 环境的 Python解释器路径
        # 优先从 config 读取，如果没有则使用默认值
        default_paddle_python = "/home/yjh/.conda/envs/paddle_env/bin/python"
        paddle_python_exec = config.get('ocr', {}).get('python_path', default_paddle_python)
        
        ocr_script_path = "run_ocr_tool.py" 
        
        if args.paper_md is None:
            if args.pdf is None:
                raise ValueError("Either --pdf or --paper_md must be provided.")
            # 1. 构建命令
            ocr_cmd = [
                paddle_python_exec, 
                ocr_script_path,
                "--pdf", args.pdf,
                "--output_dir", args.markdown_output
            ]
            
            # 2. 执行并捕获输出
            try:
                result = subprocess.run(ocr_cmd, check=True, capture_output=True, text=True)
                
                # 3. 解析输出以获取生成的 MD 文件路径
                # 我们在 run_ocr_tool.py 里输出了 "RESULT_PATH:xxx"
                for line in result.stdout.splitlines():
                    if line.startswith("RESULT_PATH:"):
                        md_path = line.split("RESULT_PATH:")[1].strip()
                        break
                
                if not md_path:
                    # 如果没抓到路径，手动推断一下作为备用方案
                    from pathlib import Path
                    filename = Path(args.pdf).stem + ".md"
                    md_path = os.path.join(args.markdown_output, filename)
                    print(f"  [Warning] Could not capture path from stdout, inferring: {md_path}")

            except subprocess.CalledProcessError as e:
                print(f"  [Error] OCR failed in paddle env.")
                print(f"  STDOUT: {e.stdout}")
                print(f"  STDERR: {e.stderr}")
                sys.exit(1)      
        
        print(f"  -> Markdown path: {md_path}")
    else:
        if md_path is None and args.pdf:
             from pathlib import Path
             filename = Path(args.pdf).stem + ".md"
             md_path = os.path.join(args.markdown_output, filename)
        print(f"Skipping Step 1. Using md_path: {md_path}")

    # ==========================================================================
    # Step 2: Code Cleaning & Separation
    # ==========================================================================
    if args.step <= 2:
        print("\n=== Step 2: Cleaning and Refactoring Code ===")
        
        potential_final_code = os.path.join(args.saving_folder, "final_code.py")
        if os.path.exists(potential_final_code):
             print(f"  [Info] Found existing final_code.py at {potential_final_code}. Skipping clean_up.")
             refactored_py_path = potential_final_code
             # Ensure the working folder file is updated with this final code
             shutil.copy(refactored_py_path, args.working_folder_file)
        else:
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
            # Ensure final_code.py exists in saving_folder (clean_up usually handles this or returns the path)
            # If clean_up returns a path, we might want to ensure it is saved as final_code.py if not already
            if not os.path.exists(potential_final_code) and os.path.exists(refactored_py_path):
                 shutil.copy(refactored_py_path, potential_final_code)
            refactored_py_path = potential_final_code
        
        print(f"  -> Refactored code saved to: {refactored_py_path}")
        

        # 实际上，这个路径是固定的： os.path.join(args.saving_folder, "final_code.py")

        # ==========================================================================
        # Step 2.1 & 2.2: Unified Unit Testing (Capture & Verify)
        # ==========================================================================
        print("\n=== Step 2.1: Running Unified Unit Testing (Data Capture & Coverage) ===")
        
        cmd_test = [
            sys.executable,  "uni_test.py",
            "--config_path", 'config.yaml',
            "--command", args.command,
            "--working_folder", args.working_folder,
            "--code_path", args.working_folder_file
        ]
        
        print(f"  [Exec] {' '.join(cmd_test)}")
        try:
            subprocess.run(cmd_test, check=True)
            print("  -> Unified testing finished successfully.")
        except subprocess.CalledProcessError as e:
            print(f"  [Error] uni_test.py failed with return code {e.returncode}")
            sys.exit(1) # 中断流程
    else:
        print("Skipping Step 2 (Cleaning & Testing).")


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

    # ==========================================================================
    # Step 3: Tutorial Writer
    # ==========================================================================
    if args.step <= 3:
        print("\n=== Step 3: Writing Tutorial ===")
        
        if not final_eval_function_lists:
            print("  [Warning] No validated functions found. Tutorial will be generated with empty function list.")

        tutorial_path = tutorial_write_and_verified(
            md_path, 
            config, 
            args.output_dir, 
            args.tutorial_name, 
            final_eval_function_lists, 
            eval_working_folder=args.working_folder,
            target_python_path = target_python_path
        )

        print(f"   Tutorial generated at: {tutorial_path}")
    else:
        print("Skipping Step 3 (Tutorial Writer).")
        tutorial_path = os.path.join(args.output_dir, f'tutorial_{args.tutorial_name}.md')

    # ==========================================================================
    # Step 4: Make Up Questions
    # ==========================================================================
    if args.step <= 4:
        print("\n=== Step 4: Make Up Questions ===")
        
        # Use tool_writer config as it is suitable for this task
        tool_writer_config = config['llm']['tool_writer']
        client = OpenAI(
            api_key=tool_writer_config['api_key'],
            base_url=tool_writer_config['base_url']
        )
        
        if not tutorial_path or not os.path.exists(tutorial_path):
            print(f"  [Error] Tutorial path not found: {tutorial_path}")
        else:
            tutorial_content = load_code_from_file(tutorial_path)
            
            if not final_eval_function_lists:
                print("  [Warning] No target functions found. Skipping question generation.")
            else:
                print(f"  Generating questions for: {final_eval_function_lists}")
                # Call the generator
                extraction = generate_coding_questions(
                    tutorial_content, 
                    final_eval_function_lists, 
                    client,
                    model_name=tool_writer_config['model']
                )
                
                if extraction:
                    question_output_path = os.path.join(args.output_dir, "coding_questions.json")
                    with open(question_output_path, 'w', encoding='utf-8') as f:
                        f.write(json.dumps(extraction.model_dump(), indent=4))
                    print(f"  -> Questions saved to: {question_output_path}")
                else:
                    print("  [Error] Failed to generate questions.")
    else:
        print("Skipping Step 4 (Make Up Questions).")

    print(f"\n✅ Pipeline complete!")

if __name__ == "__main__":
    main()