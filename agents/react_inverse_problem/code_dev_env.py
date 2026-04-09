import subprocess
import sys
import os
import threading

def run_command_streaming(command, cwd, log_file_path, timeout=None):
    """
    执行命令，实时流式输出到控制台和日志文件，支持超时控制和编码容错。
    """
    print(f"  [Exec] Streaming output to console and {log_file_path}...")
    
    # 准备日志文件
    f_log = open(log_file_path, 'w', encoding='utf-8')
    
    # 启动进程
    # errors='replace': 关键修改，遇到非UTF-8字符时用?代替，防止程序崩溃
    process = subprocess.Popen(
        command,
        cwd=cwd,
        shell=True,
        stdout=subprocess.PIPE,     
        stderr=subprocess.STDOUT,   # 将 stderr 合并到 stdout
        text=True,                  
        bufsize=1,                  
        encoding='utf-8',
        errors='replace'            
    )

    # 定义一个读取流的线程函数
    def stream_reader(proc, log_file):
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                # sys.stdout.flush() # 可选：如果希望控制台刷新更频繁
                log_file.write(line)
                log_file.flush()     # 确保实时写入文件
        except Exception as e:
            # 进程被杀掉或文件关闭时可能会有IO错误，忽略即可
            pass

    # 启动子线程读取输出
    t = threading.Thread(target=stream_reader, args=(process, f_log))
    t.daemon = True # 设置为守护线程，防止主程序退出时卡住
    t.start()
    
    result_code = -1
    status_msg = ""

    try:
        # 主线程阻塞等待，直到超时或进程结束
        # 这里的 wait 是完全受控的，不会被 stdout 阻塞
        process.wait(timeout=timeout)
        result_code = process.returncode
        status_msg = "Finished"
        
    except subprocess.TimeoutExpired:
        print(f"\n  [Error] Process timed out after {timeout} seconds. Killing...")
        process.kill() # 强杀进程
        result_code = -1
        status_msg = "TIMEOUT_EXPIRED"
        
    except Exception as e:
        print(f"\n  [Error] Exception during execution: {e}")
        process.kill()
        result_code = -1
        status_msg = str(e)
        
    finally:
        # 等待读取线程结束（通常很快，因为 stdout 管道已断开）
        t.join(timeout=1.0)
        f_log.close()

    return result_code, status_msg

def code_development_env(llm_input, working_file_path, command, cwd_location, function_name, python_path=None):
    
    if python_path:
        command = f"{python_path} test_{function_name}.py"
    else:
        command = f"python test_{function_name}.py"

    execution_log_path = os.path.join(cwd_location, f"execution_log_{function_name}.txt")
    with open(working_file_path, 'w', encoding='utf-8') as f:
                f.write(llm_input)

    return_code, _ = run_command_streaming(
                command,
                cwd=cwd_location,
                log_file_path=execution_log_path,
                timeout=None 
            )
    test_output_content = ""
    if os.path.exists(execution_log_path):
        with open(execution_log_path, 'r', encoding='utf-8', errors='ignore') as f:
            test_output_content = f.read()

    if return_code == 0:
        results = {"status": "SUCCESS", "error": None}
    else:
        # if f"Execution of {function_name}" in test_output_content:
            
        #     max_chars = 2000
        #     error_tail = test_output_content[-max_chars:] if len(test_output_content) > max_chars else test_output_content
            
        #     print(f"Error Tail: {error_tail[-200:]}...")
        #     results =  {"status": "FAIL", "error":error_tail}
        # else:
        #     print("wrong answer")
        #     results = {"status": "Wronganswer","error": None}
        if "TEST WRONG ANSWER" in test_output_content:
            print("test_output_content", test_output_content)
            print("wrong answer")
            max_chars = 2000
            error_tail = test_output_content[-max_chars:] if len(test_output_content) > max_chars else test_output_content
            results = {"status": "Wronganswer","error": error_tail}
        else:
            print("test_output_content", test_output_content)
            max_chars = 2000
            error_tail = test_output_content[-max_chars:] if len(test_output_content) > max_chars else test_output_content
            
            print(f"Error Tail: {error_tail[-500:]}...")
            results =  {"status": "FAIL", "error":error_tail}
    return results, llm_input

    # 1. Check if the script itself crashed (e.g., syntax error, module not found)
    # The traceback for this will be in stderr.
    if exec_result.returncode != 0:
        # print("\n--- error1: wrong running code STDERR (Error Traceback) ---")
        # print(exec_result.stdout)
        result = {"status": "FAIL", "error":exec_result.stderr}


    # 2. Check if the tests passed
    # We look for the specific success message in the standard output.
    elif "✅ PASS" in exec_result.stdout:
        # print("--- TEST PASSED ---")
        # print("\n--- STDOUT ---")
        # print(exec_result.stdout)
        result = {"status": "SUCCESS", "error": None}

    # 3. If it ran (returncode 0) but didn't pass, it's a test failure
    else:
        # print("--- TEST FAILED (Mismatch or Exception) ---")
        # print("The script ran, but the outputs did not match the ground truth.")
        
        # # The "mismatch info" you want is already captured in stdout/stderr
        # print("\n--- STDOUT (Failure Details) ---")
        # print(exec_result.stdout)
        result = {"status": "FAIL", "error":exec_result.stderr}
    # print(result)
    # assert 1==0
    with open(working_file_path, 'w', encoding='utf-8') as f:
        f.write('')
    return result, llm_input