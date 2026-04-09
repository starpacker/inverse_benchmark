import os
import json

def count_questions_in_json_files(root_dir="."):
    total_questions = 0
    file_count = 0
    target_filename = "coding_questions.json"

    print(f"Scanning for '{target_filename}' in: {os.path.abspath(root_dir)}\n")

    # Walk through the directory tree
    for dirpath, _, filenames in os.walk(root_dir):
        if target_filename in filenames:
            file_path = os.path.join(dirpath, target_filename)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Check if 'questions' key exists and is a list
                    if "questions" in data and isinstance(data["questions"], list):
                        num_questions = len(data["questions"])
                        total_questions += num_questions
                        file_count += 1
                        print(f"[+] Found {num_questions} questions in: {file_path}")
                    else:
                        print(f"[!] Warning: 'questions' list missing or invalid in: {file_path}")
                        
            except json.JSONDecodeError:
                print(f"[!] Error: Could not decode JSON in: {file_path}")
            except Exception as e:
                print(f"[!] Error reading {file_path}: {e}")

    print("-" * 50)
    print(f"Total files processed: {file_count}")
    print(f"Total questions found: {total_questions}")

if __name__ == "__main__":
    count_questions_in_json_files()