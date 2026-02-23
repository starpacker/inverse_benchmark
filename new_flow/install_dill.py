import re
import os
import subprocess

pipeline_script = '/home/yjh/new_flow/run_pipeline_2.sh'

with open(pipeline_script, 'r') as f:
    content = f.read()

# 1. Extract absolute python paths
# Matches /any/path/bin/python or /any/path/python
# We look for typical patterns.
# Regex: find strings starting with / and ending with python, possibly inside quotes
paths = set(re.findall(r'[\'"](/[^"\'\s]+/python)[\'"\s]', content))

# 2. Handle specific bash variables pattern (e.g., for pypet_custom)
# ENV_NAME="pypet_custom"
# PYTHON_PATH="/home/yjh/.conda/envs/$ENV_NAME/bin/python"
env_names = re.findall(r'ENV_NAME="([^"]+)"', content)
for env_name in env_names:
    potential_path = f"/home/yjh/.conda/envs/{env_name}/bin/python"
    paths.add(potential_path)

# Filter valid paths
valid_paths = []
for p in paths:
    if os.path.exists(p) and os.access(p, os.X_OK):
        valid_paths.append(p)
    else:
        print(f"Skipping invalid or inaccessible python path: {p}")

print(f"Found {len(valid_paths)} python environments to install dill.")

for py_path in valid_paths:
    print(f"\nInstalling dill for {py_path}...")
    try:
        # Check if already installed
        # check_cmd = [py_path, '-c', 'import dill; print("dill installed")']
        # res = subprocess.run(check_cmd, capture_output=True, text=True)
        # if "dill installed" in res.stdout:
        #     print(f"dill already installed in {py_path}")
        #     continue
            
        # Install
        install_cmd = [py_path, '-m', 'pip', 'install', 'dill']
        subprocess.run(install_cmd, check=True)
        print("Success.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dill in {py_path}. Error: {e}")
    except Exception as e:
        print(f"An error occurred with {py_path}: {e}")

print("\nDone.")
