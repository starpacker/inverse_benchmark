import os

# Base Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP_DIR = "/data/yjh/tmp"
DATASET_DIR = "/data/yjh/benchmark_dataset"
RUNNER_CACHE_DIR = "/data/yjh/benchmark_results/golden_references"
SANDBOX_DIR = "/data/yjh/benchmark_sandboxes_tmp"

# Ensure directories exist
os.makedirs(TMP_DIR, exist_ok=True)
os.environ["TMPDIR"] = TMP_DIR

# Model Configs
DEFAULT_MODEL = "gpt-4o"
DEFAULT_JUDGE_MODEL = "gemini-3-pro-preview"
DEFAULT_PHASE0_MODEL = "claude-opus-4-5-20251101-thinking"

# Timeouts
EXECUTION_TIMEOUT = 600
VERIFICATION_TIMEOUT = 300

# Retries
MAX_LLM_RETRIES = 3
MAX_CODE_GENERATION_RETRIES = 3
MAX_VERIFICATION_RETRIES = 3
