import sys
import os
import dill
import torch
import numpy as np
import traceback
import torch.nn.functional as F

# Ensure repo is on path
REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repo")
if os.path.isdir(REPO_DIR):
    sys.path.insert(0, REPO_DIR)

from agent_forward_operator import forward_operator
from verification_utils import recursive_check


def _patch_model_globals(model):
    """
    Recursively patch the forward method of the model (and submodules)
    so that 'F' (torch.nn.functional) is available in their globals.
    """
    import torch.nn.functional as _F
    try:
        if hasattr(model, 'forward') and hasattr(model.forward, '__globals__'):
            model.forward.__globals__['F'] = _F
    except Exception:
        pass
    if hasattr(model, 'modules'):
        try:
            for m in model.modules():
                if hasattr(m, 'forward') and hasattr(m.forward, '__globals__'):
                    m.forward.__globals__['F'] = _F
        except Exception:
            pass


def main():
    data_paths = [
        '/data/yjh/neuralop_fno_sandbox_sandbox/run_code/std_data/standard_data_forward_operator.pkl'
    ]

    outer_path = None
    inner_paths = []
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("ERROR: No outer data file found.")
        sys.exit(1)

    # --- Load outer data ---
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})

    # --- Patch model globals so 'F' is defined ---
    # The model is the first positional arg
    if outer_args:
        model_candidate = outer_args[0]
        if isinstance(model_candidate, torch.nn.Module):
            _patch_model_globals(model_candidate)

    # Also inject F into gen_std_data module if it's loaded
    try:
        import importlib
        if 'gen_std_data' in sys.modules:
            sys.modules['gen_std_data'].F = F
    except Exception:
        pass

    # Broader injection: find any loaded module missing 'F' that defines the forward
    for mod_name, mod in list(sys.modules.items()):
        if mod is not None and hasattr(mod, '__dict__'):
            if 'gelu' not in str(mod.__dict__.get('F', '')) and 'gen_std_data' in mod_name:
                try:
                    mod.__dict__['F'] = F
                except Exception:
                    pass

    # --- Phase 1: Execute forward_operator ---
    try:
        agent_result = forward_operator(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR executing forward_operator: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- Phase 2: Determine scenario ---
    if inner_paths:
        # Scenario B: factory/closure pattern
        if not callable(agent_result):
            print("ERROR: forward_operator did not return a callable for Scenario B.")
            sys.exit(1)

        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"ERROR loading inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')

            try:
                result = agent_result(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR executing inner callable: {e}")
                traceback.print_exc()
                sys.exit(1)

            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"FAILED (inner): {msg}")
                sys.exit(1)
            print(f"Inner test passed for: {inner_path}")
    else:
        # Scenario A: simple function
        expected = outer_data.get('output')
        result = agent_result

        passed, msg = recursive_check(expected, result)
        if not passed:
            print(f"FAILED: {msg}")
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()