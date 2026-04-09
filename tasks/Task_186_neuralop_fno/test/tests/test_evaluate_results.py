import sys
import os
import dill
import torch
import torch.nn.functional as F
import numpy as np
import traceback

# Inject torch.nn.functional as F into multiple scopes to fix deserialization
import builtins
builtins.F = F

# Also patch it into torch.nn.modules.module scope and any other likely places
import torch.nn.modules.module as _torch_module_mod
_torch_module_mod.F = F

def main():
    data_paths = [
        '/data/yjh/neuralop_fno_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl'
    ]

    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("ERROR: No outer data file found.")
        sys.exit(1)

    # Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print("Loaded outer data successfully.")
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    # Ensure model and tensors are on the correct device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def move_to_device(obj, dev):
        if isinstance(obj, torch.Tensor):
            return obj.to(dev)
        if isinstance(obj, torch.nn.Module):
            return obj.to(dev)
        if isinstance(obj, dict):
            return {k: move_to_device(v, dev) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            moved = [move_to_device(x, dev) for x in obj]
            return type(obj)(moved)
        return obj

    outer_args = move_to_device(list(outer_args), device)
    outer_kwargs = move_to_device(outer_kwargs, device)

    # Patch F into the model's module globals if possible
    for arg in list(outer_args) + list(outer_kwargs.values()):
        if isinstance(arg, torch.nn.Module):
            for mod in arg.modules():
                if hasattr(mod, 'forward') and hasattr(mod.forward, '__globals__'):
                    mod.forward.__globals__['F'] = F
                # Also try __func__ for bound methods
                if hasattr(mod, 'forward') and hasattr(mod.forward, '__func__'):
                    if hasattr(mod.forward.__func__, '__globals__'):
                        mod.forward.__func__.__globals__['F'] = F

    # Also patch into gen_std_data module if it's loaded
    if 'gen_std_data' in sys.modules:
        sys.modules['gen_std_data'].F = F
    # Preemptively try to load and patch it
    try:
        gen_data_path = '/data/yjh/neuralop_fno_sandbox_sandbox/gen_std_data.py'
        if os.path.exists(gen_data_path):
            import importlib.util
            spec = importlib.util.spec_from_file_location("gen_std_data", gen_data_path)
            gen_mod = importlib.util.module_from_spec(spec)
            gen_mod.F = F
            sys.modules['gen_std_data'] = gen_mod
            try:
                spec.loader.exec_module(gen_mod)
            except Exception:
                pass
            gen_mod.F = F
    except Exception:
        pass

    # Patch __main__ as well since dill often deserializes into __main__
    import __main__
    __main__.F = F

    # Determine scenario
    if inner_paths:
        print("Detected Scenario B: Factory/Closure pattern.")
    else:
        print("Detected Scenario A: Simple function call.")

    # Use a temporary results_dir to avoid polluting the real one
    import tempfile
    temp_results_dir = tempfile.mkdtemp(prefix="test_eval_results_")

    # Override results_dir in args if present
    # evaluate_results signature: (model, data_dict, device, results_dir)
    # args[3] is results_dir
    if len(outer_args) >= 4:
        outer_args[3] = temp_results_dir
    if 'results_dir' in outer_kwargs:
        outer_kwargs['results_dir'] = temp_results_dir

    outer_args = tuple(outer_args)

    # Import the target function
    try:
        # Patch F into agent module scope before import
        agent_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agent_evaluate_results.py')
        if os.path.exists(agent_path):
            # Pre-read and check if F is missing
            pass
        from agent_evaluate_results import evaluate_results
    except Exception as e:
        print(f"ERROR importing evaluate_results: {e}")
        traceback.print_exc()
        sys.exit(1)

    from verification_utils import recursive_check

    # Execute
    try:
        result = evaluate_results(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR executing evaluate_results: {e}")
        traceback.print_exc()
        sys.exit(1)

    if inner_paths:
        # Scenario B
        inner_paths.sort()
        inner_path = inner_paths[0]
        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            print("Loaded inner data successfully.")
        except Exception as e:
            print(f"ERROR loading inner data: {e}")
            traceback.print_exc()
            sys.exit(1)

        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        expected = inner_data.get('output', None)

        if not callable(result):
            print("ERROR: Expected callable from outer call (Scenario B), got non-callable.")
            sys.exit(1)

        try:
            actual_result = result(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"ERROR executing inner call: {e}")
            traceback.print_exc()
            sys.exit(1)

        passed, msg = recursive_check(expected, actual_result)
    else:
        # Scenario A
        expected = outer_output
        actual_result = result
        passed, msg = recursive_check(expected, actual_result)

    # Cleanup temp dir
    try:
        import shutil
        shutil.rmtree(temp_results_dir, ignore_errors=True)
    except Exception:
        pass

    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)


if __name__ == '__main__':
    main()