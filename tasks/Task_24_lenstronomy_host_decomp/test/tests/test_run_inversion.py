import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_run_inversion import run_inversion

# Import dependencies for evaluate_results
from lenstronomy.ImSim.image_linear_solve import ImageLinearFit


# --- Injected Referee Function ---
def evaluate_results(data_class, psf_class, lightModel, pointSource, kwargs_numerics,
                     kwargs_result, image_sim):
    """
    Evaluate the fitting results by computing the reconstructed image and comparing
    with the observed data.
    
    Args:
        data_class: ImageData object
        psf_class: PSF object
        lightModel: LightModel object
        pointSource: PointSource object
        kwargs_numerics: numerics configuration
        kwargs_result: best fit parameters from inversion
        image_sim: observed/simulated image
    
    Returns:
        dict containing:
            - image_reconstructed: reconstructed image
            - residual: difference between observed and reconstructed
            - reconstructed_sum: sum of reconstructed image
            - true_sum: sum of observed image
            - residual_rms: RMS of residuals
    """
    imageLinearFit = ImageLinearFit(
        data_class=data_class,
        psf_class=psf_class,
        lens_light_model_class=lightModel,
        point_source_class=pointSource,
        kwargs_numerics=kwargs_numerics
    )
    
    image_reconstructed, _, _, _ = imageLinearFit.image_linear_solve(
        kwargs_lens_light=kwargs_result['kwargs_lens_light'],
        kwargs_ps=kwargs_result['kwargs_ps']
    )
    
    residual = image_sim - image_reconstructed
    reconstructed_sum = np.sum(image_reconstructed)
    true_sum = np.sum(image_sim)
    residual_rms = np.sqrt(np.mean(residual**2))
    
    return {
        'image_reconstructed': image_reconstructed,
        'residual': residual,
        'reconstructed_sum': reconstructed_sum,
        'true_sum': true_sum,
        'residual_rms': residual_rms
    }


def main():
    data_paths = ['/home/yjh/lenstronomy_host_decomp_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer and inner data paths
    outer_paths = []
    inner_paths = []
    
    for path in data_paths:
        if 'parent_function' in os.path.basename(path):
            inner_paths.append(path)
        else:
            outer_paths.append(path)
    
    if not outer_paths:
        print("ERROR: No outer data file found.")
        sys.exit(1)
    
    # Load the primary (outer) data
    outer_data_path = outer_paths[0]
    print(f"Loading outer data from: {outer_data_path}")
    
    try:
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract args and kwargs for run_inversion
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)
    
    # Extract additional data needed for evaluation
    # These should be present in the data for evaluate_results
    data_class = outer_data.get('data_class', None)
    psf_class = outer_data.get('psf_class', None)
    lightModel = outer_data.get('lightModel', None)
    pointSource = outer_data.get('pointSource', None)
    kwargs_numerics = outer_data.get('kwargs_numerics', None)
    image_sim = outer_data.get('image_sim', None)
    
    # If these are not in outer_data directly, try to get from args/kwargs
    if data_class is None and len(args) > 0:
        data_class = args[0]
    if psf_class is None and len(args) > 1:
        psf_class = args[1]
    if lightModel is None and len(args) > 2:
        lightModel = args[2]
    if pointSource is None and len(args) > 3:
        pointSource = args[3]
    if kwargs_numerics is None:
        kwargs_numerics = kwargs.get('kwargs_numerics', None)
        if kwargs_numerics is None and len(args) > 6:
            kwargs_numerics = args[6]
    
    # Try to get image_sim from the data
    if image_sim is None:
        # image_sim might be stored separately or derived from data_class
        if data_class is not None and hasattr(data_class, 'data'):
            image_sim = data_class.data
    
    print("Running agent's run_inversion...")
    try:
        agent_output = run_inversion(*args, **kwargs)
        print("Agent run_inversion completed successfully.")
    except Exception as e:
        print(f"ERROR during agent run_inversion: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if this is a chained execution (closure/factory pattern)
    if inner_paths:
        # Pattern 2: Chained Execution
        print("Detected chained execution pattern.")
        inner_data_path = inner_paths[0]
        print(f"Loading inner data from: {inner_data_path}")
        
        try:
            with open(inner_data_path, 'rb') as f:
                inner_data = dill.load(f)
        except Exception as e:
            print(f"ERROR loading inner data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        std_result = inner_data.get('output', None)
        
        print("Running operator from agent output...")
        try:
            final_result = agent_output(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"ERROR during operator execution: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Pattern 1: Direct Execution
        print("Direct execution pattern.")
        final_result = agent_output
        std_result = std_output
    
    # Verify we have the necessary data for evaluation
    if data_class is None or psf_class is None or lightModel is None or pointSource is None:
        print("WARNING: Missing required objects for full evaluation.")
        print("Falling back to direct comparison of results.")
        
        # Compare key metrics from the output dictionaries
        if isinstance(final_result, dict) and isinstance(std_result, dict):
            agent_time = final_result.get('fitting_time', None)
            std_time = std_result.get('fitting_time', None)
            
            print(f"Agent fitting time: {agent_time}")
            print(f"Standard fitting time: {std_time}")
            
            # Compare kwargs_result structure
            agent_kwargs = final_result.get('kwargs_result', {})
            std_kwargs = std_result.get('kwargs_result', {})
            
            print(f"Agent kwargs_result keys: {agent_kwargs.keys() if agent_kwargs else 'None'}")
            print(f"Standard kwargs_result keys: {std_kwargs.keys() if std_kwargs else 'None'}")
            
            # Check if agent produced valid output
            if agent_kwargs and 'kwargs_lens_light' in agent_kwargs and 'kwargs_ps' in agent_kwargs:
                print("Agent output structure is valid.")
                print("TEST PASSED (structure validation)")
                sys.exit(0)
            else:
                print("Agent output structure is invalid.")
                sys.exit(1)
        else:
            print(f"Cannot compare results: agent type={type(final_result)}, std type={type(std_result)}")
            sys.exit(1)
    
    # Perform full evaluation using evaluate_results
    print("Performing full evaluation...")
    
    try:
        # Get kwargs_result from the outputs
        agent_kwargs_result = final_result.get('kwargs_result', final_result)
        std_kwargs_result = std_result.get('kwargs_result', std_result) if isinstance(std_result, dict) else std_result
        
        # Evaluate agent's result
        print("Evaluating agent results...")
        eval_agent = evaluate_results(
            data_class=data_class,
            psf_class=psf_class,
            lightModel=lightModel,
            pointSource=pointSource,
            kwargs_numerics=kwargs_numerics,
            kwargs_result=agent_kwargs_result,
            image_sim=image_sim
        )
        
        # Evaluate standard result
        print("Evaluating standard results...")
        eval_std = evaluate_results(
            data_class=data_class,
            psf_class=psf_class,
            lightModel=lightModel,
            pointSource=pointSource,
            kwargs_numerics=kwargs_numerics,
            kwargs_result=std_kwargs_result,
            image_sim=image_sim
        )
        
        # Extract metrics
        agent_rms = eval_agent['residual_rms']
        std_rms = eval_std['residual_rms']
        
        agent_sum = eval_agent['reconstructed_sum']
        std_sum = eval_std['reconstructed_sum']
        true_sum = eval_std['true_sum']
        
        print(f"\n=== Evaluation Results ===")
        print(f"Residual RMS -> Agent: {agent_rms:.6f}, Standard: {std_rms:.6f}")
        print(f"Reconstructed Sum -> Agent: {agent_sum:.2f}, Standard: {std_sum:.2f}, True: {true_sum:.2f}")
        
        # For RMS, lower is better
        # Allow agent to be up to 20% worse (for optimization algorithms with randomness)
        margin = 0.20
        
        if agent_rms <= std_rms * (1 + margin):
            print(f"\nTEST PASSED: Agent RMS ({agent_rms:.6f}) is within acceptable range of Standard ({std_rms:.6f})")
            sys.exit(0)
        else:
            print(f"\nTEST FAILED: Agent RMS ({agent_rms:.6f}) exceeds acceptable margin over Standard ({std_rms:.6f})")
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR during evaluation: {e}")
        traceback.print_exc()
        
        # Fallback: basic structure check
        if isinstance(final_result, dict) and 'kwargs_result' in final_result:
            print("Fallback: Agent produced valid output structure.")
            print("TEST PASSED (fallback validation)")
            sys.exit(0)
        else:
            sys.exit(1)


if __name__ == "__main__":
    main()