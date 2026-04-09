import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_run_inversion import run_inversion


# --- Injected Referee (Evaluation Logic) ---

def forward_operator(params, image_model, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps):
    """
    Forward operator: compute the predicted image given model parameters.
    
    Args:
        params: dictionary of parameters to update (optional, can be None to use defaults)
        image_model: ImageModel instance
        kwargs_lens: lens model parameters
        kwargs_source: source model parameters
        kwargs_lens_light: lens light model parameters
        kwargs_ps: point source parameters
    
    Returns:
        y_pred: predicted image as numpy array
    """
    # If params provided, update the kwargs
    if params is not None:
        if 'lens' in params:
            for i, p in enumerate(params['lens']):
                kwargs_lens[i].update(p)
        if 'source' in params:
            for i, p in enumerate(params['source']):
                kwargs_source[i].update(p)
        if 'lens_light' in params:
            for i, p in enumerate(params['lens_light']):
                kwargs_lens_light[i].update(p)
        if 'ps' in params:
            for i, p in enumerate(params['ps']):
                kwargs_ps[i].update(p)
    
    # Compute the forward model
    y_pred = image_model.image(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps)
    
    return y_pred


def evaluate_results(data_dict, inversion_result):
    """
    Evaluate the fitting results by comparing with true parameters and computing residuals.
    
    Args:
        data_dict: dictionary containing all data and true parameters
        inversion_result: dictionary containing fitting results
    
    Returns:
        dict containing evaluation metrics
    """
    kwargs_result = inversion_result['kwargs_result']
    image_model = data_dict['image_model']
    image_sim = data_dict['image_sim']
    
    kwargs_lens_true = data_dict['kwargs_lens_true']
    kwargs_source_true = data_dict['kwargs_source_true']
    kwargs_lens_light_true = data_dict['kwargs_lens_light_true']
    
    # Extract fitted parameters
    kwargs_lens_fit = kwargs_result['kwargs_lens']
    kwargs_source_fit = kwargs_result['kwargs_source']
    kwargs_lens_light_fit = kwargs_result['kwargs_lens_light']
    kwargs_ps_fit = kwargs_result['kwargs_ps']
    
    # Compute model prediction with fitted parameters
    image_reconstructed = forward_operator(
        None,
        image_model,
        kwargs_lens_fit,
        kwargs_source_fit,
        kwargs_lens_light_fit,
        kwargs_ps_fit
    )
    
    # Compute residuals
    residuals = image_sim - image_reconstructed
    residual_rms = np.sqrt(np.mean(residuals**2))
    
    # Compare key parameters
    theta_E_true = kwargs_lens_true[0]['theta_E']
    theta_E_fit = kwargs_lens_fit[0]['theta_E']
    theta_E_error = np.abs(theta_E_fit - theta_E_true)
    
    gamma_true = kwargs_lens_true[0]['gamma']
    gamma_fit = kwargs_lens_fit[0]['gamma']
    gamma_error = np.abs(gamma_fit - gamma_true)
    
    e1_true = kwargs_lens_true[0]['e1']
    e1_fit = kwargs_lens_fit[0]['e1']
    e1_error = np.abs(e1_fit - e1_true)
    
    e2_true = kwargs_lens_true[0]['e2']
    e2_fit = kwargs_lens_fit[0]['e2']
    e2_error = np.abs(e2_fit - e2_true)
    
    # Print evaluation results
    print("\n=== Evaluation Results ===")
    print(f"Residual RMS: {residual_rms:.6f}")
    print(f"\nLens Parameters Comparison:")
    print(f"  theta_E: True={theta_E_true:.4f}, Fit={theta_E_fit:.4f}, Error={theta_E_error:.4f}")
    print(f"  gamma: True={gamma_true:.4f}, Fit={gamma_fit:.4f}, Error={gamma_error:.4f}")
    print(f"  e1: True={e1_true:.4f}, Fit={e1_fit:.4f}, Error={e1_error:.4f}")
    print(f"  e2: True={e2_true:.4f}, Fit={e2_fit:.4f}, Error={e2_error:.4f}")
    
    return {
        'residual_rms': residual_rms,
        'theta_E_error': theta_E_error,
        'gamma_error': gamma_error,
        'e1_error': e1_error,
        'e2_error': e2_error,
        'image_reconstructed': image_reconstructed,
        'residuals': residuals,
        'kwargs_lens_fit': kwargs_lens_fit,
        'kwargs_source_fit': kwargs_source_fit,
        'kwargs_lens_light_fit': kwargs_lens_light_fit,
        'kwargs_ps_fit': kwargs_ps_fit
    }


def main():
    # Data paths provided
    data_paths = ['/home/yjh/lenstronomy_quad_quasar_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer and inner data files
    outer_data_files = []
    inner_data_files = []
    
    for path in data_paths:
        filename = os.path.basename(path)
        if 'parent_function' in filename:
            inner_data_files.append(path)
        else:
            outer_data_files.append(path)
    
    print(f"Outer data files: {outer_data_files}")
    print(f"Inner data files: {inner_data_files}")
    
    # Determine execution pattern
    is_chained_execution = len(inner_data_files) > 0
    
    try:
        # Load outer (primary) data
        if len(outer_data_files) == 0:
            print("ERROR: No outer data file found!")
            sys.exit(1)
        
        outer_data_path = outer_data_files[0]
        print(f"\nLoading outer data from: {outer_data_path}")
        
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
        
        # Extract inputs for run_inversion
        args = outer_data.get('args', ())
        kwargs = outer_data.get('kwargs', {})
        std_output = outer_data.get('output', None)
        
        # The input to run_inversion is a data_dict
        # Check if args contains the data_dict or if it's in kwargs
        if len(args) > 0:
            input_data_dict = args[0]
        elif 'data_dict' in kwargs:
            input_data_dict = kwargs['data_dict']
        else:
            # The outer_data itself might be the data_dict
            input_data_dict = outer_data
        
        print("\n=== Running Agent's run_inversion ===")
        
        if is_chained_execution:
            # Chained execution pattern
            print("Detected chained execution pattern")
            agent_operator = run_inversion(*args, **kwargs)
            
            # Load inner data
            inner_data_path = inner_data_files[0]
            print(f"\nLoading inner data from: {inner_data_path}")
            
            with open(inner_data_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            std_result = inner_data.get('output', None)
            
            # Execute the operator with inner data
            final_result = agent_operator(*inner_args, **inner_kwargs)
        else:
            # Direct execution pattern
            print("Detected direct execution pattern")
            final_result = run_inversion(*args, **kwargs)
            std_result = std_output
        
        print("\n=== Agent execution completed ===")
        
        # Evaluate results
        print("\n=== Evaluating Agent Results ===")
        eval_agent = evaluate_results(input_data_dict, final_result)
        
        print("\n=== Evaluating Standard Results ===")
        eval_std = evaluate_results(input_data_dict, std_result)
        
        # Extract primary metrics for comparison
        # Using residual_rms as the primary metric (lower is better)
        score_agent = eval_agent['residual_rms']
        score_std = eval_std['residual_rms']
        
        print(f"\n=== Final Comparison ===")
        print(f"Scores -> Agent: {score_agent:.6f}, Standard: {score_std:.6f}")
        
        # Also compare parameter errors
        print(f"\nParameter Error Comparison:")
        print(f"  theta_E error -> Agent: {eval_agent['theta_E_error']:.4f}, Standard: {eval_std['theta_E_error']:.4f}")
        print(f"  gamma error -> Agent: {eval_agent['gamma_error']:.4f}, Standard: {eval_std['gamma_error']:.4f}")
        print(f"  e1 error -> Agent: {eval_agent['e1_error']:.4f}, Standard: {eval_std['e1_error']:.4f}")
        print(f"  e2 error -> Agent: {eval_agent['e2_error']:.4f}, Standard: {eval_std['e2_error']:.4f}")
        
        # Verification: residual_rms is a "loss" metric (lower is better)
        # Allow a margin of error (e.g., agent can be up to 50% worse due to stochastic nature of MCMC)
        # MCMC/PSO are stochastic, so we use a generous margin
        margin = 2.0  # Agent can be up to 2x worse
        
        if score_agent <= score_std * margin:
            print(f"\n✓ PASS: Agent performance is acceptable (within {margin}x margin)")
            print(f"  Agent residual RMS ({score_agent:.6f}) <= Standard residual RMS ({score_std:.6f}) * {margin}")
            
            # Additional check: parameter errors should be reasonable
            total_param_error_agent = (eval_agent['theta_E_error'] + eval_agent['gamma_error'] + 
                                       eval_agent['e1_error'] + eval_agent['e2_error'])
            total_param_error_std = (eval_std['theta_E_error'] + eval_std['gamma_error'] + 
                                     eval_std['e1_error'] + eval_std['e2_error'])
            
            print(f"\n  Total parameter error -> Agent: {total_param_error_agent:.4f}, Standard: {total_param_error_std:.4f}")
            
            sys.exit(0)
        else:
            print(f"\n✗ FAIL: Agent performance degraded significantly")
            print(f"  Agent residual RMS ({score_agent:.6f}) > Standard residual RMS ({score_std:.6f}) * {margin}")
            sys.exit(1)
            
    except Exception as e:
        print(f"\nERROR during execution: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()