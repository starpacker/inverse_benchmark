import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_run_inversion import run_inversion

# Import dependencies for evaluate_results
from lenstronomy.ImSim.image_model import ImageModel


# --- Injected Referee Functions ---

def forward_operator(data_dict, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps):
    """
    Compute the forward model image given the model parameters.
    
    Args:
        data_dict: Dictionary containing model classes and data configuration.
        kwargs_lens: Lens model parameters.
        kwargs_source: Source light parameters.
        kwargs_lens_light: Lens light parameters.
        kwargs_ps: Point source parameters.
    
    Returns:
        np.ndarray: Predicted model image.
    """
    # Create ImageModel
    imageModel = ImageModel(
        data_dict['data_class'],
        data_dict['psf_class'],
        data_dict['lens_model_class'],
        data_dict['source_model_class'],
        data_dict['lens_light_model_class'],
        data_dict['point_source_class'],
        kwargs_numerics=data_dict['kwargs_numerics']
    )
    
    # Compute forward model
    image_model = imageModel.image(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps)
    
    return image_model


def evaluate_results(data_dict, inversion_result):
    """
    Evaluate the inversion results by comparing to true parameters and computing residuals.
    
    Args:
        data_dict: Dictionary containing true parameters and data.
        inversion_result: Dictionary containing best fit parameters.
    
    Returns:
        dict: Evaluation metrics.
    """
    kwargs_result = inversion_result['kwargs_result']
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    print("\nBest fit result:")
    print(kwargs_result)
    
    # Extract fitted parameters
    kwargs_lens_fit = kwargs_result['kwargs_lens']
    kwargs_source_fit = kwargs_result['kwargs_source']
    kwargs_lens_light_fit = kwargs_result['kwargs_lens_light']
    kwargs_ps_fit = kwargs_result['kwargs_ps']
    
    # True parameters
    kwargs_lens_true = data_dict['kwargs_lens_true']
    kwargs_source_true = data_dict['kwargs_source_true']
    kwargs_lens_light_true = data_dict['kwargs_lens_light_true']
    
    # Compare lens parameters
    print("\n--- Lens Model Comparison ---")
    lens_params_to_compare = ['theta_E', 'gamma', 'e1', 'e2', 'center_x', 'center_y']
    lens_residuals = {}
    for param in lens_params_to_compare:
        if param in kwargs_lens_true[0] and param in kwargs_lens_fit[0]:
            true_val = kwargs_lens_true[0][param]
            fit_val = kwargs_lens_fit[0][param]
            residual = fit_val - true_val
            lens_residuals[param] = residual
            print(f"  {param}: True={true_val:.4f}, Fit={fit_val:.4f}, Residual={residual:.4f}")
    
    # Shear comparison
    print("\n--- Shear Comparison ---")
    shear_params = ['gamma1', 'gamma2']
    shear_residuals = {}
    for param in shear_params:
        if param in kwargs_lens_true[1] and param in kwargs_lens_fit[1]:
            true_val = kwargs_lens_true[1][param]
            fit_val = kwargs_lens_fit[1][param]
            residual = fit_val - true_val
            shear_residuals[param] = residual
            print(f"  {param}: True={true_val:.4f}, Fit={fit_val:.4f}, Residual={residual:.4f}")
    
    # Compute model image with best fit
    model_image = forward_operator(
        data_dict, kwargs_lens_fit, kwargs_source_fit, kwargs_lens_light_fit, kwargs_ps_fit
    )
    
    # Compute residual image
    observed_image = data_dict['image_noisy']
    residual_image = observed_image - model_image
    
    # Compute chi-squared
    background_rms = data_dict['background_rms']
    chi2 = np.sum((residual_image / background_rms) ** 2)
    reduced_chi2 = chi2 / (observed_image.size - 1)
    
    print(f"\n--- Image Residuals ---")
    print(f"  Chi-squared: {chi2:.2f}")
    print(f"  Reduced Chi-squared: {reduced_chi2:.4f}")
    print(f"  Residual RMS: {np.std(residual_image):.4f}")
    print(f"  Max absolute residual: {np.max(np.abs(residual_image)):.4f}")
    
    # Fitting time
    print(f"\n--- Performance ---")
    print(f"  Fitting time: {inversion_result['fitting_time']:.2f} seconds")
    
    return {
        'lens_residuals': lens_residuals,
        'shear_residuals': shear_residuals,
        'chi2': chi2,
        'reduced_chi2': reduced_chi2,
        'residual_rms': np.std(residual_image),
        'model_image': model_image,
        'residual_image': residual_image,
        'fitting_time': inversion_result['fitting_time']
    }


def main():
    # Data paths provided
    data_paths = ['/home/yjh/lenstronomy_double_quasar_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer and inner data files
    outer_data_files = []
    inner_data_files = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_data_files.append(path)
        else:
            outer_data_files.append(path)
    
    print(f"Outer data files: {outer_data_files}")
    print(f"Inner data files: {inner_data_files}")
    
    # Determine execution pattern
    is_chained_execution = len(inner_data_files) > 0
    
    try:
        # Load primary (outer) data
        if not outer_data_files:
            print("ERROR: No outer data file found!")
            sys.exit(1)
        
        outer_data_path = outer_data_files[0]
        print(f"\nLoading outer data from: {outer_data_path}")
        
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
        
        print(f"Outer data keys: {outer_data.keys()}")
        
        # Extract inputs
        args = outer_data.get('args', ())
        kwargs = outer_data.get('kwargs', {})
        std_output = outer_data.get('output', None)
        
        # The data_dict is typically the first argument
        if args:
            data_dict = args[0]
        else:
            data_dict = kwargs.get('data_dict', None)
        
        print(f"\nRunning run_inversion with {len(args)} args and kwargs: {list(kwargs.keys())}")
        
        # Execute the agent function
        agent_output = run_inversion(*args, **kwargs)
        
        if is_chained_execution:
            # Chained execution pattern
            inner_data_path = inner_data_files[0]
            print(f"\nLoading inner data from: {inner_data_path}")
            
            with open(inner_data_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            std_result = inner_data.get('output', None)
            
            # Execute the operator returned by run_inversion
            final_result = agent_output(*inner_args, **inner_kwargs)
        else:
            # Direct execution pattern
            final_result = agent_output
            std_result = std_output
        
        print("\n" + "=" * 60)
        print("AGENT OUTPUT EVALUATION")
        print("=" * 60)
        
        # Evaluate agent results
        eval_agent = evaluate_results(data_dict, final_result)
        
        print("\n" + "=" * 60)
        print("STANDARD OUTPUT EVALUATION")
        print("=" * 60)
        
        # Evaluate standard results
        eval_std = evaluate_results(data_dict, std_result)
        
        # Extract primary metrics for comparison
        score_agent = eval_agent['reduced_chi2']
        score_std = eval_std['reduced_chi2']
        
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        print(f"Scores -> Agent: {score_agent:.4f}, Standard: {score_std:.4f}")
        
        # For chi-squared, lower is better
        # Allow margin of error (agent can be up to 50% worse for stochastic optimization)
        # Since MCMC/PSO are stochastic, we use a generous margin
        margin = 2.0  # Agent chi2 can be up to 2x the standard
        
        if score_agent <= score_std * margin:
            print(f"\nSUCCESS: Agent performance is acceptable (within {margin}x margin)")
            print(f"  Agent reduced chi2: {score_agent:.4f}")
            print(f"  Standard reduced chi2: {score_std:.4f}")
            
            # Additional checks on lens parameter recovery
            lens_residuals_agent = eval_agent['lens_residuals']
            lens_residuals_std = eval_std['lens_residuals']
            
            print("\n--- Lens Parameter Residuals Comparison ---")
            param_check_passed = True
            for param in lens_residuals_agent:
                agent_res = abs(lens_residuals_agent[param])
                std_res = abs(lens_residuals_std.get(param, float('inf')))
                print(f"  {param}: Agent |residual|={agent_res:.4f}, Standard |residual|={std_res:.4f}")
                # Allow agent to be up to 3x worse on individual parameters
                if agent_res > max(std_res * 3, 0.1):
                    print(f"    WARNING: Agent residual for {param} is significantly larger")
            
            sys.exit(0)
        else:
            print(f"\nFAILURE: Agent performance degraded beyond acceptable margin")
            print(f"  Agent reduced chi2: {score_agent:.4f}")
            print(f"  Standard reduced chi2: {score_std:.4f}")
            print(f"  Threshold (standard * {margin}): {score_std * margin:.4f}")
            sys.exit(1)
            
    except Exception as e:
        print(f"\nERROR during test execution: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()