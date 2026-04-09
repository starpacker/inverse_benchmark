import sys
import os
import dill
import numpy as np
import traceback

# Import target function
from agent_run_inversion import run_inversion

# Import dependencies for evaluate_results
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.image_model import ImageModel


# --- Injected Referee Functions ---

def forward_operator(
    kwargs_lens: list,
    kwargs_source: list,
    kwargs_lens_light: list,
    data_class: ImageData,
    psf_class: PSF,
    lens_model_class: LensModel,
    source_model_class: LightModel,
    lens_light_model_class: LightModel,
    kwargs_numerics: dict
) -> np.ndarray:
    """
    Forward operator: compute model image given lens, source, and lens light parameters.
    
    This creates an ImageModel and computes the predicted image.
    """
    imageModel = ImageModel(
        data_class, psf_class,
        lens_model_class=lens_model_class,
        source_model_class=source_model_class,
        lens_light_model_class=lens_light_model_class,
        kwargs_numerics=kwargs_numerics
    )
    
    y_pred = imageModel.image(
        kwargs_lens, kwargs_source,
        kwargs_lens_light=kwargs_lens_light,
        kwargs_ps=None
    )
    
    return y_pred


def evaluate_results(
    inversion_result: dict,
    preprocessed_data: dict
) -> dict:
    """
    Evaluate the fitting results by comparing to true parameters and computing residuals.
    """
    kwargs_result = inversion_result['kwargs_result']
    
    print("Best Fit Result:")
    print(kwargs_result)
    
    # Extract fitted parameters
    fitted_kwargs_lens = kwargs_result.get('kwargs_lens', [])
    fitted_kwargs_source = kwargs_result.get('kwargs_source', [])
    fitted_kwargs_lens_light = kwargs_result.get('kwargs_lens_light', [])
    
    # Get true parameters
    true_kwargs_lens = preprocessed_data['true_kwargs_lens']
    true_kwargs_source = preprocessed_data['true_kwargs_source']
    true_kwargs_lens_light = preprocessed_data['true_kwargs_lens_light']
    
    # Compute model image with fitted parameters
    model_image = forward_operator(
        fitted_kwargs_lens,
        fitted_kwargs_source,
        fitted_kwargs_lens_light,
        preprocessed_data['data_class'],
        preprocessed_data['psf_class'],
        preprocessed_data['lens_model_class'],
        preprocessed_data['source_model_class'],
        preprocessed_data['lens_light_model_class'],
        preprocessed_data['kwargs_numerics']
    )
    
    # Compute residuals
    observed_image = preprocessed_data['image_data']
    residuals = observed_image - model_image
    
    # Compute chi-squared
    background_rms = preprocessed_data['background_rms']
    chi_squared = np.sum((residuals / background_rms) ** 2)
    reduced_chi_squared = chi_squared / (observed_image.size - 1)
    
    # Compare key lens parameters
    print("\n--- Parameter Comparison ---")
    if len(fitted_kwargs_lens) > 0 and len(true_kwargs_lens) > 0:
        print("Lens Model (SIE):")
        for key in ['theta_E', 'e1', 'e2', 'center_x', 'center_y']:
            if key in fitted_kwargs_lens[0] and key in true_kwargs_lens[0]:
                fitted_val = fitted_kwargs_lens[0][key]
                true_val = true_kwargs_lens[0][key]
                print(f"  {key}: True={true_val:.4f}, Fitted={fitted_val:.4f}, Diff={fitted_val - true_val:.4f}")
    
    if len(fitted_kwargs_lens) > 1 and len(true_kwargs_lens) > 1:
        print("Shear Model:")
        for key in ['gamma1', 'gamma2']:
            if key in fitted_kwargs_lens[1] and key in true_kwargs_lens[1]:
                fitted_val = fitted_kwargs_lens[1][key]
                true_val = true_kwargs_lens[1][key]
                print(f"  {key}: True={true_val:.4f}, Fitted={fitted_val:.4f}, Diff={fitted_val - true_val:.4f}")
    
    print(f"\nChi-squared: {chi_squared:.2f}")
    print(f"Reduced Chi-squared: {reduced_chi_squared:.4f}")
    print(f"Residual RMS: {np.std(residuals):.6f}")
    print(f"Fitting time: {inversion_result['fitting_time']:.2f} seconds")
    
    return {
        'model_image': model_image,
        'residuals': residuals,
        'chi_squared': chi_squared,
        'reduced_chi_squared': reduced_chi_squared,
        'residual_rms': np.std(residuals),
        'fitted_kwargs_lens': fitted_kwargs_lens,
        'fitted_kwargs_source': fitted_kwargs_source,
        'fitted_kwargs_lens_light': fitted_kwargs_lens_light
    }


def main():
    # Data paths
    data_paths = ['/home/yjh/lenstronomy_simple_ring_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
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
    
    try:
        # Load outer (primary) data
        if not outer_data_files:
            print("ERROR: No outer data file found!")
            sys.exit(1)
        
        outer_data_path = outer_data_files[0]
        print(f"Loading outer data from: {outer_data_path}")
        
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
        
        # Extract inputs from outer data
        args = outer_data.get('args', ())
        kwargs = outer_data.get('kwargs', {})
        std_output = outer_data.get('output', None)
        
        print(f"Outer data keys: {outer_data.keys()}")
        print(f"Number of args: {len(args)}")
        print(f"Kwargs keys: {kwargs.keys() if kwargs else 'None'}")
        
        # Run the agent function
        print("\n=== Running Agent Function ===")
        agent_output = run_inversion(*args, **kwargs)
        print("Agent function completed successfully.")
        
        # Determine execution pattern
        if inner_data_files:
            # Chained execution pattern
            print("\n=== Chained Execution Pattern Detected ===")
            inner_data_path = inner_data_files[0]
            print(f"Loading inner data from: {inner_data_path}")
            
            with open(inner_data_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            std_result = inner_data.get('output', None)
            
            # Execute the operator returned by agent
            print("Executing inner operator...")
            final_result = agent_output(*inner_args, **inner_kwargs)
        else:
            # Direct execution pattern
            print("\n=== Direct Execution Pattern ===")
            final_result = agent_output
            std_result = std_output
        
        # Get preprocessed_data for evaluation
        # The first argument should be preprocessed_data
        if len(args) >= 1:
            preprocessed_data = args[0]
        else:
            preprocessed_data = kwargs.get('preprocessed_data', None)
        
        if preprocessed_data is None:
            print("ERROR: Could not find preprocessed_data for evaluation!")
            sys.exit(1)
        
        # Evaluation Phase
        print("\n=== Evaluating Agent Results ===")
        eval_agent = evaluate_results(final_result, preprocessed_data)
        
        print("\n=== Evaluating Standard Results ===")
        eval_std = evaluate_results(std_result, preprocessed_data)
        
        # Extract primary metric for comparison (reduced chi-squared - lower is better)
        score_agent = eval_agent['reduced_chi_squared']
        score_std = eval_std['reduced_chi_squared']
        
        residual_rms_agent = eval_agent['residual_rms']
        residual_rms_std = eval_std['residual_rms']
        
        print("\n" + "=" * 60)
        print("=== FINAL COMPARISON ===")
        print("=" * 60)
        print(f"Scores (Reduced Chi-Squared) -> Agent: {score_agent:.6f}, Standard: {score_std:.6f}")
        print(f"Residual RMS -> Agent: {residual_rms_agent:.6f}, Standard: {residual_rms_std:.6f}")
        
        # For reduced chi-squared, lower is better
        # Allow a 20% margin for optimization algorithms (they may find different local minima)
        margin = 1.2  # 20% tolerance
        
        # Check if agent performance is acceptable
        # Agent should have comparable or better reduced chi-squared
        if score_agent <= score_std * margin:
            print(f"\nSUCCESS: Agent reduced chi-squared ({score_agent:.6f}) is within acceptable range of standard ({score_std:.6f})")
            
            # Additional check on residual RMS
            if residual_rms_agent <= residual_rms_std * margin:
                print(f"SUCCESS: Agent residual RMS ({residual_rms_agent:.6f}) is within acceptable range of standard ({residual_rms_std:.6f})")
                sys.exit(0)
            else:
                print(f"WARNING: Agent residual RMS ({residual_rms_agent:.6f}) is higher than expected ({residual_rms_std * margin:.6f})")
                # Still pass if chi-squared is good
                sys.exit(0)
        else:
            print(f"\nFAILURE: Agent reduced chi-squared ({score_agent:.6f}) exceeds acceptable threshold ({score_std * margin:.6f})")
            sys.exit(1)
            
    except Exception as e:
        print(f"\nERROR: Exception occurred during testing!")
        print(f"Exception type: {type(e).__name__}")
        print(f"Exception message: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()