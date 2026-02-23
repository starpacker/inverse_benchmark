      
import ast
import os
import json
import re

def load_code_from_file(filepath):
    """Reads the raw source code from a file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find file: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


class FunctionSkeletonizer(ast.NodeTransformer):
    def __init__(self, target_func_name):
        self.target_func_name = target_func_name

    def visit_FunctionDef(self, node):
        if node.name == self.target_func_name:
            # 1. Capture the Docstring (if it exists)
            docstring = ast.get_docstring(node)
            
            # 2. Capture the Return Statement (if it exists)
            return_node = None
            if node.body and isinstance(node.body[-1], ast.Return):
                return_node = node.body[-1]

            # 3. Build the New Body
            new_body = []
            
            # Add Docstring back
            if docstring:
                new_body.append(ast.Expr(value=ast.Constant(value=docstring)))
            
            # Add TODO comment
            new_body.append(ast.Expr(value=ast.Constant(value="TODO: Implement logic here")))
            
            # Add 'pass' (optional if we have a return, but good style for empty blocks)
            # If we strictly just want the return, we can omit pass, 
            # but usually 'pass' helps explicitly show "code goes here".
            # Let's verify: if we have a return, 'pass' is dead code, but harmless for skeletons.
            # Actually, let's just use 'pass' to ensure valid syntax if return is missing.
            if not return_node:
                new_body.append(ast.Pass())

            # Add the preserved Return Statement back
            if return_node:
                new_body.append(return_node)
            
            node.body = new_body
            return node
        
        return self.generic_visit(node)

def create_starter_code(source_code, target_function_name):
    try:
        tree = ast.parse(source_code)
        transformer = FunctionSkeletonizer(target_function_name)
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)
        return ast.unparse(new_tree)
    except Exception as e:
        return f"Error: {e}"

def get_function_signature(source_code, target_func_name):
    try:
        tree = ast.parse(source_code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == target_func_name:
                args_str = ast.unparse(node.args)
                return_ann = ""
                if node.returns:
                    return_ann = " -> " + ast.unparse(node.returns)
                return f"{node.name}({args_str}){return_ann}"
        return f"{target_func_name}(...)"
    except Exception:
        return f"{target_func_name}(...)"

def extract_brief_explanation(prompt_text):
    match = re.search(r"Brief Explanation:\s*(.*?)(?:\n|$)", prompt_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return "No description available."

def process_questions(eval_working_folder: str, json_data):
    updated_questions = []
    
    # 1. Collect all function info first
    func_info_map = {}
    
    for q in json_data['questions']:
        func_name = q['function_name']
        
        gt_save_path = os.path.join(eval_working_folder, f"gt_{func_name}.py")
        if os.path.exists(gt_save_path):
            source_code = load_code_from_file(gt_save_path)
        else:
            gt_code = f"agent_{func_name}.py"
            gt_code_path = os.path.join(eval_working_folder, gt_code)
            if os.path.exists(gt_code_path):
                source_code = load_code_from_file(gt_code_path)
            else:
                source_code = ""

        if source_code:
            signature = get_function_signature(source_code, func_name)
        else:
            signature = f"{func_name}(...)"
            
        description = extract_brief_explanation(q['question_prompt'])
        
        func_info_map[func_name] = {
            "source_code": source_code,
            "signature": signature,
            "description": description
        }

    # 2. Process each question
    for q in json_data['questions']:
        func_name = q['function_name']
        info = func_info_map.get(func_name)
        
        if info and info["source_code"]:
            skeleton_code = create_starter_code(info["source_code"], func_name)
        else:
            skeleton_code = ""
            
        # Build system context
        context_lines = []
        for other_name, other_info in func_info_map.items():
            if other_name != func_name:
                context_lines.append(f"- {other_info['signature']}: {other_info['description']}")
        
        system_context = "\n".join(context_lines)
        
        new_q = {
            "function_name": func_name,
            "question_prompt": q['question_prompt'],
            "provided_code": skeleton_code,
            "system_context": system_context
        }
        updated_questions.append(new_q)
    
    return {"questions": updated_questions}


# --- TEST ---


# original_code = load_code_from_file("/fs-computility-new/UPDZ02_sunhe/shared/DPI/DPItorch/agent_run_inversion.py")

# json_data = {
#     "questions": [
#         {
#             "function_name": "Loss_angle_diff",
#             "question_prompt": "Function Name: Loss_angle_diff\nTarget/Goal: Computes the closure phase loss, a phase-wrapped observable robust against station phase errors.\nBrief Explanation: Uses a cosine metric to measure phase differences, handling wrapping (e.g., 359\u00b0 approximates 1\u00b0).\nDetailed Explanation: Input arguments include 'sigma' (a torch tensor representing measurement error) and 'device' for computation. Output is a function accepting 'y_true' and 'y_pred' tensors, applying the cosine metric to compute loss."
#         },
#         {
#             "function_name": "Loss_vis_diff",
#             "question_prompt": "Function Name: Loss_vis_diff\nTarget/Goal: Computes the complex visibility loss based on Euclidean distance in the complex plane.\nBrief Explanation: Measures fidelity by comparing real and imaginary parts of observed and predicted visibilities.\nDetailed Explanation: Input arguments are 'sigma' (torch tensor of measurement error) and 'device'. Output is a function taking tensors 'y_true' and 'y_pred' (complex components) and returning the computed loss."
#         },
#         {
#             "function_name": "Loss_l1",
#             "question_prompt": "Function Name: Loss_l1\nTarget/Goal: Computes sparsity loss restricting non-essential pixel values.\nBrief Explanation: Calculates the mean absolute pixel value to enforce sparsity.\nDetailed Explanation: Input 'y_pred' is a tensor representing the predicted image. Output is the computed sparsity loss, encouraging reduced noise in vacant regions."
#         },
#         {
#             "function_name": "torch_complex_mul",
#             "question_prompt": "Function Name: torch_complex_mul\nTarget/Goal: Performs element-wise complex multiplication of tensors.\nBrief Explanation: Uses simple algebra of complex numbers applied to tensors.\nDetailed Explanation: Input 'x' is a tensor of shape (Batch, N_measurements, 2) and 'y' as (2, N_vis). Output is the complex multiplied tensor."
#         },
#         {
#             "function_name": "get_rmse_numpy",
#             "question_prompt": "Function Name: get_rmse_numpy\nTarget/Goal: Computes the Root Mean Square Error (RMSE) between prediction and ground truth images.\nBrief Explanation: Utilizes standard deviation principles applied over pixel differences.\nDetailed Explanation: Inputs are 'ground_truth' and 'computed_result', both numpy arrays of equal dimensions. Output is a scalar float representing RMSE."
#         },
#         {
#             "function_name": "Loss_cross_entropy",
#             "question_prompt": "Function Name: Loss_cross_entropy\nTarget/Goal: Computes cross-entropy loss enforcing image similarity against a reference.\nBrief Explanation: Uses cross-entropy metric to gauge deviation from a given prior image.\nDetailed Explanation: Inputs are 'y_true' and 'y_pred', both tensors of image data. Output is the computed cross-entropy loss value."
#         },
#         {
#             "function_name": "load_and_preprocess_data",
#             "question_prompt": "Function Name: load_and_preprocess_data\nTarget/Goal: Loads observational data and preprocesses it for imaging.\nBrief Explanation: Involves flux estimation, pixel grid setup, and gaussian prior image generation.\nDetailed Explanation: Input 'args' is a namespace containing necessary paths and settings. Output is a dictionary with keys 'obs', 'prior', 'simim', etc., containing processed data and parameters."
#         },
#         {
#             "function_name": "forward_operator",
#             "question_prompt": "Function Name: forward_operator\nTarget/Goal: Constructs the EHT observation forward operator for mapping reconstructed images to visibilities.\nBrief Explanation: Utilizes Non-Uniform Fast Fourier Transform (NUFFT) to compute closure phases/amplitudes.\nDetailed Explanation: Inputs are numerous, including 'npix', 'nufft_ob', 'dft_mat', etc. Output is a function that computes visibilities and closure measurements given an image."
#         },
#         {
#             "function_name": "run_inversion",
#             "question_prompt": "Function Name: run_inversion\nTarget/Goal: Executes the inversion algorithm to train a generative model using variational inference.\nBrief Explanation: Iteratively adjusts model parameters to minimize KL divergence between generated and observed distributions.\nDetailed Explanation: Inputs include 'args' for configuration, 'obs' for observational data, and 'eht_obs_torch' as the forward operator. Output includes trained generative model and loss history."
#         },
#         {
#             "function_name": "evaluate_results",
#             "question_prompt": "Function Name: evaluate_results\nTarget/Goal: Evaluates the trained model's output against ground truth, checking reconstruction quality and uncertainty quantification.\nBrief Explanation: Uses techniques like RMSE computation, t-SNE visualizations, and clustering to assess model outputs.\nDetailed Explanation: Inputs include 'args' for configuration and 'img_generator' among others. Output is a dictionary with evaluation metrics such as RMSE, clustered image statistics, etc."
#         }
#     ]
# }

if __name__ == "__main__":

    # target = "run_inversion"
    # skeleton = create_starter_code(original_code, target)
    # print(skeleton)
    new_output = process_questions("/fs-computility-new/UPDZ02_sunhe/shared/DPI/DPItorch", json_data)
    print(json.dumps(new_output, indent=4))
    working_dir = "/fs-computility-new/UPDZ02_sunhe/shared/DPI/DPItorch"
    output_path = os.path.join(working_dir, "question.json")
    with open(output_path, 'w', encoding='utf8') as f:
        json.dump(new_output, f,indent=4)

    