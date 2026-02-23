import os
from typing import List
from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI
import json
import argparse

# ... (rest of imports if any)

default_funcs = [
    'Loss_angle_diff', 'Loss_logca_diff2', 'Loss_vis_diff', 
    'Loss_logamp_diff', 'Loss_l1', 'Loss_TSV', 'Loss_flux', 
    'Loss_center', 'Loss_cross_entropy', 'torch_complex_mul', 
    'get_rmse_numpy', 'Order_inverse', 'Obs_params_torch', 
    'load_and_preprocess_data', 'forward_operator', 
    'run_inversion', 'evaluate_results'
]

def load_code_from_file(filepath):
    """Reads the raw source code from a file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find file: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

# 1. SETUP CLIENT
# Replace with your API Key (OpenAI, DeepSeek, or a local LLM server)
# client = OpenAI(api_key="YOUR_API_KEY_HERE", base_url="https://api.whatai.cc/v1") 

# 2. DEFINE THE OUTPUT STRUCTURE (Pydantic)
# This forces the LLM to reply in the exact JSON format you requested.
class CodingQuestion(BaseModel):
    function_name: str = Field(..., description="The exact name of the function to be implemented.")
    question_prompt: str = Field(..., description="The detailed prompt for the student, containing target, method, input/output.")

class TutorialExtraction(BaseModel):
    questions: List[CodingQuestion]

# 3. THE CORE FUNCTION
def generate_coding_questions(tutorial_content: str, target_functions: list, client: OpenAI, model_name: str = "gpt-4o"):
    
    target_list_str = ", ".join(target_functions)
    
    # Updated system instruction to be explicit about JSON structure
    system_instruction = f"""
    You are an expert Python technical interviewer. 
    You will be given a programming tutorial text containing functions and code.
    
    Your task is to extract functions and generate coding interview questions for them.
    Generate coding interview questions ONLY for the following specific functions:
    {target_list_str}

    **CRITICAL INSTRUCTION**: 
    Do NOT generate questions for any function that is not in the list above.
    If a function in this list is not found in the code, ignore it.
        
    For each function, generate a JSON object with:
    1. 'function_name': The name of the function.
    2. 'question_prompt': A comprehensive instruction for a student to write this function.
    
    CRITICAL: The 'question_prompt' text MUST follow this structure strictly:
    - State the Function Name.
    - State the Target/Goal of the function.
    - Provide a Brief Explanation of the specific method/algorithm used (to avoid ambiguity).
    - Provide a Detailed Explanation of the Input arguments (types, shapes) and the Output (return type, structure).

    **OUTPUT FORMAT**:
    You MUST respond with a valid JSON object strictly following this schema:
    {{
        "questions": [
            {{
                "function_name": "string",
                "question_prompt": "string"
            }},
            ...
        ]
    }}
    """

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": f"Here is the tutorial content:\n\n{tutorial_content}"},
    ]

    max_retries = 3
    for attempt in range(max_retries):
        try:
            # print(f"  Attempt {attempt + 1}/{max_retries} to generate questions...")
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                response_format={"type": "json_object"},
            )

            content = completion.choices[0].message.content
            if not content:
                raise ValueError("Empty response from LLM")

            # Parse JSON
            json_data = json.loads(content)
            
            # Validate with Pydantic
            result = TutorialExtraction.model_validate(json_data)
            return result

        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            print(f"  [Warning] Attempt {attempt + 1} failed: {e}")
            # Add error to messages for next retry (Closed-Loop)
            messages.append({"role": "assistant", "content": content if 'content' in locals() and content else ""})
            messages.append({"role": "user", "content": f"Your previous response caused an error: {str(e)}. Please correct the JSON format and structure strictly."})
            
        except Exception as e:
            print(f"  [Error] Unexpected error: {e}")
            return None

    print("  [Error] Failed to generate valid questions after retries.")
    return None

# --- EXAMPLE USAGE ---

# Sample Tutorial Text (Input)
# sample_tutorial = """
# Today we will discuss basic image processing math.

# First, we look at the Mean Squared Error (MSE). 
# def calculate_mse(img1, img2):
#     # This function calculates the average squared difference
#     # It assumes img1 and img2 are numpy arrays of the same shape.
#     err = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
#     err /= float(img1.shape[0] * img1.shape[1])
#     return err

# Next, we look at Peak Signal-to-Noise Ratio (PSNR).
# def calculate_psnr(img1, img2):
#     # PSNR is calculated using the MSE. 
#     # Max pixel value is assumed to be 255.
#     mse = calculate_mse(img1, img2)
#     if mse == 0:
#         return 100
#     PIXEL_MAX = 255.0
#     return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
# """

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--tutorial_path', default=None, help='Path to PDF')
    args = parser.parse_args()
    sample_tutorial = load_code_from_file(args.tutorial_path)
    # Run the generator
    extraction = generate_coding_questions(sample_tutorial, default_funcs)

    if extraction:
        # Print as Multi-line JSON
        # We convert the Pydantic object back to a standard dictionary list
        print(json.dumps(extraction.model_dump(), indent=4))