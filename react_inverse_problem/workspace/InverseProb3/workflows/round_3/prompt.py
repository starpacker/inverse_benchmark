CODE_REVIEW_PROMPT = """
Carefully analyze the provided code solution for the given problem. Identify any:
1. Missing requirements from the problem statement
2. Potential errors or inefficiencies
3. Omitted constraints or prior knowledge
4. Opportunities for optimization

Specifically check for implementation of:
- Sparsity prior (e.g., L1 regularization, thresholding)
- Continuity prior (e.g., smoothing operations, total variation)

Provide concise improvement instructions focusing on these aspects. Do not write code - only give clear implementation guidance.
"""