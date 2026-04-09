import matplotlib

matplotlib.use('Agg')

def dim_match(A_shape, B_shape):
    A_out_shape = A_shape
    B_out_shape = B_shape
    if len(A_shape) < len(B_shape):
        for _ in range(len(A_shape), len(B_shape)):
            A_out_shape += (1,)
    elif len(A_shape) > len(B_shape):
        for _ in range(len(B_shape), len(A_shape)):
            B_out_shape += (1,)
    return A_out_shape, B_out_shape
