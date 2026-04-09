import numpy as np

def Order_inverse(order):
    order_inv = []
    for k in range(len(order)):
        for i in range(len(order)):
            if order[i] == k:
                order_inv.append(i)
    return np.array(order_inv)
